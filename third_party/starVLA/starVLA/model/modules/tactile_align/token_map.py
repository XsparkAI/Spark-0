# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Local Spark-0 addition (NOT upstream starVLA).
"""Map image-plane (u, v) / HxW heatmaps onto Qwen merged visual-token grids.

Reuses the same geometry DA3 already relies on: Qwen3-VL lays merged visual
tokens of each image in raster order; ``image_grid_thw / merge_size`` gives
the ``(gh, gw)`` student grid.

Primary alignment target is a dense head-camera **HxW heatmap** (saliency /
contact pressure raster). Sparse (u,v) points remain supported as a fallback.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def student_grid_hw(
    grid_thw_row: Sequence[int], merge_size: int
) -> Tuple[int, int]:
    """One ``image_grid_thw`` row ``(t, h, w)`` → merged ``(gh, gw)``."""
    t, h, w = (int(v) for v in grid_thw_row)
    if t != 1:
        raise ValueError(
            f"tactile_align does not support temporal image grids (t={t}); "
            "pass frames as separate views."
        )
    if h % merge_size != 0 or w % merge_size != 0:
        raise ValueError(
            f"image grid {(h, w)} not divisible by merge_size={merge_size}"
        )
    return h // merge_size, w // merge_size


def uv_to_token_index(
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    src_hw: Tuple[int, int],
    student_grid: Tuple[int, int],
) -> torch.Tensor:
    """Pixel ``(u=row, v=col)`` on ``src_hw`` → flat token index on student grid.

    Coordinates are clamped into the image; out-of-range callers should filter
    before calling. Returns long tensor with the same shape as ``u``.
    """
    src_h, src_w = src_hw
    gh, gw = student_grid
    if src_h <= 0 or src_w <= 0:
        raise ValueError(f"src_hw must be positive, got {src_hw}")
    # Resize-sync with the frame the VLM actually saw (obs_image_size).
    row = (u.float() * (gh / float(src_h))).long().clamp(0, gh - 1)
    col = (v.float() * (gw / float(src_w))).long().clamp(0, gw - 1)
    return row * gw + col


def resize_heatmap_to_obs(
    heatmap: torch.Tensor,
    obs_hw: Tuple[int, int],
) -> torch.Tensor:
    """Bilinear-resize HxW heatmap to the VLM observation resolution.

    ``heatmap`` is ``[..., H, W]`` (or already matching ``obs_hw``). This is
    the resize-sync step: labels must live on the same plane the VLM saw
    (``obs_image_size``), not the raw camera resolution.
    """
    if heatmap.ndim < 2:
        raise ValueError(f"heatmap must be [..., H, W], got {tuple(heatmap.shape)}")
    obs_h, obs_w = (int(obs_hw[0]), int(obs_hw[1]))
    if obs_h < 1 or obs_w < 1:
        raise ValueError(f"obs_hw must be positive, got {obs_hw}")
    if tuple(heatmap.shape[-2:]) == (obs_h, obs_w):
        return heatmap.float()
    leading = heatmap.shape[:-2]
    flat = heatmap.reshape(-1, 1, heatmap.shape[-2], heatmap.shape[-1]).float()
    resized = F.interpolate(flat, size=(obs_h, obs_w), mode="bilinear", align_corners=False)
    return resized.reshape(*leading, obs_h, obs_w)


def saliency_to_contact_map(
    saliency: torch.Tensor,
    student_grid: Tuple[int, int],
    *,
    obs_hw: Optional[Tuple[int, int]] = None,
    normalize: str = "max",
) -> torch.Tensor:
    """Dense HxW heatmap ``[..., H, W]`` → soft contact labels ``[..., gh*gw]``.

    1. Optional bilinear resize to ``obs_hw`` (VLM observation plane).
    2. Area-average pool onto the merged student token grid.
    3. Per-sample normalize to ``[0, 1]`` (``max`` or ``none``).
    """
    if saliency.ndim < 2:
        raise ValueError(f"saliency must be [..., H, W], got {tuple(saliency.shape)}")
    if obs_hw is not None:
        saliency = resize_heatmap_to_obs(saliency, obs_hw)
    gh, gw = student_grid
    leading = saliency.shape[:-2]
    flat = saliency.reshape(-1, 1, saliency.shape[-2], saliency.shape[-1]).float()
    pooled = F.adaptive_avg_pool2d(flat, output_size=(gh, gw))
    cmap = pooled.reshape(*leading, gh * gw)
    if normalize == "none":
        return cmap.clamp_min(0)
    if normalize != "max":
        raise ValueError(f"normalize must be 'max' or 'none', got {normalize!r}")
    if cmap.ndim == 1:
        peak = cmap.max().clamp_min(1e-6)
        return (cmap / peak).clamp(0, 1)
    peak = cmap.amax(dim=-1, keepdim=True).clamp_min(1e-6)
    return (cmap / peak).clamp(0, 1)


def prepare_heatmap_contact_map(
    heatmap: torch.Tensor,
    student_grid: Tuple[int, int],
    *,
    obs_hw: Optional[Tuple[int, int]] = None,
    normalize: str = "max",
) -> torch.Tensor:
    """Public alias: HxW heatmap → token-grid contact map (alignment primary path)."""
    return saliency_to_contact_map(
        heatmap, student_grid, obs_hw=obs_hw, normalize=normalize
    )


def scatter_points_to_contact_map(
    token_index: torch.Tensor,
    values: torch.Tensor,
    *,
    num_tokens: int,
    reduce: str = "max",
) -> torch.Tensor:
    """Sparse per-point values on token indices → ``[num_tokens]`` map.

    ``token_index`` / ``values`` are 1-D for one sample. Empty input → zeros.
    """
    device = values.device if isinstance(values, torch.Tensor) else token_index.device
    dtype = values.dtype if isinstance(values, torch.Tensor) else torch.float32
    out = torch.zeros(num_tokens, device=device, dtype=dtype)
    if token_index.numel() == 0:
        return out
    idx = token_index.long().view(-1)
    val = values.to(device=device, dtype=dtype).view(-1)
    if reduce == "max":
        out.scatter_reduce_(0, idx, val, reduce="amax", include_self=True)
    elif reduce == "sum":
        out.scatter_add_(0, idx, val)
    else:
        raise ValueError(f"reduce must be 'max' or 'sum', got {reduce!r}")
    return out
