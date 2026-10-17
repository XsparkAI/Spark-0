# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Local Spark-0 addition (NOT upstream starVLA).
"""Fine-grained contact → per-region supervision targets (§2.5 steps 4–6).

Callers (dataloader offline join, or a thin in-module path) feed sparse 2D
points and/or a dense head-camera saliency; this module aggregates to the
region / token tensors ``ContactRegionAlignment`` consumes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from starVLA.model.modules.tactile_align.token_map import (
    prepare_heatmap_contact_map,
    scatter_points_to_contact_map,
    uv_to_token_index,
)


def _as_dict(cfg: Any) -> Dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    to_container = getattr(cfg, "to_container", None)
    if callable(to_container):
        return dict(to_container(resolve=True))
    if hasattr(cfg, "items"):
        return {str(k): v for k, v in cfg.items()}
    return dict(cfg)


def load_region_map(path_or_list: Any, num_taxels: Optional[int] = None) -> Optional[torch.Tensor]:
    """Static per-taxel region ids → ``LongTensor [N]`` (shared with fusion LUT)."""
    if path_or_list is None:
        return None
    if isinstance(path_or_list, torch.Tensor):
        ids = path_or_list.long().reshape(-1)
    elif isinstance(path_or_list, (list, tuple)):
        ids = torch.tensor(list(path_or_list), dtype=torch.long)
    else:
        path = Path(str(path_or_list)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"tactile_align.region.region_map not found: {path}")
        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix in {".yaml", ".yml"}:
            import yaml

            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            ids = torch.as_tensor(torch.load(path, map_location="cpu")).long().reshape(-1)
            data = None
        if data is not None:
            if isinstance(data, dict):
                data = data.get("region_ids", data.get("ids"))
                if data is None:
                    raise ValueError(
                        f"region_map {path} needs a list or keys region_ids/ids"
                    )
            ids = torch.tensor(list(data), dtype=torch.long)
    if num_taxels is not None and ids.numel() != num_taxels:
        raise ValueError(
            f"region_map length {ids.numel()} != expected num_taxels={num_taxels}"
        )
    return ids


def force_to_bin(force: torch.Tensor, edges: Sequence[float]) -> torch.Tensor:
    """Map non-negative force → bin id in ``[0, len(edges)]`` (last bin open)."""
    if not edges:
        raise ValueError("force_bins must be non-empty")
    edges_t = torch.as_tensor(list(edges), device=force.device, dtype=force.dtype)
    # bucketize: bin i when edges[i-1] <= x < edges[i]; last = >= edges[-1]
    return torch.bucketize(force.clamp_min(0), edges_t)


def aggregate_points_to_regions(
    *,
    u: torch.Tensor,
    v: torch.Tensor,
    force: torch.Tensor,
    region_id: torch.Tensor,
    hand_id: torch.Tensor,
    num_hands: int,
    num_regions_per_hand: int,
    src_hw: Tuple[int, int],
    student_grid: Tuple[int, int],
    force_bins: Sequence[float],
) -> Dict[str, torch.Tensor]:
    """One sample's sparse points → region + token-map targets.

    Region index layout: ``hand * num_regions_per_hand + local_region``.
    Returns float/long tensors without a batch dim.
    """
    R = num_hands * num_regions_per_hand
    P = student_grid[0] * student_grid[1]
    device = force.device
    contact = torch.zeros(R, device=device, dtype=torch.float32)
    force_r = torch.zeros(R, device=device, dtype=torch.float32)
    token_mask = torch.zeros(R, P, device=device, dtype=torch.bool)

    if force.numel() == 0:
        bins = torch.zeros(R, device=device, dtype=torch.long)
        return {
            "region_contact": contact,
            "region_force": force_r,
            "region_bin": bins,
            "region_token_mask": token_mask,
            "contact_map": torch.zeros(P, device=device, dtype=torch.float32),
        }

    # Global region index; invalid hand/region rows are dropped.
    local = region_id.long()
    hand = hand_id.long()
    valid = (
        (hand >= 0)
        & (hand < num_hands)
        & (local >= 0)
        & (local < num_regions_per_hand)
        & torch.isfinite(force)
    )
    if not bool(valid.any()):
        bins = torch.zeros(R, device=device, dtype=torch.long)
        return {
            "region_contact": contact,
            "region_force": force_r,
            "region_bin": bins,
            "region_token_mask": token_mask,
            "contact_map": torch.zeros(P, device=device, dtype=torch.float32),
        }

    hand = hand[valid]
    local = local[valid]
    force_v = force[valid].float().clamp_min(0)
    u_v = u[valid]
    v_v = v[valid]
    rid = hand * num_regions_per_hand + local

    # contact = max conf (here: presence if force>0), force = max over region
    for r in rid.unique().tolist():
        m = rid == r
        contact[r] = 1.0 if bool((force_v[m] > 0).any()) else 0.0
        force_r[r] = force_v[m].max()
        tok = uv_to_token_index(u_v[m], v_v[m], src_hw=src_hw, student_grid=student_grid)
        token_mask[r].scatter_(0, tok, torch.ones_like(tok, dtype=torch.bool))

    bins = force_to_bin(force_r, force_bins)
    # Zero-force regions stay in bin 0 (none).
    bins = torch.where(contact > 0, bins, torch.zeros_like(bins))

    tok_all = uv_to_token_index(u_v, v_v, src_hw=src_hw, student_grid=student_grid)
    contact_map = scatter_points_to_contact_map(
        tok_all, force_v, num_tokens=P, reduce="max"
    )
    # Soft [0,1] map for dice/focal
    if contact_map.max() > 0:
        contact_map = contact_map / contact_map.max().clamp_min(1e-6)

    return {
        "region_contact": contact,
        "region_force": force_r,
        "region_bin": bins,
        "region_token_mask": token_mask,
        "contact_map": contact_map,
    }


def targets_from_saliency(
    saliency: torch.Tensor,
    student_grid: Tuple[int, int],
    *,
    obs_hw: Optional[Tuple[int, int]] = None,
    normalize: str = "max",
) -> Dict[str, torch.Tensor]:
    """Dense HxW heatmap ``[H,W]`` (or ``[B,H,W]``) → ``contact_map`` only.

    Primary alignment path. Region heads are omitted; ``ContactRegionAlignment``
    then runs the contact-map term unless precomputed region tensors are also
    supplied. ``obs_hw`` bilinear-resizes the heatmap onto the VLM observation
    plane before pooling onto the student token grid.
    """
    cmap = prepare_heatmap_contact_map(
        saliency, student_grid, obs_hw=obs_hw, normalize=normalize
    )
    return {"contact_map": cmap}


def stack_region_targets(per_sample: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack a list of per-sample target dicts (same keys/shapes)."""
    if not per_sample:
        raise ValueError("per_sample targets list is empty")
    keys = per_sample[0].keys()
    out: Dict[str, torch.Tensor] = {}
    for key in keys:
        out[key] = torch.stack([s[key] for s in per_sample], dim=0)
    return out


def parse_region_cfg(cfg: Any) -> Dict[str, Any]:
    """Normalize ``framework.tactile_align.region`` knobs."""
    r = _as_dict(cfg)
    edges = r.get("force_bins", [0.05, 0.2, 1.0])
    return {
        "region_map": r.get("region_map", None),
        "num_hands": int(r.get("num_hands", 2)),
        "num_regions_per_hand": int(r.get("num_regions_per_hand", 8)),
        "force_bins": [float(x) for x in edges],
        "baseline_mode": str(r.get("baseline_mode", "per_glove")),
    }
