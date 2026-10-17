# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Local Spark-0 addition (NOT upstream starVLA).
"""Per-hand tactile backbones → ``[B, R, hidden]`` region tokens.

* ``RegionSetEncoder`` — MLP + region pool / learned queries (point_set).
* ``RegionLatticeCNN`` — CNN over UV lattices (uv_lattice).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


def fourier_encode(x: torch.Tensor, num_bands: int) -> torch.Tensor:
    """``[x, sin(2^k π x), cos(...)]`` along the last dim."""
    if num_bands <= 0:
        return x
    freqs = (2.0 ** torch.arange(num_bands, device=x.device, dtype=x.dtype)) * math.pi
    xb = x.unsqueeze(-1) * freqs
    return torch.cat([x, torch.sin(xb).flatten(-2), torch.cos(xb).flatten(-2)], dim=-1)


def fourier_out_dim(in_dim: int, num_bands: int) -> int:
    return in_dim if num_bands <= 0 else in_dim * (1 + 2 * num_bands)


class RegionSetEncoder(nn.Module):
    """Free taxels ``[B, N, C]`` → ``[B, R, hidden]``."""

    def __init__(
        self,
        point_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_region_tokens: int = 8,
        use_fourier_features: bool = True,
        fourier_bands: int = 4,
        pool_mode: str = "softmax",
        trim_frac: float = 0.1,
    ) -> None:
        super().__init__()
        self.point_dim = int(point_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_region_tokens = int(num_region_tokens)
        if self.point_dim < 1 or self.num_region_tokens < 1 or num_layers < 1:
            raise ValueError("point_dim, num_region_tokens, num_layers must be >= 1")
        self.fourier_bands = int(fourier_bands) if use_fourier_features else 0
        self.pool_mode = str(pool_mode).lower()
        if self.pool_mode not in ("softmax", "masked_mean", "robust_mean"):
            raise ValueError(
                f"pool_mode must be softmax|masked_mean|robust_mean, got {pool_mode!r}"
            )
        self.trim_frac = float(trim_frac)
        if not (0.0 <= self.trim_frac < 0.5):
            raise ValueError(f"trim_frac must be in [0, 0.5), got {self.trim_frac}")

        in_dim = fourier_out_dim(self.point_dim, self.fourier_bands)
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(num_layers):
            layers.extend(
                [nn.Linear(dim, self.hidden_dim), nn.GELU(), nn.LayerNorm(self.hidden_dim)]
            )
            dim = self.hidden_dim
        self.point_mlp = nn.Sequential(*layers)
        self.queries = nn.Parameter(
            torch.randn(self.num_region_tokens, self.hidden_dim) * 0.02
        )
        self.query_attn = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.out_norm = nn.LayerNorm(self.hidden_dim)

    def forward(
        self,
        points: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        region_ids: Optional[torch.Tensor] = None,
        point_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if points.ndim != 3 or points.shape[-1] != self.point_dim:
            raise ValueError(
                f"points must be [B, N, {self.point_dim}], got {tuple(points.shape)}"
            )
        B, N, _ = points.shape
        if valid_mask is None:
            valid_mask = torch.ones(B, N, dtype=torch.bool, device=points.device)
        else:
            valid_mask = valid_mask.to(device=points.device, dtype=torch.bool)
            if valid_mask.shape != (B, N):
                raise ValueError(
                    f"valid_mask must be [B, N]={(B, N)}, got {tuple(valid_mask.shape)}"
                )

        h = self.point_mlp(
            fourier_encode(points, self.fourier_bands)
            if self.fourier_bands
            else points
        )
        if point_features is not None:
            # Extra per-point features (e.g. temporal pressure summary) are
            # fused in hidden space — never squeezed back to point_dim.
            if point_features.shape != (B, N, self.hidden_dim):
                raise ValueError(
                    f"point_features must be [B, N, {self.hidden_dim}], "
                    f"got {tuple(point_features.shape)}"
                )
            h = h + point_features.to(dtype=h.dtype)
        if region_ids is not None:
            if self.pool_mode == "softmax":
                return self._pool_by_region(h, valid_mask, region_ids)
            return self._pool_by_region_mean(
                h, valid_mask, region_ids, robust=self.pool_mode == "robust_mean"
            )
        if self.pool_mode != "softmax":
            # No region_map: learned queries latch onto stuck taxels. A
            # global masked / trimmed mean is the noise-robust fallback.
            return self._pool_global_mean(
                h, valid_mask, robust=self.pool_mode == "robust_mean"
            )
        return self._pool_by_queries(h, valid_mask)

    def _pool_by_region(
        self,
        h: torch.Tensor,
        valid_mask: torch.Tensor,
        region_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, H = h.shape
        R = self.num_region_tokens
        region_ids = region_ids.to(device=h.device, dtype=torch.long).reshape(-1)
        if region_ids.numel() != N:
            raise ValueError(f"region_ids length {region_ids.numel()} != N={N}")
        if bool((region_ids < 0).any() or (region_ids >= R).any()):
            raise ValueError(
                f"region_ids must be in [0, {R}), "
                f"got [{int(region_ids.min())}, {int(region_ids.max())}]"
            )

        scores = self.query_attn(h).sum(-1)
        tokens = h.new_zeros(B, R, H)
        token_mask = torch.zeros(B, R, dtype=torch.bool, device=h.device)
        neg = torch.finfo(scores.dtype).min
        for r in range(R):
            in_region = region_ids == r
            if not bool(in_region.any()):
                continue
            mask = valid_mask & in_region.unsqueeze(0)
            token_mask[:, r] = mask.any(dim=-1)
            # Fully-invalid rows would softmax(-inf) → NaN; force a dummy
            # position so softmax stays finite, then zero the weight mass.
            safe = mask.clone()
            safe[:, 0] = safe[:, 0] | (~token_mask[:, r])
            weights = torch.softmax(scores.masked_fill(~safe, neg), dim=-1)
            weights = weights * mask.to(scores.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            tokens[:, r] = torch.einsum("bn,bnh->bh", weights, h)
        return self.out_norm(tokens), token_mask

    def _pool_by_region_mean(
        self,
        h: torch.Tensor,
        valid_mask: torch.Tensor,
        region_ids: torch.Tensor,
        *,
        robust: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Masked (optionally trimmed) mean inside each region.

        Softmax attention concentrates on the brightest taxel — usually a
        stuck / saturated cell on a dirty glove. Mean + optional high-energy
        trim keeps one bad pad from owning the finger token.
        """
        B, N, H = h.shape
        R = self.num_region_tokens
        region_ids = region_ids.to(device=h.device, dtype=torch.long).reshape(-1)
        if region_ids.numel() != N:
            raise ValueError(f"region_ids length {region_ids.numel()} != N={N}")
        if bool((region_ids < 0).any() or (region_ids >= R).any()):
            raise ValueError(
                f"region_ids must be in [0, {R}), "
                f"got [{int(region_ids.min())}, {int(region_ids.max())}]"
            )

        tokens = h.new_zeros(B, R, H)
        token_mask = torch.zeros(B, R, dtype=torch.bool, device=h.device)
        energy = h.detach().float().norm(dim=-1)
        for r in range(R):
            in_region = region_ids == r
            if not bool(in_region.any()):
                continue
            mask = valid_mask & in_region.unsqueeze(0)
            token_mask[:, r] = mask.any(dim=-1)
            keep = self._trim_high_energy(mask, energy) if robust else mask
            weights = keep.to(h.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            tokens[:, r] = torch.einsum("bn,bnh->bh", weights, h)
        return self.out_norm(tokens), token_mask

    def _pool_global_mean(
        self,
        h: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        robust: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, H = h.shape
        R = self.num_region_tokens
        hand_valid = valid_mask.any(dim=-1)
        keep = (
            self._trim_high_energy(valid_mask, h.detach().float().norm(dim=-1))
            if robust
            else valid_mask
        )
        weights = keep.to(h.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        pooled = torch.einsum("bn,bnh->bh", weights, h)
        tokens = pooled.unsqueeze(1).expand(B, R, H) * hand_valid.view(B, 1, 1).to(h.dtype)
        return self.out_norm(tokens), hand_valid.unsqueeze(1).expand(B, R)

    def _trim_high_energy(
        self, mask: torch.Tensor, energy: torch.Tensor
    ) -> torch.Tensor:
        """Drop the brightest ``trim_frac`` of currently-valid points per row."""
        if self.trim_frac <= 0.0:
            return mask
        n_valid = mask.sum(dim=-1)
        n_drop = (n_valid.to(energy.dtype) * self.trim_frac).floor().to(torch.long)
        # Need at least two valid points left after a drop.
        n_drop = torch.minimum(n_drop, (n_valid - 1).clamp_min(0))
        if not bool((n_drop > 0).any()):
            return mask
        ranked = energy.masked_fill(~mask, torch.finfo(energy.dtype).min)
        # kth largest ≈ drop threshold; points strictly above it go.
        max_drop = int(n_drop.max().item())
        topk = ranked.topk(max_drop, dim=-1).values
        gather_k = (n_drop - 1).clamp_min(0).unsqueeze(-1)
        thresh = topk.gather(1, gather_k).squeeze(-1)
        drop = mask & (n_drop > 0).unsqueeze(-1) & (energy >= thresh.unsqueeze(-1))
        trimmed = mask & ~drop
        # If trim emptied a row that still had valids, keep the original mask.
        emptied = mask.any(dim=-1) & ~trimmed.any(dim=-1)
        return torch.where(emptied.unsqueeze(-1), mask, trimmed)

    def _pool_by_queries(
        self,
        h: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, H = h.shape
        R = self.num_region_tokens
        hand_valid = valid_mask.any(dim=-1)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        logits = torch.einsum("brh,bnh->brn", self.query_attn(q), h) / math.sqrt(H)
        # Avoid softmax(-inf) NaN on fully-invalid hands.
        safe = valid_mask.clone()
        safe[:, 0] = safe[:, 0] | (~hand_valid)
        neg = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~safe.unsqueeze(1), neg)
        attn = torch.softmax(logits, dim=-1) * hand_valid.view(B, 1, 1).to(logits.dtype)
        tokens = torch.einsum("brn,bnh->brh", attn, h)
        return self.out_norm(tokens), hand_valid.unsqueeze(1).expand(B, R)


class RegionLatticeCNN(nn.Module):
    """UV lattice ``[B, R, C, Hr, Wr]`` → ``[B, R, hidden]``."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_region_tokens: int = 8,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_dim = int(hidden_dim)
        self.num_region_tokens = int(num_region_tokens)
        if self.in_channels < 1 or self.num_region_tokens < 1 or num_layers < 1:
            raise ValueError("in_channels, num_region_tokens, num_layers must be >= 1")

        channels = [self.in_channels] + [self.hidden_dim] * num_layers
        groups = min(8, self.hidden_dim)
        while self.hidden_dim % groups != 0:
            groups -= 1
        convs: list[nn.Module] = []
        for i in range(num_layers):
            convs.extend(
                [
                    nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.GroupNorm(groups, channels[i + 1]),
                ]
            )
        self.cnn = nn.Sequential(*convs)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )

    def forward(
        self,
        lattice: torch.Tensor,
        region_valid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if lattice.ndim != 5:
            raise ValueError(f"lattice must be [B, R, C, H, W], got {tuple(lattice.shape)}")
        B, R, C, H, W = lattice.shape
        if R != self.num_region_tokens or C != self.in_channels:
            raise ValueError(
                f"lattice [B,R,C,H,W]={tuple(lattice.shape)} != "
                f"[B, {self.num_region_tokens}, {self.in_channels}, H, W]"
            )
        if region_valid is not None and tuple(region_valid.shape) != (B, R):
            raise ValueError(
                f"region_valid must be [B, R]={(B, R)}, got {tuple(region_valid.shape)}"
            )

        x = self.pool(self.cnn(lattice.reshape(B * R, C, H, W))).flatten(1)
        tokens = self.proj(x).view(B, R, self.hidden_dim)
        if region_valid is None:
            region_valid = lattice.abs().flatten(2).sum(-1) > 0
        else:
            region_valid = region_valid.to(device=lattice.device, dtype=torch.bool)
        return tokens, region_valid
