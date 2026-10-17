# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Local Spark-0 addition (NOT upstream starVLA).
"""Lightweight 1D temporal encoders for tactile history windows."""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalConvEncoder(nn.Module):
    """Encode ``[B, T, C]`` → ``[B, hidden]`` with depthwise-ish 1D convs.

    Used for fingertip ``region_pressure`` histories (full
    ``tactile_expert.temporal_layers``) and a thinner per-taxel pressure
    summary (``streams.taxel_pressure_3d.temporal_layers``, default 1).
    ``history_len`` is the configured window; shorter inputs are rejected
    by the caller (strict shape).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        *,
        num_layers: int = 2,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.kernel_size = int(kernel_size)
        if min(self.in_dim, self.hidden_dim, self.num_layers) < 1:
            raise ValueError("in_dim, hidden_dim, num_layers must be >= 1")
        if self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ValueError(
                f"temporal kernel_size must be odd and >= 1, got {self.kernel_size}"
            )

        layers: list[nn.Module] = []
        dim = self.in_dim
        pad = self.kernel_size // 2
        for _ in range(self.num_layers):
            layers.extend(
                [
                    nn.Conv1d(dim, self.hidden_dim, self.kernel_size, padding=pad),
                    nn.GELU(),
                ]
            )
            dim = self.hidden_dim
        self.conv = nn.Sequential(*layers)
        self.out_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[B, T, C]`` → ``[B, hidden]`` (mean-pool over time)."""
        if x.ndim != 3 or x.shape[-1] != self.in_dim:
            raise ValueError(
                f"temporal input must be [B, T, {self.in_dim}], got {tuple(x.shape)}"
            )
        if x.shape[1] < 1:
            raise ValueError(f"temporal T must be >= 1, got {x.shape[1]}")
        # Conv1d wants [B, C, T]
        h = self.conv(x.transpose(1, 2))
        return self.out_norm(h.mean(dim=-1))


__all__ = ["TemporalConvEncoder"]
