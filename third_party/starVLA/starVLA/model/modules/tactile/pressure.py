# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Local Spark-0 addition (NOT upstream starVLA).
"""Shared tactile pressure conditioning (dataloader + deploy).

Order, when all knobs are on:

    1. per-hand median center  (``tactile_pressure_center: median``)
    2. ``sign(p) * log1p(|p|) / scale``
    3. append temporal Δp as an extra channel (region stream only)

Defaults are all off so existing runs stay bit-identical. Deploy must call
the same helpers as ``LeRobotSingleDataset._get_tactile_step_data``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# Stem / recipe token: compute scale from the dataset's |pressure| q99.
# Explicit floats still win. Deploy must see a resolved number (train writes
# it back onto datasets.vla_data.tactile_pressure_scale).
AUTO_PRESSURE_SCALE_TOKENS = frozenset({"auto", "from_stats", "stats"})


def is_auto_pressure_scale(value) -> bool:
    """True for the ``auto`` / ``from_stats`` sentinel (not ``None``)."""
    if isinstance(value, str):
        return value.strip().lower() in AUTO_PRESSURE_SCALE_TOKENS
    return False


def scale_from_pressure_q99(q99: float, log1p: bool) -> float:
    """Divisor that puts the |p| 99th percentile at conditioned 1.0.

    Matches ``condition_tactile_pressure``: ``log1p`` uses float32
    ``log1p(q99)`` so q99 raw newtons map to exactly 1 after the log.
    All-zero / non-positive q99 → ``1.0`` (no-op divisor; avoids ``/ 0``).

    The q99 is measured on RAW |p| (``tactile_pressure_stats.py``), so the
    "q99 → 1.0" property holds exactly only when ``tactile_pressure_center``
    is ``none`` (the shipped stems). With ``center: median`` the conditioned
    q99 is ``log1p(|q99 - median|) / scale`` — close while the median is
    small against q99, but not exactly 1.
    """
    q99 = float(q99)
    if not np.isfinite(q99):
        raise ValueError(f"pressure q99 must be finite, got {q99}")
    if q99 <= 0.0:
        return 1.0
    if log1p:
        scale = float(np.log1p(np.float32(q99)))
    else:
        scale = q99
    if scale <= 0.0 or not np.isfinite(scale):
        return 1.0
    return scale


def condition_tactile_pressure(
    pressure: np.ndarray,
    log1p: bool,
    scale: Optional[float],
) -> np.ndarray:
    """``sign(p) * log1p(|p|) / scale`` — used by train and deploy."""
    out = np.asarray(pressure, dtype=np.float32)
    if log1p:
        out = np.sign(out) * np.log1p(np.abs(out))
    if scale is not None:
        out = out / float(scale)
    return out


def center_pressure(pressure: np.ndarray, mode: Optional[str]) -> np.ndarray:
    """Subtract a per-hand robust baseline from a pressure tensor.

    ``mode``:
      * ``None`` / ``none`` / ``""`` — identity
      * ``median`` — subtract the median over every non-hand axis (region ×
        history, or taxels). Donning tightness and thermal drift sit in that
        median; contact is the residual.
    The hand axis is assumed to be 0.
    """
    out = np.asarray(pressure, dtype=np.float32)
    key = str(mode or "none").lower()
    if key in ("", "none", "off", "false"):
        return out
    if key != "median":
        raise ValueError(
            f"tactile_pressure_center must be none|median, got {mode!r}"
        )
    if out.ndim < 2:
        raise ValueError(
            f"center_pressure expects [hands, ...], got shape {out.shape}"
        )
    flat = out.reshape(out.shape[0], -1)
    med = np.median(flat, axis=1).astype(np.float32)
    return out - med.reshape((out.shape[0],) + (1,) * (out.ndim - 1))


def append_temporal_delta(array: np.ndarray, history_axis: int = -2) -> np.ndarray:
    """Concatenate a same-shape temporal difference on the last axis.

    ``array`` is ``[..., history, channels]`` (default history at -2).
    Frame 0 of the delta is zeros; later frames are ``x[t] - x[t-1]``.
    """
    x = np.asarray(array, dtype=np.float32)
    if x.ndim < 2:
        raise ValueError(f"append_temporal_delta needs ndim>=2, got {x.shape}")
    hist = x.shape[history_axis]
    delta = np.zeros_like(x)
    if hist > 1:
        sl_cur = [slice(None)] * x.ndim
        sl_prev = [slice(None)] * x.ndim
        sl_cur[history_axis] = slice(1, None)
        sl_prev[history_axis] = slice(0, -1)
        sl_out = [slice(None)] * x.ndim
        sl_out[history_axis] = slice(1, None)
        delta[tuple(sl_out)] = x[tuple(sl_cur)] - x[tuple(sl_prev)]
    return np.concatenate([x, delta], axis=-1)
