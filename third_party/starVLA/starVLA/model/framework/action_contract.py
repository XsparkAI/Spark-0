# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Local Spark-0 addition (NOT upstream starVLA).
"""Packed action / state shape contract shared by train and mid-training eval.

The dataloader is supposed to emit exactly ``(action_horizon, action_dim)``
per sample (and ``(state_history_frames, state_dim)`` when state is packed).
A silent ``[:, -H:]`` slice in ``QwenPI_v3.forward`` used to hide a
horizon/packing mismatch; mid-training eval compared the unsliced packed
array to ``(B, H, D)`` preds. Both paths now go through these helpers.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np


def _as_2d(name: str, value: Any, index: int) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == object:
        raise ValueError(
            f"{name}[{index}] is ragged or non-numeric (dtype=object); "
            f"every sample must share one [T, D] layout, got type {type(value).__name__}."
        )
    if arr.ndim != 2:
        raise ValueError(
            f"{name}[{index}] must be [T, D], got shape {tuple(arr.shape)}."
        )
    return arr


def assert_action_shape(
    action: np.ndarray,
    *,
    action_horizon: int,
    action_dim: int,
    index: int = 0,
    name: str = "action",
) -> None:
    if action.shape[0] != int(action_horizon):
        raise ValueError(
            f"{name}[{index}] T={action.shape[0]} != action_horizon={int(action_horizon)}. "
            "The dataloader must emit exactly one chunk of action_horizon rows; "
            "a silent [:, -H:] slice would hide a packing/horizon mismatch."
        )
    if action.shape[1] != int(action_dim):
        raise ValueError(
            f"{name}[{index}] D={action.shape[1]} != action_dim={int(action_dim)}."
        )


def assert_state_shape(
    state: np.ndarray,
    *,
    state_dim: int,
    state_history_frames: int,
    index: int = 0,
    name: str = "state",
) -> None:
    frames = max(1, int(state_history_frames))
    if state.shape[0] != frames:
        raise ValueError(
            f"{name}[{index}] T={state.shape[0]} != state_history_frames={frames}. "
            "Train packing and deploy first/last-pad must emit the same history block."
        )
    if int(state_dim) > 0 and state.shape[1] != int(state_dim):
        raise ValueError(
            f"{name}[{index}] D={state.shape[1]} != state_dim={int(state_dim)}."
        )


def assert_packed_states(
    states: Optional[Sequence[Any]],
    *,
    batch_size: int,
    state_dim: int = 0,
    state_history_frames: int = 1,
    require_state: bool = False,
) -> None:
    """Assert the packed state history block (train or infer)."""
    if require_state and (states is None or any(s is None for s in states)):
        raise ValueError(
            "continuous_state=true requires a state array on every example."
        )
    if states is None:
        return
    if len(states) != int(batch_size):
        raise ValueError(
            f"state batch B={len(states)} != expected B={int(batch_size)}."
        )
    frames = max(1, int(state_history_frames))
    dim_s = int(state_dim)
    for i, state in enumerate(states):
        if state is None:
            continue
        assert_state_shape(
            _as_2d("state", state, i),
            state_dim=dim_s,
            state_history_frames=frames,
            index=i,
        )


def assert_packed_action_state(
    actions: Sequence[Any],
    states: Optional[Sequence[Any]] = None,
    *,
    action_horizon: int,
    action_dim: int,
    state_dim: int = 0,
    state_history_frames: int = 1,
    require_state: bool = False,
) -> None:
    """Assert every example's packed action (and optional state) matches the run.

    ``require_state`` is for ``continuous_state``: the action head's
    ``state_encoder`` must see the history block, not a missing / length-1 pad.
    When state is present but ``require_state`` is false (discretised [STATE]
    prompt only), the packed array must still be the configured history so
    train and deploy stay aligned.
    """
    if not actions:
        raise ValueError("examples list is empty; cannot pack an action batch.")
    horizon = int(action_horizon)
    dim = int(action_dim)
    if horizon < 1:
        raise ValueError(f"action_horizon must be >= 1, got {horizon}.")
    if dim < 1:
        raise ValueError(f"action_dim must be >= 1, got {dim}.")
    for i, action in enumerate(actions):
        assert_action_shape(
            _as_2d("action", action, i),
            action_horizon=horizon,
            action_dim=dim,
            index=i,
        )
    assert_packed_states(
        states,
        batch_size=len(actions),
        state_dim=state_dim,
        state_history_frames=state_history_frames,
        require_state=require_state,
    )


def stacked_action_targets(
    actions: Sequence[Any],
    *,
    action_horizon: int,
    action_dim: int,
) -> np.ndarray:
    """Stack example actions to ``(B, H, D)`` after the same asserts as train."""
    assert_packed_action_state(
        actions,
        action_horizon=action_horizon,
        action_dim=action_dim,
    )
    stacked = np.stack(
        [np.asarray(action, dtype=np.float32) for action in actions],
        axis=0,
    )
    if stacked.shape != (len(actions), int(action_horizon), int(action_dim)):
        raise ValueError(
            f"stacked action targets {stacked.shape} != "
            f"(B={len(actions)}, H={int(action_horizon)}, D={int(action_dim)})."
        )
    return stacked


def assert_pred_matches_targets(
    pred: Any,
    targets: np.ndarray,
    *,
    name: str = "normalized_actions",
) -> None:
    arr = np.asarray(pred)
    if arr.shape != targets.shape:
        raise ValueError(
            f"{name} {tuple(arr.shape)} != packed action targets {tuple(targets.shape)}; "
            "mid-training eval must score the same (B, H, D) the action loss uses."
        )
