"""Confidence-aware filtering and temporal resampling of source-robot motion."""

from __future__ import annotations

import numpy as np

from .camera_alignment import (
    _bracket,
    _ema_valid,
    _target_timestamps,
)
from .schema import (
    UnifiedHumanMotion,
    declared_frame_aligned,
)
from .transforms import (
    ensure_quaternion_continuity,
    slerp_wxyz,
)


def _resample_frame_aligned(
    values: np.ndarray,
    mode: str,
    left_idx: np.ndarray,
    right_idx: np.ndarray,
    amount: np.ndarray,
    gap_ok: np.ndarray,
) -> np.ndarray:
    """Resample one declared per-frame metadata array onto the output grid.

    ``nearest`` copies the closer source frame and therefore ignores
    ``max_gap_s`` — it exists for discrete data that cannot carry NaN.
    """
    source = np.asarray(values)
    if mode == "nearest":
        return source[np.where(amount < 0.5, left_idx, right_idx)]
    left = np.asarray(source[left_idx], dtype=np.float64)
    right = np.asarray(source[right_idx], dtype=np.float64)
    trailing = (1,) * (left.ndim - 1)
    weight = amount.reshape((-1, *trailing))
    usable = gap_ok.reshape((-1, *trailing)) & np.isfinite(left) & np.isfinite(right)
    return np.where(usable, (1.0 - weight) * left + weight * right, np.nan)


def resample_human_motion(
    motion: UnifiedHumanMotion,
    fps: float = 30.0,
    max_gap_s: float = 0.1,
    lowpass_alpha: float = 0.35,
    target_timestamps: np.ndarray | None = None,
) -> UnifiedHumanMotion:
    """Resample source-robot human/robot hand geometry (no camera stream)."""
    if target_timestamps is None:
        target_time = _target_timestamps(motion.timestamps, fps)
    else:
        target_time = np.asarray(target_timestamps, dtype=np.float64)
        if target_time.ndim != 1 or len(target_time) < 1:
            raise ValueError("target_timestamps must be a non-empty 1-D array")
    wrist_source = np.array(motion.wrist_poses, copy=True)
    local_source = _ema_valid(
        motion.joints_local,
        motion.joint_confidence[..., None] > 0,
        lowpass_alpha,
    )
    joints_world_src = motion.metadata.get("joints_world")
    if joints_world_src is not None:
        joints_world_src = _ema_valid(
            np.asarray(joints_world_src, dtype=np.float64),
            motion.joint_confidence[..., None] > 0,
            lowpass_alpha,
        )
    head_src = motion.metadata.get("head_poses")
    head_valid_src = motion.metadata.get("head_valid")
    if head_src is not None:
        head_src = np.array(head_src, dtype=np.float64, copy=True)
        head_valid_src = (
            np.asarray(head_valid_src, dtype=bool)
            if head_valid_src is not None
            else np.all(np.isfinite(head_src), axis=1)
        )
    for hand in range(2):
        valid_indices = np.flatnonzero(motion.wrist_valid[:, hand])
        if len(valid_indices):
            wrist_source[valid_indices, hand] = ensure_quaternion_continuity(
                wrist_source[valid_indices, hand]
            )
            wrist_source[:, hand, :3] = _ema_valid(
                wrist_source[:, hand, :3],
                motion.wrist_valid[:, hand, None],
                lowpass_alpha,
            )

    wrist = np.full((len(target_time), 2, 7), np.nan)
    wrist_valid = np.zeros((len(target_time), 2), dtype=bool)
    joints_local = np.full((len(target_time), 2, 5, 5, 3), np.nan)
    confidence = np.zeros((len(target_time), 2, 5, 5))
    joints_world = (
        np.full((len(target_time), 2, 5, 5, 3), np.nan)
        if joints_world_src is not None
        else None
    )
    head_poses = (
        np.full((len(target_time), 7), np.nan) if head_src is not None else None
    )
    head_valid = (
        np.zeros(len(target_time), dtype=bool) if head_src is not None else None
    )

    # Brackets are reused after the loop to resample declared metadata arrays.
    bracket_left = np.zeros(len(target_time), dtype=np.int64)
    bracket_right = np.zeros(len(target_time), dtype=np.int64)
    bracket_amount = np.zeros(len(target_time), dtype=np.float64)
    bracket_gap_ok = np.zeros(len(target_time), dtype=bool)

    for output_index, timestamp in enumerate(target_time):
        left, right, amount = _bracket(motion.timestamps, timestamp)
        gap_ok = motion.timestamps[right] - motion.timestamps[left] <= max_gap_s
        bracket_left[output_index] = left
        bracket_right[output_index] = right
        bracket_amount[output_index] = amount
        bracket_gap_ok[output_index] = gap_ok
        if (
            head_poses is not None
            and gap_ok
            and head_valid_src[left]
            and head_valid_src[right]
        ):
            head_poses[output_index, :3] = (
                (1 - amount) * head_src[left, :3] + amount * head_src[right, :3]
            )
            head_poses[output_index, 3:7] = slerp_wxyz(
                head_src[left, 3:7], head_src[right, 3:7], amount
            )
            head_valid[output_index] = True
        for hand in range(2):
            if gap_ok and motion.wrist_valid[left, hand] and motion.wrist_valid[right, hand]:
                wrist[output_index, hand, :3] = (
                    (1 - amount) * wrist_source[left, hand, :3]
                    + amount * wrist_source[right, hand, :3]
                )
                wrist[output_index, hand, 3:7] = slerp_wxyz(
                    wrist_source[left, hand, 3:7],
                    wrist_source[right, hand, 3:7],
                    amount,
                )
                wrist_valid[output_index, hand] = True
            point_valid = (
                gap_ok
                & (motion.joint_confidence[left, hand] > 0)
                & (motion.joint_confidence[right, hand] > 0)
            )
            local_interp = (
                (1 - amount) * local_source[left, hand] + amount * local_source[right, hand]
            )
            joints_local[output_index, hand][point_valid] = local_interp[point_valid]
            confidence[output_index, hand][point_valid] = np.minimum(
                motion.joint_confidence[left, hand][point_valid],
                motion.joint_confidence[right, hand][point_valid],
            )
            if joints_world is not None and joints_world_src is not None:
                world_interp = (
                    (1 - amount) * joints_world_src[left, hand]
                    + amount * joints_world_src[right, hand]
                )
                joints_world[output_index, hand][point_valid] = world_interp[point_valid]

    metadata = {
        **motion.metadata,
        "resampled_fps": fps,
        "lowpass_alpha": lowpass_alpha,
        "max_interpolation_gap_s": max_gap_s,
        "source_readonly": True,
    }
    if joints_world is not None:
        metadata["joints_world"] = joints_world
    if head_poses is not None:
        metadata["head_poses"] = head_poses
        metadata["head_valid"] = head_valid
    for key, mode in declared_frame_aligned(motion.metadata).items():
        if key not in motion.metadata:
            raise ValueError(f"metadata['{key}'] is declared frame-aligned but missing")
        metadata[key] = _resample_frame_aligned(
            motion.metadata[key],
            mode,
            bracket_left,
            bracket_right,
            bracket_amount,
            bracket_gap_ok,
        )
    return UnifiedHumanMotion(
        timestamps=target_time,
        wrist_poses=wrist,
        wrist_valid=wrist_valid,
        joints_local=joints_local,
        joint_confidence=confidence,
        source=motion.source,
        instruction=motion.instruction,
        metadata=metadata,
    )
