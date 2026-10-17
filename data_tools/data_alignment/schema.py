"""In-memory schemas shared by every source dataset adapter and writer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

if TYPE_CHECKING:
    from .camera_alignment import CameraAlignmentResult
    from .retargeting.wuji2 import WujiHandModel

HAND_SIDES = ("left", "right")
FRAME_ALIGNED_METADATA_KEY = "frame_aligned_metadata"
FRAME_ALIGNED_MODES = ("linear", "nearest")


def declared_frame_aligned(metadata: Mapping[str, Any]) -> dict[str, str]:
    declared = metadata.get(FRAME_ALIGNED_METADATA_KEY) or {}
    if not isinstance(declared, Mapping):
        raise TypeError("invalid frame-aligned metadata")
    result: dict[str, str] = {}
    for key, mode in declared.items():
        if mode not in FRAME_ALIGNED_MODES:
            raise ValueError(f"unknown frame-aligned mode {mode!r}")
        result[str(key)] = str(mode)
    return result


FINGERS = ("thumb", "index", "middle", "ring", "pinky")

# EgoPro ``/tactile/patch_pressure`` channel order (newtons).
TACTILE_PATCH_KEYS = (
    "left_palm",
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_palm",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
)
TACTILE_DIM = len(TACTILE_PATCH_KEYS)


def empty_tactile(length: int) -> tuple[np.ndarray, np.ndarray]:
    """Return zero tactile pressures and all-False validity for ``length`` frames."""
    return (
        np.zeros((int(length), TACTILE_DIM), dtype=np.float64),
        np.zeros(int(length), dtype=bool),
    )


@dataclass
class UnifiedHumanMotion:
    """Dataset-independent human hand geometry.

    ``joints_local`` uses ``[time, hand, finger, chain_point, xyz]``. Each
    finger has five ordered points from its palm-side root to fingertip.
    Coordinates are in a right-handed anatomical palm frame, in meters.
    """

    timestamps: np.ndarray
    wrist_poses: np.ndarray
    wrist_valid: np.ndarray
    joints_local: np.ndarray
    joint_confidence: np.ndarray
    source: str
    instruction: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.timestamps = np.asarray(self.timestamps, dtype=np.float64)
        self.wrist_poses = np.asarray(self.wrist_poses, dtype=np.float64)
        self.wrist_valid = np.asarray(self.wrist_valid, dtype=bool)
        self.joints_local = np.asarray(self.joints_local, dtype=np.float64)
        self.joint_confidence = np.asarray(self.joint_confidence, dtype=np.float64)
        length = self.timestamps.shape[0]
        expected = {
            "wrist_poses": (length, 2, 7),
            "wrist_valid": (length, 2),
            "joints_local": (length, 2, 5, 5, 3),
            "joint_confidence": (length, 2, 5, 5),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {getattr(self, name).shape}")
        if length > 1 and np.any(np.diff(self.timestamps) <= 0):
            raise ValueError("timestamps must be strictly increasing")
        if not np.all((self.joint_confidence >= 0) & (self.joint_confidence <= 1)):
            raise ValueError("joint_confidence must lie in [0, 1]")


@dataclass
class ReferenceFrameGeometry:
    """Source-native reference-frame geometry used for Spark 54-D alignment.

    Wrist and chain points live in a common reference frame (EgoPro/EgoDex
    world). ``joints_local`` remains anatomical-palm for the existing L2 IK
    path. Camera extrinsics register the source world axes to the Spark frame.
    """

    timestamps: np.ndarray
    wrist_poses_ref: np.ndarray
    wrist_valid: np.ndarray
    joints_ref: np.ndarray
    joint_confidence: np.ndarray
    joints_local: np.ndarray
    camera_from_world: np.ndarray
    camera_valid: np.ndarray
    intrinsics: np.ndarray
    source: str
    instruction: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    # EgoPro patch pressure (newtons); OpenAoE / missing sensors stay zero+invalid.
    tactile: np.ndarray | None = None
    tactile_valid: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.timestamps = np.asarray(self.timestamps, dtype=np.float64)
        self.wrist_poses_ref = np.asarray(self.wrist_poses_ref, dtype=np.float64)
        self.wrist_valid = np.asarray(self.wrist_valid, dtype=bool)
        self.joints_ref = np.asarray(self.joints_ref, dtype=np.float64)
        self.joint_confidence = np.asarray(self.joint_confidence, dtype=np.float64)
        self.joints_local = np.asarray(self.joints_local, dtype=np.float64)
        self.camera_from_world = np.asarray(self.camera_from_world, dtype=np.float64)
        self.camera_valid = np.asarray(self.camera_valid, dtype=bool)
        self.intrinsics = np.asarray(self.intrinsics, dtype=np.float64)
        length = self.timestamps.shape[0]
        expected = {
            "wrist_poses_ref": (length, 2, 7),
            "wrist_valid": (length, 2),
            "joints_ref": (length, 2, 5, 5, 3),
            "joint_confidence": (length, 2, 5, 5),
            "joints_local": (length, 2, 5, 5, 3),
            "camera_from_world": (length, 4, 4),
            "camera_valid": (length,),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {getattr(self, name).shape}")
        if self.intrinsics.shape != (4,):
            raise ValueError("intrinsics must be [fx, fy, cx, cy]")
        if self.tactile is None:
            self.tactile, _ = empty_tactile(length)
        else:
            self.tactile = np.asarray(self.tactile, dtype=np.float64)
        if self.tactile_valid is None:
            _, self.tactile_valid = empty_tactile(length)
        else:
            self.tactile_valid = np.asarray(self.tactile_valid, dtype=bool)
        if self.tactile.shape != (length, TACTILE_DIM):
            raise ValueError(
                f"tactile must have shape {(length, TACTILE_DIM)}, got {self.tactile.shape}"
            )
        if self.tactile_valid.shape != (length,):
            raise ValueError(
                f"tactile_valid must have shape {(length,)}, got {self.tactile_valid.shape}"
            )
        if length > 1 and np.any(np.diff(self.timestamps) <= 0):
            raise ValueError("timestamps must be strictly increasing")
        if not np.all((self.joint_confidence >= 0) & (self.joint_confidence <= 1)):
            raise ValueError("joint_confidence must lie in [0, 1]")

@dataclass
class UnifiedRobotMotion:
    """Spark-compatible 54-D robot state and optional action (L2)."""

    timestamps: np.ndarray
    state: np.ndarray
    state_valid: np.ndarray
    source: str
    retarget_error: np.ndarray | None = None
    action: np.ndarray | None = None
    action_valid: np.ndarray | None = None
    action_source: str = "not_available"
    instruction: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.timestamps = np.asarray(self.timestamps, dtype=np.float64)
        self.state = np.asarray(self.state, dtype=np.float64)
        self.state_valid = np.asarray(self.state_valid, dtype=bool)
        length = self.timestamps.shape[0]
        if self.state.shape != (length, 54):
            raise ValueError(f"state must have shape {(length, 54)}, got {self.state.shape}")
        if self.state_valid.shape != (length, 54):
            raise ValueError(
                f"state_valid must have shape {(length, 54)}, got {self.state_valid.shape}"
            )
        if self.retarget_error is not None:
            self.retarget_error = np.asarray(self.retarget_error, dtype=np.float64)
            if self.retarget_error.shape != (length, 2):
                raise ValueError("retarget_error must have shape [time, hand]")
        if self.action is not None:
            self.action = np.asarray(self.action, dtype=np.float64)
            if self.action.shape != (length, 54):
                raise ValueError("action must have shape [time, 54]")
            if self.action_valid is None:
                raise ValueError("action_valid is required when action is present")
            self.action_valid = np.asarray(self.action_valid, dtype=bool)
            if self.action_valid.shape != (length, 54):
                raise ValueError("action_valid must have shape [time, 54]")
        if length > 1 and np.any(np.diff(self.timestamps) <= 0):
            raise ValueError("timestamps must be strictly increasing")


def pack_spark_state(
    left_ee: np.ndarray,
    left_hand: np.ndarray,
    right_ee: np.ndarray,
    right_hand: np.ndarray,
) -> np.ndarray:
    """Pack ``left EE + left hand + right EE + right hand`` into 54-D."""
    arrays = [np.asarray(value) for value in (left_ee, left_hand, right_ee, right_hand)]
    dims = (7, 20, 7, 20)
    prefix = arrays[0].shape[:-1]
    for array, dim in zip(arrays, dims):
        if array.shape[:-1] != prefix or array.shape[-1] != dim:
            raise ValueError(
                f"Expected common prefix {prefix} and dimensions {dims}, got "
                f"{[item.shape for item in arrays]}"
            )
    return np.concatenate(arrays, axis=-1)


def bimanual_frame_valid(state_valid: np.ndarray) -> np.ndarray:
    """Return per-frame mask where both EE and both Wuji hand blocks are valid."""
    mask = np.asarray(state_valid, dtype=bool)
    if mask.ndim != 2 or mask.shape[1] != 54:
        raise ValueError(f"state_valid must have shape [T,54], got {mask.shape}")
    return (
        np.all(mask[:, 0:7], axis=1)
        & np.all(mask[:, 7:27], axis=1)
        & np.all(mask[:, 27:34], axis=1)
        & np.all(mask[:, 34:54], axis=1)
    )


def leading_trailing_true_span(mask: np.ndarray) -> tuple[int, int]:
    """Return ``[first_true, last_true + 1)``.

    Trims only leading/trailing False frames; interior gaps are kept.
    Raises if ``mask`` has no True entries.
    """
    values = np.asarray(mask, dtype=bool).reshape(-1)
    valid = np.flatnonzero(values)
    if not len(valid):
        raise ValueError("No valid frames to keep")
    return int(valid[0]), int(valid[-1]) + 1


def slice_robot_motion(
    motion: UnifiedRobotMotion,
    start: int,
    end: int,
) -> UnifiedRobotMotion:
    """Slice a robot motion to ``[start, end)`` without rewriting timestamps."""
    if start < 0 or end > len(motion.timestamps) or start >= end:
        raise ValueError(f"Invalid slice [{start}, {end}) for length {len(motion.timestamps)}")
    index = slice(start, end)
    return UnifiedRobotMotion(
        timestamps=motion.timestamps[index],
        state=motion.state[index],
        state_valid=motion.state_valid[index],
        source=motion.source,
        retarget_error=(
            None if motion.retarget_error is None else motion.retarget_error[index]
        ),
        action=None if motion.action is None else motion.action[index],
        action_valid=None if motion.action_valid is None else motion.action_valid[index],
        action_source=motion.action_source,
        instruction=motion.instruction,
        metadata=dict(motion.metadata),
    )


def fill_nan_frames_adjacent(
    values: np.ndarray,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fill invalid/non-finite time frames from adjacent valid frames.

    Interior gaps use linear interpolation between the previous and next valid
    frames. Leading/trailing gaps copy the nearest valid neighbor. Returns
    ``(filled_values, filled_valid, n_frames_filled)``. Frames with no valid
    neighbor anywhere are left unchanged.
    """
    filled = np.array(values, dtype=np.float64, copy=True)
    if filled.ndim != 2:
        raise ValueError(f"values must be [T, D], got {filled.shape}")
    length = filled.shape[0]
    if valid is None:
        frame_ok = np.all(np.isfinite(filled), axis=1)
    else:
        mask = np.asarray(valid, dtype=bool)
        if mask.shape != filled.shape:
            raise ValueError(
                f"valid must match values shape {filled.shape}, got {mask.shape}"
            )
        frame_ok = np.all(mask, axis=1) & np.all(np.isfinite(filled), axis=1)
    filled_valid = np.broadcast_to(frame_ok[:, None], filled.shape).copy()
    if length == 0 or np.all(frame_ok):
        return filled, filled_valid, 0
    good = np.flatnonzero(frame_ok)
    if not len(good):
        return filled, filled_valid, 0

    prev = np.full(length, -1, dtype=np.int64)
    next_idx = np.full(length, -1, dtype=np.int64)
    last = -1
    for index in range(length):
        if frame_ok[index]:
            last = index
        prev[index] = last
    last = -1
    for index in range(length - 1, -1, -1):
        if frame_ok[index]:
            last = index
        next_idx[index] = last

    filled_count = 0
    for index in range(length):
        if frame_ok[index]:
            continue
        left = int(prev[index])
        right = int(next_idx[index])
        if left < 0 and right < 0:
            continue
        if left < 0:
            filled[index] = filled[right]
        elif right < 0:
            filled[index] = filled[left]
        elif left == right:
            filled[index] = filled[left]
        else:
            amount = (index - left) / (right - left)
            filled[index] = (1.0 - amount) * filled[left] + amount * filled[right]
        filled_valid[index] = True
        filled_count += 1
    return filled, filled_valid, filled_count


def fill_hand_joint_nan_adjacent(motion: UnifiedRobotMotion) -> UnifiedRobotMotion:
    """Fill left/right Wuji hand NaN frames from adjacent valid IK solutions."""
    state = np.array(motion.state, dtype=np.float64, copy=True)
    state_valid = np.array(motion.state_valid, dtype=bool, copy=True)
    per_hand: dict[str, int] = {}
    for name, block in (("left", slice(7, 27)), ("right", slice(34, 54))):
        filled, filled_valid, count = fill_nan_frames_adjacent(
            state[:, block],
            state_valid[:, block],
        )
        state[:, block] = filled
        state_valid[:, block] = filled_valid
        per_hand[name] = count
    metadata = dict(motion.metadata)
    metadata["hand_joint_nan_fill"] = {
        "mode": "adjacent_linear",
        "left_frames": per_hand["left"],
        "right_frames": per_hand["right"],
        "total_frames": per_hand["left"] + per_hand["right"],
    }
    return UnifiedRobotMotion(
        timestamps=motion.timestamps,
        state=state,
        state_valid=state_valid,
        source=motion.source,
        retarget_error=motion.retarget_error,
        action=motion.action,
        action_valid=motion.action_valid,
        action_source=motion.action_source,
        instruction=motion.instruction,
        metadata=metadata,
    )


def make_future_state_action(motion: UnifiedRobotMotion, steps: int = 1) -> UnifiedRobotMotion:
    """Attach a pseudo action equal to a future state.

    This is intended only for retargeted human trajectories. It is deliberately
    explicit about provenance and never overwrites measured Spark commands.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    action = np.full_like(motion.state, np.nan)
    action_valid = np.zeros_like(motion.state_valid)
    if len(motion.state) > steps:
        action[:-steps] = motion.state[steps:]
        action_valid[:-steps] = motion.state_valid[steps:]
    return UnifiedRobotMotion(
        timestamps=motion.timestamps,
        state=motion.state,
        state_valid=motion.state_valid,
        source=motion.source,
        retarget_error=motion.retarget_error,
        action=action,
        action_valid=action_valid,
        action_source=f"retargeted_future_state:{steps}",
        instruction=motion.instruction,
        metadata=dict(motion.metadata),
    )


def summarize_camera_alignment(
    alignment: CameraAlignmentResult,
) -> dict[str, object]:
    """Validate mount placement and dynamic-camera invariants.

    The virtual camera is the Spark mount optical pose, so its rotation must
    match ``world_from_spark_optical``.
    """
    physical_camera = alignment.world_from_spark_optical
    virtual_camera = alignment.world_from_virtual_ego_optical
    rotation_preserved = bool(
        np.allclose(physical_camera[:3, :3], virtual_camera[:3, :3], atol=1e-9)
    )
    camera_pose_valid = int(np.count_nonzero(alignment.camera_valid))
    depth_ok = bool(alignment.positive_depth_fraction >= 0.95)
    return {
        "positive_depth_fraction": alignment.positive_depth_fraction,
        "camera_rotation_preserved": rotation_preserved,
        "dynamic_camera_valid_frames": camera_pose_valid,
        "valid": bool(
            depth_ok and rotation_preserved and camera_pose_valid > 0
        ),
    }


def summarize_robot_motion(
    motion: UnifiedRobotMotion,
    models: dict[str, WujiHandModel] | None = None,
) -> dict[str, object]:
    """Return acceptance metrics without hiding invalid or constrained frames."""
    from .retargeting.wuji2 import WujiHandModel

    models = models or {side: WujiHandModel(side) for side in ("left", "right")}
    report: dict[str, object] = {
        "frames": len(motion.timestamps),
        "source": motion.source,
        "action_source": motion.action_source,
        "valid_state_fraction": float(np.mean(motion.state_valid)),
        "nan_in_valid_state": int(
            np.count_nonzero(~np.isfinite(motion.state[motion.state_valid]))
        ),
    }
    for hand_index, side in enumerate(("left", "right")):
        hand_slice = slice(7, 27) if side == "left" else slice(34, 54)
        q = motion.state[:, hand_slice]
        valid = np.all(motion.state_valid[:, hand_slice], axis=1)
        limit_violation = np.zeros(len(q), dtype=bool)
        if np.any(valid):
            limit_violation[valid] = np.any(
                (q[valid] < models[side].lower - 1e-8)
                | (q[valid] > models[side].upper + 1e-8),
                axis=1,
            )
        hand_report: dict[str, object] = {
            "valid_frames": int(np.count_nonzero(valid)),
            "joint_limit_violations": int(np.count_nonzero(limit_violation)),
        }
        if motion.retarget_error is not None:
            errors = motion.retarget_error[:, hand_index]
            finite = errors[np.isfinite(errors)]
            hand_report["retarget_error_p50_m"] = (
                float(np.quantile(finite, 0.5)) if len(finite) else None
            )
            hand_report["retarget_error_p95_m"] = (
                float(np.quantile(finite, 0.95)) if len(finite) else None
            )
        report[side] = hand_report
    return report
