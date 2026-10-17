"""Shared Spark HDF5 writing machinery used by every dataset writer.

Everything here is dataset-agnostic: argument validation, the 54-D
state/action split, the state/action/validity/aligned/metadata group layout,
the optional ``mano/`` groups, the fixed-width JPEG ``colors`` dataset helper
and the image-bit encoding convention. Dataset-specific writers live next to
this module in ``storage/``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
from spark0.utils.image_codec import encode_image_bit

from ..schema import (
    TACTILE_PATCH_KEYS,
    UnifiedRobotMotion,
    make_future_state_action,
)

OUTPUT_IMAGE_HW = (480, 640)

# Stored frames are standard RGB JPEGs written by XPolicyLab's
# ``encode_image_bit``; its marker lets ``decode_image_bit`` return RGB.
IMAGE_BITS_COLOR_ORDER = "xpl_rgb1"

# Native Spark extrinsics describe the pose of a graphics-style camera frame
# (x right, y up, z backward), not the OpenCV optical frame (x right, y down,
# z forward). The two frames differ by this self-inverse y/z flip.
OPTICAL_FROM_SPARK_CAMERA = np.diag([1.0, -1.0, -1.0, 1.0])


def _portable_metadata(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _portable_metadata(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_portable_metadata(item) for item in value]
    if isinstance(value, Path):
        return value.name
    if isinstance(value, str) and value.startswith("/"):
        return Path(value).name
    return value


def _encode_bgr_frame(image_bgr: np.ndarray, quality: int = 95) -> bytes:
    """Encode a frame from ``cv2.imdecode`` / ``cv2.VideoCapture`` (BGR) as image bits."""
    import cv2  # type: ignore[import-not-found]

    return encode_image_bit(
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), quality=int(quality)
    )


def source_frame_indices(
    timestamps: np.ndarray, source_fps: float
) -> np.ndarray:
    """Map output timestamps onto source frame indices via ``round(t * fps)``.

    This is the single frame-selection convention for all writers: it stays an
    identity mapping for untrimmed episodes on the native fps grid, and
    recovers the correct source frame after a leading validity trim or a
    non-native output fps (``slice_robot_motion`` keeps original timestamps).
    """
    indices = np.round(
        np.asarray(timestamps, dtype=np.float64) * float(source_fps)
    ).astype(np.int64)
    return np.maximum(indices, 0)


def validate_spark_write(fps: float, action_steps: int, source_frames: int) -> int:
    """Validate common writer arguments; return ``keep = frames - action_steps``."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not np.isclose(fps, round(fps)):
        raise ValueError("Spark HDF5 frequency must be an integer")
    if action_steps < 1:
        raise ValueError("action_steps must be positive")
    if source_frames <= action_steps:
        raise ValueError(
            f"Need more than {action_steps} frames to truncate future-state action, "
            f"got {source_frames}"
        )
    return source_frames - action_steps


@dataclass
class StateActionBlocks:
    """54-D state/action split into left/right EE + hand blocks.

    Episodes are truncated by ``action_steps`` (default 1 → length ``T-1``) so
    every retained frame has a finite next-state action. This matches the
    LeRobot next-state contract and avoids a trailing NaN action frame.
    """

    keep: int
    timestamps: np.ndarray
    state_valid: np.ndarray
    retarget_error: np.ndarray
    left_ee: np.ndarray
    left_hand: np.ndarray
    right_ee: np.ndarray
    right_hand: np.ndarray
    left_ee_valid: np.ndarray
    left_hand_valid: np.ndarray
    right_ee_valid: np.ndarray
    right_hand_valid: np.ndarray
    left_action_ee: np.ndarray
    left_action: np.ndarray
    right_action_ee: np.ndarray
    right_action: np.ndarray
    left_action_ee_valid: np.ndarray
    left_action_valid: np.ndarray
    right_action_ee_valid: np.ndarray
    right_action_valid: np.ndarray

    @classmethod
    def from_robot(
        cls,
        robot: UnifiedRobotMotion,
        *,
        action_steps: int,
        keep: int,
    ) -> "StateActionBlocks":
        # Full-horizon future action, then drop the final ``action_steps``
        # frames so retained action[t] == original state[t + action_steps]
        # with no NaN tail.
        pseudo_action = make_future_state_action(robot, steps=action_steps)
        timestamps = np.asarray(robot.timestamps[:keep], dtype=np.float64)
        state = np.asarray(robot.state[:keep], dtype=np.float64)
        state_valid = np.asarray(robot.state_valid[:keep], dtype=bool)
        action = np.asarray(pseudo_action.action[:keep], dtype=np.float64)
        action_valid = np.asarray(pseudo_action.action_valid[:keep], dtype=bool)
        if robot.retarget_error is None:
            # EE-only robots (mano-only converts) carry no IK error.
            retarget_error = np.full((keep, 2), np.nan, dtype=np.float64)
        else:
            retarget_error = np.asarray(
                robot.retarget_error[:keep], dtype=np.float64
            )

        left_action_ee = action[:, 0:7].copy()
        left_action = action[:, 7:27].copy()
        right_action_ee = action[:, 27:34].copy()
        right_action = action[:, 34:54].copy()
        left_action_ee_valid = action_valid[:, 0:7]
        left_action_valid = action_valid[:, 7:27]
        right_action_ee_valid = action_valid[:, 27:34]
        right_action_valid = action_valid[:, 34:54]
        left_action_ee[~left_action_ee_valid] = np.nan
        left_action[~left_action_valid] = np.nan
        right_action_ee[~right_action_ee_valid] = np.nan
        right_action[~right_action_valid] = np.nan
        return cls(
            keep=keep,
            timestamps=timestamps,
            state_valid=state_valid,
            retarget_error=retarget_error,
            left_ee=state[:, 0:7],
            left_hand=state[:, 7:27],
            right_ee=state[:, 27:34],
            right_hand=state[:, 34:54],
            left_ee_valid=state_valid[:, 0:7],
            left_hand_valid=state_valid[:, 7:27],
            right_ee_valid=state_valid[:, 27:34],
            right_hand_valid=state_valid[:, 34:54],
            left_action_ee=left_action_ee,
            left_action=left_action,
            right_action_ee=right_action_ee,
            right_action=right_action,
            left_action_ee_valid=left_action_ee_valid,
            left_action_valid=left_action_valid,
            right_action_ee_valid=right_action_ee_valid,
            right_action_valid=right_action_valid,
        )

def write_header(
    file: Any,
    *,
    instruction: str,
    timestamps: np.ndarray,
    fps: float,
) -> None:
    """Write instruction / version / timestamps / additional_info.frequency."""
    import h5py

    # Native Spark stores these as variable-length UTF-8 strings.
    text_dtype = h5py.string_dtype(encoding="utf-8")
    file.create_dataset("instruction", data=instruction, dtype=text_dtype)
    file.create_dataset("data_format_version", data="v1.0", dtype=text_dtype)
    file.create_dataset("timestamps", data=timestamps)
    info = file.create_group("additional_info")
    info.create_dataset("frequency", data=np.int64(round(fps)))


def write_state_action_groups(
    file: Any,
    blocks: StateActionBlocks,
    *,
    tactile: np.ndarray,
    tactile_source: str,
    arm_state_left: np.ndarray,
    arm_state_right: np.ndarray,
    arm_action_left: np.ndarray,
    arm_action_right: np.ndarray,
) -> None:
    """Write the ``state/`` and ``action/`` groups from split 54-D blocks."""
    state_group = file.create_group("state")
    state_group.create_dataset("left_ee_poses", data=blocks.left_ee)
    state_group.create_dataset("right_ee_poses", data=blocks.right_ee)
    state_group.create_dataset("left_delta_ee_poses", data=blocks.left_ee)
    state_group.create_dataset("right_delta_ee_poses", data=blocks.right_ee)
    state_group.create_dataset("left_ee_joint_states", data=blocks.left_hand)
    state_group.create_dataset("right_ee_joint_states", data=blocks.right_hand)
    state_group.create_dataset("left_arm_joint_states", data=arm_state_left)
    state_group.create_dataset("right_arm_joint_states", data=arm_state_right)
    tactile_ds = state_group.create_dataset("tactile", data=tactile)
    tactile_ds.attrs["unit"] = "newton"
    tactile_ds.attrs["source"] = tactile_source
    tactile_ds.attrs["channels"] = ",".join(TACTILE_PATCH_KEYS)
    tactile_ds.attrs["order"] = "left_palm_to_pinky_then_right"

    action_group = file.create_group("action")
    action_group.create_dataset("left_ee_poses", data=blocks.left_action_ee)
    action_group.create_dataset("right_ee_poses", data=blocks.right_action_ee)
    action_group.create_dataset("left_ee_joint_states", data=blocks.left_action)
    action_group.create_dataset("right_ee_joint_states", data=blocks.right_action)
    action_group.create_dataset("left_arm_joint_states", data=arm_action_left)
    action_group.create_dataset("right_arm_joint_states", data=arm_action_right)


def write_validity_datasets(
    file: Any,
    blocks: StateActionBlocks,
    *,
    tactile_valid: np.ndarray,
    arm_valid_left: np.ndarray,
    arm_valid_right: np.ndarray,
    arm_action_valid_left: np.ndarray,
    arm_action_valid_right: np.ndarray,
    state_valid_name: str = "state_54",
) -> Any:
    """Write the shared ``validity/`` datasets; return the group for extras."""
    validity = file.require_group("validity")
    validity.create_dataset(state_valid_name, data=blocks.state_valid)
    validity.create_dataset("tactile", data=tactile_valid)
    validity.create_dataset("left_ee_pose", data=blocks.left_ee_valid)
    validity.create_dataset("right_ee_pose", data=blocks.right_ee_valid)
    validity.create_dataset("left_ee_joint_states", data=blocks.left_hand_valid)
    validity.create_dataset("right_ee_joint_states", data=blocks.right_hand_valid)
    validity.create_dataset("left_arm_joint_states", data=arm_valid_left)
    validity.create_dataset("right_arm_joint_states", data=arm_valid_right)
    validity.create_dataset("action_left_ee_pose", data=blocks.left_action_ee_valid)
    validity.create_dataset("action_right_ee_pose", data=blocks.right_action_ee_valid)
    validity.create_dataset(
        "action_left_ee_joint_states", data=blocks.left_action_valid
    )
    validity.create_dataset(
        "action_right_ee_joint_states", data=blocks.right_action_valid
    )
    validity.create_dataset(
        "action_left_arm_joint_states", data=arm_action_valid_left
    )
    validity.create_dataset(
        "action_right_arm_joint_states", data=arm_action_valid_right
    )
    return validity


def write_aligned_group(
    file: Any,
    *,
    joints_world: np.ndarray,
    joint_confidence: np.ndarray,
    retarget_error: np.ndarray,
) -> None:
    aligned = file.create_group("aligned")
    aligned.create_dataset("joints_world", data=joints_world)
    aligned.create_dataset("joint_confidence", data=joint_confidence)
    aligned.create_dataset("retarget_error", data=retarget_error)


def write_common_metadata(
    file: Any,
    *,
    source: str,
    action_steps: int,
    source_frames: int,
    arm_state_action_source: bytes,
    camera_alignment: Any,
    provenance: dict[str, Any],
) -> None:
    """Write the ``metadata/`` group shared by every dataset writer."""
    metadata = file.create_group("metadata")
    metadata.create_dataset("source", data=np.bytes_(source.encode("utf-8")))
    metadata.create_dataset(
        "episode_truncation",
        data=np.int64(action_steps),
    )
    metadata.create_dataset(
        "source_frame_count",
        data=np.int64(source_frames),
    )
    metadata.create_dataset(
        "action_source",
        data=np.bytes_(f"retargeted_future_state:{action_steps}".encode("utf-8")),
    )
    metadata.create_dataset(
        "ee_action_source",
        data=np.bytes_(f"retargeted_future_state:{action_steps}".encode("utf-8")),
    )
    metadata.create_dataset(
        "arm_state_action_source",
        data=np.bytes_(arm_state_action_source),
    )
    metadata.create_dataset(
        "delta_ee_pose_semantics",
        data=np.bytes_(b"legacy_absolute_copy_matches_native_spark_v1"),
    )
    metadata.create_dataset(
        "camera_alignment_json",
        data=np.bytes_(
            json.dumps(
                camera_alignment,
                ensure_ascii=False,
                default=str,
            ).encode("utf-8")
        ),
    )
    metadata.create_dataset(
        "provenance_json",
        data=np.bytes_(
            json.dumps(provenance, ensure_ascii=False, default=str).encode("utf-8")
        ),
    )


def create_fixed_width_jpeg_colors(
    camera_group: Any,
    jpegs: list[bytes],
    max_jpeg_bytes: int,
) -> None:
    """Write an in-memory JPEG list as a fixed-width ``colors`` dataset."""
    colors = np.empty(len(jpegs), dtype=f"|S{max_jpeg_bytes}")
    for index, encoded in enumerate(jpegs):
        colors[index] = np.bytes_(encoded)
    camera_group.create_dataset("colors", data=colors)


_MANO_JOINT_ORDER = (
    "index_mcp,index_pip,index_dip,"
    "middle_mcp,middle_pip,middle_dip,"
    "pinky_mcp,pinky_pip,pinky_dip,"
    "ring_mcp,ring_pip,ring_dip,"
    "thumb_cmc,thumb_mcp,thumb_ip"
)


def _write_mano_groups(
    file: Any,
    mano_arrays: dict[str, np.ndarray] | None,
    action_steps: int,
    keep: int,
    left_convention: str = "mirror_model",
    *,
    replace: bool = False,
    extra_attrs: dict[str, str] | None = None,
) -> None:
    """Write the optional ``mano/`` state/action/validity groups.

    ``mano_arrays`` keys are slash-joined paths (``state/left_ee_poses``,
    ``validity/left_ee_pose``, ``validity/action_left_ee_pose``, ...)
    produced by ``mano_to_hdf5_arrays``. Frames with invalid parameters are
    NaN with matching validity masks. All arrays are truncated to ``keep``
    frames (``source_frames - action_steps``) so the MANO state/action stay
    frame-aligned with the Wuji 54-D state/action under the LeRobot
    next-state contract.
    """
    if not mano_arrays:
        return
    if "mano" in file:
        if not replace:
            raise ValueError("HDF5 already has a mano/ group")
        del file["mano"]
    mano_root = file.create_group("mano")
    mano_root.attrs["parameterization"] = (
        "MANO hand: global_orient(3 axis-angle) + hand_pose(45, 15 joints "
        "axis-angle) + betas(10); transl is not stored - ee_poses xyz holds "
        "the FK wrist position instead"
    )
    convention_attr = {
        "mirror_model": "mirror_model_true_left_hand",
        "flip_aa": "flip_aa",
        "right_then_mirror": "right_then_mirror",
    }.get(left_convention, str(left_convention))
    mano_root.attrs["left_convention"] = convention_attr
    mano_root.attrs["joint_order"] = _MANO_JOINT_ORDER
    mano_root.attrs["action_semantics"] = f"next_state_truncated:{action_steps}"
    mano_root.attrs["ee_position"] = (
        "physical_wrist_joint_fk_joints21_0_matches_wuji_state_ee_xyz"
    )
    mano_root.attrs["ee_orientation"] = (
        "mano_global_orient_wxyz_not_wuji_palm_to_ee"
    )
    if extra_attrs:
        for key, value in extra_attrs.items():
            mano_root.attrs[key] = value
    groups: dict[str, Any] = {}
    for group_name in ("state", "action", "validity"):
        groups[group_name] = mano_root.create_group(group_name)
    for key, values in mano_arrays.items():
        group_name, separator, dataset_name = key.partition("/")
        if not separator or group_name not in groups:
            raise ValueError(f"Invalid mano_arrays key {key!r}")
        groups[group_name].create_dataset(
            dataset_name, data=np.asarray(values[:keep])
        )
    for group_name in ("state", "action"):
        group = groups[group_name]
        for side in ("left", "right"):
            group[f"{side}_ee_poses"].attrs["order"] = "x,y,z,qw,qx,qy,qz"
            group[f"{side}_ee_poses"].attrs["semantics"] = (
                "xyz=physical_wrist_matches_wuji_ee; "
                "quat=mano_global_orient_wxyz_not_wuji_palm_to_ee"
            )
            group[f"{side}_ee_poses"].attrs["position"] = (
                "physical_wrist_joint_fk_joints21_0_rest_wrist_plus_transl"
            )
            group[f"{side}_ee_joint_states"].attrs["order"] = _MANO_JOINT_ORDER
            group[f"{side}_ee_joint_states"].attrs["unit"] = "axis_angle_radians"
            group[f"{side}_hand_betas"].attrs["order"] = "mano_shape_first_10"
