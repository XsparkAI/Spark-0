"""EgoDex HDF5+MP4 readers for camera-aligned Spark retargeting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..common import _decode_scalar
from ...schema import ReferenceFrameGeometry
from ...transforms import (
    canonicalize_hand_points,
    ensure_quaternion_continuity,
    matrix_to_pose_wxyz,
)


def _egodex_chain_names(side: str) -> list[list[str]]:
    """5x5 chains matching apple/ml-egodex finger lists + hand root for thumb."""
    prefix = "left" if side == "left" else "right"
    return [
        [
            f"{prefix}Hand",
            f"{prefix}ThumbKnuckle",
            f"{prefix}ThumbIntermediateBase",
            f"{prefix}ThumbIntermediateTip",
            f"{prefix}ThumbTip",
        ],
        *[
            [
                f"{prefix}{finger}FingerMetacarpal",
                f"{prefix}{finger}FingerKnuckle",
                f"{prefix}{finger}FingerIntermediateBase",
                f"{prefix}{finger}FingerIntermediateTip",
                f"{prefix}{finger}FingerTip",
            ]
            for finger in ("Index", "Middle", "Ring", "Little")
        ],
    ]


def is_egodex_episode(path: str | Path) -> bool:
    """Return True for a paired EgoDex ``*.hdf5`` + sibling ``*.mp4`` episode."""
    source = Path(path)
    return (
        source.is_file()
        and source.suffix.lower() == ".hdf5"
        and source.with_suffix(".mp4").is_file()
    )


def _egodex_instruction(attributes: dict[str, Any]) -> str:
    """Match apple/ml-egodex ``simple_dataset`` language attribute selection."""
    llm_type = str(attributes.get("llm_type", "")).strip().lower()
    if llm_type == "reversible" and str(attributes.get("which_llm_description", "1")) == "2":
        text = attributes.get("llm_description2", attributes.get("llm_description", ""))
    else:
        text = attributes.get("llm_description", attributes.get("task", ""))
    return str(text or attributes.get("task", "") or "")


def _egodex_needed_joint_names() -> list[str]:
    names: list[str] = []
    for side in ("left", "right"):
        for chain in _egodex_chain_names(side):
            names.extend(chain)
    return names


def read_egodex_reference(path: str | Path) -> ReferenceFrameGeometry:
    """Read one EgoDex episode into camera-aligned reference geometry.

    Wrists and chain points stay in the ARKit origin (session world), matching
    EgoPro/OpenAoE ``*_reference`` readers. ``transforms/camera`` is the official
    world-from-camera extrinsic; ``camera_from_world = inv(camera)``. RGB is the
    sibling ``*.mp4`` indexed by integer frame id (no fabricated gap frames).
    """
    try:
        import h5py
    except ImportError as error:
        raise ImportError("Reading EgoDex requires `h5py`") from error

    source_path = Path(path)
    if not is_egodex_episode(source_path):
        raise FileNotFoundError(
            f"EgoDex episode incomplete at {source_path}: need paired .hdf5 + .mp4"
        )
    video_path = source_path.with_suffix(".mp4")

    with h5py.File(source_path, "r") as file:
        transforms = file["transforms"]
        if "camera" not in transforms:
            raise ValueError(f"EgoDex HDF5 missing transforms/camera: {source_path}")
        length = int(transforms["camera"].shape[0])
        world_from_camera = np.asarray(transforms["camera"], dtype=np.float64)
        camera_from_world = np.linalg.inv(world_from_camera)
        intrinsic = np.asarray(file["camera"]["intrinsic"], dtype=np.float64)
        attributes = {key: _decode_scalar(value) for key, value in file.attrs.items()}
        confidence_group = file.get("confidences")
        world_joints: dict[str, np.ndarray] = {}
        confidences: dict[str, np.ndarray] = {}
        for name in _egodex_needed_joint_names():
            if name in transforms:
                world_joints[name] = np.asarray(transforms[name], dtype=np.float64)
            if confidence_group is not None and name in confidence_group:
                confidences[name] = np.asarray(confidence_group[name], dtype=np.float64)

    if intrinsic.shape != (3, 3):
        raise ValueError(f"Unexpected EgoDex intrinsic shape {intrinsic.shape}")

    timestamps = np.arange(length, dtype=np.float64) / 30.0
    wrist_poses = np.full((length, 2, 7), np.nan, dtype=np.float64)
    wrist_valid = np.zeros((length, 2), dtype=bool)
    joints_ref = np.full((length, 2, 5, 5, 3), np.nan, dtype=np.float64)
    joints_local = np.full((length, 2, 5, 5, 3), np.nan, dtype=np.float64)
    confidence = np.zeros((length, 2, 5, 5), dtype=np.float64)
    camera_valid = np.all(np.isfinite(camera_from_world), axis=(1, 2))
    if length:
        camera_valid &= np.abs(np.linalg.det(camera_from_world[:, :3, :3]) - 1.0) < 1e-2

    for hand_index, side in enumerate(("left", "right")):
        chains = _egodex_chain_names(side)
        hand_name = chains[0][0]
        if hand_name not in world_joints:
            continue
        world_from_hand = world_joints[hand_name]
        wrist_poses[:, hand_index] = matrix_to_pose_wxyz(world_from_hand)
        hand_ok = np.all(np.isfinite(world_from_hand), axis=(1, 2))
        if hand_name in confidences:
            hand_ok &= confidences[hand_name] > 0.0
        wrist_valid[:, hand_index] = hand_ok
        valid_indices = np.flatnonzero(hand_ok)
        if len(valid_indices):
            wrist_poses[valid_indices, hand_index] = ensure_quaternion_continuity(
                wrist_poses[valid_indices, hand_index]
            )

        for frame in valid_indices.tolist():
            points = np.full((5, 5, 3), np.nan, dtype=np.float64)
            weights = np.zeros((5, 5), dtype=np.float64)
            for finger_index, names in enumerate(chains):
                for point_index, name in enumerate(names):
                    if name not in world_joints:
                        continue
                    points[finger_index, point_index] = world_joints[name][frame, :3, 3]
                    if name in confidences:
                        weights[finger_index, point_index] = float(
                            np.clip(confidences[name][frame], 0.0, 1.0)
                        )
                    else:
                        weights[finger_index, point_index] = 1.0
            joints_ref[frame, hand_index] = points
            local, ok = canonicalize_hand_points(
                points,
                wrist_poses[frame, hand_index, :3],
                weights,
                knuckle_index=1,
            )
            if ok:
                joints_local[frame, hand_index] = local
                confidence[frame, hand_index] = weights

    intrinsics = np.asarray(
        [intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]],
        dtype=np.float64,
    )
    return ReferenceFrameGeometry(
        timestamps=timestamps,
        wrist_poses_ref=wrist_poses,
        wrist_valid=wrist_valid,
        joints_ref=joints_ref,
        joint_confidence=confidence,
        joints_local=joints_local,
        camera_from_world=camera_from_world,
        camera_valid=camera_valid,
        intrinsics=intrinsics,
        source="egodex",
        instruction=_egodex_instruction(attributes),
        metadata={
            "source_path": str(source_path),
            "source_fps": 30.0,
            "coordinate_frame": "egodex_arkit_origin",
            "hand_local_frame": "arkit_hand_anchor_knuckle_index",
            "camera_frame": "opencv_optical",
            "camera_from_world_composition": "inv(transforms/camera)",
            "camera_from_world_status": "matches_apple_ml_egodex_convert_to_camera_frame",
            "source_readonly": True,
            "video_path": str(video_path),
            "task": attributes.get("task"),
            "attributes": attributes,
            "source_image_height": 1080,
            "source_image_width": 1920,
        },
    )


