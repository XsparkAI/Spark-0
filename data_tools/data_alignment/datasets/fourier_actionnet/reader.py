"""Fourier ActionNet episode discovery and source-geometry reader."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from ..common import _decode_scalar
from ...schema import UnifiedHumanMotion
from ...transforms import ensure_quaternion_continuity


def fourier_rgb_path(path: str | Path) -> Path:
    """Sibling ``<stem>/top/rgb.mp4`` for an ActionNet episode HDF5."""
    source = Path(path)
    return source.parent / source.stem / "top" / "rgb.mp4"


def is_fourier_episode(path: str | Path) -> bool:
    """Return True for ActionNet ``*.hdf5`` + ``<stem>/top/rgb.mp4``.

    Path-only probe (no HDF5 open) so batch discovery stays cheap. Key layout is
    validated inside :func:`read_fourier_source`.
    """
    source = Path(path)
    if not (source.is_file() and source.suffix.lower() == ".hdf5"):
        return False
    # EgoDex uses a sibling ``*.mp4``; ActionNet keeps RGB under ``<stem>/top/``.
    if source.with_suffix(".mp4").is_file():
        return False
    return fourier_rgb_path(source).is_file()


def read_fourier_source(path: str | Path) -> UnifiedHumanMotion:
    """Read one Fourier ActionNet episode into source-robot hand geometry.

    EE poses come from ``state/pose`` (GR1 end-link, source robot frame). Hand
    5×5 chains are Fourier-hand FK of ``state/hand`` command units, then
    anatomical-palm canonicalization for Wuji IK. Does **not** apply Spark
    camera alignment.
    """
    try:
        import h5py
    except ImportError as error:
        raise ImportError("Reading Fourier ActionNet requires h5py") from error

    from .hand import (
        FourierHandModel,
        build_hand_geometry_frame,
        decode_state_pose_ee,
        decode_state_pose_head,
        resolve_fourier_hand_space,
    )

    source_path = Path(path)
    if not is_fourier_episode(source_path):
        raise ValueError(
            f"Not a Fourier ActionNet episode (need *.hdf5 + <stem>/top/rgb.mp4): "
            f"{source_path}"
        )
    rgb_path = fourier_rgb_path(source_path)
    ts_json = rgb_path.parent / "timestamps.json"
    video_frames = None
    if ts_json.is_file():
        video_frames = len(json.loads(ts_json.read_text(encoding="utf-8")))
    if video_frames is None:
        try:
            import cv2
            capture = cv2.VideoCapture(str(rgb_path))
            if capture.isOpened():
                count = int(round(float(capture.get(cv2.CAP_PROP_FRAME_COUNT))))
                video_frames = count if count > 1 else None
            capture.release()
        except Exception:
            video_frames = None

    with h5py.File(source_path, "r") as file:
        timestamps_abs = np.asarray(file["timestamp"][()], dtype=np.float64)
        hand_cmd = np.asarray(file["state/hand"][()], dtype=np.float64)
        pose = np.asarray(file["state/pose"][()], dtype=np.float64)
        attrs = {key: _decode_scalar(file.attrs[key]) for key in file.attrs}
    if timestamps_abs.ndim != 1 or hand_cmd.shape != (len(timestamps_abs), 12):
        raise ValueError(
            f"Fourier episode has inconsistent lengths: timestamp={timestamps_abs.shape}, "
            f"hand={hand_cmd.shape}"
        )
    if pose.shape[0] != len(timestamps_abs) or pose.shape[1] < 18:
        raise ValueError(f"Fourier state/pose shape invalid: {pose.shape}")

    timestamps = timestamps_abs - timestamps_abs[0]
    ee = decode_state_pose_ee(pose)
    for hand in range(2):
        valid = np.all(np.isfinite(ee[:, hand]), axis=1)
        if np.any(valid):
            ee[valid, hand] = ensure_quaternion_continuity(ee[valid, hand])
    head_poses = decode_state_pose_head(pose)
    head_valid = np.all(np.isfinite(head_poses), axis=1)
    if np.any(head_valid):
        head_poses[head_valid] = ensure_quaternion_continuity(head_poses[head_valid])

    models = {
        "left": FourierHandModel("left"),
        "right": FourierHandModel("right"),
    }
    hand_space = resolve_fourier_hand_space(
        hand_cmd, str(attrs.get("task_name", "") or "")
    )
    length = len(timestamps)
    joints_world = np.full((length, 2, 5, 5, 3), np.nan, dtype=np.float64)
    joints_local = np.full((length, 2, 5, 5, 3), np.nan, dtype=np.float64)
    confidence = np.zeros((length, 2, 5, 5), dtype=np.float64)
    wrist_valid = np.zeros((length, 2), dtype=bool)
    for hand_index, side in enumerate(("left", "right")):
        model = models[side]
        cmd = hand_cmd[:, hand_index * 6 : (hand_index + 1) * 6]
        for index in range(length):
            if not np.all(np.isfinite(ee[index, hand_index])):
                continue
            if not np.all(np.isfinite(cmd[index])):
                continue
            world, local, conf, ok = build_hand_geometry_frame(
                ee[index, hand_index],
                cmd[index],
                model,
                space=hand_space,
            )
            joints_world[index, hand_index] = world
            confidence[index, hand_index] = conf
            wrist_valid[index, hand_index] = True
            if ok:
                joints_local[index, hand_index] = local
            else:
                confidence[index, hand_index] = 0.0

    instruction = str(attrs.get("task_name", "") or "")
    source_hz = (
        float(length - 1) / float(timestamps[-1])
        if length > 1 and timestamps[-1] > 0
        else 60.0
    )
    return UnifiedHumanMotion(
        timestamps=timestamps,
        wrist_poses=ee,
        wrist_valid=wrist_valid,
        joints_local=joints_local,
        joint_confidence=confidence,
        source="fourier_actionnet",
        instruction=instruction,
        metadata={
            "source_path": str(source_path.resolve()),
            "rgb_path": str(rgb_path.resolve()),
            "source_fps": source_hz,
            "video_fps": 30.0,
            "video_frame_count": video_frames,
            "coordinate_frame": "source_robot_as_is",
            "ee_frame": "fourier_end_link",
            "hand_command_space": hand_space,
            "joints_world": joints_world,
            "head_poses": head_poses,
            "head_valid": head_valid,
            "head_link": "head_yaw_link",
            "episode_id": attrs.get("episode_id"),
            "episode_duration": attrs.get("episode_duration"),
            "source_readonly": True,
        },
    )


def read_fourier_reference(path: str | Path):
    """Read ActionNet and expose source-robot geometry via the shared schema."""
    from ..common import source_motion_to_reference
    motion = read_fourier_source(path)
    nominal = np.asarray([563.0, 563.0, 640.0, 400.0], dtype=np.float64)
    return source_motion_to_reference(motion, intrinsics=nominal)
