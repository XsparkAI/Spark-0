"""Egoverse zarr v3 human-hand reader for camera-aligned Spark 54-D retargeting.

Egoverse episodes are zarr v3 stores aggregated from many capture labs. The
human subsets (``human_bimanual`` / ``human_right_arm`` / ``lw_bimanual``)
carry, per episode:

- ``{left,right}.obs_keypoints`` ``(T, 63)``: 21 hand keypoints in MediaPipe /
  MANO order (0 = wrist, then four points per finger thumb..pinky), in the
  SLAM world frame (gravity +z, meters).  Some capture stacks (``scale``)
  ship a non-metric SLAM scale; :func:`read_egoverse_reference` detects this
  from convention-stable finger bone lengths against the wujihand2 canonical
  hand and renormalises all world translations (see
  ``WORLD_SCALE_DEADBAND``).
- ``{left,right}.obs_wrist_pose`` and ``obs_head_pose`` ``(T, 7)`` as
  ``[x, y, z, qw, qx, qy, qz]`` in the same world frame (verified by
  reprojection: ``inv(head)`` maps world keypoints onto ``images.front_1``);
- ``images.front_1``: standard JPEG bytes (true RGB after ``cv2.imdecode``);
- root ``zarr.json`` attributes with ``total_frames``, ``fps``,
  ``task_description`` / ``task``, pinhole ``intrinsics.front_1`` and
  ``annotation_v1`` language segments.

The ``lightwheel`` subset ships a different capture layout (MCAP directory):
``pose.json`` (per-frame head/wrist/hand-21 poses), ``metadata.json`` and
``head_{left,right}_camera/`` with fisheye calibration
(``distortion_model`` + coefficients), per-frame optical-axis ``R_w2c`` /
``t_w2c`` (1-based frame numbers), ``undistorted_intrinsics`` and the
pre-undistorted ``head_*_camera_undistorted.mp4`` used as front RGB
(verified by keypoint reprojection; no FLIP).  Both layouts feed the same
:func:`read_egoverse_reference` contract.

Robot-teleop subsets (``cmd_joints`` / ``cmd_gripper``) are NOT handled here;
they belong to the source-robot retarget family, not this reader.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ...schema import ReferenceFrameGeometry
from ...transforms import (
    canonicalize_hand_points,
    ensure_quaternion_continuity,
    pose_wxyz_to_matrix,
)

# 21-point layout: 0 = wrist, then four points per finger (thumb..pinky).
_FINGER_KEYPOINTS = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)
# Chain point 0 of the four fingers (metacarpal base, matching the EgoDex
# ``*FingerMetacarpal`` slot) is extrapolated this fraction along the
# wrist->MCP segment; ARKit metacarpal bases sit on that segment near the
# wrist.  Only chain points >= 1 (knuckle_index=1) anchor the palm frame, so
# this constant mainly shapes the IK root segment.
METACARPAL_ROOT_FRACTION = 0.4

# Non-metric SLAM stacks (e.g. ``scale``) share one unknown world scale factor
# across keypoints / wrist / head poses.  Finger phalanx lengths are
# convention-stable, so the observed/URDF-canonical bone-length ratio
# estimates that factor.  Metric stacks measure ~0.85-0.9 (human hands vs the
# URDF prior); only factors outside this deadband trigger renormalisation so
# genuine metric episodes stay untouched.
WORLD_SCALE_DEADBAND = (0.75, 1.35)

_FINGER_BONE_REFERENCE_M: float | None = None


def _zarr_attributes(path: Path) -> dict[str, Any]:
    payload = json.loads((path / "zarr.json").read_text(encoding="utf-8"))
    return payload.get("attributes", {})


def is_lightwheel_episode(path: str | Path) -> bool:
    """True for a lightwheel capture directory (MCAP layout, undistorted RGB)."""
    source = Path(path)
    camera = source / "head_left_camera"
    return (
        source.is_dir()
        and (source / "pose.json").is_file()
        and (camera / "head_left_camera_params.json").is_file()
        and (camera / "head_left_camera_undistorted.mp4").is_file()
    )


def is_egoverse_episode(path: str | Path) -> bool:
    """True for a human-hand egoverse episode (zarr store or lightwheel dir)."""
    source = Path(path)
    if is_lightwheel_episode(source):
        return True
    if not source.is_dir() or not source.name.endswith(".zarr"):
        return False
    if not (source / "zarr.json").is_file():
        return False
    return (
        (source / "left.obs_keypoints" / "zarr.json").is_file()
        or (source / "right.obs_keypoints" / "zarr.json").is_file()
    )


def _as_bytes(value: Any) -> bytes:
    """Unwrap zarr v3 variable-length-bytes scalar (nested 0-d object arrays)."""
    while isinstance(value, np.ndarray):
        value = value.item()
    return value


def _open_store(path: Path):
    try:
        import zarr  # type: ignore[import-not-found]
    except ImportError as error:
        raise ImportError("Reading egoverse requires the `zarr` package") from error
    return zarr.open(path, mode="r")


def egoverse_source_hw(path: str | Path) -> tuple[int, int]:
    """Return ``(height, width)`` of the front RGB source (JPEGs or mp4)."""
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as error:
        raise ImportError("Reading egoverse RGB requires cv2") from error
    source = Path(path)
    if is_lightwheel_episode(source):
        capture = cv2.VideoCapture(
            str(source / "head_left_camera" / "head_left_camera_undistorted.mp4")
        )
        ok, image = capture.read()
        capture.release()
        if not ok or image is None:
            raise ValueError(f"Could not decode lightwheel head RGB at {path}")
        return int(image.shape[0]), int(image.shape[1])
    store = _open_store(source)
    image = cv2.imdecode(
        np.frombuffer(_as_bytes(store["images.front_1"][0]), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise ValueError(f"Could not decode egoverse front RGB at {path}")
    return int(image.shape[0]), int(image.shape[1])


def egoverse_rgb_timestamps(path: str | Path) -> np.ndarray | None:
    """Zero-based monotonic ``images.front_1`` timestamps (s), or None.

    Mirrors ``read_egoverse_reference`` exactly: slice ``obs_rgb_timestamps_ns``
    to the same effective length (min of declared ``total_frames`` and keypoint
    array lengths) and require strict monotonicity there. Returns None when the
    stream is absent or non-monotonic on that slice (tail placeholder /
    wrap-around, as in microagi); callers then fall back to a uniform
    ``arange / fps`` grid. Monotonic streams may still drop frames (aria), so
    consumers must map output times by nearest neighbour, not ``t * fps``.
    """
    source_path = Path(path)
    if is_lightwheel_episode(source_path):
        # Uniform 30 fps mp4 grid; pose frames and video frames align 1:1.
        return None
    store = _open_store(source_path)
    if "obs_rgb_timestamps_ns" not in store:
        return None
    declared = int(_zarr_attributes(source_path).get("total_frames", 0))
    key_names = [
        name
        for name in ("left.obs_keypoints", "right.obs_keypoints")
        if name in store
    ]
    if not key_names:
        return None
    array_length = min(int(store[name].shape[0]) for name in key_names)
    length = min(declared, array_length) if declared else array_length
    raw = np.asarray(
        store["obs_rgb_timestamps_ns"][:length], dtype=np.float64
    ) / 1e9
    if raw.size < 2 or not np.all(np.diff(raw) > 0):
        return None
    return raw - raw[0]


def load_egoverse_head_frames(
    path: str | Path, indices: list[int]
) -> dict[int, np.ndarray]:
    """Decode front RGB frames at source indices as BGR arrays."""
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as error:
        raise ImportError("Reading egoverse RGB requires cv2") from error
    source = Path(path)
    if is_lightwheel_episode(source):
        capture = cv2.VideoCapture(
            str(source / "head_left_camera" / "head_left_camera_undistorted.mp4")
        )
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        out: dict[int, np.ndarray] = {}
        for index in sorted(indices):
            if index < 0 or index >= total:
                continue
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, image = capture.read()
            if ok and image is not None:
                out[int(index)] = image
        capture.release()
        return out
    store = _open_store(source)
    frames = store["images.front_1"][indices]
    out: dict[int, np.ndarray] = {}
    for index, payload in zip(indices, frames):
        image = cv2.imdecode(
            np.frombuffer(_as_bytes(payload), dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if image is not None:
            out[int(index)] = image
    return out


def egoverse_head_source_topic(path: str | Path) -> str:
    """Provenance topic of the front RGB stream for ``cam_head`` attrs."""
    if is_lightwheel_episode(Path(path)):
        return "lightwheel_head_left_undistorted"
    return "egoverse_front_1"


def _egoverse_instruction(attributes: dict[str, Any], store: Any) -> str:
    for key in ("task_description", "task"):
        value = attributes.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    try:
        segments = store["annotations"][:]
    except KeyError:
        return ""
    for payload in segments:
        try:
            record = json.loads(_as_bytes(payload).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            continue
        text = record.get("text") if isinstance(record, dict) else None
        if isinstance(text, str) and text.strip():
            return text.strip()
    return ""


def _finger_bone_reference_m() -> float:
    """Total wujihand2 canonical phalanx length (15 wrist-free segments)."""
    global _FINGER_BONE_REFERENCE_M
    if _FINGER_BONE_REFERENCE_M is None:
        from ...retargeting.wuji2 import WujiHandModel

        model = WujiHandModel("left", variant="wujihand2")
        points = model.forward_points(
            np.zeros(20, dtype=np.float64), canonical=True
        ).reshape(5, 5, 3)
        _FINGER_BONE_REFERENCE_M = float(
            sum(
                np.linalg.norm(points[finger, seg + 1] - points[finger, seg])
                for finger in range(5)
                for seg in range(1, 4)
            )
        )
    return _FINGER_BONE_REFERENCE_M


def egoverse_world_scale(hands: list[np.ndarray]) -> float:
    """World scale factor of an episode, or 1.0 inside the deadband.

    ``hands`` holds ``(T, 21, 3)`` keypoint arrays per available hand.  The
    estimate is the median over frames/hands of the observed total phalanx
    length divided by :func:`_finger_bone_reference_m`; factors inside
    ``WORLD_SCALE_DEADBAND`` are treated as metric and left as-is.
    """
    reference = _finger_bone_reference_m()
    ratios: list[np.ndarray] = []
    for keypoints in hands:
        finite = np.all(np.isfinite(keypoints), axis=(1, 2))
        if not finite.any():
            continue
        segments = [
            np.diff(keypoints[finite][:, list(finger), :], axis=1)
            for finger in _FINGER_KEYPOINTS
        ]
        totals = np.concatenate(
            [np.linalg.norm(segment, axis=2) for segment in segments], axis=1
        ).sum(axis=1)
        ratios.append(totals / reference)
    if not ratios:
        return 1.0
    scale = float(np.median(np.concatenate(ratios)))
    if not np.isfinite(scale) or scale <= 0.0:
        return 1.0
    if WORLD_SCALE_DEADBAND[0] <= scale <= WORLD_SCALE_DEADBAND[1]:
        return 1.0
    return scale


def _chain_from_keypoints(
    keypoints: np.ndarray, wrist_position: np.ndarray
) -> np.ndarray:
    """Build the 5x5 chain (metacarpal-base root) from 21 keypoints."""
    points = np.full((5, 5, 3), np.nan, dtype=np.float64)
    points[0, 0] = wrist_position
    points[0, 1:] = keypoints[list(_FINGER_KEYPOINTS[0])]
    for finger, (mcp, pip, dip, tip) in enumerate(_FINGER_KEYPOINTS[1:], start=1):
        points[finger, 1] = keypoints[mcp]
        points[finger, 2] = keypoints[pip]
        points[finger, 3] = keypoints[dip]
        points[finger, 4] = keypoints[tip]
        points[finger, 0] = wrist_position + METACARPAL_ROOT_FRACTION * (
            keypoints[mcp] - wrist_position
        )
    return points


def _fill_hand_geometry(
    hand_index: int,
    keypoints: np.ndarray,
    wrist_poses: np.ndarray,
    wrist_valid: np.ndarray,
    joints_ref: np.ndarray,
    joints_local: np.ndarray,
    confidence: np.ndarray,
) -> None:
    """Wrist continuity + 5x5 chains + palm-canonical joints for one hand."""
    wrist_positions = wrist_poses[:, hand_index, :3]
    hand_ok = np.all(np.isfinite(wrist_poses[:, hand_index]), axis=1)
    hand_ok &= np.all(np.isfinite(keypoints), axis=(1, 2))
    wrist_valid[:, hand_index] = hand_ok
    valid_indices = np.flatnonzero(hand_ok)
    if len(valid_indices):
        wrist_poses[valid_indices, hand_index] = ensure_quaternion_continuity(
            wrist_poses[valid_indices, hand_index]
        )
    for frame in valid_indices.tolist():
        points = _chain_from_keypoints(keypoints[frame], wrist_positions[frame])
        weights = np.ones((5, 5), dtype=np.float64)
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


def _new_hand_arrays(length: int) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    wrist_poses = np.full((length, 2, 7), np.nan, dtype=np.float64)
    wrist_valid = np.zeros((length, 2), dtype=bool)
    joints_ref = np.full((length, 2, 5, 5, 3), np.nan, dtype=np.float64)
    joints_local = np.full((length, 2, 5, 5, 3), np.nan, dtype=np.float64)
    confidence = np.zeros((length, 2, 5, 5), dtype=np.float64)
    return wrist_poses, wrist_valid, joints_ref, joints_local, confidence


def read_egoverse_reference(path: str | Path) -> ReferenceFrameGeometry:
    """Read one human-hand egoverse episode into reference geometry.

    Wrists, chains and head poses stay in the egoverse SLAM world frame;
    ``camera_from_world = inv(obs_head_pose)`` (OpenCV optical axes, verified
    by keypoint reprojection onto ``images.front_1``).  Lightwheel capture
    directories dispatch to :func:`_read_lightwheel_reference`.
    """
    source_path = Path(path)
    if is_lightwheel_episode(source_path):
        return _read_lightwheel_reference(source_path)
    if not is_egoverse_episode(source_path):
        raise FileNotFoundError(f"Not a human-hand egoverse episode: {source_path}")
    attributes = _zarr_attributes(source_path)
    store = _open_store(source_path)

    declared = int(attributes.get("total_frames", 0))
    key_names = [
        name
        for name in ("left.obs_keypoints", "right.obs_keypoints")
        if name in store
    ]
    array_length = min(int(store[name].shape[0]) for name in key_names)
    length = min(declared, array_length) if declared else array_length
    if length <= 0:
        raise ValueError(f"Egoverse episode has no frames: {source_path}")
    fps = float(attributes.get("fps", 30.0))

    rgb_timestamps = egoverse_rgb_timestamps(source_path)
    if rgb_timestamps is not None:
        timestamps = rgb_timestamps
    else:
        timestamps = np.arange(length, dtype=np.float64) / fps

    head = np.asarray(store["obs_head_pose"][:length], dtype=np.float64)
    world_scale = egoverse_world_scale(
        [
            np.asarray(store[name][:length], dtype=np.float64).reshape(
                length, 21, 3
            )
            for name in key_names
        ]
    )
    if world_scale != 1.0:
        head[:, :3] /= world_scale
    world_from_head = np.stack(
        [pose_wxyz_to_matrix(head[frame]) for frame in range(length)]
    )
    camera_from_world = np.linalg.inv(world_from_head)
    camera_valid = np.all(np.isfinite(camera_from_world), axis=(1, 2))
    camera_valid &= np.abs(np.linalg.det(camera_from_world[:, :3, :3]) - 1.0) < 1e-2

    intrinsic = np.asarray(attributes["intrinsics"]["front_1"], dtype=np.float64)
    if intrinsic.shape not in ((3, 3), (3, 4)):
        raise ValueError(f"Unexpected egoverse intrinsics shape {intrinsic.shape}")
    intrinsic = intrinsic[:3, :3]
    intrinsics = np.asarray(
        [intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]],
        dtype=np.float64,
    )

    wrist_poses, wrist_valid, joints_ref, joints_local, confidence = (
        _new_hand_arrays(length)
    )

    for hand_index, side in enumerate(("left", "right")):
        key_name = f"{side}.obs_keypoints"
        if key_name not in store:
            continue
        keypoints = np.asarray(
            store[key_name][:length], dtype=np.float64
        ).reshape(length, 21, 3)
        if world_scale != 1.0:
            keypoints /= world_scale
        wrist_name = f"{side}.obs_wrist_pose"
        if wrist_name in store:
            wrist_poses[:, hand_index] = np.asarray(
                store[wrist_name][:length], dtype=np.float64
            )
            if world_scale != 1.0:
                wrist_poses[:, hand_index, :3] /= world_scale
        else:
            wrist_poses[:, hand_index, :3] = keypoints[:, 0]
            wrist_poses[:, hand_index, 3:] = np.array([1.0, 0.0, 0.0, 0.0])
        _fill_hand_geometry(
            hand_index,
            keypoints,
            wrist_poses,
            wrist_valid,
            joints_ref,
            joints_local,
            confidence,
        )

    source_hw = egoverse_source_hw(source_path)
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
        source="egoverse",
        instruction=_egoverse_instruction(attributes, store),
        metadata={
            "source_path": str(source_path),
            "source_fps": fps,
            "coordinate_frame": "egoverse_slam_world",
            "hand_local_frame": "wrist_root_metacarpal_extrapolated_knuckle_index",
            "camera_frame": "opencv_optical",
            "camera_from_world_composition": "inv(obs_head_pose)",
            "camera_from_world_status": "verified_by_keypoint_reprojection",
            "pose_quaternion_order": "wxyz",
            "keypoints_order": "mediapipe_mano_21",
            "metacarpal_root_fraction": METACARPAL_ROOT_FRACTION,
            "world_scale_factor": world_scale,
            "world_scale_deadband": list(WORLD_SCALE_DEADBAND),
            "source_readonly": True,
            "embodiment": attributes.get("embodiment"),
            "task_name": attributes.get("task_name"),
            "source_image_height": source_hw[0],
            "source_image_width": source_hw[1],
        },
    )


def _read_lightwheel_reference(source_path: Path) -> ReferenceFrameGeometry:
    """Read one lightwheel capture directory into reference geometry.

    Front RGB is the shipped fisheye-undistorted ``head_left_camera`` video
    with ``undistorted_intrinsics``; the per-frame ``R_w2c`` / ``t_w2c`` in
    ``head_left_camera_params.json`` are OpenCV optical axes directly
    (verified by keypoint reprojection; no FLIP) with 1-based frame numbers
    against the 0-based ``pose.json`` frame list.
    """
    poses = json.loads((source_path / "pose.json").read_text(encoding="utf-8"))
    records = poses["frames"]
    length = len(records)
    if length <= 0:
        raise ValueError(f"Lightwheel episode has no frames: {source_path}")

    metadata = json.loads(
        (source_path / "metadata.json").read_text(encoding="utf-8")
    )
    fps = 30.0
    for device in metadata.get("capture_devices", {}).get("devices", []):
        if device.get("device_type") == "head_left_camera":
            fps = float(device.get("fps", 30.0))
    timestamps = np.arange(length, dtype=np.float64) / fps

    keypoints_per_hand: dict[int, np.ndarray] = {}
    wrist_poses, wrist_valid, joints_ref, joints_local, confidence = (
        _new_hand_arrays(length)
    )
    for hand_index, side in enumerate(("left", "right")):
        keypoints = np.full((length, 21, 3), np.nan, dtype=np.float64)
        for frame, record in enumerate(records):
            entries = record.get(f"{side}_hand", [])
            if len(entries) != 21:
                continue
            for index, entry in enumerate(entries):
                keypoints[frame, index] = (entry["x"], entry["y"], entry["z"])
            wrist = entries[0]
            quaternion = wrist["quat"]
            wrist_poses[frame, hand_index] = (
                wrist["x"],
                wrist["y"],
                wrist["z"],
                quaternion["w"],
                quaternion["x"],
                quaternion["y"],
                quaternion["z"],
            )
        keypoints_per_hand[hand_index] = keypoints

    world_scale = egoverse_world_scale(list(keypoints_per_hand.values()))
    if world_scale != 1.0:
        for keypoints in keypoints_per_hand.values():
            keypoints /= world_scale
        wrist_poses[:, :, :3] /= world_scale

    params = json.loads(
        (
            source_path / "head_left_camera" / "head_left_camera_params.json"
        ).read_text(encoding="utf-8")
    )
    undistorted = params["undistorted_intrinsics"]
    intrinsics = np.asarray(
        [undistorted["fx"], undistorted["fy"], undistorted["cx"], undistorted["cy"]],
        dtype=np.float64,
    )
    camera_from_world = np.full((length, 4, 4), np.nan, dtype=np.float64)
    for record in params["frames"]:
        index = int(record["frame"]) - 1
        if not 0 <= index < length:
            continue
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = np.asarray(record["R_w2c"], dtype=np.float64)
        matrix[:3, 3] = np.asarray(record["t_w2c"], dtype=np.float64)
        if world_scale != 1.0:
            matrix[:3, 3] /= world_scale
        camera_from_world[index] = matrix
    camera_valid = np.all(np.isfinite(camera_from_world), axis=(1, 2))
    camera_valid &= np.abs(np.linalg.det(camera_from_world[:, :3, :3]) - 1.0) < 1e-2

    for hand_index in (0, 1):
        _fill_hand_geometry(
            hand_index,
            keypoints_per_hand[hand_index],
            wrist_poses,
            wrist_valid,
            joints_ref,
            joints_local,
            confidence,
        )

    task_info = metadata.get("task_info", {})
    source_hw = egoverse_source_hw(source_path)
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
        source="egoverse",
        instruction=str(task_info.get("task_description", "") or "").strip(),
        metadata={
            "source_path": str(source_path),
            "source_fps": fps,
            "coordinate_frame": "lightwheel_capture_world",
            "hand_local_frame": "wrist_root_metacarpal_extrapolated_knuckle_index",
            "camera_frame": "opencv_optical",
            "camera_from_world_composition": "head_left_camera R_w2c/t_w2c",
            "camera_from_world_status": "verified_by_keypoint_reprojection",
            "pose_quaternion_order": "wxyz",
            "keypoints_order": "mediapipe_mano_21",
            "metacarpal_root_fraction": METACARPAL_ROOT_FRACTION,
            "world_scale_factor": world_scale,
            "world_scale_deadband": list(WORLD_SCALE_DEADBAND),
            "source_readonly": True,
            "embodiment": "lightwheel",
            "task_name": task_info.get("task_name"),
            "rgb_source": "head_left_camera_undistorted",
            "fisheye_distortion_model": params.get("distortion_model"),
            "source_image_height": source_hw[0],
            "source_image_width": source_hw[1],
        },
    )
