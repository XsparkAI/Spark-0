"""AgiBotWorld-Beta(hand) → source-robot geometry for Spark retargeting.

Reads ``proprio_stats.h5``:
- Spark EE **position** = ``state/end`` flange translation.
- Spark EE **orientation** + Inspire FK parent = flange × the built-in Inspire
  mount rotation :data:`INSPIRE_MOUNT_ROTATION`. Dropping that R flips the
  palm; applying the mount translation too would slide the origin off the
  flange, so only the rotation is baked in.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .hand import InspireHandModel
from ...schema import UnifiedHumanMotion
from ...transforms import (
    canonicalize_hand_points,
    ensure_quaternion_continuity,
    matrix_to_pose_wxyz,
    pose_wxyz_to_matrix,
    transform_points,
)

_OPENCV_FROM_SPARK = np.diag([1.0, -1.0, -1.0, 1.0])

# ``R_handbase_from_flange``, right-multiplied onto ``world_from_flange``.
# Solved once by manual fingertip PnP on the Beta dexhand rig: left from frame
# 195 (9.53 px RMS), right from frames 400/330/0 (26.89 px RMS). The mount is
# physically skewed, so these are not axis-aligned (left sits 42 deg off the
# nearest signed permutation) and cannot be written as plain axis swaps.
INSPIRE_MOUNT_ROTATION: dict[str, np.ndarray] = {
    "left": np.array(
        [
            [0.7454988551475814, -0.3426037979612214, 0.5717115484194737],
            [-0.6039560172956808, 0.015554104601504637, 0.7968658601059433],
            [-0.2819017313639363, -0.9393512162405455, -0.19532205815376597],
        ],
        dtype=np.float64,
    ),
    "right": np.array(
        [
            [0.045205876193839756, -0.12570190148775867, 0.9910375677641636],
            [-0.9985636379547254, -0.03424570140642699, 0.04120549586895117],
            [0.028759167445730094, -0.9914768095611435, -0.12706945498539288],
        ],
        dtype=np.float64,
    ),
}
MOUNT_SOURCE = "builtin_inspire_pnp_rotation"


def mount_matrix(side: str) -> np.ndarray:
    """Mount rotation as SE(3) with zero translation (flange origin stays put)."""
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = INSPIRE_MOUNT_ROTATION[side]
    return out


def _xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def flange_apply_mount_rotation(
    flange_wxyz: np.ndarray, R_mount: np.ndarray
) -> np.ndarray:
    """(T,7) flange → same xyz, orientation right-multiplied by the mount R."""
    out = np.empty_like(flange_wxyz)
    for i in range(len(flange_wxyz)):
        T_f = pose_wxyz_to_matrix(flange_wxyz[i])
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = T_f[:3, :3] @ R_mount
        T[:3, 3] = T_f[:3, 3]
        out[i] = matrix_to_pose_wxyz(T)
    return out


def parse_agibot_episode(path: str | Path) -> tuple[Path, str, str]:
    """Parse ``…/proprio_stats/<task>/<ep>/proprio_stats.h5`` → root, task, ep."""
    proprio = Path(path).resolve()
    if proprio.name != "proprio_stats.h5":
        raise ValueError(
            f"AgiBot episode input must be proprio_stats.h5, got {proprio.name}"
        )
    ep_dir = proprio.parent
    task_dir = ep_dir.parent
    proprio_root = task_dir.parent
    if proprio_root.name != "proprio_stats":
        raise ValueError(
            f"Expected …/proprio_stats/<task>/<ep>/proprio_stats.h5, got {proprio}"
        )
    root = proprio_root.parent
    return root, task_dir.name, ep_dir.name


def agibot_episode_paths(
    root: Path, task: str, ep: str
) -> dict[str, Path]:
    """Resolve observation / parameter / proprio paths for one episode."""
    return {
        "proprio": root / "proprio_stats" / task / ep / "proprio_stats.h5",
        "videos": root / "observations" / task / ep / "videos",
        "camera": root / "parameters" / task / ep / "parameters" / "camera",
        "head_video": root
        / "observations"
        / task
        / ep
        / "videos"
        / "head_color.mp4",
    }


def is_agibot_episode(path: str | Path) -> bool:
    """True when ``path`` is a proprio_stats.h5 with sibling videos + camera params."""
    try:
        root, task, ep = parse_agibot_episode(path)
    except ValueError:
        return False
    paths = agibot_episode_paths(root, task, ep)
    return (
        paths["proprio"].is_file()
        and paths["head_video"].is_file()
        and (paths["camera"] / "head_intrinsic_params.json").is_file()
    )


def _instruction_for_episode(root: Path, task: str, ep: str) -> str:
    task_json = root / "task_info" / f"task_{task}.json"
    if not task_json.is_file():
        return f"agibot_task_{task}"
    try:
        entries = json.loads(task_json.read_text())
    except json.JSONDecodeError:
        return f"agibot_task_{task}"
    if not isinstance(entries, list):
        return f"agibot_task_{task}"
    ep_int = int(ep) if str(ep).isdigit() else None
    for entry in entries:
        eid = entry.get("episode_id")
        if eid == ep or (ep_int is not None and eid == ep_int):
            name = entry.get("task_name") or entry.get("init_scene_text")
            if name:
                return str(name)
    return f"agibot_task_{task}"


def _video_stream_info(video: Path) -> tuple[int, float]:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as error:
        raise ImportError("Reading AgiBot episodes requires cv2") from error
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open AgiBot video {video}")
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if frames <= 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"{video}: invalid frame count {frames} or fps {fps}")
    return frames, fps


def agibot_video_grid(paths: dict[str, Path]) -> tuple[int, float]:
    """Return ``(frames, fps)`` of the video grid: shortest of head + wrist streams."""
    from .wrist import detect_wrist_variant, resolve_wrist_paths

    videos = [paths["head_video"]]
    try:
        variant = detect_wrist_variant(paths["camera"])
    except FileNotFoundError:
        variant = None
    if variant is not None:
        for side in ("left", "right"):
            resolved = resolve_wrist_paths(paths["videos"], paths["camera"], side, variant)
            if resolved is not None:
                videos.append(resolved[0])
    infos = [_video_stream_info(video) for video in videos]
    fps = infos[0][1]
    for video, (_frames, other_fps) in zip(videos, infos):
        if not np.isclose(other_fps, fps):
            raise ValueError(f"{video}: fps {other_fps} differs from head video fps {fps}")
    return min(frames for frames, _fps in infos), fps


def read_agibot_source(path: str | Path) -> UnifiedHumanMotion:
    """Load AgiBot dexhand; flange xyz + mount R for EE ori / finger parent."""
    try:
        import h5py
    except ImportError as error:
        raise ImportError("Reading AgiBot episodes requires h5py") from error

    root, task, ep = parse_agibot_episode(path)
    paths = agibot_episode_paths(root, task, ep)
    proprio = paths["proprio"]
    if not is_agibot_episode(proprio):
        raise ValueError(f"Not a complete AgiBot episode: {proprio}")

    with h5py.File(proprio, "r") as file:
        eff = np.asarray(file["state/effector/position"][()], dtype=np.float64)
        end_pos = np.asarray(file["state/end/position"][()], dtype=np.float64)
        end_ori = np.asarray(file["state/end/orientation"][()], dtype=np.float64)
        ts = np.asarray(file["timestamp"][()], dtype=np.float64)
    if eff.shape[1] != 12:
        raise ValueError(f"not dexhand: effector {eff.shape}")
    frames = len(eff)
    span_ns = float(ts[-1] - ts[0]) if frames > 1 else 0.0
    measured_fps = 1e9 * (frames - 1) / span_ns if span_ns > 0 else None
    # Proprio samples and video frames are logged 1:1 at a measured ~29.9-30 Hz.
    # Timing them on the measured clock and resampling to exact video fps
    # yields more frames than the videos hold, so stay on the video grid.
    video_frames, fps = agibot_video_grid(paths)
    timestamps = np.arange(frames, dtype=np.float64) / fps

    models = {side: InspireHandModel(side) for side in ("left", "right")}
    wrist = np.empty((frames, 2, 7), dtype=np.float64)
    joints_world = np.full((frames, 2, 5, 5, 3), np.nan)
    joints_local = np.full((frames, 2, 5, 5, 3), np.nan)
    confidence = np.zeros((frames, 2, 5, 5))
    wrist_valid = np.zeros((frames, 2), dtype=bool)

    for hand_index, side in enumerate(("left", "right")):
        model = models[side]
        flange = np.concatenate(
            [end_pos[:, hand_index], _xyzw_to_wxyz(end_ori[:, hand_index])],
            axis=-1,
        )
        mount = mount_matrix(side)
        ee_pose = flange_apply_mount_rotation(flange, INSPIRE_MOUNT_ROTATION[side])
        valid = np.all(np.isfinite(ee_pose), axis=1)
        if np.any(valid):
            ee_pose[valid] = ensure_quaternion_continuity(ee_pose[valid])
        wrist[:, hand_index] = ee_pose
        q = model.agibot_to_q(eff[:, hand_index * 6 : (hand_index + 1) * 6])
        for frame_index in range(frames):
            if not valid[frame_index] or not np.all(np.isfinite(q[frame_index])):
                continue
            local = model.forward_chains_local(q[frame_index])
            world_from_flange = pose_wxyz_to_matrix(flange[frame_index])
            world_from_chain = world_from_flange @ mount
            world = transform_points(
                world_from_chain, local.reshape(-1, 3)
            ).reshape(5, 5, 3)
            conf = np.ones((5, 5), dtype=np.float64)
            canon, ok = canonicalize_hand_points(
                world, ee_pose[frame_index, :3], conf, knuckle_index=1
            )
            joints_world[frame_index, hand_index] = world
            confidence[frame_index, hand_index] = conf
            wrist_valid[frame_index, hand_index] = True
            if ok:
                joints_local[frame_index, hand_index] = canon
            else:
                confidence[frame_index, hand_index] = 0.0

    return UnifiedHumanMotion(
        timestamps=timestamps,
        wrist_poses=wrist,
        wrist_valid=wrist_valid,
        joints_local=joints_local,
        joint_confidence=confidence,
        source="agibot_world_beta_hand",
        instruction=_instruction_for_episode(root, task, ep),
        metadata={
            "source_path": str(proprio.resolve()),
            "dataset_root": str(root.resolve()),
            "source_fps": fps,
            "video_fps": fps,
            "video_frame_count": video_frames,
            "proprio_measured_fps": measured_fps,
            "coordinate_frame": "source_robot_as_is",
            "ee_frame": "agibot_flange_xyz_pnp_mount_R",
            "finger_parent": "pnp_mount_rotation",
            "hand_mount": MOUNT_SOURCE,
            "joints_world": joints_world,
            "task": task,
            "episode": ep,
            "head_video": str(paths["head_video"]),
            "camera_params": str(paths["camera"]),
            "videos_dir": str(paths["videos"]),
            "mount_mode": "rotation_only",
            "source_readonly": True,
        },
    )


def load_agibot_head_camera(
    params_dir: str | Path, n_frames: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return head ``[fx,fy,cx,cy]``, Spark-axes ``(T,7)`` poses, shape ``[480,640,3]``."""
    params_dir = Path(params_dir)
    intr = json.loads((params_dir / "head_intrinsic_params.json").read_text())[
        "intrinsic"
    ]
    intrinsics = np.asarray(
        [intr["fx"], intr["fy"], intr["ppx"], intr["ppy"]], dtype=np.float64
    )
    aligned = json.loads(
        (params_dir / "head_extrinsic_params_aligned.json").read_text()
    )
    poses = []
    for entry in aligned:
        ext = entry["extrinsic"]
        # AgiBotWorld aligned extrinsics store world_from_camera directly
        # (official projection is p_cam = R^T @ (P_world - t)):
        #   rotation_matrix  = camera→world rotation (NOT world→camera)
        #   translation_vector = camera position in world frame
        R_wc = np.asarray(ext["rotation_matrix"], dtype=np.float64)
        t_cam = np.asarray(ext["translation_vector"], dtype=np.float64)
        T_w_cv = np.eye(4, dtype=np.float64)
        T_w_cv[:3, :3] = R_wc
        T_w_cv[:3, 3] = t_cam  # camera position IS world_from_camera t
        T_w_spark = T_w_cv @ _OPENCV_FROM_SPARK
        poses.append(matrix_to_pose_wxyz(T_w_spark))
    poses_arr = np.asarray(poses, dtype=np.float64)
    if len(poses_arr) == 1:
        poses_arr = np.repeat(poses_arr, n_frames, axis=0)
    elif len(poses_arr) < n_frames:
        raise ValueError(
            f"{params_dir}: head extrinsics have {len(poses_arr)} entries, "
            f"expected {n_frames}"
        )
    else:
        poses_arr = poses_arr[:n_frames]
    shape = np.asarray([480, 640, 3], dtype=np.int64)
    return intrinsics, poses_arr, shape

def read_agibot_reference(path: str | Path):
    """Read AgiBot and expose source-robot geometry via the shared schema."""
    from ..common import source_motion_to_reference
    motion = read_agibot_source(path)
    # Camera alignment is performed by the source-robot pipeline; keep identity
    # registration here so the common adapter contract remains shape-complete.
    return source_motion_to_reference(motion, intrinsics=np.zeros(4, dtype=np.float64))
