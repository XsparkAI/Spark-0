"""Fit and store MANO root + articulation on the EgoPro HDF5 timeline.

``ee_poses`` is ``xyz + wxyz`` (7). ``ee_joint_states`` is the native MANO
hand pose, 15 joints × axis-angle (45). ``hand_betas`` is MANO shape
``(T, 10)``. EgoPro optimizes one subject shape per valid track.  The standard
EgoDex-compatible tuple reconstructs with scale fixed at 1; no hidden size
parameter is discarded.
Action is the next-state copy of those arrays, matching robot ``state`` /
``action`` truncation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ...tactile.collector import (
    UNITY_TO_OPENVR,
    manus_joints21,
)
from ...camera_alignment import (
    ensure_quaternion_continuity,
    pose_wxyz_to_matrix,
    rotvec_to_wxyz,
)

MANO_POSE_DIM = 7
MANO_JOINT_DIM = 45
MANO_BETA_DIM = 10
MANO_HANDS = ("left", "right")
MANO_CAMERA_FRAME = "camera_opencv_upright_head"
def slice_next_state(
    values: np.ndarray, action_steps: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Split a full-horizon series into truncated state and next-state action."""
    if action_steps < 1:
        raise ValueError("action_steps must be positive")
    length = len(values)
    if length <= action_steps:
        raise ValueError(
            f"Need more than {action_steps} frames to write next-state MANO, "
            f"got {length}"
        )
    keep = length - action_steps
    return values[:keep], values[action_steps : action_steps + keep]


def pack_ee_pose(translation: np.ndarray, global_orient: np.ndarray) -> np.ndarray:
    """``xyz`` + ``wxyz`` from IK translation and axis-angle ``global_orient``."""
    pose = np.full(MANO_POSE_DIM, np.nan, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    global_orient = np.asarray(global_orient, dtype=np.float64).reshape(3)
    if not (np.isfinite(translation).all() and np.isfinite(global_orient).all()):
        return pose
    pose[:3] = translation
    pose[3:7] = rotvec_to_wxyz(global_orient)
    return pose


def manus_joints_to_camera(
    joints21: Mapping[str, np.ndarray],
    *,
    source_coordinate_frame: str,
    world_from_egopro: np.ndarray,
    camera_from_world: np.ndarray,
) -> dict[str, np.ndarray]:
    """Map stored MANUS targets into the per-frame head optical frame.

    ``manus_joints21`` stores Unity-world sources after the Unity→OpenVR z
    flip.  Undo that self-inverse flip, apply the calibrated EgoPro→Spark
    world map, then the dynamic optical-from-world camera pose.  Fitting in
    this frame matches the camera-frame MANO convention used by EgoDex and
    makes the exported root pose directly projectable with head intrinsics.
    """
    world_map = np.asarray(world_from_egopro, dtype=np.float64)
    if world_map.shape != (4, 4):
        raise ValueError(f"world_from_egopro must be 4x4, got {world_map.shape}")
    camera_poses = np.asarray(camera_from_world, dtype=np.float64)
    if camera_poses.ndim != 2 or camera_poses.shape[1] != 7:
        raise ValueError(
            f"camera_from_world must have shape [T,7], got {camera_poses.shape}"
        )
    camera_matrices = pose_wxyz_to_matrix(camera_poses)
    result: dict[str, np.ndarray] = {}
    for hand in MANO_HANDS:
        points = np.asarray(joints21[hand], dtype=np.float64)
        if points.shape != (len(camera_poses), 21, 3):
            raise ValueError(
                f"{hand} MANUS joints must have shape "
                f"({len(camera_poses)},21,3), got {points.shape}"
            )
        if source_coordinate_frame == "manus_core_unity_world":
            points = points @ UNITY_TO_OPENVR.T
        world = np.einsum(
            "ij,tkj->tki", world_map[:3, :3], points, optimize=True
        ) + world_map[:3, 3]
        camera = np.einsum(
            "tij,tkj->tki", camera_matrices[:, :3, :3], world, optimize=True
        ) + camera_matrices[:, None, :3, 3]
        result[hand] = camera
    return result


def fit_episode_mano_params(
    joints21: Mapping[str, np.ndarray],
    *,
    mano_assets: Any = None,
    device: str = "cpu",
    bootstrap_iterations: int = 80,
    live_iterations: int = 32,
) -> dict[str, dict[str, np.ndarray]]:
    """IK each hand's ``(T, 21, 3)`` joints into standard MANO parameters.

    Shape and size are encoded by ``hand_betas`` and scale is fixed at 1, so
    the exported pose / articulation / beta tuple is self-contained.
    """
    from .ik import cached_runtime, fit_hand_mesh_fast

    runtime = cached_runtime(mano_assets=mano_assets, device=device)
    result: dict[str, dict[str, np.ndarray]] = {}
    for hand in MANO_HANDS:
        targets = np.asarray(joints21[hand], dtype=np.float64)
        if targets.ndim != 3 or targets.shape[1:] != (21, 3):
            raise ValueError(
                f"manus/{hand}_ee must have shape [T,21,3], got {targets.shape}"
            )
        length = len(targets)
        poses = np.full((length, MANO_POSE_DIM), np.nan, dtype=np.float32)
        joints = np.full((length, MANO_JOINT_DIM), np.nan, dtype=np.float32)
        betas = np.full((length, MANO_BETA_DIM), np.nan, dtype=np.float32)
        scale = np.full(length, np.nan, dtype=np.float32)
        previous = None
        backend = runtime.backends[hand]
        for index, target in enumerate(targets):
            if index == 0 or (index + 1) % 40 == 0 or index + 1 == length:
                print(f"MANO IK {hand} {index + 1}/{length}", flush=True)
            if target.shape != (21, 3) or not np.all(np.isfinite(target)):
                previous = None
                continue
            _vertices, state = fit_hand_mesh_fast(
                backend,
                target,
                previous,
                bootstrap_iterations=bootstrap_iterations,
                live_iterations=live_iterations,
                need_vertices=False,
            )
            previous = state
            poses[index] = pack_ee_pose(state["translation"], state["global_orient"])
            joints[index] = np.asarray(state["handPose"], dtype=np.float32).reshape(45)
            betas[index] = np.asarray(state["betas"], dtype=np.float32).reshape(10)
            scale[index] = float(state["scale"])
        finite = np.isfinite(poses).all(axis=1)
        if np.any(finite):
            poses[finite] = ensure_quaternion_continuity(poses[finite]).astype(
                np.float32
            )
        result[hand] = {
            "poses": poses,
            "joints": joints,
            "betas": betas,
            "scale": scale,
        }
    return result


def write_fitted_mano_tree(
    file: Any,
    fitted: Mapping[str, Mapping[str, np.ndarray]],
    *,
    root_name: str = "mano",
    action_steps: int = 1,
    coordinate_frame: str = "",
    source_coordinate_frame: str = "",
    extra_attrs: Mapping[str, Any] | None = None,
) -> None:
    """Write a MANO ``state|action`` tree below ``root_name``."""
    if not root_name or root_name.startswith("/"):
        raise ValueError(f"root_name must be a nonempty relative HDF5 path: {root_name!r}")
    for hand in MANO_HANDS:
        scale = np.asarray(fitted[hand].get("scale", 1.0), dtype=np.float64)
        finite = scale[np.isfinite(scale)]
        if finite.size and not np.allclose(finite, 1.0, atol=1e-6, rtol=0.0):
            raise ValueError(
                f"{hand} fitted MANO contains non-unit scale; the exported "
                "EgoDex schema has no scale field, so size must be encoded in betas"
            )
    mano = file.require_group(root_name)
    mano.attrs["layout"] = (
        "ee_poses=xyz_wxyz; ee_joint_states=mano_pose_15x3_axis_angle; "
        "hand_betas=mano_shape_10"
    )
    mano.attrs["pose_order"] = "x_y_z_qw_qx_qy_qz"
    mano.attrs["coordinate_frame"] = coordinate_frame
    mano.attrs["source_coordinate_frame"] = source_coordinate_frame
    mano.attrs["action_source"] = f"retargeted_future_state:{action_steps}"
    mano.attrs["global_orient"] = "wxyz_from_mano_axis_angle"
    mano.attrs["scale_convention"] = "fixed_1_shape_encoded_by_betas"
    if extra_attrs:
        for key, value in extra_attrs.items():
            mano.attrs[key] = value

    groups = [mano.require_group(name) for name in ("state", "action")]
    for hand in MANO_HANDS:
        for source_key, dataset_key in (
            ("poses", "ee_poses"),
            ("joints", "ee_joint_states"),
            ("betas", "hand_betas"),
        ):
            values = slice_next_state(fitted[hand][source_key], action_steps)
            for group, value in zip(groups, values):
                group.create_dataset(
                    f"{hand}_{dataset_key}",
                    data=np.asarray(value, dtype=np.float32),
                    compression="gzip",
                    compression_opts=4,
                    shuffle=True,
                )


def write_egopro_mano_hdf5(
    file: Any,
    context: Any,
    source_frames: np.ndarray,
    *,
    action_steps: int = 1,
    mano_assets: Any = None,
    world_from_egopro: np.ndarray | None = None,
    camera_from_world: np.ndarray | None = None,
) -> None:
    """Write camera-frame ``mano/state|action`` from full-horizon MANUS joints."""
    import time

    if mano_assets is None or str(mano_assets).strip() == "":
        raise ValueError("EgoPro MANO write requires --mano-assets")
    if world_from_egopro is None or camera_from_world is None:
        raise ValueError(
            "EgoPro MANO write requires calibrated world_from_egopro and "
            "full-horizon camera_from_world"
        )
    source_indices = np.asarray(source_frames, dtype=np.int64)
    raw_joints21 = {
        hand: manus_joints21(context, hand, source_indices) for hand in MANO_HANDS
    }
    manus = file["raw/manus"]
    source_coordinate_frame = str(
        manus.attrs.get("source_coordinate_frame", "")
    )
    joints21 = manus_joints_to_camera(
        raw_joints21,
        source_coordinate_frame=source_coordinate_frame,
        world_from_egopro=np.asarray(world_from_egopro, dtype=np.float64),
        camera_from_world=np.asarray(camera_from_world, dtype=np.float64),
    )
    from .ik import limit_cpu_threads

    limit_cpu_threads()
    started = time.perf_counter()
    fitted = fit_episode_mano_params(
        joints21, mano_assets=mano_assets, device="cpu"
    )
    elapsed = time.perf_counter() - started
    length = int(source_indices.shape[0])
    print(
        f"MANO IK cpu {length} frames x 2 hands in {elapsed:.1f}s",
        flush=True,
    )
    write_fitted_mano_tree(
        file,
        fitted,
        action_steps=action_steps,
        coordinate_frame=MANO_CAMERA_FRAME,
        source_coordinate_frame=source_coordinate_frame,
        extra_attrs={
            "source": "egopro_manus21_ik",
            "ik": "adam_21j_root_shape_scale1",
            "ik_bootstrap_iterations": 80,
            "ik_live_iterations": 32,
            "camera_alignment": "unity_to_openvr_undo_then_egopro_to_spark_to_head_optical",
        },
    )


def pack_openaoe_mano_params(hands: Mapping[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    """Pack OpenAoE ``hands.npz`` camera-frame MANO into the HDF5 tree."""
    pred_valid = np.asarray(hands["pred_valid"], dtype=np.float64) > 0.5
    pred_trans = np.asarray(hands["pred_trans_cam"], dtype=np.float64)
    pred_rot = np.asarray(hands["pred_rot_cam"], dtype=np.float64)
    pred_pose = np.asarray(hands["pred_hand_pose"], dtype=np.float64)
    pred_betas = np.asarray(hands["pred_betas"], dtype=np.float64)
    if pred_trans.ndim != 3 or pred_trans.shape[0] != 2 or pred_trans.shape[2] != 3:
        raise ValueError(f"Unexpected pred_trans_cam shape {pred_trans.shape}")
    length = int(pred_trans.shape[1])
    if pred_betas.shape != (2, length, MANO_BETA_DIM):
        raise ValueError(f"Unexpected pred_betas shape {pred_betas.shape}")
    result: dict[str, dict[str, np.ndarray]] = {}
    for hand_index, hand in enumerate(MANO_HANDS):
        poses = np.full((length, MANO_POSE_DIM), np.nan, dtype=np.float32)
        joints = np.full((length, MANO_JOINT_DIM), np.nan, dtype=np.float32)
        betas = np.full((length, MANO_BETA_DIM), np.nan, dtype=np.float32)
        scale = np.full(length, np.nan, dtype=np.float32)
        for index in range(length):
            if not bool(pred_valid[hand_index, index]):
                continue
            poses[index] = pack_ee_pose(
                pred_trans[hand_index, index], pred_rot[hand_index, index]
            )
            joints[index] = np.asarray(
                pred_pose[hand_index, index], dtype=np.float32
            ).reshape(45)
            betas[index] = np.asarray(
                pred_betas[hand_index, index], dtype=np.float32
            ).reshape(10)
            scale[index] = 1.0
        finite = np.isfinite(poses).all(axis=1)
        if np.any(finite):
            poses[finite] = ensure_quaternion_continuity(poses[finite]).astype(
                np.float32
            )
        result[hand] = {
            "poses": poses,
            "joints": joints,
            "betas": betas,
            "scale": scale,
        }
    return result


def _gather_fitted(
    fitted: Mapping[str, Mapping[str, np.ndarray]],
    indices: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    indices = np.asarray(indices, dtype=np.int64)
    gathered: dict[str, dict[str, np.ndarray]] = {}
    for hand, payload in fitted.items():
        length = len(payload["poses"])
        clipped = np.clip(indices, 0, max(0, length - 1))
        gathered[hand] = {
            key: np.asarray(values)[clipped] for key, values in payload.items()
        }
    return gathered


def write_openaoe_mano_hdf5(
    file: Any,
    sample_dir: str | Path,
    source_frames: np.ndarray,
    *,
    action_steps: int = 1,
) -> None:
    """Write ``mano/state|action`` from OpenAoE ``hands.npz`` (FK params, no IK)."""
    hands_path = (
        Path(sample_dir) / "ego_process" / "ego_hands_reconstruction" / "hands.npz"
    )
    if not hands_path.is_file():
        raise FileNotFoundError(f"OpenAoE hands.npz not found: {hands_path}")
    fitted = _gather_fitted(
        pack_openaoe_mano_params(np.load(hands_path)),
        np.asarray(source_frames, dtype=np.int64),
    )
    write_fitted_mano_tree(
        file,
        fitted,
        action_steps=action_steps,
        coordinate_frame="openaoe_camera",
        source_coordinate_frame="openaoe_pred_cam",
        extra_attrs={"source": "openaoe_hands_npz"},
    )
