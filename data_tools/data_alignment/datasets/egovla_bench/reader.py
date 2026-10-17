"""EgoVLA-Humanoid-Sim (IsaacLab H1 + Inspire) → source-robot geometry.

Raw episodes are HuggingFace ``EgoVLA/EgoVLA-Humanoid-Sim`` HDF5 files:
``observations/images/main`` (T, 384, 384, 3) RGB, ``observations/qpos`` (T, 50),
world-frame EE 7-D, and optional wrist RGB. Hands are Inspire 12-D (6 actuated
+ 6 mimic) packed into the 50-D qpos. The source reader extracts canonical
Inspire12 values; the EgoVLA adapter applies the reviewed direct index/sign
mapping to Wuji2 without Inspire/Wuji FK or IK.

Raw files do not store camera K/extrinsics. Converts leave those slots empty
(NaN + invalid). Do not invent look-at / pinhole / official-sim calibration
unless the user explicitly asks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .hand import (
    ACTUATED,
    INSPIRE12_ORDER,
    InspireHandModel,
)
from ...schema import FRAME_ALIGNED_METADATA_KEY, UnifiedHumanMotion
from ...transforms import (
    canonicalize_hand_points,
    ensure_quaternion_continuity,
    pose_wxyz_to_matrix,
    transform_points,
)

# Fixed by the raw dataset. Converts must not resample away from it: qpos and
# the arm joints derived from it ride in ``metadata`` and are never resampled.
EGOVLA_SOURCE_FPS = 30.0

# 50-D qpos order from egovla-dataset-tool provenance (H1 26 + Inspire 24).
QPOS_DIM = 50
BODY_JOINT_COUNT = 26

# Actuated Inspire joints in ``InspireHandModel.actuated_names`` order.
_LEFT_ACTUATED_QPOS = (
    30,  # thumb_proximal_yaw
    26,  # index_proximal
    27,  # middle_proximal
    29,  # ring_proximal
    28,  # pinky_proximal
    40,  # thumb_proximal_pitch
)
_RIGHT_ACTUATED_QPOS = (
    35,
    31,
    32,
    34,
    33,
    45,
)

# 7-D H1 arms (shoulder pitch/roll/yaw, elbow pitch/roll, wrist pitch/yaw).
_LEFT_ARM_QPOS = (4, 8, 12, 16, 20, 22, 24)
_RIGHT_ARM_QPOS = (5, 9, 13, 17, 21, 23, 25)

TASK_INSTRUCTIONS = {
    "Close-Drawer": "Close the opened drawer",
    "Open-Drawer": "Open the drawer",
    "Flip-Mug": "Flip the mug",
    "Open-Laptop": "Open the laptop",
    "Pour-Balls": "Pour the balls",
    "Push-Box": "Push the box",
    "Stack-Can": "Stack the can",
    "Stack-Can-Into-Drawer": "Stack the can into the drawer",
    "Unload-Cans": "Unload the cans",
    "Insert-Cans": "Insert the cans",
    "Sort-Cans": "Sort the cans",
    "Insert-And-Unload-Cans": "Insert and unload the cans",
}

EGOVLA_IMAGE_HW = (384, 384)

_EE_KEY_CANDIDATES = ("{side}_ee_pose", "{side}_curr_ee_pose")


def is_egovla_episode(path: str | Path) -> bool:
    """True for a raw EgoVLA-Humanoid-Sim HDF5 (not a Spark product)."""
    source = Path(path)
    if not source.is_file() or source.suffix.lower() not in {".hdf5", ".h5"}:
        return False
    try:
        import h5py
    except ImportError:
        return False
    try:
        with h5py.File(source, "r") as file:
            if "state/left_ee_poses" in file:
                return False
            if "observations/images/main" not in file:
                return False
            if "observations/qpos" not in file:
                return False
            qpos = file["observations/qpos"]
            main = file["observations/images/main"]
            if qpos.ndim != 2 or qpos.shape[1] != QPOS_DIM:
                return False
            if main.ndim != 4 or main.shape[1:3] != EGOVLA_IMAGE_HW:
                return False
            return qpos.shape[0] == main.shape[0]
    except OSError:
        return False


def parse_egovla_episode(path: str | Path) -> tuple[Path, str, str]:
    """``…/<Task>/episode_N.hdf5`` → (task_parent, task, episode stem)."""
    source = Path(path).resolve()
    if not is_egovla_episode(source):
        raise ValueError(f"Not an EgoVLA-Humanoid-Sim episode: {source}")
    return source.parent.parent, source.parent.name, source.stem


def instruction_for_task(task: str, override: str | None = None) -> str:
    if override is not None and str(override).strip():
        return str(override).strip()
    if task in TASK_INSTRUCTIONS:
        return TASK_INSTRUCTIONS[task]
    return task.replace("-", " ").replace("_", " ").strip() or "manipulate the object"


def _ee_dataset_name(keys: set[str], side: str) -> str:
    for template in _EE_KEY_CANDIDATES:
        name = template.format(side=side)
        if name in keys:
            return name
    raise ValueError(
        f"EgoVLA episode missing {side} EE "
        f"(tried {[t.format(side=side) for t in _EE_KEY_CANDIDATES]})"
    )


def inspire12_actuated_from_qpos(qpos: np.ndarray, side: str) -> np.ndarray:
    """Slice the 6 actuated Inspire joints from a (T, 50) qpos."""
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != QPOS_DIM:
        raise ValueError(f"expected qpos (T, {QPOS_DIM}), got {qpos.shape}")
    index = _LEFT_ACTUATED_QPOS if side == "left" else _RIGHT_ACTUATED_QPOS
    return qpos[:, list(index)]


def arm_joints_from_qpos(qpos: np.ndarray, side: str) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float64)
    index = _LEFT_ARM_QPOS if side == "left" else _RIGHT_ARM_QPOS
    return qpos[:, list(index)]


def read_egovla_source(
    path: str | Path,
    *,
    instruction: str | None = None,
    need_finger_chains: bool = True,
) -> UnifiedHumanMotion:
    """Load one EgoVLA-Humanoid-Sim episode into source-robot hand geometry.

    ``need_finger_chains=False`` skips Inspire FK (used by ``--to egovla``).
    Wrist validity still follows finite EE poses.
    """
    try:
        import h5py
    except ImportError as error:
        raise ImportError("Reading EgoVLA episodes requires h5py") from error

    source_path = Path(path)
    if not is_egovla_episode(source_path):
        raise ValueError(f"Not an EgoVLA-Humanoid-Sim episode: {source_path}")
    _root, task, episode = parse_egovla_episode(source_path)

    with h5py.File(source_path, "r") as file:
        obs = file["observations"]
        keys = set(obs.keys())
        qpos = np.asarray(obs["qpos"][()], dtype=np.float64)
        frames = int(qpos.shape[0])
        wrist = np.empty((frames, 2, 7), dtype=np.float64)
        for hand_index, side in enumerate(("left", "right")):
            ee = np.asarray(obs[_ee_dataset_name(keys, side)][()], dtype=np.float64)
            if ee.shape != (frames, 7):
                raise ValueError(f"{side} EE shape {ee.shape}, expected ({frames}, 7)")
            wrist[:, hand_index] = ee
        tip_keys = {
            "left": "left_finger_tip_pos" if "left_finger_tip_pos" in keys else None,
            "right": "right_finger_tip_pos" if "right_finger_tip_pos" in keys else None,
        }
        stored_tips = {}
        for side, key in tip_keys.items():
            if key is None:
                stored_tips[side] = None
                continue
            tips = np.asarray(obs[key][()], dtype=np.float64)
            stored_tips[side] = tips.reshape(frames, 5, 3)
        image_keys = ["main"]
        if "images" in obs:
            for extra in ("left_hand", "right_hand"):
                if extra in obs["images"]:
                    image_keys.append(extra)

    for hand in range(2):
        finite = np.all(np.isfinite(wrist[:, hand]), axis=1)
        if np.any(finite):
            wrist[finite, hand] = ensure_quaternion_continuity(wrist[finite, hand])

    joints_world = np.full((frames, 2, 5, 5, 3), np.nan, dtype=np.float64)
    joints_local = np.full((frames, 2, 5, 5, 3), np.nan, dtype=np.float64)
    confidence = np.zeros((frames, 2, 5, 5), dtype=np.float64)
    wrist_valid = np.all(np.isfinite(wrist), axis=2)
    q_actuated = {
        "left": inspire12_actuated_from_qpos(qpos, "left"),
        "right": inspire12_actuated_from_qpos(qpos, "right"),
    }

    if need_finger_chains:
        models = {side: InspireHandModel(side) for side in ("left", "right")}
        for hand_index, side in enumerate(("left", "right")):
            model = models[side]
            q_hand = q_actuated[side]
            valid = wrist_valid[:, hand_index] & np.all(np.isfinite(q_hand), axis=1)
            for frame_index in np.flatnonzero(valid):
                local = model.forward_chains_local(q_hand[frame_index])
                world_from_ee = pose_wxyz_to_matrix(wrist[frame_index, hand_index])
                world = transform_points(world_from_ee, local.reshape(-1, 3)).reshape(
                    5, 5, 3
                )
                conf = np.ones((5, 5), dtype=np.float64)
                canon, ok = canonicalize_hand_points(
                    world, wrist[frame_index, hand_index, :3], conf, knuckle_index=1
                )
                joints_world[frame_index, hand_index] = world
                confidence[frame_index, hand_index] = conf
                if ok:
                    joints_local[frame_index, hand_index] = canon
                else:
                    confidence[frame_index, hand_index] = 0.0

    timestamps = np.arange(frames, dtype=np.float64) / EGOVLA_SOURCE_FPS
    return UnifiedHumanMotion(
        timestamps=timestamps,
        wrist_poses=wrist,
        wrist_valid=wrist_valid,
        joints_local=joints_local,
        joint_confidence=confidence,
        source="egovla_humanoid_sim",
        instruction=instruction_for_task(task, instruction),
        metadata={
            "source_path": source_path.name,
            "source_fps": EGOVLA_SOURCE_FPS,
            "video_fps": EGOVLA_SOURCE_FPS,
            "video_frame_count": frames,
            "coordinate_frame": "source_robot_as_is",
            "ee_frame": "egovla_inspire_ee",
            "finger_parent": "ee_pose",
            "joints_world": joints_world,
            "task": task,
            "episode": episode,
            "image_keys": image_keys,
            "qpos": qpos,
            "left_arm_q": arm_joints_from_qpos(qpos, "left"),
            "right_arm_q": arm_joints_from_qpos(qpos, "right"),
            # H1 arm joints have no slot in UnifiedHumanMotion, so they ride in
            # metadata; declaring them keeps resampling and trimming in sync.
            FRAME_ALIGNED_METADATA_KEY: {
                "qpos": "linear",
                "left_arm_q": "linear",
                "right_arm_q": "linear",
            },
            "source_image_hw": EGOVLA_IMAGE_HW,
            "stored_fingertips": stored_tips,
            "actuated_names": ACTUATED,
            "inspire12_order": INSPIRE12_ORDER,
            "need_finger_chains": bool(need_finger_chains),
            "source_readonly": True,
        },
    )


def load_egovla_rgb(
    path: str | Path,
    camera: str,
    indices: np.ndarray | int,
) -> np.ndarray:
    """Return ``(len(indices), 384, 384, 3)`` RGB uint8 for ``main`` / wrist cams.

    ``indices`` selects source frames (one per output frame); an ``int`` keeps
    the legacy prefix semantics ``0..n-1``. Indices past the stored frames
    reuse the last frame (same padding as before).
    """
    try:
        import h5py
    except ImportError as error:
        raise ImportError("Reading EgoVLA RGB requires h5py") from error
    if isinstance(indices, (int, np.integer)):
        indices = np.arange(int(indices), dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    key = {
        "main": "observations/images/main",
        "left_hand": "observations/images/left_hand",
        "right_hand": "observations/images/right_hand",
    }.get(camera)
    if key is None:
        raise ValueError(f"unknown EgoVLA camera {camera!r}")
    with h5py.File(path, "r") as file:
        if key not in file:
            raise KeyError(key)
        stored = file[key]
        if not len(stored):
            raise ValueError(f"{key} is empty")
        upper = min(int(indices.max()) + 1, len(stored)) if len(indices) else 0
        rgb = np.asarray(stored[:upper])
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if rgb.ndim != 4 or rgb.shape[1:3] != EGOVLA_IMAGE_HW or rgb.shape[-1] != 3:
        raise ValueError(f"{key} shape {rgb.shape}, expected (T, 384, 384, 3)")
    return rgb[np.clip(indices, 0, len(rgb) - 1)]
