"""Fourier 6-DoF hand kinematics for ActionNet → Spark retargeting.

ActionNet stores per-hand commands in the teleop ``fourier_dexpilot_dhx`` order
and ~0–10.3 command units (thumb pitch inverted). This module maps those to
URDF radians, runs FK (including mimic joints), and builds anatomical 5×5
chains for the existing Wuji IK path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from ...transforms import (
    axis_angle_matrix,
    canonicalize_hand_points,
    matrix_to_pose_wxyz,
    rpy_matrix,
)

FOURIER_HAND_ASSET_DIR = str(Path(__file__).resolve().parents[2] / "assets" / "fourier_hand_6dof_description")

# Teleop / ActionNet per-hand layout (L then R in the 12-D vector).
HAND_CMD_SUFFIXES = (
    "pinky_proximal_joint",
    "ring_proximal_joint",
    "middle_proximal_joint",
    "index_proximal_joint",
    "thumb_proximal_pitch_joint",
    "thumb_proximal_yaw_joint",
)
# Matches FFTAI/teleoperation configs/hand/fourier_dexpilot_dhx.yaml
HAND_CMD_RANGE_MIN = np.asarray([0.0, 0.0, 0.0, 0.0, 10.3, 0.0], dtype=np.float64)
HAND_CMD_RANGE_MAX = np.asarray([10.3, 10.3, 10.3, 10.3, 0.0, 10.3], dtype=np.float64)
# factory_01/02-style ActionNet logs 0–10.3 teleop units. Later factory_04/05/06
# shards store the same 6-D order as URDF radians (fingers ~[-1.8, 0],
# thumb_pitch ~[0, 1.2]). Treating those as dexpilot units clips every
# negative flex to 0 = fully open.
HAND_SPACE_DEXPILOT = "fourier_dexpilot_dhx_0_10p3"
HAND_SPACE_URDF_RAD = "fourier_urdf_rad"
# 12-D layout: L then R, each pinky/ring/middle/index/thumb_pitch/thumb_yaw.
_FINGER_COLS = (0, 1, 2, 3, 6, 7, 8, 9)
_THUMB_PITCH_COLS = (4, 10)
# URDF fingers upper=0, thumb_pitch upper~1.16. Values above this cannot be rad.
_DEX_FINGER_MAX = 2.0
_DEX_THUMB_PITCH_MAX = 2.0
# Dexpilot fingers stay in [0, 10.3]. Negatives cannot be dexpilot.
_URDF_FINGER_MIN = -0.2
OFFICIAL_DEX_FAMILIES = frozenset({"factory_01", "factory_02", "factory_2"})
OFFICIAL_URDF_FAMILIES = frozenset({"factory_04", "factory_05", "factory_06"})
QUARANTINE_FAMILIES = frozenset({"factory_03", "T5"})
OFFICIAL_REWRITE_FAMILIES = OFFICIAL_DEX_FAMILIES | OFFICIAL_URDF_FAMILIES

SIDE_PREFIX = {"left": "L", "right": "R"}

# Native ActionNet top RGB size (height, width). Writer center-crops to 4:3
# then uniformly scales to Spark 640×480 (same path as EgoDex/OpenAoE).
FOURIER_SOURCE_IMAGE_HW = (800, 1280)

# Approximate OAK-D W color intrinsics (OV9782, 1280x800): EFL 1.69 mm at
# 3 um pixels -> fx = fy ~= 563 px (pinhole-equivalent HFOV ~= 97 deg, the
# "97" in the camera name). Principal point assumed at the image center.
# Values are in the native 1280×800 grid; Spark writers scale them after the
# 4:3 crop (see ``scale_intrinsics_for_output_image``).
FOURIER_NOMINAL_INTRINSICS = np.asarray([563.0, 563.0, 640.0, 400.0], dtype=np.float64)

# Approximate head_yaw_link -> OAK optical-frame mount (optical axes:
# x right, y down, z forward). Solved 2026-08-06 via PnP on 7 manually
# annotated wrist pixels against `state/pose` EE positions (episode
# 01JGZXNR33FF5ZC6G3; reprojection RMS ~43 px, cross-episode spot checks
# within ~5 cm). Camera looks ~74 deg below the horizon.
FOURIER_HEAD_T_CAM_OPTICAL = np.asarray(
    [
        [-0.094960, -0.956710, 0.275114, 0.261770],
        [-0.995325, 0.086355, -0.043251, -0.031368],
        [0.017622, -0.277935, -0.960438, -0.003606],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
# Anatomical finger order for 5×5 chains (thumb=0 … pinky=4).
# Per finger: joint0 (MCP/yaw), joint1, joint2 (pre-tip), tip joint.
ANATOMICAL_CHAIN_SUFFIXES = (
    (
        "thumb_proximal_yaw_joint",
        "thumb_proximal_pitch_joint",
        "thumb_distal_joint",
        "thumb_tip_joint",
    ),
    (
        "index_proximal_joint",
        "index_intermediate_joint",
        "index_tip_joint",
        "index_tip_joint",
    ),
    (
        "middle_proximal_joint",
        "middle_intermediate_joint",
        "middle_tip_joint",
        "middle_tip_joint",
    ),
    (
        "ring_proximal_joint",
        "ring_intermediate_joint",
        "ring_tip_joint",
        "ring_tip_joint",
    ),
    (
        "pinky_proximal_joint",
        "pinky_intermediate_joint",
        "pinky_tip_joint",
        "pinky_tip_joint",
    ),
)


@dataclass(frozen=True)
class _Joint:
    name: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float
    fixed: bool
    mimic_joint: str | None = None
    mimic_multiplier: float = 1.0
    mimic_offset: float = 0.0


def decode_pose9(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode one 9-D ActionNet end-link: xyz + first two columns of R."""
    values = np.asarray(values, dtype=np.float64).reshape(9)
    position = values[:3].copy()
    col0 = values[3:6]
    col1 = values[6:9]
    col2 = np.cross(col0, col1)
    rotation = np.stack([col0, col1, col2], axis=1)
    # Re-orthonormalize in case of mild numerical drift.
    col0 = col0 / np.linalg.norm(col0)
    col1 = col1 - col0 * np.dot(col0, col1)
    col1 = col1 / np.linalg.norm(col1)
    col2 = np.cross(col0, col1)
    rotation = np.stack([col0, col1, col2], axis=1)
    return position, rotation


def pose9_to_wxyz(values: np.ndarray) -> np.ndarray:
    """Convert one 9-D pose block to Spark ``[x,y,z,qw,qx,qy,qz]``."""
    position, rotation = decode_pose9(values)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return matrix_to_pose_wxyz(transform)


def decode_state_pose_ee(pose: np.ndarray) -> np.ndarray:
    """Decode ``state/pose`` ``(T,27)`` → left/right EE ``(T,2,7)`` wxyz."""
    pose = np.asarray(pose, dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] < 18:
        raise ValueError(f"state/pose must be (T, >=18), got {pose.shape}")
    out = np.empty((pose.shape[0], 2, 7), dtype=np.float64)
    for hand, offset in enumerate((0, 9)):
        for index in range(pose.shape[0]):
            out[index, hand] = pose9_to_wxyz(pose[index, offset : offset + 9])
    return out


def decode_state_pose_head(pose: np.ndarray) -> np.ndarray:
    """Decode ``state/pose`` head block ``[...,18:27]`` → ``(T,7)`` wxyz.

    Teleop maps this to GR1 ``head_yaw_link`` (not the OAK-D optical frame).
    """
    pose = np.asarray(pose, dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] < 27:
        raise ValueError(f"state/pose must be (T, >=27) for head, got {pose.shape}")
    out = np.empty((pose.shape[0], 7), dtype=np.float64)
    for index in range(pose.shape[0]):
        out[index] = pose9_to_wxyz(pose[index, 18:27])
    return out


def cmd_to_joint_q(
    cmd6: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    cmd_min: np.ndarray = HAND_CMD_RANGE_MIN,
    cmd_max: np.ndarray = HAND_CMD_RANGE_MAX,
) -> np.ndarray:
    """Map one hand's 6 command units onto URDF limits (teleop order).

    ``amount`` is the *closing* fraction: ``cmd = cmd_min`` is the fully open
    hand (URDF ``upper``: fingers straight at q=0) and ``cmd = cmd_max`` is
    fully closed (URDF ``lower``). Verified against ActionNet videos
    (2026-08-06): open hands log finger cmds ~0, grasps log ~3-6. The thumb
    pitch range arrays are inverted (min=10.3, max=0), so the same rule holds.
    """
    cmd6 = np.asarray(cmd6, dtype=np.float64).reshape(6)
    lower = np.asarray(lower, dtype=np.float64).reshape(6)
    upper = np.asarray(upper, dtype=np.float64).reshape(6)
    cmd_min = np.asarray(cmd_min, dtype=np.float64).reshape(6)
    cmd_max = np.asarray(cmd_max, dtype=np.float64).reshape(6)
    span = cmd_max - cmd_min
    # Inverted joints (thumb pitch) have negative span; still a linear map.
    amount = np.where(np.abs(span) < 1e-12, 0.0, (cmd6 - cmd_min) / span)
    amount = np.clip(amount, 0.0, 1.0)
    return upper + amount * (lower - upper)


class FourierHandSpaceError(ValueError):
    """``state/hand`` encoding is ambiguous, quarantined, or family-mismatched."""


@dataclass(frozen=True)
class HandSpaceEvidence:
    space: str | None
    dex: bool
    urdf: bool
    finger_min: float
    finger_max: float
    thumb_pitch_max: float
    finite: int


def fourier_task_family(task_name: str | None) -> str:
    """Map ``task_name`` onto a coarse family used for official rewrite policy."""
    name = str(task_name or "")
    if name.startswith("T5"):
        return "T5"
    for family in (
        "factory_03",
        "factory_01",
        "factory_02",
        "factory_04",
        "factory_05",
        "factory_06",
    ):
        if name == family or name.startswith(f"{family}_"):
            return family
    if name == "factory_2" or name.startswith("factory_2_"):
        return "factory_2"
    if name.startswith("factory_2") and (len(name) == 9 or not name[9].isdigit()):
        return "factory_2"
    return "other"


def hand_space_evidence(hand_cmd: np.ndarray) -> HandSpaceEvidence:
    """Exclusive range evidence; both or neither means ``space is None``.

    Uses finite min/max, not a percentile guess. One spike that trips both
    tests fails closed instead of flipping the episode.
    """
    values = np.asarray(hand_cmd, dtype=np.float64)
    if values.size == 0 or values.size % 12 != 0:
        return HandSpaceEvidence(None, False, False, np.nan, np.nan, np.nan, 0)
    flat = values.reshape(-1, 12)
    fingers = flat[:, list(_FINGER_COLS)]
    thumb_pitch = flat[:, list(_THUMB_PITCH_COLS)]
    finger_finite = fingers[np.isfinite(fingers)]
    pitch_finite = thumb_pitch[np.isfinite(thumb_pitch)]
    finite = int(finger_finite.size + pitch_finite.size)
    if finger_finite.size == 0:
        return HandSpaceEvidence(None, False, False, np.nan, np.nan, np.nan, finite)
    finger_min = float(np.min(finger_finite))
    finger_max = float(np.max(finger_finite))
    thumb_pitch_max = (
        float(np.max(pitch_finite)) if pitch_finite.size else float("nan")
    )
    dex = finger_max > _DEX_FINGER_MAX or (
        np.isfinite(thumb_pitch_max) and thumb_pitch_max > _DEX_THUMB_PITCH_MAX
    )
    urdf = finger_min < _URDF_FINGER_MIN
    if dex and not urdf:
        space: str | None = HAND_SPACE_DEXPILOT
    elif urdf and not dex:
        space = HAND_SPACE_URDF_RAD
    else:
        space = None
    return HandSpaceEvidence(
        space, dex, urdf, finger_min, finger_max, thumb_pitch_max, finite
    )


def infer_hand_command_space(hand_cmd: np.ndarray) -> str:
    """Return the unique encoding, or raise if both/neither evidence is present."""
    evidence = hand_space_evidence(hand_cmd)
    if evidence.space is None:
        raise FourierHandSpaceError(
            "ambiguous Fourier state/hand encoding "
            f"(dex={evidence.dex}, urdf={evidence.urdf}, "
            f"finger=[{evidence.finger_min:.3f}, {evidence.finger_max:.3f}], "
            f"thumb_pitch_max={evidence.thumb_pitch_max})"
        )
    return evidence.space


def resolve_fourier_hand_space(
    hand_cmd: np.ndarray,
    task_name: str | None,
) -> str:
    """Official rewrite gate: 01/02/04/05/06 only, evidence must match family.

    ``factory_03`` / ``T5*`` / other task names are refused even if the numbers
    look like one of the two known encodings.
    """
    family = fourier_task_family(task_name)
    if family in QUARANTINE_FAMILIES:
        raise FourierHandSpaceError(
            f"quarantined Fourier task family {family!r} (task_name={task_name!r})"
        )
    if family not in OFFICIAL_REWRITE_FAMILIES:
        raise FourierHandSpaceError(
            f"Fourier task family {family!r} is outside official rewrite "
            f"(task_name={task_name!r})"
        )
    space = infer_hand_command_space(hand_cmd)
    expected = (
        HAND_SPACE_DEXPILOT
        if family in OFFICIAL_DEX_FAMILIES
        else HAND_SPACE_URDF_RAD
    )
    if space != expected:
        raise FourierHandSpaceError(
            f"hand-space/family mismatch: task {family!r} expects {expected}, "
            f"evidence says {space}"
        )
    return space


def hand_values_to_q(
    values6: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    space: str,
) -> np.ndarray:
    """Map one hand's 6-D ``state/hand`` slice onto URDF actuated q."""
    if space == HAND_SPACE_URDF_RAD:
        return np.clip(
            np.asarray(values6, dtype=np.float64).reshape(6),
            np.asarray(lower, dtype=np.float64).reshape(6),
            np.asarray(upper, dtype=np.float64).reshape(6),
        )
    if space == HAND_SPACE_DEXPILOT:
        return cmd_to_joint_q(values6, lower, upper)
    raise FourierHandSpaceError(f"unsupported Fourier hand space: {space!r}")


class FourierHandModel:
    """Kinematic FK for one Fourier 6-DoF hand (left or right)."""

    def __init__(self, side: str, urdf_path: str | Path | None = None) -> None:
        if side not in SIDE_PREFIX:
            raise ValueError("side must be 'left' or 'right'")
        self.side = side
        self.prefix = SIDE_PREFIX[side]
        path = (
            Path(urdf_path)
            if urdf_path is not None
            else Path(FOURIER_HAND_ASSET_DIR) / f"{side}.urdf"
        )
        if not path.is_file():
            raise FileNotFoundError(f"Fourier hand URDF not found: {path}")
        self.urdf_path = path
        self._joints = self._parse_urdf(path)
        self.base_link = f"{self.prefix}_hand_base_link"
        self.actuated_names = tuple(
            f"{self.prefix}_{suffix}" for suffix in HAND_CMD_SUFFIXES
        )
        missing = [name for name in self.actuated_names if name not in self._joints]
        if missing:
            raise ValueError(f"URDF {path} missing actuated joints: {missing}")
        self.lower = np.asarray(
            [self._joints[name].lower for name in self.actuated_names],
            dtype=np.float64,
        )
        self.upper = np.asarray(
            [self._joints[name].upper for name in self.actuated_names],
            dtype=np.float64,
        )
        self._children: dict[str, list[str]] = {}
        for joint in self._joints.values():
            self._children.setdefault(joint.parent, []).append(joint.name)

    @staticmethod
    def _parse_urdf(path: Path) -> dict[str, _Joint]:
        root = ET.parse(path).getroot()
        joints: dict[str, _Joint] = {}
        for element in root.findall("joint"):
            name = element.attrib["name"]
            joint_type = element.attrib.get("type", "fixed")
            fixed = joint_type == "fixed"
            origin_element = element.find("origin")
            xyz = np.fromstring(
                (
                    origin_element.attrib.get("xyz", "0 0 0")
                    if origin_element is not None
                    else "0 0 0"
                ),
                sep=" ",
            )
            rpy = np.fromstring(
                (
                    origin_element.attrib.get("rpy", "0 0 0")
                    if origin_element is not None
                    else "0 0 0"
                ),
                sep=" ",
            )
            origin = np.eye(4, dtype=np.float64)
            origin[:3, :3] = rpy_matrix(rpy)
            origin[:3, 3] = xyz
            axis_element = element.find("axis")
            axis = np.fromstring(
                axis_element.attrib.get("xyz", "0 0 1")
                if axis_element is not None
                else "0 0 1",
                sep=" ",
            )
            limit = element.find("limit")
            lower = upper = 0.0
            if not fixed and limit is not None:
                lower = float(limit.attrib["lower"])
                upper = float(limit.attrib["upper"])
            mimic = element.find("mimic")
            mimic_joint = None
            mimic_multiplier = 1.0
            mimic_offset = 0.0
            if mimic is not None:
                mimic_joint = mimic.attrib["joint"]
                mimic_multiplier = float(mimic.attrib.get("multiplier", "1"))
                mimic_offset = float(mimic.attrib.get("offset", "0"))
            joints[name] = _Joint(
                name=name,
                parent=element.find("parent").attrib["link"],
                child=element.find("child").attrib["link"],
                origin=origin,
                axis=axis,
                lower=lower,
                upper=upper,
                fixed=fixed,
                mimic_joint=mimic_joint,
                mimic_multiplier=mimic_multiplier,
                mimic_offset=mimic_offset,
            )
        return joints

    def cmd_to_q(
        self,
        cmd6: np.ndarray,
        space: str = HAND_SPACE_DEXPILOT,
    ) -> np.ndarray:
        return hand_values_to_q(cmd6, self.lower, self.upper, space)

    def _angle_for_joint(self, name: str, q_actuated: np.ndarray) -> float:
        joint = self._joints[name]
        if joint.mimic_joint is not None:
            master = joint.mimic_joint
            if master not in self.actuated_names:
                # Mimic of another mimic is not expected; resolve via actuated only.
                raise ValueError(f"Unsupported mimic chain for {name} -> {master}")
            master_q = float(q_actuated[self.actuated_names.index(master)])
            return joint.mimic_multiplier * master_q + joint.mimic_offset
        if joint.fixed:
            return 0.0
        if name in self.actuated_names:
            return float(q_actuated[self.actuated_names.index(name)])
        raise ValueError(f"No angle available for joint {name}")

    def link_poses(self, q_actuated: np.ndarray) -> dict[str, np.ndarray]:
        """Return 4×4 poses of every link in the hand-base frame."""
        q_actuated = np.asarray(q_actuated, dtype=np.float64).reshape(6)
        poses = {self.base_link: np.eye(4, dtype=np.float64)}
        stack = [self.base_link]
        while stack:
            parent = stack.pop()
            for joint_name in self._children.get(parent, ()):
                joint = self._joints[joint_name]
                motion = np.eye(4, dtype=np.float64)
                # Mimic joints stay ``type="revolute"`` in the URDF; only true
                # fixed tips have no angle. Actuated + mimic both rotate.
                if not joint.fixed or joint.mimic_joint is not None:
                    angle = self._angle_for_joint(joint_name, q_actuated)
                    motion[:3, :3] = axis_angle_matrix(joint.axis, angle)
                child_pose = poses[parent] @ joint.origin @ motion
                poses[joint.child] = child_pose
                stack.append(joint.child)
        return poses

    def forward_chains_local(self, q_actuated: np.ndarray) -> np.ndarray:
        """5×5 xyz chains in the hand-base frame (anatomical finger order).

        Layout matches EgoPro-style chains with ``knuckle_index=1``:
        wrist/base → MCP → next joint → pre-tip → tip. Joint *origins* are
        used for the middle points so consecutive samples are never identical
        (child-link pose of joint *k* coincides with the origin of joint *k*).
        """
        poses = self.link_poses(q_actuated)
        chains = np.empty((5, 5, 3), dtype=np.float64)
        base = poses[self.base_link][:3, 3]

        def _joint_origin(joint_name: str) -> np.ndarray:
            joint = self._joints[joint_name]
            return (poses[joint.parent] @ joint.origin)[:3, 3]

        for finger, suffixes in enumerate(ANATOMICAL_CHAIN_SUFFIXES):
            tip = self._joints[f"{self.prefix}_{suffixes[-1]}"]
            tip_pos = poses[tip.child][:3, 3]
            chains[finger, 0] = base
            chains[finger, 1] = _joint_origin(f"{self.prefix}_{suffixes[0]}")
            chains[finger, 2] = _joint_origin(f"{self.prefix}_{suffixes[1]}")
            if finger == 0:
                # Thumb has an extra distal revolute before the tip.
                chains[finger, 3] = _joint_origin(f"{self.prefix}_{suffixes[2]}")
            else:
                # Fingers only have proximal+intermediate+fixed tip: insert a
                # synthetic DIP between PIP and tip so 5×5 segments stay nonzero.
                chains[finger, 3] = 0.5 * (chains[finger, 2] + tip_pos)
            chains[finger, 4] = tip_pos
        return chains


def chains_world_from_ee(
    ee_pose_wxyz: np.ndarray,
    chains_local: np.ndarray,
) -> np.ndarray:
    """Apply EE (hand-base) pose to local 5×5 chains → world/robot frame."""
    from ...transforms import pose_wxyz_to_matrix, transform_points

    transform = pose_wxyz_to_matrix(ee_pose_wxyz)
    return transform_points(transform, chains_local)


def build_hand_geometry_frame(
    ee_pose_wxyz: np.ndarray,
    cmd6: np.ndarray,
    model: FourierHandModel,
    space: str = HAND_SPACE_DEXPILOT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Return joints_world, joints_local, confidence, valid for one hand/frame."""
    q = model.cmd_to_q(cmd6, space=space)
    local_base = model.forward_chains_local(q)
    joints_world = chains_world_from_ee(ee_pose_wxyz, local_base)
    confidence = np.ones((5, 5), dtype=np.float64)
    joints_local, ok = canonicalize_hand_points(
        joints_world,
        ee_pose_wxyz[:3],
        confidence,
        knuckle_index=1,
    )
    return joints_world, joints_local, confidence, ok
