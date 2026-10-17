"""Shared source-dataset helpers and InspireHand URDF kinematics."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from .._paths import ASSET_ROOT
from ..schema import ReferenceFrameGeometry
from ..transforms import axis_angle_matrix, rpy_matrix

DEFAULT_INSPIRE_ASSET_DIR = ASSET_ROOT / "inspire_hand"

INSPIRE_ACTUATED = (
    "thumb_proximal_yaw_joint",
    "index_proximal_joint",
    "middle_proximal_joint",
    "ring_proximal_joint",
    "pinky_proximal_joint",
    "thumb_proximal_pitch_joint",
)

INSPIRE_ANATOMICAL_CHAIN = (
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

# Thumb: sample anatomical *links*, skipping ``thumb_proximal_base``.
# Joint-origin sampling (yaw→pitch) keeps the CMC dogleg (~1.7 cm, ~100° bend
# even at q=0); Wuji IK then locks mcp≈π/2. Link chain:
# hand_base → proximal → intermediate → distal → tip.
INSPIRE_THUMB_LINK_CHAIN = (
    "thumb_proximal",
    "thumb_intermediate",
    "thumb_distal",
    "thumb_tip",
)


@dataclass(frozen=True)
class _InspireJoint:
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


class InspireHandKinematics:
    """Shared URDF forward kinematics for one Inspire hand (dex-urdf assets)."""

    def __init__(self, side: str, urdf_path: str | Path | None = None) -> None:
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        self.side = side
        path = (
            Path(urdf_path)
            if urdf_path is not None
            else DEFAULT_INSPIRE_ASSET_DIR / f"inspire_hand_{side}.urdf"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        self.urdf_path = path
        self._joints = self._parse_urdf(path)
        self.base_link = "hand_base_link"
        self.actuated_names = INSPIRE_ACTUATED
        missing = [n for n in self.actuated_names if n not in self._joints]
        if missing:
            raise ValueError(f"URDF missing actuated joints: {missing}")
        self.lower = np.asarray(
            [self._joints[n].lower for n in self.actuated_names], dtype=np.float64
        )
        self.upper = np.asarray(
            [self._joints[n].upper for n in self.actuated_names], dtype=np.float64
        )
        self._children: dict[str, list[str]] = {}
        for joint in self._joints.values():
            self._children.setdefault(joint.parent, []).append(joint.name)
        bj = self._joints["base_joint"]
        self.T_base_from_handbase = np.linalg.inv(bj.origin)
        self.T_handbase_from_base = bj.origin.copy()

    @staticmethod
    def _parse_urdf(path: Path) -> dict[str, _InspireJoint]:
        root = ET.parse(path).getroot()
        joints: dict[str, _InspireJoint] = {}
        for element in root.findall("joint"):
            name = element.attrib["name"]
            joint_type = element.attrib.get("type", "fixed")
            fixed = joint_type == "fixed"
            origin_element = element.find("origin")
            xyz = np.fromstring(
                origin_element.attrib.get("xyz", "0 0 0") if origin_element is not None else "0 0 0",
                sep=" ",
            )
            rpy = np.fromstring(
                origin_element.attrib.get("rpy", "0 0 0") if origin_element is not None else "0 0 0",
                sep=" ",
            )
            origin = np.eye(4, dtype=np.float64)
            origin[:3, :3] = rpy_matrix(rpy)
            origin[:3, 3] = xyz
            axis_element = element.find("axis")
            axis = np.fromstring(
                axis_element.attrib.get("xyz", "0 0 1") if axis_element is not None else "0 0 1",
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
            joints[name] = _InspireJoint(
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

    def _angle_for_joint(self, name: str, q_actuated: np.ndarray) -> float:
        joint = self._joints[name]
        if joint.mimic_joint is not None:
            master = joint.mimic_joint
            if master not in self.actuated_names:
                raise ValueError(f"Unsupported mimic chain for {name} -> {master}")
            master_q = float(q_actuated[self.actuated_names.index(master)])
            return joint.mimic_multiplier * master_q + joint.mimic_offset
        if joint.fixed:
            return 0.0
        if name in self.actuated_names:
            return float(q_actuated[self.actuated_names.index(name)])
        raise ValueError(f"No angle available for joint {name}")

    def link_poses(self, q_actuated: np.ndarray) -> dict[str, np.ndarray]:
        """4×4 poses of every link in the ``hand_base_link`` frame."""
        q_actuated = np.asarray(q_actuated, dtype=np.float64).reshape(6)
        poses = {self.base_link: np.eye(4, dtype=np.float64)}
        stack = [self.base_link]
        while stack:
            parent = stack.pop()
            for joint_name in self._children.get(parent, ()):
                joint = self._joints[joint_name]
                motion = np.eye(4, dtype=np.float64)
                if not joint.fixed or joint.mimic_joint is not None:
                    angle = self._angle_for_joint(joint_name, q_actuated)
                    motion[:3, :3] = axis_angle_matrix(joint.axis, angle)
                child_pose = poses[parent] @ joint.origin @ motion
                poses[joint.child] = child_pose
                stack.append(joint.child)
        return poses

    def forward_chains_local(self, q_actuated: np.ndarray) -> np.ndarray:
        """5×5 xyz chains in ``hand_base_link`` (thumb..pinky anatomical).

        Thumb uses link origins (skipping the CMC ``thumb_proximal_base``
        dogleg); other fingers use actuated/mimic joint origins as before.
        """
        poses = self.link_poses(q_actuated)
        chains = np.empty((5, 5, 3), dtype=np.float64)
        base = poses[self.base_link][:3, 3]

        def _joint_origin(joint_name: str) -> np.ndarray:
            joint = self._joints[joint_name]
            return (poses[joint.parent] @ joint.origin)[:3, 3]

        for finger, names in enumerate(INSPIRE_ANATOMICAL_CHAIN):
            if finger == 0:
                chains[finger, 0] = base
                for i, link_name in enumerate(INSPIRE_THUMB_LINK_CHAIN, start=1):
                    chains[finger, i] = poses[link_name][:3, 3]
                continue
            tip = self._joints[names[-1]]
            tip_pos = poses[tip.child][:3, 3]
            chains[finger, 0] = base
            chains[finger, 1] = _joint_origin(names[0])
            chains[finger, 2] = _joint_origin(names[1])
            chains[finger, 3] = 0.5 * (chains[finger, 2] + tip_pos)
            chains[finger, 4] = tip_pos
        return chains


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="replace")
    return value

def source_motion_to_reference(motion: Any, *, intrinsics: np.ndarray | None = None,
                               camera_from_world: np.ndarray | None = None,
                               camera_valid: np.ndarray | None = None) -> ReferenceFrameGeometry:
    """Expose source-robot motion through the shared reference geometry contract."""
    n = int(len(motion.timestamps))
    joints_world = np.asarray(motion.metadata.get("joints_world", motion.joints_local), dtype=np.float64)
    if joints_world.shape != (n, 2, 5, 5, 3):
        joints_world = np.asarray(motion.joints_local, dtype=np.float64)
    if camera_from_world is None:
        camera_from_world = np.repeat(np.eye(4, dtype=np.float64)[None], n, axis=0)
    if camera_valid is None:
        camera_valid = np.ones(n, dtype=bool)
    if intrinsics is None:
        intrinsics = np.zeros(4, dtype=np.float64)
    metadata = dict(motion.metadata)
    metadata.update({"source_robot_motion": True, "source_motion": motion,
                     "joints_world": joints_world, "camera_registration": "identity_placeholder"})
    return ReferenceFrameGeometry(
        timestamps=motion.timestamps, wrist_poses_ref=motion.wrist_poses,
        wrist_valid=motion.wrist_valid, joints_ref=joints_world,
        joint_confidence=motion.joint_confidence, joints_local=motion.joints_local,
        camera_from_world=np.asarray(camera_from_world, dtype=np.float64),
        camera_valid=np.asarray(camera_valid, dtype=bool),
        intrinsics=np.asarray(intrinsics, dtype=np.float64).reshape(4),
        source=motion.source, instruction=motion.instruction, metadata=metadata)
