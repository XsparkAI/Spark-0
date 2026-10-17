"""Wuji URDF forward kinematics and NumPy-only human-to-hand retargeting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from ..schema import UnifiedHumanMotion, UnifiedRobotMotion, pack_spark_state
from ..transforms import (
    axis_angle_matrix,
    rpy_matrix,
)

from .._paths import ASSET_ROOT  # noqa: E402

# Finger geometry is always anatomical: the human readers, ``canonicalize_hand_points``
# and ``_palm_rotation_ref`` all index chains as thumb=0 .. pinky=4. Only the 20-D
# joint-vector order is hand-generation specific.
ANATOMICAL_FINGERS = ("thumb", "index", "middle", "ring", "pinky")
JOINTS_PER_FINGER = 4
HAND_DOF = len(ANATOMICAL_FINGERS) * JOINTS_PER_FINGER

_STAGE_MAJOR_ANATOMICAL = tuple(
    (finger, stage)
    for stage in range(JOINTS_PER_FINGER)
    for finger in range(len(ANATOMICAL_FINGERS))
)
# Isaac articulation order for wujihand2: stage-major, but siblings run
# index, middle, pinky, ring, thumb. Equivalent to applying
# ``data_trasfomer/hand_mapping.WUJIHAND2_STREAM_TO_ISAAC`` to the recorder's
# finger-major stream layout.
_STAGE_MAJOR_ISAAC_HAND2 = tuple(
    (finger, stage)
    for stage in range(JOINTS_PER_FINGER)
    for finger in (1, 2, 4, 3, 0)
)


@dataclass(frozen=True)
class HandVariantSpec:
    """One Wuji hand generation: URDF assets, joint naming, and 20-D ordering.

    ``joint_order`` lists the driven joints as ``(finger, stage)`` pairs, where
    ``finger`` indexes :data:`ANATOMICAL_FINGERS` and ``stage`` is the position
    along that finger. It is both the IK solve order and the on-disk Spark
    ``*_ee_joint_states`` order, so the pipeline needs no separate permutation.
    """

    name: str
    asset_dir: str
    order_name: str
    source_suffix: str
    ee_frame_label: str
    ee_frame_note: str
    side_prefixes: Mapping[str, str]
    finger_segments: tuple[str, ...]
    stages: tuple[tuple[str, ...], ...]
    joint_order: tuple[tuple[int, int], ...]
    urdf_basename: str = "{side}.urdf"

    def __post_init__(self) -> None:
        if len(self.finger_segments) != len(ANATOMICAL_FINGERS):
            raise ValueError(
                f"{self.name}: expected {len(ANATOMICAL_FINGERS)} finger segments"
            )
        if len(self.stages) != len(ANATOMICAL_FINGERS) or any(
            len(stages) != JOINTS_PER_FINGER for stages in self.stages
        ):
            raise ValueError(
                f"{self.name}: expected {JOINTS_PER_FINGER} stages per finger"
            )
        if sorted(self.joint_order) != sorted(_STAGE_MAJOR_ANATOMICAL):
            raise ValueError(
                f"{self.name}: joint_order must permute all {HAND_DOF} (finger, stage) pairs"
            )

    def urdf_path(self, side: str) -> Path:
        return ASSET_ROOT / self.asset_dir / self.urdf_basename.format(side=side)

    def joint_name(self, side: str, finger: int, stage: int) -> str:
        prefix = self.side_prefixes[side]
        return f"{prefix}_{self.finger_segments[finger]}_{self.stages[finger][stage]}"

    def tip_joint_name(self, side: str, finger: int) -> str:
        prefix = self.side_prefixes[side]
        return f"{prefix}_{self.finger_segments[finger]}_tip_fixed"

    def joint_names(self, side: str) -> tuple[str, ...]:
        """Driven joint names in 20-D output order."""
        return tuple(
            self.joint_name(side, finger, stage) for finger, stage in self.joint_order
        )

    def q_indices(self) -> np.ndarray:
        """``(5, 4)`` map from anatomical ``(finger, stage)`` to 20-D index."""
        indices = np.empty((len(ANATOMICAL_FINGERS), JOINTS_PER_FINGER), dtype=np.int64)
        for position, (finger, stage) in enumerate(self.joint_order):
            indices[finger, stage] = position
        return indices


WUJIHAND2_V2 = HandVariantSpec(
    name="wujihand2",
    # Whole-arm Tianji v2 asset (arm + hand); the FK/IK only consume the
    # hand subchain parented on {l,r}_wrist.
    asset_dir="wuji_hand2_description",
    urdf_basename="tianji_wujihand2_{side}.urdf",
    order_name="isaac_wujihand2",
    source_suffix="wujihand2",
    # The EE stage is shared verbatim with v1 and applies no Link7->wrist hand
    # mount, matching data_trasfomer/to_spark_world_frame.py --ee-frame link7.
    ee_frame_label="link7",
    ee_frame_note=(
        "EE denotes the arm flange (Link7_L / Link7_R) as in "
        "data/v2_processed/spark_real_*; no hand-mount offset is applied and the "
        "orientation anchor is the reused v1 target_ee_orientation_wxyz."
    ),
    side_prefixes={"left": "l", "right": "r"},
    finger_segments=("thumb", "index_finger", "middle_finger", "ring_finger", "pinky"),
    stages=(
        ("cmc_flex", "cmc_abd", "mcp", "ip"),
        *(("mcp_flex", "mcp_abd", "pip", "dip") for _ in range(4)),
    ),
    joint_order=_STAGE_MAJOR_ISAAC_HAND2,
)

HAND_VARIANTS: Mapping[str, HandVariantSpec] = {
    WUJIHAND2_V2.name: WUJIHAND2_V2,
}
DEFAULT_HAND_VARIANT = WUJIHAND2_V2.name


def resolve_hand_variant(variant: str | HandVariantSpec) -> HandVariantSpec:
    if isinstance(variant, HandVariantSpec):
        return variant
    try:
        return HAND_VARIANTS[variant]
    except KeyError:
        raise ValueError(
            f"Unknown Wuji hand variant {variant!r}; expected one of "
            f"{sorted(HAND_VARIANTS)}"
        ) from None


@dataclass(frozen=True)
class JointSpec:
    name: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float
    fixed: bool = False


@dataclass
class RetargetResult:
    joint_positions: np.ndarray
    valid: np.ndarray
    rms_error_m: np.ndarray
    iterations: np.ndarray
    stats: dict[str, object]


class WujiHandModel:
    """Kinematic model backed by a pinned Wuji hand URDF.

    ``variant`` selects the hand generation (see :data:`HAND_VARIANTS`). It
    changes the URDF, joint names, limits, axes, and the 20-D joint order; the
    FK, analytic Jacobian, and IK below are shared because both generations
    have the same topology of four revolute joints plus a fixed tip per finger.
    """

    def __init__(
        self,
        side: str,
        urdf_path: str | Path | None = None,
        variant: str | HandVariantSpec = DEFAULT_HAND_VARIANT,
    ) -> None:
        if side not in {"left", "right"}:
            raise ValueError("side must be 'left' or 'right'")
        self.side = side
        self.variant = resolve_hand_variant(variant)
        path = (
            Path(urdf_path)
            if urdf_path is not None
            else self.variant.urdf_path(side)
        )
        self.urdf_path = path
        self._joints = self._parse_urdf(path)
        self.joint_names = self.variant.joint_names(side)
        tip_names = tuple(
            self.variant.tip_joint_name(side, finger)
            for finger in range(len(ANATOMICAL_FINGERS))
        )
        missing = [
            name
            for name in self.joint_names + tip_names
            if name not in self._joints
        ]
        if missing:
            raise ValueError(
                f"URDF {path} is missing expected {self.variant.name} joints: {missing}"
            )
        self._q_indices = self.variant.q_indices()
        self._chain = tuple(
            tuple(
                self._joints[self.variant.joint_name(side, finger, stage)]
                for stage in range(JOINTS_PER_FINGER)
            )
            for finger in range(len(ANATOMICAL_FINGERS))
        )
        self._tips = tuple(self._joints[name] for name in tip_names)
        self.lower = np.array([self._joints[name].lower for name in self.joint_names])
        self.upper = np.array([self._joints[name].upper for name in self.joint_names])
        self.neutral = np.clip(np.zeros(HAND_DOF, dtype=np.float64), self.lower, self.upper)
        self.last_solve_used_fallback = False
        neutral_raw, _ = self._forward_raw(self.neutral, with_jacobian=False)
        x_axis = neutral_raw[1, 0] - neutral_raw[4, 0]
        x_axis /= np.linalg.norm(x_axis)
        y_hint = neutral_raw[2, 0].copy()
        y_hint /= np.linalg.norm(y_hint)
        z_axis = np.cross(x_axis, y_hint)
        z_axis /= np.linalg.norm(z_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        self._canonical_rotation = np.stack([x_axis, y_axis, z_axis], axis=1)
        self._neutral_points = neutral_raw @ self._canonical_rotation
        self.palm_width = float(
            np.linalg.norm(self._neutral_points[1, 0] - self._neutral_points[4, 0])
        )

    @staticmethod
    def _parse_urdf(path: Path) -> dict[str, JointSpec]:
        if not path.is_file():
            raise FileNotFoundError(f"Wuji URDF not found: {path}")
        root = ET.parse(path).getroot()
        joints: dict[str, JointSpec] = {}
        for element in root.findall("joint"):
            name = element.attrib["name"]
            fixed = element.attrib.get("type") == "fixed"
            origin_element = element.find("origin")
            xyz = np.fromstring(
                (origin_element.attrib.get("xyz", "0 0 0") if origin_element is not None else "0 0 0"),
                sep=" ",
            )
            rpy = np.fromstring(
                (origin_element.attrib.get("rpy", "0 0 0") if origin_element is not None else "0 0 0"),
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
            joints[name] = JointSpec(
                name=name,
                parent=element.find("parent").attrib["link"],
                child=element.find("child").attrib["link"],
                origin=origin,
                axis=axis,
                lower=lower,
                upper=upper,
                fixed=fixed,
            )
        return joints

    def _forward_raw(
        self,
        q_spark: np.ndarray,
        with_jacobian: bool,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        q = np.asarray(q_spark, dtype=np.float64)
        if q.shape != (HAND_DOF,):
            raise ValueError(f"Expected q shape ({HAND_DOF},), got {q.shape}")
        points = np.empty((5, 5, 3), dtype=np.float64)
        jacobian = (
            np.zeros((5, 5, 3, HAND_DOF), dtype=np.float64) if with_jacobian else None
        )
        for finger in range(len(ANATOMICAL_FINGERS)):
            transform = np.eye(4, dtype=np.float64)
            origins: list[np.ndarray] = []
            axes: list[np.ndarray] = []
            indices: list[int] = []
            for stage in range(JOINTS_PER_FINGER):
                joint = self._chain[finger][stage]
                transform = transform @ joint.origin
                points[finger, stage] = transform[:3, 3]
                q_index = int(self._q_indices[finger, stage])
                origins.append(transform[:3, 3].copy())
                axes.append(transform[:3, :3] @ joint.axis)
                indices.append(q_index)
                rotation = np.eye(4, dtype=np.float64)
                rotation[:3, :3] = axis_angle_matrix(joint.axis, q[q_index])
                transform = transform @ rotation
            transform = transform @ self._tips[finger].origin
            points[finger, 4] = transform[:3, 3]
            if jacobian is not None:
                for stage, (origin, axis, q_index) in enumerate(
                    zip(origins, axes, indices)
                ):
                    for point_index in range(stage + 1, 5):
                        jacobian[finger, point_index, :, q_index] = np.cross(
                            axis,
                            points[finger, point_index] - origin,
                        )
        return points, jacobian

    def forward_points(self, q_spark: np.ndarray, canonical: bool = True) -> np.ndarray:
        """Return ``[finger, five chain points, xyz]`` in palm coordinates."""
        points, _ = self._forward_raw(q_spark, with_jacobian=False)
        if not canonical:
            return points
        return points @ self._canonical_rotation

    def forward_points_with_jacobian(
        self,
        q_spark: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return canonical points and analytic ``d(point_xyz)/d(q)``."""
        points, jacobian = self._forward_raw(q_spark, with_jacobian=True)
        assert jacobian is not None
        return (
            points @ self._canonical_rotation,
            np.einsum("fpcj,cd->fpdj", jacobian, self._canonical_rotation),
        )

    def scale_target(self, human_points: np.ndarray) -> np.ndarray:
        """Morphology-normalize human chains to Wuji segment lengths.

        A single palm-width scale leaves large systematic errors because the
        human wrist-to-MCP offsets and finger proportions differ from Wuji.
        Retargeting should preserve segment directions, not human bone lengths.
        """
        human_points = np.asarray(human_points, dtype=np.float64)
        target = np.full_like(human_points, np.nan)
        for finger in range(5):
            target[finger, 0] = self._neutral_points[finger, 0]
            for point in range(1, 5):
                human_segment = human_points[finger, point] - human_points[finger, point - 1]
                human_length = np.linalg.norm(human_segment)
                robot_length = np.linalg.norm(
                    self._neutral_points[finger, point]
                    - self._neutral_points[finger, point - 1]
                )
                if np.isfinite(human_length) and human_length > 1e-5:
                    target[finger, point] = (
                        target[finger, point - 1]
                        + human_segment / human_length * robot_length
                    )
        return target

    def solve(
        self,
        human_points: np.ndarray,
        confidence: np.ndarray,
        initial: np.ndarray | None = None,
        previous: np.ndarray | None = None,
        max_iterations: int = 35,
        tolerance_m: float = 0.03,
        solver: str = "analytic",
    ) -> tuple[np.ndarray, bool, float, int]:
        """Fit Wuji q to one human hand frame using damped least squares."""
        self.last_solve_used_fallback = False
        if solver == "analytic":
            try:
                result = self._solve_analytic(
                    human_points,
                    confidence,
                    initial=initial,
                    previous=previous,
                    max_iterations=max_iterations,
                    tolerance_m=tolerance_m,
                )
                if result[3] == 0 or np.all(np.isfinite(result[0])):
                    return result
            except (np.linalg.LinAlgError, FloatingPointError):
                pass
            self.last_solve_used_fallback = True
            return self._solve_finite_difference(
                human_points,
                confidence,
                initial=initial,
                previous=previous,
                max_iterations=max_iterations,
                tolerance_m=tolerance_m,
            )
        if solver == "finite_difference":
            return self._solve_finite_difference(
                human_points,
                confidence,
                initial=initial,
                previous=previous,
                max_iterations=max_iterations,
                tolerance_m=tolerance_m,
            )
        raise ValueError(f"Unknown IK solver: {solver}")

    def _prepare_target(
        self,
        human_points: np.ndarray,
        confidence: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        target = self.scale_target(human_points)
        confidence = np.asarray(confidence, dtype=np.float64)
        available = np.isfinite(target).all(axis=-1) & (confidence > 0)
        if np.count_nonzero(available) < 10 or np.count_nonzero(available[:, -1]) < 3:
            return None
        point_importance = np.array([0.25, 0.75, 1.0, 1.5, 2.0])[None, :]
        weights = np.sqrt(np.clip(confidence, 0, 1) * point_importance)
        weights[~available] = 0
        return target, available, weights

    def evaluate(
        self,
        q: np.ndarray,
        human_points: np.ndarray,
        confidence: np.ndarray,
    ) -> tuple[bool, float, float]:
        """Evaluate one candidate against the unweighted target and collisions."""
        prepared = self._prepare_target(human_points, confidence)
        if prepared is None:
            return False, float("nan"), float("nan")
        target, available, _weights = prepared
        predicted = self.forward_points(q)
        errors = np.linalg.norm(predicted[available] - target[available], axis=-1)
        rms = float(np.sqrt(np.mean(errors**2)))
        distances = [
            np.linalg.norm(predicted[first, point] - predicted[second, point])
            for first in range(5)
            for second in range(first + 1, 5)
            for point in (2, 3, 4)
        ]
        min_distance = float(min(distances))
        valid = bool(np.isfinite(rms))
        return valid, rms, min_distance

    def _residual_and_jacobian(
        self,
        candidate: np.ndarray,
        target: np.ndarray,
        weights: np.ndarray,
        previous_q: np.ndarray,
        with_jacobian: bool,
        regularization_weight: float = 0.01,
        temporal_weight: float = 0.03,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if with_jacobian:
            predicted, point_jacobian = self.forward_points_with_jacobian(candidate)
        else:
            predicted = self.forward_points(candidate)
            point_jacobian = None
        point_residual = ((predicted - target) * weights[..., None]).reshape(-1)
        point_residual[~np.isfinite(point_residual)] = 0
        regularization = regularization_weight * (candidate - self.neutral)
        temporal = temporal_weight * (candidate - previous_q)
        collision: list[float] = []
        collision_jacobian: list[np.ndarray] = []
        for first in range(5):
            for second in range(first + 1, 5):
                for point_index in (2, 3, 4):
                    delta = predicted[first, point_index] - predicted[second, point_index]
                    distance = float(np.linalg.norm(delta))
                    active = distance < 0.004
                    collision.append(max(0.0, 0.004 - distance) * 2.0)
                    if point_jacobian is not None:
                        derivative = np.zeros(HAND_DOF, dtype=np.float64)
                        if active and distance > 1e-12:
                            derivative = -2.0 * (
                                (delta / distance)
                                @ (
                                    point_jacobian[first, point_index]
                                    - point_jacobian[second, point_index]
                                )
                            )
                        collision_jacobian.append(derivative)
        residual = np.concatenate(
            [point_residual, regularization, temporal, np.asarray(collision)]
        )
        if point_jacobian is None:
            return residual, None
        point_rows = (
            point_jacobian * weights[..., None, None]
        ).reshape(-1, HAND_DOF)
        point_rows[~np.isfinite(point_rows)] = 0
        jacobian = np.concatenate(
            [
                point_rows,
                regularization_weight * np.eye(HAND_DOF),
                temporal_weight * np.eye(HAND_DOF),
                np.stack(collision_jacobian),
            ],
            axis=0,
        )
        return residual, jacobian

    def _solve_analytic(
        self,
        human_points: np.ndarray,
        confidence: np.ndarray,
        initial: np.ndarray | None,
        previous: np.ndarray | None,
        max_iterations: int,
        tolerance_m: float,
    ) -> tuple[np.ndarray, bool, float, int]:
        prepared = self._prepare_target(human_points, confidence)
        if prepared is None:
            return self.neutral.copy(), False, float("nan"), 0
        target, available, weights = prepared
        q = np.array(initial if initial is not None else self.neutral, dtype=np.float64)
        q = np.clip(q, self.lower, self.upper)
        previous_q = (
            np.asarray(previous, dtype=np.float64) if previous is not None else self.neutral
        )
        damping = 2e-4
        iterations = 0
        refinement_iterations = min(8, max_iterations // 3)
        for iterations in range(1, max_iterations + 1):
            fit_only = iterations > max_iterations - refinement_iterations
            current, jacobian = self._residual_and_jacobian(
                q,
                target,
                weights,
                previous_q,
                with_jacobian=True,
                regularization_weight=0.0 if fit_only else 0.01,
                temporal_weight=0.0 if fit_only else 0.03,
            )
            assert jacobian is not None
            lhs = jacobian.T @ jacobian + damping * np.eye(HAND_DOF)
            update = -np.linalg.solve(lhs, jacobian.T @ current)
            q_next = np.clip(q + update, self.lower, self.upper)
            if np.linalg.norm(q_next - q) < 1e-5:
                q = q_next
                predicted = self.forward_points(q)
                errors = np.linalg.norm(predicted[available] - target[available], axis=-1)
                if float(np.sqrt(np.mean(errors**2))) <= tolerance_m or fit_only:
                    break
                continue
            next_residual, _ = self._residual_and_jacobian(
                q_next,
                target,
                weights,
                previous_q,
                with_jacobian=False,
                regularization_weight=0.0 if fit_only else 0.01,
                temporal_weight=0.0 if fit_only else 0.03,
            )
            if np.linalg.norm(next_residual) > np.linalg.norm(current):
                damping = min(damping * 5, 1.0)
            else:
                q = q_next
                damping = max(damping / 2, 1e-6)
            predicted = self.forward_points(q)
            errors = np.linalg.norm(predicted[available] - target[available], axis=-1)
            rms = float(np.sqrt(np.mean(errors**2)))
            if rms <= tolerance_m and iterations >= 2:
                break
        predicted = self.forward_points(q)
        errors = np.linalg.norm(predicted[available] - target[available], axis=-1)
        rms = float(np.sqrt(np.mean(errors**2)))
        valid = bool(np.isfinite(rms) and rms <= tolerance_m)
        return q, valid, rms, iterations

    def _solve_finite_difference(
        self,
        human_points: np.ndarray,
        confidence: np.ndarray,
        initial: np.ndarray | None,
        previous: np.ndarray | None,
        max_iterations: int,
        tolerance_m: float,
    ) -> tuple[np.ndarray, bool, float, int]:
        prepared = self._prepare_target(human_points, confidence)
        if prepared is None:
            return self.neutral.copy(), False, float("nan"), 0
        target, available, weights = prepared
        q = np.array(initial if initial is not None else self.neutral, dtype=np.float64)
        q = np.clip(q, self.lower, self.upper)
        previous_q = (
            np.asarray(previous, dtype=np.float64) if previous is not None else self.neutral
        )

        def residual(candidate: np.ndarray) -> np.ndarray:
            values, _ = self._residual_and_jacobian(
                candidate,
                target,
                weights,
                previous_q,
                with_jacobian=False,
            )
            return values

        epsilon = 1e-5
        damping = 2e-4
        iterations = 0
        for iterations in range(1, max_iterations + 1):
            current = residual(q)
            jacobian = np.empty((len(current), HAND_DOF), dtype=np.float64)
            for joint_index in range(HAND_DOF):
                perturbed = q.copy()
                perturbed[joint_index] = min(q[joint_index] + epsilon, self.upper[joint_index])
                step = perturbed[joint_index] - q[joint_index]
                if step < epsilon * 0.1:
                    perturbed[joint_index] = max(
                        q[joint_index] - epsilon, self.lower[joint_index]
                    )
                    step = perturbed[joint_index] - q[joint_index]
                jacobian[:, joint_index] = (residual(perturbed) - current) / step
            lhs = jacobian.T @ jacobian + damping * np.eye(HAND_DOF)
            update = -np.linalg.solve(lhs, jacobian.T @ current)
            q_next = np.clip(q + update, self.lower, self.upper)
            if np.linalg.norm(q_next - q) < 1e-5:
                q = q_next
                break
            if np.linalg.norm(residual(q_next)) > np.linalg.norm(current):
                damping = min(damping * 5, 1.0)
            else:
                q = q_next
                damping = max(damping / 2, 1e-6)

        predicted = self.forward_points(q)
        errors = np.linalg.norm(predicted[available] - target[available], axis=-1)
        rms = float(np.sqrt(np.mean(errors**2)))
        valid = bool(np.isfinite(rms) and rms <= tolerance_m)
        return q, valid, rms, iterations


def hand_models(
    variant: str | HandVariantSpec = DEFAULT_HAND_VARIANT,
    sides: Sequence[str] = ("left", "right"),
) -> dict[str, WujiHandModel]:
    """Build one :class:`WujiHandModel` per side for a hand generation."""
    spec = resolve_hand_variant(variant)
    return {side: WujiHandModel(side, variant=spec) for side in sides}


def retarget_hands(
    motion: UnifiedHumanMotion,
    models: dict[str, WujiHandModel] | None = None,
    tolerance_m: float = 0.03,
    solver: str = "analytic",
    stride: int = 1,
    max_iterations: int = 35,
    interpolation_collision_m: float = 0.002,
    hand_variant: str | HandVariantSpec = DEFAULT_HAND_VARIANT,
) -> RetargetResult:
    """Retarget valid hands, optionally interpolating between IK keyframes.

    ``hand_variant`` is only used to build the default models; explicit
    ``models`` win and must agree with each other on the hand generation.
    """
    if stride < 1:
        raise ValueError("IK stride must be at least 1")
    if max_iterations < 1:
        raise ValueError("IK max_iterations must be at least 1")
    models = models or hand_models(hand_variant)
    variants = {model.variant.name for model in models.values()}
    if len(variants) != 1:
        raise ValueError(f"Hand models mix Wuji generations: {sorted(variants)}")
    length = len(motion.timestamps)
    q = np.full((length, 2, HAND_DOF), np.nan, dtype=np.float64)
    valid = np.zeros((length, 2), dtype=bool)
    errors = np.full((length, 2), np.nan, dtype=np.float64)
    iterations = np.zeros((length, 2), dtype=np.int64)
    hand_stats: dict[str, dict[str, int]] = {}
    for hand_index, side in enumerate(("left", "right")):
        model = models[side]
        usable = np.asarray(
            [
                model._prepare_target(
                    motion.joints_local[frame, hand_index],
                    motion.joint_confidence[frame, hand_index],
                )
                is not None
                for frame in range(length)
            ],
            dtype=bool,
        )
        keyframes: list[int] = []
        cursor = 0
        while cursor < length:
            if not usable[cursor]:
                cursor += 1
                continue
            segment_start = cursor
            while cursor + 1 < length and usable[cursor + 1]:
                cursor += 1
            segment_end = cursor
            segment_keys = list(range(segment_start, segment_end + 1, stride))
            if segment_keys[-1] != segment_end:
                segment_keys.append(segment_end)
            keyframes.extend(segment_keys)
            cursor += 1

        solved_frames = 0
        interpolated_frames = 0
        fallback_frames = 0
        legacy_fallback_frames = 0
        previous = model.neutral
        for frame in keyframes:
            result = model.solve(
                motion.joints_local[frame, hand_index],
                motion.joint_confidence[frame, hand_index],
                initial=previous,
                previous=previous,
                max_iterations=max_iterations,
                tolerance_m=tolerance_m,
                solver=solver,
            )
            q[frame, hand_index], valid[frame, hand_index], errors[frame, hand_index], iterations[
                frame, hand_index
            ] = result
            solved_frames += 1
            legacy_fallback_frames += int(model.last_solve_used_fallback)
            if valid[frame, hand_index]:
                previous = q[frame, hand_index]

        if stride > 1:
            for left, right in zip(keyframes[:-1], keyframes[1:]):
                if right - left <= 1 or not (
                    valid[left, hand_index] and valid[right, hand_index]
                ):
                    continue
                left_mask = motion.joint_confidence[left, hand_index] > 0
                right_mask = motion.joint_confidence[right, hand_index] > 0
                for frame in range(left + 1, right):
                    if not usable[frame] or not np.array_equal(
                        motion.joint_confidence[frame, hand_index] > 0,
                        left_mask,
                    ) or not np.array_equal(left_mask, right_mask):
                        continue
                    amount = (frame - left) / (right - left)
                    candidate = np.clip(
                        (1.0 - amount) * q[left, hand_index]
                        + amount * q[right, hand_index],
                        model.lower,
                        model.upper,
                    )
                    evaluable, rms, min_distance = model.evaluate(
                        candidate,
                        motion.joints_local[frame, hand_index],
                        motion.joint_confidence[frame, hand_index],
                    )
                    if (
                        evaluable
                        and rms <= tolerance_m
                        and min_distance >= interpolation_collision_m
                    ):
                        q[frame, hand_index] = candidate
                        valid[frame, hand_index] = True
                        errors[frame, hand_index] = rms
                        interpolated_frames += 1

        previous = model.neutral
        for frame in range(length):
            if valid[frame, hand_index]:
                previous = q[frame, hand_index]
                continue
            if not usable[frame]:
                continue
            initial = (
                q[frame, hand_index]
                if np.all(np.isfinite(q[frame, hand_index]))
                else previous
            )
            result = model.solve(
                motion.joints_local[frame, hand_index],
                motion.joint_confidence[frame, hand_index],
                initial=initial,
                previous=previous,
                max_iterations=max_iterations,
                tolerance_m=tolerance_m,
                solver=solver,
            )
            q[frame, hand_index], valid[frame, hand_index], errors[frame, hand_index], iterations[
                frame, hand_index
            ] = result
            solved_frames += 1
            fallback_frames += 1
            legacy_fallback_frames += int(model.last_solve_used_fallback)
            if valid[frame, hand_index]:
                previous = q[frame, hand_index]
        hand_stats[side] = {
            "usable_frames": int(np.count_nonzero(usable)),
            "solved_frames": solved_frames,
            "interpolated_frames": interpolated_frames,
            "fallback_frames": fallback_frames,
            "legacy_fallback_frames": legacy_fallback_frames,
            "iteration_sum": int(np.sum(iterations[:, hand_index])),
        }
    return RetargetResult(
        q,
        valid,
        errors,
        iterations,
        stats={
            "solver": solver,
            "stride": stride,
            "max_iterations": max_iterations,
            "interpolation_collision_m": interpolation_collision_m,
            "hand_variant": next(iter(variants)),
            "hands": hand_stats,
        },
    )


def retarget_camera_aligned_motion(
    motion: UnifiedHumanMotion,
    models: dict[str, WujiHandModel] | None = None,
    tolerance_m: float = 0.03,
    solver: str = "analytic",
    stride: int = 1,
    max_iterations: int = 35,
    hand_variant: str | HandVariantSpec = DEFAULT_HAND_VARIANT,
) -> UnifiedRobotMotion:
    """Produce Spark 54-D from camera-aligned EE poses.

    ``motion.wrist_poses`` must already contain EE poses generated by the
    camera-alignment stage. Hand IK still fits the morphology-normalized
    5×5 Wuji target returned by :meth:`WujiHandModel.scale_target`.

    ``hand_variant`` swaps only the hand: the EE block is copied from the shared
    camera-alignment stage for every generation, so switching generations never
    moves an EE pose.
    """
    spec = models[next(iter(models))].variant if models else resolve_hand_variant(
        hand_variant
    )
    hands = retarget_hands(
        motion,
        models=models,
        tolerance_m=tolerance_m,
        solver=solver,
        stride=stride,
        max_iterations=max_iterations,
        hand_variant=spec,
    )
    ee = np.array(motion.wrist_poses, copy=True)
    state = pack_spark_state(
        ee[:, 0],
        hands.joint_positions[:, 0],
        ee[:, 1],
        hands.joint_positions[:, 1],
    )
    state_valid = np.zeros_like(state, dtype=bool)
    state_valid[:, 0:7] = motion.wrist_valid[:, 0, None]
    state_valid[:, 7:27] = hands.valid[:, 0, None]
    state_valid[:, 27:34] = motion.wrist_valid[:, 1, None]
    state_valid[:, 34:54] = hands.valid[:, 1, None]
    state[~state_valid] = np.nan
    return UnifiedRobotMotion(
        timestamps=motion.timestamps,
        state=state,
        state_valid=state_valid,
        source=f"{motion.source}:camera_aligned_retargeted_to_{spec.source_suffix}",
        retarget_error=hands.rms_error_m,
        action_source="not_available",
        instruction=motion.instruction,
        metadata={
            **motion.metadata,
            "hand_variant": spec.name,
            "hand_urdf_assets": spec.asset_dir,
            "wuji_order": spec.order_name,
            "ee_frame": spec.ee_frame_label,
            "ee_frame_note": spec.ee_frame_note,
            "ee_alignment": "spark_camera_mount_axes",
            "retargeter": "official_urdf_scaled_target_damped_least_squares",
            "retarget_error_definition": "wuji_fk_vs_morphology_normalized_target_rms_m",
            "ik": hands.stats,
        },
    )


def retarget_source_robot_motion(
    motion: UnifiedHumanMotion,
    models: dict[str, WujiHandModel] | None = None,
    tolerance_m: float = 0.03,
    solver: str = "analytic",
    stride: int = 1,
    max_iterations: int = 35,
    hand_variant: str | HandVariantSpec = DEFAULT_HAND_VARIANT,
) -> UnifiedRobotMotion:
    """Produce Spark 54-D from source-robot EE poses (no camera alignment).

    Used by Fourier ActionNet: ``motion.wrist_poses`` are GR1 end-link poses in
    the source robot frame and ``joints_local`` come from Fourier-hand FK.
    """
    if motion.metadata.get("coordinate_frame") != "source_robot_as_is":
        raise ValueError(
            "retarget_source_robot_motion requires coordinate_frame=source_robot_as_is"
        )
    spec = models[next(iter(models))].variant if models else resolve_hand_variant(
        hand_variant
    )
    hands = retarget_hands(
        motion,
        models=models,
        tolerance_m=tolerance_m,
        solver=solver,
        stride=stride,
        max_iterations=max_iterations,
        hand_variant=spec,
    )
    ee = np.array(motion.wrist_poses, copy=True)
    state = pack_spark_state(
        ee[:, 0],
        hands.joint_positions[:, 0],
        ee[:, 1],
        hands.joint_positions[:, 1],
    )
    state_valid = np.zeros_like(state, dtype=bool)
    state_valid[:, 0:7] = motion.wrist_valid[:, 0, None]
    state_valid[:, 7:27] = hands.valid[:, 0, None]
    state_valid[:, 27:34] = motion.wrist_valid[:, 1, None]
    state_valid[:, 34:54] = hands.valid[:, 1, None]
    state[~state_valid] = np.nan
    # Drop bulky arrays from robot metadata; writer reads them from ``motion``.
    metadata = {
        key: value
        for key, value in motion.metadata.items()
        if key != "joints_world"
    }
    return UnifiedRobotMotion(
        timestamps=motion.timestamps,
        state=state,
        state_valid=state_valid,
        source=f"{motion.source}:source_robot_retargeted_to_{spec.source_suffix}",
        retarget_error=hands.rms_error_m,
        action_source="not_available",
        instruction=motion.instruction,
        metadata={
            **metadata,
            "hand_variant": spec.name,
            "hand_urdf_assets": spec.asset_dir,
            "wuji_order": spec.order_name,
            "ee_frame": "fourier_end_link",
            "ee_frame_note": (
                "EE is the Fourier GR1 arm end-link from state/pose (9-D xyz + R "
                "columns); not Tianji Link7. Do not mix with spark_real_processed "
                "as the same embodiment without an extra calibration."
            ),
            "ee_alignment": "fourier_state_pose_as_is",
            "coordinate_frame": "source_robot_as_is",
            "retargeter": "fourier_fk_to_wuji_scaled_target_damped_least_squares",
            "retarget_error_definition": "wuji_fk_vs_morphology_normalized_target_rms_m",
            "ik": hands.stats,
            "camera_alignment": {
                "skipped": True,
                "reason": "source_robot_as_is_fourier_actionnet",
            },
        },
    )
