#!/usr/bin/env python
"""InspireHand FK / IK for AgiBot and EgoVLA dexhand (dex-urdf assets).

AgiBot per-hand 6-D (rad, calibrated) → URDF actuated joints → 5×5 anatomical
chains in ``hand_base_link``. Mimic joints are applied from the URDF.

``solve`` is the inverse of that FK (same DLS + ``scale_target`` as Wuji):
Wuji 5×5 → Inspire 6-D, then :meth:`pack_inspire12`. The product map is
``spark-data convert --from wuji2 --to egovla``. The forward pair
(Inspire 6-D FK → Wuji 20-D IK) is ``--from egovla --to wuji2``.

Thumb chains sample link origins (skip CMC ``thumb_proximal_base`` dogleg);
non-thumb fingers use joint origins.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...transforms import axis_angle_matrix
from ..common import (
    DEFAULT_INSPIRE_ASSET_DIR as DEFAULT_ASSET_DIR,
    INSPIRE_ACTUATED as ACTUATED,
    INSPIRE_ANATOMICAL_CHAIN as ANATOMICAL_CHAIN,
    INSPIRE_THUMB_LINK_CHAIN as THUMB_LINK_CHAIN,
    InspireHandKinematics,
)

INSPIRE_DOF = len(ACTUATED)

# 6 URDF mimic joints packed after the 6 actuated names (EgoVLA / --to egovla).
INSPIRE12_MIMIC = (
    "thumb_intermediate_joint",
    "thumb_distal_joint",
    "index_intermediate_joint",
    "middle_intermediate_joint",
    "ring_intermediate_joint",
    "pinky_intermediate_joint",
)
INSPIRE12_ORDER = ACTUATED + INSPIRE12_MIMIC
INSPIRE12_DIM = len(INSPIRE12_ORDER)


class InspireHandModel(InspireHandKinematics):
    """Kinematic FK / IK for one Inspire hand (dex-urdf naming, no side prefix)."""

    def __init__(self, side: str, urdf_path: str | Path | None = None) -> None:
        super().__init__(side, urdf_path=urdf_path)
        self.neutral = np.clip(
            np.zeros(INSPIRE_DOF, dtype=np.float64), self.lower, self.upper
        )
        self.last_solve_used_fallback = False
        neutral_raw = self.forward_chains_local(self.neutral)
        # Knuckle slot 1 matches EgoVLA canonicalize_hand_points(knuckle_index=1).
        x_axis = neutral_raw[1, 1] - neutral_raw[4, 1]
        x_axis /= np.linalg.norm(x_axis)
        y_hint = neutral_raw[2, 1].copy()
        y_hint /= np.linalg.norm(y_hint)
        z_axis = np.cross(x_axis, y_hint)
        z_axis /= np.linalg.norm(z_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        self._canonical_rotation = np.stack([x_axis, y_axis, z_axis], axis=1)
        self._neutral_points = neutral_raw @ self._canonical_rotation

    def _angle_partials(self, name: str) -> np.ndarray:
        """``d(angle)/d(q_actuated)`` for one URDF joint (mimic included)."""
        partials = np.zeros(INSPIRE_DOF, dtype=np.float64)
        joint = self._joints[name]
        if joint.mimic_joint is not None:
            master = joint.mimic_joint
            if master not in self.actuated_names:
                raise ValueError(f"Unsupported mimic chain for {name} -> {master}")
            partials[self.actuated_names.index(master)] = joint.mimic_multiplier
            return partials
        if joint.fixed:
            return partials
        if name in self.actuated_names:
            partials[self.actuated_names.index(name)] = 1.0
            return partials
        raise ValueError(f"No angle available for joint {name}")

    def _link_poses_and_jac_terms(
        self, q_actuated: np.ndarray
    ) -> tuple[dict[str, np.ndarray], dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]]:
        """Link poses plus geometric Jacobian terms ``(axis, origin, d_angle/dq)``."""
        q_actuated = np.asarray(q_actuated, dtype=np.float64).reshape(INSPIRE_DOF)
        poses = {self.base_link: np.eye(4, dtype=np.float64)}
        terms: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
            self.base_link: []
        }
        stack = [self.base_link]
        while stack:
            parent = stack.pop()
            for joint_name in self._children.get(parent, ()):
                joint = self._joints[joint_name]
                origin_pose = poses[parent] @ joint.origin
                motion = np.eye(4, dtype=np.float64)
                child_terms = list(terms[parent])
                if not joint.fixed or joint.mimic_joint is not None:
                    angle = self._angle_for_joint(joint_name, q_actuated)
                    motion[:3, :3] = axis_angle_matrix(joint.axis, angle)
                    partials = self._angle_partials(joint_name)
                    if np.any(partials):
                        axis_world = origin_pose[:3, :3] @ joint.axis
                        child_terms.append(
                            (axis_world, origin_pose[:3, 3].copy(), partials)
                        )
                child_pose = origin_pose @ motion
                poses[joint.child] = child_pose
                terms[joint.child] = child_terms
                stack.append(joint.child)
        return poses, terms

    def _chains_and_jacobian_raw(
        self, q_actuated: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        poses, terms = self._link_poses_and_jac_terms(q_actuated)
        chains = np.empty((5, 5, 3), dtype=np.float64)
        jacobian = np.zeros((5, 5, 3, INSPIRE_DOF), dtype=np.float64)
        base = poses[self.base_link][:3, 3]

        def _apply(
            finger: int,
            slot: int,
            point: np.ndarray,
            point_terms: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        ) -> None:
            chains[finger, slot] = point
            for axis, origin, partials in point_terms:
                jacobian[finger, slot] += np.outer(np.cross(axis, point - origin), partials)

        def _joint_origin(joint_name: str) -> tuple[np.ndarray, list]:
            joint = self._joints[joint_name]
            origin_pose = poses[joint.parent] @ joint.origin
            return origin_pose[:3, 3], terms[joint.parent]

        for finger, names in enumerate(ANATOMICAL_CHAIN):
            if finger == 0:
                _apply(finger, 0, base, terms[self.base_link])
                for i, link_name in enumerate(THUMB_LINK_CHAIN, start=1):
                    _apply(finger, i, poses[link_name][:3, 3], terms[link_name])
                continue
            tip = self._joints[names[-1]]
            tip_pos = poses[tip.child][:3, 3]
            knuckle, knuckle_terms = _joint_origin(names[0])
            mid, mid_terms = _joint_origin(names[1])
            _apply(finger, 0, base, terms[self.base_link])
            _apply(finger, 1, knuckle, knuckle_terms)
            _apply(finger, 2, mid, mid_terms)
            mid_tip = 0.5 * (mid + tip_pos)
            # Midpoint: average the two endpoint jacobians (same as 0.5 * (J_mid + J_tip)).
            chains[finger, 3] = mid_tip
            jacobian[finger, 3] = 0.0
            for axis, origin, partials in mid_terms:
                jacobian[finger, 3] += 0.5 * np.outer(
                    np.cross(axis, mid - origin), partials
                )
            for axis, origin, partials in terms[tip.child]:
                jacobian[finger, 3] += 0.5 * np.outer(
                    np.cross(axis, tip_pos - origin), partials
                )
            _apply(finger, 4, tip_pos, terms[tip.child])
        return chains, jacobian

    def forward_points(self, q_actuated: np.ndarray, canonical: bool = True) -> np.ndarray:
        """Return ``[finger, five chain points, xyz]`` in the Inspire palm frame."""
        points = self.forward_chains_local(q_actuated)
        if not canonical:
            return points
        return points @ self._canonical_rotation

    def forward_points_with_jacobian(
        self, q_actuated: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Canonical 5×5 points and analytic ``d(point_xyz)/d(q6)`` (mimics included)."""
        points, jacobian = self._chains_and_jacobian_raw(q_actuated)
        return (
            points @ self._canonical_rotation,
            np.einsum("fpcj,cd->fpdj", jacobian, self._canonical_rotation),
        )

    def pack_inspire12(self, q_actuated: np.ndarray) -> np.ndarray:
        """6 actuated + 6 URDF mimic, ``INSPIRE12_ORDER``."""
        q_actuated = np.asarray(q_actuated, dtype=np.float64).reshape(INSPIRE_DOF)
        packed = np.empty(INSPIRE12_DIM, dtype=np.float64)
        packed[:INSPIRE_DOF] = q_actuated
        for offset, name in enumerate(INSPIRE12_MIMIC):
            packed[INSPIRE_DOF + offset] = self._angle_for_joint(name, q_actuated)
        return packed

    def scale_target(self, source_points: np.ndarray) -> np.ndarray:
        """Morphology-normalize a 5×5 chain onto Inspire segment lengths.

        Same recipe as :meth:`WujiHandModel.scale_target`: keep source segment
        directions, replace lengths with the Inspire neutral chain. Used by the
        inverse of EgoVLA ``inspire_fk → wujihand2_ik``.
        """
        source_points = np.asarray(source_points, dtype=np.float64)
        target = np.full_like(source_points, np.nan)
        for finger in range(5):
            target[finger, 0] = self._neutral_points[finger, 0]
            for point in range(1, 5):
                source_segment = (
                    source_points[finger, point] - source_points[finger, point - 1]
                )
                source_length = np.linalg.norm(source_segment)
                robot_length = np.linalg.norm(
                    self._neutral_points[finger, point]
                    - self._neutral_points[finger, point - 1]
                )
                if np.isfinite(source_length) and source_length > 1e-5:
                    target[finger, point] = (
                        target[finger, point - 1]
                        + source_segment / source_length * robot_length
                    )
        return target

    def solve(
        self,
        source_points: np.ndarray,
        confidence: np.ndarray,
        initial: np.ndarray | None = None,
        previous: np.ndarray | None = None,
        max_iterations: int = 35,
        tolerance_m: float = 0.03,
        solver: str = "analytic",
    ) -> tuple[np.ndarray, bool, float, int]:
        """Fit Inspire 6-D q to one 5×5 chain using damped least squares."""
        self.last_solve_used_fallback = False
        if solver == "analytic":
            try:
                result = self._solve_analytic(
                    source_points,
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
                source_points,
                confidence,
                initial=initial,
                previous=previous,
                max_iterations=max_iterations,
                tolerance_m=tolerance_m,
            )
        if solver == "finite_difference":
            return self._solve_finite_difference(
                source_points,
                confidence,
                initial=initial,
                previous=previous,
                max_iterations=max_iterations,
                tolerance_m=tolerance_m,
            )
        raise ValueError(f"Unknown IK solver: {solver}")

    def _prepare_target(
        self,
        source_points: np.ndarray,
        confidence: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        target = self.scale_target(source_points)
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
        source_points: np.ndarray,
        confidence: np.ndarray,
    ) -> tuple[bool, float, float]:
        prepared = self._prepare_target(source_points, confidence)
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
        min_distance = float(min(distances)) if distances else float("nan")
        return bool(np.isfinite(rms)), rms, min_distance

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
                        derivative = np.zeros(INSPIRE_DOF, dtype=np.float64)
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
        point_rows = (point_jacobian * weights[..., None, None]).reshape(-1, INSPIRE_DOF)
        point_rows[~np.isfinite(point_rows)] = 0
        jacobian = np.concatenate(
            [
                point_rows,
                regularization_weight * np.eye(INSPIRE_DOF),
                temporal_weight * np.eye(INSPIRE_DOF),
                np.stack(collision_jacobian),
            ],
            axis=0,
        )
        return residual, jacobian

    def _solve_analytic(
        self,
        source_points: np.ndarray,
        confidence: np.ndarray,
        initial: np.ndarray | None,
        previous: np.ndarray | None,
        max_iterations: int,
        tolerance_m: float,
    ) -> tuple[np.ndarray, bool, float, int]:
        prepared = self._prepare_target(source_points, confidence)
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
            lhs = jacobian.T @ jacobian + damping * np.eye(INSPIRE_DOF)
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
        source_points: np.ndarray,
        confidence: np.ndarray,
        initial: np.ndarray | None,
        previous: np.ndarray | None,
        max_iterations: int,
        tolerance_m: float,
    ) -> tuple[np.ndarray, bool, float, int]:
        prepared = self._prepare_target(source_points, confidence)
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
            jacobian = np.empty((len(current), INSPIRE_DOF), dtype=np.float64)
            for joint_index in range(INSPIRE_DOF):
                perturbed = q.copy()
                perturbed[joint_index] = min(
                    q[joint_index] + epsilon, self.upper[joint_index]
                )
                step = perturbed[joint_index] - q[joint_index]
                if step < epsilon * 0.1:
                    perturbed[joint_index] = max(
                        q[joint_index] - epsilon, self.lower[joint_index]
                    )
                    step = perturbed[joint_index] - q[joint_index]
                jacobian[:, joint_index] = (residual(perturbed) - current) / step
            lhs = jacobian.T @ jacobian + damping * np.eye(INSPIRE_DOF)
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
