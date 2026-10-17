#!/usr/bin/env python3
"""Compute pose B expressed in pose A's coordinate frame.

Pose convention:
  [x, y, z, qw, qx, qy, qz]

Both input poses must be expressed in the same parent/world coordinate frame.
The output is B's pose in A-centered coordinates:

  T_A_from_B = inverse(T_world_from_A) @ T_world_from_B

Usage (from the repository root):

  python spark0/relative_pose.py \\
    --pose-a '[0,0,0,1,0,0,0]' \\
    --pose-b '[1,0,0,1,0,0,0]'
"""

from __future__ import annotations

import argparse
import json

import numpy as np

try:
    from .ee_delta_action import matrix_to_quat_wxyz, quat_wxyz_to_matrix
except ImportError:  # ``python spark0/relative_pose.py``
    from ee_delta_action import matrix_to_quat_wxyz, quat_wxyz_to_matrix


def pose_wxyz_to_matrix(pose: np.ndarray) -> np.ndarray:
    """``[xyz(3), quat_wxyz(4)]`` -> 4x4 ``T_world_from_pose``."""
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError(f"Pose must have shape (7,), got {pose.shape}")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quat_wxyz_to_matrix(pose[3:7])
    transform[:3, 3] = pose[:3]
    return transform


def matrix_to_pose_wxyz(transform: np.ndarray) -> np.ndarray:
    """4x4 rigid transform -> ``[xyz(3), quat_wxyz(4)]`` (``w >= 0``)."""
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"Transform must have shape (4,4), got {transform.shape}")
    return np.concatenate(
        [transform[:3, 3], matrix_to_quat_wxyz(transform[:3, :3])],
        axis=0,
    )


def invert_rigid(transform: np.ndarray) -> np.ndarray:
    """Invert a 4x4 rigid transform via ``R.T`` / ``-R.T @ t``."""
    transform = np.asarray(transform, dtype=np.float64)
    result = np.eye(4, dtype=np.float64)
    rotation_t = transform[:3, :3].T
    result[:3, :3] = rotation_t
    result[:3, 3] = -(rotation_t @ transform[:3, 3])
    return result


def relative_pose_b_in_a(pose_a: np.ndarray, pose_b: np.ndarray) -> np.ndarray:
    """Return pose B expressed in A's coordinate frame.

    Args:
        pose_a: A pose in the shared world frame, ``[x,y,z,qw,qx,qy,qz]``.
        pose_b: B pose in the same shared world frame, ``[x,y,z,qw,qx,qy,qz]``.

    Returns:
        B pose in A frame, ``[x,y,z,qw,qx,qy,qz]`` (quaternion sign ``w >= 0``).
    """
    world_from_a = pose_wxyz_to_matrix(pose_a)
    world_from_b = pose_wxyz_to_matrix(pose_b)
    a_from_b = invert_rigid(world_from_a) @ world_from_b
    return matrix_to_pose_wxyz(a_from_b)


# ── Batched forms: MANY poses against ONE frame A ────────────────────────
# The deploy path always transforms a whole block against a single camera
# pose (a state history every control step, a 50-row action chunk per
# inference). The quaternion helpers below are already batched, so the
# per-pose loop was paying their fixed numpy overhead N times over AND
# rebuilding A's matrix (and its inverse) once per pose. These do the same
# arithmetic with A hoisted out; `BatchedPoseTransformTest` pins them to the
# scalar functions above, which stay the reference definition.


def poses_wxyz_to_matrix(poses: np.ndarray) -> np.ndarray:
    """Batched :func:`pose_wxyz_to_matrix`: (..., 7) -> (..., 4, 4)."""
    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape[-1] != 7:
        raise ValueError(f"Poses must have shape (..., 7), got {poses.shape}")
    transform = np.zeros(poses.shape[:-1] + (4, 4), dtype=np.float64)
    transform[..., :3, :3] = quat_wxyz_to_matrix(poses[..., 3:7])
    transform[..., :3, 3] = poses[..., :3]
    transform[..., 3, 3] = 1.0
    return transform


def matrices_to_poses_wxyz(transforms: np.ndarray) -> np.ndarray:
    """Batched :func:`matrix_to_pose_wxyz`: (..., 4, 4) -> (..., 7), ``w >= 0``."""
    transforms = np.asarray(transforms, dtype=np.float64)
    if transforms.shape[-2:] != (4, 4):
        raise ValueError(
            f"Transforms must have shape (..., 4, 4), got {transforms.shape}"
        )
    return np.concatenate(
        [transforms[..., :3, 3], matrix_to_quat_wxyz(transforms[..., :3, :3])],
        axis=-1,
    )


def relative_poses_b_in_a(pose_a: np.ndarray, poses_b: np.ndarray) -> np.ndarray:
    """Batched :func:`relative_pose_b_in_a`: one frame A, ``(..., 7)`` B poses."""
    a_from_world = invert_rigid(pose_wxyz_to_matrix(pose_a))
    return matrices_to_poses_wxyz(a_from_world @ poses_wxyz_to_matrix(poses_b))


def compose_poses_a_with_rel(pose_a: np.ndarray, poses_b_in_a: np.ndarray) -> np.ndarray:
    """Batched :func:`compose_pose_a_with_rel`: one frame A, ``(..., 7)`` poses."""
    world_from_a = pose_wxyz_to_matrix(pose_a)
    return matrices_to_poses_wxyz(world_from_a @ poses_wxyz_to_matrix(poses_b_in_a))


# ── Camera-extrinsics conventions ────────────────────────────────────────
# The training pipeline (ee_delta_dataset) normalizes the stored extrinsics
# column to ``world_from_camera`` and REFUSES an ambiguous declaration, but it
# applies no axis correction at all: ``camera_axes`` from ``meta/modality.json``
# only enters the converted-stats cache key. Deployment therefore has to hand
# the codec extrinsics in the very same convention the dataset was recorded
# in, and reading them the wrong way corrupts every EE pose without raising.
# These helpers make that contract explicit on the deploy side too.
WORLD_FROM_CAMERA = "world_from_camera"
CAMERA_FROM_WORLD = "camera_from_world"
EXTRINSICS_SEMANTICS = (WORLD_FROM_CAMERA, CAMERA_FROM_WORLD)

# Pipeline default (OpenGL / Blender style): +X right, +Y up, +Z backwards.
TRAINING_CAMERA_AXES = "x_right_y_up_z_back"
# OpenCV / most RGB-D SDKs: +X right, +Y down, +Z forward. The two differ by a
# 180 deg rotation about the camera X axis, i.e. diag(1, -1, -1) — a proper
# rotation (det +1), so the corrected pose is still a valid rigid transform.
OPENCV_CAMERA_AXES = "x_right_y_down_z_forward"
CAMERA_AXES = (TRAINING_CAMERA_AXES, OPENCV_CAMERA_AXES)

_AXES_TO_TRAINING = {
    TRAINING_CAMERA_AXES: np.eye(4, dtype=np.float64),
    OPENCV_CAMERA_AXES: np.diag([1.0, -1.0, -1.0, 1.0]),
}


def check_pose7(pose: np.ndarray, *, label: str = "pose") -> np.ndarray:
    """Validate a ``[xyz, quat_wxyz]`` pose and return it with a unit quat.

    Catches the failure modes that would otherwise propagate silently: a wrong
    field length, NaN/inf from an unpublished sensor, and a quaternion that is
    not a rotation (all-zero placeholders, or a field that simply is not a
    quaternion). It cannot catch an ``xyzw`` ordering — that stays a
    documented contract.
    """
    pose = np.asarray(pose, dtype=np.float64).reshape(-1)
    if pose.shape != (7,):
        raise ValueError(f"{label} must be 7-d [xyz, quat_wxyz], got shape {pose.shape}")
    if not np.all(np.isfinite(pose)):
        raise ValueError(f"{label} contains non-finite values: {pose.tolist()}")
    norm = float(np.linalg.norm(pose[3:7]))
    if not 0.5 <= norm <= 2.0:
        raise ValueError(
            f"{label} quaternion has norm {norm:.6g}; expected ~1. Check that the "
            "field really is [x, y, z, qw, qx, qy, qz] (wxyz, not xyzw)."
        )
    pose = pose.copy()
    pose[3:7] /= norm
    return pose


def normalize_camera_pose(
    pose: np.ndarray,
    *,
    semantics: str = WORLD_FROM_CAMERA,
    axes: str = TRAINING_CAMERA_AXES,
) -> np.ndarray:
    """Bring published camera extrinsics into the training convention.

    Returns ``T_world_from_camera`` with the camera axes the training pipeline
    assumed (``x_right_y_up_z_back``), so the result can be used directly as
    the frame argument of :func:`relative_pose_b_in_a`.

    Args:
        pose: published extrinsics as ``[xyz, quat_wxyz]``.
        semantics: direction of ``pose`` — ``world_from_camera`` (the pipeline
            default) or ``camera_from_world`` (inverted here).
        axes: axis convention of the camera frame ``pose`` refers to.
    """
    if semantics not in EXTRINSICS_SEMANTICS:
        raise ValueError(
            f"camera extrinsics semantics={semantics!r} must be one of {EXTRINSICS_SEMANTICS}"
        )
    if axes not in _AXES_TO_TRAINING:
        raise ValueError(f"camera axes={axes!r} must be one of {CAMERA_AXES}")

    transform = pose_wxyz_to_matrix(check_pose7(pose, label="camera extrinsics"))
    if semantics == CAMERA_FROM_WORLD:
        transform = invert_rigid(transform)
    # Right-multiply: the correction re-expresses points given in the source
    # camera frame as points in the training camera frame.
    return matrix_to_pose_wxyz(transform @ _AXES_TO_TRAINING[axes])


def _parse_pose(value: str) -> np.ndarray:
    payload = json.loads(value)
    pose = np.asarray(payload, dtype=np.float64)
    if pose.shape != (7,):
        raise argparse.ArgumentTypeError("pose must be a JSON list of 7 numbers")
    return pose


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-a", required=True, type=_parse_pose)
    parser.add_argument("--pose-b", required=True, type=_parse_pose)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = relative_pose_b_in_a(args.pose_a, args.pose_b)
    print(json.dumps(result.tolist()))


if __name__ == "__main__":
    main()
