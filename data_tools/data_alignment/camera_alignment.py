"""Coordinate transforms, temporal resampling, and Spark camera alignment."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from .schema import ReferenceFrameGeometry, UnifiedHumanMotion


def normalize(vector: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if norm < eps:
        raise ValueError("Cannot normalize a near-zero vector")
    return vector / norm


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    """Return ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = normalize(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


def rotvec_to_wxyz(rotvec: np.ndarray) -> np.ndarray:
    """Axis-angle rotation vector -> normalized ``wxyz`` quaternion."""
    rotvec = np.asarray(rotvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return matrix_to_quaternion_wxyz(axis_angle_matrix(rotvec / angle, angle)).reshape(4)


def wxyz_to_rotvec(wxyz: np.ndarray) -> np.ndarray:
    """Normalized ``wxyz`` quaternion -> axis-angle rotation vector."""
    wxyz = np.asarray(wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(wxyz))
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float64)
    wxyz = wxyz / norm
    if wxyz[0] < 0:
        wxyz = -wxyz
    w, x, y, z = wxyz
    sine = math.sqrt(max(1.0 - w * w, 0.0))
    if sine < 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.acos(float(np.clip(w, -1.0, 1.0)))
    return (angle / sine) * np.array([x, y, z], dtype=np.float64)


def quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    matrix = np.empty(quaternion.shape[:-1] + (3, 3), dtype=np.float64)
    matrix[..., 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[..., 0, 1] = 2 * (x * y - z * w)
    matrix[..., 0, 2] = 2 * (x * z + y * w)
    matrix[..., 1, 0] = 2 * (x * y + z * w)
    matrix[..., 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[..., 1, 2] = 2 * (y * z - x * w)
    matrix[..., 2, 0] = 2 * (x * z - y * w)
    matrix[..., 2, 1] = 2 * (y * z + x * w)
    matrix[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return matrix


def matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert one or more proper rotation matrices to normalized wxyz."""
    matrices = np.asarray(matrix, dtype=np.float64)
    flat = matrices.reshape(-1, 3, 3)
    result = np.empty((len(flat), 4), dtype=np.float64)
    for index, rotation in enumerate(flat):
        trace = np.trace(rotation)
        if trace > 0:
            scale = math.sqrt(trace + 1.0) * 2
            result[index] = [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        else:
            diagonal = np.diag(rotation)
            axis = int(np.argmax(diagonal))
            if axis == 0:
                scale = math.sqrt(1 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
                result[index] = [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ]
            elif axis == 1:
                scale = math.sqrt(1 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
                result[index] = [
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ]
            else:
                scale = math.sqrt(1 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
                result[index] = [
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                ]
    result /= np.linalg.norm(result, axis=-1, keepdims=True)
    return result.reshape(matrices.shape[:-2] + (4,))


def pose_wxyz_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    result = np.zeros(pose.shape[:-1] + (4, 4), dtype=np.float64)
    result[..., :3, :3] = quaternion_wxyz_to_matrix(pose[..., 3:7])
    result[..., :3, 3] = pose[..., :3]
    result[..., 3, 3] = 1
    return result


def matrix_to_pose_wxyz(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    return np.concatenate(
        [transform[..., :3, 3], matrix_to_quaternion_wxyz(transform[..., :3, :3])],
        axis=-1,
    )


def ensure_quaternion_continuity(poses: np.ndarray) -> np.ndarray:
    """Flip equivalent quaternion signs so adjacent dot products are positive."""
    result = np.array(poses, dtype=np.float64, copy=True)
    for index in range(1, len(result)):
        if np.dot(result[index - 1, 3:7], result[index, 3:7]) < 0:
            result[index, 3:7] *= -1
    return result


def canonicalize_hand_points(
    points: np.ndarray,
    wrist_position: np.ndarray,
    confidence: np.ndarray,
    knuckle_index: int = 0,
) -> tuple[np.ndarray, bool]:
    """Express a 5x5 hand chain in a stable anatomical palm frame.

    The x-axis points from the pinky knuckle to the index knuckle, y points
    from the wrist toward the middle knuckle, and z completes a right-handed
    frame. ``knuckle_index`` selects the chain point holding the anatomical
    knuckle (MCP): 0 for the Wuji URDF chains whose roots are the joint1
    origins, 1 for EgoPro/EgoDex chains whose point 0 is the metacarpal base.
    Using the metacarpal-base points for a human hand yields a frame rotated
    ~75 deg away from the Wuji canonical frame and breaks retargeting.
    """
    points = np.asarray(points, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    if points.shape != (5, 5, 3) or confidence.shape != (5, 5):
        raise ValueError("Expected points [5,5,3] and confidence [5,5]")
    required = (
        confidence[1, knuckle_index] > 0
        and confidence[2, knuckle_index] > 0
        and confidence[4, knuckle_index] > 0
        and np.all(np.isfinite(points[[1, 2, 4], knuckle_index]))
    )
    if not required or not np.all(np.isfinite(wrist_position)):
        return np.full_like(points, np.nan), False
    try:
        x_axis = normalize(points[1, knuckle_index] - points[4, knuckle_index])
        y_hint = normalize(points[2, knuckle_index] - wrist_position)
        z_axis = normalize(np.cross(x_axis, y_hint))
        y_axis = normalize(np.cross(z_axis, x_axis))
    except ValueError:
        return np.full_like(points, np.nan), False
    world_from_palm = np.stack([x_axis, y_axis, z_axis], axis=1)
    local = (points - wrist_position) @ world_from_palm
    local[confidence <= 0] = np.nan
    return local, True


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert one or more rigid 4×4 transforms."""
    transform = np.asarray(transform, dtype=np.float64)
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3]
    result = np.zeros_like(transform)
    result[..., :3, :3] = np.swapaxes(rotation, -1, -2)
    result[..., :3, 3] = -(result[..., :3, :3] @ translation[..., None])[..., 0]
    result[..., 3, 3] = 1.0
    return result


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a rigid transform to points with a trailing xyz axis."""
    transform = np.asarray(transform, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    return points @ transform[..., :3, :3].swapaxes(-1, -2) + transform[..., :3, 3]


def optical_upright_from_inverted(transform: np.ndarray) -> np.ndarray:
    """Compose a 180° optical-axis correction matching ``cv2.ROTATE_180``."""
    transform = np.asarray(transform, dtype=np.float64)
    upright = np.diag([-1.0, -1.0, 1.0, 1.0])
    return upright @ transform


def rpy_deg_matrix(rpy_deg: np.ndarray) -> np.ndarray:
    return rpy_matrix(np.deg2rad(np.asarray(rpy_deg, dtype=np.float64)))


def slerp_wxyz(q0: np.ndarray, q1: np.ndarray, amount: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 /= np.linalg.norm(q0)
    q1 /= np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1, 1))
    if dot > 0.9995:
        result = q0 + amount * (q1 - q0)
        return result / np.linalg.norm(result)
    theta = math.acos(dot)
    return (
        math.sin((1 - amount) * theta) / math.sin(theta) * q0
        + math.sin(amount * theta) / math.sin(theta) * q1
    )


def _target_timestamps(source: np.ndarray, fps: float) -> np.ndarray:
    if fps <= 0:
        raise ValueError("fps must be positive")
    if len(source) == 0:
        return np.empty(0, dtype=np.float64)
    count = int(np.floor((source[-1] - source[0]) * fps + 1e-9)) + 1
    return source[0] + np.arange(count, dtype=np.float64) / fps


def _bracket(source: np.ndarray, timestamp: float) -> tuple[int, int, float]:
    right = int(np.searchsorted(source, timestamp, side="left"))
    if right == 0:
        return 0, 0, 0.0
    if right == len(source):
        return len(source) - 1, len(source) - 1, 0.0
    if source[right] == timestamp:
        return right, right, 0.0
    left = right - 1
    amount = float((timestamp - source[left]) / (source[right] - source[left]))
    return left, right, amount


def _ema_valid(values: np.ndarray, valid: np.ndarray, alpha: float) -> np.ndarray:
    result = np.array(values, copy=True)
    if not 0 < alpha <= 1:
        raise ValueError("lowpass_alpha must lie in (0, 1]")
    for index in range(1, len(result)):
        mask = valid[index] & valid[index - 1]
        blended = alpha * result[index] + (1 - alpha) * result[index - 1]
        result[index] = np.where(mask, blended, result[index])
    return result


def resample_reference_geometry(
    geometry: ReferenceFrameGeometry,
    fps: float = 25.0,
    max_gap_s: float = 0.1,
    lowpass_alpha: float = 0.35,
) -> ReferenceFrameGeometry:
    """Resample reference wrists, chains, and camera poses before alignment."""
    target_time = _target_timestamps(geometry.timestamps, fps)
    wrist_source = np.array(geometry.wrist_poses_ref, copy=True)
    joints_source = _ema_valid(
        geometry.joints_ref,
        geometry.joint_confidence[..., None] > 0,
        lowpass_alpha,
    )
    local_source = _ema_valid(
        geometry.joints_local,
        geometry.joint_confidence[..., None] > 0,
        lowpass_alpha,
    )
    for hand in range(2):
        valid_indices = np.flatnonzero(geometry.wrist_valid[:, hand])
        if len(valid_indices):
            wrist_source[valid_indices, hand] = ensure_quaternion_continuity(
                wrist_source[valid_indices, hand]
            )
            wrist_source[:, hand, :3] = _ema_valid(
                wrist_source[:, hand, :3],
                geometry.wrist_valid[:, hand, None],
                lowpass_alpha,
            )

    camera_pose = np.full((len(geometry.timestamps), 7), np.nan)
    camera_reflection = np.eye(3, dtype=np.float64)
    valid_cam = np.flatnonzero(geometry.camera_valid)
    if len(valid_cam):
        determinants = np.linalg.det(
            geometry.camera_from_world[valid_cam, :3, :3]
        )
        reflected = determinants < 0
        if np.any(reflected) and not np.all(reflected):
            raise ValueError("Camera handedness changes within one trajectory")
        if np.all(reflected):
            # OpenCV x-right/y-down/z-forward relative to a Y-up source is an
            # improper frame. Factor out the fixed Y reflection so only a
            # proper SO(3) rotation is converted to quaternion / SLERP.
            camera_reflection = np.diag([1.0, -1.0, 1.0])
    for index in range(len(geometry.timestamps)):
        if geometry.camera_valid[index]:
            world_from_camera = invert_transform(geometry.camera_from_world[index])
            world_from_camera[:3, :3] = (
                world_from_camera[:3, :3] @ camera_reflection
            )
            camera_pose[index] = matrix_to_pose_wxyz(world_from_camera)
    if len(valid_cam):
        camera_pose[valid_cam] = ensure_quaternion_continuity(camera_pose[valid_cam])
        camera_pose[:, :3] = _ema_valid(
            camera_pose[:, :3],
            geometry.camera_valid[:, None],
            lowpass_alpha,
        )

    wrist = np.full((len(target_time), 2, 7), np.nan)
    wrist_valid = np.zeros((len(target_time), 2), dtype=bool)
    joints = np.full((len(target_time), 2, 5, 5, 3), np.nan)
    joints_local = np.full((len(target_time), 2, 5, 5, 3), np.nan)
    confidence = np.zeros((len(target_time), 2, 5, 5))
    camera_from_world = np.full((len(target_time), 4, 4), np.nan)
    camera_valid = np.zeros(len(target_time), dtype=bool)

    for output_index, timestamp in enumerate(target_time):
        left, right, amount = _bracket(geometry.timestamps, timestamp)
        gap_ok = geometry.timestamps[right] - geometry.timestamps[left] <= max_gap_s
        if gap_ok and geometry.camera_valid[left] and geometry.camera_valid[right]:
            position = (1 - amount) * camera_pose[left, :3] + amount * camera_pose[right, :3]
            orientation = slerp_wxyz(camera_pose[left, 3:7], camera_pose[right, 3:7], amount)
            world_from_camera = pose_wxyz_to_matrix(
                np.concatenate([position, orientation])
            )
            world_from_camera[:3, :3] = (
                world_from_camera[:3, :3] @ camera_reflection
            )
            camera_from_world[output_index] = invert_transform(world_from_camera)
            camera_valid[output_index] = True
        for hand in range(2):
            if gap_ok and geometry.wrist_valid[left, hand] and geometry.wrist_valid[right, hand]:
                wrist[output_index, hand, :3] = (
                    (1 - amount) * wrist_source[left, hand, :3]
                    + amount * wrist_source[right, hand, :3]
                )
                wrist[output_index, hand, 3:7] = slerp_wxyz(
                    wrist_source[left, hand, 3:7],
                    wrist_source[right, hand, 3:7],
                    amount,
                )
                wrist_valid[output_index, hand] = True
            point_valid = (
                gap_ok
                & (geometry.joint_confidence[left, hand] > 0)
                & (geometry.joint_confidence[right, hand] > 0)
            )
            interpolated = (
                (1 - amount) * joints_source[left, hand] + amount * joints_source[right, hand]
            )
            joints[output_index, hand][point_valid] = interpolated[point_valid]
            local_interp = (
                (1 - amount) * local_source[left, hand] + amount * local_source[right, hand]
            )
            joints_local[output_index, hand][point_valid] = local_interp[point_valid]
            confidence[output_index, hand][point_valid] = np.minimum(
                geometry.joint_confidence[left, hand][point_valid],
                geometry.joint_confidence[right, hand][point_valid],
            )

    return ReferenceFrameGeometry(
        timestamps=target_time,
        wrist_poses_ref=wrist,
        wrist_valid=wrist_valid,
        joints_ref=joints,
        joint_confidence=confidence,
        joints_local=joints_local,
        camera_from_world=camera_from_world,
        camera_valid=camera_valid,
        intrinsics=geometry.intrinsics,
        source=geometry.source,
        instruction=geometry.instruction,
        metadata={
            **geometry.metadata,
            "resampled_fps": fps,
            "lowpass_alpha": lowpass_alpha,
            "max_interpolation_gap_s": max_gap_s,
            "source_readonly": True,
        },
    )


@dataclass(frozen=True)
class SparkCameraConfig:
    """Orientation of the Spark camera mount; the mount sits at the world origin."""

    rotation_rpy_deg: np.ndarray
    mount_from_optical: np.ndarray
    initial_window_s: float = 0.5

    def __post_init__(self) -> None:
        rpy = np.asarray(self.rotation_rpy_deg, dtype=np.float64)
        if rpy.shape != (3,):
            raise ValueError("Spark camera RPY must be 3-D")
        if not np.all(np.isfinite(rpy)):
            raise ValueError("Spark camera RPY must be finite")
        if self.initial_window_s <= 0:
            raise ValueError("initial_window_s must be positive")
        object.__setattr__(self, "rotation_rpy_deg", rpy)
        axes = np.asarray(self.mount_from_optical, dtype=np.float64)
        _validate_rotation(axes, "mount_from_optical")
        object.__setattr__(self, "mount_from_optical", axes)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SparkCameraConfig":
        camera = payload.get("camera", payload)
        return cls(
            rotation_rpy_deg=camera["rotation_rpy_deg"],
            initial_window_s=float(camera.get("initial_window_s", 0.5)),
            mount_from_optical=camera["mount_from_optical"],
        )

    def world_from_mount(self) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rpy_deg_matrix(self.rotation_rpy_deg)
        return transform


@dataclass(frozen=True)
class SparkWorkspaceConfig:
    """Fixed EE orientation anchors."""

    target_ee_orientation_wxyz: np.ndarray

    def __post_init__(self) -> None:
        target = np.asarray(self.target_ee_orientation_wxyz, dtype=np.float64)
        if target.shape != (2, 4) or not np.all(np.isfinite(target)):
            raise ValueError("workspace target EE orientations must be finite [2,4]")
        norms = np.linalg.norm(target, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-6):
            raise ValueError("workspace target EE quaternions must be normalized")
        object.__setattr__(self, "target_ee_orientation_wxyz", target)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SparkWorkspaceConfig":
        workspace = payload["workspace"]
        orientations = workspace["target_ee_orientation_wxyz"]
        return cls(
            target_ee_orientation_wxyz=np.asarray(
                [orientations["left"], orientations["right"]],
                dtype=np.float64,
            ),
        )


@dataclass(frozen=True)
class CameraAlignmentResult:
    """Camera-aligned human motion whose wrist poses are EE poses."""

    motion: UnifiedHumanMotion
    world_from_spark_optical: np.ndarray
    world_from_virtual_ego_optical: np.ndarray
    camera_from_world: np.ndarray
    camera_valid: np.ndarray
    intrinsics: np.ndarray
    mount_from_optical: np.ndarray
    positive_depth_fraction: float
    palm_to_ee_wxyz: np.ndarray
    aligned_joints_world: np.ndarray


def slice_camera_alignment(
    alignment: CameraAlignmentResult,
    start: int,
    end: int,
) -> CameraAlignmentResult:
    """Slice time-varying alignment fields to ``[start, end)``."""
    length = len(alignment.motion.timestamps)
    if start < 0 or end > length or start >= end:
        raise ValueError(f"Invalid slice [{start}, {end}) for length {length}")
    index = slice(start, end)
    motion = alignment.motion
    sliced_motion = UnifiedHumanMotion(
        timestamps=motion.timestamps[index],
        wrist_poses=motion.wrist_poses[index],
        wrist_valid=motion.wrist_valid[index],
        joints_local=motion.joints_local[index],
        joint_confidence=motion.joint_confidence[index],
        source=motion.source,
        instruction=motion.instruction,
        metadata=dict(motion.metadata),
    )
    return CameraAlignmentResult(
        motion=sliced_motion,
        world_from_spark_optical=alignment.world_from_spark_optical,
        world_from_virtual_ego_optical=alignment.world_from_virtual_ego_optical,
        camera_from_world=alignment.camera_from_world[index],
        camera_valid=alignment.camera_valid[index],
        intrinsics=alignment.intrinsics,
        mount_from_optical=alignment.mount_from_optical,
        positive_depth_fraction=alignment.positive_depth_fraction,
        palm_to_ee_wxyz=alignment.palm_to_ee_wxyz,
        aligned_joints_world=alignment.aligned_joints_world[index],
    )


def _validate_rotation(rotation: np.ndarray, name: str) -> None:
    if rotation.shape != (3, 3):
        raise ValueError(f"{name} must be 3x3")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7):
        raise ValueError(f"{name} must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7):
        raise ValueError(f"{name} must be right-handed")


def _initial_common_indices(
    timestamps: np.ndarray,
    valid: np.ndarray,
    window_s: float,
) -> np.ndarray:
    common = np.flatnonzero(np.all(valid, axis=1))
    if not len(common):
        raise ValueError("No frames have both left and right hands valid")
    start = timestamps[common[0]]
    indices = common[timestamps[common] <= start + window_s + 1e-9]
    if not len(indices):
        raise ValueError("Initial valid-hand window is empty")
    return indices


def _mean_rotation(rotations: np.ndarray) -> np.ndarray:
    if not len(rotations):
        raise ValueError("Cannot average an empty rotation sequence")
    u, _singular, vh = np.linalg.svd(np.sum(rotations, axis=0))
    result = u @ vh
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1
        result = u @ vh
    return result


def _mean_reference_camera_from_world(camera_from_world: np.ndarray) -> np.ndarray:
    """Return one fixed initial camera pose for world-to-world axis registration.

    EgoPro camera transforms contain a handedness reflection, so averaging them
    as SO(3) rotations would silently remove that reflection. Average the
    inverse camera poses while preserving their common determinant, then invert
    the result back to ``reference_camera_from_world``.
    """
    transforms = np.asarray(camera_from_world, dtype=np.float64)
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4) or not len(transforms):
        raise ValueError("camera_from_world must be a non-empty [time,4,4] array")
    world_from_cameras = np.linalg.inv(transforms)
    rotations = world_from_cameras[:, :3, :3]
    determinants = np.linalg.det(rotations)
    target_sign = float(np.sign(determinants[0]))
    if target_sign == 0 or not np.all(np.sign(determinants) == target_sign):
        raise ValueError("Initial camera transforms must share one handedness")
    u, _singular, vh = np.linalg.svd(np.sum(rotations, axis=0))
    mean_rotation = u @ vh
    if np.sign(np.linalg.det(mean_rotation)) != target_sign:
        u[:, -1] *= -1
        mean_rotation = u @ vh
    world_from_camera = np.eye(4, dtype=np.float64)
    world_from_camera[:3, :3] = mean_rotation
    world_from_camera[:3, 3] = np.mean(world_from_cameras[:, :3, 3], axis=0)
    return np.linalg.inv(world_from_camera)


def _palm_rotation_ref(
    points: np.ndarray,
    wrist_position: np.ndarray,
    confidence: np.ndarray,
) -> np.ndarray | None:
    required = (1, 2, 4)
    if not all(confidence[finger, 0] > 0 for finger in required):
        return None
    roots = points[list(required), 0]
    if not np.all(np.isfinite(roots)) or not np.all(np.isfinite(wrist_position)):
        return None
    try:
        x_axis = normalize(points[1, 0] - points[4, 0])
        y_hint = normalize(points[2, 0] - wrist_position)
        z_axis = normalize(np.cross(x_axis, y_hint))
        y_axis = normalize(np.cross(z_axis, x_axis))
    except ValueError:
        return None
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def align_reference_to_spark_camera(
    geometry: ReferenceFrameGeometry,
    config: SparkCameraConfig,
    workspace: SparkWorkspaceConfig,
) -> CameraAlignmentResult:
    """Register source world axes once onto the Spark camera mount.

    The initial valid camera window defines one fixed source-world-to-Spark-world
    transform that places that camera at the origin with the configured mount
    orientation. Later head-camera motion is deliberately excluded, so a wrist
    stationary in the source world remains stationary in the output. No
    translation is applied. Each hand receives one fixed orientation offset so
    its initial palm matches ``target_ee_orientation_wxyz``.
    """
    world_from_mount = config.world_from_mount()
    mount_from_optical = config.mount_from_optical

    world_from_spark_optical = world_from_mount.copy()
    world_from_spark_optical[:3, :3] = (
        world_from_mount[:3, :3] @ mount_from_optical
    )

    length = len(geometry.timestamps)
    camera_reference_valid = (
        geometry.wrist_valid & geometry.camera_valid[:, None]
    )
    camera_reference_indices = _initial_common_indices(
        geometry.timestamps,
        camera_reference_valid,
        config.initial_window_s,
    )
    reference_camera_from_world = _mean_reference_camera_from_world(
        geometry.camera_from_world[camera_reference_indices]
    )
    # This transform is constant for the full trajectory. Using
    # geometry.camera_from_world[frame] here would inject head motion into EE.
    world_from_reference = world_from_spark_optical @ reference_camera_from_world

    mapped_positions = np.full((length, 2, 3), np.nan, dtype=np.float64)
    mapped_palm_rotations = np.full((length, 2, 3, 3), np.nan, dtype=np.float64)
    mapped_joints_world = np.full_like(geometry.joints_ref, np.nan)
    mapped_joints_local = np.full_like(geometry.joints_local, np.nan)
    mapped_confidence = np.zeros_like(geometry.joint_confidence)
    aligned_valid = np.zeros((length, 2), dtype=bool)
    for frame in range(length):
        for hand in range(2):
            if not geometry.wrist_valid[frame, hand]:
                continue
            reference_from_wrist = pose_wxyz_to_matrix(
                geometry.wrist_poses_ref[frame, hand]
            )
            world_from_wrist = world_from_reference @ reference_from_wrist
            mapped_positions[frame, hand] = world_from_wrist[:3, 3]
            point_valid = (
                geometry.joint_confidence[frame, hand] > 0
            ) & np.isfinite(geometry.joints_ref[frame, hand]).all(axis=-1)
            mapped_joints_world[frame, hand][point_valid] = transform_points(
                world_from_reference,
                geometry.joints_ref[frame, hand][point_valid],
            )
            # The EgoPro optical convention contains a handedness reflection.
            # Rebuild a proper right-handed palm frame after transforming the
            # points instead of multiplying an SO(3) palm by an improper matrix.
            palm_world = _palm_rotation_ref(
                mapped_joints_world[frame, hand],
                mapped_positions[frame, hand],
                geometry.joint_confidence[frame, hand],
            )
            if palm_world is None:
                mapped_positions[frame, hand] = np.nan
                mapped_joints_world[frame, hand] = np.nan
                continue
            mapped_palm_rotations[frame, hand] = palm_world
            # Hand articulation stays in the source right-handed anatomical
            # palm frame. Re-canonicalizing after the camera handedness
            # reflection flips palm-normal coordinates and changes the pose.
            mapped_joints_local[frame, hand] = geometry.joints_local[frame, hand]
            mapped_confidence[frame, hand] = geometry.joint_confidence[frame, hand]
            aligned_valid[frame, hand] = True

    if not np.any(aligned_valid):
        raise ValueError("No valid EgoPro hands after camera-axis alignment")
    ego_initial = _initial_common_indices(
        geometry.timestamps, aligned_valid, config.initial_window_s
    )

    offsets = np.empty((2, 3, 3), dtype=np.float64)
    target_rotations = quaternion_wxyz_to_matrix(
        workspace.target_ee_orientation_wxyz
    )
    for hand in range(2):
        source_mean = _mean_rotation(mapped_palm_rotations[ego_initial, hand])
        offsets[hand] = source_mean.T @ target_rotations[hand]

    ee_poses = np.full((length, 2, 7), np.nan, dtype=np.float64)
    ee_poses[..., :3] = mapped_positions
    for hand in range(2):
        valid_indices = np.flatnonzero(aligned_valid[:, hand])
        if not len(valid_indices):
            continue
        ee_poses[valid_indices, hand, 3:7] = matrix_to_quaternion_wxyz(
            mapped_palm_rotations[valid_indices, hand] @ offsets[hand]
        )
        ee_poses[valid_indices, hand] = ensure_quaternion_continuity(
            ee_poses[valid_indices, hand]
        )

    virtual_camera = world_from_spark_optical.copy()
    reference_from_aligned_world = np.linalg.inv(world_from_reference)
    dynamic_camera_matrices = (
        geometry.camera_from_world @ reference_from_aligned_world
    )
    dynamic_camera_valid = (
        geometry.camera_valid
        & np.all(np.isfinite(dynamic_camera_matrices), axis=(1, 2))
        & (np.abs(np.linalg.det(dynamic_camera_matrices[:, :3, :3]) - 1.0) < 1e-6)
    )
    dynamic_camera_poses = np.full((length, 7), np.nan, dtype=np.float64)
    valid_camera_indices = np.flatnonzero(dynamic_camera_valid)
    if len(valid_camera_indices):
        dynamic_camera_poses[valid_camera_indices] = matrix_to_pose_wxyz(
            dynamic_camera_matrices[valid_camera_indices]
        )
        dynamic_camera_poses[valid_camera_indices] = ensure_quaternion_continuity(
            dynamic_camera_poses[valid_camera_indices]
        )
    valid_ee_positions = ee_poses[..., :3][aligned_valid]
    spark_optical_from_world = np.linalg.inv(world_from_spark_optical)
    aligned_optical_positions = transform_points(
        spark_optical_from_world,
        valid_ee_positions,
    )
    aligned_positive_depth_fraction = float(
        np.mean(aligned_optical_positions[:, 2] > 0.03)
    )
    palm_to_ee_wxyz = matrix_to_quaternion_wxyz(offsets)
    motion = UnifiedHumanMotion(
        timestamps=geometry.timestamps,
        wrist_poses=ee_poses,
        wrist_valid=aligned_valid,
        joints_local=mapped_joints_local,
        joint_confidence=mapped_confidence,
        source=geometry.source,
        instruction=geometry.instruction,
        metadata={
            **{
                key: value
                for key, value in geometry.metadata.items()
                if key != "coordinate_frame"
            },
            "camera_alignment": {
                "method": "fixed_initial_camera_axes_at_spark_mount",
                "initial_window_s": config.initial_window_s,
                "ego_initial_indices": ego_initial.tolist(),
                "camera_reference_indices": camera_reference_indices.tolist(),
                "reference_camera_from_world": reference_camera_from_world.tolist(),
                "source_world_mapping": "fixed_full_trajectory",
                "head_motion_in_ee": False,
                "positive_depth_fraction": aligned_positive_depth_fraction,
                "mount_from_optical": mount_from_optical.tolist(),
                "world_from_spark_optical": world_from_spark_optical.tolist(),
                "world_from_virtual_ego_optical": virtual_camera.tolist(),
                "palm_to_ee_wxyz": palm_to_ee_wxyz.tolist(),
                "target_ee_orientation_wxyz": (
                    workspace.target_ee_orientation_wxyz.tolist()
                ),
                "target_ee_orientation_source": "configured_frame_zero_statistics",
                "ik_geometry_frame": "source_right_handed_anatomical_palm",
                "aligned_joints_world_role": "camera_alignment_qc_not_ik_target",
            },
        },
    )
    return CameraAlignmentResult(
        motion=motion,
        world_from_spark_optical=world_from_spark_optical,
        world_from_virtual_ego_optical=virtual_camera,
        camera_from_world=dynamic_camera_poses,
        camera_valid=dynamic_camera_valid,
        intrinsics=geometry.intrinsics,
        mount_from_optical=mount_from_optical,
        positive_depth_fraction=aligned_positive_depth_fraction,
        palm_to_ee_wxyz=palm_to_ee_wxyz,
        aligned_joints_world=mapped_joints_world,
    )
