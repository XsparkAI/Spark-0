"""Official EgoVLA_Release ``main_camera`` → Spark ``cam_head`` slots.

Raw HDF5 has no K/E. EgoVLA_Release treats ``images/main`` as the world-fixed
Isaac ``main_camera`` (1280×720, ``focal_length=8``,
``horizontal_aperture=20.955``, OpenGL offset ``(0.09, 0, 1.7)`` +
``(0.66446, 0.24184, -0.24184, -0.664464)``). Published RGB is already
384×384; default ``stretch`` scales that 1280×720 K independently on x/y
(same as ``cv2.resize`` to 384).

Spark stores ``camera_to_world`` (x right, y up, z back). The rotation is
chosen so EgoVLA_Release ``CAM_AXIS_TRANSFORM @ inv(T)`` matches Spark viz
(invert then ``diag(1,-1,-1)``). Translation stays the Isaac world position.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...transforms import matrix_to_quaternion_wxyz, quaternion_wxyz_to_matrix

CAMERA_SOURCE = "egovla_release_main_camera_official"
CAMERA_STATUS = "official"
K_MODE = "stretch"
SOURCE_RESOLUTION = "1280x720"

ISAAC_MAIN_CAM_XYZ = np.array([0.09, 0.0, 1.7], dtype=np.float64)
# OpenGL spawn quat; EgoVLA_Release CHANGE @ R_isaac == EGOVLA_GT_CAM_WXYZ.
ISAAC_MAIN_CAM_WXYZ = np.array(
    [0.66446, 0.24184, -0.24184, -0.664464], dtype=np.float64
)
EGOVLA_GT_CAM_WXYZ = np.array(
    [0.9063077870366499, 0.0, 0.42261826174069944, 0.0], dtype=np.float64
)
CAM_AXIS_TRANSFORM = np.array(
    [
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)
GRAPHICS_TO_OPENCV = np.diag([1.0, -1.0, -1.0])
K_1280_720 = np.array([488.6662, 488.6662, 640.0, 360.0], dtype=np.float64)
SRC_WH = (1280, 720)
DST_WH = (384, 384)


def official_head_intrinsics(*, k_mode: str = K_MODE) -> np.ndarray:
    fx, fy, cx, cy = K_1280_720
    src_w, src_h = SRC_WH
    dst_w, dst_h = DST_WH
    if k_mode == "stretch":
        return np.array(
            [fx * dst_w / src_w, fy * dst_h / src_h, cx * dst_w / src_w, cy * dst_h / src_h],
            dtype=np.float64,
        )
    if k_mode == "letterbox":
        scale = dst_w / src_w
        return np.array(
            [
                fx * scale,
                fy * scale,
                cx * scale,
                cy * scale + 0.5 * (dst_h - src_h * scale),
            ],
            dtype=np.float64,
        )
    raise ValueError(f"unknown EgoVLA k-mode {k_mode!r}")


def official_head_pose() -> np.ndarray:
    isaac_rot = quaternion_wxyz_to_matrix(ISAAC_MAIN_CAM_WXYZ)
    rotation_t = quaternion_wxyz_to_matrix(EGOVLA_GT_CAM_WXYZ)
    # CHANGE @ R_isaac == R_gt; keep the GT form EgoVLA_Release actually uses.
    if not np.allclose(
        (rotation_t @ np.linalg.inv(isaac_rot)) @ isaac_rot, rotation_t, atol=1e-6
    ):
        raise RuntimeError("EgoVLA official camera frame-change identity failed")
    rotation_spark = (
        rotation_t @ np.linalg.inv(CAM_AXIS_TRANSFORM) @ GRAPHICS_TO_OPENCV
    )
    pose = np.empty(7, dtype=np.float64)
    pose[:3] = ISAAC_MAIN_CAM_XYZ
    quat = matrix_to_quaternion_wxyz(rotation_spark)
    if quat[0] < 0.0:
        quat = -quat
    pose[3:] = quat
    return pose


def official_head_poses(keep: int) -> tuple[np.ndarray, np.ndarray]:
    if keep < 1:
        raise ValueError(f"keep must be >= 1, got {keep}")
    poses = np.tile(official_head_pose(), (keep, 1))
    valid = np.ones(keep, dtype=bool)
    return poses, valid


def write_official_intrinsic_attrs(dataset: Any, *, k_mode: str = K_MODE) -> None:
    dataset.attrs["order"] = "fx,fy,cx,cy"
    dataset.attrs["status"] = CAMERA_STATUS
    dataset.attrs["source"] = CAMERA_SOURCE
    dataset.attrs["semantics"] = "pinhole_fx_fy_cx_cy_for_cam_head_rgb"
    dataset.attrs["image_width"] = DST_WH[0]
    dataset.attrs["image_height"] = DST_WH[1]
    dataset.attrs["k_mode"] = k_mode
    dataset.attrs["source_resolution"] = SOURCE_RESOLUTION


def write_official_extrinsic_attrs(dataset: Any, *, constant: bool) -> None:
    dataset.attrs["order"] = "x,y,z,qw,qx,qy,qz"
    dataset.attrs["frame"] = "camera_to_world"
    dataset.attrs["camera_axes"] = "x_right_y_up_z_back"
    dataset.attrs["status"] = CAMERA_STATUS
    dataset.attrs["source"] = CAMERA_SOURCE
    dataset.attrs["constant"] = bool(constant)
