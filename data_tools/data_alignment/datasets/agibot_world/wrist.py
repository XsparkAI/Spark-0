"""AgiBotWorld-Beta(hand) wrist RGB: undistort → exact 4:3 crop → 640×480.

Supports two hardware variants detected per episode:
- fisheye (960×768, equidistant ``k1..k4``)
- D405 (640×480, plumb bob)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ...readers import egopro_center_crop_to_aspect, scale_intrinsics_for_output_image
from ...storage.writer_common import (
    OPTICAL_FROM_SPARK_CAMERA as _OPENCV_FROM_SPARK,
    OUTPUT_IMAGE_HW,
    _encode_bgr_frame,
)
from ...transforms import matrix_to_pose_wxyz


WristVariant = Literal["fisheye", "d405"]

_DEFAULT_HW: dict[WristVariant, tuple[int, int]] = {
    "fisheye": (768, 960),
    "d405": (480, 640),
}


def detect_wrist_variant(params_dir: Path) -> WristVariant:
    """Prefer fisheye params when present; else D405 pinhole hand_* files."""
    params_dir = Path(params_dir)
    if (params_dir / "hand_left_fisheye_intrinsic_params.json").is_file():
        return "fisheye"
    if (params_dir / "hand_left_intrinsic_params.json").is_file():
        return "d405"
    raise FileNotFoundError(
        f"No wrist intrinsic JSON under {params_dir} "
        "(expected hand_left_fisheye_intrinsic_params.json or "
        "hand_left_intrinsic_params.json)"
    )


def wrist_video_name(side: str, variant: WristVariant) -> str:
    if side not in ("left", "right"):
        raise ValueError(f"side must be left/right, got {side!r}")
    if variant == "fisheye":
        return f"hand_{side}_fisheye_color.mp4"
    return f"hand_{side}_color.mp4"


def wrist_intrinsic_name(side: str, variant: WristVariant) -> str:
    if variant == "fisheye":
        return f"hand_{side}_fisheye_intrinsic_params.json"
    return f"hand_{side}_intrinsic_params.json"


def wrist_extrinsic_aligned_name(side: str, variant: WristVariant) -> str:
    if variant == "fisheye":
        return f"hand_{side}_fisheye_extrinsic_params_aligned.json"
    return f"hand_{side}_extrinsic_params_aligned.json"


def resolve_wrist_paths(
    observations_videos: Path,
    params_dir: Path,
    side: str,
    variant: WristVariant,
) -> tuple[Path, Path, Path] | None:
    """Return ``(video, intrinsic_json, extrinsic_aligned_json)`` or None."""
    video = Path(observations_videos) / wrist_video_name(side, variant)
    intrinsic = Path(params_dir) / wrist_intrinsic_name(side, variant)
    extrinsic = Path(params_dir) / wrist_extrinsic_aligned_name(side, variant)
    if video.is_file() and intrinsic.is_file() and extrinsic.is_file():
        return video, intrinsic, extrinsic
    return None


def _source_hw(params_dir: Path, side: str, variant: WristVariant) -> tuple[int, int]:
    params_dir = Path(params_dir)
    if variant == "fisheye":
        info_path = params_dir / "fisheye_camera_info.json"
        want = f"hand_{side}_fisheye"
        if info_path.is_file():
            info = json.loads(info_path.read_text())
            for entry in info.values():
                if str(entry.get("name", "")) == want:
                    return int(entry["height"]), int(entry["width"])
    else:
        info_path = params_dir / "rs_camera_info.json"
        want = f"hand_{side}"
        if info_path.is_file():
            info = json.loads(info_path.read_text())
            for entry in info.values():
                if str(entry.get("name", "")) == want:
                    return int(entry["height"]), int(entry["width"])
    return _DEFAULT_HW[variant]


def load_wrist_intrinsics_distortion(
    intrinsic_json: Path,
    *,
    variant: WristVariant,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return ``(K 3x3, D, model_name)`` from an AgiBot intrinsic JSON."""
    intr = json.loads(Path(intrinsic_json).read_text())["intrinsic"]
    if variant == "fisheye":
        fu = float(intr["fu"])
        fv = float(intr["fv"])
        pu = float(intr["pu"])
        pv = float(intr["pv"])
        K = np.asarray([[fu, 0.0, pu], [0.0, fv, pv], [0.0, 0.0, 1.0]], dtype=np.float64)
        D = np.asarray(
            [intr["k1"], intr["k2"], intr["k3"], intr["k4"]], dtype=np.float64
        ).reshape(4, 1)
        model = str(intr.get("distortion_model", "equidistant")).replace(" ", "_")
        return K, D, model
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["ppx"])
    cy = float(intr["ppy"])
    K = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    D = np.asarray(
        [intr["k1"], intr["k2"], intr["p1"], intr["p2"], intr["k3"]],
        dtype=np.float64,
    )
    model = str(intr.get("distortion_model", "plumb bob")).replace(" ", "_")
    return K, D, model


def build_undistort_maps(
    K: np.ndarray,
    D: np.ndarray,
    source_hw: tuple[int, int],
    *,
    variant: WristVariant,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(map1, map2, K_new)`` for remap; output size = source size."""
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as error:
        raise ImportError("AgiBot wrist undistort requires cv2") from error

    h, w = int(source_hw[0]), int(source_hw[1])
    size = (w, h)
    if variant == "fisheye":
        # Keep calibrated K as rectified matrix (estimateNewCameraMatrix
        # collapses focal length on these AgiBot equidistant calibrations).
        R = np.eye(3, dtype=np.float64)
        K_new = np.asarray(K, dtype=np.float64).copy()
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, R, K_new, size, cv2.CV_16SC2
        )
        return map1, map2, K_new
    K_new, _roi = cv2.getOptimalNewCameraMatrix(
        K, D, size, alpha=0.0, newImgSize=size
    )
    map1, map2 = cv2.initUndistortRectifyMap(
        K, D, None, K_new, size, cv2.CV_16SC2
    )
    return map1, map2, np.asarray(K_new, dtype=np.float64)


def scaled_intrinsics_640x480(
    K_undistorted: np.ndarray,
    source_hw: tuple[int, int],
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    fx = float(K_undistorted[0, 0])
    fy = float(K_undistorted[1, 1])
    cx = float(K_undistorted[0, 2])
    cy = float(K_undistorted[1, 2])
    return scale_intrinsics_for_output_image(
        np.asarray([fx, fy, cx, cy], dtype=np.float64),
        source_hw=source_hw,
        output_hw=OUTPUT_IMAGE_HW,
        upright_rotate_180=False,
    )


def load_wrist_extrinsics_spark(
    extrinsic_aligned_json: Path,
    n_frames: int,
) -> np.ndarray:
    """Load aligned OpenCV camera_to_world → Spark-axes ``(T,7)`` wxyz."""
    aligned = json.loads(Path(extrinsic_aligned_json).read_text())
    poses = []
    for entry in aligned:
        ext = entry["extrinsic"]
        # AgiBotWorld aligned extrinsics store world_from_camera directly
        # (official projection is p_cam = R^T @ (P_world - t)):
        #   rotation_matrix  = camera→world rotation (NOT world→camera)
        #   translation_vector = camera position in world frame
        R_wc = np.asarray(ext["rotation_matrix"], dtype=np.float64)
        t_cam = np.asarray(ext["translation_vector"], dtype=np.float64)
        T_w_spark = np.eye(4, dtype=np.float64)
        T_w_spark[:3, :3] = R_wc
        T_w_spark[:3, 3] = t_cam  # camera position IS world_from_camera t
        T_w_spark = T_w_spark @ _OPENCV_FROM_SPARK
        poses.append(matrix_to_pose_wxyz(T_w_spark))
    poses_arr = np.asarray(poses, dtype=np.float64)
    if len(poses_arr) == 1:
        poses_arr = np.repeat(poses_arr, n_frames, axis=0)
    elif len(poses_arr) < n_frames:
        raise ValueError(
            f"{extrinsic_aligned_json}: {len(poses_arr)} extrinsics, expected {n_frames}"
        )
    else:
        poses_arr = poses_arr[:n_frames]
    return poses_arr


def _pyav_bgr_frames(video: Path):
    import av  # type: ignore[import-not-found]

    with av.open(str(video)) as container:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="bgr24")


def _cv2_bgr_frames(video: Path):
    import cv2  # type: ignore[import-not-found]

    capture = cv2.VideoCapture(str(video))
    try:
        while capture.isOpened():
            ok, bgr = capture.read()
            if not ok or bgr is None:
                return
            yield bgr
    finally:
        capture.release()


def _ffmpeg_bgr_frames(video: Path):
    import subprocess

    import cv2  # type: ignore[import-not-found]

    command = ["ffmpeg", "-v", "error", "-i", str(video),
               "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "2", "pipe:1"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    buffer = bytearray()
    try:
        while process.stdout is not None:
            chunk = process.stdout.read(1 << 20)
            if not chunk:
                break
            buffer.extend(chunk)
            while True:
                start_idx = buffer.find(b"\xff\xd8")
                if start_idx < 0:
                    if len(buffer) > 2:
                        del buffer[:-2]
                    break
                end_idx = buffer.find(b"\xff\xd9", start_idx + 2)
                if end_idx < 0:
                    if start_idx:
                        del buffer[:start_idx]
                    break
                jpeg = bytes(buffer[start_idx:end_idx + 2])
                del buffer[:end_idx + 2]
                image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is not None:
                    yield image
    finally:
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
        process.wait()


def _iter_bgr_frames(video: Path, n_frames: int):
    """Yield up to n_frames BGR frames, falling back PyAV -> OpenCV -> ffmpeg.

    Every decoder starts at the head of the file, so a fallback skips the
    frames already yielded; restarting would splice the video onto itself.
    """
    yielded = 0

    def resume(frames):
        nonlocal yielded
        for index, bgr in enumerate(frames):
            if yielded >= n_frames:
                return
            if index < yielded:
                continue
            yielded += 1
            yield bgr

    try:
        import av  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        pass
    else:
        try:
            yield from resume(_pyav_bgr_frames(video))
        except Exception:
            pass
    for decode in (_cv2_bgr_frames, _ffmpeg_bgr_frames):
        if yielded >= n_frames:
            return
        yield from resume(decode(video))


def process_wrist_video(
    video: Path,
    map1: np.ndarray,
    map2: np.ndarray,
    n_frames: int,
    *,
    quality: int = 95,
) -> list[bytes]:
    """Decode → undistort → 4:3 crop → 640×480 → Spark image bits.

    Geometry runs on the OpenCV BGR frames; ``_encode_bgr_frame`` converts once
    at encode time.
    """
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as error:
        raise ImportError("AgiBot wrist video requires cv2") from error

    out: list[bytes] = []
    target_h, target_w = OUTPUT_IMAGE_HW
    for i, bgr in enumerate(_iter_bgr_frames(video, n_frames)):
        undistorted = cv2.remap(
            bgr,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        cropped = egopro_center_crop_to_aspect(undistorted, aspect_wh=(4, 3))
        if cropped.shape[:2] != (target_h, target_w):
            cropped = cv2.resize(
                cropped,
                (target_w, target_h),
                interpolation=cv2.INTER_AREA,
            )
        try:
            out.append(_encode_bgr_frame(cropped, quality=quality))
        except ValueError as error:
            raise RuntimeError(
                f"JPEG encode failed at frame {i} for {video}"
            ) from error
    if len(out) != n_frames:
        raise RuntimeError(
            f"{video}: decoded {len(out)} frames after undistort, expected {n_frames}"
        )
    return out


def pack_one_wrist(
    *,
    videos_dir: Path,
    params_dir: Path,
    side: str,
    variant: WristVariant,
    n_frames: int,
    quality: int = 90,
) -> dict[str, Any] | None:
    """Build one wrist stream payload, or None if files are incomplete."""
    resolved = resolve_wrist_paths(videos_dir, params_dir, side, variant)
    if resolved is None:
        return None
    video, intrinsic_json, extrinsic_json = resolved
    source_hw = _source_hw(params_dir, side, variant)
    K, D, model = load_wrist_intrinsics_distortion(intrinsic_json, variant=variant)
    map1, map2, K_new = build_undistort_maps(K, D, source_hw, variant=variant)
    jpegs = process_wrist_video(video, map1, map2, n_frames, quality=quality)
    intrinsics, crop_meta = scaled_intrinsics_640x480(K_new, source_hw)
    extrinsics = load_wrist_extrinsics_spark(extrinsic_json, n_frames)
    return {
        "jpegs": jpegs,
        "intrinsics": intrinsics,
        "extrinsics": extrinsics,
        "shape": np.asarray([*OUTPUT_IMAGE_HW, 3], dtype=np.int64),
        "source_topic": video.name,
        "undistort_model": model,
        "variant": variant,
        "source_image_width": int(source_hw[1]),
        "source_image_height": int(source_hw[0]),
        "crop_meta": crop_meta,
    }
