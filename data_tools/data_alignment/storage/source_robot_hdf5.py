"""Write Fourier ActionNet and AgiBot World source-robot episodes as Spark HDF5."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from ..readers import (
    egopro_center_crop_to_aspect,
    scale_intrinsics_for_output_image,
)
from ..datasets.fourier_actionnet.reader import fourier_rgb_path, is_fourier_episode
from ..schema import (
    TACTILE_PATCH_KEYS,
    UnifiedHumanMotion,
    UnifiedRobotMotion,
    empty_tactile,
    make_future_state_action,
)
from .writer_common import (
    IMAGE_BITS_COLOR_ORDER,
    OPTICAL_FROM_SPARK_CAMERA,
    OUTPUT_IMAGE_HW,
    _encode_bgr_frame,
    _portable_metadata,
    _write_mano_groups,
)
from ..transforms import (
    matrix_to_pose_wxyz,
    pose_wxyz_to_matrix,
)

# EgoPro wrist streams are already 4:3 (960x1280); head needs upright + crop.
EGOPRO_CAMERA_STREAMS: tuple[dict[str, Any], ...] = (
    {
        "camera_name": "cam_head",
        "topic": "/ego_video/head",
        "rotate_upright": "from_calibration",
        "center_crop_aspect": (4, 3),
        "resize_attr": "1920x1080_center_crop_1440x1080_to_640x480_inter_area",
    },
    {
        "camera_name": "cam_left_wrist",
        "topic": "/ego_video/left",
        "rotate_upright": False,
        "center_crop_aspect": None,
        "resize_attr": "1280x960_to_640x480_inter_area",
    },
    {
        "camera_name": "cam_right_wrist",
        "topic": "/ego_video/right",
        "rotate_upright": False,
        "center_crop_aspect": None,
        "resize_attr": "1280x960_to_640x480_inter_area",
    },
)


def _write_mp4_head_video(
    file: Any,
    video_path: str | Path,
    timestamps: np.ndarray,
    source_fps: float,
    staging_path: str | Path,
    *,
    source_hw: tuple[int, int],
    source_topic: str,
    image_hw: tuple[int, int] = OUTPUT_IMAGE_HW,
) -> np.ndarray:
    """Write a source MP4 head stream as fixed-width Spark JPEG bytes.

    Frames are selected by integer source index ``round(t * source_fps)`` so a
    dense native-fps timeline maps 1:1 without inventing gap frames.
    """
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as error:
        raise ImportError("Writing Spark HDF5 vision requires cv2") from error

    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Missing head video: {video_path}")

    selected: dict[int, list[int]] = {}
    for output_index, timestamp in enumerate(timestamps):
        source_index = int(round(float(timestamp) * source_fps))
        selected.setdefault(source_index, []).append(output_index)
    if not selected:
        raise ValueError(f"Cannot write an empty video trajectory for {source_topic}")

    vision = file.require_group("vision")
    camera = vision.require_group("cam_head")
    valid = np.zeros(len(timestamps), dtype=bool)
    offsets = np.full(len(timestamps), -1, dtype=np.int64)
    lengths = np.zeros(len(timestamps), dtype=np.int64)
    max_jpeg_bytes = 0
    image_shape: tuple[int, int, int] | None = None
    stage = Path(staging_path)
    stage.parent.mkdir(parents=True, exist_ok=True)
    stage.unlink(missing_ok=True)
    target_height, target_width = image_hw
    resize_attr = (
        f"{source_hw[1]}x{source_hw[0]}_center_crop_4x3_to_"
        f"{target_width}x{target_height}_inter_area"
    )

    try:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video {video_path}")
        with stage.open("wb") as staged:
            source_index = 0
            last_needed = max(selected)
            while source_index <= last_needed:
                ok, image = capture.read()
                if not ok or image is None:
                    break
                if source_index in selected:
                    image = egopro_center_crop_to_aspect(image, aspect_wh=(4, 3))
                    if image.shape[:2] != (target_height, target_width):
                        image = cv2.resize(
                            image,
                            (target_width, target_height),
                            interpolation=cv2.INTER_AREA,
                        )
                    try:
                        encoded = _encode_bgr_frame(image, quality=95)
                    except ValueError as error:
                        raise ValueError(
                            f"Could not encode {source_topic} frame {source_index}"
                        ) from error
                    image_shape = tuple(int(value) for value in image.shape)
                    for output_index in selected[source_index]:
                        offsets[output_index] = staged.tell()
                        lengths[output_index] = len(encoded)
                        staged.write(encoded)
                        valid[output_index] = True
                        max_jpeg_bytes = max(max_jpeg_bytes, len(encoded))
                source_index += 1
        capture.release()

        if not np.all(valid):
            missing = np.flatnonzero(~valid).tolist()
            raise ValueError(
                f"Missing {source_topic} frames for output indices {missing[:10]}"
            )
        if image_shape is None or max_jpeg_bytes < 1:
            raise ValueError(
                f"Could not determine {source_topic} image shape or JPEG size"
            )

        colors = camera.create_dataset(
            "colors",
            shape=(len(timestamps),),
            dtype=np.dtype(f"S{max_jpeg_bytes}"),
        )
        with stage.open("rb") as staged:
            for output_index, (offset, length) in enumerate(zip(offsets, lengths)):
                staged.seek(int(offset))
                encoded = staged.read(int(length))
                if len(encoded) != int(length):
                    raise ValueError(
                        f"Truncated staged JPEG at output index {output_index}"
                    )
                colors[output_index] = np.bytes_(encoded)
        camera.create_dataset("shape", data=np.asarray(image_shape, dtype=np.int64))
        camera.attrs["encoding"] = "jpeg"
        camera.attrs["storage_dtype"] = f"|S{max_jpeg_bytes}"
        camera.attrs["color_order"] = IMAGE_BITS_COLOR_ORDER
        camera.attrs["upright_correction"] = "none"
        camera.attrs["resize"] = resize_attr
        camera.attrs["source_topic"] = source_topic
        return valid
    finally:
        stage.unlink(missing_ok=True)


def _write_fourier_head_video(
    file: Any,
    rgb_path: str | Path,
    timestamps: np.ndarray,
    source_fps: float,
    staging_path: str | Path,
    image_hw: tuple[int, int] = OUTPUT_IMAGE_HW,
) -> np.ndarray:
    """Write ActionNet top RGB after 4:3 center-crop + INTER_AREA to 640×480."""
    from ..datasets.fourier_actionnet.hand import FOURIER_SOURCE_IMAGE_HW

    return _write_mp4_head_video(
        file,
        rgb_path,
        timestamps,
        source_fps,
        staging_path,
        source_hw=FOURIER_SOURCE_IMAGE_HW,
        source_topic="fourier_actionnet_top_rgb",
        image_hw=image_hw,
    )


def _write_fourier_cam_head_calibration(
    file: Any,
    frames: int,
    motion: UnifiedHumanMotion,
) -> None:
    """Write approximate ActionNet head-camera calibration.

    ActionNet publishes no K/extrinsics. Intrinsics are OAK-D W OV9782
    nominals scaled onto the written 640×480 head RGB (4:3 center-crop +
    uniform INTER_AREA); extrinsics compose the ``state/pose``
    ``head_yaw_link`` pose with an annotated PnP mount (see
    ``fourier_hand.FOURIER_HEAD_T_CAM_OPTICAL``). Stored poses follow native
    Spark ``camera_to_world`` (x right, y up, z back).
    """
    from ..datasets.fourier_actionnet.hand import (
        FOURIER_HEAD_T_CAM_OPTICAL,
        FOURIER_NOMINAL_INTRINSICS,
        FOURIER_SOURCE_IMAGE_HW,
    )

    intrinsics, crop_meta = scale_intrinsics_for_output_image(
        FOURIER_NOMINAL_INTRINSICS,
        source_hw=FOURIER_SOURCE_IMAGE_HW,
        output_hw=OUTPUT_IMAGE_HW,
        upright_rotate_180=False,
    )
    intrinsic_attrs = {
        "order": "fx,fy,cx,cy",
        "status": "approximate",
        "source": "oak_d_w_ov9782_nominal_1280x800_center_crop_4x3_to_640x480",
        "semantics": "pinhole_fx_fy_cx_cy_for_cam_head_rgb",
        "image_width": int(OUTPUT_IMAGE_HW[1]),
        "image_height": int(OUTPUT_IMAGE_HW[0]),
        "source_image_width": int(FOURIER_SOURCE_IMAGE_HW[1]),
        "source_image_height": int(FOURIER_SOURCE_IMAGE_HW[0]),
        "upright_correction": "none",
        **crop_meta,
    }
    head_poses = motion.metadata.get("head_poses")
    head_valid = motion.metadata.get("head_valid")
    dynamic = np.full((frames, 7), np.nan, dtype=np.float64)
    dynamic_valid = np.zeros(frames, dtype=bool)
    if head_poses is not None:
        head_poses = np.asarray(head_poses, dtype=np.float64)[:frames]
        head_valid = (
            np.asarray(head_valid, dtype=bool)[:frames]
            if head_valid is not None
            else np.all(np.isfinite(head_poses), axis=1)
        )
        valid_indices = np.flatnonzero(head_valid)
        if len(valid_indices):
            world_from_optical = (
                pose_wxyz_to_matrix(head_poses[valid_indices])
                @ FOURIER_HEAD_T_CAM_OPTICAL
            )
            dynamic[valid_indices] = matrix_to_pose_wxyz(
                world_from_optical @ OPTICAL_FROM_SPARK_CAMERA
            )
            dynamic_valid[valid_indices] = True

    pose_attrs = {
        "frame": "camera_to_world",
        "camera_axes": "x_right_y_up_z_back",
        "order": "x,y,z,qw,qx,qy,qz",
        "coordinate_frame": "source_robot_as_is",
        "status": "approximate",
        "source": "fourier_head_yaw_link_pose_times_annotated_mount",
        "calibration_note": (
            "head_T_cam solved 2026-08-06 by PnP on 7 annotated wrist pixels "
            "(episode 01JGZXNR33FF5ZC6G3, reprojection RMS ~43 px); "
            "intrinsics are OAK-D W OV9782 nominals scaled after the same "
            "4:3 center-crop + INTER_AREA resize as cam_head JPEG"
        ),
    }
    first_valid = np.flatnonzero(dynamic_valid)
    fixed_pose = (
        dynamic[first_valid[0]]
        if len(first_valid)
        else np.full(7, np.nan, dtype=np.float64)
    )
    fixed = file.create_dataset("extrinsics", data=fixed_pose)
    for key, value in pose_attrs.items():
        fixed.attrs[key] = value
    fixed.attrs["semantics"] = "fixed_first_valid_frame_world_from_camera"
    fixed.attrs["constant"] = True
    intrinsic_ds = file.create_dataset("intrinsics", data=intrinsics)
    for key, value in intrinsic_attrs.items():
        intrinsic_ds.attrs[key] = value
    camera = file.require_group("vision").require_group("cam_head")
    cam_intrinsics = camera.create_dataset("intrinsics", data=intrinsics)
    for key, value in intrinsic_attrs.items():
        cam_intrinsics.attrs[key] = value
    poses = camera.create_dataset("extrinsics", data=dynamic)
    for key, value in pose_attrs.items():
        poses.attrs[key] = value
    poses.attrs["semantics"] = "dynamic_world_from_camera"
    poses.attrs["constant"] = False
    poses.attrs["time_alignment"] = "output_timestamps"
    validity = file.require_group("validity")
    validity.create_dataset("cam_head_pose", data=dynamic_valid)


def write_fourier_spark_hdf5(
    output_path: str | Path,
    robot: UnifiedRobotMotion,
    motion: UnifiedHumanMotion,
    episode_path: str | Path,
    fps: float,
    action_steps: int = 1,
    mano_arrays: dict[str, np.ndarray] | None = None,
    mano_left_convention: str = "mirror_model",
) -> None:
    """Write Fourier ActionNet → Spark 54-D HDF5 (source robot frame, 640×480 RGB)."""
    try:
        import h5py
    except ImportError as error:
        raise ImportError("Writing Spark HDF5 requires h5py") from error

    if fps <= 0:
        raise ValueError("fps must be positive")
    if not np.isclose(fps, round(fps)):
        raise ValueError("Spark HDF5 frequency must be an integer")
    if action_steps < 1:
        raise ValueError("action_steps must be positive")
    source_frames = len(robot.timestamps)
    if source_frames <= action_steps:
        raise ValueError(
            f"Need more than {action_steps} frames to truncate future-state action, "
            f"got {source_frames}"
        )
    keep = source_frames - action_steps
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    episode_path = Path(episode_path)
    if not is_fourier_episode(episode_path):
        raise ValueError(f"Not a Fourier episode: {episode_path}")
    rgb_path = Path(motion.metadata.get("rgb_path", fourier_rgb_path(episode_path)))

    pseudo_action = make_future_state_action(robot, steps=action_steps)
    timestamps = np.asarray(robot.timestamps[:keep], dtype=np.float64)
    state = np.asarray(robot.state[:keep], dtype=np.float64)
    state_valid = np.asarray(robot.state_valid[:keep], dtype=bool)
    action = np.asarray(pseudo_action.action[:keep], dtype=np.float64)
    action_valid = np.asarray(pseudo_action.action_valid[:keep], dtype=bool)
    retarget_error = np.asarray(robot.retarget_error[:keep], dtype=np.float64)
    joints_world = np.asarray(motion.metadata["joints_world"][:keep], dtype=np.float64)
    joint_confidence = np.asarray(motion.joint_confidence[:keep], dtype=np.float64)
    tactile, tactile_valid = empty_tactile(keep)
    tactile_valid[:] = False

    left_ee = state[:, 0:7]
    left_hand = state[:, 7:27]
    right_ee = state[:, 27:34]
    right_hand = state[:, 34:54]
    left_ee_valid = state_valid[:, 0:7]
    left_hand_valid = state_valid[:, 7:27]
    right_ee_valid = state_valid[:, 27:34]
    right_hand_valid = state_valid[:, 34:54]
    left_action_ee = action[:, 0:7].copy()
    left_action = action[:, 7:27].copy()
    right_action_ee = action[:, 27:34].copy()
    right_action = action[:, 34:54].copy()
    left_action_ee_valid = action_valid[:, 0:7]
    left_action_valid = action_valid[:, 7:27]
    right_action_ee_valid = action_valid[:, 27:34]
    right_action_valid = action_valid[:, 34:54]
    left_action_ee[~left_action_ee_valid] = np.nan
    left_action[~left_action_valid] = np.nan
    right_action_ee[~right_action_ee_valid] = np.nan
    right_action[~right_action_valid] = np.nan
    unavailable_arm = np.full((keep, 7), np.nan, dtype=np.float64)
    unavailable_arm_valid = np.zeros_like(unavailable_arm, dtype=bool)
    writer_started = time.perf_counter()

    with h5py.File(output, "w") as file:
        text_dtype = h5py.string_dtype(encoding="utf-8")
        file.create_dataset("instruction", data=robot.instruction, dtype=text_dtype)
        file.create_dataset("data_format_version", data="v1.0", dtype=text_dtype)
        file.create_dataset("timestamps", data=timestamps)
        info = file.create_group("additional_info")
        info.create_dataset("frequency", data=np.int64(round(fps)))
        _write_fourier_cam_head_calibration(file, keep, motion)

        state_group = file.create_group("state")
        state_group.create_dataset("left_ee_poses", data=left_ee)
        state_group.create_dataset("right_ee_poses", data=right_ee)
        state_group.create_dataset("left_delta_ee_poses", data=left_ee)
        state_group.create_dataset("right_delta_ee_poses", data=right_ee)
        state_group.create_dataset("left_ee_joint_states", data=left_hand)
        state_group.create_dataset("right_ee_joint_states", data=right_hand)
        state_group.create_dataset("left_arm_joint_states", data=unavailable_arm)
        state_group.create_dataset("right_arm_joint_states", data=unavailable_arm)
        tactile_ds = state_group.create_dataset("tactile", data=tactile)
        tactile_ds.attrs["unit"] = "newton"
        tactile_ds.attrs["source"] = "not_available"
        tactile_ds.attrs["channels"] = ",".join(TACTILE_PATCH_KEYS)
        tactile_ds.attrs["order"] = "left_palm_to_pinky_then_right"

        action_group = file.create_group("action")
        action_group.create_dataset("left_ee_poses", data=left_action_ee)
        action_group.create_dataset("right_ee_poses", data=right_action_ee)
        action_group.create_dataset("left_ee_joint_states", data=left_action)
        action_group.create_dataset("right_ee_joint_states", data=right_action)
        action_group.create_dataset("left_arm_joint_states", data=unavailable_arm)
        action_group.create_dataset("right_arm_joint_states", data=unavailable_arm)
        _write_mano_groups(
            file, mano_arrays, action_steps, keep, mano_left_convention
        )

        validity = file.require_group("validity")
        validity.create_dataset("state_54", data=state_valid)
        validity.create_dataset("tactile", data=tactile_valid)
        validity.create_dataset("left_ee_pose", data=left_ee_valid)
        validity.create_dataset("right_ee_pose", data=right_ee_valid)
        validity.create_dataset("left_ee_joint_states", data=left_hand_valid)
        validity.create_dataset("right_ee_joint_states", data=right_hand_valid)
        validity.create_dataset("left_arm_joint_states", data=unavailable_arm_valid)
        validity.create_dataset("right_arm_joint_states", data=unavailable_arm_valid)
        validity.create_dataset("action_left_ee_pose", data=left_action_ee_valid)
        validity.create_dataset("action_right_ee_pose", data=right_action_ee_valid)
        validity.create_dataset("action_left_ee_joint_states", data=left_action_valid)
        validity.create_dataset("action_right_ee_joint_states", data=right_action_valid)
        validity.create_dataset("action_left_arm_joint_states", data=unavailable_arm_valid)
        validity.create_dataset("action_right_arm_joint_states", data=unavailable_arm_valid)

        aligned = file.create_group("aligned")
        aligned.create_dataset("joints_world", data=joints_world)
        aligned.create_dataset("joint_confidence", data=joint_confidence)
        aligned.create_dataset("retarget_error", data=retarget_error)

        video_started = time.perf_counter()
        source_fps = float(motion.metadata.get("video_fps", fps))
        staging_prefix = output.with_name(f".{output.name}.jpeg-stage")
        head_stage = staging_prefix.with_name(f"{staging_prefix.name}.cam_head")
        image_valid = _write_fourier_head_video(
            file,
            rgb_path,
            timestamps,
            source_fps=source_fps,
            staging_path=head_stage,
        )
        performance = robot.metadata.setdefault("performance", {})
        stage_s = performance.setdefault("stage_s", {})
        stage_s["video_hdf5"] = time.perf_counter() - video_started
        stage_s["hdf5_before_metadata"] = time.perf_counter() - writer_started
        validity.create_dataset("cam_head", data=image_valid)

        metadata = file.create_group("metadata")
        metadata.create_dataset("source", data=np.bytes_(robot.source.encode("utf-8")))
        metadata.create_dataset(
            "episode_truncation",
            data=np.int64(action_steps),
        )
        metadata.create_dataset(
            "source_frame_count",
            data=np.int64(source_frames),
        )
        metadata.create_dataset(
            "action_source",
            data=np.bytes_(f"retargeted_future_state:{action_steps}".encode("utf-8")),
        )
        metadata.create_dataset(
            "ee_action_source",
            data=np.bytes_(f"retargeted_future_state:{action_steps}".encode("utf-8")),
        )
        metadata.create_dataset(
            "arm_state_action_source",
            data=np.bytes_(b"not_available_nan"),
        )
        metadata.create_dataset(
            "delta_ee_pose_semantics",
            data=np.bytes_(b"legacy_absolute_copy_matches_native_spark_v1"),
        )
        metadata.create_dataset(
            "camera_alignment_json",
            data=np.bytes_(
                json.dumps(
                    robot.metadata.get("camera_alignment", {"skipped": True}),
                    ensure_ascii=False,
                    default=str,
                ).encode("utf-8")
            ),
        )
        provenance = {
            key: value
            for key, value in robot.metadata.items()
            if key != "joints_world"
        }
        provenance["episode_truncation"] = int(action_steps)
        provenance["source_frame_count"] = int(source_frames)
        provenance["output_frame_count"] = int(keep)
        provenance["action_semantics"] = f"next_state_truncated:{action_steps}"
        provenance["vision_policy"] = (
            "top_rgb_center_crop_4x3_to_640x480_inter_area"
        )
        provenance = _portable_metadata(provenance)
        metadata.create_dataset(
            "provenance_json",
            data=np.bytes_(
                json.dumps(provenance, ensure_ascii=False, default=str).encode("utf-8")
            ),
        )


def _encode_agibot_head_jpegs_from_mp4(
    video_path: str | Path,
    n_frames: int,
    *,
    quality: int = 95,
) -> tuple[list[bytes], tuple[int, int]]:
    """Decode AgiBot head MP4 and encode Spark-native JPEG bytes.

    Prefers PyAV because AgiBot ``head_color`` is often AV1 (OpenCV VideoCapture
    may fail on this host). Geometry stays in true BGR; encode uses
    :func:`_encode_bgr_frame` so channels match EgoPro/Fourier/Spark. Frames
    that are not 640x480 are center-cropped to 4:3 before resizing; the
    returned source ``(H, W)`` lets the caller scale the intrinsics to match.
    """
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as error:
        raise ImportError("Writing AgiBot vision requires cv2") from error

    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    out: list[bytes] = []
    source_hw: list[tuple[int, int]] = []
    try:
        import av  # type: ignore[import-not-found]
    except ImportError:
        av = None  # type: ignore[assignment]

    def _append(image: np.ndarray, index: int) -> None:
        hw = (int(image.shape[0]), int(image.shape[1]))
        if not source_hw:
            source_hw.append(hw)
        elif hw != source_hw[0]:
            raise ValueError(
                f"{video_path}: frame {index} is {hw}, expected {source_hw[0]}"
            )
        if hw != (480, 640):
            image = egopro_center_crop_to_aspect(image, aspect_wh=(4, 3))
            image = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
        try:
            out.append(_encode_bgr_frame(image, quality=quality))
        except ValueError as error:
            raise ValueError(f"JPEG encode failed at frame {index}") from error

    def _decode_ffmpeg_mjpeg() -> None:
        """Decode through the system ffmpeg when PyAV/OpenCV lack AV1 support."""
        import subprocess
        command = [
            "ffmpeg", "-v", "error", "-i", str(video_path),
            "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "2", "pipe:1",
        ]
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        buffer = bytearray()
        try:
            while len(out) < n_frames and process.stdout is not None:
                chunk = process.stdout.read(1 << 20)
                if not chunk:
                    break
                buffer.extend(chunk)
                while len(out) < n_frames:
                    start = buffer.find(b"\xff\xd8")
                    if start < 0:
                        if len(buffer) > 2:
                            del buffer[:-2]
                        break
                    end = buffer.find(b"\xff\xd9", start + 2)
                    if end < 0:
                        if start:
                            del buffer[:start]
                        break
                    jpeg = bytes(buffer[start:end + 2])
                    del buffer[:end + 2]
                    image = cv2.imdecode(
                        np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    if image is not None:
                        _append(image, len(out))
        finally:
            if process.stdout is not None:
                process.stdout.close()
            stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr is not None else ""
            process.stderr.close() if process.stderr is not None else None
            process.wait()
        if not out and stderr:
            raise RuntimeError(f"ffmpeg could not decode {video_path}: {stderr[-500:]}")

    if av is not None:
        try:
            with av.open(str(video_path)) as container:
                for index, frame in enumerate(container.decode(video=0)):
                    if index >= n_frames:
                        break
                    _append(frame.to_ndarray(format="bgr24"), index)
        except Exception:
            pass
    if len(out) < n_frames:
        out.clear()
        source_hw.clear()
        capture = cv2.VideoCapture(str(video_path))
        if capture.isOpened():
            try:
                for index in range(n_frames):
                    ok, image = capture.read()
                    if not ok or image is None:
                        break
                    _append(image, index)
            finally:
                capture.release()
        else:
            capture.release()
    if len(out) < n_frames:
        out.clear()
        source_hw.clear()
        _decode_ffmpeg_mjpeg()
    if len(out) < n_frames:
        raise ValueError(f"{video_path}: decoded {len(out)} frames, expected {n_frames}")
    return out[:n_frames], source_hw[0]


def write_agibot_spark_hdf5(
    output_path: str | Path,
    robot: UnifiedRobotMotion,
    motion: UnifiedHumanMotion,
    episode_path: str | Path,
    fps: float,
    action_steps: int = 1,
    mano_arrays: dict[str, np.ndarray] | None = None,
    mano_left_convention: str = "mirror_model",
) -> None:
    """Write AgiBotWorld-Beta(hand) → Spark 54-D HDF5 (source robot, head+wrists)."""
    try:
        import h5py
    except ImportError as error:
        raise ImportError("Writing Spark HDF5 requires h5py") from error

    from ..datasets.agibot_world.reader import (
        agibot_episode_paths,
        is_agibot_episode,
        load_agibot_head_camera,
        parse_agibot_episode,
    )
    from ..datasets.agibot_world.wrist import detect_wrist_variant, pack_one_wrist

    if fps <= 0:
        raise ValueError("fps must be positive")
    video_fps = motion.metadata.get("video_fps")
    if video_fps is not None and not np.isclose(fps, float(video_fps)):
        raise ValueError(
            f"AgiBot output fps {fps} must equal the source video fps {video_fps}: "
            "head and wrist frames are taken one per output frame"
        )
    if not np.isclose(fps, round(fps)):
        raise ValueError("Spark HDF5 frequency must be an integer")
    if action_steps < 1:
        raise ValueError("action_steps must be positive")
    source_frames = len(robot.timestamps)
    if source_frames <= action_steps:
        raise ValueError(
            f"Need more than {action_steps} frames to truncate future-state action, "
            f"got {source_frames}"
        )
    keep = source_frames - action_steps
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    episode_path = Path(episode_path)
    if not is_agibot_episode(episode_path):
        raise ValueError(f"Not an AgiBot episode: {episode_path}")
    root, task, ep = parse_agibot_episode(episode_path)
    paths = agibot_episode_paths(root, task, ep)

    pseudo_action = make_future_state_action(robot, steps=action_steps)
    timestamps = np.asarray(robot.timestamps[:keep], dtype=np.float64)
    state = np.asarray(robot.state[:keep], dtype=np.float64)
    state_valid = np.asarray(robot.state_valid[:keep], dtype=bool)
    action = np.asarray(pseudo_action.action[:keep], dtype=np.float64)
    action_valid = np.asarray(pseudo_action.action_valid[:keep], dtype=bool)
    retarget_error = np.asarray(robot.retarget_error[:keep], dtype=np.float64)
    joints_world = np.asarray(motion.metadata["joints_world"][:keep], dtype=np.float64)
    joint_confidence = np.asarray(motion.joint_confidence[:keep], dtype=np.float64)
    tactile, tactile_valid = empty_tactile(keep)
    tactile_valid[:] = False

    left_ee = state[:, 0:7]
    left_hand = state[:, 7:27]
    right_ee = state[:, 27:34]
    right_hand = state[:, 34:54]
    left_ee_valid = state_valid[:, 0:7]
    left_hand_valid = state_valid[:, 7:27]
    right_ee_valid = state_valid[:, 27:34]
    right_hand_valid = state_valid[:, 34:54]
    left_action_ee = action[:, 0:7].copy()
    left_action = action[:, 7:27].copy()
    right_action_ee = action[:, 27:34].copy()
    right_action = action[:, 34:54].copy()
    left_action_ee_valid = action_valid[:, 0:7]
    left_action_valid = action_valid[:, 7:27]
    right_action_ee_valid = action_valid[:, 27:34]
    right_action_valid = action_valid[:, 34:54]
    left_action_ee[~left_action_ee_valid] = np.nan
    left_action[~left_action_valid] = np.nan
    right_action_ee[~right_action_ee_valid] = np.nan
    right_action[~right_action_valid] = np.nan
    unavailable_arm = np.full((keep, 7), np.nan, dtype=np.float64)
    unavailable_arm_valid = np.zeros_like(unavailable_arm, dtype=bool)

    head_intrinsics, head_poses, head_shape = load_agibot_head_camera(
        paths["camera"], keep
    )
    head_jpegs, head_source_hw = _encode_agibot_head_jpegs_from_mp4(
        paths["head_video"], keep
    )
    head_intrinsics, head_crop = scale_intrinsics_for_output_image(
        head_intrinsics, source_hw=head_source_hw, output_hw=(480, 640)
    )
    wrist_variant = detect_wrist_variant(paths["camera"])
    wrist_payloads: dict[str, dict[str, Any] | None] = {}
    max_jpeg = max(len(b) for b in head_jpegs)
    for side in ("left", "right"):
        payload = pack_one_wrist(
            videos_dir=paths["videos"],
            params_dir=paths["camera"],
            side=side,
            variant=wrist_variant,
            n_frames=keep,
        )
        wrist_payloads[side] = payload
        if payload is not None:
            max_jpeg = max(max_jpeg, max(len(b) for b in payload["jpegs"]))

    with h5py.File(output, "w") as file:
        text_dtype = h5py.string_dtype(encoding="utf-8")
        file.create_dataset("instruction", data=robot.instruction, dtype=text_dtype)
        file.create_dataset("data_format_version", data="v1.0", dtype=text_dtype)
        file.create_dataset("timestamps", data=timestamps)
        info = file.create_group("additional_info")
        info.create_dataset("frequency", data=np.int64(round(fps)))

        vision = file.create_group("vision")
        cam = vision.create_group("cam_head")
        colors = np.empty(keep, dtype=f"|S{max_jpeg}")
        for index, encoded in enumerate(head_jpegs):
            colors[index] = np.bytes_(encoded)
        cam.create_dataset("colors", data=colors)
        cam.create_dataset("intrinsics", data=head_intrinsics)
        head_ext = cam.create_dataset("extrinsics", data=head_poses)
        head_ext.attrs["order"] = "x,y,z,qw,qx,qy,qz"
        head_ext.attrs["frame"] = "camera_to_world"
        head_ext.attrs["camera_axes"] = "x_right_y_up_z_back"
        cam.create_dataset("shape", data=head_shape)
        cam.attrs["encoding"] = "jpeg"
        cam.attrs["color_order"] = IMAGE_BITS_COLOR_ORDER
        cam.attrs["source_topic"] = "head_color.mp4"
        cam.attrs["resize"] = (
            "640x480_center_crop_4x3_to_640x480_native"
            if head_source_hw == (480, 640)
            else str(head_crop["resize"])
        )

        wrist_valid: dict[str, np.ndarray] = {}
        wrist_pose_valid: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            payload = wrist_payloads[side]
            if payload is None:
                wrist_valid[side] = np.zeros(keep, dtype=bool)
                wrist_pose_valid[side] = np.zeros(keep, dtype=bool)
                continue
            wcam = vision.create_group(f"cam_{side}_wrist")
            w_colors = np.empty(keep, dtype=f"|S{max_jpeg}")
            for index, encoded in enumerate(payload["jpegs"]):
                w_colors[index] = np.bytes_(encoded)
            wcam.create_dataset("colors", data=w_colors)
            wintr = wcam.create_dataset("intrinsics", data=payload["intrinsics"])
            wintr.attrs["order"] = "fx,fy,cx,cy"
            wintr.attrs["semantics"] = "pinhole_fx_fy_cx_cy_for_wrist_rgb"
            wintr.attrs["undistort_model"] = payload["undistort_model"]
            wintr.attrs["wrist_variant"] = payload["variant"]
            wintr.attrs["image_width"] = 640
            wintr.attrs["image_height"] = 480
            wintr.attrs["source_image_width"] = payload["source_image_width"]
            wintr.attrs["source_image_height"] = payload["source_image_height"]
            for key, value in payload["crop_meta"].items():
                wintr.attrs[key] = value
            wext = wcam.create_dataset("extrinsics", data=payload["extrinsics"])
            wext.attrs["order"] = "x,y,z,qw,qx,qy,qz"
            wext.attrs["frame"] = "camera_to_world"
            wext.attrs["camera_axes"] = "x_right_y_up_z_back"
            wcam.create_dataset("shape", data=payload["shape"])
            wcam.attrs["encoding"] = "jpeg"
            wcam.attrs["color_order"] = IMAGE_BITS_COLOR_ORDER
            wcam.attrs["source_topic"] = payload["source_topic"]
            wcam.attrs["undistort_model"] = payload["undistort_model"]
            wcam.attrs["wrist_variant"] = payload["variant"]
            wcam.attrs["resize"] = str(payload["crop_meta"].get("resize", "native"))
            wrist_valid[side] = np.ones(keep, dtype=bool)
            wrist_pose_valid[side] = np.all(
                np.isfinite(payload["extrinsics"]), axis=1
            )

        # Compatibility top-level intrinsics/extrinsics = head.
        file.create_dataset("intrinsics", data=head_intrinsics)
        fixed = file.create_dataset("extrinsics", data=head_poses[0])
        fixed.attrs["order"] = "x,y,z,qw,qx,qy,qz"
        fixed.attrs["frame"] = "camera_to_world"
        fixed.attrs["camera_axes"] = "x_right_y_up_z_back"
        fixed.attrs["constant"] = True

        state_group = file.create_group("state")
        state_group.create_dataset("left_ee_poses", data=left_ee)
        state_group.create_dataset("right_ee_poses", data=right_ee)
        state_group.create_dataset("left_delta_ee_poses", data=left_ee)
        state_group.create_dataset("right_delta_ee_poses", data=right_ee)
        state_group.create_dataset("left_ee_joint_states", data=left_hand)
        state_group.create_dataset("right_ee_joint_states", data=right_hand)
        state_group.create_dataset("left_arm_joint_states", data=unavailable_arm)
        state_group.create_dataset("right_arm_joint_states", data=unavailable_arm)
        tactile_ds = state_group.create_dataset("tactile", data=tactile)
        tactile_ds.attrs["unit"] = "newton"
        tactile_ds.attrs["source"] = "not_available"
        tactile_ds.attrs["channels"] = ",".join(TACTILE_PATCH_KEYS)
        tactile_ds.attrs["order"] = "left_palm_to_pinky_then_right"

        action_group = file.create_group("action")
        action_group.create_dataset("left_ee_poses", data=left_action_ee)
        action_group.create_dataset("right_ee_poses", data=right_action_ee)
        action_group.create_dataset("left_ee_joint_states", data=left_action)
        action_group.create_dataset("right_ee_joint_states", data=right_action)
        action_group.create_dataset("left_arm_joint_states", data=unavailable_arm)
        action_group.create_dataset("right_arm_joint_states", data=unavailable_arm)
        _write_mano_groups(
            file, mano_arrays, action_steps, keep, mano_left_convention
        )

        validity = file.create_group("validity")
        validity.create_dataset("state_54", data=state_valid)
        validity.create_dataset("tactile", data=tactile_valid)
        validity.create_dataset("left_ee_pose", data=left_ee_valid)
        validity.create_dataset("right_ee_pose", data=right_ee_valid)
        validity.create_dataset("left_ee_joint_states", data=left_hand_valid)
        validity.create_dataset("right_ee_joint_states", data=right_hand_valid)
        validity.create_dataset("left_arm_joint_states", data=unavailable_arm_valid)
        validity.create_dataset("right_arm_joint_states", data=unavailable_arm_valid)
        validity.create_dataset("action_left_ee_pose", data=left_action_ee_valid)
        validity.create_dataset("action_right_ee_pose", data=right_action_ee_valid)
        validity.create_dataset("action_left_ee_joint_states", data=left_action_valid)
        validity.create_dataset("action_right_ee_joint_states", data=right_action_valid)
        validity.create_dataset("action_left_arm_joint_states", data=unavailable_arm_valid)
        validity.create_dataset("action_right_arm_joint_states", data=unavailable_arm_valid)
        validity.create_dataset("cam_head", data=np.ones(keep, dtype=bool))
        validity.create_dataset(
            "cam_head_pose", data=np.all(np.isfinite(head_poses), axis=1)
        )
        for side in ("left", "right"):
            validity.create_dataset(f"cam_{side}_wrist", data=wrist_valid[side])
            validity.create_dataset(
                f"cam_{side}_wrist_pose", data=wrist_pose_valid[side]
            )

        aligned = file.create_group("aligned")
        aligned.create_dataset("joints_world", data=joints_world)
        aligned.create_dataset("joint_confidence", data=joint_confidence)
        aligned.create_dataset("retarget_error", data=retarget_error)

        metadata = file.create_group("metadata")
        metadata.create_dataset(
            "source",
            data=np.bytes_(
                f"agibot_world_beta_hand/{task}/{ep}:inspire_pnp_fk_wujihand2".encode(
                    "utf-8"
                )
            ),
        )
        metadata.create_dataset("episode_truncation", data=np.int64(action_steps))
        metadata.create_dataset("source_frame_count", data=np.int64(source_frames))
        metadata.create_dataset(
            "action_source",
            data=np.bytes_(f"retargeted_future_state:{action_steps}".encode("utf-8")),
        )
        metadata.create_dataset(
            "ee_action_source",
            data=np.bytes_(f"retargeted_future_state:{action_steps}".encode("utf-8")),
        )
        metadata.create_dataset(
            "arm_state_action_source",
            data=np.bytes_(b"not_available_nan"),
        )
        metadata.create_dataset(
            "delta_ee_pose_semantics",
            data=np.bytes_(b"legacy_absolute_copy_matches_native_spark_v1"),
        )
        metadata.create_dataset(
            "camera_alignment_json",
            data=np.bytes_(
                json.dumps(
                    {
                        "ee_frame": robot.metadata.get(
                            "ee_frame", "agibot_state_end_flange"
                        ),
                        "coordinate_frame": "agibot_robot_base_as_is",
                        "hand_variant": robot.metadata.get("hand_variant", "wujihand2"),
                        "wuji_order": "isaac_wujihand2",
                        "fk_anchor_hint": "palm-from-joints",
                        "skipped": True,
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
            ),
        )
        provenance = {
            key: value
            for key, value in robot.metadata.items()
            if key != "joints_world"
        }
        provenance.update(
            {
                "converter": "data_tools.data_alignment.retarget-agibot",
                "source_task": task,
                "source_episode": ep,
                "hand_pipeline": "agibot6 -> inspire_fk -> wujihand2_ik",
                "hand_urdf": "data_tools/data_alignment/assets/inspire_hand/",
                "hand_mount": motion.metadata.get("hand_mount"),
                "ee_frame": robot.metadata.get(
                    "ee_frame", "agibot_state_end_flange"
                ),
                "coordinate_frame": "agibot_robot_base_as_is",
                "camera_axes_convert": "agibot_opencv_zfwd -> spark x_right_y_up_z_back",
                "wrist_camera_variant": wrist_variant,
                "wrist_resize": (
                    "undistort_then_exact_4x3_center_crop_to_640x480_inter_area"
                ),
                "wrist_sides_present": {
                    side: wrist_payloads[side] is not None for side in ("left", "right")
                },
                "episode_truncation": int(action_steps),
                "source_frame_count": int(source_frames),
                "output_frame_count": int(keep),
                "action_semantics": f"next_state_truncated:{action_steps}",
                "vision_policy": (
                    "head_native_640x480_plus_wrist_undistort_640x480"
                ),
            }
        )
        provenance = _portable_metadata(provenance)
        metadata.create_dataset(
            "provenance_json",
            data=np.bytes_(
                json.dumps(provenance, ensure_ascii=False, default=str).encode("utf-8")
            ),
        )
