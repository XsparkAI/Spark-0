"""Write camera-aligned retarget results in the native Spark HDF5 layout."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from spark0.utils.image_codec import encode_image_bit

from ..camera_alignment import CameraAlignmentResult
from ..readers import (
    base_source_dataset_name,
    egopro_center_crop_to_aspect,
    egopro_head_image_is_inverted_for_source,
    egopro_intrinsics_image_size,
    is_openaoe_sample,
    load_egopro_camera_calibration,
    resolve_raw_instructions,
    scale_intrinsics_for_output_image,
)
from ..schema import UnifiedRobotMotion
from ..datasets.egodex.reader import is_egodex_episode
from ..datasets.egoverse.reader import is_egoverse_episode
from ..tactile.collector import (
    EgoProTactileContext,
    episode_tactile_lr_swapped,
    palm_patch_usable_from_meta,
    world_from_egopro_aligned,
    write_egopro_tactile_hdf5,
)
from ..camera_alignment import invert_transform, matrix_to_pose_wxyz, pose_wxyz_to_matrix
from .writer_common import (
    IMAGE_BITS_COLOR_ORDER,
    OPTICAL_FROM_SPARK_CAMERA,
    OUTPUT_IMAGE_HW,
    _portable_metadata,
)


def _output_source_frame_indices(
    timestamps: np.ndarray,
    source_fps: float,
) -> np.ndarray:
    """Map final output timestamps onto nearest source frames for every modality."""
    if source_fps <= 0:
        raise ValueError("source_fps must be positive")
    return np.rint(np.asarray(timestamps, dtype=np.float64) * source_fps).astype(
        np.int64
    )


def _extract_jpeg(payload: bytes) -> bytes | None:
    """Slice JPEG bytes out of an EgoPro MCAP / protobuf image payload.
    Search for ``FFD8 FF`` rather than a bare ``FFD8``. Foxglove protobuf
    varints for the JPEG length can contain ``FF D8`` (for example length
    109695 encodes as ``FF D8 06``), and that false SOI makes OpenCV/PIL
    fail to decode an otherwise valid frame.
    """
    start = payload.find(b"\xff\xd8\xff")
    end = payload.rfind(b"\xff\xd9")
    if start < 0 or end < start:
        return None
    return payload[start : end + 2]


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

# HumanTouch2 canonical MCAPs contain Annex-B H264 in Foxglove
# ``CompressedVideo.data``. Head pixels were rotated upright during the
# canonical H264 stage; wrist pixels retain their original 4:3 orientation.
HUMANTOUCH2_CAMERA_STREAMS: tuple[dict[str, Any], ...] = (
    {
        "camera_name": "cam_head",
        "topic": "/views/head/image",
        "center_crop_aspect": (4, 3),
        "resize_attr": "1920x1080_center_crop_1440x1080_to_640x480_inter_area",
        "upright_correction": "source_pre_rotated_180",
    },
    {
        "camera_name": "cam_left_wrist",
        "topic": "/views/left_hand/image",
        "center_crop_aspect": None,
        "resize_attr": "1280x960_to_640x480_inter_area",
        "upright_correction": "none",
    },
    {
        "camera_name": "cam_right_wrist",
        "topic": "/views/right_hand/image",
        "center_crop_aspect": None,
        "resize_attr": "1280x960_to_640x480_inter_area",
        "upright_correction": "none",
    },
)


def _protobuf_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(data) and shift < 70:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, offset
        shift += 7
    raise ValueError("Invalid protobuf varint")


def _protobuf_length_delimited_field(data: bytes, field_number: int) -> bytes | None:
    """Extract one length-delimited protobuf field without generated classes."""
    offset = 0
    while offset < len(data):
        tag, offset = _protobuf_varint(data, offset)
        wire_type = tag & 0x7
        current = tag >> 3
        if wire_type == 0:
            _value, offset = _protobuf_varint(data, offset)
        elif wire_type == 1:
            offset += 8
        elif wire_type == 2:
            length, offset = _protobuf_varint(data, offset)
            end = offset + length
            if end > len(data):
                raise ValueError("Truncated protobuf length-delimited field")
            if current == field_number:
                return data[offset:end]
            offset = end
        elif wire_type == 5:
            offset += 4
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire_type}")
        if offset > len(data):
            raise ValueError("Truncated protobuf field")
    return None


def _write_egopro_camera_video(
    file: Any,
    mcap_path: str | Path,
    timestamps: np.ndarray,
    source_fps: float,
    staging_path: str | Path,
    *,
    camera_name: str,
    topic: str,
    rotate_upright: bool,
    center_crop_aspect: tuple[int, int] | None,
    resize_attr: str,
    image_hw: tuple[int, int] = OUTPUT_IMAGE_HW,
    allow_missing_jpeg: bool = False,
) -> np.ndarray:
    """Stage processed JPEGs for one EgoPro topic, then write fixed-width bytes.

    When ``allow_missing_jpeg`` is enabled, a source message without a JPEG is
    represented by the nearest valid encoded frame and its camera validity bit
    remains false.  The default is intentionally strict so malformed source
    data is still surfaced unless a caller opts into this policy explicitly.
    """
    try:
        import cv2  # type: ignore[import-not-found]
        from mcap.reader import make_reader
    except ImportError as error:
        raise ImportError("Writing Spark HDF5 vision requires cv2 and mcap") from error

    selected: dict[int, list[int]] = {}
    for output_index, source_index in enumerate(
        _output_source_frame_indices(timestamps, source_fps)
    ):
        selected.setdefault(source_index, []).append(output_index)
    if not selected:
        raise ValueError("Cannot write an empty video trajectory")

    vision = file.require_group("vision")
    camera = vision.require_group(camera_name)
    valid = np.zeros(len(timestamps), dtype=bool)
    offsets = np.full(len(timestamps), -1, dtype=np.int64)
    lengths = np.zeros(len(timestamps), dtype=np.int64)
    max_jpeg_bytes = 0
    image_shape: tuple[int, int, int] | None = None
    last_encoded: bytes | None = None
    last_source_index: int | None = None
    deferred_output_indices: list[tuple[int, int]] = []
    fallback_records: list[dict[str, int]] = []
    stage = Path(staging_path)
    stage.parent.mkdir(parents=True, exist_ok=True)
    stage.unlink(missing_ok=True)
    target_height, target_width = image_hw

    try:
        with stage.open("wb") as staged, Path(mcap_path).open("rb") as stream:
            reader = make_reader(stream)
            source_index = 0
            last_needed = max(selected)
            for _schema, _channel, message in reader.iter_messages(topics=[topic]):
                if source_index in selected:
                    jpeg = _extract_jpeg(bytes(message.data))
                    if jpeg is None:
                        if not allow_missing_jpeg:
                            raise ValueError(
                                f"No JPEG payload in EgoPro {camera_name} frame "
                                f"{source_index}"
                            )
                        if last_encoded is None:
                            deferred_output_indices.extend(
                                (source_index, output_index)
                                for output_index in selected[source_index]
                            )
                            print(
                                f"JPEG_FALLBACK_DEFER camera={camera_name} "
                                f"source_frame={source_index}",
                                flush=True,
                            )
                            source_index += 1
                            if source_index > last_needed:
                                break
                            continue
                        encoded = last_encoded
                        assert last_source_index is not None
                        for output_index in selected[source_index]:
                            offsets[output_index] = staged.tell()
                            lengths[output_index] = len(encoded)
                            staged.write(encoded)
                            max_jpeg_bytes = max(max_jpeg_bytes, len(encoded))
                            fallback_records.append(
                                {
                                    "output_index": int(output_index),
                                    "missing_source_frame": int(source_index),
                                    "fallback_source_frame": int(last_source_index),
                                }
                            )
                        print(
                            f"JPEG_FALLBACK camera={camera_name} "
                            f"source_frame={source_index} "
                            f"fallback_source_frame={last_source_index}",
                            flush=True,
                        )
                        source_index += 1
                        if source_index > last_needed:
                            break
                        continue
                    image = cv2.imdecode(
                        np.frombuffer(jpeg, dtype=np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    if image is None:
                        raise ValueError(
                            f"Could not decode EgoPro {camera_name} frame "
                            f"{source_index}"
                        )
                    # EgoPro publishes ordinary camera JPEGs, which OpenCV
                    # decodes as BGR. Convert once here so everything below
                    # this line — geometry, encode, on-disk bits — is RGB.
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    if rotate_upright:
                        image = cv2.rotate(image, cv2.ROTATE_180)
                    if center_crop_aspect is not None:
                        image = egopro_center_crop_to_aspect(
                            image, aspect_wh=center_crop_aspect
                        )
                    if image.shape[:2] != (target_height, target_width):
                        image = cv2.resize(
                            image,
                            (target_width, target_height),
                            interpolation=cv2.INTER_AREA,
                        )
                    try:
                        encoded = encode_image_bit(image, quality=95)
                    except ValueError as error:
                        raise ValueError(
                            f"Could not encode EgoPro {camera_name} frame "
                            f"{source_index}"
                        ) from error
                    image_shape = tuple(int(value) for value in image.shape)
                    if deferred_output_indices:
                        for missing_source_index, output_index in deferred_output_indices:
                            offsets[output_index] = staged.tell()
                            lengths[output_index] = len(encoded)
                            staged.write(encoded)
                            max_jpeg_bytes = max(max_jpeg_bytes, len(encoded))
                            fallback_records.append(
                                {
                                    "output_index": int(output_index),
                                    "missing_source_frame": int(missing_source_index),
                                    "fallback_source_frame": int(source_index),
                                }
                            )
                        deferred_output_indices.clear()
                    for output_index in selected[source_index]:
                        offsets[output_index] = staged.tell()
                        lengths[output_index] = len(encoded)
                        staged.write(encoded)
                        valid[output_index] = True
                        max_jpeg_bytes = max(max_jpeg_bytes, len(encoded))
                    last_encoded = encoded
                    last_source_index = source_index
                if source_index >= last_needed:
                    break
                source_index += 1

        if deferred_output_indices:
            if not allow_missing_jpeg or last_encoded is None or last_source_index is None:
                missing = [output_index for _source, output_index in deferred_output_indices]
                raise ValueError(
                    f"Missing EgoPro {camera_name} frames for output indices "
                    f"{missing[:10]}"
                )
            with stage.open("ab") as staged:
                for missing_source_index, output_index in deferred_output_indices:
                    offsets[output_index] = staged.tell()
                    lengths[output_index] = len(last_encoded)
                    staged.write(last_encoded)
                    max_jpeg_bytes = max(max_jpeg_bytes, len(last_encoded))
                    fallback_records.append(
                        {
                            "output_index": int(output_index),
                            "missing_source_frame": int(missing_source_index),
                            "fallback_source_frame": int(last_source_index),
                        }
                    )
            deferred_output_indices.clear()

        if not allow_missing_jpeg and not np.all(valid):
            missing = np.flatnonzero(~valid).tolist()
            raise ValueError(
                f"Missing EgoPro {camera_name} frames for output indices "
                f"{missing[:10]}"
            )
        if np.any(offsets < 0) or np.any(lengths < 1):
            missing = np.flatnonzero((offsets < 0) | (lengths < 1)).tolist()
            raise ValueError(
                f"Missing staged EgoPro {camera_name} frames for output indices "
                f"{missing[:10]}"
            )
        if image_shape is None or max_jpeg_bytes < 1:
            raise ValueError(
                f"Could not determine EgoPro {camera_name} image shape or JPEG size"
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
        camera.attrs["upright_correction"] = (
            "rotate_180" if rotate_upright else "none"
        )
        camera.attrs["resize"] = resize_attr
        camera.attrs["source_topic"] = topic
        if fallback_records:
            camera.attrs["missing_jpeg_policy"] = "duplicate_nearest_valid_frame"
            camera.attrs["missing_jpeg_fallback_count"] = np.int64(
                len(fallback_records)
            )
            camera.attrs["missing_jpeg_fallbacks_json"] = json.dumps(
                fallback_records, separators=(",", ":")
            )
        return valid
    finally:
        stage.unlink(missing_ok=True)


def _write_egopro_videos(
    file: Any,
    mcap_path: str | Path,
    timestamps: np.ndarray,
    source_fps: float,
    staging_prefix: str | Path,
    image_hw: tuple[int, int] = OUTPUT_IMAGE_HW,
    source_dataset: str = "egopro",
    allow_missing_jpeg: bool = False,
) -> dict[str, np.ndarray]:
    """Write head and wrist EgoPro RGB streams; return per-camera validity."""
    extrinsics = load_egopro_camera_calibration(mcap_path)["extrinsics"]
    head_upright = egopro_head_image_is_inverted_for_source(
        extrinsics, source_dataset
    )
    prefix = Path(staging_prefix)
    validity: dict[str, np.ndarray] = {}
    for stream in EGOPRO_CAMERA_STREAMS:
        rotate = stream["rotate_upright"]
        if rotate == "from_calibration":
            rotate_upright = bool(head_upright)
        else:
            rotate_upright = bool(rotate)
        camera_name = str(stream["camera_name"])
        validity[camera_name] = _write_egopro_camera_video(
            file,
            mcap_path,
            timestamps,
            source_fps,
            staging_path=prefix.with_name(f"{prefix.name}.{camera_name}"),
            camera_name=camera_name,
            topic=str(stream["topic"]),
            rotate_upright=rotate_upright,
            center_crop_aspect=stream["center_crop_aspect"],
            resize_attr=str(stream["resize_attr"]),
            image_hw=image_hw,
            allow_missing_jpeg=allow_missing_jpeg,
        )
    return validity


def _write_humantouch2_videos(
    file: Any,
    mcap_path: str | Path,
    timestamps: np.ndarray,
    source_fps: float,
    staging_prefix: str | Path,
    image_hw: tuple[int, int] = OUTPUT_IMAGE_HW,
) -> dict[str, np.ndarray]:
    """Decode all three aligned HumanTouch2 H264 streams in one MCAP pass."""
    try:
        import av  # type: ignore[import-not-found]
        import cv2  # type: ignore[import-not-found]
        from mcap.reader import make_reader
    except ImportError as error:
        raise ImportError(
            "Writing HumanTouch2 vision requires PyAV, cv2, and mcap"
        ) from error
    from contextlib import ExitStack

    selected: dict[int, list[int]] = {}
    for output_index, source_index in enumerate(
        _output_source_frame_indices(timestamps, source_fps)
    ):
        selected.setdefault(int(source_index), []).append(output_index)
    if not selected:
        raise ValueError("Cannot write an empty HumanTouch2 video trajectory")
    last_needed = max(selected)
    target_height, target_width = image_hw
    prefix = Path(staging_prefix)
    runtimes: dict[str, dict[str, Any]] = {}
    stage_paths: list[Path] = []
    for spec in HUMANTOUCH2_CAMERA_STREAMS:
        camera_name = str(spec["camera_name"])
        stage = prefix.with_name(f"{prefix.name}.{camera_name}")
        stage.parent.mkdir(parents=True, exist_ok=True)
        stage.unlink(missing_ok=True)
        stage_paths.append(stage)
        runtimes[str(spec["topic"])] = {
            "spec": spec,
            "stage": stage,
            "codec": av.CodecContext.create("h264", "r"),
            "decoded_index": 0,
            "valid": np.zeros(len(timestamps), dtype=bool),
            "offsets": np.full(len(timestamps), -1, dtype=np.int64),
            "lengths": np.zeros(len(timestamps), dtype=np.int64),
            "max_jpeg_bytes": 0,
            "image_shape": None,
        }

    def store_frame(runtime: dict[str, Any], frame: Any) -> None:
        source_index = int(runtime["decoded_index"])
        runtime["decoded_index"] = source_index + 1
        if source_index not in selected:
            return
        spec = runtime["spec"]
        image = frame.to_ndarray(format="rgb24")
        aspect = spec["center_crop_aspect"]
        if aspect is not None:
            image = egopro_center_crop_to_aspect(image, aspect_wh=aspect)
        if image.shape[:2] != (target_height, target_width):
            image = cv2.resize(
                image,
                (target_width, target_height),
                interpolation=cv2.INTER_AREA,
            )
        encoded = encode_image_bit(image, quality=95)
        runtime["image_shape"] = tuple(int(value) for value in image.shape)
        staged = runtime["stream"]
        for output_index in selected[source_index]:
            runtime["offsets"][output_index] = staged.tell()
            runtime["lengths"][output_index] = len(encoded)
            staged.write(encoded)
            runtime["valid"][output_index] = True
            runtime["max_jpeg_bytes"] = max(
                int(runtime["max_jpeg_bytes"]), len(encoded)
            )

    def feed(runtime: dict[str, Any], bitstream: bytes) -> None:
        for packet in runtime["codec"].parse(bitstream):
            for frame in runtime["codec"].decode(packet):
                store_frame(runtime, frame)

    try:
        with ExitStack() as stack:
            for runtime in runtimes.values():
                runtime["stream"] = stack.enter_context(runtime["stage"].open("wb"))
            source_stream = stack.enter_context(Path(mcap_path).open("rb"))
            reader = make_reader(source_stream)
            for _schema, channel, message in reader.iter_messages(
                topics=list(runtimes)
            ):
                runtime = runtimes[channel.topic]
                bitstream = _protobuf_length_delimited_field(bytes(message.data), 3)
                if bitstream is None:
                    raise ValueError(
                        "No CompressedVideo.data field in HumanTouch2 "
                        f"{runtime['spec']['camera_name']} message"
                    )
                feed(runtime, bitstream)
                if all(
                    int(item["decoded_index"]) > last_needed
                    for item in runtimes.values()
                ):
                    break
            for runtime in runtimes.values():
                if np.all(runtime["valid"]):
                    continue
                feed(runtime, b"")
                for frame in runtime["codec"].decode(None):
                    store_frame(runtime, frame)

        validity: dict[str, np.ndarray] = {}
        vision = file.require_group("vision")
        for runtime in runtimes.values():
            spec = runtime["spec"]
            camera_name = str(spec["camera_name"])
            valid = runtime["valid"]
            if not np.all(valid):
                missing = np.flatnonzero(~valid).tolist()
                raise ValueError(
                    f"Missing HumanTouch2 {camera_name} frames for output indices "
                    f"{missing[:10]}"
                )
            image_shape = runtime["image_shape"]
            max_jpeg_bytes = int(runtime["max_jpeg_bytes"])
            if image_shape is None or max_jpeg_bytes < 1:
                raise ValueError(
                    f"Could not determine HumanTouch2 {camera_name} image output"
                )
            camera = vision.require_group(camera_name)
            colors = camera.create_dataset(
                "colors",
                shape=(len(timestamps),),
                dtype=np.dtype(f"S{max_jpeg_bytes}"),
            )
            with runtime["stage"].open("rb") as staged:
                for output_index, (offset, length) in enumerate(
                    zip(runtime["offsets"], runtime["lengths"])
                ):
                    staged.seek(int(offset))
                    encoded = staged.read(int(length))
                    if len(encoded) != int(length):
                        raise ValueError(
                            f"Truncated staged JPEG at output index {output_index}"
                        )
                    colors[output_index] = np.bytes_(encoded)
            camera.create_dataset(
                "shape", data=np.asarray(image_shape, dtype=np.int64)
            )
            camera.attrs["encoding"] = "jpeg"
            camera.attrs["source_encoding"] = "h264_annexb"
            camera.attrs["storage_dtype"] = f"|S{max_jpeg_bytes}"
            camera.attrs["color_order"] = IMAGE_BITS_COLOR_ORDER
            camera.attrs["upright_correction"] = str(spec["upright_correction"])
            camera.attrs["resize"] = str(spec["resize_attr"])
            camera.attrs["source_topic"] = str(spec["topic"])
            validity[camera_name] = valid
        return validity
    finally:
        for stage in stage_paths:
            stage.unlink(missing_ok=True)


def _openaoe_source_hw(sample_dir: str | Path) -> tuple[int, int]:
    """Return ``(height, width)`` of the delivered undistorted ego video."""
    from ..readers import _openaoe_video_hw

    return _openaoe_video_hw(Path(sample_dir))


def _output_intrinsics(
    alignment: CameraAlignmentResult,
    source_path: str | Path,
    image_hw: tuple[int, int] = OUTPUT_IMAGE_HW,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return intrinsics aligned to upright edge-cropped 640x480 RGB."""
    if is_openaoe_sample(source_path):
        upright = False
        upright_label = "none"
        source_hw = _openaoe_source_hw(source_path)
    elif is_egodex_episode(source_path):
        upright = False
        upright_label = "none"
        source_hw = (1080, 1920)
    elif is_egoverse_episode(source_path):
        upright = False
        upright_label = "none"
        from ..datasets.egoverse.reader import egoverse_source_hw
        source_hw = egoverse_source_hw(source_path)
    else:
        calibration = load_egopro_camera_calibration(source_path)
        source_dataset = base_source_dataset_name(alignment.motion.source)
        upright = egopro_head_image_is_inverted_for_source(
            calibration["extrinsics"],
            source_dataset,
        )
        upright_label = "rotate_180" if upright else "none"
        if source_dataset == "humantouch2" and upright:
            # The canonical H264 stage already rotated the pixels, but K still
            # comes from the original camera calibration and needs the same
            # principal-point mapping.
            upright_label = "source_pre_rotated_180"
        source_hw = egopro_intrinsics_image_size(calibration.get("intrinsics"))
    scaled, crop_meta = scale_intrinsics_for_output_image(
        alignment.intrinsics,
        source_hw=source_hw,
        output_hw=image_hw,
        upright_rotate_180=upright,
        aspect_wh=(4, 3),
    )
    attrs: dict[str, Any] = {
        "camera_model": "pinhole",
        "order": "fx,fy,cx,cy",
        "semantics": "pinhole_fx_fy_cx_cy_for_cam_head_rgb",
        "image_width": int(image_hw[1]),
        "image_height": int(image_hw[0]),
        "source_image_width": int(source_hw[1]),
        "source_image_height": int(source_hw[0]),
        "upright_correction": upright_label,
        **crop_meta,
    }
    return scaled, attrs


def _write_openaoe_head_video(
    file: Any,
    sample_dir: str | Path,
    timestamps: np.ndarray,
    source_fps: float,
    staging_path: str | Path,
    image_hw: tuple[int, int] = OUTPUT_IMAGE_HW,
) -> np.ndarray:
    """Write OpenAoE undistorted head RGB as fixed-width JPEG bytes."""
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as error:
        raise ImportError("Writing Spark HDF5 vision requires cv2") from error

    source_path = Path(sample_dir)
    egodex = is_egodex_episode(source_path)
    video_path = (
        source_path.with_suffix(".mp4")
        if egodex
        else source_path
        / "ego_process"
        / "ego_undistorted_video"
        / "raw_video_undistorted.mp4"
    )
    if not video_path.is_file():
        raise FileNotFoundError(f"Missing source RGB video: {video_path}")

    selected: dict[int, list[int]] = {}
    for output_index, source_index in enumerate(
        _output_source_frame_indices(timestamps, source_fps)
    ):
        selected.setdefault(source_index, []).append(output_index)
    if not selected:
        raise ValueError("Cannot write an empty OpenAoE video trajectory")

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
    source_hw = (1080, 1920) if egodex else _openaoe_source_hw(sample_dir)
    resize_attr = (
        f"{source_hw[1]}x{source_hw[0]}_center_crop_4x3_to_"
        f"{target_width}x{target_height}_inter_area"
    )

    try:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open OpenAoE video {video_path}")
        with stage.open("wb") as staged:
            source_index = 0
            last_needed = max(selected)
            while source_index <= last_needed:
                ok, image = capture.read()
                if not ok or image is None:
                    break
                if source_index in selected:
                    # cv2.VideoCapture returns BGR; everything below is RGB.
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = egopro_center_crop_to_aspect(image, aspect_wh=(4, 3))
                    if image.shape[:2] != (target_height, target_width):
                        image = cv2.resize(
                            image,
                            (target_width, target_height),
                            interpolation=cv2.INTER_AREA,
                        )
                    try:
                        encoded = encode_image_bit(image, quality=95)
                    except ValueError as error:
                        raise ValueError(
                            f"Could not encode OpenAoE frame {source_index}"
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
                f"Missing OpenAoE frames for output indices {missing[:10]}"
            )
        if image_shape is None or max_jpeg_bytes < 1:
            raise ValueError("Could not determine OpenAoE image shape or JPEG size")

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
        camera.attrs["source_topic"] = "egodex_rgb_mp4" if egodex else "openaoe_undistorted_head"
        return valid
    finally:
        stage.unlink(missing_ok=True)


def _write_egoverse_head_video(
    file: Any,
    source_path: str | Path,
    timestamps: np.ndarray,
    source_fps: float,
    staging_path: str | Path,
    image_hw: tuple[int, int] = OUTPUT_IMAGE_HW,
) -> np.ndarray:
    """Write EgoVerse front RGB as Spark JPEG bits using the canonical RGB rule."""
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as error:
        raise ImportError("Writing EgoVerse vision requires cv2") from error
    from ..datasets.egoverse.reader import (
        egoverse_head_source_topic,
        egoverse_rgb_timestamps,
        egoverse_source_hw,
        load_egoverse_head_frames,
    )
    source = Path(source_path)
    target_timestamps = np.asarray(timestamps, dtype=np.float64)
    source_rgb_timestamps = egoverse_rgb_timestamps(source)
    if source_rgb_timestamps is not None and source_rgb_timestamps.size:
        right = np.searchsorted(source_rgb_timestamps, target_timestamps, side="left")
        right = np.clip(right, 0, len(source_rgb_timestamps) - 1)
        left = np.clip(right - 1, 0, len(source_rgb_timestamps) - 1)
        choose_left = np.abs(target_timestamps - source_rgb_timestamps[left]) <= np.abs(target_timestamps - source_rgb_timestamps[right])
        selected = np.where(choose_left, left, right).astype(np.int64)
    else:
        selected = _output_source_frame_indices(target_timestamps, source_fps)
    if selected.size == 0:
        raise ValueError("Cannot write an empty EgoVerse video trajectory")
    frames = load_egoverse_head_frames(source, sorted(set(int(i) for i in selected)))
    target_height, target_width = image_hw
    stage = Path(staging_path)
    stage.parent.mkdir(parents=True, exist_ok=True)
    stage.unlink(missing_ok=True)
    valid = np.zeros(len(timestamps), dtype=bool)
    offsets = np.full(len(timestamps), -1, dtype=np.int64)
    lengths = np.zeros(len(timestamps), dtype=np.int64)
    max_jpeg_bytes = 0
    image_shape: tuple[int, int, int] | None = None
    source_hw = egoverse_source_hw(source)
    resize_attr = (
        f"{source_hw[1]}x{source_hw[0]}_center_crop_4x3_to_"
        f"{target_width}x{target_height}_inter_area"
    )
    with stage.open("wb") as staged:
        for out_index, source_index in enumerate(selected.tolist()):
            image = frames.get(int(source_index))
            if image is None:
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = egopro_center_crop_to_aspect(image, aspect_wh=(4, 3))
            if image.shape[:2] != (target_height, target_width):
                image = cv2.resize(
                    image,
                    (target_width, target_height),
                    interpolation=cv2.INTER_AREA,
                )
            encoded = encode_image_bit(image, quality=95)
            offsets[out_index] = staged.tell()
            lengths[out_index] = len(encoded)
            staged.write(encoded)
            valid[out_index] = True
            max_jpeg_bytes = max(max_jpeg_bytes, len(encoded))
            image_shape = tuple(int(value) for value in image.shape)
    if not np.all(valid):
        missing = np.flatnonzero(~valid).tolist()
        raise ValueError(f"Missing EgoVerse frames for output indices {missing[:10]}")
    if image_shape is None or max_jpeg_bytes < 1:
        raise ValueError("Could not determine EgoVerse image shape or JPEG size")
    camera = file.require_group("vision").require_group("cam_head")
    colors = camera.create_dataset(
        "colors", shape=(len(timestamps),), dtype=np.dtype(f"S{max_jpeg_bytes}")
    )
    with stage.open("rb") as staged:
        for out_index, (offset, length) in enumerate(zip(offsets, lengths)):
            staged.seek(int(offset))
            encoded = staged.read(int(length))
            colors[out_index] = np.bytes_(encoded)
    camera.create_dataset("shape", data=np.asarray(image_shape, dtype=np.int64))
    camera.attrs["encoding"] = "jpeg"
    camera.attrs["storage_dtype"] = f"|S{max_jpeg_bytes}"
    camera.attrs["color_order"] = IMAGE_BITS_COLOR_ORDER
    camera.attrs["upright_correction"] = "none"
    camera.attrs["resize"] = resize_attr
    camera.attrs["source_topic"] = egoverse_head_source_topic(source)
    stage.unlink(missing_ok=True)
    return valid


def _write_cam_head_calibration(
    file: Any,
    alignment: CameraAlignmentResult,
    *,
    ego_mcap_path: str | Path,
    camera_from_world: np.ndarray,
    camera_valid: np.ndarray,
) -> None:
    """Write fixed compatibility and dynamic world-from-camera extrinsics.

    Stored poses follow the native Spark convention (``camera_to_world``):
    the pose of the camera in robot-base world, expressed for the native
    Spark camera frame (x right, y up, z backward). Consumers projecting to
    OpenCV pixels must apply ``diag(1,-1,-1)`` after inverting, exactly as
    for native Spark episodes.
    """
    world_from_camera = np.asarray(alignment.world_from_virtual_ego_optical, dtype=np.float64)
    if world_from_camera.shape != (4, 4):
        raise ValueError(
            "alignment.world_from_virtual_ego_optical must have shape [4, 4], "
            f"got {world_from_camera.shape}"
        )
    intrinsics, intrinsic_attrs = _output_intrinsics(alignment, ego_mcap_path)
    if intrinsics.shape != (4,):
        raise ValueError(f"alignment intrinsics must be [fx, fy, cx, cy], got {intrinsics.shape}")
    dynamic_cam_from_world = np.asarray(camera_from_world, dtype=np.float64)
    dynamic_valid = np.asarray(camera_valid, dtype=bool)
    frames = int(dynamic_valid.shape[0])
    if dynamic_cam_from_world.shape != (frames, 7):
        raise ValueError(
            f"camera_from_world must have shape {(frames, 7)}, got {dynamic_cam_from_world.shape}"
        )
    if dynamic_valid.shape != (frames,):
        raise ValueError(
            f"camera_valid must have shape {(frames,)}, got {dynamic_valid.shape}"
        )
    dynamic = np.full_like(dynamic_cam_from_world, np.nan)
    valid_indices = np.flatnonzero(dynamic_valid)
    if len(valid_indices):
        dynamic[valid_indices] = matrix_to_pose_wxyz(
            invert_transform(pose_wxyz_to_matrix(dynamic_cam_from_world[valid_indices]))
            @ OPTICAL_FROM_SPARK_CAMERA
        )

    fixed = file.create_dataset(
        "extrinsics",
        data=matrix_to_pose_wxyz(world_from_camera @ OPTICAL_FROM_SPARK_CAMERA),
    )
    fixed.attrs["semantics"] = "fixed_virtual_world_from_camera"
    fixed.attrs["frame"] = "camera_to_world"
    fixed.attrs["camera_axes"] = "x_right_y_up_z_back"
    fixed.attrs["order"] = "x,y,z,qw,qx,qy,qz"
    fixed.attrs["constant"] = True
    # Top-level copy for older readers; Spark reference keeps K under cam_head.
    intrinsic_ds = file.create_dataset("intrinsics", data=intrinsics)
    for key, value in intrinsic_attrs.items():
        intrinsic_ds.attrs[key] = value
    camera = file.require_group("vision").require_group("cam_head")
    cam_intrinsics = camera.create_dataset("intrinsics", data=intrinsics)
    for key, value in intrinsic_attrs.items():
        cam_intrinsics.attrs[key] = value
    poses = camera.create_dataset("extrinsics", data=dynamic)
    poses.attrs["semantics"] = "dynamic_world_from_camera"
    poses.attrs["frame"] = "camera_to_world"
    poses.attrs["camera_axes"] = "x_right_y_up_z_back"
    poses.attrs["order"] = "x,y,z,qw,qx,qy,qz"
    poses.attrs["constant"] = False
    poses.attrs["time_alignment"] = "output_timestamps"
    poses.attrs["source"] = (
        "resampled_openaoe_slam_pose"
        if is_openaoe_sample(ego_mcap_path)
        else (
            "resampled_humantouch2_head_pose"
            if alignment.motion.source == "humantouch2"
            else (
                "resampled_humantouch2_real_robot_head_pose"
                if alignment.motion.source == "humantouch2_real_robot"
                else "resampled_egopro_head_pose"
            )
        )
    )
    validity = file.require_group("validity")
    validity.create_dataset("cam_head_pose", data=dynamic_valid)


def _utf8_bytes(value: object) -> np.ndarray:
    text = "" if value is None else str(value)
    return np.bytes_(text.encode("utf-8"))


def write_episode_identity_metadata(
    metadata: Any,
    episode_meta: dict[str, Any] | None,
) -> None:
    """Write capture date, original MCAP stem, and tactile-QC fields."""
    if not episode_meta:
        return
    qc = episode_meta.get("tactile_qc")
    if not isinstance(qc, dict):
        qc = {}
    usable = bool(episode_meta.get("tactile_usable"))
    swapped = episode_tactile_lr_swapped(episode_meta)
    palm_ok = palm_patch_usable_from_meta(episode_meta)
    metadata.create_dataset(
        "capture_date", data=_utf8_bytes(episode_meta.get("capture_date"))
    )
    metadata.create_dataset(
        "mcap_stem", data=_utf8_bytes(episode_meta.get("mcap_stem"))
    )
    metadata.create_dataset(
        "mcap_name", data=_utf8_bytes(episode_meta.get("mcap_name"))
    )
    metadata.create_dataset("task_id", data=_utf8_bytes(episode_meta.get("task_id")))
    metadata.create_dataset(
        "task_id_full", data=_utf8_bytes(episode_meta.get("task_id_full"))
    )
    metadata.create_dataset(
        "dataset_name", data=_utf8_bytes(episode_meta.get("dataset_name"))
    )
    metadata.create_dataset(
        "source_format", data=_utf8_bytes(episode_meta.get("source_format"))
    )
    metadata.create_dataset(
        "block_index", data=_utf8_bytes(episode_meta.get("block_index"))
    )
    metadata.create_dataset(
        "block_id", data=_utf8_bytes(episode_meta.get("block_id"))
    )
    metadata.create_dataset(
        "instruction_source",
        data=_utf8_bytes(episode_meta.get("instruction_source")),
    )
    metadata.create_dataset("tactile_usable", data=np.uint8(1 if usable else 0))
    metadata.create_dataset(
        "tactile_lr_swapped", data=np.uint8(1 if swapped else 0)
    )
    metadata.create_dataset(
        "palm_patch_usable", data=np.uint8(1 if palm_ok else 0)
    )
    metadata.create_dataset(
        "tactile_unusable_reason",
        data=_utf8_bytes(episode_meta.get("tactile_unusable_reason")),
    )
    metadata.create_dataset(
        "tactile_unusable_reason_label",
        data=_utf8_bytes(episode_meta.get("tactile_unusable_reason_label")),
    )
    payload = {
        "capture_date": episode_meta.get("capture_date"),
        "mcap_stem": episode_meta.get("mcap_stem"),
        "mcap_name": episode_meta.get("mcap_name"),
        "task_id": episode_meta.get("task_id"),
        "task_id_full": episode_meta.get("task_id_full"),
        "dataset_name": episode_meta.get("dataset_name"),
        "source_format": episode_meta.get("source_format"),
        "block_index": episode_meta.get("block_index"),
        "block_id": episode_meta.get("block_id"),
        "instruction_source": episode_meta.get("instruction_source"),
        "split": episode_meta.get("split"),
        "episode_index": episode_meta.get("episode_index"),
        "tactile_usable": usable,
        "tactile_lr_swapped": swapped,
        "palm_patch_usable": palm_ok,
        "tactile_unusable_reason": episode_meta.get("tactile_unusable_reason") or "",
        "tactile_unusable_reason_label": episode_meta.get(
            "tactile_unusable_reason_label"
        )
        or "",
        "tactile_qc": qc,
    }
    metadata.create_dataset(
        "humantouch_json",
        data=_utf8_bytes(json.dumps(payload, ensure_ascii=False, default=str)),
    )


def write_raw_instructions(
    file: Any,
    episode_meta: dict[str, Any] | None,
    text_dtype: Any,
) -> None:
    """Write ``raw/instructions`` only when this episode has review subtasks.

    Each element is ``[caption, start_frame_60, end_frame_60_exclusive]``.
    Episodes that only reused ``task_name`` via the task_id map get no dataset.
    """
    rows = resolve_raw_instructions(episode_meta)
    if not rows:
        return
    raw = file.require_group("raw")
    if "instructions" in raw:
        del raw["instructions"]
    dataset = raw.create_dataset(
        "instructions",
        data=json.dumps(rows, ensure_ascii=False),
        dtype=text_dtype,
    )
    dataset.attrs["item"] = "[caption, start_frame_60, end_frame_60_exclusive]"
    dataset.attrs["frame_rate"] = np.int64(60)
    dataset.attrs["frame_interval"] = "half_open"


def write_camera_aligned_spark_hdf5(
    output_path: str | Path,
    robot: UnifiedRobotMotion,
    alignment: CameraAlignmentResult,
    ego_mcap_path: str | Path,
    fps: float,
    action_steps: int = 1,
    tactile_context: EgoProTactileContext | None = None,
    mano_assets: str | Path | None = None,
    episode_meta: dict[str, Any] | None = None,
    allow_missing_jpeg: bool = False,
) -> None:
    """Write Spark-compatible state/action/vision plus explicit validity metadata.

    Episodes are truncated by ``action_steps`` (default 1 → length ``T-1``) so
    every retained frame has a finite next-state action. This matches the
    LeRobot next-state contract and avoids a trailing NaN action frame.
    """
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

    # Keep matching state/action slices; no unused padded action tail is built.
    timestamps = np.asarray(robot.timestamps[:keep], dtype=np.float64)
    state = np.asarray(robot.state[:keep], dtype=np.float64)
    state_valid = np.asarray(robot.state_valid[:keep], dtype=bool)
    action = np.asarray(robot.state[action_steps:], dtype=np.float64).copy()
    action_valid = np.asarray(robot.state_valid[action_steps:], dtype=bool)
    action[~action_valid] = np.nan
    retarget_error = np.asarray(robot.retarget_error[:keep], dtype=np.float64)
    aligned_joints = np.asarray(alignment.aligned_joints_world[:keep], dtype=np.float64)
    joint_confidence = np.asarray(alignment.motion.joint_confidence[:keep], dtype=np.float64)
    camera_from_world = np.asarray(alignment.camera_from_world[:keep], dtype=np.float64)
    camera_valid = np.asarray(alignment.camera_valid[:keep], dtype=bool)

    unavailable_arm = np.full((keep, 7), np.nan, dtype=np.float64)
    unavailable_arm_valid = np.zeros_like(unavailable_arm, dtype=bool)
    writer_started = time.perf_counter()

    with h5py.File(output, "w") as file:
        # Native Spark stores these as variable-length UTF-8 strings.
        text_dtype = h5py.string_dtype(encoding="utf-8")
        file.create_dataset("instruction", data=robot.instruction, dtype=text_dtype)
        file.create_dataset("data_format_version", data="v1.0", dtype=text_dtype)
        file.create_dataset("timestamps", data=timestamps)
        info = file.create_group("additional_info")
        info.create_dataset("frequency", data=np.int64(round(fps)))
        _write_cam_head_calibration(
            file,
            alignment,
            ego_mcap_path=ego_mcap_path,
            camera_from_world=camera_from_world,
            camera_valid=camera_valid,
        )

        state_group = file.create_group("state")
        action_group = file.create_group("action")
        validity = file.require_group("validity")
        validity.create_dataset("state_54", data=state_valid)
        for side, offset in (("left", 0), ("right", 27)):
            for name, valid_name, columns in (
                ("ee_poses", "ee_pose", slice(offset, offset + 7)),
                ("ee_joint_states", "ee_joint_states", slice(offset + 7, offset + 27)),
            ):
                state_group.create_dataset(f"{side}_{name}", data=state[:, columns])
                action_group.create_dataset(f"{side}_{name}", data=action[:, columns])
                validity.create_dataset(f"{side}_{valid_name}", data=state_valid[:, columns])
                validity.create_dataset(f"action_{side}_{valid_name}", data=action_valid[:, columns])
            # Native Spark v1 keeps absolute EE poses under this legacy key.
            state_group.create_dataset(
                f"{side}_delta_ee_poses", data=state[:, offset : offset + 7]
            )
            for group in (state_group, action_group):
                group.create_dataset(f"{side}_arm_joint_states", data=unavailable_arm)
            for prefix in ("", "action_"):
                validity.create_dataset(
                    f"{prefix}{side}_arm_joint_states", data=unavailable_arm_valid
                )

        aligned = file.create_group("aligned")
        aligned.create_dataset("joints_world", data=aligned_joints)
        aligned.create_dataset("joint_confidence", data=joint_confidence)
        aligned.create_dataset("retarget_error", data=retarget_error)

        video_started = time.perf_counter()
        source_fps = float(alignment.motion.metadata["source_fps"])
        staging_prefix = output.with_name(f".{output.name}.jpeg-stage")
        if is_openaoe_sample(ego_mcap_path) or is_egodex_episode(ego_mcap_path):
            image_validity = {
                "cam_head": _write_openaoe_head_video(
                    file,
                    ego_mcap_path,
                    timestamps,
                    source_fps=source_fps,
                    staging_path=staging_prefix.with_name(
                        f"{staging_prefix.name}.cam_head"
                    ),
                )
            }
            if is_openaoe_sample(ego_mcap_path):
                from ..retargeting.mano.params import write_openaoe_mano_hdf5

                write_openaoe_mano_hdf5(
                    file,
                    ego_mcap_path,
                    _output_source_frame_indices(
                        np.asarray(robot.timestamps, dtype=np.float64),
                        source_fps,
                    ),
                    action_steps=action_steps,
                )
        elif is_egoverse_episode(ego_mcap_path):
            image_validity = {
                "cam_head": _write_egoverse_head_video(
                    file,
                    ego_mcap_path,
                    timestamps,
                    source_fps=source_fps,
                    staging_path=staging_prefix.with_name(
                        f"{staging_prefix.name}.cam_head"
                    ),
                )
            }
        else:
            # ``retarget_camera_aligned_motion`` appends a provenance suffix
            # to ``robot.source``.  Select the writer from the base dataset,
            # otherwise canonical HumanTouch2 would incorrectly enter the
            # EgoPro JPEG path and real-robot policy would be obscured.
            source_dataset = base_source_dataset_name(robot.source)
            if source_dataset == "humantouch2":
                image_validity = _write_humantouch2_videos(
                    file,
                    ego_mcap_path,
                    timestamps,
                    source_fps=source_fps,
                    staging_prefix=staging_prefix,
                )
            else:
                image_validity = _write_egopro_videos(
                    file,
                    ego_mcap_path,
                    timestamps,
                    source_fps=source_fps,
                    staging_prefix=staging_prefix,
                    source_dataset=source_dataset,
                    allow_missing_jpeg=allow_missing_jpeg,
                )
            if tactile_context is None:
                raise ValueError(
                    "EgoPro/HumanTouch2 HDF5 write requires tactile context "
                    "for manus and MANO"
                )
            alignment_payload = robot.metadata.get("camera_alignment", {})
            if not isinstance(alignment_payload, dict) or not alignment_payload:
                raise ValueError(
                    "EgoPro tactile write requires camera_alignment metadata"
                )
            world_from_egopro = world_from_egopro_aligned(alignment_payload)
            write_egopro_tactile_hdf5(
                file,
                tactile_context,
                _output_source_frame_indices(timestamps, source_fps),
                cam_head_extrinsics=np.asarray(
                    file["vision/cam_head/extrinsics"], dtype=np.float64
                ),
                world_from_egopro=world_from_egopro,
                episode_meta=(
                    episode_meta
                    if episode_meta is not None
                    else robot.metadata.get("episode_meta")
                ),
            )
            from ..retargeting.mano.params import write_egopro_mano_hdf5

            write_egopro_mano_hdf5(
                file,
                tactile_context,
                _output_source_frame_indices(
                    np.asarray(robot.timestamps, dtype=np.float64),
                    source_fps,
                ),
                action_steps=action_steps,
                mano_assets=mano_assets,
                world_from_egopro=world_from_egopro,
                camera_from_world=np.asarray(
                    alignment.camera_from_world, dtype=np.float64
                ),
            )
        performance = robot.metadata.setdefault("performance", {})
        stage_s = performance.setdefault("stage_s", {})
        stage_s["video_hdf5"] = time.perf_counter() - video_started
        stage_s["hdf5_before_metadata"] = time.perf_counter() - writer_started
        for camera_name, image_valid in image_validity.items():
            validity.create_dataset(camera_name, data=image_valid)

        metadata = file.create_group("metadata")
        metadata.create_dataset("source", data=np.bytes_(robot.source.encode("utf-8")))
        resolved_meta = (
            episode_meta
            if episode_meta is not None
            else robot.metadata.get("episode_meta")
        )
        write_raw_instructions(file, resolved_meta, text_dtype)
        write_episode_identity_metadata(
            metadata,
            resolved_meta,
        )
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
                    robot.metadata.get("camera_alignment", {}),
                    ensure_ascii=False,
                    default=str,
                ).encode("utf-8")
            ),
        )
        provenance = dict(robot.metadata)
        provenance["episode_truncation"] = int(action_steps)
        provenance["source_frame_count"] = int(source_frames)
        provenance["output_frame_count"] = int(keep)
        provenance["action_semantics"] = (
            f"next_state_truncated:{action_steps}"
        )
        if resolved_meta:
            provenance["humantouch"] = resolved_meta
        provenance = _portable_metadata(provenance)
        metadata.create_dataset(
            "provenance_json",
            data=np.bytes_(
                json.dumps(provenance, ensure_ascii=False, default=str).encode("utf-8")
            ),
        )
