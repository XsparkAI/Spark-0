"""Head-camera tactile contact maps (``tactile.2d``) from stored MANO and taxels."""

from __future__ import annotations

import json
import multiprocessing
import os
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..retargeting.mano.model import resolve_mano_assets
from ..retargeting.mano.ik import cached_runtime, vertices_from_mano_params
from .proximity import (
    DEFAULT_PRESSURE_DEADBAND,
    PARTS,
    build_mapping,
    display_field,
    fingertip_vertex_masks,
    geodesic_table,
    rasterize_posed_mesh,
    vertex_field_from_tip_pressure,
)
from ..storage.writer_common import OUTPUT_IMAGE_HW
from .collector import (
    CHANNEL_COUNT,
    RAW_CHANNEL_COUNT,
    channel_ids,
    optical_from_world,
    world_from_egopro_aligned,
)
from ..camera_alignment import transform_points


OUTPUT_HW = OUTPUT_IMAGE_HW
UNITY_TO_OPENVR = np.diag([1.0, 1.0, -1.0])
UNITY_SOURCE_FRAME = "manus_core_unity_world"
CAMERA_OPTICAL_FRAMES = {"camera_opencv_upright_head", "openaoe_camera"}
IDENTITY_4 = np.eye(4, dtype=np.float64)


def pressure_to_mapping_order(
    pressure: np.ndarray,
    hand: str,
    mapping: Mapping[str, Any],
) -> np.ndarray:
    """Scatter stored 320-ch taxels onto the mapping's channel order."""
    pressure = np.asarray(pressure, dtype=np.float64)
    if pressure.ndim != 2 or pressure.shape[1] != CHANNEL_COUNT:
        raise ValueError(
            f"taxel pressure must have shape [T,{CHANNEL_COUNT}], got {pressure.shape}"
        )
    ids = channel_ids(hand) - 1
    mapping_ids = np.asarray(mapping["channel_ids"], dtype=np.int64) - 1
    full = np.zeros((len(pressure), RAW_CHANNEL_COUNT), dtype=np.float64)
    full[:, ids] = pressure
    return full[:, mapping_ids]


def _cached_hand_mapping(hand: str, mano_assets: Path | str | None = None) -> Any:
    assets = resolve_mano_assets(mano_assets)
    key = (hand, str(assets))
    mapping = _MAPPING_CACHE.get(key)
    if mapping is not None:
        return mapping
    with _CACHE_LOCK:
        mapping = _MAPPING_CACHE.get(key)
        if mapping is None:
            mapping = build_mapping(hand, 320, mano_assets=assets)
            palmar_finger, all_finger = fingertip_vertex_masks(
                mapping["mano"], hand, assets
            )
            palm_verts = (np.asarray(mapping["mano"].part) == "palm") & np.asarray(
                mapping["mano"].palmar, dtype=bool
            )
            glove_part = np.asarray(mapping["glove_part"])
            mapping["taxel_mask"] = np.ones(len(mapping["channel_ids"]), dtype=bool)
            mapping["tip_groups"] = {
                name: np.flatnonzero(glove_part == name) for name in PARTS
            }
            mapping["palmar_tip"] = palmar_finger | palm_verts
            mapping["all_tip"] = all_finger | palm_verts
            mapping["tip_weights"] = np.asarray(mapping["weights"], dtype=np.float64)
            mapping["_geodesic"] = geodesic_table(
                mapping["mano"].verts, mapping["mano"].faces
            )
            _MAPPING_CACHE[key] = mapping
        return mapping


def analyze_hand_from_pressure(
    pressure: np.ndarray,
    frames: np.ndarray,
    hand: str,
    *,
    mapping: Mapping[str, Any] | None = None,
    mano_assets: Path | str | None = None,
    pressure_deadband: float = DEFAULT_PRESSURE_DEADBAND,
) -> np.ndarray:
    """Return ``float64[T, 778]`` geodesic proximity including the palm."""
    mapping = mapping or _cached_hand_mapping(hand, mano_assets)
    frames = np.asarray(frames, dtype=np.int64)
    ordered = pressure_to_mapping_order(pressure, hand, mapping)
    if len(ordered) != len(frames):
        raise ValueError("pressure rows must match frame keys")
    return vertex_field_from_tip_pressure(
        ordered, mapping, pressure_deadband=pressure_deadband
    )


def _load_json_attr(value: Any) -> Any:
    if isinstance(value, (bytes, np.bytes_)):
        value = bytes(value).decode()
    if isinstance(value, str):
        return json.loads(value)
    return value


def ik_vertices_to_aligned_world(
    vertices: np.ndarray,
    *,
    source_coordinate_frame: str,
    world_from_egopro: np.ndarray,
) -> np.ndarray:
    """Map ManoRuntime vertices into the HDF5 camera-extrinsics world.

    IK targets are stored in OpenVR after the Unity ``diag(1,1,-1)`` flip.
    ``_mesh_from_retarget`` sends those vertices back to EgoPro/Unity before
    the head-camera projection; the same undo is required here, then the
    camera-alignment rigid map into the HDF5 extrinsics world.
    """
    points = np.asarray(vertices, dtype=np.float64)
    if source_coordinate_frame == UNITY_SOURCE_FRAME:
        points = points @ UNITY_TO_OPENVR.T
    return transform_points(world_from_egopro, points)


def vertices_to_hdf5_world(
    vertices: np.ndarray,
    *,
    source_coordinate_frame: str = "",
    world_from_egopro: np.ndarray | None = None,
) -> np.ndarray:
    """Return raw EgoPro/OpenVR vertices in the camera-extrinsics world.

    Stored MANO follows its declared coordinate frame.
    Camera-frame MANO bypasses this helper at the caller.
    """
    points = np.asarray(vertices, dtype=np.float64)
    if world_from_egopro is None:
        raise ValueError("stored MANO vertices require world_from_egopro alignment")
    return ik_vertices_to_aligned_world(
        points,
        source_coordinate_frame=source_coordinate_frame,
        world_from_egopro=world_from_egopro,
    )


def project_vertices_optical(
    vertices_world: np.ndarray,
    transform: np.ndarray,
    camera_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    vertices_world = np.asarray(vertices_world, dtype=np.float64)
    camera = vertices_world @ transform[:3, :3].T + transform[:3, 3]
    uv = np.full((len(camera), 2), np.nan, dtype=np.float64)
    depth = camera[:, 2].astype(np.float64)
    valid = np.isfinite(camera).all(1) & (depth > 0.03)
    if np.any(valid):
        image_points, _ = cv2.projectPoints(
            camera[valid].reshape(-1, 1, 3),
            np.zeros((3, 1), dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
            np.asarray(camera_matrix, dtype=np.float64),
            None,
        )
        uv[valid] = image_points.reshape(-1, 2)
    return uv, depth


def _load_stored_mano(
    file: Any, start: int, stop: int
) -> dict[str, dict[str, np.ndarray]] | None:
    if "mano/state/left_ee_poses" not in file:
        return None
    length = stop - start
    stored: dict[str, dict[str, np.ndarray]] = {}
    for hand in ("left", "right"):
        poses = np.asarray(
            file[f"mano/state/{hand}_ee_poses"][start:stop], dtype=np.float64
        )
        joints = np.asarray(
            file[f"mano/state/{hand}_ee_joint_states"][start:stop], dtype=np.float64
        )
        beta_key = f"mano/state/{hand}_hand_betas"
        if beta_key in file:
            betas = np.asarray(file[beta_key][start:stop], dtype=np.float64)
        else:
            betas = np.zeros((length, 10), dtype=np.float64)
        scale = np.asarray(
            file["mano/state"].attrs.get(f"{hand}_scale", []), dtype=np.float64
        ).reshape(-1)
        if scale.size == 0:
            scale = np.ones(length, dtype=np.float64)
        elif scale.size != length:
            scale = scale[start:stop]
        if (
            poses.shape != (length, 7)
            or joints.shape != (length, 45)
            or betas.shape != (length, 10)
        ):
            raise ValueError(
                f"mano/state/{hand}_ee_* has poses {poses.shape} joints "
                f"{joints.shape} betas {betas.shape}, expected ({length}, 7), "
                f"({length}, 45), ({length}, 10)"
            )
        stored[hand] = {
            "poses": poses,
            "joints": joints,
            "betas": betas,
            "scale": scale,
        }
    return stored


_CACHE_LOCK = threading.Lock()
_MAPPING_CACHE: dict[tuple[str, str], Any] = {}


def _process_pool_safe() -> bool:
    """Fork/spawn while other threads exist (convert pool, image writer) can deadlock."""
    if threading.current_thread() is not threading.main_thread():
        return False
    return threading.active_count() <= 1


def _resolve_device(explicit: str | None) -> str:
    raw = (explicit or os.environ.get("TACTILE_2D_DEVICE") or "cpu").strip()
    if raw == "auto":
        return "cpu"
    return raw or "cpu"


def _resolve_workers(explicit: int | None) -> int:
    if explicit is not None:
        return max(1, int(explicit))
    raw = os.environ.get("TACTILE_2D_WORKERS", "").strip()
    if raw:
        return max(1, int(raw))
    return 1


def _chunk_bounds(length: int, workers: int) -> list[tuple[int, int]]:
    workers = min(max(1, workers), max(1, length))
    size, extra = divmod(length, workers)
    bounds: list[tuple[int, int]] = []
    start = 0
    for index in range(workers):
        count = size + (1 if index < extra else 0)
        stop = start + count
        if count > 0:
            bounds.append((start, stop))
        start = stop
    return bounds


def _chunk_worker(payload: dict[str, Any]) -> np.ndarray:
    return episode_proximity_maps(
        payload["path"],
        output_hw=tuple(payload["output_hw"]),
        mano_assets=payload["mano_assets"],
        device=payload["device"],
        start_index=int(payload["start"]),
        max_frames=int(payload["count"]),
        workers=1,
        pressure_deadband=float(
            payload.get("pressure_deadband", DEFAULT_PRESSURE_DEADBAND)
        ),
    )


def episode_proximity_maps(
    hdf5_path: str | Path,
    *,
    output_hw: tuple[int, int] = OUTPUT_HW,
    mano_assets: Path | str | None = None,
    device: str | None = None,
    start_index: int = 0,
    max_frames: int | None = None,
    workers: int | None = None,
    pressure_deadband: float = DEFAULT_PRESSURE_DEADBAND,
) -> np.ndarray:
    """Return ``float32[T, H, W]`` proximity fields in ``[0, 1]``.

    Reuse the stored ``mano/state`` produced by camera-aligned retargeting.

    pressure_deadband is in raw taxel units and is applied independently
    to every taxel before the level/rise contact gate.
    """
    import h5py

    hdf5_path = Path(hdf5_path)
    device_name = _resolve_device(device)
    worker_count = _resolve_workers(workers)
    if worker_count > 1 and not _process_pool_safe():
        print(
            f"proximity 2D workers={worker_count} disabled "
            "(already in a thread or other threads live); using 1",
            flush=True,
        )
        worker_count = 1
    height, width = int(output_hw[0]), int(output_hw[1])
    assets = resolve_mano_assets(mano_assets)
    start = max(0, int(start_index))

    if worker_count > 1:
        with h5py.File(hdf5_path, "r") as file:
            total = int(
                np.asarray(
                    file["tactile/taxel_pressure_3D"].attrs["source_frame_indices"]
                ).shape[0]
            )
        stop = total if max_frames is None else min(total, start + max(0, int(max_frames)))
        length = max(0, stop - start)
        bounds = _chunk_bounds(length, worker_count)
        if len(bounds) > 1:
            print(
                f"proximity 2D device={device_name} "
                f"workers={len(bounds)} frames={length} chunked",
                flush=True,
            )
            payloads = [
                {
                    "path": str(hdf5_path),
                    "output_hw": [height, width],
                    "mano_assets": str(assets),
                    "device": device_name,
                    "start": start + chunk_start,
                    "count": chunk_stop - chunk_start,
                    "pressure_deadband": float(pressure_deadband),
                }
                for chunk_start, chunk_stop in bounds
            ]
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=len(payloads), mp_context=ctx
            ) as pool:
                parts = list(pool.map(_chunk_worker, payloads))
            return np.concatenate(parts, axis=0)

    runtime = cached_runtime(mano_assets=assets, device=device_name)
    mappings = {
        hand: _cached_hand_mapping(hand, assets)
        for hand in ("left", "right")
    }

    with h5py.File(hdf5_path, "r") as file:
        source_frames = np.asarray(
            file["tactile/taxel_pressure_3D"].attrs["source_frame_indices"],
            dtype=np.int64,
        )
        fx, fy, cx, cy = np.asarray(
            file["vision/cam_head/intrinsics"][()], dtype=np.float64
        ).reshape(4)
        camera_matrix = np.asarray(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        extrinsics = np.asarray(
            file["vision/cam_head/extrinsics"][()], dtype=np.float64
        )
        stop = len(source_frames) if max_frames is None else min(
            len(source_frames), start + max(0, int(max_frames))
        )
        source_frames = source_frames[start:stop]
        extrinsics = extrinsics[start:stop]
        stored_mano = _load_stored_mano(file, start, stop)
        if stored_mano is None:
            raise ValueError(
                f"{hdf5_path} has no mano/state; re-run retarget-camera-aligned "
                "so LeRobot 2D can reuse stored MANO."
            )
        mano_coordinate_frame = str(
            file["mano"].attrs.get("coordinate_frame", "")
        )
        stored_camera_optical = mano_coordinate_frame in CAMERA_OPTICAL_FRAMES
        world_from_egopro = None
        source_coordinate_frame = ""
        if not stored_camera_optical:
            alignment = _load_json_attr(file["metadata/camera_alignment_json"][()])
            if not isinstance(alignment, Mapping):
                raise ValueError("metadata/camera_alignment_json must be an object")
            world_from_egopro = world_from_egopro_aligned(alignment)
            source_coordinate_frame = str(
                file["mano"].attrs.get("source_coordinate_frame", "")
            )
        vertex_fields: dict[str, np.ndarray] = {}
        for hand in ("left", "right"):
            pressure = np.asarray(
                file[f"tactile/taxel_pressure_3D/{hand}_ee"][start:stop, :, 3],
                dtype=np.float64,
            )
            vertex_fields[hand] = analyze_hand_from_pressure(
                pressure,
                np.arange(len(pressure), dtype=np.int64),
                hand,
                mapping=mappings[hand],
                mano_assets=assets,
                pressure_deadband=pressure_deadband,
            ).astype(np.float32)

    print(
        f"proximity 2D device={device_name} mesh=stored MANO "
        f"backend={next(iter(runtime.backends.values())).device}",
        flush=True,
    )

    maps = np.zeros((len(source_frames), height, width), dtype=np.float32)
    faces = {
        hand: np.asarray(runtime.backends[hand].faces, dtype=np.int64)
        for hand in ("left", "right")
    }
    for index in range(len(source_frames)):
        if index == 0 or (index + 1) % 40 == 0 or index + 1 == len(source_frames):
            print(
                f"proximity 2D {index + 1}/{len(source_frames)}",
                flush=True,
            )
        values = {}
        active_hands: list[str] = []
        for hand in ("left", "right"):
            field_1d = vertex_fields[hand][index]
            values[hand] = field_1d
            if bool(field_1d.max() > 0):
                active_hands.append(hand)
        if not active_hands:
            continue
        transform = (
            IDENTITY_4
            if stored_camera_optical
            else optical_from_world(extrinsics[index])
        )
        if transform is None:
            continue
        field = np.zeros((height, width), dtype=np.float32)
        for hand in active_hands:
            vertices_raw = vertices_from_mano_params(
                runtime.backends[hand],
                stored_mano[hand]["poses"][index],
                stored_mano[hand]["joints"][index],
                float(stored_mano[hand]["scale"][index]),
                stored_mano[hand]["betas"][index],
            )
            if vertices_raw is None:
                continue
            vertices = (
                np.asarray(vertices_raw, dtype=np.float64)
                if stored_camera_optical
                else vertices_to_hdf5_world(
                    vertices_raw,
                    source_coordinate_frame=source_coordinate_frame,
                    world_from_egopro=world_from_egopro,
                )
            )
            uv, depth = project_vertices_optical(vertices, transform, camera_matrix)
            np.maximum(
                field,
                rasterize_posed_mesh(
                    uv, depth, faces[hand], values[hand], height, width
                ),
                out=field,
            )
        maps[index] = display_field(field)
    return maps
