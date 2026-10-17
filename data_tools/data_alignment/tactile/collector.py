"""EgoPro tactile extraction: camera-frame 320 taxels + 6-D region history.

Layout matches the 320 mapping in ``mano_mapping.py``:

    palm    rows 0-14, 16 active columns     240 taxels
    fingers rows 19-22, 4x4 per finger        80 taxels

Stored order is always palm, then thumb, index, middle, ring, pinky.
Source data is 60 Hz; the HDF5 output is downsampled to the output frame
grid (30 Hz by default) by sampling the nearest 60 Hz source frame.

``taxel_pressure_3D`` xyz is OpenCV optical (x right, y down, z forward).
Taxel centers are lifted from EgoPro/Manus world into the HDF5 extrinsics
world with the camera-alignment rigid map, then
``optical_from_world(vision/cam_head/extrinsics[t])`` — the same Spark
``camera_to_world`` poses used by MANO overlay and proximity 2D.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ..readers import egopro_frame_key
from .geometry import skeleton_joints, taxel_world_points
from ..camera_alignment import invert_transform, pose_wxyz_to_matrix, transform_points

RAW_CHANNEL_COUNT = 460
CHANNEL_COUNT = 320
MOXIAN_COLS = 20
PALM_LAST_ROW = 14
PALM_COLUMNS = {"left": (4, 19), "right": (0, 15)}
FINGER_ROWS = (19, 22)
FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
REGION_ORDER = ("palm", *FINGER_ORDER)
HISTORY_60 = 50
_GZIP = dict(compression="gzip", compression_opts=4, shuffle=True)
REGION_SLICES = {
    "palm": (0, 240),
    "thumb": (240, 256),
    "index": (256, 272),
    "middle": (272, 288),
    "ring": (288, 304),
    "pinky": (304, 320),
}
UNITY_TO_OPENVR = np.diag([1.0, 1.0, -1.0])
# Spark ``camera_to_world`` (x right, y up, z back) -> OpenCV optical.
# Self-inverse; keep local to avoid importing spark_hdf5 (circular).
_OPTICAL_FROM_SPARK_CAMERA = np.diag([1.0, -1.0, -1.0, 1.0])
MANO_JOINT_ORDER = (
    "wrist",
    "thumb_mcp",
    "thumb_pip",
    "thumb_dip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)
MANO_JOINT_TO_SKELETON = {
    "wrist": "hand",
    "thumb_mcp": "thumb_metacarpal",
    "thumb_pip": "thumb_proximal",
    "thumb_dip": "thumb_distal",
    "thumb_tip": "thumb_tip",
    "index_mcp": "index_proximal",
    "index_pip": "index_intermediate",
    "index_dip": "index_distal",
    "index_tip": "index_tip",
    "middle_mcp": "middle_proximal",
    "middle_pip": "middle_intermediate",
    "middle_dip": "middle_distal",
    "middle_tip": "middle_tip",
    "ring_mcp": "ring_proximal",
    "ring_pip": "ring_intermediate",
    "ring_dip": "ring_distal",
    "ring_tip": "ring_tip",
    "pinky_mcp": "pinky_proximal",
    "pinky_pip": "pinky_intermediate",
    "pinky_dip": "pinky_distal",
    "pinky_tip": "pinky_tip",
}

_FINGER_COLUMNS = {
    "left": {
        "thumb": (0, 3),
        "index": (4, 7),
        "middle": (8, 11),
        "ring": (12, 15),
        "pinky": (16, 19),
    },
    "right": {
        "pinky": (0, 3),
        "ring": (4, 7),
        "middle": (8, 11),
        "index": (12, 15),
        "thumb": (16, 19),
    },
}


def channel_ids(hand: str) -> np.ndarray:
    """Return 320 one-based Moxian ids: palm, then five fingers."""
    if hand not in {"left", "right"}:
        raise ValueError("hand must be left or right")
    palm0, palm1 = PALM_COLUMNS[hand]
    ids = [
        row * MOXIAN_COLS + col + 1
        for row in range(PALM_LAST_ROW, -1, -1)
        for col in range(palm0, palm1 + 1)
    ]
    for finger in FINGER_ORDER:
        start, stop = _FINGER_COLUMNS[hand][finger]
        ids.extend(
            row * MOXIAN_COLS + col + 1
            for row in range(FINGER_ROWS[1], FINGER_ROWS[0] - 1, -1)
            for col in range(start, stop + 1)
        )
    result = np.asarray(ids, dtype=np.int64)
    if result.shape != (CHANNEL_COUNT,):
        raise AssertionError(f"unexpected channel map shape {result.shape}")
    return result


def region_slice_meta() -> dict[str, list[int]]:
    """Inclusive-start exclusive-end index ranges in stored 320 order."""
    return {name: [start, stop] for name, (start, stop) in REGION_SLICES.items()}


def _write_region_attrs(group: Any, **extra: Any) -> None:
    group.attrs["channel_count"] = CHANNEL_COUNT
    group.attrs["region_order"] = np.asarray(REGION_ORDER, dtype="S16")
    group.attrs["region_slices"] = json.dumps(region_slice_meta())
    for key, value in extra.items():
        group.attrs[key] = value


def optical_from_world(extrinsics_wxyz: np.ndarray) -> np.ndarray | None:
    """Spark ``camera_to_world`` pose -> OpenCV optical-from-world 4x4."""
    pose = np.asarray(extrinsics_wxyz, dtype=np.float64).reshape(-1)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        return None
    return _OPTICAL_FROM_SPARK_CAMERA @ invert_transform(pose_wxyz_to_matrix(pose))


def world_from_egopro_aligned(alignment: Mapping[str, Any]) -> np.ndarray:
    """EgoPro/Unity world -> HDF5 extrinsics world.

    ``world_from_virtual_ego_optical`` is the optical pose written as the fixed
    HDF5 ``extrinsics``, so the map matches the stored extrinsics world of the
    episode.
    """
    world_from_virtual_optical = np.asarray(
        alignment["world_from_virtual_ego_optical"], dtype=np.float64
    )
    reference_camera_from_world = np.asarray(
        alignment["reference_camera_from_world"], dtype=np.float64
    )
    if world_from_virtual_optical.shape != (4, 4):
        raise ValueError(
            "world_from_virtual_ego_optical must be 4x4, "
            f"got {world_from_virtual_optical.shape}"
        )
    if reference_camera_from_world.shape != (4, 4):
        raise ValueError(
            "reference_camera_from_world must be 4x4, "
            f"got {reference_camera_from_world.shape}"
        )
    return world_from_virtual_optical @ reference_camera_from_world


@dataclass
class EgoProTactileContext:
    skeleton: dict[str, dict[int, dict[str, Any]]]
    wrist: dict[str, dict[int, dict[str, Any]]]
    head: dict[int, dict[str, Any]]
    raw: dict[str, dict[int, dict[str, Any]]]
    patches: dict[int, dict[str, dict[str, float]]]
    calibration: dict[str, Any]
    baseline: dict[str, np.ndarray]
    coordinate_frame: str = ""
    tactile_lr_swapped: bool = False


def episode_tactile_lr_swapped(episode_meta: Mapping[str, Any] | None) -> bool:
    """Return True when QC marked this MCAP's left/right tactile topics swapped."""
    if not isinstance(episode_meta, Mapping):
        return False
    if bool(episode_meta.get("tactile_lr_swapped")):
        return True
    qc = episode_meta.get("tactile_qc")
    return isinstance(qc, Mapping) and bool(qc.get("tactile_lr_swapped"))


def palm_patch_usable_from_meta(episode_meta: Mapping[str, Any] | None) -> bool:
    """Palm patches are unusable after an L/R tactile topic swap.

    Left/right palm molds occupy different columns (left 4–19, right 0–15).
    Swapping the raw streams aligns fingertips with the skeleton, but palm
    taxels and ``palm`` patch Newtons stay on the wrong mold.
    """
    if episode_tactile_lr_swapped(episode_meta):
        return False
    if isinstance(episode_meta, Mapping) and "tactile_usable" in episode_meta:
        return bool(episode_meta.get("tactile_usable"))
    return True


def swap_egopro_tactile_hands(context: EgoProTactileContext) -> None:
    """Swap left/right tactile raw streams and patch scalars in place.

    Does not touch skeleton, wrist, video, or point mapping. Marks the
    context so a second call is a no-op.
    """
    if context.tactile_lr_swapped:
        return
    context.raw["left"], context.raw["right"] = context.raw["right"], context.raw["left"]
    context.baseline["left"], context.baseline["right"] = (
        context.baseline["right"],
        context.baseline["left"],
    )
    for record in context.patches.values():
        if not isinstance(record, dict):
            continue
        record["left"], record["right"] = (
            record.get("right") or {},
            record.get("left") or {},
        )
    for side in ("left", "right"):
        for payload in context.raw[side].values():
            if isinstance(payload, dict):
                payload["hand"] = side
    context.tactile_lr_swapped = True


def maybe_swap_egopro_tactile_hands(
    context: EgoProTactileContext | None,
    episode_meta: Mapping[str, Any] | None,
) -> bool:
    """Swap tactile L/R when QC says the MCAP topics were reversed."""
    if context is None or not episode_tactile_lr_swapped(episode_meta):
        return bool(context.tactile_lr_swapped) if context is not None else False
    swap_egopro_tactile_hands(context)
    return True


def _raw_values(raw_row: Mapping[str, Any] | None) -> np.ndarray | None:
    """Return one finite 460-channel Moxian sample (``values`` or ``pressureData``)."""
    if raw_row is None or not bool(raw_row.get("valid", True)):
        return None
    values = np.asarray(
        raw_row.get("values", raw_row.get("pressureData", [])),
        dtype=np.float32,
    ).reshape(-1)
    if values.shape != (RAW_CHANNEL_COUNT,):
        raise ValueError(
            f"Valid tactile sample must contain {RAW_CHANNEL_COUNT} values, "
            f"got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Valid tactile sample contains NaN/Inf")
    return values


@dataclass
class EgoProTactileCollector:
    """Mutable sink filled by the EgoPro MCAP JSON scan."""

    raw: dict[str, dict[int, dict[str, Any]]] = field(
        default_factory=lambda: {"left": {}, "right": {}}
    )
    patches: dict[int, dict[str, dict[str, float]]] = field(
        default_factory=dict
    )
    context: EgoProTactileContext | None = None

    def add(self, topic: str, payload: dict[str, Any]) -> None:
        if topic in {
            "/human/tactile/left_patch_force",
            "/human/tactile/right_patch_force",
        }:
            frame = egopro_frame_key(payload)
            body = payload.get("payload")
            body = body if isinstance(body, Mapping) else {}
            side = str(body.get("side") or ("left" if "/left_" in topic else "right"))
            aliases = {
                "palm_center": "palm",
                "thumb_distal": "thumb",
                "index_distal": "index",
                "middle_distal": "middle",
                "ring_distal": "ring",
                "little_distal": "pinky",
                "pinky_distal": "pinky",
            }
            regions = {name: 0.0 for name in REGION_ORDER}
            for item in body.get("patches") or []:
                if not isinstance(item, Mapping):
                    continue
                region = aliases.get(str(item.get("region_name") or ""))
                if region is None:
                    continue
                regions[region] = max(float(item.get("normal_force_n", 0.0)), 0.0)
            self.patches.setdefault(frame, {})[side] = regions
            return
        if topic in {
            "/human/tactile/left_raw",
            "/human/tactile/right_raw",
        }:
            side = "left" if "/left_" in topic else "right"
            normalized = dict(payload)
            normalized["valid"] = bool(payload.get("is_valid", True))
            self.raw[side][egopro_frame_key(payload)] = normalized
            return
        if topic == "/tactile/patch_pressure":
            self.patches[egopro_frame_key(payload)] = {
                side: {
                    region: max(float(payload.get(f"{side}_{region}", 0.0)), 0.0)
                    for region in REGION_ORDER
                }
                for side in ("left", "right")
            }
            return
        if topic.startswith("/tactile/raw/"):
            side = topic.rsplit("/", 1)[-1]
            self.raw[side][egopro_frame_key(payload)] = payload

    def finalize(
        self,
        *,
        skeleton: dict[str, dict[int, dict[str, Any]]],
        wrist: dict[str, dict[int, dict[str, Any]]],
        head: dict[int, dict[str, Any]],
        calibration: dict[str, Any],
    ) -> EgoProTactileContext | None:
        if not any(self.raw[side] for side in ("left", "right")):
            self.context = None
            return None
        baseline: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            first_valid = None
            for frame in sorted(self.raw[side]):
                values = _raw_values(self.raw[side][frame])
                if values is not None and first_valid is None:
                    first_valid = values
            baseline[side] = (
                first_valid
                if first_valid is not None
                else np.zeros(RAW_CHANNEL_COUNT, dtype=np.float32)
            )
        coordinate_frame = ""
        for side in ("left", "right"):
            for payload in skeleton.get(side, {}).values():
                if isinstance(payload, Mapping):
                    coordinate_frame = str(
                        payload.get("coordinate_frame")
                        or payload.get("coordinateFrame")
                        or ""
                    )
                    if coordinate_frame:
                        break
            if coordinate_frame:
                break
        self.context = EgoProTactileContext(
            skeleton=skeleton,
            wrist=wrist,
            head=head,
            raw=self.raw,
            patches=self.patches,
            calibration=calibration,
            baseline=baseline,
            coordinate_frame=coordinate_frame,
        )
        return self.context


def _raw_pressure(raw_row: Mapping[str, Any], baseline: np.ndarray) -> np.ndarray:
    values = np.asarray(
        raw_row.get("values", raw_row.get("pressureData", [])),
        dtype=np.float32,
    ).reshape(-1)
    padded = np.zeros(RAW_CHANNEL_COUNT, dtype=np.float32)
    padded[: min(len(values), RAW_CHANNEL_COUNT)] = values[:RAW_CHANNEL_COUNT]
    return np.maximum(padded - baseline, 0.0)


def _exact_taxel_points(
    hand: str,
    joints: Mapping[str, np.ndarray],
    wrist_tracker_position: Sequence[float] | None,
    selected_ids: np.ndarray,
) -> np.ndarray:
    """Return the 320 taxel centers in the shared Manus world frame."""
    world = taxel_world_points(
        hand,
        joints,
        wrist_tracker_position=wrist_tracker_position,
    )
    selected = np.asarray(world[selected_ids], dtype=np.float64)
    if selected.shape != (CHANNEL_COUNT, 3):
        raise ValueError(
            f"taxel world points must have shape ({CHANNEL_COUNT}, 3), "
            f"got {selected.shape}"
        )
    missing = ~np.isfinite(selected).all(axis=1)
    if missing.any():
        good = np.flatnonzero(~missing)
        if good.size == 0:
            raise ValueError("no finite taxel world points")
        filled = selected.copy()
        for index in np.flatnonzero(missing):
            filled[index] = filled[good[np.argmin(np.abs(good - index))]]
        selected = filled
    return selected


def _frame_surface(
    context: EgoProTactileContext,
    side: str,
    frame: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(world[320, 3], pressure[320])`` for one 60 Hz source frame."""
    skeleton_row = context.skeleton[side].get(frame)
    raw_row = context.raw[side].get(frame)
    if skeleton_row is None or raw_row is None:
        return None
    joints = skeleton_joints(skeleton_row, side)
    if "hand" not in joints:
        return None
    pressure = _raw_pressure(raw_row, context.baseline[side])
    wrist_row = context.wrist[side].get(frame) or {}
    tracker_position = (
        wrist_row.get("position_m")
        or wrist_row.get("position")
        or wrist_row.get("pos")
    )
    ids = channel_ids(side) - 1
    world = _exact_taxel_points(side, joints, tracker_position, ids)
    return world, pressure[ids]


def camera_taxel_320(
    context: EgoProTactileContext,
    side: str,
    source_frames: Sequence[int],
    *,
    cam_head_extrinsics: np.ndarray,
    world_from_egopro: np.ndarray,
) -> np.ndarray:
    """Return camera-frame ``float32[T, 320, 4]`` (xyz plus pressure).

    ``cam_head_extrinsics`` is the stored Spark ``vision/cam_head/extrinsics``
    ``(T, 7)`` ``camera_to_world`` (x right, y up, z back). Taxel centers from the Manus surface stay in
    EgoPro world, then:

    ``optical = optical_from_world(extrinsics[t]) @ world_from_egopro @ p_egopro``
    """
    source_indices = np.asarray(source_frames, dtype=np.int64)
    extrinsics = np.asarray(cam_head_extrinsics, dtype=np.float64)
    world_map = np.asarray(world_from_egopro, dtype=np.float64)
    if extrinsics.shape != (len(source_indices), 7):
        raise ValueError(
            "cam_head_extrinsics must have shape "
            f"{(len(source_indices), 7)}, got {extrinsics.shape}"
        )
    if world_map.shape != (4, 4):
        raise ValueError(f"world_from_egopro must be 4x4, got {world_map.shape}")
    result = np.full(
        (len(source_indices), CHANNEL_COUNT, 4), np.nan, dtype=np.float32
    )
    for output_index, frame in enumerate(source_indices):
        surface = _frame_surface(context, side, int(frame))
        if surface is None:
            continue
        world, pressure = surface
        spark = transform_points(world_map, world)
        transform = optical_from_world(extrinsics[output_index])
        if transform is not None:
            result[output_index, :, :3] = (
                spark @ transform[:3, :3].T + transform[:3, 3]
            ).astype(np.float32)
        result[output_index, :, 3] = pressure
    return result


def region_pressure_history(
    context: EgoProTactileContext,
    side: str,
    source_frames: Sequence[int],
) -> np.ndarray:
    """Return ``float32[T, HISTORY_60, 6]`` palm+finger patch-force history.

    History is stored most-recent-first: index 0 is the patch pressure at the
    output frame and index ``HISTORY_60 - 1`` is 49 frames earlier (timestamp
    descending). Missing frames before the first available patch are zero;
    gaps between samples are forward-filled with the most recent patch.
    """
    source_indices = np.asarray(source_frames, dtype=np.int64)
    patch_frames = sorted(context.patches)
    last_patch = len(patch_frames) - 1

    def patch_at(frame: int) -> np.ndarray:
        if not patch_frames:
            return np.zeros(len(REGION_ORDER), dtype=np.float32)
        position = np.searchsorted(patch_frames, int(frame), side="right") - 1
        position = min(max(position, 0), last_patch)
        record = context.patches[patch_frames[position]].get(side, {})
        return np.asarray(
            [record.get(region, 0.0) for region in REGION_ORDER],
            dtype=np.float32,
        )

    result = np.zeros(
        (len(source_indices), HISTORY_60, len(REGION_ORDER)),
        dtype=np.float32,
    )
    for output_index, frame in enumerate(source_indices):
        frame = int(frame)
        for history_index in range(HISTORY_60):
            result[output_index, history_index] = patch_at(frame - history_index)
    return result


def _context_coordinate_frame(context: EgoProTactileContext) -> str:
    if context.coordinate_frame:
        return context.coordinate_frame
    for side in ("left", "right"):
        for payload in context.skeleton.get(side, {}).values():
            if isinstance(payload, Mapping):
                value = payload.get("coordinate_frame") or payload.get("coordinateFrame")
                if value:
                    return str(value)
    return ""


def manus_joints21(
    context: EgoProTactileContext,
    side: str,
    source_frames: Sequence[int],
) -> np.ndarray:
    """Return world-frame ``float32[T, 21, 3]`` in ManoRuntime joint order."""
    source_indices = np.asarray(source_frames, dtype=np.int64)
    result = np.full((len(source_indices), 21, 3), np.nan, dtype=np.float32)
    flip_unity = _context_coordinate_frame(context) == "manus_core_unity_world"
    for output_index, frame in enumerate(source_indices):
        skeleton_row = context.skeleton[side].get(int(frame))
        if skeleton_row is None:
            continue
        joints = skeleton_joints(skeleton_row, side)
        points = []
        for joint in MANO_JOINT_ORDER:
            point = joints.get(MANO_JOINT_TO_SKELETON[joint])
            if point is None or not np.all(np.isfinite(point)):
                points = []
                break
            points.append(np.asarray(point, dtype=np.float64).reshape(3))
        if len(points) != 21:
            continue
        stacked = np.stack(points, axis=0)
        if flip_unity:
            stacked = stacked @ UNITY_TO_OPENVR.T
        result[output_index] = stacked.astype(np.float32)
    return result


def finger_pressure_60hz(
    context: EgoProTactileContext,
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(frame_keys[N], float32[N, 320])`` baseline-subtracted taxels."""
    frames = np.asarray(sorted(context.raw[side]), dtype=np.int64)
    ids = channel_ids(side) - 1
    values = np.zeros((len(frames), CHANNEL_COUNT), dtype=np.float32)
    for index, frame in enumerate(frames):
        raw_row = context.raw[side][int(frame)]
        if _raw_values(raw_row) is None:
            continue
        values[index] = _raw_pressure(raw_row, context.baseline[side])[ids]
    return frames, values


def tactile_raw_460_60hz(
    context: EgoProTactileContext,
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(frame_keys[N], float32[N, 460])`` original Moxian ADC."""
    frames = np.asarray(sorted(context.raw[side]), dtype=np.int64)
    values = np.zeros((len(frames), RAW_CHANNEL_COUNT), dtype=np.float32)
    for index, frame in enumerate(frames):
        raw = _raw_values(context.raw[side][int(frame)])
        if raw is not None:
            values[index] = raw
    return frames, values


def tactile_patch_60hz(
    context: EgoProTactileContext,
    side: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(frame_keys[N], float32[N, 6])`` patch Newtons at native 60 Hz."""
    frames = np.asarray(sorted(context.patches), dtype=np.int64)
    values = np.zeros((len(frames), len(REGION_ORDER)), dtype=np.float32)
    for index, frame in enumerate(frames):
        record = (context.patches[int(frame)] or {}).get(side) or {}
        values[index] = np.asarray(
            [float(record.get(region, 0.0)) for region in REGION_ORDER],
            dtype=np.float32,
        )
    return frames, values


def write_egopro_tactile_hdf5(
    file: Any,
    context: EgoProTactileContext,
    source_frames: np.ndarray,
    *,
    cam_head_extrinsics: np.ndarray,
    world_from_egopro: np.ndarray,
    episode_meta: Mapping[str, Any] | None = None,
) -> None:
    """Stream 30 Hz camera-frame taxels, patch history, and MANO targets.

    Taxel xyz uses the calibrated Spark camera extrinsics and EgoPro→Spark
    world map supplied by the episode writer.
    """
    swapped = maybe_swap_egopro_tactile_hands(context, episode_meta)
    palm_ok = palm_patch_usable_from_meta(episode_meta) and not swapped
    source_indices = np.asarray(source_frames, dtype=np.int64)
    # HDF5 stores attributes in the owning object's header.  Long EgoPro
    # episodes exceed that header's ~64 KiB limit when this per-frame mapping
    # is kept as int64.  Source frame indices are small non-negative integers,
    # so int32 preserves the mapping while halving both attribute payloads.
    int32 = np.iinfo(np.int32)
    if source_indices.size and (
        int(source_indices.min()) < int32.min
        or int(source_indices.max()) > int32.max
    ):
        raise ValueError("source frame indices do not fit in int32")
    source_indices_attr = source_indices.astype(np.int32, copy=False)
    frame_count = len(source_indices)
    extrinsics = np.asarray(cam_head_extrinsics, dtype=np.float64)
    world_map = np.asarray(world_from_egopro, dtype=np.float64)
    if extrinsics.shape != (frame_count, 7):
        raise ValueError(f"cam_head extrinsics must have shape {(frame_count, 7)}, got {extrinsics.shape}")
    if world_map.shape != (4, 4):
        raise ValueError(f"world_from_egopro must be 4x4, got {world_map.shape}")
    tactile = file.require_group("tactile")
    coordinate_frame = _context_coordinate_frame(context)
    stored_manus_frame = (
        "openvr_raw_uncalibrated_meters"
        if coordinate_frame == "manus_core_unity_world"
        else (coordinate_frame or "egopro_world")
    )

    taxel = tactile.create_group("taxel_pressure_3D")
    for side in ("left", "right"):
        taxel.create_dataset(
            f"{side}_ee",
            shape=(frame_count, CHANNEL_COUNT, 4),
            dtype=np.float32,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
    taxel.create_dataset("channel_ids_left", data=channel_ids("left"))
    taxel.create_dataset("channel_ids_right", data=channel_ids("right"))
    _write_region_attrs(
        taxel,
        fields="x,y,z,pressure",
        coordinate_frame="camera_opencv_upright_head",
        extrinsics_source="vision/cam_head/extrinsics",
        source_fps=60.0,
        source_frame_indices=source_indices_attr,
        layout="palm_then_thumb_index_middle_ring_pinky",
    )
    taxel.attrs["palm_patch_usable"] = np.uint8(1 if palm_ok else 0)
    taxel.attrs["tactile_lr_swapped"] = np.uint8(1 if swapped else 0)

    patches = tactile.create_group("region_pressure_vector")
    for side in ("left", "right"):
        patches.create_dataset(
            f"{side}_ee",
            shape=(frame_count, HISTORY_60, len(REGION_ORDER)),
            dtype=np.float32,
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
    patches.attrs["regions"] = np.asarray(REGION_ORDER, dtype="S16")
    patches.attrs["unit"] = "newton"
    patches.attrs["history_fps"] = 60.0
    patches.attrs["history_frames"] = HISTORY_60
    patches.attrs["history_order"] = "most_recent_first"
    patches.attrs["palm_patch_usable"] = np.uint8(1 if palm_ok else 0)
    patches.attrs["tactile_lr_swapped"] = np.uint8(1 if swapped else 0)
    patches.attrs["palm_region_index"] = np.int64(0)

    raw_root = file.require_group("raw")
    manus = raw_root.require_group("manus")
    for side in ("left", "right"):
        manus.create_dataset(
            f"{side}_ee",
            shape=(frame_count, 21, 3),
            dtype=np.float32,
            **_GZIP,
        )
    manus.attrs["joint_order"] = np.asarray(MANO_JOINT_ORDER, dtype="S16")
    manus.attrs["coordinate_frame"] = stored_manus_frame
    manus.attrs["source_coordinate_frame"] = coordinate_frame
    manus.attrs["source_frame_indices"] = source_indices_attr

    raw_adc = raw_root.create_group("tactile_raw_60hz")
    raw_adc.attrs["channel_count"] = RAW_CHANNEL_COUNT
    raw_adc.attrs["layout"] = "moxian_23x20_460"
    raw_adc.attrs["unit"] = "adc"
    raw_adc.attrs["baseline_subtracted"] = False
    raw_adc.attrs["source_fps"] = 60.0

    patch60 = raw_root.create_group("tactile_patch_60hz")
    patch60.attrs["regions"] = np.asarray(REGION_ORDER, dtype="S16")
    patch60.attrs["unit"] = "newton"
    patch60.attrs["source_fps"] = 60.0
    patch60.attrs["history_order"] = "native_60hz"
    patch60.attrs["palm_patch_usable"] = np.uint8(1 if palm_ok else 0)
    patch60.attrs["tactile_lr_swapped"] = np.uint8(1 if swapped else 0)
    patch60.attrs["palm_region_index"] = np.int64(0)

    raw60 = tactile.create_group("finger_pressure_60hz")
    _write_region_attrs(
        raw60,
        source_fps=60.0,
        layout="palm_then_thumb_index_middle_ring_pinky",
    )

    for side in ("left", "right"):
        taxel[f"{side}_ee"][:] = camera_taxel_320(
            context,
            side,
            source_indices,
            cam_head_extrinsics=extrinsics,
            world_from_egopro=world_map,
        )
        patches[f"{side}_ee"][:] = region_pressure_history(
            context, side, source_indices
        )
        manus[f"{side}_ee"][:] = manus_joints21(context, side, source_indices)
        frames60, pressure60 = finger_pressure_60hz(context, side)
        raw60.create_dataset(f"{side}_ee", data=pressure60, **_GZIP)
        raw60.create_dataset(f"{side}_frames", data=frames60)
        frames460, values460 = tactile_raw_460_60hz(context, side)
        raw_adc.create_dataset(f"{side}_ee", data=values460, **_GZIP)
        raw_adc.create_dataset(f"{side}_frames", data=frames460)
        frames_p, values_p = tactile_patch_60hz(context, side)
        patch60.create_dataset(f"{side}_ee", data=values_p, **_GZIP)
        patch60.create_dataset(f"{side}_frames", data=frames_p)
