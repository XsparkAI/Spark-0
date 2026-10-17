"""Moxian palm and finger geometry for Spark tactile conversion.

Vendored from ``ego_pro/tactile_utils/lib/taxel_geometry.py`` so the Spark
conversion pipeline is self-contained. The module only depends on numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, Optional, Sequence

import numpy as np


MOXIAN_COLS = 20
MOXIAN_ROWS = 23
MOXIAN_CHANNELS = MOXIAN_COLS * MOXIAN_ROWS
FINGER_FIRST_ROW = 15
FINGER_LAST_ROW = 22
FINGER_INACTIVE_ROWS_NEAR_PALM = 2
FINGER_ACTIVE_FIRST_ROW = FINGER_FIRST_ROW + FINGER_INACTIVE_ROWS_NEAR_PALM
FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
FINGER_ROW_ROLL_DEG = -20.0
FINGER_TAXEL_COLUMN_SPAN_M = 0.01
FINGER_SENSOR_START_FRACTION = 0.10
FINGER_SENSOR_END_FRACTION = 1.00
PALM_METACARPAL_START_FRACTION = 0.00
PALM_METACARPAL_END_FRACTION = 1.00
PALM_SURFACE_FINGERS = ("pinky", "ring", "middle", "index")

_FINGER_COL_RANGES = {
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


@dataclass(frozen=True)
class TaxelBinding:
    hand: str
    channel: int
    row: int
    col: int
    region: str
    finger: Optional[str]
    row_fraction: float
    lateral_fraction: float


def _vec3(value: object) -> np.ndarray | None:
    if isinstance(value, Mapping):
        values = (value.get("x"), value.get("y"), value.get("z"))
    elif isinstance(value, np.ndarray) and value.size >= 3:
        values = value.reshape(-1)[:3]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 3:
        values = value[:3]
    else:
        return None
    result = np.asarray(values, dtype=np.float64)
    return result if np.all(np.isfinite(result)) else None


def skeleton_joints(row: Mapping[str, Any], side: str) -> dict[str, np.ndarray]:
    """Return EgoPro joints without their ``l_``/``r_`` prefix."""
    prefix = "l_" if side == "left" else "r_"
    result: dict[str, np.ndarray] = {}
    for joint in row.get("joints") or []:
        if not isinstance(joint, Mapping) or not bool(joint.get("valid", True)):
            continue
        point = _vec3(joint.get("position_m") or joint.get("position"))
        if point is None:
            continue
        name = str(joint.get("name", ""))
        result[name[len(prefix) :] if name.startswith(prefix) else name] = point
    return result

def _normalize_hand(hand: str) -> str:
    value = str(hand).strip().lower()
    if value not in {"left", "right"}:
        raise ValueError("hand must be 'left' or 'right'")
    return value


def _finger_for_col(hand: str, col: int) -> tuple[str, int, int]:
    for finger, (start, end) in _FINGER_COL_RANGES[hand].items():
        if start <= col <= end:
            return finger, start, end
    raise ValueError(f"No Moxian finger column mapping for {hand} col {col}")


def moxian_channel_binding(hand: str, channel: int) -> TaxelBinding:
    hand = _normalize_hand(hand)
    channel = int(channel)
    if channel < 0 or channel >= MOXIAN_CHANNELS:
        raise ValueError(f"Moxian channel must be 0..{MOXIAN_CHANNELS - 1}")
    row, col = divmod(channel, MOXIAN_COLS)
    if row < FINGER_FIRST_ROW:
        return TaxelBinding(
            hand=hand,
            channel=channel,
            row=row,
            col=col,
            region="palm",
            finger=None,
            row_fraction=row / max(1, FINGER_FIRST_ROW - 1),
            lateral_fraction=(col / max(1, MOXIAN_COLS - 1)) - 0.5,
        )
    if row < FINGER_ACTIVE_FIRST_ROW:
        return TaxelBinding(
            hand=hand,
            channel=channel,
            row=row,
            col=col,
            region="unused",
            finger=None,
            row_fraction=row / max(1, FINGER_FIRST_ROW - 1),
            lateral_fraction=(col / max(1, MOXIAN_COLS - 1)) - 0.5,
        )
    finger, start, end = _finger_for_col(hand, col)
    return TaxelBinding(
        hand=hand,
        channel=channel,
        row=row,
        col=col,
        region="finger",
        finger=finger,
        row_fraction=(row - FINGER_ACTIVE_FIRST_ROW) / max(1, FINGER_LAST_ROW - FINGER_ACTIVE_FIRST_ROW),
        lateral_fraction=((col - start) / max(1, end - start)) - 0.5,
    )


@lru_cache(maxsize=2)
def moxian_channel_bindings(hand: str) -> tuple[TaxelBinding, ...]:
    hand = _normalize_hand(hand)
    return tuple(moxian_channel_binding(hand, channel) for channel in range(MOXIAN_CHANNELS))


def taxel_world_points(
    hand: str,
    joints: Mapping[str, np.ndarray],
    *,
    wrist_tracker_position: object = None,
) -> np.ndarray:
    """Return dense 460-channel centers; inactive transition rows remain NaN."""
    hand = _normalize_hand(hand)
    bindings = moxian_channel_bindings(hand)
    chains = _hand_chains(joints)
    tracker_point = _as_vec3(wrist_tracker_position)
    palm_lateral_axis = _palm_lateral_axis(joints, tracker_point)
    palm_normal_axis = _palm_normal_axis(joints, hand, tracker_point)
    centers = np.full((MOXIAN_CHANNELS, 3), np.nan, dtype=np.float64)
    finger_channels: dict[str, list[int]] = {finger: [] for finger in FINGER_ORDER}
    for channel, binding in enumerate(bindings):
        if binding.region == "palm":
            center = _palm_center(binding, joints, tracker_point)
            if center is not None:
                centers[channel] = center
            continue
        if binding.region == "finger" and binding.finger is not None:
            finger_channels[binding.finger].append(channel)
            continue

    for finger, channels in finger_channels.items():
        if not channels:
            continue
        chain = chains.get(finger)
        if not chain:
            continue
        finger_points = chain
        finger_lateral_axis = _finger_row_lateral_axis(finger_points, palm_lateral_axis, palm_normal_axis)
        revolution_correction_deg = _finger_row_roll_deg(hand)
        finger_lateral_axis = _rotate_vector_around_axis(
            finger_lateral_axis,
            _finger_longitudinal_axis(finger_points),
            revolution_correction_deg,
        )
        arc_fractions = np.asarray(
            [_finger_sensor_fraction(bindings[channel].row_fraction) for channel in channels],
            dtype=np.float64,
        )
        centerline = _interpolate_polyline_many(finger_points, arc_fractions)
        lateral = np.asarray(
            [float(bindings[channel].lateral_fraction) for channel in channels],
            dtype=np.float64,
        )
        finger_centers = (
            centerline
            + lateral[:, None]
            * finger_lateral_axis.reshape(1, 3)
            * FINGER_TAXEL_COLUMN_SPAN_M
        )
        for channel, center in zip(channels, finger_centers):
            centers[channel] = center

    return centers


def _as_vec3(value: object) -> Optional[np.ndarray]:
    if isinstance(value, Mapping):
        items = [value.get("x"), value.get("y"), value.get("z")]
    elif isinstance(value, np.ndarray) and value.size >= 3:
        items = value.reshape(-1)[:3]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 3:
        items = [value[0], value[1], value[2]]
    else:
        return None
    vector = np.asarray(items, dtype=np.float64)
    return vector if np.isfinite(vector).all() else None


def _finger_key(name: str) -> str:
    lower = name.lower()
    if "thumb" in lower:
        return "thumb"
    if "index" in lower:
        return "index"
    if "middle" in lower:
        return "middle"
    if "ring" in lower:
        return "ring"
    if "pinky" in lower or "little" in lower:
        return "pinky"
    if "hand" in lower or "wrist" in lower:
        return "wrist"
    return ""


def _bone_order(name: str) -> int:
    lower = name.lower()
    if "hand" in lower or "wrist" in lower:
        return 0
    if "metacarpal" in lower:
        return 1
    if "proximal" in lower:
        return 2
    if "intermediate" in lower:
        return 3
    if "distal" in lower:
        return 4
    if "tip" in lower:
        return 5
    return 9


def _hand_chains(joints: Mapping[str, np.ndarray]) -> dict[str, list[np.ndarray]]:
    groups: dict[str, list[tuple[str, int, np.ndarray]]] = {}
    for name, point in joints.items():
        key = _finger_key(name)
        if key:
            groups.setdefault(key, []).append((name, _bone_order(name), point))
    for key in groups:
        groups[key].sort(key=lambda item: (item[1], item[0]))
    chains: dict[str, list[np.ndarray]] = {}
    for finger in FINGER_ORDER:
        points = [item[2] for item in groups.get(finger, []) if item[1] >= 2]
        if points:
            chains[finger] = points
    return chains


def _interpolate_polyline_many(points: list[np.ndarray], fractions: np.ndarray) -> np.ndarray:
    if len(points) == 1:
        return np.repeat(points[0].reshape(1, 3), int(fractions.size), axis=0)
    point_array = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    segment_vectors = point_array[1:] - point_array[:-1]
    lengths = np.linalg.norm(segment_vectors, axis=1)
    total = float(lengths.sum())
    if total <= 1e-9:
        return np.repeat(point_array[0:1], int(fractions.size), axis=0)
    targets = np.clip(fractions.astype(np.float64), 0.0, 1.0) * total
    cumulative = np.cumsum(lengths)
    segment_indices = np.searchsorted(cumulative, targets, side="left")
    segment_indices = np.clip(segment_indices, 0, len(lengths) - 1)
    starts = np.concatenate(([0.0], cumulative[:-1]))[segment_indices]
    segment_lengths = np.maximum(lengths[segment_indices], 1e-9)
    ratios = ((targets - starts) / segment_lengths).reshape(-1, 1)
    return point_array[segment_indices] * (1.0 - ratios) + point_array[segment_indices + 1] * ratios


def _finger_frame_axes(
    points: list[np.ndarray], palm_lateral_axis: np.ndarray, palm_normal_axis: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finger_axis = _finger_longitudinal_axis(points)
    lateral_axis = _unit_perpendicular_to_axis(palm_lateral_axis, finger_axis)
    if lateral_axis is None:
        lateral_axis = _unit_perpendicular_to_axis(np.asarray([1.0, 0.0, 0.0], dtype=np.float64), finger_axis)
    if lateral_axis is None:
        lateral_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    normal_axis = np.cross(lateral_axis, finger_axis)
    normal_norm = float(np.linalg.norm(normal_axis))
    if normal_norm > 1e-9:
        normal_axis = normal_axis / normal_norm
    else:
        normal_axis = _unit_perpendicular_to_axis(palm_normal_axis, finger_axis)
        if normal_axis is None:
            normal_axis = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    if float(np.dot(normal_axis, palm_normal_axis)) < 0.0:
        normal_axis = -normal_axis
        lateral_axis = -lateral_axis
    return finger_axis, lateral_axis, normal_axis


def _finger_row_lateral_axis(
    points: list[np.ndarray],
    palm_lateral_axis: np.ndarray,
    palm_normal_axis: np.ndarray,
) -> np.ndarray:
    _finger_axis, lateral_axis, _normal_axis = _finger_frame_axes(points, palm_lateral_axis, palm_normal_axis)
    bend_plane_normal = _unit_perpendicular_to_axis(lateral_axis, palm_normal_axis)
    if bend_plane_normal is not None:
        projected_lateral = _unit_perpendicular_to_axis(bend_plane_normal, palm_normal_axis)
        if projected_lateral is not None:
            lateral_axis = projected_lateral
    return lateral_axis


def _finger_longitudinal_axis(points: list[np.ndarray]) -> np.ndarray:
    if len(points) >= 2:
        axis = points[-1] - points[0]
        norm = float(np.linalg.norm(axis))
        if norm > 1e-9:
            return axis / norm
    return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)


def _finger_row_roll_deg(hand: str) -> float:
    return -FINGER_ROW_ROLL_DEG if _normalize_hand(hand) == "right" else FINGER_ROW_ROLL_DEG


def _rotate_vector_around_axis(vector: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-9:
        return vector
    unit_axis = axis / axis_norm
    angle_rad = np.deg2rad(float(angle_deg))
    return (
        vector * np.cos(angle_rad)
        + np.cross(unit_axis, vector) * np.sin(angle_rad)
        + unit_axis * float(np.dot(unit_axis, vector)) * (1.0 - np.cos(angle_rad))
    )


def _unit_perpendicular_to_axis(vector: np.ndarray, axis: np.ndarray) -> Optional[np.ndarray]:
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-9:
        return None
    unit_axis = axis / axis_norm
    perpendicular = vector - unit_axis * float(np.dot(vector, unit_axis))
    norm = float(np.linalg.norm(perpendicular))
    if norm <= 1e-9:
        return None
    return perpendicular / norm


def _palm_points(
    joints: Mapping[str, np.ndarray],
    fallback_wrist: Optional[np.ndarray] = None,
) -> tuple[Optional[np.ndarray], list[np.ndarray]]:
    wrist: Optional[np.ndarray] = None
    bases_by_finger: dict[str, tuple[int, np.ndarray]] = {}
    for name, point in joints.items():
        key = _finger_key(name)
        order = _bone_order(name)
        if key == "wrist":
            wrist = point
        elif key in FINGER_ORDER and order <= 2:
            previous = bases_by_finger.get(key)
            if previous is None or order < previous[0]:
                bases_by_finger[key] = (order, point)
    bases = [bases_by_finger[finger][1] for finger in FINGER_ORDER if finger in bases_by_finger]
    if wrist is None and fallback_wrist is not None:
        wrist = fallback_wrist
    return wrist, bases


def _palm_lateral_axis(
    joints: Mapping[str, np.ndarray],
    fallback_wrist: Optional[np.ndarray] = None,
) -> np.ndarray:
    surface = _palm_surface_anchors(joints)
    if surface is not None:
        _start_center, _end_center, lateral_axis = surface
        return lateral_axis
    _wrist, bases = _palm_points(joints, fallback_wrist)
    if len(bases) >= 2:
        lateral_axis = bases[-1] - bases[0]
        norm = float(np.linalg.norm(lateral_axis))
        if norm > 1e-9:
            return lateral_axis / norm
    return np.asarray([1.0, 0.0, 0.0], dtype=np.float64)


def _palm_longitudinal_axis(
    joints: Mapping[str, np.ndarray],
    fallback_wrist: Optional[np.ndarray] = None,
) -> np.ndarray:
    surface = _palm_surface_anchors(joints)
    if surface is not None:
        start_center, end_center, _lateral_axis = surface
        longitudinal_axis = end_center - start_center
        norm = float(np.linalg.norm(longitudinal_axis))
        if norm > 1e-9:
            return longitudinal_axis / norm
    wrist, bases = _palm_points(joints, fallback_wrist)
    if wrist is not None and bases:
        longitudinal_axis = np.mean(np.asarray(bases, dtype=np.float64), axis=0) - wrist
        norm = float(np.linalg.norm(longitudinal_axis))
        if norm > 1e-9:
            return longitudinal_axis / norm
    return np.asarray([0.0, 0.0, 1.0], dtype=np.float64)


def _palm_normal_axis(
    joints: Mapping[str, np.ndarray],
    hand: str,
    fallback_wrist: Optional[np.ndarray] = None,
) -> np.ndarray:
    normal = np.cross(_palm_lateral_axis(joints, fallback_wrist), _palm_longitudinal_axis(joints, fallback_wrist))
    norm = float(np.linalg.norm(normal))
    if norm > 1e-9:
        normal = normal / norm
        if hand == "right":
            normal = -normal
        return normal
    return np.asarray([0.0, 1.0, 0.0], dtype=np.float64)


def _palm_center(
    binding: TaxelBinding,
    joints: Mapping[str, np.ndarray],
    fallback_wrist: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    surface_center = _palm_metacarpal_surface_center(binding, joints)
    if surface_center is not None:
        return surface_center
    palm_fraction = _palm_metacarpal_fraction(binding.row_fraction)
    surface = _palm_surface_anchors(joints)
    if surface is not None:
        start_center, end_center, lateral_axis = surface
        centerline = start_center * (1.0 - palm_fraction) + end_center * palm_fraction
        return centerline + lateral_axis * _palm_lateral_fraction(binding) * 0.08
    wrist, bases = _palm_points(joints, fallback_wrist)
    if wrist is None or not bases:
        return None
    base_center = np.mean(np.asarray(bases, dtype=np.float64), axis=0)
    lateral_axis = _palm_lateral_axis(joints, fallback_wrist)
    longitudinal_axis = _palm_longitudinal_axis(joints, fallback_wrist)
    centerline = wrist * (1.0 - palm_fraction) + base_center * palm_fraction
    return centerline + lateral_axis * _palm_lateral_fraction(binding) * 0.08 + longitudinal_axis * 0.01


def _palm_lateral_fraction(binding: TaxelBinding) -> float:
    return float(binding.lateral_fraction)


def _palm_metacarpal_fraction(row_fraction: float) -> float:
    fraction = min(1.0, max(0.0, float(row_fraction)))
    span = PALM_METACARPAL_END_FRACTION - PALM_METACARPAL_START_FRACTION
    return PALM_METACARPAL_START_FRACTION + fraction * span


def _finger_sensor_fraction(row_fraction: float) -> float:
    fraction = min(1.0, max(0.0, float(row_fraction)))
    span = FINGER_SENSOR_END_FRACTION - FINGER_SENSOR_START_FRACTION
    return FINGER_SENSOR_START_FRACTION + fraction * span


def _palm_visual_col(hand: str, raw_col: int) -> int:
    hand = _normalize_hand(hand)
    return (MOXIAN_COLS - 1 - int(raw_col)) if hand == "left" else int(raw_col)


def _palm_metacarpal_surface_center(binding: TaxelBinding, joints: Mapping[str, np.ndarray]) -> Optional[np.ndarray]:
    metacarpals, proximals = _palm_metacarpal_and_proximal_points(joints)
    surface_fingers = [finger for finger in PALM_SURFACE_FINGERS if finger in metacarpals and finger in proximals]
    if len(surface_fingers) < 2:
        return None
    palm_fraction = _palm_metacarpal_fraction(binding.row_fraction)
    visual_col = max(0, min(MOXIAN_COLS - 1, _palm_visual_col(binding.hand, binding.col)))
    patch_position = (visual_col / max(1, MOXIAN_COLS - 1)) * (len(surface_fingers) - 1)
    patch_index = min(len(surface_fingers) - 2, int(np.floor(patch_position)))
    patch_fraction = patch_position - patch_index
    left_finger = surface_fingers[patch_index]
    right_finger = surface_fingers[patch_index + 1]
    left_rail = metacarpals[left_finger] * (1.0 - palm_fraction) + proximals[left_finger] * palm_fraction
    right_rail = metacarpals[right_finger] * (1.0 - palm_fraction) + proximals[right_finger] * palm_fraction
    return left_rail * (1.0 - patch_fraction) + right_rail * patch_fraction


def _palm_metacarpal_and_proximal_points(
    joints: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    metacarpals: dict[str, np.ndarray] = {}
    proximals: dict[str, np.ndarray] = {}
    for name, point in joints.items():
        key = _finger_key(name)
        order = _bone_order(name)
        if key not in FINGER_ORDER:
            continue
        if order == 1:
            metacarpals[key] = point
        elif order == 2:
            proximals[key] = point
    return metacarpals, proximals


def _palm_surface_anchors(
    joints: Mapping[str, np.ndarray],
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    metacarpals, proximals = _palm_metacarpal_and_proximal_points(joints)
    fingers = [finger for finger in FINGER_ORDER if finger in metacarpals and finger in proximals]
    if len(fingers) >= 2:
        start_points = [metacarpals[finger] for finger in fingers]
        end_points = [proximals[finger] for finger in fingers]
        lateral_axis = start_points[-1] - start_points[0]
        norm = float(np.linalg.norm(lateral_axis))
        if norm <= 1e-9:
            lateral_axis = end_points[-1] - end_points[0]
            norm = float(np.linalg.norm(lateral_axis))
        if norm > 1e-9:
            return (
                np.mean(np.asarray(start_points, dtype=np.float64), axis=0),
                np.mean(np.asarray(end_points, dtype=np.float64), axis=0),
                lateral_axis / norm,
            )
    return None
