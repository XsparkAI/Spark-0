"""Map glove taxels to MANO and rasterize geodesic contact proximity.

Uses the official MANO left/right templates supplied through ``--mano-assets``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..retargeting.mano.model import ManoModel, resolve_mano_assets


MOXIAN_COLS = 20
PALM_LAST_ROW = 14
FINGER_LAST_ROW = 22
MAPPING_FINGER_FIRST_ROW = {320: 19, 360: 17}
PIP_ROW = 17.5
TIP_ROW_EDGE = 22.5

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
PARTS = ("palm", *FINGERS)
FINGER_COL_RANGES = {
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
PALM_COLS = {"left": (4, 19), "right": (0, 15)}
MANO_CHAIN = {
    "index": (1, 2, 3),
    "middle": (4, 5, 6),
    "pinky": (7, 8, 9),
    "ring": (10, 11, 12),
    "thumb": (14, 15),
}


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else vector


def polyline_project(
    poly: np.ndarray, pts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Arc fraction, closest point and unit tangent of ``pts`` on a polyline."""
    start, vec = poly[:-1], poly[1:] - poly[:-1]
    seg_len = np.linalg.norm(vec, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(seg_len)))
    ratio = np.clip(
        ((pts[:, None, :] - start[None]) * vec[None]).sum(-1)
        / np.maximum(seg_len**2, 1e-18)[None],
        0.0,
        1.0,
    )
    proj = start[None] + ratio[..., None] * vec[None]
    nearest = np.linalg.norm(pts[:, None, :] - proj, axis=-1).argmin(1)
    index = np.arange(len(pts))
    return (
        (cumulative[nearest] + ratio[index, nearest] * seg_len[nearest])
        / cumulative[-1],
        proj[index, nearest],
        vec[nearest] / np.maximum(seg_len[nearest], 1e-12)[:, None],
    )


def _vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    face_n = np.cross(v1 - v0, v2 - v0)
    normals = np.zeros_like(verts)
    np.add.at(normals, faces[:, 0], face_n)
    np.add.at(normals, faces[:, 1], face_n)
    np.add.at(normals, faces[:, 2], face_n)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(length, 1e-12)


@dataclass(frozen=True)
class ManoGeometry:
    """MANO template plus the anatomical coordinates used for matching."""

    hand: str
    verts: np.ndarray
    faces: np.ndarray
    joints: np.ndarray
    normals: np.ndarray
    part: np.ndarray
    palmar: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    plane: np.ndarray
    axes: np.ndarray
    part_size: dict
    pip_arc: dict


def load_mano_geometry(
    hand: str, mano_assets: Path | str | None = None
) -> ManoGeometry:
    if hand not in ("left", "right"):
        raise ValueError("hand must be 'left' or 'right'")
    model = ManoModel.from_official(resolve_mano_assets(mano_assets), hand)
    verts = model.v_template.copy()
    faces = model.faces.copy()
    joints = model.j_regressor @ verts
    normals = _vertex_normals(verts, faces)
    weights = model.weights
    tip_vertex = {
        "thumb": int(model.tip_ids[0]),
        "index": int(model.tip_ids[1]),
        "middle": int(model.tip_ids[2]),
        "ring": int(model.tip_ids[3]),
        "pinky": int(model.tip_ids[4]),
    }

    joint_label = weights.argmax(1)
    part = np.array(["palm"] * len(verts), dtype=object)
    for finger, chain in MANO_CHAIN.items():
        part[np.isin(joint_label, chain)] = finger

    origin = joints[0]
    mcp = joints[[MANO_CHAIN[f][0] for f in ("index", "middle", "ring", "pinky")]]
    lon_axis = unit(mcp.mean(0) - origin)
    span = joints[MANO_CHAIN["pinky"][0]] - joints[MANO_CHAIN["index"][0]]
    lat_axis = unit(np.cross(lon_axis, np.cross(span, lon_axis)))
    if np.dot(span, lat_axis) < 0:
        lat_axis = -lat_axis
    nrm_axis = unit(np.cross(lat_axis, lon_axis))
    if np.dot(joints[MANO_CHAIN["thumb"][-1]] - origin, nrm_axis) < 0:
        nrm_axis = -nrm_axis
    axes = np.column_stack((lat_axis, lon_axis, nrm_axis))

    local = (verts - origin) @ axes
    palmar = (normals @ nrm_axis) > 0.0
    lon = np.zeros(len(verts))
    lat = np.zeros(len(verts))
    part_size: dict[str, tuple[float, float]] = {}
    pip_arc: dict[str, float] = {}

    palm = part == "palm"
    palm_length = float(np.dot(mcp.mean(0) - origin, lon_axis))
    lon[palm] = local[palm, 1] / max(palm_length, 1e-9)
    low, high = np.percentile(local[palm & palmar, 0], [2, 98])
    lat[palm] = (local[palm, 0] - 0.5 * (low + high)) / max(high - low, 1e-9)
    part_size["palm"] = (palm_length, float(high - low))

    for finger in FINGERS:
        mask = part == finger
        poly = np.vstack(
            [joints[list(MANO_CHAIN[finger])], verts[tip_vertex[finger]]]
        )
        arc, closest, tangent = polyline_project(poly, verts[mask])
        bone_normal = np.cross(np.cross(tangent, nrm_axis[None, :]), tangent)
        bone_normal /= np.linalg.norm(bone_normal, axis=1, keepdims=True) + 1e-12
        bone_lateral = np.cross(bone_normal, tangent)
        bone_lateral /= np.linalg.norm(bone_lateral, axis=1, keepdims=True) + 1e-12
        if np.dot(bone_lateral.mean(0), lat_axis) < 0:
            bone_lateral = -bone_lateral
        offset = ((verts[mask] - closest) * bone_lateral).sum(1)
        width = float(np.percentile(np.abs(offset), 95) * 2)
        lon[mask] = arc
        lat[mask] = offset / max(width, 1e-9)
        segment = np.linalg.norm(np.diff(poly, axis=0), axis=1)
        part_size[finger] = (float(segment.sum()), width)
        pip_arc[finger] = float(segment[0] / segment.sum())

    return ManoGeometry(
        hand,
        verts,
        faces,
        joints,
        normals,
        part,
        palmar,
        lon,
        lat,
        local[:, :2],
        axes,
        part_size,
        pip_arc,
    )


@dataclass(frozen=True)
class GloveLayout:
    hand: str
    mapping: int
    channel_ids: np.ndarray
    row: np.ndarray
    col: np.ndarray
    part: np.ndarray
    lon: np.ndarray
    lat: np.ndarray


def channel_ids(hand: str, mapping: int = 320) -> np.ndarray:
    if hand not in PALM_COLS:
        raise ValueError("hand must be 'left' or 'right'")
    if mapping not in MAPPING_FINGER_FIRST_ROW:
        raise ValueError(f"mapping must be one of {sorted(MAPPING_FINGER_FIRST_ROW)}")
    first_row = MAPPING_FINGER_FIRST_ROW[mapping]
    low, high = PALM_COLS[hand]
    ids = [
        r * MOXIAN_COLS + c + 1
        for r in range(PALM_LAST_ROW, -1, -1)
        for c in range(low, high + 1)
    ]
    order = FINGERS if hand == "left" else FINGERS[::-1]
    for finger in order:
        start, end = FINGER_COL_RANGES[hand][finger]
        ids.extend(
            r * MOXIAN_COLS + c + 1
            for r in range(FINGER_LAST_ROW, first_row - 1, -1)
            for c in range(start, end + 1)
        )
    result = np.asarray(ids, dtype=np.int64)
    if len(result) != mapping:
        raise AssertionError(f"expected {mapping} channels, built {len(result)}")
    return result


def load_glove_layout(
    hand: str, mano: ManoGeometry, mapping: int = 320
) -> GloveLayout:
    ids = channel_ids(hand, mapping)
    row, col = np.divmod(ids - 1, MOXIAN_COLS)
    part = np.array(
        [
            "palm"
            if r <= PALM_LAST_ROW
            else next(
                f
                for f, (s, e) in FINGER_COL_RANGES[hand].items()
                if s <= c <= e
            )
            for r, c in zip(row, col)
        ],
        dtype=object,
    )

    lon = np.zeros(len(ids))
    lat = np.zeros(len(ids))
    palm_direction = 1.0 if hand == "left" else -1.0
    for name in PARTS:
        mask = part == name
        if not mask.any():
            continue
        if name == "palm":
            low, high = PALM_COLS[hand]
            lon[mask] = row[mask] / PALM_LAST_ROW
            lat[mask] = palm_direction * ((col[mask] - low) / (high - low) - 0.5)
        else:
            pip = mano.pip_arc[name]
            lon[mask] = pip + (row[mask] - PIP_ROW) * (1.0 - pip) / (
                TIP_ROW_EDGE - PIP_ROW
            )
            start, end = FINGER_COL_RANGES[hand][name]
            lat[mask] = (col[mask] - start) / (end - start) - 0.5
    return GloveLayout(hand, mapping, ids, row, col, part, lon, lat)


def taxel_pitch(mano: ManoGeometry, glove: GloveLayout) -> dict:
    pitch = {}
    for name in PARTS:
        taxels = np.flatnonzero(glove.part == name)
        if len(taxels) < 2:
            continue
        length, width = mano.part_size[name]
        coords = np.column_stack(
            (glove.lon[taxels] * length, glove.lat[taxels] * width)
        )
        distance = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
        np.fill_diagonal(distance, np.inf)
        pitch[name] = float(np.median(distance.min(1)))
    return pitch


def intrinsic_cost(
    mano: ManoGeometry, glove: GloveLayout, cutoff: float | None = 2.5
) -> np.ndarray:
    pitch = taxel_pitch(mano, glove)
    cost = np.full((len(glove.channel_ids), len(mano.verts)), np.inf)
    for name in PARTS:
        taxels = np.flatnonzero(glove.part == name)
        verts = np.flatnonzero((mano.part == name) & mano.palmar)
        if not len(taxels) or not len(verts):
            continue
        length, width = mano.part_size[name]
        along = (glove.lon[taxels][:, None] - mano.lon[verts][None, :]) * length
        across = (glove.lat[taxels][:, None] - mano.lat[verts][None, :]) * width
        block = np.hypot(along, across)
        if cutoff is not None:
            block[block > cutoff * pitch[name]] = np.inf
        cost[np.ix_(taxels, verts)] = block
    return cost


def soft_weights(
    mano: ManoGeometry,
    glove: GloveLayout,
    neighbours: int = 4,
    cutoff: float = 2.5,
) -> np.ndarray:
    cost = intrinsic_cost(mano, glove, cutoff)
    sigma = np.full(len(mano.verts), np.nan)
    for name, pitch in taxel_pitch(mano, glove).items():
        sigma[mano.part == name] = pitch
    weights = np.zeros_like(cost)
    for vertex in np.flatnonzero(np.isfinite(cost).any(0) & np.isfinite(sigma)):
        column = cost[:, vertex]
        finite = np.flatnonzero(np.isfinite(column))
        near = finite[np.argsort(column[finite])[:neighbours]]
        kernel = np.exp(-0.5 * (column[near] / sigma[vertex]) ** 2)
        if kernel.sum() > 1e-12:
            weights[near, vertex] = kernel / kernel.sum()
    return weights


def build_mapping(
    hand: str,
    mapping: int = 320,
    neighbours: int = 4,
    cutoff: float = 2.5,
    mano_assets: Path | str | None = None,
) -> dict:
    """Build the taxel -> MANO vertex weight matrix."""
    mano = load_mano_geometry(hand, mano_assets)
    glove = load_glove_layout(hand, mano, mapping)
    weights = soft_weights(mano, glove, neighbours=neighbours, cutoff=cutoff)
    return {
        "hand": hand,
        "mapping": mapping,
        "cutoff": cutoff,
        "channel_ids": glove.channel_ids,
        "weights": weights,
        "glove_part": glove.part.astype(str),
        "glove_lon": glove.lon,
        "glove_lat": glove.lat,
        "mano_part": mano.part.astype(str),
        "mano_palmar": mano.palmar,
        "mano_plane": mano.plane,
        "mano": mano,
        "glove": glove,
    }


def apply_mapping(
    weights: np.ndarray, pressure: np.ndarray
) -> np.ndarray:
    pressure = np.asarray(pressure, dtype=np.float64)
    if pressure.shape[-1] != weights.shape[0]:
        raise ValueError(
            f"pressure has {pressure.shape[-1]} channels, mapping expects "
            f"{weights.shape[0]}"
        )
    return pressure @ weights


REST_FRAMES = 60
NOISE_FLOOR = 1.0
PROXIMITY_EPS = 0.05
DEFAULT_W_L = 3
DEFAULT_W_R = 24
DEFAULT_K_MAG = 2.0
DEFAULT_K_RISE = 2.0
DEFAULT_K_END = 5
DEFAULT_SIGMA_M = 0.030
DEFAULT_FADE_FRAMES = 8
DEFAULT_BASELINE_FRAMES = 10
DEFAULT_PRESSURE_DEADBAND = 10.0


def fingertip_vertex_masks(
    mano: ManoGeometry,
    hand: str,
    mano_assets: Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(palmar_finger, all_finger)`` masks over the 778 vertices."""
    model = ManoModel.from_official(resolve_mano_assets(mano_assets), hand)
    joint_label = model.weights.argmax(1)
    all_tip = np.zeros(len(mano.verts), dtype=bool)
    for chain in MANO_CHAIN.values():
        all_tip[np.isin(joint_label, chain)] = True
    all_tip[np.asarray(model.tip_ids, dtype=np.int64)] = True
    palmar_tip = all_tip & np.asarray(mano.palmar, dtype=bool)
    return palmar_tip, all_tip


def rolling_max(values: np.ndarray, window: int) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view

    values = np.asarray(values, dtype=np.float64)
    window = max(1, int(window))
    if window == 1:
        return values.copy()
    if values.ndim == 1:
        pad = np.empty(len(values) + window - 1, dtype=np.float64)
        pad[: window - 1] = -np.inf
        pad[window - 1 :] = values
        return sliding_window_view(pad, window).max(-1)
    pad = np.empty((len(values) + window - 1,) + values.shape[1:], dtype=np.float64)
    pad[: window - 1] = -np.inf
    pad[window - 1 :] = values
    return sliding_window_view(pad, window, axis=0).max(axis=-1)


def window_level_and_change(
    trace: np.ndarray, w_l: int, w_r: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trace = np.asarray(trace, dtype=np.float64).reshape(-1)
    length = len(trace)
    w_l = max(1, int(w_l))
    w_r = max(1, int(w_r))
    level = rolling_max(trace, w_l)
    rise = np.empty(length, dtype=np.float64)
    fall = np.empty(length, dtype=np.float64)
    for index in range(length):
        long = trace[max(0, index - w_r + 1) : index + 1]
        peak_at = int(long.argmax())
        peak = float(long[peak_at])
        rise[index] = peak - float(long[: peak_at + 1].min())
        fall[index] = peak - level[index]
    return level, rise, fall


def _mad_sigma(samples: np.ndarray) -> float:
    samples = np.asarray(samples, dtype=np.float64).reshape(-1)
    if samples.size == 0:
        return NOISE_FLOOR
    center = float(np.median(samples))
    return float(1.4826 * np.median(np.abs(samples - center)))


def rest_noise(
    trace: np.ndarray,
    baseline_frames: int = DEFAULT_BASELINE_FRAMES,
    rest_frames: int = REST_FRAMES,
    w_l: int = DEFAULT_W_L,
) -> float:
    trace = np.asarray(trace, dtype=np.float64).reshape(-1)
    start = min(len(trace), max(0, int(baseline_frames)))
    stop = min(len(trace), start + int(rest_frames))
    rest = trace[start:stop]
    primary = _mad_sigma(rest)
    level = rolling_max(trace, w_l)
    quiet = level[level <= np.percentile(level, 20)] if len(level) else level
    fallback = _mad_sigma(quiet)
    contaminated = rest.size > 0 and (
        primary > max(3.0, 3.0 * max(fallback, NOISE_FLOOR))
        or float(rest.max()) > float(np.percentile(level, 70))
    )
    return max(fallback if contaminated else primary, NOISE_FLOOR)


def detect_contact_events(
    level: np.ndarray,
    rise: np.ndarray,
    fall: np.ndarray,
    noise: float,
    k_mag: float,
    k_rise: float,
    k_end: int = DEFAULT_K_END,
) -> dict[str, np.ndarray]:
    z_l = np.asarray(level, dtype=np.float64) / max(float(noise), NOISE_FLOOR)
    z_rise = np.asarray(rise, dtype=np.float64) / max(float(noise), NOISE_FLOOR)
    z_fall = np.asarray(fall, dtype=np.float64) / max(float(noise), NOISE_FLOOR)
    active = np.zeros(len(z_l), dtype=bool)
    enter_rise = np.zeros(len(z_l), dtype=np.float64)
    in_contact = False
    held_rise = 0.0
    low_count = 0
    hold_floor = 0.5 * float(k_mag)
    for index, (level_z, rise_z, fall_z) in enumerate(zip(z_l, z_rise, z_fall)):
        entered = level_z >= k_mag and rise_z >= k_rise
        if not in_contact:
            if entered:
                in_contact = True
                held_rise = float(rise_z)
                low_count = 0
        elif level_z >= hold_floor:
            low_count = 0
        else:
            low_count += 1
            if low_count >= k_end or (fall_z >= k_rise and level_z < hold_floor):
                in_contact = False
                held_rise = 0.0
                low_count = 0
        active[index] = in_contact
        enter_rise[index] = held_rise
    return {
        "active": active,
        "z_l": z_l,
        "z_rise": z_rise,
        "z_fall": z_fall,
        "enter_rise": enter_rise,
    }


def source_amplitude(
    z_l: np.ndarray, enter_rise: np.ndarray, k_mag: float, k_rise: float
) -> np.ndarray:
    level_term = np.clip(z_l / max(3.0 * k_mag, 1e-6), 0.0, 1.0)
    rise_term = np.clip(0.35 + 0.65 * enter_rise / max(3.0 * k_rise, 1e-6), 0.0, 1.0)
    return level_term * rise_term


def geodesic_table(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """``(V, V)`` rest-pose geodesic distances in metres."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    start = np.concatenate([faces[:, 0], faces[:, 1], faces[:, 2]])
    end = np.concatenate([faces[:, 1], faces[:, 2], faces[:, 0]])
    length = np.linalg.norm(verts[start] - verts[end], axis=1)
    graph = csr_matrix((length, (start, end)), shape=(len(verts), len(verts)))
    graph = graph.maximum(graph.T)
    return dijkstra(graph, directed=False)


def vertex_proximity(
    distance: np.ndarray,
    sources: np.ndarray,
    amplitude: float,
    sigma_m: float = DEFAULT_SIGMA_M,
    eps: float = PROXIMITY_EPS,
) -> np.ndarray:
    field = np.zeros(distance.shape[0], dtype=np.float64)
    if sources.size == 0 or amplitude <= 0.0:
        return field
    nearest = distance[:, sources].min(1)
    finite = nearest[np.isfinite(nearest)]
    if finite.size == 0:
        return field
    d_ref = max(float(np.percentile(finite, 98)), float(sigma_m), 1e-4)
    decay = np.clip(1.0 - nearest / d_ref, 0.0, 1.0)
    strength = 0.35 + 0.65 * float(np.clip(amplitude, 0.0, 1.0))
    return np.maximum(strength * decay, eps)


def fade_amplitudes(
    active: np.ndarray,
    amplitude: np.ndarray,
    fade_frames: int = DEFAULT_FADE_FRAMES,
) -> np.ndarray:
    faded = np.zeros_like(amplitude)
    last = 0.0
    remaining = 0
    fade_frames = max(1, int(fade_frames))
    for index, (on, value) in enumerate(zip(active, amplitude)):
        if on:
            last = float(value)
            remaining = fade_frames
            faded[index] = last
            continue
        if remaining > 0:
            remaining -= 1
            faded[index] = last * remaining / fade_frames
        else:
            faded[index] = 0.0
    return faded


def vertex_field_from_tip_pressure(
    tip_pressure: np.ndarray,
    mapping: dict[str, Any],
    *,
    w_l: int = DEFAULT_W_L,
    w_r: int = DEFAULT_W_R,
    k_mag: float = DEFAULT_K_MAG,
    k_rise: float = DEFAULT_K_RISE,
    k_end: int = DEFAULT_K_END,
    sigma_m: float = DEFAULT_SIGMA_M,
    baseline_frames: int = DEFAULT_BASELINE_FRAMES,
    pressure_deadband: float = DEFAULT_PRESSURE_DEADBAND,
) -> np.ndarray:
    """Geodesic 778-vertex field from mapping-ordered taxel traces.

    Contact detection is performed independently for every taxel.  The raw
    pressure deadband is applied before estimating temporal changes and noise,
    so a sparse palm contact cannot be diluted by averaging it with the other
    palm taxels.
    """
    tip_pressure = np.asarray(tip_pressure, dtype=np.float64)
    pressure_deadband = float(pressure_deadband)
    if not np.isfinite(pressure_deadband) or pressure_deadband < 0.0:
        raise ValueError("pressure_deadband must be a finite non-negative value")
    # A missing source frame can contain NaN/Inf. Treat missing or negative
    # pressure as zero so one dropped frame cannot poison episode-wide noise.
    tip_pressure = np.maximum(
        np.nan_to_num(tip_pressure, nan=0.0, posinf=0.0, neginf=0.0), 0.0
    )
    if pressure_deadband > 0.0:
        tip_pressure = np.maximum(tip_pressure - pressure_deadband, 0.0)
    tip_index = np.flatnonzero(np.asarray(mapping["taxel_mask"], dtype=bool))
    groups = mapping["tip_groups"]
    palmar_tip = np.asarray(mapping["palmar_tip"], dtype=bool)
    all_tip = np.asarray(mapping["all_tip"], dtype=bool)
    distance = mapping.get("_geodesic")
    if distance is None:
        distance = geodesic_table(mapping["mano"].verts, mapping["mano"].faces)
        mapping["_geodesic"] = distance
    weights = mapping["tip_weights"]
    local_of = {int(index): local for local, index in enumerate(tip_index)}

    finger_active: dict[str, np.ndarray] = {}
    finger_amplitude: dict[str, np.ndarray] = {}
    taxel_active = np.zeros_like(tip_pressure, dtype=bool)
    taxel_score = np.zeros_like(tip_pressure, dtype=np.float64)

    for finger, rows in groups.items():
        if rows.size == 0:
            continue
        local = np.asarray([local_of[int(row)] for row in rows], dtype=np.int64)
        group_active = np.zeros(len(tip_pressure), dtype=bool)
        group_amplitude = np.zeros(len(tip_pressure), dtype=np.float64)
        for column in local:
            channel = tip_pressure[:, int(column)]
            ch_level, ch_rise, ch_fall = window_level_and_change(channel, w_l, w_r)
            ch_sigma = rest_noise(channel, baseline_frames, REST_FRAMES, w_l)
            ch_events = detect_contact_events(
                ch_level, ch_rise, ch_fall, ch_sigma, k_mag, k_rise, k_end
            )
            ch_amplitude = fade_amplitudes(
                ch_events["active"],
                source_amplitude(
                    ch_events["z_l"], ch_events["enter_rise"], k_mag, k_rise
                ),
            )
            taxel_active[:, column] = ch_events["active"]
            taxel_score[:, column] = (
                ch_events["z_l"] * ch_events["z_rise"] * taxel_active[:, column]
            )
            group_active |= ch_events["active"]
            np.maximum(group_amplitude, ch_amplitude, out=group_amplitude)
        finger_active[finger] = group_active
        finger_amplitude[finger] = group_amplitude

    vertex = np.zeros((len(tip_pressure), 778), dtype=np.float64)
    for time_index in range(len(tip_pressure)):
        sources: list[int] = []
        amplitudes: list[float] = []
        for finger, rows in groups.items():
            if finger not in finger_active or not finger_active[finger][time_index]:
                continue
            local = np.asarray([local_of[int(row)] for row in rows], dtype=np.int64)
            active_local = local[taxel_active[time_index, local]]
            scores = taxel_score[time_index, active_local]
            chosen = active_local[np.argsort(-scores)[:3]]
            pressure = np.zeros(weights.shape[0], dtype=np.float64)
            pressure[tip_index[chosen]] = 1.0
            mapped = apply_mapping(weights, pressure)
            hits = np.flatnonzero((mapped > 1e-6) & palmar_tip)
            if hits.size == 0:
                hits = np.flatnonzero((mapping["mano"].part == finger) & all_tip)
            sources.extend(hits.tolist())
            amplitudes.append(float(finger_amplitude[finger][time_index]))
        if not sources:
            continue
        unique = np.unique(np.asarray(sources, dtype=np.int64))
        vertex[time_index] = vertex_proximity(
            distance, unique, float(max(amplitudes)), sigma_m
        )
    return vertex


def rasterize_posed_mesh(
    uv: np.ndarray,
    depth: np.ndarray,
    faces: np.ndarray,
    values: np.ndarray,
    height: int,
    width: int,
    eps: float = PROXIMITY_EPS,
) -> np.ndarray:
    """Z-buffered barycentric raster. Off-mesh stays 0."""
    import cv2

    del eps
    uv = np.asarray(uv, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    valid_vertex = (
        np.isfinite(uv).all(1)
        & np.isfinite(depth)
        & (depth > 0.03)
        & (uv[:, 0] >= -8)
        & (uv[:, 1] >= -8)
        & (uv[:, 0] < width + 8)
        & (uv[:, 1] < height + 8)
    )
    valid_face = valid_vertex[faces].all(1)
    if not np.any(valid_face):
        return np.zeros((height, width), dtype=np.float32)
    faces = faces[valid_face]
    order = np.argsort(-depth[faces].mean(1))
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    kept = faces[order]
    for face_id, face in enumerate(kept, start=1):
        pts = np.round(uv[face]).astype(np.int32)
        cv2.fillConvexPoly(canvas, pts, (face_id & 0xFF, (face_id >> 8) & 0xFF, 0))
    painted = (
        canvas[:, :, 0].astype(np.int64) | (canvas[:, :, 1].astype(np.int64) << 8)
    ).ravel()
    pixel = np.flatnonzero(painted)
    if pixel.size == 0:
        return np.zeros((height, width), dtype=np.float32)
    vertex = kept[painted[pixel] - 1]
    corner = uv[vertex]
    row, column = np.divmod(pixel, width)
    offset = np.column_stack((column, row)).astype(np.float64) - corner[:, 0]
    edge1 = corner[:, 1] - corner[:, 0]
    edge2 = corner[:, 2] - corner[:, 0]
    area = edge1[:, 0] * edge2[:, 1] - edge2[:, 0] * edge1[:, 1]
    area = np.where(np.abs(area) < 1e-9, np.nan, area)
    second = (offset[:, 0] * edge2[:, 1] - edge2[:, 0] * offset[:, 1]) / area
    third = (edge1[:, 0] * offset[:, 1] - offset[:, 0] * edge1[:, 1]) / area
    weight = np.clip(np.column_stack((1.0 - second - third, second, third)), 0.0, None)
    weight = np.where(np.isfinite(weight), weight, (1.0, 0.0, 0.0))
    weight /= np.maximum(weight.sum(1, keepdims=True), 1e-12)
    field = (weight * values[vertex]).sum(1)
    out = np.zeros(height * width, dtype=np.float32)
    out[pixel] = field.astype(np.float32)
    return out.reshape(height, width)


def display_field(field: np.ndarray) -> np.ndarray:
    """Keep the geodesic [eps, 1] field; off-mesh stays 0."""
    field = np.asarray(field, dtype=np.float32)
    vis = np.zeros_like(field)
    mesh = field > 0
    if np.any(mesh):
        vis[mesh] = np.clip(field[mesh], 0.0, 1.0)
    return vis


def _proximity_lut() -> np.ndarray:
    """BGR LUT matching manus2mano ``fingertip_proximity._proximity_lut``."""
    lavender = (168, 150, 186)
    stops = np.asarray(
        [
            (0.00, lavender),
            (0.08, lavender),
            (0.20, (210, 190, 70)),
            (0.34, (50, 200, 40)),
            (0.48, (0, 255, 255)),
            (0.58, (0, 130, 255)),
            (0.68, (0, 0, 255)),
            (1.00, (0, 0, 255)),
        ],
        dtype=object,
    )
    lut = np.zeros((256, 3), dtype=np.uint8)
    positions = [float(item[0]) for item in stops]
    colors = np.asarray([item[1] for item in stops], dtype=np.float64)
    for index in range(256):
        t = index / 255.0
        hi = 1
        while hi < len(positions) - 1 and t > positions[hi]:
            hi += 1
        lo = hi - 1
        span = max(positions[hi] - positions[lo], 1e-6)
        weight = (t - positions[lo]) / span
        lut[index] = np.clip((1.0 - weight) * colors[lo] + weight * colors[hi], 0, 255)
    return lut


PROXIMITY_LUT = _proximity_lut()


PALMAR_BASE_COLOR = np.array([232.0, 237.0, 240.0], dtype=np.float32)
