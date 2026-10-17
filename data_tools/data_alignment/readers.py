"""Readers that normalize source-specific semantics without choosing a file format."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .schema import ReferenceFrameGeometry
from .camera_alignment import (
    canonicalize_hand_points,
    ensure_quaternion_continuity,
    invert_transform,
    matrix_to_pose_wxyz,
    optical_upright_from_inverted,
    pose_wxyz_to_matrix,
    rpy_deg_matrix,
)


DATASET_NAMES = (
    "auto",
    "human_touch",
    "openaoe",
    "egodex",
    "fourier_actionnet",
    "agibot_world_beta",
    "egovla_bench",
    "egoverse",
)
HUMANTOUCH2_MARKER_TOPICS = frozenset(
    {
        "/views/head/image",
        "/human/hands/left_joints",
        "/human/head/pose",
        "/meta/global_parameters",
    }
)
_HUMANTOUCH2_BLOCK_RE = re.compile(
    r"(?:^|_)TASK_([A-Za-z][A-Za-z0-9-]*)(?:_|$)", re.IGNORECASE
)


def humantouch2_block_index_from_path(path: str | Path) -> str | None:
    """Infer HumanTouch2 ``blockIndex`` (for example ``B002``) from its path."""
    for part in reversed(Path(path).parts):
        match = _HUMANTOUCH2_BLOCK_RE.search(part)
        if match is not None:
            return match.group(1).upper()
    return None


# HumanTouch / EgoPro task codes look like T001 or X012. Dates such as
# 20260706 must not match.
_EGOPRO_TASK_CODE_RE = re.compile(r"(?<![A-Za-z0-9])([TX]\d{2,4})(?![A-Za-z0-9])")
_EGOPRO_DATE_RE = re.compile(r"^(\d{8})")


def _normalize_block_name(name: str) -> str:
    return name.replace("_", " ").lower()


def egopro_task_code_from_text(text: str) -> str | None:
    """Return the last Txxx/Xxxx code in ``text``, or None."""
    matches = _EGOPRO_TASK_CODE_RE.findall(text)
    return matches[-1].upper() if matches else None


def annotations_root_from_mcap(source: str | Path) -> Path:
    """``<dataset>/mcap/...`` → ``<dataset>/annotations``."""
    resolved = Path(source).resolve()
    parts = resolved.parts
    if "mcap" not in parts:
        raise ValueError(
            "EgoPro MCAP path must contain an 'mcap' directory so annotations "
            f"resolve as <dataset>/annotations: {resolved}"
        )
    index = parts.index("mcap")
    if index == 0:
        raise ValueError(
            "EgoPro MCAP path has 'mcap' as the filesystem root; "
            f"cannot resolve a sibling annotations directory: {resolved}"
        )
    return Path(*parts[:index]) / "annotations"


def _block_dir_and_date(source: Path) -> tuple[str | None, str | None]:
    """Return ``(YYYYMMDD, block_dir_name)`` parsed from an MCAP path."""
    for part in reversed(source.parts[:-1]):
        match = _EGOPRO_DATE_RE.match(part)
        if match is not None and egopro_task_code_from_text(part):
            return match.group(1), part
    for part in reversed(source.parts[:-1]):
        match = _EGOPRO_DATE_RE.match(part)
        if match is not None:
            return match.group(1), part
    return None, None


def find_egopro_semantic_review(
    source: str | Path,
    annotations_root: str | Path | None = None,
) -> Path | None:
    """Locate the ``*semantic_review.json`` that matches one EgoPro MCAP."""
    source = Path(source)
    root = (
        Path(annotations_root)
        if annotations_root is not None
        else annotations_root_from_mcap(source)
    )
    if not root.is_dir():
        return None
    date, block = _block_dir_and_date(source)
    stem = source.stem
    if date is not None and block is not None:
        dated = root / date
        if dated.is_dir():
            normalized = _normalize_block_name(block)
            candidates = [dated / block]
            candidates.extend(
                path
                for path in sorted(dated.iterdir())
                if path.is_dir() and _normalize_block_name(path.name) == normalized
            )
            for annotation_dir in dict.fromkeys(candidates):
                if not annotation_dir.is_dir():
                    continue
                matches = sorted(annotation_dir.glob(f"{stem}*semantic_review.json"))
                if matches:
                    return matches[0]
    matches = sorted(root.rglob(f"{stem}*semantic_review.json"))
    return matches[0] if matches else None


_SHORT_TASK_ID_RE = re.compile(r"^((?:T\d+(?:_\d+)?)|(?:X\d+))", re.IGNORECASE)


def short_task_id(task_id: str | None) -> str | None:
    """``T001_task`` → ``T001``, ``T035_1`` → ``T035_1``, ``X001_...`` → ``X001``."""
    if not isinstance(task_id, str) or not task_id.strip():
        return None
    match = _SHORT_TASK_ID_RE.match(task_id.strip())
    return match.group(1) if match else None


def load_review_json(path: str | Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _task_name_from_review(path: Path) -> str | None:
    payload = load_review_json(path)
    if payload is None:
        return None
    name = str(payload.get("task_name", "")).strip()
    return name or None


def parse_review_subtask_instructions(
    payload: Mapping[str, Any] | None,
) -> list[list[Any]]:
    """Return ``[caption, start_frame_60, end_frame_60_exclusive]`` rows.

    Only the episode's own review JSON should be passed in. Empty captions are
    dropped. Frame fields come from the annotation schema (60 Hz, half-open).
    """
    if not isinstance(payload, Mapping):
        return []
    rows: list[list[Any]] = []
    for item in payload.get("subtasks") or []:
        if not isinstance(item, Mapping):
            continue
        caption = str(item.get("caption") or "").strip()
        if not caption:
            continue
        try:
            start = int(item.get("start_frame_60"))
            end = int(item.get("end_frame_60_exclusive"))
        except (TypeError, ValueError):
            continue
        rows.append([caption, start, end])
    return rows


def load_task_name_map(path: str | Path) -> dict[str, str]:
    """Load task_id / short-code → episode instruction.

    Prefers English ``task_id_to_instruction`` / ``short_id_to_instruction``.
    Falls back to Chinese ``task_name`` maps when those keys are absent.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"task-name map must be a JSON object: {path}")
    mapping: dict[str, str] = {}
    blocks = [
        payload.get("task_id_to_instruction"),
        payload.get("short_id_to_instruction"),
        payload.get("task_id_to_task_name"),
        payload.get("short_id_to_task_name"),
    ]
    found_nested = False
    for block in blocks:
        if not isinstance(block, dict):
            continue
        found_nested = True
        for key, value in block.items():
            if not isinstance(key, str) or not isinstance(value, str):
                continue
            tid = key.strip()
            text = value.strip()
            if not tid or not text or tid in mapping:
                continue
            mapping[tid] = text
    if found_nested:
        return mapping
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, str) and key.strip() and value.strip():
            mapping[key.strip()] = value.strip()
    return mapping


def load_block_instruction_map(path: str | Path) -> dict[str, str]:
    """Load a HumanTouch2 task/block ID → English instruction JSON mapping.

    A flat object is accepted, as are the task-name mapping files used by
    HumanTouch1.  HumanTouch2 directory names encode the task as
    ``..._TASK_A001_...``; older captures may call the same value
    ``blockIndex``, so both schemas intentionally remain supported.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"instruction map must be a JSON object: {path}")
    nested_keys = (
        "task_id_to_instruction",
        "short_id_to_instruction",
        "block_index_to_instruction",
        "blockIndexToInstruction",
        "block_index_to_task_description",
        "blockIndexToTaskDescription",
        "instructions",
        "task_id_to_task_name",
        "short_id_to_task_name",
    )
    candidates: list[Mapping[str, Any]] = []
    for key in nested_keys:
        value = payload.get(key)
        if isinstance(value, Mapping):
            candidates.append(value)
    if not candidates:
        candidates.append(payload)
    mapping: dict[str, str] = {}
    for candidate in candidates:
        for key, value in candidate.items():
            if not isinstance(key, str) or not isinstance(value, str):
                continue
            task_id = key.strip().upper()
            instruction = value.strip()
            if task_id and instruction and task_id not in mapping:
                mapping[task_id] = instruction
    return mapping


def lookup_block_instruction(
    block_index: str | None, mapping: Mapping[str, str]
) -> str | None:
    """Look up one HumanTouch2 instruction case-insensitively."""
    if not isinstance(block_index, str) or not block_index.strip():
        return None
    value = mapping.get(block_index.strip().upper())
    return value.strip() if isinstance(value, str) and value.strip() else None


def lookup_task_name(
    *ids: str | None,
    mapping: Mapping[str, str],
) -> str | None:
    """Return the episode instruction for the first matching full or short task id."""
    seen: set[str] = set()
    candidates: list[str] = []
    for raw in ids:
        if not isinstance(raw, str):
            continue
        key = raw.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        candidates.append(key)
        short = short_task_id(key)
        if short and short not in seen:
            seen.add(short)
            candidates.append(short)
    for key in candidates:
        name = mapping.get(key)
        if isinstance(name, str) and name.strip():
            return name.strip()
    for key in candidates:
        names = {
            value.strip()
            for map_key, value in mapping.items()
            if isinstance(value, str)
            and value.strip()
            and (map_key == key or map_key.startswith(f"{key}_"))
        }
        if len(names) == 1:
            return next(iter(names))
    return None


def resolve_raw_instructions(
    episode_meta: Mapping[str, Any] | None,
) -> list[list[Any]]:
    """Subtask rows only when this episode has a review JSON (or explicit list)."""
    if not isinstance(episode_meta, Mapping):
        return []
    direct = episode_meta.get("instructions")
    if isinstance(direct, list) and direct:
        rows: list[list[Any]] = []
        for item in direct:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            caption = str(item[0]).strip()
            if not caption:
                continue
            try:
                rows.append([caption, int(item[1]), int(item[2])])
            except (TypeError, ValueError):
                continue
        if rows:
            return rows
    path = episode_meta.get("annotation_json")
    if not isinstance(path, str) or not path:
        qc = episode_meta.get("tactile_qc")
        if isinstance(qc, Mapping):
            path = qc.get("annotation_json")
    if not isinstance(path, str) or not path or not Path(path).is_file():
        return []
    return parse_review_subtask_instructions(load_review_json(path))


def resolve_egopro_instruction(
    source: str | Path,
    *,
    annotations_root: str | Path | None = None,
) -> str:
    """Episode-level EgoPro instruction from sibling annotation ``task_name``.

    Looks up ``<dataset>/annotations`` next to ``<dataset>/mcap``. Raises if
    the matching ``*semantic_review.json`` or its ``task_name`` is missing.
    """
    source = Path(source)
    root = (
        Path(annotations_root)
        if annotations_root is not None
        else annotations_root_from_mcap(source)
    )
    if not root.is_dir():
        raise ValueError(
            f"EgoPro annotations directory not found next to mcap: {root} "
            f"(source={source})"
        )
    review = find_egopro_semantic_review(source, root)
    if review is None:
        raise ValueError(
            "No semantic_review.json for "
            f"{source.name} under {root} "
            f"(expected <date>/<block>/{source.stem}*semantic_review.json)"
        )
    name = _task_name_from_review(review)
    if not name:
        raise ValueError(f"semantic_review.json has empty task_name: {review}")
    return name


def load_egopro_camera_calibration(path: str | Path) -> dict[str, Any]:
    """Load EgoPro camera calibration attachments from one annotated MCAP."""
    try:
        from mcap.reader import make_reader
    except ImportError as error:
        raise ImportError("Reading EgoPro requires the `mcap` package") from error

    source_path = Path(path)
    extrinsics: dict[str, Any] | None = None
    intrinsics: dict[str, Any] | None = None
    with source_path.open("rb") as stream:
        reader = make_reader(stream)
        for attachment in reader.iter_attachments():
            name = attachment.name
            if name.endswith("calibration_extrinsics.json"):
                extrinsics = json.loads(attachment.data)
            elif name.endswith("calibration_intrinsics_A3.json") or (
                "calibration_intrinsics" in name and name.endswith(".json")
            ):
                intrinsics = json.loads(attachment.data)
    if extrinsics is None:
        raise ValueError(f"No EgoPro camera extrinsics attachment in {source_path}")
    return {"extrinsics": extrinsics, "intrinsics": intrinsics}


def egopro_frame_key(payload: Mapping[str, Any]) -> int:
    """Return the 60 Hz frame index from EgoPro/HumanTouch2 conventions."""
    frame = payload.get("frame_60")
    if frame is None:
        frame = payload.get("frameIndex")
    if frame is None:
        frame = payload.get("sequence_id")
    if frame is None:
        raise KeyError(
            "Motion message has none of frame_60, frameIndex, or sequence_id"
        )
    return int(frame)


def _egopro_v3(value: object, size: int) -> np.ndarray | None:
    if isinstance(value, Mapping):
        value = [value.get(key) for key in ("x", "y", "z", "w")[:size]]
    try:
        result = np.asarray(value, dtype=np.float64).reshape(-1)[:size]
    except (TypeError, ValueError):
        return None
    return result if result.shape == (size,) and np.isfinite(result).all() else None


def normalize_egopro_skeleton_row(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize v2 ``/manus/*`` bones into the legacy ``joints`` row shape.

    The real-robot HumanTouch2 exporter writes the canonical 21 MANUS names
    (``wrist``, ``thumb_cmc``, ``index_mcp``, ...), while older EgoPro v2
    files already use the 25 prefixed names consumed by ``_egopro_chain_names``.
    Normalize only the former layout and synthesize the four unpublished
    non-thumb metacarpal roots exactly as the canonical HumanTouch2 reader.
    """
    joints: list[dict[str, Any]] = []
    for bone in payload.get("bones") or payload.get("joints") or []:
        if not isinstance(bone, Mapping):
            continue
        name = bone.get("name")
        position = _egopro_v3(
            bone.get("position_m") or bone.get("position") or bone.get("pos"), 3
        )
        if name is None or position is None:
            continue
        orientation = _egopro_v3(
            bone.get("orientation_xyzw")
            or bone.get("rotation")
            or bone.get("rot"),
            4,
        )
        joint: dict[str, Any] = {
            "name": str(name),
            "position_m": position.tolist(),
            "valid": bool(bone.get("valid", True)),
            "tracking_confidence": float(bone.get("tracking_confidence", 1.0)),
        }
        if orientation is not None and float(np.linalg.norm(orientation)) > 1e-8:
            joint["orientation_xyzw"] = orientation.tolist()
        joints.append(joint)

    side = str(payload.get("hand") or "").strip().lower()
    source_names = {str(joint["name"]) for joint in joints}
    if side in {"left", "right"} and "wrist" in source_names:
        prefix = "l" if side == "left" else "r"
        renamed: list[dict[str, Any]] = []
        for joint in joints:
            target = _HUMANTOUCH2_JOINT_NAMES.get(str(joint["name"]))
            if target is None:
                continue
            normalized = dict(joint)
            normalized["name"] = f"{prefix}_{target}"
            renamed.append(normalized)
        by_name = {str(joint["name"]): joint for joint in renamed}
        wrist = by_name.get(f"{prefix}_hand")
        if wrist is not None:
            wrist_position = np.asarray(wrist["position_m"], dtype=np.float64)
            for finger in ("index", "middle", "ring", "pinky"):
                mcp = by_name.get(f"{prefix}_{finger}_proximal")
                if mcp is None:
                    continue
                mcp_position = np.asarray(mcp["position_m"], dtype=np.float64)
                root = dict(wrist)
                root["name"] = f"{prefix}_{finger}_metacarpal"
                root["position_m"] = (
                    wrist_position * 0.6 + mcp_position * 0.4
                ).tolist()
                root["valid"] = bool(wrist.get("valid", False)) and bool(
                    mcp.get("valid", False)
                )
                root["tracking_confidence"] = min(
                    float(wrist.get("tracking_confidence", 1.0)),
                    float(mcp.get("tracking_confidence", 1.0)),
                )
                root["synthetic"] = "interpolate_0.4_wrist_to_mcp"
                renamed.append(root)
        joints = renamed
    row = {"valid": bool(payload.get("valid", True)), "joints": joints}
    coordinate_frame = payload.get("coordinateFrame") or payload.get("coordinate_frame")
    if coordinate_frame:
        row["coordinate_frame"] = str(coordinate_frame)
    return row


def normalize_egopro_pose_row(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """Normalize v2 tracker pose rows into legacy ``position_m`` rows."""
    position = _egopro_v3(
        payload.get("position_m") or payload.get("position") or payload.get("pos"), 3
    )
    orientation = _egopro_v3(
        payload.get("orientation_xyzw")
        or payload.get("rotation")
        or payload.get("orientation"),
        4,
    )
    if position is None or orientation is None:
        return None
    return {
        "valid": bool(payload.get("valid", True)),
        "position_m": position.tolist(),
        "orientation_xyzw": orientation.tolist(),
    }


_HUMANTOUCH2_JOINT_NAMES = {
    "wrist": "hand",
    "thumb_cmc": "thumb_metacarpal",
    "thumb_mcp": "thumb_proximal",
    "thumb_ip": "thumb_distal",
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
    "little_mcp": "pinky_proximal",
    "little_pip": "pinky_intermediate",
    "little_dip": "pinky_distal",
    "little_tip": "pinky_tip",
}


def normalize_humantouch2_pose_row(
    observation: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Unwrap one canonical HumanTouch2 pose observation."""
    payload = observation.get("payload")
    if not isinstance(payload, Mapping):
        return None
    row = normalize_egopro_pose_row(payload)
    if row is None:
        return None
    row["valid"] = bool(observation.get("is_valid", True)) and bool(
        row.get("valid", True)
    )
    return row


def normalize_humantouch2_skeleton_row(
    observation: Mapping[str, Any], side: str
) -> dict[str, Any]:
    """Convert canonical 21-joint HumanTouch2 rows to EgoPro's 25-point names.

    HumanTouch2 does not publish the four non-thumb metacarpal-base joints.
    Those roots are estimated 40% along each measured wrist-to-MCP ray; the
    measured wrist/MCP/PIP/DIP/tip values remain untouched. This preserves the
    five-point chain expected by Wuji retargeting and tactile geometry without
    making the world-frame palm basis degenerate.
    """
    if side not in {"left", "right"}:
        raise ValueError(f"Unknown HumanTouch2 hand side: {side!r}")
    prefix = "l" if side == "left" else "r"
    payload = observation.get("payload")
    payload = payload if isinstance(payload, Mapping) else {}
    joints: list[dict[str, Any]] = []
    wrist: dict[str, Any] | None = None
    for item in payload.get("joints") or []:
        if not isinstance(item, Mapping):
            continue
        source_name = str(item.get("joint_name") or "")
        target_name = _HUMANTOUCH2_JOINT_NAMES.get(source_name)
        pose = item.get("joint_pose")
        if target_name is None or not isinstance(pose, Mapping):
            continue
        position = _egopro_v3(pose.get("position_m"), 3)
        orientation = _egopro_v3(pose.get("orientation_xyzw"), 4)
        confidence = float(pose.get("tracking_confidence", 1.0))
        if position is None:
            continue
        joint: dict[str, Any] = {
            "name": f"{prefix}_{target_name}",
            "position_m": position.tolist(),
            "valid": bool(observation.get("is_valid", True)) and confidence > 0.0,
            "tracking_confidence": float(np.clip(confidence, 0.0, 1.0)),
        }
        if orientation is not None:
            joint["orientation_xyzw"] = orientation.tolist()
        joints.append(joint)
        if source_name == "wrist":
            wrist = joint
    if wrist is not None:
        by_target = {str(joint["name"]): joint for joint in joints}
        wrist_position = np.asarray(wrist["position_m"], dtype=np.float64)
        for finger in ("index", "middle", "ring", "pinky"):
            root = dict(wrist)
            root["name"] = f"{prefix}_{finger}_metacarpal"
            mcp = by_target.get(f"{prefix}_{finger}_proximal")
            if mcp is not None:
                mcp_position = np.asarray(mcp["position_m"], dtype=np.float64)
                # Approximate the unpublished metacarpal base along its
                # measured wrist→MCP ray. Keeping the four rays distinct is
                # required for a non-degenerate world-frame palm basis.
                root["position_m"] = (
                    wrist_position * 0.6 + mcp_position * 0.4
                ).tolist()
                root["tracking_confidence"] = min(
                    float(root.get("tracking_confidence", 1.0)),
                    float(mcp.get("tracking_confidence", 1.0)),
                )
            root["synthetic"] = "interpolate_0.4_wrist_to_mcp"
            joints.append(root)
    return {
        "valid": bool(observation.get("is_valid", True)) and bool(joints),
        "joints": joints,
        "coordinate_frame": "manus_core_unity_world",
    }


def egopro_head_image_is_inverted(extrinsics: dict[str, Any]) -> bool:
    """Return whether ``/ego_video/head`` JPEG needs a 180° upright correction.

    EgoPro stores ``camera_from_head_tracker`` in
    ``Calibration/camera/calibration_extrinsics.json``. Local demo sessions use
    ``rotation_rpy_deg ≈ [50, 178, 180]``, which mounts the head camera inverted
    relative to an upright OpenCV image (arms enter from the top of the raw
    frame). ``unity_to_opencv_y_flip`` only documents Unity→OpenCV axis mapping
    for overlays; it is not itself the RGB upright flag.
    """
    rpy = extrinsics.get("rotation_rpy_deg")
    if not isinstance(rpy, (list, tuple)) or len(rpy) != 3:
        return False
    pitch = abs(float(rpy[1]))
    yaw = abs(float(rpy[2]))
    return pitch >= 150.0 and yaw >= 150.0


EGOPRO_HEAD_IMAGE_ROTATION_POLICY: dict[str, str] = {
    # Keep the legacy calibration heuristic for the original EgoPro path.
    "egopro": "from_calibration",
    # Keep canonical HumanTouch2 behavior unchanged.
    "humantouch2": "from_calibration",
    # HumanTouch2 real-robot UVC JPEGs are already visually upright.
    "humantouch2_real_robot": "none",
}


def base_source_dataset_name(source_dataset: str | None = None) -> str:
    """Return the dataset key without a downstream provenance suffix.

    Retargeting appends a stage suffix to ``UnifiedRobotMotion.source`` (for
    example ``:camera_aligned_retargeted_to_wujihand2``).  Rotation policy is about
    the original dataset, so that suffix must not change the selected policy.
    """
    source = str(source_dataset or "egopro").strip().lower().replace("-", "_")
    return source.split(":", 1)[0]


def egopro_head_image_rotation_policy(source_dataset: str | None = None) -> str:
    """Return the explicit head-image rotation policy for a source dataset."""
    source = base_source_dataset_name(source_dataset)
    return EGOPRO_HEAD_IMAGE_ROTATION_POLICY.get(source, "from_calibration")


def egopro_head_image_is_inverted_for_source(
    extrinsics: dict[str, Any],
    source_dataset: str | None = None,
) -> bool:
    """Return whether a source-specific head JPEG should be rotated 180°."""
    policy = egopro_head_image_rotation_policy(source_dataset)
    if policy == "none":
        return False
    if policy == "from_calibration":
        return egopro_head_image_is_inverted(extrinsics)
    raise ValueError(f"Unsupported head image rotation policy: {policy!r}")


def egopro_camera_from_head_tracker(
    extrinsics: dict[str, Any],
    *,
    source_dataset: str | None = None,
) -> np.ndarray:
    """Build upright OpenCV ``camera_from_head_tracker`` like the overlay tool.

    The overlay first expresses world points in the head frame, applies the
    Unity Y-up → OpenCV Y-down convention, and then applies the calibrated
    camera offset. Its final 180° image correction is represented here as an
    upright optical-axis transform.
    """
    convention = np.eye(4, dtype=np.float64)
    if bool(extrinsics.get("unity_to_opencv_y_flip", False)):
        convention[1, 1] = -1.0
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rpy_deg_matrix(extrinsics["rotation_rpy_deg"])
    transform[:3, 3] = np.asarray(extrinsics["translation_m"], dtype=np.float64)
    transform = transform @ convention
    if egopro_head_image_is_inverted_for_source(extrinsics, source_dataset):
        # Align 3-D camera axes with the source-specific RGB convention.
        transform = optical_upright_from_inverted(transform)
    return transform


def egopro_intrinsics_vector(intrinsics: dict[str, Any] | None) -> np.ndarray:
    if intrinsics is None:
        return np.full(4, np.nan, dtype=np.float64)
    matrix = intrinsics.get("camera_matrix", {})
    if "fx" in matrix:
        return np.asarray(
            [matrix["fx"], matrix["fy"], matrix["cx"], matrix["cy"]],
            dtype=np.float64,
        )
    values = np.asarray(matrix.get("matrix"), dtype=np.float64)
    return np.asarray([values[0, 0], values[1, 1], values[0, 2], values[1, 2]], dtype=np.float64)


def egopro_intrinsics_image_size(
    intrinsics: dict[str, Any] | None,
    *,
    default_hw: tuple[int, int] = (1080, 1920),
) -> tuple[int, int]:
    """Return ``(height, width)`` for the native EgoPro head calibration image."""
    if intrinsics is None:
        return default_hw
    width = intrinsics.get("image_width")
    height = intrinsics.get("image_height")
    if width is None or height is None:
        return default_hw
    return int(height), int(width)


def egopro_center_crop_box_for_aspect(
    source_hw: tuple[int, int],
    *,
    aspect_wh: tuple[int, int] = (4, 3),
) -> tuple[int, int, int, int]:
    """Return ``(x0, y0, crop_w, crop_h)`` for a centered aspect crop.

    EgoPro head RGB is 16:9; Spark output is 4:3. Default keeps full height and
    trims left/right edges (1920x1080 → 1440x1080).
    """
    source_h, source_w = (int(source_hw[0]), int(source_hw[1]))
    aspect_w, aspect_h = (int(aspect_wh[0]), int(aspect_wh[1]))
    if min(source_h, source_w, aspect_w, aspect_h) < 1:
        raise ValueError(f"Invalid sizes source_hw={source_hw} aspect_wh={aspect_wh}")
    g = math.gcd(aspect_w, aspect_h)
    aspect_w, aspect_h = aspect_w // g, aspect_h // g
    k = min(source_w // aspect_w, source_h // aspect_h)
    if k < 1:
        raise ValueError(f"Source {source_w}x{source_h} cannot fit {aspect_w}:{aspect_h}")
    crop_w, crop_h = k * aspect_w, k * aspect_h
    return (source_w - crop_w) // 2, (source_h - crop_h) // 2, crop_w, crop_h


def egopro_center_crop_to_aspect(
    image: np.ndarray,
    *,
    aspect_wh: tuple[int, int] = (4, 3),
) -> np.ndarray:
    """Center-crop an image to ``aspect_wh`` without stretching."""
    x0, y0, crop_w, crop_h = egopro_center_crop_box_for_aspect(
        (int(image.shape[0]), int(image.shape[1])),
        aspect_wh=aspect_wh,
    )
    return image[y0 : y0 + crop_h, x0 : x0 + crop_w]


def scale_intrinsics_for_output_image(
    intrinsics: np.ndarray,
    *,
    source_hw: tuple[int, int],
    output_hw: tuple[int, int],
    upright_rotate_180: bool = False,
    aspect_wh: tuple[int, int] = (4, 3),
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Map native pinhole ``[fx,fy,cx,cy]`` onto upright edge-cropped RGB.

    Pipeline matching the written head JPEG:
    1. optional ``cv2.ROTATE_180`` on the source grid (principal-point flip);
    2. centered crop to 4:3 (trim left/right on 16:9 EgoPro);
    3. uniform ``INTER_AREA`` scale to 640x480.

    Returns scaled intrinsics and crop/scale provenance for dataset attrs.
    """
    values = np.asarray(intrinsics, dtype=np.float64).reshape(4)
    source_h, source_w = (int(source_hw[0]), int(source_hw[1]))
    output_h, output_w = (int(output_hw[0]), int(output_hw[1]))
    if min(source_h, source_w, output_h, output_w) < 1:
        raise ValueError(
            f"Invalid image sizes source_hw={source_hw} output_hw={output_hw}"
        )
    x0, y0, crop_w, crop_h = egopro_center_crop_box_for_aspect(
        (source_h, source_w),
        aspect_wh=aspect_wh,
    )
    scale = output_w / crop_w
    scale_y = output_h / crop_h
    if not np.isclose(scale, scale_y, rtol=0, atol=1e-9):
        raise ValueError(
            f"Crop {crop_w}x{crop_h} is not the same aspect as output "
            f"{output_w}x{output_h}"
        )
    meta: dict[str, float | int | str] = {
        "crop_x0": int(x0),
        "crop_y0": int(y0),
        "crop_width": int(crop_w),
        "crop_height": int(crop_h),
        "uniform_scale": float(scale),
        "resize": (
            f"{source_w}x{source_h}_center_crop_{crop_w}x{crop_h}"
            f"_to_{output_w}x{output_h}_inter_area"
        ),
    }
    if not np.all(np.isfinite(values)):
        return values.copy(), meta
    fx, fy, cx, cy = values.tolist()
    if upright_rotate_180:
        cx = float(source_w - 1) - cx
        cy = float(source_h - 1) - cy
    cx = cx - float(x0)
    cy = cy - float(y0)
    scaled = np.asarray(
        [fx * scale, fy * scale, cx * scale, cy * scale],
        dtype=np.float64,
    )
    return scaled, meta


def _egopro_chain_names(side: str) -> list[list[str]]:
    prefix = "l" if side == "left" else "r"
    return [
        [
            f"{prefix}_hand",
            f"{prefix}_thumb_metacarpal",
            f"{prefix}_thumb_proximal",
            f"{prefix}_thumb_distal",
            f"{prefix}_thumb_tip",
        ],
        *[
            [
                f"{prefix}_{finger}_metacarpal",
                f"{prefix}_{finger}_proximal",
                f"{prefix}_{finger}_intermediate",
                f"{prefix}_{finger}_distal",
                f"{prefix}_{finger}_tip",
            ]
            for finger in ("index", "middle", "ring", "pinky")
        ],
    ]


def read_egopro_reference(
    path: str | Path,
    *,
    tactile_collector: Any | None = None,
    annotations_root: str | Path | None = None,
    resolve_instruction: bool = True,
    source_dataset: str = "egopro",
) -> ReferenceFrameGeometry:
    """Read EgoPro world-frame geometry plus head-camera extrinsics (read-only)."""
    try:
        from mcap.reader import make_reader
    except ImportError as error:
        raise ImportError("Reading EgoPro requires the `mcap` package") from error

    source_path = Path(path)
    calibration = load_egopro_camera_calibration(source_path)
    camera_from_head = egopro_camera_from_head_tracker(
        calibration["extrinsics"],
        source_dataset=source_dataset,
    )
    intrinsics = egopro_intrinsics_vector(calibration.get("intrinsics"))

    skeleton_frames: dict[tuple[str, int], dict[str, Any]] = {}
    wrist_frames: dict[tuple[str, int], dict[str, Any]] = {}
    head_frames: dict[int, dict[str, Any]] = {}
    session_info: dict[str, Any] = {}
    task_info: dict[str, Any] = {}

    with source_path.open("rb") as stream:
        reader = make_reader(stream)
        for _, channel, message in reader.iter_messages():
            if channel.message_encoding != "json":
                continue
            topic = channel.topic
            if tactile_collector is not None and topic in {
                "/tactile/raw/left",
                "/tactile/raw/right",
                "/tactile/patch_pressure",
            }:
                tactile_collector.add(topic, json.loads(message.data))
                continue
            if topic in {"/manus/left", "/manus/right"}:
                payload = json.loads(message.data)
                side = topic.rsplit("/", 1)[-1]
                skeleton_frames[(side, egopro_frame_key(payload))] = (
                    normalize_egopro_skeleton_row(payload)
                )
                continue
            if topic == "/tracking/vive_tracker_manus":
                payload = json.loads(message.data)
                frame = egopro_frame_key(payload)
                head = normalize_egopro_pose_row(payload.get("head") or {})
                if head is not None:
                    head_frames[frame] = head
                for side, key in (("left", "left_wrist"), ("right", "right_wrist")):
                    wrist = normalize_egopro_pose_row(payload.get(key) or {})
                    if wrist is not None:
                        wrist_frames[(side, frame)] = wrist
                continue
            if topic == "/vive/wrist":
                payload = json.loads(message.data)
                frame = egopro_frame_key(payload)
                role = str(payload.get("role", ""))
                if role == "head":
                    head = normalize_egopro_pose_row(payload)
                    if head is not None:
                        head_frames[frame] = head
                elif role in {"left_wrist", "right_wrist"}:
                    side = role.split("_", 1)[0]
                    wrist = normalize_egopro_pose_row(payload)
                    if wrist is not None:
                        wrist_frames[(side, frame)] = wrist
                continue
            if not (
                topic.startswith("/hand_skeleton/")
                or topic.startswith("/pose/wrist/")
                or topic in {"/pose/head", "/meta/session", "/meta/block"}
            ):
                continue
            payload = json.loads(message.data)
            if topic == "/meta/session":
                session_info = payload
            elif topic == "/meta/block":
                task_info = payload
            elif topic == "/pose/head":
                head_frames[egopro_frame_key(payload)] = payload
            elif topic.startswith("/hand_skeleton/"):
                side = topic.rsplit("/", 1)[-1]
                skeleton_frames[(side, egopro_frame_key(payload))] = payload
            else:
                side = topic.rsplit("/", 1)[-1]
                wrist_frames[(side, egopro_frame_key(payload))] = payload

    frame_ids = (
        [frame for _, frame in skeleton_frames]
        + [frame for _, frame in wrist_frames]
        + list(head_frames)
    )
    if not frame_ids:
        raise ValueError(f"No EgoPro hand motion found in {source_path}")
    length = int(session_info.get("num_frames_60", max(frame_ids) + 1))
    fps = float(task_info.get("frame_rate_hz", 60.0))
    timestamps = np.arange(length, dtype=np.float64) / fps
    wrist_poses = np.full((length, 2, 7), np.nan, dtype=np.float64)
    wrist_valid = np.zeros((length, 2), dtype=bool)
    joints_ref = np.full((length, 2, 5, 5, 3), np.nan, dtype=np.float64)
    joints_local = np.full((length, 2, 5, 5, 3), np.nan, dtype=np.float64)
    confidence = np.zeros((length, 2, 5, 5), dtype=np.float64)
    camera_from_world = np.full((length, 4, 4), np.nan, dtype=np.float64)
    camera_valid = np.zeros(length, dtype=bool)

    for frame, head in head_frames.items():
        if frame >= length or not bool(head.get("valid", False)):
            continue
        position = np.asarray(head["position_m"], dtype=np.float64)
        xyzw = np.asarray(head["orientation_xyzw"], dtype=np.float64)
        world_from_head = pose_wxyz_to_matrix(
            np.concatenate([position, xyzw[[3, 0, 1, 2]]])
        )
        # Match scripts/render_egocentric_skeleton_overlay.py exactly:
        # world → head tracker → calibrated upright camera.
        camera_from_world[frame] = camera_from_head @ invert_transform(world_from_head)
        camera_valid[frame] = True

    for hand_index, side in enumerate(("left", "right")):
        chains = _egopro_chain_names(side)
        for frame in range(length):
            wrist = wrist_frames.get((side, frame))
            skeleton = skeleton_frames.get((side, frame))
            if wrist is not None and bool(wrist.get("valid", False)):
                position = np.asarray(wrist["position_m"], dtype=np.float64)
                xyzw = np.asarray(wrist["orientation_xyzw"], dtype=np.float64)
                wrist_poses[frame, hand_index] = np.concatenate([position, xyzw[[3, 0, 1, 2]]])
                wrist_valid[frame, hand_index] = True
            if skeleton is None or not bool(skeleton.get("valid", False)):
                continue
            by_name = {joint["name"]: joint for joint in skeleton.get("joints", [])}
            points = np.full((5, 5, 3), np.nan, dtype=np.float64)
            weights = np.zeros((5, 5), dtype=np.float64)
            for finger_index, names in enumerate(chains):
                for point_index, name in enumerate(names):
                    joint = by_name.get(name)
                    if joint is not None and bool(joint.get("valid", False)):
                        points[finger_index, point_index] = joint["position_m"]
                        weights[finger_index, point_index] = 1.0
            if not wrist_valid[frame, hand_index]:
                hand_joint = by_name.get(chains[0][0])
                if hand_joint is not None and bool(hand_joint.get("valid", False)):
                    position = np.asarray(hand_joint["position_m"], dtype=np.float64)
                    xyzw = np.asarray(hand_joint["orientation_xyzw"], dtype=np.float64)
                    wrist_poses[frame, hand_index] = np.concatenate(
                        [position, xyzw[[3, 0, 1, 2]]]
                    )
                    wrist_valid[frame, hand_index] = True
            if wrist_valid[frame, hand_index]:
                joints_ref[frame, hand_index] = points
                # Same anchor and mirror correction as read_egopro: hand-root
                # anchored palm frame with z negated (left-handed source data).
                hand_root_ok = weights[0, 0] > 0 and np.all(np.isfinite(points[0, 0]))
                if hand_root_ok:
                    local, valid = canonicalize_hand_points(
                        points, points[0, 0], weights, knuckle_index=1
                    )
                    if valid:
                        local[..., 2] *= -1.0
                        joints_local[frame, hand_index] = local
                        confidence[frame, hand_index] = weights
        valid_indices = np.flatnonzero(wrist_valid[:, hand_index])
        if len(valid_indices):
            wrist_poses[valid_indices, hand_index] = ensure_quaternion_continuity(
                wrist_poses[valid_indices, hand_index]
            )

    result = ReferenceFrameGeometry(
        timestamps=timestamps,
        wrist_poses_ref=wrist_poses,
        wrist_valid=wrist_valid,
        joints_ref=joints_ref,
        joint_confidence=confidence,
        joints_local=joints_local,
        camera_from_world=camera_from_world,
        camera_valid=camera_valid,
        intrinsics=intrinsics,
        source="egopro",
        instruction=(
            resolve_egopro_instruction(
                source_path,
                annotations_root=annotations_root,
            )
            if resolve_instruction
            else ""
        ),
        metadata={
            "source_path": str(source_path),
            "session_id": session_info.get("session_id"),
            "quality_status": session_info.get("quality_status"),
            "source_fps": fps,
            "coordinate_frame": "egopro_world",
            "hand_local_frame": "hand_root_anchor_mirror_corrected",
            "camera_frame": "opencv_upright_head",
            "head_image_rotation_policy": egopro_head_image_rotation_policy(
                source_dataset
            ),
            "camera_from_world_composition": "camera_from_head @ inv(world_from_head)",
            "camera_from_world_status": "overlay_verified",
            "source_readonly": True,
            "extrinsics": calibration["extrinsics"],
        },
    )
    if tactile_collector is not None:
        tactile_collector.finalize(
            skeleton={
                side: {
                    frame: payload
                    for (row_side, frame), payload in skeleton_frames.items()
                    if row_side == side
                }
                for side in ("left", "right")
            },
            wrist={
                side: {
                    frame: payload
                    for (row_side, frame), payload in wrist_frames.items()
                    if row_side == side
                }
                for side in ("left", "right")
            },
            head=head_frames,
            calibration=calibration,
        )
    return result


def read_humantouch2_real_robot_reference(
    path: str | Path,
    *,
    tactile_collector: Any | None = None,
) -> ReferenceFrameGeometry:
    """Read the HumanTouch2 real-robot MCAP layout.

    This release uses the EgoPro-v2 transport topics (``/manus/*``,
    ``/tracking/*``, and ``/ego_video/*``), but remains a distinct dataset so
    downstream metadata and format-specific video handling are unambiguous.
    """
    source_path = Path(path)
    result = read_egopro_reference(
        source_path,
        tactile_collector=tactile_collector,
        resolve_instruction=False,
        source_dataset="humantouch2_real_robot",
    )
    block_index = humantouch2_block_index_from_path(source_path) or ""
    result.source = "humantouch2_real_robot"
    result.metadata = {
        **result.metadata,
        "dataset_name": "humantouch2_real_robot",
        "source_coordinate_frame": "manus_core_unity_world",
        "camera_from_world_status": "humantouch2_real_robot",
        "head_video_pre_rotated_180": False,
        "non_thumb_metacarpal_roots": "synthetic_0.4_wrist_to_mcp",
        "block_index": block_index,
        "block_id": source_path.parent.name,
    }
    return result


def read_humantouch2_reference(
    path: str | Path,
    *,
    tactile_collector: Any | None = None,
    instruction_map: str | Path | None = None,
) -> ReferenceFrameGeometry:
    """Read one canonical HumanTouch2 H264 MCAP without modifying the source.

    HumanTouch2 publishes aligned 60 Hz JSON observations under ``/human/*``
    and H264 camera streams under ``/views/*``. Its hand skeleton has 21
    measured joints, so missing non-thumb metacarpal roots are interpolated
    along wrist-to-MCP rays while all measured joints retain native values.
    """
    try:
        from mcap.reader import make_reader
    except ImportError as error:
        raise ImportError("Reading HumanTouch2 requires the `mcap` package") from error

    source_path = Path(path)
    calibration = load_egopro_camera_calibration(source_path)
    camera_from_head = egopro_camera_from_head_tracker(calibration["extrinsics"])
    intrinsics = egopro_intrinsics_vector(calibration.get("intrinsics"))

    skeleton_frames: dict[tuple[str, int], dict[str, Any]] = {}
    wrist_frames: dict[tuple[str, int], dict[str, Any]] = {}
    head_frames: dict[int, dict[str, Any]] = {}
    global_parameters: dict[str, Any] = {}
    relevant = {
        "/human/hands/left_joints",
        "/human/hands/right_joints",
        "/human/hands/left_pose",
        "/human/hands/right_pose",
        "/human/head/pose",
        "/human/tactile/left_raw",
        "/human/tactile/right_raw",
        "/human/tactile/left_patch_force",
        "/human/tactile/right_patch_force",
        "/meta/global_parameters",
    }
    with source_path.open("rb") as stream:
        reader = make_reader(stream)
        for _schema, channel, message in reader.iter_messages(topics=list(relevant)):
            if channel.message_encoding != "json":
                continue
            topic = channel.topic
            payload = json.loads(message.data)
            if topic == "/meta/global_parameters":
                if isinstance(payload, dict):
                    global_parameters = payload
                continue
            if tactile_collector is not None and topic.startswith("/human/tactile/"):
                tactile_collector.add(topic, payload)
                continue
            frame = egopro_frame_key(payload)
            if topic == "/human/head/pose":
                pose = normalize_humantouch2_pose_row(payload)
                if pose is not None:
                    head_frames[frame] = pose
            elif topic.endswith("_joints"):
                side = "left" if "/left_" in topic else "right"
                skeleton_frames[(side, frame)] = normalize_humantouch2_skeleton_row(
                    payload, side
                )
            elif topic.endswith("_pose"):
                side = "left" if "/left_" in topic else "right"
                pose = normalize_humantouch2_pose_row(payload)
                if pose is not None:
                    wrist_frames[(side, frame)] = pose

    frame_ids = (
        [frame for _side, frame in skeleton_frames]
        + [frame for _side, frame in wrist_frames]
        + list(head_frames)
    )
    if not frame_ids:
        raise ValueError(f"No HumanTouch2 hand/head motion found in {source_path}")
    length = max(frame_ids) + 1
    fps = 60.0
    timestamps = np.arange(length, dtype=np.float64) / fps
    wrist_poses = np.full((length, 2, 7), np.nan, dtype=np.float64)
    wrist_valid = np.zeros((length, 2), dtype=bool)
    joints_ref = np.full((length, 2, 5, 5, 3), np.nan, dtype=np.float64)
    joints_local = np.full((length, 2, 5, 5, 3), np.nan, dtype=np.float64)
    confidence = np.zeros((length, 2, 5, 5), dtype=np.float64)
    camera_from_world = np.full((length, 4, 4), np.nan, dtype=np.float64)
    camera_valid = np.zeros(length, dtype=bool)

    for frame, head in head_frames.items():
        if frame >= length or not bool(head.get("valid", False)):
            continue
        position = np.asarray(head["position_m"], dtype=np.float64)
        xyzw = np.asarray(head["orientation_xyzw"], dtype=np.float64)
        world_from_head = pose_wxyz_to_matrix(
            np.concatenate([position, xyzw[[3, 0, 1, 2]]])
        )
        camera_from_world[frame] = camera_from_head @ invert_transform(world_from_head)
        camera_valid[frame] = True

    for hand_index, side in enumerate(("left", "right")):
        chains = _egopro_chain_names(side)
        for frame in range(length):
            wrist = wrist_frames.get((side, frame))
            skeleton = skeleton_frames.get((side, frame))
            if wrist is not None and bool(wrist.get("valid", False)):
                position = np.asarray(wrist["position_m"], dtype=np.float64)
                xyzw = np.asarray(wrist["orientation_xyzw"], dtype=np.float64)
                wrist_poses[frame, hand_index] = np.concatenate(
                    [position, xyzw[[3, 0, 1, 2]]]
                )
                wrist_valid[frame, hand_index] = True
            if skeleton is None or not bool(skeleton.get("valid", False)):
                continue
            by_name = {joint["name"]: joint for joint in skeleton.get("joints", [])}
            points = np.full((5, 5, 3), np.nan, dtype=np.float64)
            weights = np.zeros((5, 5), dtype=np.float64)
            for finger_index, names in enumerate(chains):
                for point_index, name in enumerate(names):
                    joint = by_name.get(name)
                    if joint is None or not bool(joint.get("valid", False)):
                        continue
                    points[finger_index, point_index] = joint["position_m"]
                    weights[finger_index, point_index] = float(
                        np.clip(joint.get("tracking_confidence", 1.0), 0.0, 1.0)
                    )
            if not wrist_valid[frame, hand_index]:
                hand_joint = by_name.get(chains[0][0])
                if hand_joint is not None and bool(hand_joint.get("valid", False)):
                    orientation = hand_joint.get("orientation_xyzw")
                    if orientation is not None:
                        position = np.asarray(hand_joint["position_m"], dtype=np.float64)
                        xyzw = np.asarray(orientation, dtype=np.float64)
                        wrist_poses[frame, hand_index] = np.concatenate(
                            [position, xyzw[[3, 0, 1, 2]]]
                        )
                        wrist_valid[frame, hand_index] = True
            if wrist_valid[frame, hand_index]:
                joints_ref[frame, hand_index] = points
                hand_root_ok = weights[0, 0] > 0 and np.all(
                    np.isfinite(points[0, 0])
                )
                if hand_root_ok:
                    local, valid = canonicalize_hand_points(
                        points, points[0, 0], weights, knuckle_index=1
                    )
                    if valid:
                        # Canonical observations preserve the Unity/Manus
                        # left-handed world convention used by EgoPro.
                        local[..., 2] *= -1.0
                        joints_local[frame, hand_index] = local
                        confidence[frame, hand_index] = weights
        valid_indices = np.flatnonzero(wrist_valid[:, hand_index])
        if len(valid_indices):
            wrist_poses[valid_indices, hand_index] = ensure_quaternion_continuity(
                wrist_poses[valid_indices, hand_index]
            )

    block_index = str(
        global_parameters.get("blockIndex")
        or humantouch2_block_index_from_path(source_path)
        or ""
    ).strip().upper()
    mapping = (
        load_block_instruction_map(instruction_map)
        if instruction_map is not None and str(instruction_map).strip()
        else {}
    )
    instruction = lookup_block_instruction(block_index, mapping) or ""
    result = ReferenceFrameGeometry(
        timestamps=timestamps,
        wrist_poses_ref=wrist_poses,
        wrist_valid=wrist_valid,
        joints_ref=joints_ref,
        joint_confidence=confidence,
        joints_local=joints_local,
        camera_from_world=camera_from_world,
        camera_valid=camera_valid,
        intrinsics=intrinsics,
        source="humantouch2",
        instruction=instruction,
        metadata={
            "source_path": str(source_path),
            "source_fps": fps,
            "coordinate_frame": "egopro_world",
            "source_coordinate_frame": "manus_core_unity_world",
            "hand_local_frame": "hand_root_anchor_mirror_corrected",
            "camera_frame": "opencv_upright_head",
            "camera_from_world_composition": "camera_from_head @ inv(world_from_head)",
            "camera_from_world_status": "canonical_humantouch2",
            "head_video_pre_rotated_180": True,
            "non_thumb_metacarpal_roots": "synthetic_0.4_wrist_to_mcp",
            "block_index": block_index,
            "block_id": global_parameters.get("blockId"),
            "session_id": global_parameters.get("session"),
            "global_parameters": global_parameters,
            "source_readonly": True,
            "extrinsics": calibration["extrinsics"],
        },
    )
    if tactile_collector is not None:
        tactile_collector.finalize(
            skeleton={
                side: {
                    frame: payload
                    for (row_side, frame), payload in skeleton_frames.items()
                    if row_side == side
                }
                for side in ("left", "right")
            },
            wrist={
                side: {
                    frame: payload
                    for (row_side, frame), payload in wrist_frames.items()
                    if row_side == side
                }
                for side in ("left", "right")
            },
            head=head_frames,
            calibration=calibration,
        )
    return result


def is_openaoe_sample(path: str | Path) -> bool:
    """Return True when ``path`` looks like an OpenAoE sample directory."""
    root = Path(path)
    return (
        root.is_dir()
        and (root / "ego_process" / "ego_hands_reconstruction" / "hands.npz").is_file()
    )


def _openaoe_instruction(sample_dir: Path) -> str:
    annotation = sample_dir / "ego_annotation" / "ego_action_annotation.json"
    if not annotation.is_file():
        return sample_dir.name
    try:
        payload = json.loads(annotation.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return sample_dir.name
    phrases: list[str] = []
    if isinstance(payload, list):
        for segment in payload:
            actions = segment.get("atomic_action") or []
            for action in actions:
                verb = str(action.get("verb", "")).strip()
                obj = str(action.get("object", "")).strip()
                if verb and obj:
                    phrases.append(f"{verb} {obj}")
                elif verb:
                    phrases.append(verb)
                description = str(action.get("description", "")).strip()
                if description and description not in phrases:
                    phrases.append(description)
    if not phrases:
        return sample_dir.name
    # Keep a compact episode-level instruction.
    unique: list[str] = []
    for phrase in phrases:
        if phrase not in unique:
            unique.append(phrase)
    return "; ".join(unique[:8])


def _openaoe_video_hw(sample_dir: Path) -> tuple[int, int]:
    """Return ``(height, width)`` of the undistorted ego video file."""
    video_path = (
        sample_dir
        / "ego_process"
        / "ego_undistorted_video"
        / "raw_video_undistorted.mp4"
    )
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as error:
        raise ImportError("Reading OpenAoE video size requires cv2") from error
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Cannot open OpenAoE video {video_path}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if min(width, height) < 1:
        raise ValueError(f"Invalid OpenAoE video size {width}x{height}")
    return height, width


def _scale_openaoe_intrinsics(
    intrinsic: np.ndarray,
    sample_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Scale MANO/SLAM intrinsics onto the delivered undistorted video grid.

    Some OpenAoE samples ship 1280x720 video while ``video_info.json`` /
    ``camera_traj.npz`` still carry 1920x1080 principal points. Scale by the
    ratio of actual video size to the metadata resolution when they differ.
    """
    matrix = np.asarray(intrinsic, dtype=np.float64).reshape(3, 3).copy()
    video_h, video_w = _openaoe_video_hw(sample_dir)
    meta_w, meta_h = video_w, video_h
    for info_name in (
        sample_dir / "video_info.json",
        sample_dir
        / "ego_process"
        / "ego_undistorted_video"
        / "undistorted_video_info.json",
    ):
        if not info_name.is_file():
            continue
        info = json.loads(info_name.read_text(encoding="utf-8"))
        resolution = str(info.get("cameraParams", {}).get("resolution", ""))
        if "x" not in resolution.lower():
            continue
        width_s, height_s = resolution.lower().split("x", 1)
        candidate_w, candidate_h = int(width_s), int(height_s)
        # Prefer a metadata size that explains an out-of-frame principal point.
        if candidate_w >= video_w and candidate_h >= video_h:
            meta_w, meta_h = candidate_w, candidate_h
            break
    # Fallback: infer from principal point (~image center).
    cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
    if meta_w == video_w and (cx > video_w * 0.9 or cy > video_h * 0.9):
        meta_w = int(round(cx * 2))
        meta_h = int(round(cy * 2))
    scale_x = video_w / meta_w
    scale_y = video_h / meta_h
    if not (np.isclose(scale_x, 1.0) and np.isclose(scale_y, 1.0)):
        matrix[0, 0] *= scale_x
        matrix[1, 1] *= scale_y
        matrix[0, 2] *= scale_x
        matrix[1, 2] *= scale_y
    return matrix, {
        "video_width": video_w,
        "video_height": video_h,
        "metadata_width": meta_w,
        "metadata_height": meta_h,
        "intrinsic_scale_x": scale_x,
        "intrinsic_scale_y": scale_y,
    }


def read_openaoe_reference(
    path: str | Path,
    *,
    mano_model: str | Path | None = None,
    flat_hand_mean: bool = True,
    left_convention: str = "mirror_model",
) -> ReferenceFrameGeometry:
    """Read one OpenAoE sample into camera-aligned reference geometry.

    Source files under ``ego_process/`` are read-only. MANO is evaluated in the
    camera frame (``pred_*_cam``) then lifted to SLAM world with ``R_c2w`` /
    ``t_c2w``. World MANO parameters are not used for FK because MANO's root
    joint is not rotated by ``global_orient`` before ``transl`` is applied, so
    ``FK(world)`` is not equal to ``c2w @ FK(camera)``.
    """
    from .retargeting.mano.model import (
        ManoModel,
        default_mano_asset,
        forward_hand,
        joints21_to_chains,
    )

    sample_dir = Path(path)
    if not is_openaoe_sample(sample_dir):
        raise FileNotFoundError(
            f"OpenAoE sample incomplete at {sample_dir}: missing "
            "ego_process/ego_hands_reconstruction/hands.npz"
        )
    hands_path = sample_dir / "ego_process" / "ego_hands_reconstruction" / "hands.npz"
    traj_path = sample_dir / "ego_process" / "ego_hands_reconstruction" / "camera_traj.npz"
    undistorted_info = (
        sample_dir / "ego_process" / "ego_undistorted_video" / "undistorted_video_info.json"
    )
    hands = np.load(hands_path)
    if traj_path.is_file():
        traj = np.load(traj_path)
        intrinsic = np.asarray(traj["intrinsic"], dtype=np.float64)
    else:
        intrinsic = None

    fps = 30.0
    if undistorted_info.is_file():
        info = json.loads(undistorted_info.read_text(encoding="utf-8"))
        fps = float(info.get("fps", 30.0))
        if intrinsic is None:
            camera = info.get("cameraParams", {})
            intrinsic = np.array(
                [
                    [float(camera["fx_pixels"]), 0.0, float(camera["cx_pixels"])],
                    [0.0, float(camera["fy_pixels"]), float(camera["cy_pixels"])],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
    if intrinsic is None:
        raise ValueError(f"No camera intrinsics found for OpenAoE sample {sample_dir}")
    intrinsic, intrinsic_meta = _scale_openaoe_intrinsics(intrinsic, sample_dir)

    pred_valid = np.asarray(hands["pred_valid"], dtype=np.float64) > 0.5
    pred_trans_cam = np.asarray(hands["pred_trans_cam"], dtype=np.float64)
    pred_rot_cam = np.asarray(hands["pred_rot_cam"], dtype=np.float64)
    pred_hand_pose = np.asarray(hands["pred_hand_pose"], dtype=np.float64)
    pred_betas = np.asarray(hands["pred_betas"], dtype=np.float64)
    r_w2c = np.asarray(hands["R_w2c"], dtype=np.float64)
    t_w2c = np.asarray(hands["t_w2c"], dtype=np.float64)
    r_c2w = np.asarray(hands["R_c2w"], dtype=np.float64)
    t_c2w = np.asarray(hands["t_c2w"], dtype=np.float64)
    length = int(pred_trans_cam.shape[1])
    if pred_trans_cam.shape != (2, length, 3):
        raise ValueError(f"Unexpected pred_trans_cam shape {pred_trans_cam.shape}")

    asset = Path(mano_model) if mano_model is not None else default_mano_asset()
    model = ManoModel(asset)
    timestamps = np.arange(length, dtype=np.float64) / fps
    wrist_poses = np.full((length, 2, 7), np.nan, dtype=np.float64)
    wrist_valid = np.zeros((length, 2), dtype=bool)
    joints_ref = np.full((length, 2, 5, 5, 3), np.nan, dtype=np.float64)
    joints_local = np.full((length, 2, 5, 5, 3), np.nan, dtype=np.float64)
    confidence = np.zeros((length, 2, 5, 5), dtype=np.float64)
    camera_from_world = np.full((length, 4, 4), np.nan, dtype=np.float64)
    camera_valid = np.zeros(length, dtype=bool)

    for frame in range(length):
        if not np.all(np.isfinite(r_w2c[frame])) or not np.all(np.isfinite(t_w2c[frame])):
            continue
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = r_w2c[frame]
        transform[:3, 3] = t_w2c[frame]
        camera_from_world[frame] = transform
        camera_valid[frame] = abs(np.linalg.det(r_w2c[frame]) - 1.0) < 1e-3

    for hand_index, is_right in enumerate((False, True)):
        valid_frames = np.flatnonzero(pred_valid[hand_index])
        if not len(valid_frames):
            continue
        joints_cam = forward_hand(
            model,
            is_right=is_right,
            global_orient=pred_rot_cam[hand_index, valid_frames],
            hand_pose=pred_hand_pose[hand_index, valid_frames],
            betas=pred_betas[hand_index, valid_frames],
            transl=pred_trans_cam[hand_index, valid_frames],
            flat_hand_mean=flat_hand_mean,
            left_convention=left_convention,  # type: ignore[arg-type]
        )
        for local_index, frame in enumerate(valid_frames):
            joints21_cam = joints_cam[local_index]
            joints21 = (r_c2w[frame] @ joints21_cam.T).T + t_c2w[frame]
            wrist = joints21[0]
            # Wrist orientation: camera global_orient lifted by R_c2w.
            rotation_cam = _axis_angle_to_matrix_single(pred_rot_cam[hand_index, frame])
            rotation_world = r_c2w[frame] @ rotation_cam
            wrist_poses[frame, hand_index] = matrix_to_pose_wxyz(
                _pose_matrix(rotation_world, wrist)
            )
            wrist_valid[frame, hand_index] = True
            points, weights = joints21_to_chains(joints21)
            joints_ref[frame, hand_index] = points
            local, ok = canonicalize_hand_points(
                points, wrist, weights, knuckle_index=1
            )
            if ok:
                joints_local[frame, hand_index] = local
                confidence[frame, hand_index] = weights
        continuity = np.flatnonzero(wrist_valid[:, hand_index])
        if len(continuity):
            wrist_poses[continuity, hand_index] = ensure_quaternion_continuity(
                wrist_poses[continuity, hand_index]
            )

    intrinsics = np.asarray(
        [intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]],
        dtype=np.float64,
    )
    return ReferenceFrameGeometry(
        timestamps=timestamps,
        wrist_poses_ref=wrist_poses,
        wrist_valid=wrist_valid,
        joints_ref=joints_ref,
        joint_confidence=confidence,
        joints_local=joints_local,
        camera_from_world=camera_from_world,
        camera_valid=camera_valid,
        intrinsics=intrinsics,
        source="openaoe",
        instruction=_openaoe_instruction(sample_dir),
        metadata={
            "source_path": str(sample_dir),
            "source_fps": fps,
            "coordinate_frame": "openaoe_slam_world",
            "hand_local_frame": "mano_wrist_anchor_from_camera_fk",
            "camera_frame": "opencv_optical",
            "mano_fk_frame": "camera_then_c2w",
            "mano_model": str(asset),
            "mano_flat_hand_mean": bool(flat_hand_mean),
            "mano_left_convention": left_convention,
            "intrinsic_scaling": intrinsic_meta,
            "source_readonly": True,
            "video_path": str(
                sample_dir
                / "ego_process"
                / "ego_undistorted_video"
                / "raw_video_undistorted.mp4"
            ),
            "undistorted_info_path": str(undistorted_info),
        },
    )


def _pose_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _axis_angle_to_matrix_single(axis_angle: np.ndarray) -> np.ndarray:
    from .retargeting.mano.model import _axis_angle_to_matrix

    return _axis_angle_to_matrix(np.asarray(axis_angle, dtype=np.float64).reshape(1, 3))[0]
