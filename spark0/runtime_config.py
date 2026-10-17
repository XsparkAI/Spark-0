from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


_AUTO_VALUES = {"", "auto", "none", "null"}


def _is_auto(value: Any) -> bool:
    """True when a config value asks for checkpoint-side resolution.

    Non-string, non-None values (a real ``[W, H]`` list, a bool) are never
    ``auto``, so this is safe to call on every override type.
    """
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _AUTO_VALUES
    return False


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"Expected a boolean value, got {value!r}.")


def _checkpoint_run_dir(checkpoint_path: str | Path) -> Path:
    path = Path(checkpoint_path).expanduser()
    if path.is_dir():
        return path.parent if path.name in {"checkpoints", "final_model"} else path
    if path.parent.name in {"checkpoints", "final_model"}:
        return path.parent.parent
    return path.parent


def _read_include_state(config_path: Path) -> bool | None:
    if not config_path.is_file():
        return None
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    try:
        value = config["datasets"]["vla_data"]["include_state"]
    except (KeyError, TypeError):
        return None
    return _parse_bool(value)


def _load_checkpoint_configs(checkpoint_path: str | Path | None) -> list[dict]:
    """All config files saved next to a checkpoint, in lookup-priority order.

    ``config.full.yaml`` is the authoritative complete merged configuration.
    ``config.yaml`` is the trainer's accessed-only snapshot and is retained as
    a fallback for legacy checkpoints that do not have the full configuration.
    """
    if checkpoint_path in (None, "", "null", "None"):
        return []
    run_dir = _checkpoint_run_dir(checkpoint_path)
    configs = []
    for name in ("config.full.yaml", "config.yaml"):
        config_path = run_dir / name
        if not config_path.is_file():
            continue
        with config_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        if isinstance(config, dict):
            configs.append(config)
    return configs


def resolve_embodiment_spec(
    checkpoint_path: str | Path | None,
    embodiment_dir: str | Path | None = None,
) -> dict | None:
    """Resolve the training-time embodiment schema for a checkpoint.

    Resolution order (train.sh records both keys in the generated config that
    is saved next to the checkpoint, so trained runs are self-describing):
      1. ``embodiment_spec`` embedded in the checkpoint's
         ``config.full.yaml`` — the authoritative training-time schema;
         ``config.yaml`` is an accessed-only legacy fallback.
      2. ``<embodiment_dir>/<embodiment>.yaml`` looked up via the config's
         ``embodiment`` name (fallback for configs saved without the spec).

    Returns ``None`` when neither is available (legacy checkpoints).

    Either way the run-level decisions the embodiment YAML does not pin are
    re-applied from the checkpoint's own training config, so the fallback path
    cannot drift from ``configs/embodiments/*.yaml``: the pose
    switches
    (``datasets.vla_data.{state_pose_format,action_pose_format,
    pose_normalization}``) onto ``pose_representation.{state,action,
    normalization}``, and the top-level schema knobs
    (``framework.action_model.action_horizon`` plus
    ``datasets.vla_data.{state,action}_normalization`` and
    ``observation_indices``) onto the spec itself. train.sh already freezes all
    of them into the embedded spec, so this only matters for the name lookup
    and for checkpoints saved before that.
    """
    configs = _load_checkpoint_configs(checkpoint_path)
    if not configs:
        return None

    def _config_value(*path_keys: str) -> Any:
        for config in configs:
            value: Any = config
            for key in path_keys:
                value = value.get(key) if isinstance(value, Mapping) else None
                if value is None:
                    break
            if value not in (None, "", [], {}):
                return value
        return None

    def _with_run_overrides(spec: dict) -> dict:
        patched = dict(spec)
        for field, path_keys in (
            ("action_horizon", ("framework", "action_model", "action_horizon")),
            ("state_normalization", ("datasets", "vla_data", "state_normalization")),
            ("action_normalization", ("datasets", "vla_data", "action_normalization")),
            ("observation_indices", ("datasets", "vla_data", "observation_indices")),
        ):
            value = _config_value(*path_keys)
            if value is not None:
                patched[field] = value

        pose_repr = spec.get("pose_representation")
        if isinstance(pose_repr, Mapping) and pose_repr.get("keys"):
            pose_patched = dict(pose_repr)
            for field, key in (
                ("state", "state_pose_format"),
                ("action", "action_pose_format"),
                ("normalization", "pose_normalization"),
            ):
                value = _config_value("datasets", "vla_data", key)
                if value:
                    pose_patched[field] = str(value)
            patched["pose_representation"] = pose_patched
        return patched

    for config in configs:
        spec = config.get("embodiment_spec")
        if isinstance(spec, dict) and spec:
            return _with_run_overrides(spec)

    if embodiment_dir is not None:
        for config in configs:
            embodiment = config.get("embodiment")
            if not embodiment:
                continue
            spec_path = Path(embodiment_dir).expanduser() / f"{embodiment}.yaml"
            if spec_path.is_file():
                with spec_path.open("r", encoding="utf-8") as stream:
                    spec = yaml.safe_load(stream)
                if isinstance(spec, dict) and spec:
                    return _with_run_overrides(spec)
    return None


def resolve_include_state(value: Any, checkpoint_path: str | Path | None) -> bool:
    """Resolve state input from an override or checkpoint-side configuration."""

    if not _is_auto(value):
        return _parse_bool(value)
    if checkpoint_path in (None, "", "null", "None"):
        return False

    run_dir = _checkpoint_run_dir(checkpoint_path)
    for name in ("config.full.yaml", "config.yaml"):
        include_state = _read_include_state(run_dir / name)
        if include_state is not None:
            return include_state
    return False


def _parse_image_size(value: Any) -> tuple[int, int]:
    """Accept ``[W, H]`` lists/tuples, single ints and their string forms.

    Exactly two entries are required: a longer sequence is a typo (e.g. an
    ``[H, W, C]`` shape copied from the dataloader) that would otherwise be
    silently truncated to a wrong resize. PIL ``Image.resize`` takes
    ``(width, height)``; configs and this helper use that convention.
    """
    if isinstance(value, str):
        value = yaml.safe_load(value)
    if isinstance(value, bool):
        raise ValueError(f"Expected an image size like [384, 384], got {value!r}.")
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"Expected an image size like [384, 384], got {value!r}.")


def _is_hw_size(value: Any) -> bool:
    """True for a single ``[W, H]`` / ``(W, H)`` / int size (not a mapping)."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, (list, tuple)) and len(value) == 2 and not isinstance(
        value[0], (list, tuple, dict)
    )


def _short_camera_name(video_key: str) -> str:
    text = str(video_key)
    return text.split(".", 1)[-1] if "." in text else text


def resolve_active_video_keys(
    catalog_video_keys: Sequence[str],
    *,
    video_keys: Any = None,
    obs_image_size: Any = None,
) -> list[str]:
    """Resolve the active camera list for one training run.

    The embodiment YAML's ``video_keys`` is the *catalog* of cameras the robot
    / dataset can provide. Which cameras a run actually feeds the model is a
    train-config decision:

      1. Explicit ``datasets.vla_data.video_keys`` (short names or ``video.*``)
         selects a subset of the catalog (catalog order is preserved).
      2. Else, when ``obs_image_size`` is a per-camera mapping, the named
         cameras in that mapping are the enable switch (``default`` /
         ``_default`` are size fallbacks, not cameras). This couples the
         camera switch to the resize targets so a head-only recipe only
         needs to list ``cam_high``.
      3. Else (``obs_image_size`` absent) → the full catalog.

    A uniform ``obs_image_size`` (int / ``[W, H]``) is REJECTED: it would
    silently enable every catalog camera instead of selecting a subset.

    When both an explicit list and a named mapping are present they must
    name the same camera set (mapping may still carry a ``default``).
    Unknown cameras and an empty selection raise.
    """
    catalog = [str(key) for key in catalog_video_keys]
    if not catalog:
        raise ValueError("embodiment video_keys catalog is empty.")
    catalog_by_short = {_short_camera_name(key): key for key in catalog}
    if len(catalog_by_short) != len(catalog):
        raise ValueError(
            f"embodiment video_keys have duplicate short names: {catalog}."
        )

    def _normalize_requested(raw_keys: Sequence[Any], *, source: str) -> list[str]:
        requested_short: list[str] = []
        seen: set[str] = set()
        for raw in raw_keys:
            short = _short_camera_name(raw)
            if short in seen:
                raise ValueError(
                    f"{source} lists camera {short!r} more than once."
                )
            seen.add(short)
            requested_short.append(short)
        unknown = sorted(set(requested_short) - set(catalog_by_short))
        if unknown:
            raise ValueError(
                f"{source} has camera(s) {unknown} not in the embodiment "
                f"catalog {catalog}."
            )
        if not requested_short:
            raise ValueError(f"{source} selected an empty camera list.")
        # Preserve embodiment catalog order (not the request order).
        wanted = set(requested_short)
        return [key for key in catalog if _short_camera_name(key) in wanted]

    explicit: list[str] | None = None
    if video_keys is not None:
        if isinstance(video_keys, str):
            raise ValueError(
                "datasets.vla_data.video_keys must be a list of camera names, "
                f"got a string {video_keys!r}."
            )
        if not isinstance(video_keys, (list, tuple)):
            raise ValueError(
                "datasets.vla_data.video_keys must be a list of camera names, "
                f"got {video_keys!r}."
            )
        explicit = _normalize_requested(
            video_keys, source="datasets.vla_data.video_keys"
        )

    mapping_cameras: list[str] | None = None
    if obs_image_size is not None and not isinstance(obs_image_size, Mapping):
        raise ValueError(
            "datasets.vla_data.obs_image_size must be a per-camera mapping "
            "naming the active cameras, e.g. {cam_high: [384, 384]} or "
            "{cam_high: [384, 384], cam_left_wrist: [256, 256]}. A uniform "
            f"int / [W, H] (got {obs_image_size!r}) is NOT accepted: it would "
            f"silently enable the whole embodiment catalog {catalog}. Name the "
            "cameras you want explicitly (the mapping keys are the enable "
            "switch; a 'default' entry only fills unlisted sizes)."
        )
    if isinstance(obs_image_size, Mapping) and not _is_hw_size(obs_image_size):
        named = [
            key
            for key in obs_image_size
            if str(key) not in {"default", "_default"}
        ]
        if named:
            mapping_cameras = _normalize_requested(
                named, source="datasets.vla_data.obs_image_size"
            )

    if explicit is not None and mapping_cameras is not None:
        if set(explicit) != set(mapping_cameras):
            raise ValueError(
                "datasets.vla_data.video_keys and obs_image_size name different "
                f"cameras: video_keys={explicit}, "
                f"obs_image_size={mapping_cameras}. "
                "List the same cameras in both (or drop video_keys and let "
                "obs_image_size be the switch)."
            )
        return explicit
    if explicit is not None:
        return explicit
    if mapping_cameras is not None:
        return mapping_cameras
    # Neither an explicit video_keys list nor a named obs_image_size mapping:
    # fall back to the full catalog (the pre-existing behaviour when
    # obs_image_size is omitted entirely).
    return list(catalog)


def expand_obs_image_sizes(
    spec: Any, video_keys: Sequence[str], *, default: tuple[int, int] = (384, 384)
) -> list[tuple[int, int]]:
    """Expand an ``obs_image_size`` spec into one ``(W, H)`` per video key.

    Accepted forms (same as the training dataloader):
      - ``384`` / ``[384, 384]`` — uniform size for every camera
      - mapping keyed by short camera name (``cam_high``) or full video key
        (``video.cam_high``); optional ``default`` entry covers cameras not
        listed explicitly. Every ``video_keys`` entry must resolve.
    """
    if not video_keys:
        raise ValueError("expand_obs_image_sizes requires a non-empty video_keys list.")
    if isinstance(spec, str):
        spec = yaml.safe_load(spec)
    if spec is None:
        return [default for _ in video_keys]
    if _is_hw_size(spec):
        size = _parse_image_size(spec)
        return [size for _ in video_keys]
    if not isinstance(spec, Mapping):
        raise ValueError(
            "obs_image_size must be an int, [W, H], or a per-camera mapping "
            f"(e.g. {{cam_high: [384, 384], cam_left_wrist: [224, 224]}}), got {spec!r}."
        )

    # Normalize keys to short camera names; keep an optional default.
    by_short: dict[str, tuple[int, int]] = {}
    mapping_default: tuple[int, int] | None = None
    for raw_key, raw_size in spec.items():
        key = str(raw_key)
        if key in {"default", "_default"}:
            mapping_default = _parse_image_size(raw_size)
            continue
        by_short[_short_camera_name(key)] = _parse_image_size(raw_size)

    sizes: list[tuple[int, int]] = []
    missing: list[str] = []
    for video_key in video_keys:
        short = _short_camera_name(video_key)
        if short in by_short:
            sizes.append(by_short[short])
        elif mapping_default is not None:
            sizes.append(mapping_default)
        else:
            missing.append(str(video_key))
    if missing:
        known = sorted(by_short)
        raise ValueError(
            "obs_image_size mapping is missing camera(s) "
            f"{missing}; known keys={known}. Add each camera or a 'default' entry."
        )
    return sizes


# The training dataloader's fallback when datasets.vla_data.obs_image_size is
# absent (see LeRobotSingleDataset._resolve_obs_image_size).
_DEFAULT_IMAGE_SIZE = (384, 384)


def resolve_ee_in_camera_frame(
    value: Any, checkpoint_path: str | Path | None
) -> bool:
    """``auto`` → checkpoint ``datasets.vla_data.ee_in_camera_frame``; else bool.

    Missing key on legacy checkpoints → ``False`` (world-frame EE).
    """
    if not _is_auto(value):
        return _parse_bool(value)
    for config in _load_checkpoint_configs(checkpoint_path):
        try:
            flag = config["datasets"]["vla_data"]["ee_in_camera_frame"]
        except (AttributeError, KeyError, TypeError):
            continue
        if flag is not None:
            return _parse_bool(flag)
    return False


def resolve_flexible_views(
    value: Any, checkpoint_path: str | Path | None
) -> bool:
    """``auto`` → checkpoint ``datasets.vla_data.flexible_views``; else bool.

    Missing key on legacy checkpoints → ``False`` (every active camera required).
    """
    if not _is_auto(value):
        return _parse_bool(value)
    for config in _load_checkpoint_configs(checkpoint_path):
        try:
            flag = config["datasets"]["vla_data"]["flexible_views"]
        except (AttributeError, KeyError, TypeError):
            continue
        if flag is not None:
            return _parse_bool(flag)
    return False


def assert_trailing_view_mask(
    view_mask: Sequence[Any], slot_keys: Sequence[str]
) -> None:
    """Reject non-prefix ``view_mask`` patterns (the flexible_views contract).

    The VLM only ever sees the present subsequence of the slot list (no
    slot-id tokens), so a mask like ``[1, 0, 1]`` (head+right, no left)
    packs ``[head, right]`` and makes ``image[1]`` a different camera than
    under ``[1, 1, 0]`` -> ``[head, left]``. Training enforces this in the
    dataloader (``LeRobotSingleDataset._assert_trailing_absent_mask``); this
    is the deploy-side mirror so a client that drops a middle camera fails
    loudly instead of silently shifting every later view one slot left.
    """
    seen_absent = False
    for present in view_mask:
        if not present:
            seen_absent = True
        elif seen_absent:
            raise ValueError(
                "flexible_views requires absent cameras to be a trailing "
                f"suffix of the camera slots {list(slot_keys)} (got "
                f"view_mask={list(view_mask)}). Non-prefix masks make packed "
                "image positions ambiguous to the VLM (no slot-id tokens). "
                "Reorder the cameras or provide the missing middle camera."
            )


def resolve_required_views(
    value: Any,
    checkpoint_path: str | Path | None,
    video_keys: Sequence[str] | None = None,
) -> list[str]:
    """``auto`` → checkpoint ``required_views``, else explicit list; short names.

    Default when unset: first active ``video_keys`` entry (typically the head).
    """
    raw = value
    if _is_auto(value):
        raw = None
        for config in _load_checkpoint_configs(checkpoint_path):
            try:
                raw = config["datasets"]["vla_data"]["required_views"]
            except (AttributeError, KeyError, TypeError):
                continue
            break
    if raw is None:
        if video_keys:
            return [str(video_keys[0]).split(".")[-1]]
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        raise ValueError(
            f"required_views must be a list of camera names or 'auto', got {raw!r}."
        )
    return [str(name).split(".")[-1] for name in raw]


def resolve_camera_extrinsics_convention(
    semantics: Any, axes: Any
) -> tuple[str, str]:
    """Validate the deploy-side camera-extrinsics convention knobs.

    Training normalizes the stored extrinsics column to ``world_from_camera``
    and refuses an ambiguous declaration, but it applies NO axis correction —
    ``camera_axes`` is only a label in ``meta/modality.json``. Neither value
    reaches the checkpoint, so deployment cannot auto-resolve them: ``auto``
    means "the bench publishes what the pipeline recorded". Unknown values
    fail here, at startup, instead of silently mirroring every EE pose.
    """
    from .relative_pose import (
        CAMERA_AXES,
        EXTRINSICS_SEMANTICS,
        TRAINING_CAMERA_AXES,
        WORLD_FROM_CAMERA,
    )

    resolved_semantics = (
        WORLD_FROM_CAMERA if _is_auto(semantics) else str(semantics).strip().lower()
    )
    if resolved_semantics not in EXTRINSICS_SEMANTICS:
        raise ValueError(
            f"ee_camera_extrinsics_semantics={semantics!r} must be 'auto' or one of "
            f"{list(EXTRINSICS_SEMANTICS)}."
        )
    resolved_axes = (
        TRAINING_CAMERA_AXES if _is_auto(axes) else str(axes).strip().lower()
    )
    if resolved_axes not in CAMERA_AXES:
        raise ValueError(
            f"ee_camera_axes={axes!r} must be 'auto' or one of {list(CAMERA_AXES)}."
        )
    return resolved_semantics, resolved_axes


# Deploy-side tactile stream layout. The names are the batch keys the
# dataloader emits and the tactile encoder resolves by default.
TACTILE_REGION_KEY = "region_pressure"
TACTILE_TAXEL_KEY = "taxel_pressure_3d"


def condition_tactile_pressure(
    pressure: np.ndarray,
    log1p: bool,
    scale: float | None,
    *,
    center: str | None = None,
) -> np.ndarray:
    """``center`` then ``sign(p) * log1p(|p|) / scale`` — train transform.

    Delegates to ``starVLA.model.modules.tactile.pressure`` so deploy and the
    dataloader cannot drift.
    """
    from starVLA.model.modules.tactile.pressure import (
        center_pressure,
        condition_tactile_pressure as _condition,
    )

    return _condition(center_pressure(pressure, center), log1p, scale)


def _for_stream(value: Any, batch_key: str) -> Any:
    """Per-stream mappings and plain scalars share one config spelling."""
    if isinstance(value, dict):
        return value.get(batch_key)
    return value


def _as_role_flag(value: Any, default: bool = True) -> bool:
    if value is None or _is_auto(value):
        return bool(default)
    return _parse_bool(value)


_NULL_STEM = (None, "", "null", "None")


def tactile_stem_ref(value: Any, *, key: str) -> Any:
    """Normalize a top-level tactile stem pointer.

    ``null`` / absent / empty → ``None`` (off). A boolean is a common
    mistake (``tactile_fusion: true``) and is rejected — pick a stem name.
    """
    if value in _NULL_STEM:
        return None
    if isinstance(value, bool):
        raise ValueError(
            f"{key} must be null or a stem under configs/{key}/, "
            f"not a boolean ({value!r}). Example: {key}: default."
        )
    return value


_PRESSURE_VLA_KEYS = (
    ("log1p", "tactile_pressure_log1p"),
    ("scale", "tactile_pressure_scale"),
    ("center", "tactile_pressure_center"),
    ("append_delta", "tactile_append_delta"),
)


def apply_tactile_stem_pressure(vla_data: dict, tactile_cfg: Any) -> None:
    """Copy ``framework.tactile.pressure`` into ``datasets.vla_data`` when unset.

    Sensor / export conditioning lives on the fusion stem, not the recipe.
    Recipe-inline ``tactile_pressure_*`` still wins.
    """
    if not isinstance(vla_data, dict):
        raise TypeError("vla_data must be a mapping")
    tac = tactile_cfg if isinstance(tactile_cfg, dict) else {}
    pressure = tac.get("pressure")
    if not isinstance(pressure, dict):
        return
    for src, dest in _PRESSURE_VLA_KEYS:
        if src not in pressure:
            continue
        if dest not in vla_data or vla_data.get(dest) in _NULL_STEM:
            vla_data[dest] = pressure[src]


def apply_tactile_stem_enables(
    framework: dict,
    *,
    fusion_selected: bool,
    align_selected: bool,
) -> None:
    """A selected stem is on unless the recipe already set ``enable``.

    Fusion stem → ``framework.tactile.enable`` and
    ``framework.tactile.action_fusion.enable`` (encoder + DiT). Align stem
    → ``framework.tactile_align.enable``. Recipe-inline ``enable: false``
    still wins (setdefault).
    """
    if not isinstance(framework, dict):
        raise TypeError("framework must be a mapping")
    if fusion_selected:
        tac = framework.setdefault("tactile", {})
        if not isinstance(tac, dict):
            raise TypeError("framework.tactile must be a mapping")
        tac.setdefault("enable", True)
        af = tac.get("action_fusion")
        if af is None:
            tac["action_fusion"] = {"enable": True}
        elif isinstance(af, dict):
            af.setdefault("enable", True)
        else:
            raise TypeError("framework.tactile.action_fusion must be a mapping")
    if align_selected:
        ta = framework.setdefault("tactile_align", {})
        if not isinstance(ta, dict):
            raise TypeError("framework.tactile_align must be a mapping")
        ta.setdefault("enable", True)


def resolve_tactile_runtime(
    value: Any,
    checkpoint_path: str | Path | None,
    *,
    fast: Any = "auto",
    slow: Any = "auto",
) -> dict | None:
    """Deploy-side tactile spec from the checkpoint, or ``None`` when off.

    Everything is read from the config saved next to the checkpoint, so the
    server cannot disagree with the weights it loads: the stream shapes come
    from ``framework.tactile`` (what the encoder was BUILT with) and the
    pressure conditioning from ``datasets.vla_data`` (what the dataloader FED
    it). A checkpoint trained without tactile resolves to ``None`` and the
    deploy path stays byte-identical to the pre-tactile server.

    ``value`` is the serve-config override: ``auto`` (default) reads the
    checkpoint, ``false`` force-disables — useful to A/B the same checkpoint
    with and without the glove, since the model keeps its ``no_tactile``
    fallback.
    """
    if value is not None and not _is_auto(value) and not _parse_bool(value):
        return None

    tactile_cfg: Any = None
    vla_data: Any = {}
    for config in _load_checkpoint_configs(checkpoint_path):
        try:
            candidate = config["framework"]["tactile"]
        except (AttributeError, KeyError, TypeError):
            continue
        if candidate:
            tactile_cfg = candidate
            try:
                vla_data = config["datasets"]["vla_data"] or {}
            except (AttributeError, KeyError, TypeError):
                vla_data = {}
            break
    if not tactile_cfg or not _parse_bool(tactile_cfg.get("enable", False)):
        return None

    streams = tactile_cfg.get("streams") or {}
    # Construct-time flags win: deploy `true` cannot resurrect an expert
    # the checkpoint never built. `auto` follows the ckpt; `false` drops
    # that role on a both-ckpt (same architecture, unused slots = no_tactile).
    ckpt_fast = _as_role_flag(tactile_cfg.get("fast_enable"), True)
    ckpt_slow = _as_role_flag(tactile_cfg.get("slow_enable"), True)
    fast_on = ckpt_fast and _as_role_flag(fast, True)
    slow_on = ckpt_slow and _as_role_flag(slow, True)
    log1p_cfg = vla_data.get("tactile_pressure_log1p")
    scale_cfg = vla_data.get("tactile_pressure_scale")
    center_cfg = vla_data.get("tactile_pressure_center")
    delta_cfg = vla_data.get("tactile_append_delta")

    spec: dict[str, dict] = {}
    for batch_key, stream in streams.items():
        if not isinstance(stream, dict):
            continue
        if not _parse_bool(stream.get("enable", True)):
            continue
        role = str(
            stream.get("role")
            or ("fast" if batch_key == TACTILE_REGION_KEY else "slow")
        ).lower()
        if role == "fast" and not fast_on:
            continue
        if role == "slow" and not slow_on:
            continue
        scale = _for_stream(scale_cfg, batch_key)
        if isinstance(scale, str) and scale.strip().lower() in (
            "auto",
            "from_stats",
            "stats",
        ):
            raise ValueError(
                f"tactile_pressure_scale for {batch_key} is still {scale!r}. "
                "Training resolves auto from the tactile-pressure stats cache "
                "and must freeze the number into config.full.yaml."
            )
        center = _for_stream(center_cfg, batch_key)
        center = str(center or "none").lower()
        if center in ("", "off", "false"):
            center = "none"
        entry = {
            "num_hands": int(stream.get("num_hands", 2)),
            # Defaults match TactileEncoder: region window 5, taxel 1.
            "history_len": int(
                stream.get(
                    "history_len",
                    5 if batch_key == TACTILE_REGION_KEY else 1,
                )
            ),
            "pressure_log1p": bool(_for_stream(log1p_cfg, batch_key)),
            "pressure_scale": None if scale is None else float(scale),
            "pressure_center": center,
            "append_delta": bool(_for_stream(delta_cfg, batch_key)),
        }
        if batch_key == TACTILE_REGION_KEY:
            entry["num_units"] = int(stream["num_regions"])
            # Published live frames stay 1 force channel; Δp is appended
            # after the history window is stacked (same as the dataloader).
            entry["num_fields"] = 1
        elif batch_key == TACTILE_TAXEL_KEY:
            entry["num_units"] = int(stream["num_taxels"])
            entry["num_fields"] = 4
        else:
            continue
        if entry["pressure_scale"] is not None and entry["pressure_scale"] <= 0.0:
            raise ValueError(
                f"tactile_pressure_scale for {batch_key} must be > 0, "
                f"got {entry['pressure_scale']}."
            )
        spec[batch_key] = entry
    return spec or None


def resolve_tactile_refresh(
    checkpoint_path: str | Path | None,
) -> dict | None:
    """Infer-only 3D refresh knobs from the checkpoint, or ``None``.

    Returns ``{"mode": str, "every": int}`` when ``action_fusion`` is on.
    ``every`` counts **infer calls**, not control steps — see
    ``tactile_slow_refresh_control_period``.
    """
    for config in _load_checkpoint_configs(checkpoint_path):
        try:
            tactile_cfg = config["framework"]["tactile"]
        except (AttributeError, KeyError, TypeError):
            continue
        if not tactile_cfg or not _parse_bool(tactile_cfg.get("enable", False)):
            continue
        fusion = tactile_cfg.get("action_fusion") or {}
        if not _parse_bool(fusion.get("enable", False)):
            return None
        every = int(fusion.get("slow_refresh_every", 1) or 1)
        if every < 1:
            raise ValueError(
                f"framework.tactile.action_fusion.slow_refresh_every must be >= 1, "
                f"got {every}."
            )
        return {
            "mode": str(fusion.get("refresh_mode", "both")).lower(),
            "every": every,
        }
    return None


def tactile_slow_refresh_control_period(
    infer_period_steps: int, slow_refresh_every: int
) -> int:
    """Control steps between full 3D encodes (infer period × infer-count N)."""
    period = int(infer_period_steps)
    every = int(slow_refresh_every)
    if period < 1 or every < 1:
        raise ValueError(
            "infer_period_steps and slow_refresh_every must be >= 1 "
            f"(got period={period}, every={every})"
        )
    return period * every


def resolve_state_history_frames(
    value: Any, checkpoint_path: str | Path | None
) -> int:
    """``auto`` → checkpoint ``datasets.vla_data.state_history_frames``; else int.

    Missing key (every checkpoint trained before the state-history feature)
    → ``1`` (single-frame state, the legacy behaviour).
    """
    if not _is_auto(value):
        frames = int(value)
    else:
        frames = 1
        for config in _load_checkpoint_configs(checkpoint_path):
            try:
                raw = config["datasets"]["vla_data"]["state_history_frames"]
            except (AttributeError, KeyError, TypeError):
                continue
            if raw is not None:
                frames = int(raw)
                break
    if frames < 1:
        raise ValueError(f"state_history_frames must be >= 1, got {frames}.")
    return frames


def _load_checkpoint_obs_image_size(checkpoint_path: str | Path | None) -> Any:
    """Raw ``obs_image_size`` / ``image_size`` from the checkpoint, or None."""
    for config in _load_checkpoint_configs(checkpoint_path):
        try:
            vla_data = config["datasets"]["vla_data"]
            size = vla_data.get("obs_image_size")
            if size is None:
                size = vla_data.get("image_size")
        except (AttributeError, KeyError, TypeError):
            continue
        if size is not None:
            return size
    return None


def resolve_obs_image_sizes(
    value: Any,
    checkpoint_path: str | Path | None,
    video_keys: Sequence[str],
) -> list[tuple[int, int]]:
    """Resolve per-camera ``(W, H)`` sizes aligned with ``video_keys``.

    ``auto`` replays the checkpoint's ``datasets.vla_data.obs_image_size``
    (or ``image_size`` alias) — uniform or per-camera mapping. An explicit
    override (uniform or mapping) wins. Legacy checkpoints without a saved
    size fall back to (384, 384) for every camera.
    """
    if not video_keys:
        raise ValueError("resolve_obs_image_sizes requires a non-empty video_keys list.")
    if _is_auto(value):
        spec = _load_checkpoint_obs_image_size(checkpoint_path)
    else:
        spec = value
    return expand_obs_image_sizes(spec, video_keys, default=_DEFAULT_IMAGE_SIZE)


def resolve_image_size(value: Any, checkpoint_path: str | Path | None) -> tuple[int, int]:
    """Resolve a single uniform eval-side camera resize from the TRAINING config.

    ``auto`` (the default) replays the checkpoint's saved
    ``datasets.vla_data.obs_image_size`` so the eval client resizes frames to
    exactly the training resolution with the same PIL kernel — no second
    server-side resample. ``image_size`` is accepted as the same alias the
    training dataloader honours; reading only the canonical key would silently
    evaluate an alias-configured checkpoint at the default resolution. An
    explicit ``[W, H]`` override still wins (useful only to deliberately
    diverge from training). Legacy checkpoints without a saved config fall
    back to (384, 384).

    Per-camera mappings are supported only when every listed size is identical
    (then that size is returned). Mixed per-camera sizes must go through
    ``resolve_obs_image_sizes`` with the embodiment's ``video_keys``.
    """
    def _as_uniform(spec: Any) -> tuple[int, int]:
        if isinstance(spec, str):
            spec = yaml.safe_load(spec)
        if _is_hw_size(spec):
            return _parse_image_size(spec)
        if isinstance(spec, Mapping):
            keys = [str(k) for k in spec.keys() if str(k) not in {"default", "_default"}]
            if not keys:
                if "default" in spec or "_default" in spec:
                    return _parse_image_size(spec.get("default", spec.get("_default")))
                raise ValueError(f"Empty per-camera obs_image_size mapping: {dict(spec)!r}.")
            sizes = expand_obs_image_sizes(spec, keys, default=_DEFAULT_IMAGE_SIZE)
            unique = set(sizes)
            if len(unique) != 1:
                raise ValueError(
                    "resolve_image_size cannot collapse a mixed per-camera "
                    f"obs_image_size mapping {dict(spec)!r}; use "
                    "resolve_obs_image_sizes(video_keys=...) instead."
                )
            return next(iter(unique))
        return _parse_image_size(spec)

    if not _is_auto(value):
        return _as_uniform(value)

    size = _load_checkpoint_obs_image_size(checkpoint_path)
    if size is None:
        return _DEFAULT_IMAGE_SIZE
    return _as_uniform(size)
