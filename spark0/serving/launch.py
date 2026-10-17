"""Resolve a Spark-0 serving launch from a checkpoint and the live configs.

Everything here is pure Python (no torch), so it can be unit-tested and reused
by other launchers. ``spark0.serving.server`` calls :func:`resolve_launch` and
then builds the inference backend and the policy from the result.

Train/test consistency is the point of this module:

* the TRAINING config saved next to the checkpoint is replayed into the
  ``SPARK_0_XPOLICY_*`` environment the StarVLA data registry reads, so the
  server rebuilds the training-time normalization and embodiment schema;
* deploy-time knobs (``rtc``, ``execute_horizon``) come from the LIVE model
  config under ``configs/``, never from the checkpoint, so editing them takes
  effect without retraining;
* the only checkpoint-config keys an operator may diverge on are the
  inference-only allowlist in :func:`validate_backend_override`.
"""

from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_SERVE_CONFIG = CONFIG_DIR / "serve.yaml"
DEFAULT_CKPT_ROOT = REPO_ROOT / "checkpoints"
DEFAULT_STARVLA_ROOT = REPO_ROOT / "third_party" / "starVLA"
GENERATED_DIR = REPO_ROOT / ".generated"

_NULLS = (None, "", "null", "None", "auto")
_WEIGHT_SUFFIXES = (".pt", ".safetensors")

RTC_SERVER_KEYS = (
    "enable",
    "inference_delay",
    "execution_horizon",
    "prefix_attention_schedule",
    "max_guidance_weight",
    "mode",
)
RTC_CLIENT_KEYS = ("fire_at_remaining", "delay_window", "debug", "max_failures")

# Training contracts the operator may pin by env var only when the checkpoint
# records nothing (legacy runs); a recorded value must match exactly.
_POSE_CONTRACT_ENV = (
    ("SPARK_0_XPOLICY_STATE_POSE_FORMAT", "state_pose_format"),
    ("SPARK_0_XPOLICY_ACTION_POSE_FORMAT", "action_pose_format"),
    ("SPARK_0_XPOLICY_POSE_NORMALIZATION", "pose_normalization"),
)


@dataclass
class LaunchPlan:
    checkpoint_path: Path
    starvla_root: Path
    backend_env: dict[str, str]
    backend_overrides: list[str]
    policy_cfg: dict[str, Any]
    model_config_path: Path | None
    notes: list[str] = field(default_factory=list)


def _stem(text: Any) -> str:
    text = str(text)
    return text[:-5] if text.endswith(".yaml") else text


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def parse_value(text: str) -> Any:
    """``key=value`` value parsing: Python literals when they parse, else str."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def _natural_key(text: str) -> list:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def _newest_weights(run_dir: Path) -> Path | None:
    candidates = [
        path
        for sub in ("final_model", "checkpoints", ".")
        for path in (run_dir / sub).glob("*")
        if path.is_file() and path.suffix in _WEIGHT_SUFFIXES
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: _natural_key(str(path)))


def resolve_checkpoint(ckpt: str, ckpt_root: Path = DEFAULT_CKPT_ROOT) -> Path:
    """A weight file from a file path, a run directory, or a run name.

    A run directory resolves to its newest ``final_model/`` or
    ``checkpoints/`` weight file (natural sort, so ``steps_80000`` beats
    ``steps_9000``). A bare name is looked up under ``ckpt_root`` first.
    """
    raw = Path(ckpt).expanduser()
    candidates = [raw] if raw.is_absolute() else [ckpt_root / raw, Path.cwd() / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
        if candidate.is_dir():
            found = _newest_weights(candidate)
            if found is not None:
                return found.resolve()
    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"No .pt / .safetensors checkpoint found for {ckpt!r} (searched {searched}; "
        "a run directory is searched under final_model/ and checkpoints/)."
    )


def _run_dir(checkpoint_path: Path) -> Path:
    """Run directory of a weight file (same rule as ``runtime_config``).

    ``<run>/final_model/x.pt`` and ``<run>/checkpoints/x.pt`` resolve to
    ``<run>``; a weight stored directly in the run directory resolves to its
    parent, so ``config.full.yaml`` is found in every layout
    :func:`_newest_weights` accepts.
    """
    if checkpoint_path.parent.name in ("checkpoints", "final_model"):
        return checkpoint_path.parents[1]
    return checkpoint_path.parent


def load_training_configs(checkpoint_path: Path) -> list[dict]:
    """``config.full.yaml`` (authoritative) then ``config.yaml`` of the run."""
    configs = []
    for name in ("config.full.yaml", "config.yaml"):
        path = _run_dir(checkpoint_path) / name
        if path.is_file():
            loaded = _load_yaml(path)
            if isinstance(loaded, dict):
                configs.append(loaded)
    return configs


def _first(configs: list[dict], *path_keys: str) -> Any:
    for cfg in configs:
        value: Any = cfg
        for key in path_keys:
            value = value.get(key) if isinstance(value, dict) else None
            if value is None:
                break
        if value not in (None, "", [], {}):
            return value
    return None


def checkpoint_registry(checkpoint_path: Path) -> dict[str, str]:
    """Data-registry inputs replayed from the checkpoint's training config.

    Empty values when the run saved no config (legacy checkpoints); the
    server then relies on statically registered mixtures.
    """
    keys = (
        "data_mix",
        "robot_type",
        "embodiment_dir",
        "mixture_json",
        "state_pose_format",
        "action_pose_format",
        "pose_normalization",
        "spec_overrides",
    )
    configs = load_training_configs(checkpoint_path)
    if not configs:
        return {key: "" for key in keys}

    data_mix = _first(configs, "datasets", "vla_data", "data_mix") or "xpolicylab_runtime"
    schema = _first(configs, "datasets", "vla_data", "xpolicylab_schema") or {}
    schema_robot = schema.get("robot_type") if isinstance(schema, dict) else None
    # robodojo_arx_x5_h50_q99 is the built-in spec checkpoints saved before
    # `embodiment` was recorded were trained against.
    embodiment = schema_robot or _first(configs, "embodiment") or "robodojo_arx_x5_h50_q99"

    embodiment_dir = CONFIG_DIR / "embodiments"
    spec = _first(configs, "embodiment_spec")
    if isinstance(spec, dict) and spec:
        # Materialize the frozen schema so eval uses the exact training-time
        # spec even if configs/embodiments/*.yaml changed since training.
        embodiment_dir = GENERATED_DIR / "eval_embodiments" / _run_dir(checkpoint_path).name
        embodiment_dir.mkdir(parents=True, exist_ok=True)
        with (embodiment_dir / f"{embodiment}.yaml").open("w", encoding="utf-8") as fp:
            yaml.safe_dump(spec, fp, sort_keys=False)

    mixture = _first(configs, "data_mixture") or []
    pose_repr = (
        spec.get("pose_representation")
        if isinstance(spec, dict) and isinstance(spec.get("pose_representation"), dict)
        else {}
    )
    state_pose_format = (
        _first(configs, "datasets", "vla_data", "state_pose_format") or pose_repr.get("state") or ""
    )
    action_pose_format = (
        _first(configs, "datasets", "vla_data", "action_pose_format") or pose_repr.get("action") or ""
    )
    # Legacy pose checkpoints have no key because q99 was the only path; make
    # that explicit so an ambient env var cannot reinterpret an old run.
    pose_normalization = (
        _first(configs, "datasets", "vla_data", "pose_normalization")
        or pose_repr.get("normalization")
        or ("q99" if action_pose_format else "")
    )

    spec_overrides: dict[str, Any] = {}
    for name, path_keys in (
        ("action_horizon", ("framework", "action_model", "action_horizon")),
        ("state_normalization", ("datasets", "vla_data", "state_normalization")),
        ("action_normalization", ("datasets", "vla_data", "action_normalization")),
        ("observation_indices", ("datasets", "vla_data", "observation_indices")),
    ):
        value = _first(configs, *path_keys)
        if value is not None:
            spec_overrides[name] = value
    video_keys = spec.get("video_keys") if isinstance(spec, dict) else None
    if not video_keys:
        video_keys = _first(configs, "datasets", "vla_data", "video_keys")
    if video_keys is not None:
        spec_overrides["video_keys"] = list(video_keys)

    return {
        "data_mix": str(data_mix),
        "robot_type": str(embodiment),
        "embodiment_dir": str(embodiment_dir),
        "mixture_json": json.dumps(mixture, separators=(",", ":")) if mixture else "",
        "state_pose_format": str(state_pose_format),
        "action_pose_format": str(action_pose_format),
        "pose_normalization": str(pose_normalization),
        "spec_overrides": json.dumps(spec_overrides, separators=(",", ":")) if spec_overrides else "",
    }


def backend_env(checkpoint_path: Path, environ: Mapping[str, str]) -> dict[str, str]:
    """``SPARK_0_*`` variables for the StarVLA backend; pre-set values win."""
    ckpt = checkpoint_registry(checkpoint_path)
    for env_name, label in _POSE_CONTRACT_ENV:
        env_value = environ.get(env_name, "")
        if env_value and ckpt[label] and env_value != ckpt[label]:
            raise ValueError(
                f"{env_name}={env_value} conflicts with checkpoint {label}={ckpt[label]}; "
                "pose representation and normalization must match training."
            )

    def pick(env_name: str, ckpt_value: str) -> str:
        return environ.get(env_name) or ckpt_value

    data_mix = pick("SPARK_0_XPOLICY_DATA_MIX", ckpt["data_mix"])
    # Any non-empty dataset name registers the runtime mix entry; only its
    # robot_type is consumed for eval-time normalization.
    dataset_name = environ.get("SPARK_0_XPOLICY_DATASET_NAME") or ("xpolicy_eval" if data_mix else "")
    return {
        "SPARK_0_XPOLICY_DATASET_NAME": dataset_name,
        "SPARK_0_XPOLICY_DATA_MIX": data_mix,
        "SPARK_0_XPOLICY_ROBOT_TYPE": pick("SPARK_0_XPOLICY_ROBOT_TYPE", ckpt["robot_type"]),
        "SPARK_0_EMBODIMENT_DIR": pick("SPARK_0_EMBODIMENT_DIR", ckpt["embodiment_dir"]),
        "SPARK_0_XPOLICY_MIXTURE_JSON": pick("SPARK_0_XPOLICY_MIXTURE_JSON", ckpt["mixture_json"]),
        "SPARK_0_XPOLICY_STATE_POSE_FORMAT": pick(
            "SPARK_0_XPOLICY_STATE_POSE_FORMAT", ckpt["state_pose_format"]
        ),
        "SPARK_0_XPOLICY_ACTION_POSE_FORMAT": pick(
            "SPARK_0_XPOLICY_ACTION_POSE_FORMAT", ckpt["action_pose_format"]
        ),
        "SPARK_0_XPOLICY_POSE_NORMALIZATION": pick(
            "SPARK_0_XPOLICY_POSE_NORMALIZATION", ckpt["pose_normalization"]
        ),
        "SPARK_0_XPOLICY_SPEC_OVERRIDES": pick(
            "SPARK_0_XPOLICY_SPEC_OVERRIDES", ckpt["spec_overrides"]
        ),
    }


def validate_backend_override(key: str, value: str, source: str) -> None:
    """Only inference-time keys may diverge from the checkpoint config."""

    def fail(message: str) -> None:
        raise ValueError(f"{key}: {message} (got {value!r}, from {source})")

    if key in (
        "framework.action_model.num_inference_timesteps",
        "framework.action_model.rtc.inference_delay",
        "framework.action_model.rtc.execution_horizon",
    ):
        if not re.fullmatch(r"[1-9][0-9]*", value):
            fail("must be a positive integer")
    elif key == "framework.action_model.rtc.enable":
        if value not in ("true", "false"):
            fail("must be true|false")
    elif key == "framework.action_model.rtc.prefix_attention_schedule":
        if value not in ("exp", "linear", "ones", "zeros"):
            fail("must be exp|linear|ones|zeros")
    elif key == "framework.action_model.rtc.max_guidance_weight":
        # 0 silently disables PiGDM guidance and a negative value inverts it.
        if not re.fullmatch(r"[1-9][0-9]*(\.[0-9]+)?|0\.[0-9]*[1-9][0-9]*", value):
            fail("must be a positive number")
    elif key == "framework.action_model.rtc.mode":
        # simulated_delay needs a checkpoint trained with that loss.
        if value != "pigdm":
            fail("must be pigdm")
    else:
        raise ValueError(
            f"Backend overrides may only set inference-only keys "
            "(framework.action_model.num_inference_timesteps, framework.action_model.rtc.*); "
            f"got {key} from {source}. Training-contract keys (action_horizon, dimensions, "
            "normalization, architecture) are replayed from the checkpoint."
        )


class _OverrideSet:
    """Dotlist overrides; the first source to set a key wins."""

    def __init__(self) -> None:
        self.items: list[str] = []
        self.applied: list[str] = []
        self._sources: dict[str, str] = {}

    def add(self, item: str, source: str) -> str | None:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got {item!r} ({source}).")
        key, value = item.split("=", 1)
        previous = self._sources.get(key)
        if previous is not None:
            if previous == source:
                raise ValueError(f"{source} sets {key} twice; remove the duplicate.")
            return f"{source} also sets {key}; keeping the value from {previous}."
        validate_backend_override(key, value, source)
        self._sources[key] = source
        self.items.append(item)
        self.applied.append(f"{item} ({source})")
        return None


def resolve_model_config(
    model_config: str | None, ckpt_hint: str | None, checkpoint_path: Path
) -> tuple[Path | None, str, list[str]]:
    """(path, why, warnings) of the live model config whose deploy knobs apply.

    Order: explicit ``model_config`` (stem or path), ``configs/<ckpt_hint>.yaml``,
    then the stem the checkpoint recorded at training time.
    """
    warnings = []
    if model_config:
        candidate = Path(model_config).expanduser()
        if candidate.is_file():
            return candidate.resolve(), "--model-config", warnings
        candidate = CONFIG_DIR / f"{_stem(model_config)}.yaml"
        if candidate.is_file():
            return candidate, "--model-config", warnings
        raise FileNotFoundError(f"Model config {model_config!r} not found (also tried {candidate}).")
    if ckpt_hint:
        candidate = CONFIG_DIR / f"{_stem(ckpt_hint)}.yaml"
        if candidate.is_file():
            return candidate, "the checkpoint name", warnings
    recorded = _first(load_training_configs(checkpoint_path), "model_config")
    if recorded not in _NULLS:
        candidate = CONFIG_DIR / f"{_stem(recorded)}.yaml"
        if candidate.is_file():
            return candidate, "the checkpoint's recorded model_config", warnings
        warnings.append(
            f"The checkpoint was trained from model_config={recorded}, which no longer "
            f"exists under {CONFIG_DIR}; falling back to its frozen rtc pointer."
        )
    return None, "", warnings


def _deploy_knobs(
    model_cfg_path: Path | None, checkpoint_path: Path
) -> tuple[list[str], dict[str, str], str]:
    """(backend dotlist, policy overrides, log line) from the live rtc pointer."""
    backend: list[str] = []
    policy: dict[str, str] = {}
    model_cfg = _load_yaml(model_cfg_path) if model_cfg_path is not None else {}
    if model_cfg_path is not None:
        rtc_ref = model_cfg.get("rtc")
    else:
        rtc_ref = _first(load_training_configs(checkpoint_path), "rtc")

    if model_cfg_path is not None:
        value = model_cfg.get("execute_horizon")
        if value not in _NULLS:
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(
                    f"{model_cfg_path.name} execute_horizon must be a positive integer "
                    f"or null, got {value!r}."
                )
            policy["execute_horizon"] = str(value)

    if rtc_ref in _NULLS:
        # A found model config is an explicit decision: pin the synchronous
        # path. No model config at all (legacy checkpoint) keeps auto.
        if model_cfg_path is not None:
            policy["rtc"] = "false"
            policy["rtc_enable"] = "false"
            return backend, policy, "rtc=off (model config rtc: null)"
        return backend, policy, "rtc=auto (no model config)"

    rtc_path = Path(str(rtc_ref)).expanduser()
    if not rtc_path.is_file():
        rtc_path = CONFIG_DIR / "rtc" / f"{_stem(rtc_ref)}.yaml"
    if not rtc_path.is_file():
        raise FileNotFoundError(f"rtc: {rtc_ref} points at a missing RTC config ({rtc_path}).")
    rtc_cfg = _load_yaml(rtc_path)
    unknown = sorted(set(rtc_cfg) - set(RTC_SERVER_KEYS) - set(RTC_CLIENT_KEYS))
    if unknown:
        raise ValueError(
            f"{rtc_path} has unknown keys {unknown}; server keys: {list(RTC_SERVER_KEYS)}, "
            f"client keys: {list(RTC_CLIENT_KEYS)}."
        )
    enable = rtc_cfg.get("enable", True)
    if isinstance(enable, str):
        enable = enable.strip().lower() in ("true", "1", "yes", "on")
    enable_str = "true" if enable else "false"
    backend.append(f"framework.action_model.rtc.enable={enable_str}")
    for key in RTC_SERVER_KEYS:
        value = rtc_cfg.get(key)
        if key != "enable" and value not in _NULLS:
            backend.append(f"framework.action_model.rtc.{key}={value}")
    policy["rtc"] = enable_str
    policy["rtc_enable"] = enable_str
    for key in RTC_CLIENT_KEYS:
        value = rtc_cfg.get(key)
        if value in _NULLS:
            continue
        if isinstance(value, bool):
            value = "true" if value else "false"
        policy[f"rtc_{key}"] = str(value)
    return backend, policy, f"rtc={rtc_path} (training-free, from the model config)"


def resolve_launch(
    *,
    ckpt: str,
    serve_config: Path = DEFAULT_SERVE_CONFIG,
    model_config: str | None = None,
    ckpt_root: Path = DEFAULT_CKPT_ROOT,
    backend_overrides: list[str] | None = None,
    policy_overrides: list[str] | None = None,
    execute_horizon: int | None = None,
    environ: Mapping[str, str] | None = None,
) -> LaunchPlan:
    environ = os.environ if environ is None else environ
    checkpoint_path = resolve_checkpoint(ckpt, ckpt_root)
    notes = [f"checkpoint: {checkpoint_path}"]

    env = backend_env(checkpoint_path, environ)

    overrides = _OverrideSet()
    for item in backend_overrides or []:
        overrides.add(item, "--backend-override")
    for item in (environ.get("SPARK_0_CONFIG_OVERRIDE") or "").split():
        overrides.add(item, "SPARK_0_CONFIG_OVERRIDE")

    ckpt_hint = ckpt if os.sep not in ckpt and not ckpt.endswith(_WEIGHT_SUFFIXES) else None
    cfg_path, cfg_why, warnings = resolve_model_config(model_config, ckpt_hint, checkpoint_path)
    notes.extend(f"WARN {w}" for w in warnings)
    notes.append(
        f"model config: {cfg_path} (via {cfg_why})" if cfg_path else "model config: none"
    )
    rtc_backend, knob_overrides, rtc_note = _deploy_knobs(cfg_path, checkpoint_path)
    notes.append(rtc_note)
    for item in rtc_backend:
        warning = overrides.add(item, f"rtc config of {cfg_path.name if cfg_path else 'checkpoint'}")
        if warning:
            notes.append(f"WARN {warning}")

    env_execute = environ.get("SPARK_0_EXECUTE_HORIZON")
    if execute_horizon is None and env_execute:
        if not re.fullmatch(r"[1-9][0-9]*", env_execute):
            raise ValueError(f"SPARK_0_EXECUTE_HORIZON must be a positive integer, got {env_execute!r}.")
        execute_horizon = int(env_execute)
    if execute_horizon is not None:
        if execute_horizon < 1:
            raise ValueError(f"execute_horizon must be positive, got {execute_horizon}.")
        knob_overrides["execute_horizon"] = str(execute_horizon)

    serve_cfg = _load_yaml(Path(serve_config)) if serve_config else {}
    policy_cfg: dict[str, Any] = dict(serve_cfg)
    for key, env_name in (
        ("include_state", "SPARK_0_INCLUDE_STATE"),
        ("unnorm_key", "SPARK_0_UNNORM_KEY"),
        ("image_size", "SPARK_0_IMAGE_SIZE"),
    ):
        if environ.get(env_name):
            policy_cfg[key] = parse_value(environ[env_name])
    for key, value in knob_overrides.items():
        policy_cfg[key] = parse_value(value)
    for item in policy_overrides or []:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}.")
        key, value = item.split("=", 1)
        policy_cfg[key] = parse_value(value)
    policy_cfg["checkpoint_path"] = str(checkpoint_path)
    notes.append(
        "execute_horizon: "
        + (str(policy_cfg["execute_horizon"]) if policy_cfg.get("execute_horizon") not in _NULLS
           else "auto (training action_horizon)")
    )
    if overrides.applied:
        notes.append("WARN diverging from the checkpoint config: " + ", ".join(overrides.applied))

    starvla_root = Path(environ.get("SPARK_0_STARVLA_ROOT") or DEFAULT_STARVLA_ROOT).resolve()
    return LaunchPlan(
        checkpoint_path=checkpoint_path,
        starvla_root=starvla_root,
        backend_env=env,
        backend_overrides=overrides.items,
        policy_cfg=policy_cfg,
        model_config_path=cfg_path,
        notes=notes,
    )
