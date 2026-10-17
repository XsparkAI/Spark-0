# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""Policy server wrapper.

Encapsulates a `baseframework` instance plus a :class:`PolicyNormProcessor`
that reuses the *training-time* :class:`ComposedModalityTransform` for action
un-normalization (no hand-rolled math). The websocket server returns
already-unnormalized actions.

Client-side responsibilities that REMAIN on the client:
  - environment-specific adapters (image_history, gripper sticky, action
    ensembling)
  - chunk-cache scheduling (`step % chunk_size == 0` triggers a new infer)

Exposed API:
  - ``metadata`` (dict, sent at handshake): ``action_chunk_size``,
    ``available_unnorm_keys``, ``action_keys``, ``state_keys``.
  - ``predict_action(examples, unnorm_key=None, **kwargs)`` returns
    ``{"actions": np.ndarray[B, T, action_dim]}``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import yaml

from starVLA.dataloader.gr00t_lerobot.transform.state_action import NORMALIZED_CLAMP_LIMIT
from starVLA.model.framework.base_framework import baseframework, merge_config_overrides
from starVLA.model.framework.share_tools import read_mode_config
from starVLA.model.framework.VLM4A.QwenPI_v3 import validate_rtc_deploy_config

from deployment.model_server.policy_norm_processor import PolicyNormProcessor


def _training_sample_dtype(model_cfg: Dict[str, Any], ckpt_path: str) -> str:
    """dtype the training dataloader packed state/action with (_pack_sample).

    Reads ``datasets.vla_data.sample_dtype``. ``config.yaml`` (the
    accessed-only snapshot loaded by ``read_mode_config``) may omit the key
    because dataloader-worker config accesses are not tracked, so fall back to
    ``config.full.yaml`` (the complete merged config) in the run dir. Absent
    everywhere -> legacy ``float16``.
    """

    def _lookup(cfg: Any) -> Optional[str]:
        try:
            value = cfg["datasets"]["vla_data"]["sample_dtype"]
        except (KeyError, TypeError):
            return None
        return str(value) if value is not None else None

    value = _lookup(model_cfg)
    if value is None:
        full_config_path = Path(ckpt_path).resolve().parents[1] / "config.full.yaml"
        if full_config_path.is_file():
            with full_config_path.open("r", encoding="utf-8") as fp:
                value = _lookup(yaml.safe_load(fp) or {})
    return value or "float16"


def _training_obs_image_size(model_cfg: Dict[str, Any]) -> Optional[List[int]]:
    """Return training image size metadata when it is explicitly configured.

    This is only a consistency check for eval clients. It is not used to infer
    camera order or to rewrite eval observations. Uniform sizes only; a
    per-camera mapping is exported by ``_training_obs_image_sizes`` instead.
    """
    vla_data_cfg = model_cfg.get("datasets", {}).get("vla_data", {})
    size = vla_data_cfg.get("obs_image_size") or vla_data_cfg.get("image_size")
    if isinstance(size, dict):
        # Uniform only when every listed size agrees; a mixed mapping cannot
        # collapse into one [W, H] (that is what the plural variant is for).
        named = [v for k, v in size.items() if str(k) not in {"default", "_default"}]
        default = size.get("default", size.get("_default"))
        candidates = named if named else ([default] if default is not None else [])
        as_int = [
            [int(v[0]), int(v[1])]
            for v in candidates
            if isinstance(v, (list, tuple)) and len(v) == 2
        ]
        if as_int and all(v == as_int[0] for v in as_int):
            return as_int[0]
        return None
    if size is None and "default_image_resolution" in vla_data_cfg:
        default_resolution = vla_data_cfg["default_image_resolution"]
        if isinstance(default_resolution, (list, tuple)) and len(default_resolution) >= 2:
            size = default_resolution[-2:]
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        return None
    return [int(size[0]), int(size[1])]


def _training_video_keys(model_cfg: Dict[str, Any]) -> Optional[List[str]]:
    """The ACTIVE training camera list (order = model input order).

    Resolution order mirrors the training side:
      1. ``embodiment_spec.video_keys`` — train.sh freezes the resolved active
         set (catalog order) into the checkpoint; authoritative.
      2. explicit ``datasets.vla_data.video_keys``;
      3. named keys of a per-camera ``obs_image_size`` mapping (mapping order,
         which for a single-camera recipe is the same one camera).
    Returns ``None`` when the checkpoint does not pin a camera set (legacy).
    """
    spec = model_cfg.get("embodiment_spec")
    if isinstance(spec, dict):
        keys = spec.get("video_keys")
        if isinstance(keys, (list, tuple)) and keys:
            return [str(k) for k in keys]
    vla_data_cfg = model_cfg.get("datasets", {}).get("vla_data", {})
    keys = vla_data_cfg.get("video_keys")
    if isinstance(keys, (list, tuple)) and keys:
        return [str(k) for k in keys]
    size = vla_data_cfg.get("obs_image_size") or vla_data_cfg.get("image_size")
    if isinstance(size, dict):
        named = [str(k) for k in size if str(k) not in {"default", "_default"}]
        if named:
            return named
    return None


def _training_obs_image_sizes(
    model_cfg: Dict[str, Any], video_keys: Optional[Sequence[str]]
) -> Optional[List[List[int]]]:
    """Per-camera training resize targets aligned with ``video_keys`` order.

    ``None`` when either the camera set or the size spec is unknown, or when
    a camera named in ``video_keys`` has no size and the mapping carries no
    ``default`` — an incomplete export is worse than none (clients would
    compare the wrong entries).
    """
    if not video_keys:
        return None
    vla_data_cfg = model_cfg.get("datasets", {}).get("vla_data", {})
    size = vla_data_cfg.get("obs_image_size") or vla_data_cfg.get("image_size")
    if size is None:
        return None

    def _short(key: str) -> str:
        return key.split(".", 1)[-1] if "." in key else key

    if isinstance(size, dict):
        by_short = {
            _short(str(k)): v for k, v in size.items() if str(k) not in {"default", "_default"}
        }
        default = size.get("default", size.get("_default"))
        sizes = []
        for key in video_keys:
            entry = by_short.get(_short(str(key)), default)
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                return None
            sizes.append([int(entry[0]), int(entry[1])])
        return sizes
    if isinstance(size, (list, tuple)) and len(size) == 2:
        return [[int(size[0]), int(size[1])] for _ in video_keys]
    return None


def _assert_trailing_view_mask(
    view_mask: Sequence[Any], training_video_keys: Sequence[str]
) -> None:
    """Reject non-prefix ``view_mask`` patterns (the flexible_views contract).

    Mirrors ``LeRobotSingleDataset._assert_trailing_absent_mask`` (training)
    and ``spark0.runtime_config.assert_trailing_view_mask`` (deploy client);
    duplicated here because this server tree cannot import the client-side
    module. The VLM sees only the present subsequence (no slot-id tokens), so
    a mask like ``[1, 0, 1]`` would make ``image[1]`` a different camera than
    under ``[1, 1, 0]`` — fail loudly instead of silently shifting views.
    """
    seen_absent = False
    for present in view_mask:
        if not present:
            seen_absent = True
        elif seen_absent:
            raise ValueError(
                "predict_action: flexible_views requires absent cameras to be "
                "a trailing suffix of the training slots "
                f"{list(training_video_keys)} (got view_mask="
                f"{list(view_mask)}). Non-prefix masks make packed image "
                "positions ambiguous to the VLM (no slot-id tokens)."
            )


class PolicyServerWrapper:
    """Wraps a `baseframework` for use as a websocket-server policy."""

    # Guidance-target saturation is per-request but its diagnosis is not, so
    # it is logged once per server. Class-level so instances built without
    # ``__init__`` (protocol adapters, tests) still work; ``+=`` shadows it.
    _rtc_saturation_events = 0

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        use_bf16: bool = False,
        unnorm_key: Optional[str] = None,
        config_overrides: Sequence[str] | None = None,
    ) -> None:
        self._ckpt_path = str(ckpt_path)

        # Co-located metadata FIRST (cheap: config + stats JSONs, no model
        # load) so an unservable RTC block fails in seconds instead of after
        # minutes of checkpoint loading. model_cfg is already merged with
        # config_overrides, so both the checkpoint's saved block and
        # eval-time overrides are covered.
        model_cfg, norm_stats = read_mode_config(self._ckpt_path)
        model_cfg = merge_config_overrides(model_cfg, config_overrides)
        self._model_cfg = model_cfg

        # action_chunk_size = future_action_window_size + 1 (matches old client).
        action_model_cfg = model_cfg["framework"]["action_model"]

        if "action_horizon" in action_model_cfg:
            self._action_chunk_size = int(action_model_cfg["action_horizon"])
        elif "future_action_window_size" in action_model_cfg:
            self._action_chunk_size = int(action_model_cfg["future_action_window_size"]) + 1
        else:
            raise ValueError(
                f"PolicyServerWrapper: no action_horizon or future_action_window_size found in model config for {self._ckpt_path}"
            )

        # Real-time chunking (RTC, arXiv:2506.07339) — inference-time async
        # chunk splicing guided by the previous chunk's un-executed tail.
        # Disabled by default; when off, predict_action below is the exact
        # pre-RTC path (same framework call, same response keys). The client
        # gates its async loop on the `rtc` handshake metadata entry.
        self._rtc_cfg = dict(action_model_cfg.get("rtc") or {})
        self._rtc_enabled = bool(self._rtc_cfg.get("enable", False))
        # Fail BEFORE the model load, not mid-episode, on RTC configs this
        # repo cannot serve.
        if self._rtc_enabled:
            validate_rtc_deploy_config(self._rtc_cfg)

        logging.info("PolicyServerWrapper: loading framework from %s", self._ckpt_path)
        framework = baseframework.from_pretrained(self._ckpt_path, config_overrides=config_overrides)
        if use_bf16:
            framework = framework.to(torch.bfloat16)
        framework = framework.to(device).eval()
        self._framework = framework

        if self._rtc_enabled and not hasattr(self._framework, "predict_action_realtime"):
            logging.warning(
                "PolicyServerWrapper: framework.action_model.rtc.enable=true but %s has no "
                "predict_action_realtime — falling back to plain predict_action.",
                type(self._framework).__name__,
            )
            self._rtc_enabled = False
        # Cache of PolicyNormProcessor instances per unnorm_key.
        # For single-dataset ckpts unnorm_key is auto-selected; for multi-dataset
        # ckpts clients must pass unnorm_key per request.
        self._default_unnorm_key = unnorm_key
        self._norm_processors: Dict[str, PolicyNormProcessor] = {}

        # State-quantization replay: legacy runs packed the normalized state
        # as float16 in the dataloader, so eval must quantize the same way
        # before the 256-bin discretization; runs trained with
        # `sample_dtype: float32` must NOT be re-quantized.
        self._replay_state_fp16 = (
            _training_sample_dtype(model_cfg, self._ckpt_path) == "float16"
        )
        logging.info(
            "PolicyServerWrapper: training sample_dtype replay -> %s",
            "float16 round-trip" if self._replay_state_fp16 else "float32 (no re-quantization)",
        )

        # Training camera contract: count (and, for the handshake, order +
        # per-camera sizes). The server cannot reorder views — images arrive
        # anonymous — but it CAN reject a request whose image count disagrees
        # with the checkpoint, which is the most common silent eval bug
        # (head-only checkpoint fed three cameras, or vice versa).
        self._training_video_keys = _training_video_keys(model_cfg)
        self._training_obs_image_sizes = _training_obs_image_sizes(
            model_cfg, self._training_video_keys
        )
        vla_data_cfg = model_cfg.get("datasets", {}).get("vla_data", {}) or {}
        # Match spark0.runtime_config._parse_bool: the string "false" must NOT
        # enable flexible_views (raw bool("false") is True).
        raw_flex = vla_data_cfg.get("flexible_views", False)
        if isinstance(raw_flex, str):
            normalized = raw_flex.strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                self._flexible_views = True
            elif normalized in {"false", "0", "no", "off", ""}:
                self._flexible_views = False
            else:
                raise ValueError(
                    f"datasets.vla_data.flexible_views must be a boolean, got {raw_flex!r}."
                )
        else:
            self._flexible_views = bool(raw_flex)

        # Peek at available keys without building a full processor.
        self._available_unnorm_keys: List[str] = list(norm_stats.keys())

        # Eagerly build when unambiguous; defer for multi-key / no explicit key.
        if unnorm_key is not None or len(self._available_unnorm_keys) == 1:
            default_proc = self._get_processor(unnorm_key)
            self._default_unnorm_key = default_proc.unnorm_key
            logging.info(
                "PolicyServerWrapper ready: action_chunk_size=%d, default_unnorm_key=%s, "
                "available_unnorm_keys=%s, action_keys=%s, state_keys=%s",
                self._action_chunk_size,
                default_proc.unnorm_key,
                default_proc.available_unnorm_keys,
                default_proc.action_keys,
                default_proc.state_keys,
            )
        else:
            logging.info(
                "PolicyServerWrapper ready (multi-key): action_chunk_size=%d, "
                "available_unnorm_keys=%s — clients must pass unnorm_key per request.",
                self._action_chunk_size,
                self._available_unnorm_keys,
            )

    def _get_processor(self, unnorm_key: Optional[str]) -> PolicyNormProcessor:
        cache_key = unnorm_key if unnorm_key is not None else "__default__"
        if cache_key not in self._norm_processors:
            self._norm_processors[cache_key] = PolicyNormProcessor(
                self._ckpt_path, unnorm_key=unnorm_key
            )
        return self._norm_processors[cache_key]

    def get_norm_processor(self, unnorm_key: Optional[str] = None) -> PolicyNormProcessor:
        """Public accessor for the per-unnorm_key normalization processor.

        Protocol adapters (e.g. the GR00T ZMQ compat server) use this to read
        the training-time ``action_keys`` / ``state_keys`` and their per-key
        dims so they can split the flat action chunk into named groups.
        """
        return self._get_processor(
            unnorm_key if unnorm_key is not None else self._default_unnorm_key
        )

    @property
    def metadata(self) -> Dict[str, Any]:
        """Model-invariant metadata; sent to client at websocket handshake."""
        base = {
            "env": "starvla_policy_server",
            "ckpt_path": self._ckpt_path,
            "action_chunk_size": self._action_chunk_size,
            "available_unnorm_keys": self._available_unnorm_keys,
            "default_unnorm_key": self._default_unnorm_key,
            "training_data_mix": self._model_cfg.get("datasets", {}).get("vla_data", {}).get("data_mix"),
            "training_obs_image_size": _training_obs_image_size(self._model_cfg),
            # Full camera contract: the ACTIVE training cameras in model-input
            # order, plus their per-camera resize targets (both None on legacy
            # checkpoints). Requests with a different image COUNT are rejected
            # in predict_action; ORDER remains the client's responsibility.
            "training_video_keys": self._training_video_keys,
            "training_obs_image_sizes": self._training_obs_image_sizes,
            "flexible_views": self._flexible_views,
            "eval_image_contract": (
                "Eval clients must explicitly choose image order. "
                "The server validates image count against training_video_keys "
                "(or present-camera count + view_mask when flexible_views) "
                "but cannot infer or reorder camera views."
            ),
            # RTC handshake: enable flag + the client's async-loop parameters.
            # enable=false -> clients must use the plain synchronous path.
            "rtc": {
                "enable": self._rtc_enabled,
                "execution_horizon": self._rtc_cfg.get("execution_horizon"),
                "inference_delay": self._rtc_cfg.get("inference_delay"),
            },
        }
        # Enrich with per-embodiment keys when a default processor already exists.
        if self._default_unnorm_key is not None:
            proc = self._get_processor(self._default_unnorm_key)
            base["action_keys"] = proc.action_keys
            base["state_keys"] = proc.state_keys
        return base

    def _normalize_example_states(
        self, examples: List[dict], proc: PolicyNormProcessor
    ) -> List[dict]:
        normalized_examples = []
        for example in examples:
            normalized = dict(example)
            if "state" in normalized and normalized["state"] is not None:
                state = proc.apply_state(normalized["state"])
                if self._replay_state_fp16:
                    # Round-trip through float16 to replay the training
                    # dataloader exactly: legacy LeRobotSingleDataset
                    # _pack_sample (sample_dtype float16) casts the normalized
                    # state to fp16 before it reaches the framework (and its
                    # 256-bin discretization), so eval must quantize the same
                    # way or values near bin edges land in different bins.
                    # Runs trained with sample_dtype float32 skip this.
                    state = state.astype(np.float16)
                normalized["state"] = state.astype(np.float32)
            normalized_examples.append(normalized)
        return normalized_examples

    def _normalize_prev_chunk(
        self, prev_action_chunk_model: Any, proc: PolicyNormProcessor
    ) -> tuple[List[np.ndarray], Optional[Dict[str, Any]]]:
        """RTC leftover: un-normalized model space -> the head's input space.

        The client sends the previous chunk's un-executed tail already
        re-encoded against the CURRENT observation's reference pose (absolute
        env poses are the only frame-stable cache — see
        ``ActionCodec.action_raw_to_model``), so only normalization is left.
        It must use the same row indices the head predicts on: per-row modes
        (``hold_centered_perrow``) scale row j by row j's own span, and they
        reject any input whose length differs from the training horizon. The
        buffer is therefore padded to the horizon and sliced back — the
        padding never leaves this method, because a padded row handed to the
        head would widen ``leftover_len`` and pick up a non-zero prefix weight
        instead of falling in the zero-weight tail.

        Accepts a uniform (B, T_prev, D) / (T_prev, D) array or a RAGGED list
        of per-sample (T_prev_i, D) arrays — batched parallel-env fires carry
        one leftover per env and the lengths generally differ.

        Returns ``(normalized, saturation_info)`` with ``normalized`` a
        per-sample list of (T_prev_i, D) arrays (the framework zero-pads the
        ragged batch); the info dict rides back to the client in the
        response's ``rtc`` metadata so the saturation warning is visible on
        the eval-driver side too.
        """
        if isinstance(prev_action_chunk_model, (list, tuple)):
            samples = [np.asarray(c, dtype=np.float32) for c in prev_action_chunk_model]
        else:
            chunk = np.asarray(prev_action_chunk_model, dtype=np.float32)
            if chunk.ndim == 2:
                chunk = chunk[None]
            if chunk.ndim != 3:
                raise ValueError(
                    "prev_action_chunk_model must be (B, T_prev, D), (T_prev, D), or "
                    f"a list of (T_prev_i, D) arrays, got {chunk.shape}"
                )
            samples = [chunk[b] for b in range(chunk.shape[0])]

        horizon = self._action_chunk_size
        normalized: List[np.ndarray] = []
        for rows in samples:
            if rows.ndim != 2 and rows.size > 0:
                raise ValueError(
                    f"each leftover must be (T_prev_i, D), got {rows.shape}"
                )
            leftover_len = min(int(rows.shape[0]) if rows.ndim == 2 else 0, horizon)
            if leftover_len <= 0:
                # One env of a batch can run out of leftover (it re-planned
                # from a drained queue); it rides along unguided rather than
                # failing the whole batched request. A request where EVERY
                # sample is empty has nothing to guide with and is a caller
                # bug — rejected below.
                normalized.append(rows.reshape(0, rows.shape[-1] if rows.ndim == 2 else 0))
                continue
            rows = rows[:leftover_len]
            if leftover_len < horizon:
                rows = np.concatenate(
                    [rows, np.repeat(rows[-1:], horizon - leftover_len, axis=0)], axis=0
                )
            normalized.append(proc.apply_actions(rows)[:leftover_len])
        if all(len(rows) == 0 for rows in normalized):
            raise ValueError("prev_action_chunk_model carries no rows")
        return normalized, self._warn_if_leftover_saturates(normalized)

    def _warn_if_leftover_saturates(
        self, normalized_leftover: List[np.ndarray]
    ) -> Optional[Dict[str, Any]]:
        """Report a guidance target the normalized action space cannot express.

        Every bounded normalization mode saturates at
        ``NORMALIZED_CLAMP_LIMIT``, so a leftover row further from the CURRENT
        reference pose than the training data ever was is silently shortened
        and the frozen prefix is guided to a pose short of the one the robot
        was actually commanded to — the very splice discontinuity RTC exists
        to remove. Clamping is still the right call (the head never saw
        targets outside that range), so this reports instead of failing, but
        it must not be invisible: the usual cause is controller tracking lag,
        which makes ``leftover[j]`` sit ``j + lag`` steps from the measured
        state while row ``j``'s span only covers ``j``.

        Returns an info dict for the response's ``rtc`` metadata, ``None``
        when nothing saturated. Besides the aggregate fields, ``per_sample``
        carries one entry per batch sample (aligned with the request's
        example order) so a batched parallel-env client can map the report
        back to the env that saturated.
        """
        saturated = [np.abs(rows) >= NORMALIZED_CLAMP_LIMIT - 1e-6 for rows in normalized_leftover]
        count = int(sum(int(s.sum()) for s in saturated))
        if count == 0:
            return None
        total = int(sum(int(s.size) for s in saturated))
        rows = sorted({int(r) for s in saturated for r in np.unique(np.nonzero(s)[0])})
        per_sample = []
        for s in saturated:
            sample_count = int(s.sum())
            if sample_count == 0:
                per_sample.append({"leftover_saturated": False})
            else:
                sample_rows = np.unique(np.nonzero(s)[0])
                per_sample.append(
                    {
                        "leftover_saturated": True,
                        "first_saturated_row": int(sample_rows[0]),
                        "saturated_cells": sample_count,
                    }
                )
        self._rtc_saturation_events += 1
        if self._rtc_saturation_events == 1:
            logging.warning(
                "PolicyServerWrapper: %d/%d cells of the RTC guidance leftover saturate "
                "at +-%.1f (rows %s). The frozen prefix will be guided toward a "
                "SHORTENED target, so the splice keeps some of the discontinuity RTC "
                "removes. Usual cause: controller tracking lag — the re-encoded "
                "leftover row j sits j+lag steps from the measured state while row "
                "j's normalization span only covers ~j. NOTE: the client's fire "
                "threshold is NOT a lever here — row j's displacement is j+lag steps "
                "regardless of the leftover length, and per-row spans grow with j, so "
                "shrinking the leftover only drops the least-saturated far rows. "
                "Reduce the lag. Logged once per server.",
                count,
                total,
                NORMALIZED_CLAMP_LIMIT,
                rows,
            )
        return {
            "leftover_saturated": True,
            "first_saturated_row": int(rows[0]),
            "saturated_rows": [int(r) for r in rows[:16]],
            "saturated_cells": count,
            "per_sample": per_sample,
        }

    def predict_action(
        self,
        examples: List[dict],
        unnorm_key: Optional[str] = None,
        prev_action_chunk_model: Optional[Any] = None,
        inference_delay: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Run the framework, then un-normalize via training-time transforms.

        Args:
            examples: list of dicts (each with ``image`` / ``lang`` / optional ``state``).
            unnorm_key: dataset key for un-normalization stats. ``None`` -->
                use the wrapper's default (auto-picked at startup).
            prev_action_chunk_model: RTC only — UN-NORMALIZED model-space
                leftover of the previous chunk (``prev[consumed:]`` at
                inference start), re-encoded by the client against the
                reference pose of THIS request's observation. Either a uniform
                (B, T_prev, D) / (T_prev, D) array or a ragged list of B
                per-sample (T_prev_i, D) arrays (batched parallel-env fires,
                one leftover per env). Normalized here, never sent
                pre-normalized: a chunk normalized for its own inference is
                expressed against that inference's reference pose and row
                indices, and would guide this one toward a stale frame.
            inference_delay: RTC only — rows guaranteed to execute from the
                previous chunk while this inference runs (measured
                client-side). A scalar, or one entry per batch sample.
            **kwargs: forwarded to the framework's ``predict_action``.
                Note: QwenPI_v3 ignores sampler kwargs — the flow-matching
                step count comes from the checkpoint's
                ``framework.action_model.num_inference_timesteps``.

        Returns:
            ``{"actions": np.ndarray[B, T, D]}`` -- un-normalized MODEL-SPACE
            actions in every mode, i.e. still in the checkpoint's converted
            representation (e.g. chunk-wise ``delta_xyz_rot6d`` against THIS
            request's reference pose, not env-absolute targets). The client
            decodes them to world-frame absolute actions itself
            (``Model._decode_chunk``) and caches THAT; the next RTC leftover
            is derived from the decoded cache. On a guided (RTC) call the
            dict also carries an additive ``"rtc"`` metadata entry (e.g.
            ``leftover_saturated``) — plain calls never set it, and older
            clients ignore it.
        """
        effective_key = unnorm_key if unnorm_key is not None else self._default_unnorm_key
        if effective_key is None:
            if len(self._available_unnorm_keys) == 1:
                effective_key = self._available_unnorm_keys[0]
            else:
                raise ValueError(
                    f"predict_action: unnorm_key not specified and no default set. "
                    f"Pass one of {self._available_unnorm_keys}."
                )
        proc = self._get_processor(effective_key)

        # Camera-count contract: images arrive anonymous, so ORDER cannot be
        # checked server-side, but a wrong COUNT always means the client is
        # feeding a different camera set than the checkpoint was trained on
        # (e.g. wrists added to a head-only run) — fail loudly instead of
        # silently misaligning every view. With flexible_views, a view_mask
        # of length = training slots may accompany a shorter present-only
        # image list (same packing as the training dataloader).
        if self._training_video_keys is not None:
            expected_cameras = len(self._training_video_keys)
            for i, example in enumerate(examples):
                images = example.get("image") if isinstance(example, dict) else None
                if images is None:
                    raise ValueError(
                        f"predict_action: example {i} has no 'image' entry; expected "
                        f"{expected_cameras} camera image(s) {self._training_video_keys}."
                    )
                view_mask = example.get("view_mask") if isinstance(example, dict) else None
                if self._flexible_views and view_mask is not None:
                    if len(view_mask) != expected_cameras:
                        raise ValueError(
                            f"predict_action: example {i} view_mask length "
                            f"{len(view_mask)} != training camera slots "
                            f"{expected_cameras} ({self._training_video_keys})."
                        )
                    present = int(sum(1 for m in view_mask if m))
                    if len(images) != present:
                        raise ValueError(
                            f"predict_action: example {i} carries {len(images)} image(s) "
                            f"but view_mask selects {present} present camera(s) "
                            f"(mask={list(view_mask)}, slots={self._training_video_keys})."
                        )
                    # Count alone cannot catch a dropped MIDDLE camera (e.g.
                    # [1,0,1] packs [head, right] but reads as [head, left]).
                    _assert_trailing_view_mask(view_mask, self._training_video_keys)
                elif len(images) != expected_cameras:
                    raise ValueError(
                        f"predict_action: example {i} carries {len(images)} camera image(s) "
                        f"but the checkpoint was trained on {expected_cameras} "
                        f"({self._training_video_keys}, order = model input order). "
                        "Select/resize cameras client-side to match the training set."
                    )

        model_examples = self._normalize_example_states(examples, proc)

        use_rtc = (
            self._rtc_enabled
            and prev_action_chunk_model is not None
            and inference_delay is not None
        )
        if use_rtc:
            # Per-sample delays (a scalar broadcast by the client is expanded
            # here so the framework's per-sample validation always sees one
            # entry per example).
            if isinstance(inference_delay, (list, tuple, np.ndarray)):
                delays = [int(d) for d in inference_delay]
            else:
                delays = [int(inference_delay)] * len(examples)
            use_rtc = any(d > 0 for d in delays)
        rtc_meta: Optional[Dict[str, Any]] = None
        if use_rtc:
            prev_normalized, saturation = self._normalize_prev_chunk(
                prev_action_chunk_model, proc
            )
            out = self._framework.predict_action_realtime(
                examples=model_examples,
                prev_action_chunk_normalized=prev_normalized,
                inference_delay=delays,
                **kwargs,
            )
            rtc_meta = saturation or {"leftover_saturated": False}
        else:
            out = self._framework.predict_action(examples=model_examples, **kwargs)
        normalized = np.asarray(out["normalized_actions"])  # (B, T, D)

        unnorm = np.stack(
            [proc.unapply_actions(normalized[b]) for b in range(normalized.shape[0])],
            axis=0,
        )
        result: Dict[str, Any] = {"actions": unnorm}
        if rtc_meta is not None:
            result["rtc"] = rtc_meta
        return result
