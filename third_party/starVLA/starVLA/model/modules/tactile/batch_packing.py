# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Local Spark-0 addition (NOT upstream starVLA).
"""
Turn a list of ``examples`` dicts into ``TactileEncoder.encode`` kwargs.

The dataloader packs fusion columns (`region_pressure` / `taxel_pressure_3d`)
and the optional head-camera heatmap (`tactile_2d`) when `meta/modality.json`
declares the matching tactile streams. A key that is missing on a row means
"this row has no tactile", which the encoder answers with its learned
``no_tactile`` token. That makes a fusion-enabled run on non-tactile data
behave like the V-only run plus a constant placeholder, and it makes MIXED
batches (some episodes gloved, some not) legal without a second collate path.

Key names are resolved from ``framework.tactile.streams.<name>.batch_key``
(falling back to ``data_key`` and its trailing component, then to built-in
aliases), so a recipe can point at its own column without a code change.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Aliases are deliberately tactile-specific. Bare names like `present`,
# `points` or `valid_mask` are NOT accepted: they are plausible column names
# for view masks / DA3 geometry, and silently reading one as tactile would be
# far worse than the miss it saves. Point at anything else via `batch_key`.
DEFAULT_FAST_KEYS: Tuple[str, ...] = ("region_pressure", "tactile_region_pressure")
DEFAULT_SLOW_KEYS: Tuple[str, ...] = (
    "taxel_pressure_3d",
    "tactile_taxel_pressure_3d",
)
DEFAULT_SLOW_VALID_KEYS: Tuple[str, ...] = ("taxel_valid_mask",)
DEFAULT_PRESENT_FAST_KEYS: Tuple[str, ...] = ("present_fast", "present_region_pressure")
DEFAULT_PRESENT_SLOW_KEYS: Tuple[str, ...] = ("present_slow", "present_taxel")
DEFAULT_PRESENT_KEYS: Tuple[str, ...] = ("present_tactile",)
DEFAULT_REGION_FUTURE_KEYS: Tuple[str, ...] = ("region_pressure_future",)
DEFAULT_REGION_FUTURE_VALID_KEYS: Tuple[str, ...] = (
    "region_pressure_future_valid",
)


def resolve_tactile_batch_keys(tac_cfg: Any) -> Dict[str, List[str]]:
    """Candidate ``examples[i]`` keys per stream, most specific first."""
    from starVLA.model.modules.tactile.tactile_encoder import (
        _as_role_flag,
        stream_is_active,
    )

    streams = tac_cfg.get("streams", None) if tac_cfg else None
    fast_on = _as_role_flag(tac_cfg.get("fast_enable", True) if tac_cfg else True)
    slow_on = _as_role_flag(tac_cfg.get("slow_enable", True) if tac_cfg else True)

    def _candidates(stream_name: str, defaults: Sequence[str]) -> List[str]:
        keys: List[str] = []
        scfg = streams.get(stream_name, None) if streams else None
        if scfg:
            for cfg_key in ("batch_key", "data_key"):
                value = scfg.get(cfg_key, None)
                if not value:
                    continue
                keys.append(str(value))
                # `tactile/taxel_pressure_3D` also matches a flattened
                # `taxel_pressure_3D` column.
                tail = str(value).rsplit("/", 1)[-1]
                if tail != str(value):
                    keys.append(tail)
        keys.extend(defaults)
        return list(dict.fromkeys(keys))  # de-dup, keep order

    def _wanted(stream_name: str) -> bool:
        # No streams block → both columns (legacy). A listed stream is on
        # unless enable: false or its role flag is off. An unlisted name
        # is off — do not invent a region column for a taxel-only stem.
        if streams is None:
            return fast_on if stream_name == "region_pressure" else slow_on
        if stream_name not in streams:
            return False
        return stream_is_active(
            stream_name,
            streams.get(stream_name),
            fast_enable=fast_on,
            slow_enable=slow_on,
            default_enable=True,
        )

    # Encode kwargs are stream-named (region_pressure / taxel_pressure_3d).
    # The "fast"/"slow" key groups here are those kwargs, not a role swap.
    want_region = _wanted("region_pressure")
    want_taxel = _wanted("taxel_pressure_3d")
    return {
        "fast": _candidates("region_pressure", DEFAULT_FAST_KEYS) if want_region else [],
        "slow": _candidates("taxel_pressure_3d", DEFAULT_SLOW_KEYS) if want_taxel else [],
        "slow_valid": list(DEFAULT_SLOW_VALID_KEYS) if want_taxel else [],
        "present_fast": list(DEFAULT_PRESENT_FAST_KEYS) if want_region else [],
        "present_slow": list(DEFAULT_PRESENT_SLOW_KEYS) if want_taxel else [],
        "present": list(DEFAULT_PRESENT_KEYS),
    }


def first_key(example: dict, keys: Sequence[str]):
    for key in keys:
        value = example.get(key, None) if isinstance(example, dict) else None
        if value is not None:
            return value
    return None


def stack_optional_stream(
    examples: List[dict],
    keys: Sequence[str],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Stack one stream to ``[B, ...]`` plus a per-row present flag.

    Returns ``(None, None)`` when no row carries the key. Rows that lack it in
    a partially-packed batch are zero-filled and marked absent. ``dtype`` must
    be the encoder's parameter dtype: ``TactileEncoder.encode`` adopts the
    dtype of whatever tensor it is handed, so an fp32 batch against bf16
    weights (DeepSpeed bf16) would hit a dtype mismatch inside the trunk.
    """
    raw = [first_key(ex, keys) for ex in examples]
    if all(value is None for value in raw):
        return None, None
    tensors: List[Optional[torch.Tensor]] = []
    for value in raw:
        if value is None:
            tensors.append(None)
            continue
        tensor = (
            value.detach()
            if isinstance(value, torch.Tensor)
            else torch.as_tensor(np.asarray(value))
        )
        tensors.append(tensor.to(device=device, dtype=dtype))
    ref = next(t for t in tensors if t is not None)
    packed = torch.zeros((len(examples), *ref.shape), device=device, dtype=dtype)
    present = torch.zeros(len(examples), dtype=torch.bool, device=device)
    for row, tensor in enumerate(tensors):
        if tensor is None:
            continue
        if tensor.shape != ref.shape:
            raise ValueError(
                f"tactile stream rows must share a shape; row {row} has "
                f"{tuple(tensor.shape)} vs {tuple(ref.shape)}"
            )
        packed[row] = tensor
        present[row] = True
    return packed, present


def stack_present_flag(
    examples: List[dict], keys: Sequence[str], device: torch.device
) -> Optional[torch.Tensor]:
    raw = [first_key(ex, keys) for ex in examples]
    if all(value is None for value in raw):
        return None
    return torch.tensor(
        [bool(value) if value is not None else False for value in raw],
        dtype=torch.bool,
        device=device,
    )


def pack_tactile_encode_kwargs(
    examples: List[dict],
    batch_keys: Dict[str, List[str]],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[dict, bool]:
    """Return ``(encode_kwargs, found_any)``.

    ``found_any`` is False when the batch carries no tactile column at all —
    the caller warns once and the encoder emits ``no_tactile`` for every row.
    ``dtype`` should be the encoder's parameter dtype (see
    ``stack_optional_stream``).
    """
    region, region_present = stack_optional_stream(
        examples, batch_keys["fast"], device, dtype
    )
    taxel, taxel_present = stack_optional_stream(
        examples, batch_keys["slow"], device, dtype
    )
    present_fast = stack_present_flag(examples, batch_keys["present_fast"], device)
    present_slow = stack_present_flag(examples, batch_keys["present_slow"], device)
    present = stack_present_flag(examples, batch_keys["present"], device)

    if present_fast is None:
        present_fast = region_present
    if present_slow is None:
        present_slow = taxel_present

    found_any = not (
        present_fast is None and present_slow is None and present is None
    )
    if not found_any:
        present = torch.zeros(len(examples), dtype=torch.bool, device=device)

    kwargs: dict = {
        "batch_size": len(examples),
        "device": device,
        "dtype": dtype,
    }
    if region is not None:
        kwargs["region_pressure"] = region
    if taxel is not None:
        kwargs["taxel_pressure_3d"] = taxel
        taxel_valid, valid_present = stack_optional_stream(
            examples, batch_keys["slow_valid"], device, dtype
        )
        if taxel_valid is not None:
            valid_mask = taxel_valid > 0.5
            # Rows that carry no validity column default to ALL VALID. The
            # zero-fill from stack_optional_stream would otherwise read as
            # "every taxel invalid" on exactly the rows that never opted into
            # validity — e.g. a nested-format tactile dataset mixed with an
            # EgoPro converter dataset that does pack `taxel_valid_mask`.
            # Those rows still have present_slow=True, so the encoder would
            # train a fully-masked point cloud as if it were real contact.
            if valid_present is not None and not bool(valid_present.all()):
                missing = ~valid_present
                valid_mask = valid_mask.clone()
                valid_mask[missing] = True
            kwargs["taxel_valid_mask"] = valid_mask
    if present_fast is not None:
        kwargs["present_fast"] = present_fast
    if present_slow is not None:
        kwargs["present_slow"] = present_slow
    if present is not None:
        kwargs["present"] = present
    return kwargs, found_any


def pack_future_contact_tensors(
    examples: List[dict],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Stack ``region_pressure_future`` + valid mask for variant A.

    Returns ``(None, None)`` when no row carries the future window (fusion
    without future_contact, or a V-only member). ``future_valid`` is bool
    ``[B, H]``; missing rows are all-False.
    """
    future, present = stack_optional_stream(
        examples, DEFAULT_REGION_FUTURE_KEYS, device, dtype
    )
    valid, valid_present = stack_optional_stream(
        examples, DEFAULT_REGION_FUTURE_VALID_KEYS, device, dtype
    )
    if future is None:
        return None, None
    if valid is None:
        # Packed future without a mask: treat every step as present on rows
        # that actually had the future key.
        horizon = int(future.shape[2])
        valid_mask = torch.zeros(
            future.shape[0], horizon, dtype=torch.bool, device=device
        )
        if present is not None:
            valid_mask[present] = True
        return future, valid_mask
    valid_mask = valid > 0.5
    if valid_present is not None and not bool(valid_present.all()):
        missing = ~valid_present
        valid_mask = valid_mask.clone()
        valid_mask[missing] = False
    return future, valid_mask
