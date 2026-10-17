"""DDP-safe NaN/Inf fuse for the training loop.

Importable without Accelerate / transformers so unit tests can pin the
trip conditions without constructing a trainer.
"""

from __future__ import annotations

import math
import zlib

import torch
import torch.distributed as dist


def tensor_is_finite(value) -> bool:
    """True if ``value`` is None or every element is finite (NaN/Inf fail)."""
    if value is None:
        return True
    if not torch.is_tensor(value):
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            return False
    if value.numel() == 0:
        return True
    return bool(torch.isfinite(value.detach().abs().max()).item())


def grads_are_finite(model) -> bool:
    """True if every locally visible ``param.grad`` is finite.

    Under ZeRO each rank only sees its shard; pair this with
    ``assert_finite_training_tensors(..., extra_ok=...)`` so one rank's
    non-finite shard trips every rank. DeepSpeed's ``get_global_grad_norm``
    is intentionally unused: it is often stale until ``engine.step()``.
    """
    if model is None:
        return True
    parameters = model.parameters() if hasattr(model, "parameters") else []
    for param in parameters:
        grad = getattr(param, "grad", None)
        if grad is not None and not tensor_is_finite(grad):
            return False
    return True


def _collective_flag_device(parts) -> torch.device:
    for value in parts.values():
        if torch.is_tensor(value):
            return value.device
    if dist.is_available() and dist.is_initialized() and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def assert_finite_training_tensors(
    parts,
    *,
    step: int,
    extra_ok: bool = True,
    extra_name: str = "extra",
) -> None:
    """Abort every rank if a named tensor is non-finite, or ``extra_ok`` is False.

    All-reduces a 0/1 flag with MIN so a single rank's NaN cannot skip
    ``backward``/``step`` and leave the others hanging in a collective.
    There is no skip-and-continue path: that is not a fuse, and skipping
    backward on one rank desynchronizes DDP.
    """
    named = {str(key): value for key, value in parts.items() if value is not None}
    bad = [key for key, value in named.items() if not tensor_is_finite(value)]
    if not extra_ok:
        bad.append(str(extra_name))
    device = _collective_flag_device(parts)
    flag = (
        torch.zeros((), device=device, dtype=torch.float32)
        if bad
        else torch.ones((), device=device, dtype=torch.float32)
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if float(flag.item()) == 0.0:
        rank = dist.get_rank() if dist.is_initialized() else 0
        detail = ", ".join(bad) if bad else "non-finite on another rank"
        raise RuntimeError(
            f"[starVLA] NaN fuse tripped at step {step} (rank {rank}): {detail}. "
            "Training aborted so a poisoned backward/optimizer step cannot continue. "
            "Reload the last save_interval checkpoint."
        )


def assert_identical_rank_keys(keys, *, device, step: int, name: str = "metric keys") -> None:
    """Abort every rank if the local key set disagrees with any other rank.

    ``accelerator.reduce`` is one collective per key. This hash compares the
    *set*, not insertion order — trainers must then reduce in **sorted**
    key order so the same keys cannot pair the wrong tensors (or hang).
    A rank that is missing a key (or has an extra one) issues a different
    number of collectives and the job deadlocks. Hash the sorted key set
    and MIN/MAX-compare so a mismatch trips everyone *before* the reduce
    loop. Single-process is a no-op.
    """
    payload = "\0".join(sorted(str(key) for key in keys))
    digest = zlib.crc32(payload.encode("utf-8")) & 0xFFFFFFFF
    flag = torch.tensor([digest], device=device, dtype=torch.int64)
    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return
    low = flag.clone()
    high = flag.clone()
    dist.all_reduce(low, op=dist.ReduceOp.MIN)
    dist.all_reduce(high, op=dist.ReduceOp.MAX)
    if int(low.item()) != int(high.item()):
        rank = dist.get_rank()
        raise RuntimeError(
            f"[starVLA] {name} disagree across ranks at step {step} "
            f"(rank {rank} has {sorted(str(key) for key in keys)}). A key that "
            "appears on some ranks only would desynchronise accelerator.reduce. "
            "Emit the same metric keys on every rank (dummy zeros on skip)."
        )
