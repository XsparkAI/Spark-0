# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Local Spark-0 addition (NOT upstream starVLA).
"""
Future-contact variant A: train-only aux head on region expert tokens.

N0-VTLA's claim: the action chunk is decided by contact that is about to
happen, not the reading already in the glove. Variant A does not change
inference — it only shapes the same region tokens the Action-DiT already
reads, by asking them to predict region-level contact over the action
horizon. Variant B (runtime latent z) is not implemented; ``variant``
must be ``aux_head``.

Target (per region, per subwindow; §5.4.2):
  * contact transition ∈ {none, onset, release, repeated}
  * pressure bin     ∈ {none, light, medium, heavy}
  * mean Δpressure   relative to the current frame

The future window is a packed column, not an episode walk: the dataset
exports the next frames of region force alongside the history window
(``region_pressure_vector_future``, slot 0 = nearest to now), and the
loader slices the first ``horizon`` of them. ``horizon`` therefore counts
GLOVE frames at the sensor rate (EgoPro 60 Hz), not policy steps — the
default 10 is ~167 ms ahead, which is the contact the current chunk is
about to make, and it stays cheap because it costs no extra parquet row.

Only pressure / contact is supervised (``supervise_xyz`` is rejected if
true). Subwindows are masked independently so an episode tail keeps
near/mid. Absolute loss is dominated by "no change"; the module always
emits persistence and linear-extrapolation baselines so a probe can be
judged against those, not against a raw CE.

Disable: ``enable: false`` → do not construct (framework sets
``tactile_future_contact=None``). ``predict_action`` never calls this.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Future glove frames the head supervises when the stem leaves ``horizon``
# unset. Sensor frames (EgoPro 60 Hz), NOT policy steps: the packed future
# column shares its rate with the region history window.
DEFAULT_FUTURE_HORIZON = 10

TRANSITION_NONE = 0
TRANSITION_ONSET = 1
TRANSITION_RELEASE = 2
TRANSITION_REPEATED = 3
N_TRANSITION = 4
N_BIN = 4

# Logged on every rank whenever the module is on (empty batch still emits
# zeros) so the trainer reduce cannot hang on a missing key.
FUTURE_CONTACT_METRIC_KEYS: Tuple[str, ...] = (
    "future_contact_transition_loss",
    "future_contact_bin_loss",
    "future_contact_delta_l1",
    "future_contact_transition_acc",
    "future_contact_bin_acc",
    "future_contact_present_frac",
    "future_contact_persist_transition_acc",
    "future_contact_persist_bin_acc",
    "future_contact_persist_delta_l1",
    "future_contact_extrap_transition_acc",
    "future_contact_extrap_bin_acc",
    "future_contact_extrap_delta_l1",
)


def _as_dict(cfg: Any) -> Dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    to_container = getattr(cfg, "to_container", None)
    if callable(to_container):
        return dict(to_container(resolve=True))
    if hasattr(cfg, "items"):
        return {str(k): v for k, v in cfg.items()}
    return dict(cfg)


def subwindow_bounds(
    horizon: int,
    num_subwindows: int,
    splits: Optional[Sequence[int]] = None,
) -> List[Tuple[int, int]]:
    """Half-open ``[start, end)`` slices of a length-``horizon`` future.

    Even partitions, last window absorbing the remainder (H=10, K=3 →
    ``[0,3) / [3,6) / [6,10)``). H=50 keeps the §5.4.2 near/mid/far split
    ``[0,16) / [16,33) / [33,50)``.
    """
    if horizon < 1:
        raise ValueError(f"future_contact horizon must be >= 1, got {horizon}")
    if num_subwindows < 1:
        raise ValueError(
            f"future_contact.num_subwindows must be >= 1, got {num_subwindows}"
        )
    if splits is not None:
        edges = [0, *[int(x) for x in splits], horizon]
        if len(edges) != num_subwindows + 1:
            raise ValueError(
                f"future_contact.subwindow_splits must have "
                f"{num_subwindows - 1} interior edges, got {splits!r}"
            )
        if edges != sorted(edges) or edges[0] != 0 or edges[-1] != horizon:
            raise ValueError(
                f"future_contact.subwindow_splits must be increasing in "
                f"(0, {horizon}), got {splits!r}"
            )
        if any(edges[i] >= edges[i + 1] for i in range(len(edges) - 1)):
            raise ValueError(
                f"future_contact subwindows must be non-empty, got {splits!r}"
            )
        return [(edges[i], edges[i + 1]) for i in range(num_subwindows)]
    if num_subwindows == 3 and horizon == 50:
        return [(0, 16), (16, 33), (33, 50)]
    step = horizon // num_subwindows
    if step < 1:
        raise ValueError(
            f"future_contact horizon={horizon} cannot fit "
            f"{num_subwindows} subwindows"
        )
    bounds = []
    start = 0
    for k in range(num_subwindows):
        end = horizon if k == num_subwindows - 1 else start + step
        bounds.append((start, end))
        start = end
    return bounds


def flatten_region_pressure(window: torch.Tensor) -> torch.Tensor:
    """``[B, hands, regions, T, C]`` → ``[B, hands*regions, T]`` (channel 0).

    Channel 0 is the conditioned pressure. Extra Δp channels are ignored —
    mean Δ is computed here from the future window, not from a packed Δp.
    """
    if window.ndim != 5:
        raise ValueError(
            f"region pressure must be [B, H, R, T, C], got {tuple(window.shape)}"
        )
    batch, hands, regions, steps, _ = window.shape
    return window[..., 0].reshape(batch, hands * regions, steps)


def pressure_bin_index(
    magnitude: torch.Tensor, edges: Sequence[float]
) -> torch.Tensor:
    """``|p|`` → ``{0:none, 1:light, 2:medium, 3:heavy}`` from three edges."""
    if len(edges) != 3:
        raise ValueError(
            f"future_contact.bin_edges must be 3 floats "
            f"(none|light|medium|heavy), got {edges!r}"
        )
    e0, e1, e2 = (float(x) for x in edges)
    if not (0.0 <= e0 <= e1 <= e2):
        raise ValueError(
            f"future_contact.bin_edges must be non-decreasing and >= 0, "
            f"got {edges!r}"
        )
    bins = torch.zeros(magnitude.shape, dtype=torch.long, device=magnitude.device)
    bins = torch.where(magnitude >= e0, torch.ones_like(bins), bins)
    bins = torch.where(magnitude >= e1, torch.full_like(bins, 2), bins)
    bins = torch.where(magnitude >= e2, torch.full_like(bins, 3), bins)
    return bins


def transition_from_contact(
    current: torch.Tensor, future: torch.Tensor
) -> torch.Tensor:
    """Contact flags ``[B,R]`` + ``[B,R,T]`` → transition class ``[B,R]``.

    Events are measured on ``[current] + future`` so "onset" means contact
    appears relative to now, not only inside the window.
    """
    seq = torch.cat([current.unsqueeze(-1), future], dim=-1)
    prev = seq[..., :-1]
    nxt = seq[..., 1:]
    onset = ((~prev) & nxt).any(dim=-1)
    release = (prev & (~nxt)).any(dim=-1)
    label = torch.full(
        current.shape, TRANSITION_NONE, dtype=torch.long, device=current.device
    )
    label = torch.where(onset & ~release, torch.full_like(label, TRANSITION_ONSET), label)
    label = torch.where(
        release & ~onset, torch.full_like(label, TRANSITION_RELEASE), label
    )
    label = torch.where(
        onset & release, torch.full_like(label, TRANSITION_REPEATED), label
    )
    return label


def build_future_contact_targets(
    current: torch.Tensor,
    future: torch.Tensor,
    future_valid: torch.Tensor,
    bounds: Sequence[Tuple[int, int]],
    *,
    contact_abs: float,
    bin_edges: Sequence[float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack readable targets for one future window.

    Args:
        current: ``[B, R]`` conditioned pressure at the observation frame.
        future: ``[B, R, H]`` next ``H`` policy frames (t+1 … t+H).
        future_valid: ``[B, H]`` True when that future row exists.
        bounds: subwindow ``[start, end)`` in the H axis.

    Returns:
        ``transition, pressure_bin, mean_delta, mask`` each ``[B, R, K]``.
        A subwindow is valid only when every step in it is valid (and the
        current reading is finite). Incomplete far windows at episode end
        drop; near/mid stay.
    """
    if current.ndim != 2 or future.ndim != 3:
        raise ValueError(
            "current must be [B, R] and future [B, R, H], "
            f"got {tuple(current.shape)} / {tuple(future.shape)}"
        )
    if future_valid.ndim != 2 or future_valid.shape[0] != future.shape[0]:
        raise ValueError(
            f"future_valid must be [B, H], got {tuple(future_valid.shape)}"
        )
    if future_valid.shape[1] != future.shape[2]:
        raise ValueError(
            f"future_valid H={future_valid.shape[1]} != future H={future.shape[2]}"
        )
    batch, regions, _ = future.shape
    n_win = len(bounds)
    device = future.device
    transition = torch.zeros(batch, regions, n_win, dtype=torch.long, device=device)
    bins = torch.zeros(batch, regions, n_win, dtype=torch.long, device=device)
    mean_delta = torch.zeros(batch, regions, n_win, dtype=future.dtype, device=device)
    mask = torch.zeros(batch, regions, n_win, dtype=torch.bool, device=device)
    current_ok = torch.isfinite(current)
    current_contact = current.abs() > float(contact_abs)
    for k, (start, end) in enumerate(bounds):
        win = future[:, :, start:end]
        win_valid = future_valid[:, start:end]
        complete = win_valid.all(dim=-1) if win_valid.numel() else current.new_zeros(
            batch, dtype=torch.bool
        )
        finite_win = torch.isfinite(win).all(dim=-1)
        slot = complete.unsqueeze(1) & finite_win & current_ok
        contact = win.abs() > float(contact_abs)
        transition[:, :, k] = transition_from_contact(current_contact, contact)
        peak = win.abs().amax(dim=-1)
        bins[:, :, k] = pressure_bin_index(peak, bin_edges)
        mean_delta[:, :, k] = (win - current.unsqueeze(-1)).mean(dim=-1)
        mask[:, :, k] = slot
    return transition, bins, mean_delta, mask


def linear_extrap_future(
    history: torch.Tensor, horizon: int
) -> torch.Tensor:
    """Constant-velocity rollout from a chronological history window.

    ``history`` is ``[B, R, T]`` (oldest → current). Velocity is
    ``(p_now - p_oldest) / (T-1)``; ``T=1`` degenerates to persistence.
    Step ``i`` (0-based) is ``p_now + (i+1) * vel`` — the next policy frame.
    """
    if history.ndim != 3:
        raise ValueError(f"history must be [B, R, T], got {tuple(history.shape)}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    now = history[:, :, -1]
    steps = history.shape[-1]
    if steps <= 1:
        vel = torch.zeros_like(now)
    else:
        vel = (now - history[:, :, 0]) / float(steps - 1)
    idx = torch.arange(1, horizon + 1, device=history.device, dtype=history.dtype)
    return now.unsqueeze(-1) + vel.unsqueeze(-1) * idx.view(1, 1, -1)


def get_future_contact(
    cfg: Any, *, token_dim: int, num_regions: int
) -> "FutureContactHead":
    """Factory used by ``QwenPI_v3`` when ``tactile_expert.future_contact.enable``."""
    return FutureContactHead(
        token_dim=token_dim,
        num_regions=num_regions,
        cfg=cfg,
    )


class FutureContactHead(nn.Module):
    """2-layer MLP (+ optional region self-attn) → three linear heads."""

    def __init__(
        self,
        *,
        token_dim: int,
        num_regions: int,
        cfg: Any,
    ):
        super().__init__()
        cfg = _as_dict(cfg)
        if not bool(cfg.get("enable", False)):
            raise ValueError(
                "FutureContactHead requires enable=true; the framework "
                "must skip construction when future_contact.enable=false."
            )
        variant = str(cfg.get("variant", "aux_head")).lower()
        if variant != "aux_head":
            raise ValueError(
                f"future_contact.variant={variant!r} is not implemented; "
                "only aux_head (variant A) is wired. latent_z is Stage D+."
            )
        if bool(cfg.get("supervise_xyz", False)):
            raise ValueError(
                "future_contact.supervise_xyz must stay false: future xyz "
                "is the policy's own hand command and would leak action "
                "into the tactile branch (§5.4.1)."
            )
        granularity = str(cfg.get("mask_granularity", "subwindow")).lower()
        if granularity != "subwindow":
            raise ValueError(
                f"future_contact.mask_granularity={granularity!r} is not "
                "implemented; only subwindow mask is supported."
            )
        horizon = cfg.get("horizon", None)
        self.horizon = int(
            DEFAULT_FUTURE_HORIZON if horizon in (None, "", "null") else horizon
        )
        self.num_subwindows = int(cfg.get("num_subwindows", 3))
        splits = cfg.get("subwindow_splits", None)
        self.bounds = subwindow_bounds(self.horizon, self.num_subwindows, splits)
        self.num_regions = int(num_regions)
        self.token_dim = int(token_dim)
        if self.num_regions < 1 or self.token_dim < 1:
            raise ValueError("future_contact needs num_regions >= 1 and token_dim >= 1")
        self.contact_abs = float(cfg.get("contact_abs", 0.05))
        if self.contact_abs < 0.0:
            raise ValueError("future_contact.contact_abs must be >= 0")
        edges = cfg.get("bin_edges", [0.05, 0.25, 0.6])
        self.bin_edges = tuple(float(x) for x in edges)
        # Validate edges once (pressure_bin_index raises with the same rule).
        pressure_bin_index(torch.zeros(1), self.bin_edges)
        self.loss_coeff = float(cfg.get("loss_coeff", 0.05))
        if self.loss_coeff < 0.0:
            raise ValueError("future_contact.loss_coeff must be >= 0")
        hidden = int(cfg.get("hidden_dim", token_dim))
        if hidden < 1:
            raise ValueError("future_contact.hidden_dim must be >= 1")
        self.mix = nn.Sequential(
            nn.LayerNorm(self.token_dim),
            nn.Linear(self.token_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        use_attn = bool(cfg.get("use_self_attention", True))
        if use_attn:
            n_heads = 4 if hidden % 4 == 0 else 1
            self.attn = nn.MultiheadAttention(
                hidden, num_heads=n_heads, batch_first=True
            )
            self.attn_norm = nn.LayerNorm(hidden)
        else:
            self.attn = None
            self.attn_norm = None
        self.transition_head = nn.Linear(hidden, self.num_subwindows * N_TRANSITION)
        self.bin_head = nn.Linear(hidden, self.num_subwindows * N_BIN)
        self.delta_head = nn.Linear(hidden, self.num_subwindows)

    def _heads(
        self, region_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.mix(region_tokens)
        if self.attn is not None:
            attn_out, _ = self.attn(hidden, hidden, hidden, need_weights=False)
            hidden = self.attn_norm(hidden + attn_out)
        batch, regions, _ = hidden.shape
        k = self.num_subwindows
        transition = self.transition_head(hidden).view(batch, regions, k, N_TRANSITION)
        bins = self.bin_head(hidden).view(batch, regions, k, N_BIN)
        delta = self.delta_head(hidden).view(batch, regions, k)
        return transition, bins, delta

    def _zero_metrics(
        self, touch: torch.Tensor, *, device: torch.device, dtype: torch.dtype
    ) -> Dict[str, torch.Tensor]:
        # Keep the head in the graph (ZeRO-3). Logged metrics are detached.
        loss = touch.to(device=device, dtype=dtype) * 0
        zero = torch.zeros((), device=device, dtype=dtype)
        metrics = {"loss": loss}
        for key in FUTURE_CONTACT_METRIC_KEYS:
            metrics[key] = zero
        return metrics

    def forward(
        self,
        region_tokens: torch.Tensor,
        *,
        history: Optional[torch.Tensor],
        future: Optional[torch.Tensor],
        future_valid: Optional[torch.Tensor],
        present: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Predict from current tokens; supervise with the packed future.

        ``region_tokens`` is the projected region block the DiT reads
        (``TactileEncoder.region_token_span``). History / future are the
        packed region-pressure windows. Every call runs the heads (ZeRO-3);
        a batch with no valid subwindow still returns the full metric key
        set as zeros.
        """
        if region_tokens.ndim != 3:
            raise ValueError(
                f"region_tokens must be [B, R, D], got {tuple(region_tokens.shape)}"
            )
        batch, regions, dim = region_tokens.shape
        if regions != self.num_regions:
            raise ValueError(
                f"region_tokens R={regions} != future_contact.num_regions="
                f"{self.num_regions} (encoder region_token_span)"
            )
        if dim != self.token_dim:
            raise ValueError(
                f"region_tokens D={dim} != future_contact.token_dim={self.token_dim}"
            )
        trans_logits, bin_logits, delta_pred = self._heads(region_tokens)
        touch = (
            trans_logits.float().sum() * 0
            + bin_logits.float().sum() * 0
            + delta_pred.float().sum() * 0
        )
        device = region_tokens.device
        dtype = torch.float32
        if future is None or history is None:
            return self._zero_metrics(touch, device=device, dtype=dtype)

        # Targets are built in fp32: under DeepSpeed bf16 the packed windows
        # arrive in the encoder's dtype, and a bf16 mean over the subwindow
        # would quantise the Δp target to ~3 decimal digits.
        hist_flat = flatten_region_pressure(history).float()
        fut_flat = flatten_region_pressure(future).float()
        if hist_flat.shape[1] != regions or fut_flat.shape[1] != regions:
            raise ValueError(
                "packed region hands×regions must match region tokens: "
                f"history R={hist_flat.shape[1]}, future R={fut_flat.shape[1]}, "
                f"tokens R={regions}"
            )
        if fut_flat.shape[-1] != self.horizon:
            raise ValueError(
                f"packed region_pressure_future H={fut_flat.shape[-1]} != "
                f"future_contact.horizon={self.horizon}. The loader slices "
                "the exported future column to datasets.vla_data."
                "tactile_future_horizon; sync it with the stem."
            )
        current = hist_flat[:, :, -1]
        if future_valid is None:
            future_valid = torch.ones(
                batch, self.horizon, dtype=torch.bool, device=device
            )
        else:
            future_valid = future_valid.to(device=device, dtype=torch.bool)
            if future_valid.ndim != 2 or future_valid.shape != (batch, self.horizon):
                raise ValueError(
                    f"future_valid must be [B, {self.horizon}], "
                    f"got {tuple(future_valid.shape)}"
                )
        transition, bins, mean_delta, mask = build_future_contact_targets(
            current,
            fut_flat,
            future_valid,
            self.bounds,
            contact_abs=self.contact_abs,
            bin_edges=self.bin_edges,
        )
        if present is not None:
            present = present.to(device=device, dtype=torch.bool)
            if present.shape != (batch,):
                raise ValueError(f"present must be [B], got {tuple(present.shape)}")
            mask = mask & present.view(batch, 1, 1)

        weight = mask.to(dtype=dtype)
        denom = weight.sum().clamp(min=1.0)
        present_frac = mask.float().mean()

        trans_ce = F.cross_entropy(
            trans_logits.float().reshape(-1, N_TRANSITION),
            transition.reshape(-1),
            reduction="none",
        ).view_as(transition)
        bin_ce = F.cross_entropy(
            bin_logits.float().reshape(-1, N_BIN),
            bins.reshape(-1),
            reduction="none",
        ).view_as(bins)
        delta_l1 = (delta_pred.float() - mean_delta.float()).abs()
        trans_loss = (trans_ce * weight).sum() / denom
        bin_loss = (bin_ce * weight).sum() / denom
        delta_loss = (delta_l1 * weight).sum() / denom
        raw = trans_loss + bin_loss + delta_loss
        loss = raw * self.loss_coeff + touch.to(dtype=raw.dtype)

        trans_acc = (
            (trans_logits.argmax(dim=-1) == transition).to(dtype=dtype) * weight
        ).sum() / denom
        bin_acc = (
            (bin_logits.argmax(dim=-1) == bins).to(dtype=dtype) * weight
        ).sum() / denom

        persist_trans = torch.full_like(transition, TRANSITION_NONE)
        persist_bin = pressure_bin_index(current.abs(), self.bin_edges).unsqueeze(-1)
        persist_bin = persist_bin.expand_as(bins)
        persist_delta = torch.zeros_like(mean_delta)
        persist_t_acc, persist_b_acc, persist_d = _hard_baseline(
            persist_trans, persist_bin, persist_delta, transition, bins, mean_delta, weight
        )

        extrap = linear_extrap_future(hist_flat, self.horizon)
        # Baselines are judged on the same valid subwindows as the model,
        # including steps the linear rollout invents past episode end —
        # those slots stay masked, so the comparison is on real future only.
        e_trans, e_bin, e_delta, _ = build_future_contact_targets(
            current,
            extrap,
            torch.ones_like(future_valid),
            self.bounds,
            contact_abs=self.contact_abs,
            bin_edges=self.bin_edges,
        )
        extrap_t_acc, extrap_b_acc, extrap_d = _hard_baseline(
            e_trans, e_bin, e_delta, transition, bins, mean_delta, weight
        )

        # A fully-masked batch (V-only / episode tail / dropout) still has
        # to emit the same keys. Acc/L1 on an empty mask are zeros, not 1.
        if not bool(mask.any()):
            return self._zero_metrics(touch, device=device, dtype=dtype)

        return {
            "loss": loss,
            "future_contact_transition_loss": trans_loss.detach(),
            "future_contact_bin_loss": bin_loss.detach(),
            "future_contact_delta_l1": delta_loss.detach(),
            "future_contact_transition_acc": trans_acc.detach(),
            "future_contact_bin_acc": bin_acc.detach(),
            "future_contact_present_frac": present_frac.detach(),
            "future_contact_persist_transition_acc": persist_t_acc.detach(),
            "future_contact_persist_bin_acc": persist_b_acc.detach(),
            "future_contact_persist_delta_l1": persist_d.detach(),
            "future_contact_extrap_transition_acc": extrap_t_acc.detach(),
            "future_contact_extrap_bin_acc": extrap_b_acc.detach(),
            "future_contact_extrap_delta_l1": extrap_d.detach(),
        }


def _hard_baseline(
    pred_trans: torch.Tensor,
    pred_bin: torch.Tensor,
    pred_delta: torch.Tensor,
    true_trans: torch.Tensor,
    true_bin: torch.Tensor,
    true_delta: torch.Tensor,
    weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Accuracy / L1 of a discrete baseline vs the same targets."""
    denom = weight.sum().clamp(min=1.0)
    t_acc = ((pred_trans == true_trans).to(dtype=weight.dtype) * weight).sum() / denom
    b_acc = ((pred_bin == true_bin).to(dtype=weight.dtype) * weight).sum() / denom
    d_l1 = ((pred_delta.float() - true_delta.float()).abs() * weight).sum() / denom
    return t_acc, b_acc, d_l1


__all__ = [
    "DEFAULT_FUTURE_HORIZON",
    "FUTURE_CONTACT_METRIC_KEYS",
    "FutureContactHead",
    "N_BIN",
    "N_TRANSITION",
    "TRANSITION_NONE",
    "TRANSITION_ONSET",
    "TRANSITION_RELEASE",
    "TRANSITION_REPEATED",
    "build_future_contact_targets",
    "flatten_region_pressure",
    "get_future_contact",
    "linear_extrap_future",
    "pressure_bin_index",
    "subwindow_bounds",
    "transition_from_contact",
]
