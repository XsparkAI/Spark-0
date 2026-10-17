# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Local Spark-0 addition (NOT upstream starVLA).
"""
Contact-region → Qwen visual-token alignment (train-only attention regularizer).

What this implements (Spark-0 design §2.5)
─────────────────────────────────────────
Fine-grained 2D contact evidence on the head camera is aggregated to region /
token-grid targets and used to regularize intermediate Qwen visual tokens so
they encode "where contact is / will be". At inference the whole pathway is
deleted (never constructed on the predict_action path).

Orthogonal to:
  * ``framework.spatial_alignment`` (DA3 geometry distillation)
  * ``framework.tactile`` / ``tactile_fusion`` (3D → Action-DiT context)

Disable contract
────────────────
``enable: false`` → do not construct (framework sets ``tactile_align=None``).
Constructing with ``enable=false`` raises. Missing / non-alignable rows use
``present2d=False``; the framework emits a zero loss that still touches every
trainable parameter (DDP/ZeRO-safe), same pattern as DA3 flex-view skips.

Config (``framework.tactile_align``, usually via ``tactile_align: <stem>``)
──────────────────────────────────────────────────────────────────────────
See ``configs/tactile_align/default.yaml`` and the region / loss /
projector / skip blocks therein.

Sparse contact gate (``skip.min_contact_frac``)
───────────────────────────────────────────────
Heatmaps are mostly zero: supervising every visual token on an empty frame
teaches the aligned layer "no contact anywhere", fighting the DA3 geometry
loss on the same layer — and per-sample ``normalize: max`` pulls a noise-only
frame up to peak 1, minting fake contact positives. The sample-level gate
judges the RAW heatmap (physical units, or log1p of it — never the
max-normalized map) before targets are built:

  * ``mass >= min_contact_frac`` (fraction of pixels > ``min_contact_abs``)
    → supervised as before;
  * all-zero frame (explicitly idle) → kept only with ``idle_keep_prob``
    (train-only draw, same RNG pattern as ``align_dropout_prob``);
  * ``0 < mass < min_contact_frac`` (noise frame) → skipped.

Code default ``min_contact_frac: 0.0`` = gate off = bit-identical baseline
(no RNG consumed). The shipped ``tactile_align/default.yaml`` turns the gate
on (``0.005`` / ``idle_keep_prob: 0.05``). DA3 stays dense — this gate only
lives here. The kept fraction is logged as ``contact_align_sparse_keep_frac``
(1.0 when the gate is off or no heatmap rows exist).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.modules.spatial_alignment.layout_groups import parse_align_view_indices
from starVLA.model.modules.tactile_align.region_targets import (
    parse_region_cfg,
    targets_from_saliency,
)
from starVLA.model.modules.tactile_align.token_map import (
    prepare_heatmap_contact_map,
    student_grid_hw,
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


def get_tactile_align(cfg: Any, vl_hidden_dim: int) -> "ContactRegionAlignment":
    """Factory used by ``QwenPI_v3`` when ``framework.tactile_align.enable``."""
    return ContactRegionAlignment(vl_hidden_dim, cfg)


class ContactRegionAlignment(nn.Module):
    """Trainable projector + contact / InfoNCE / pressure-bin losses."""

    def __init__(self, vl_hidden_dim: int, cfg: Any):
        super().__init__()
        cfg = _as_dict(cfg)
        if not bool(cfg.get("enable", False)):
            raise ValueError(
                "ContactRegionAlignment requires enable=true; the framework "
                "must skip construction when tactile_align.enable=false."
            )

        self.vl_hidden_dim = int(vl_hidden_dim)
        parsed_views = parse_align_view_indices(
            cfg.get("align_view_indices", [0]),
            field="tactile_align.align_view_indices",
        )
        self.align_view_indices: Optional[List[int]] = (
            [0] if parsed_views is None else parsed_views
        )
        if len(self.align_view_indices) != 1:
            # Targets (contact_map / region_token_mask) are built for ONE
            # student grid. Multiple views would concat tokens to P=ΣP_v
            # while the target stays P_0. Fail at construct rather than
            # mid-training (or, worse, as a silent zero-loss skip).
            raise ValueError(
                "tactile_align supports exactly one supervised view; got "
                f"align_view_indices={self.align_view_indices}. Use [0] "
                "(head camera) until multi-view targets are implemented."
            )

        self.align_layer_index = int(cfg.get("align_layer_index", 25))

        loss_cfg = _as_dict(cfg.get("loss", {}))
        self.infonce_coeff = float(loss_cfg.get("infonce_coeff", 0.1))
        self.contact_map_coeff = float(loss_cfg.get("contact_map_coeff", 0.1))
        self.pressure_bin_coeff = float(loss_cfg.get("pressure_bin_coeff", 0.05))
        self.dice_coeff = float(loss_cfg.get("dice_coeff", 0.1))
        self.temperature = float(loss_cfg.get("temperature", 0.07))
        self.focal_gamma = float(loss_cfg.get("focal_gamma", 2.0))
        if self.temperature <= 0:
            raise ValueError("tactile_align.loss.temperature must be > 0")

        # HxW heatmap → token-grid contact map (primary alignment path).
        heat_cfg = _as_dict(cfg.get("heatmap", {}))
        self.heatmap_normalize = str(heat_cfg.get("normalize", "max"))
        obs = heat_cfg.get("obs_hw", None)
        if obs is None:
            self.heatmap_obs_hw: Optional[Tuple[int, int]] = None
        else:
            if not isinstance(obs, (list, tuple)) or len(obs) != 2:
                raise ValueError(
                    f"tactile_align.heatmap.obs_hw must be [H, W] or null, got {obs!r}"
                )
            self.heatmap_obs_hw = (int(obs[0]), int(obs[1]))

        proj_cfg = _as_dict(cfg.get("projector", {}))
        out_dim = int(proj_cfg.get("out_dim", 128))
        hidden_dim = int(proj_cfg.get("hidden_dim", 256))
        use_norm = bool(proj_cfg.get("use_vlm_norm", True))
        self.out_dim = out_dim
        self.vlm_norm = nn.LayerNorm(vl_hidden_dim) if use_norm else nn.Identity()
        self.fc1 = nn.Linear(vl_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        # Dedicated token → contact logit head for HxW-pooled maps (not a
        # mean-of-embedding proxy).
        self.map_head = nn.Linear(out_dim, 1)

        region_cfg = parse_region_cfg(cfg.get("region", {}))
        self.num_hands = region_cfg["num_hands"]
        self.num_regions_per_hand = region_cfg["num_regions_per_hand"]
        self.num_regions = self.num_hands * self.num_regions_per_hand
        self.force_bins = region_cfg["force_bins"]
        # bin classes = len(edges) + 1 (open last bin), including "none"
        self.num_bins = len(self.force_bins) + 1
        self.bin_head = nn.Linear(out_dim, self.num_bins)

        skip_cfg = _as_dict(cfg.get("skip", {}))
        self.require_present2d = bool(skip_cfg.get("require_present2d", True))
        self.min_align_weight = float(skip_cfg.get("min_align_weight", 1e-6))
        # Sparse contact gate (see module docstring). 0.0 = off = baseline.
        self.min_contact_frac = float(skip_cfg.get("min_contact_frac", 0.0))
        self.min_contact_abs = float(skip_cfg.get("min_contact_abs", 0.05))
        self.contact_log1p = bool(skip_cfg.get("contact_log1p", False))
        self.idle_keep_prob = float(skip_cfg.get("idle_keep_prob", 0.0))
        if not (0.0 <= self.min_contact_frac <= 1.0):
            raise ValueError(
                f"tactile_align.skip.min_contact_frac must be in [0, 1], "
                f"got {self.min_contact_frac}"
            )
        if self.min_contact_abs < 0.0:
            raise ValueError(
                f"tactile_align.skip.min_contact_abs must be >= 0, "
                f"got {self.min_contact_abs}"
            )
        if not (0.0 <= self.idle_keep_prob <= 1.0):
            raise ValueError(
                f"tactile_align.skip.idle_keep_prob must be in [0, 1], "
                f"got {self.idle_keep_prob}"
            )

        self.align_dropout_prob = float(cfg.get("align_dropout_prob", 0.0))
        if not (0.0 <= self.align_dropout_prob <= 1.0):
            raise ValueError(
                f"align_dropout_prob must be in [0, 1], got {self.align_dropout_prob}"
            )

    def project(self, tokens: torch.Tensor) -> torch.Tensor:
        """``[..., C]`` visual tokens → L2-normalized contact subspace."""
        x = self.vlm_norm(tokens)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return F.normalize(x.float(), dim=-1)

    def apply_align_dropout(self, present2d: torch.Tensor) -> torch.Tensor:
        """Train-only random clear of ``present2d`` bits."""
        if (
            not self.training
            or self.align_dropout_prob <= 0.0
            or not bool(present2d.any())
        ):
            return present2d
        keep = torch.rand(present2d.shape[0], device=present2d.device) >= self.align_dropout_prob
        return present2d & keep

    def _sparse_gate_keep(self, heatmap: Any) -> bool:
        """Raw-heatmap contact gate (see module docstring). Train-only draw
        for all-zero idle frames; contact / noise decisions are deterministic.
        Non-finite pixels count as 0 so a NaN frame cannot mint contact.
        """
        raw = torch.as_tensor(heatmap, dtype=torch.float32)
        raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        if self.contact_log1p:
            raw = torch.log1p(raw.clamp_min(0))
        mass = float((raw > self.min_contact_abs).float().mean())
        if mass >= self.min_contact_frac:
            return True
        if float(raw.max()) <= 0.0:
            # Explicitly idle frame: keep a small fraction so "no contact
            # this frame" is not unlearned entirely.
            return self.training and float(torch.rand(())) < self.idle_keep_prob
        return False  # noise frame: 0 < mass < min_contact_frac

    def resolve_view_indices(self, views_per_sample: int) -> List[int]:
        assert self.align_view_indices is not None
        out_of_range = [i for i in self.align_view_indices if not 0 <= i < views_per_sample]
        if out_of_range:
            raise ValueError(
                f"tactile_align.align_view_indices={self.align_view_indices} "
                f"references view(s) {out_of_range}, but the batch only provides "
                f"{views_per_sample} packed camera(s)."
            )
        return list(self.align_view_indices)

    def forward(
        self,
        student_tokens: torch.Tensor,
        *,
        present2d: torch.Tensor,
        contact_map: Optional[torch.Tensor] = None,
        region_contact: Optional[torch.Tensor] = None,
        region_bin: Optional[torch.Tensor] = None,
        region_token_mask: Optional[torch.Tensor] = None,
        align_weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute weighted contact-align loss on one (concatenated) view grid.

        Args:
            student_tokens: ``[B, P, C]`` visual tokens of the supervised view(s)
                already gathered / concatenated in packed view order.
            present2d: ``[B]`` bool — row contributes to the loss when True.
            contact_map: optional ``[B, P]`` soft labels in ``[0, 1]``.
            region_contact: optional ``[B, R]`` contact existence.
            region_bin: optional ``[B, R]`` long pressure-bin ids.
            region_token_mask: optional ``[B, R, P]`` bool token membership.
            align_weight: optional ``[B]`` soft weights (default 1 for present).

        Returns:
            dict with ``loss`` (already coeff-weighted sum) and detached metrics.
            When no row is present, returns zeros that still touch parameters.
        """
        packed = {
            "present2d": present2d,
            "contact_map": contact_map,
            "region_contact": region_contact,
            "region_bin": region_bin,
            "region_token_mask": region_token_mask,
            "align_weight": align_weight,
        }
        return self.forward_groups(
            [(student_tokens, packed)],
            batch_size=int(student_tokens.shape[0]),
        )

    def forward_groups(
        self,
        segments: Sequence[Tuple[torch.Tensor, Dict[str, Any]]],
        *,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, torch.Tensor]:
        """One projector / map_head call over mixed-layout segments.

        Each segment is ``(student_tokens [B_g, P_g, C], packed targets)``.
        Different ``P_g`` (head-only vs 3-view student grids, or different
        ``obs_image_size``) cannot stack on the batch axis, so tokens are
        flattened, projected once, then split for the per-grid losses.
        ``batch_size`` is the original mixed batch (including layout groups
        this caller skipped) so ``contact_align_present_frac`` stays a
        fraction of every sample. Empty ``segments`` still projects a dummy
        row (DDP/ZeRO-safe, rank-invariant call count).
        """
        if int(batch_size) <= 0:
            raise ValueError(f"forward_groups batch_size must be > 0, got {batch_size}.")
        tokens_list: List[torch.Tensor] = []
        packed_list: List[Dict[str, Any]] = []
        shapes: List[Tuple[int, int, int]] = []
        for tokens, packed in segments:
            if tokens.ndim != 3:
                raise ValueError(
                    f"student_tokens must be [B, P, C], got {tuple(tokens.shape)}"
                )
            b, p, c = tokens.shape
            if c != self.vl_hidden_dim:
                raise ValueError(
                    f"student_tokens C={c} != vl_hidden_dim={self.vl_hidden_dim}"
                )
            tokens_list.append(tokens)
            packed_list.append(packed)
            shapes.append((b, p, c))
            if device is None:
                device = tokens.device
            if dtype is None:
                dtype = tokens.dtype
        if device is None:
            device = torch.device("cpu")
        if dtype is None:
            dtype = torch.float32

        if tokens_list:
            flat = torch.cat(
                [tokens.reshape(-1, tokens.shape[-1]) for tokens in tokens_list],
                dim=0,
            )
        else:
            flat = torch.zeros(
                1, self.vl_hidden_dim, device=device, dtype=dtype
            )

        projected_flat = self.project(flat)  # [N, D]
        map_logits_flat = self.map_head(projected_flat).squeeze(-1)  # [N]
        touch = (
            projected_flat.float().sum() * 0
            + map_logits_flat.float().sum() * 0
            + self.bin_head.weight.float().sum() * 0
        )
        if self.bin_head.bias is not None:
            touch = touch + self.bin_head.bias.float().sum() * 0

        if not tokens_list:
            metrics = {
                "contact_map_loss": touch.detach(),
                "dice_loss": touch.detach(),
                "infonce_loss": touch.detach(),
                "pressure_bin_loss": touch.detach(),
                "contact_align_present_frac": torch.zeros(
                    (), device=device, dtype=torch.float32
                ),
                # No heatmap rows at all → nothing dropped.
                "contact_align_sparse_keep_frac": torch.ones(
                    (), device=device, dtype=torch.float32
                ),
            }
            return {"loss": touch, **metrics}

        offset = 0
        weighted_losses: List[Tuple[torch.Tensor, torch.Tensor]] = []
        metric_acc = {
            "contact_map_loss": [],
            "dice_loss": [],
            "infonce_loss": [],
            "pressure_bin_loss": [],
        }
        present_count = torch.zeros((), device=device, dtype=torch.float32)
        for (b, p, _), packed in zip(shapes, packed_list):
            n = b * p
            projected = projected_flat[offset : offset + n].view(b, p, -1)
            map_logits = map_logits_flat[offset : offset + n].view(b, p)
            offset += n
            group_loss, group_metrics, group_w, n_present = self._losses_from_projected(
                projected,
                map_logits,
                packed,
                param_touch=touch,
            )
            weighted_losses.append((group_loss, group_w))
            present_count = present_count + n_present
            for key in metric_acc:
                metric_acc[key].append((group_metrics[key], group_w.detach()))

        weight_sum = torch.stack([w for _, w in weighted_losses]).sum().clamp_min(1e-6)
        # group_loss is the SUM of per-sample terms (mean * group_weight).
        total = touch + sum(loss for loss, _ in weighted_losses) / weight_sum
        sparse_kept = sum(packed.get("sparse_keep", (0, 0))[0] for packed in packed_list)
        sparse_total = sum(packed.get("sparse_keep", (0, 0))[1] for packed in packed_list)
        keep_frac = sparse_kept / sparse_total if sparse_total > 0 else 1.0
        metrics = {
            "contact_align_present_frac": (present_count / float(batch_size)).detach(),
            "contact_align_sparse_keep_frac": torch.tensor(
                keep_frac, device=device, dtype=torch.float32
            ),
        }
        for key, parts in metric_acc.items():
            metrics[key] = (
                sum(val * w for val, w in parts) / weight_sum
            ).detach()
        return {"loss": total, **metrics}

    def _losses_from_projected(
        self,
        projected: torch.Tensor,
        map_logits: torch.Tensor,
        packed: Dict[str, Any],
        *,
        param_touch: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Per-segment losses given already-projected tokens.

        Returns ``(loss, metrics, weight_sum, n_present)``. ``loss`` is the
        coeff-weighted sum **before** dividing by ``weight_sum`` so the
        caller can re-normalize across segments.
        """
        B, P, _ = projected.shape
        device = projected.device
        present2d = packed["present2d"].to(device=device, dtype=torch.bool).view(B)
        present2d = self.apply_align_dropout(present2d)
        align_weight = packed.get("align_weight")
        if align_weight is None:
            align_weight = present2d.to(dtype=torch.float32)
        else:
            align_weight = align_weight.to(device=device, dtype=torch.float32).view(B)
            align_weight = torch.where(
                present2d, align_weight, torch.zeros_like(align_weight)
            )
        align_weight = torch.where(
            align_weight >= self.min_align_weight,
            align_weight,
            torch.zeros_like(align_weight),
        )
        present2d = align_weight > 0
        n_present = present2d.float().sum()
        metrics: Dict[str, torch.Tensor] = {}
        w = align_weight
        weight_sum = w.sum()
        contact_map = packed.get("contact_map")
        region_contact = packed.get("region_contact")
        region_bin = packed.get("region_bin")
        region_token_mask = packed.get("region_token_mask")

        if not bool(present2d.any()):
            zero = param_touch
            metrics["contact_map_loss"] = zero.detach()
            metrics["dice_loss"] = zero.detach()
            metrics["infonce_loss"] = zero.detach()
            metrics["pressure_bin_loss"] = zero.detach()
            return zero, metrics, torch.zeros((), device=device, dtype=torch.float32), n_present

        total = param_touch * 0
        if (
            (self.contact_map_coeff > 0 or self.dice_coeff > 0)
            and contact_map is not None
        ):
            cmap = contact_map.to(device=device, dtype=torch.float32)
            if cmap.shape != (B, P):
                raise ValueError(
                    f"contact_map must be [B, P]=({B}, {P}), got {tuple(cmap.shape)}"
                )
            focal = self._focal_bce_map(map_logits, cmap, w)
            dice = self._dice_loss(map_logits, cmap, w)
            # Helpers already divide by weight.sum(); undo so we can merge.
            total = total + (
                self.contact_map_coeff * focal + self.dice_coeff * dice
            ) * weight_sum
            metrics["contact_map_loss"] = focal.detach()
            metrics["dice_loss"] = dice.detach()
        else:
            metrics["contact_map_loss"] = param_touch.detach()
            metrics["dice_loss"] = param_touch.detach()

        if (
            self.infonce_coeff > 0
            and region_contact is not None
            and region_token_mask is not None
        ):
            nce = self._region_infonce(
                projected, region_contact, region_token_mask, w
            )
            total = total + self.infonce_coeff * nce * weight_sum
            metrics["infonce_loss"] = nce.detach()
        else:
            metrics["infonce_loss"] = param_touch.detach()

        if (
            self.pressure_bin_coeff > 0
            and region_contact is not None
            and region_bin is not None
            and region_token_mask is not None
        ):
            bin_loss = self._pressure_bin_loss(
                projected, region_contact, region_bin, region_token_mask, w
            )
            total = total + self.pressure_bin_coeff * bin_loss * weight_sum
            metrics["pressure_bin_loss"] = bin_loss.detach()
        else:
            metrics["pressure_bin_loss"] = param_touch.detach()

        return total, metrics, weight_sum, n_present

    def _focal_bce_map(
        self,
        logits: torch.Tensor,
        contact_map: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """Token-wise focal BCE on learned map-head logits vs HxW-pooled labels."""
        targets = contact_map.clamp(0, 1)
        prob = torch.sigmoid(logits.float())
        bce = F.binary_cross_entropy_with_logits(
            logits.float(), targets, reduction="none"
        )
        p_t = prob * targets + (1 - prob) * (1 - targets)
        focal = (1 - p_t).pow(self.focal_gamma) * bce  # [B, P]
        per = focal.mean(dim=-1)  # [B]
        denom = weight.sum().clamp_min(1e-6)
        return (per * weight).sum() / denom

    def _dice_loss(
        self,
        logits: torch.Tensor,
        contact_map: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """Soft Dice on the token-grid contact map (good for sparse heatmaps)."""
        prob = torch.sigmoid(logits.float())
        targets = contact_map.clamp(0, 1)
        # Per-sample Dice; empty maps → 0 contribution via weight.
        inter = (prob * targets).sum(dim=-1)
        denom = prob.sum(dim=-1) + targets.sum(dim=-1)
        dice = 1.0 - (2.0 * inter + 1e-6) / (denom + 1e-6)
        return (dice * weight).sum() / weight.sum().clamp_min(1e-6)

    def _masked_pool(
        self,
        projected: torch.Tensor,
        region_token_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pool tokens per region → ``[B, R, D]`` + ``[B, R]`` valid mask."""
        mask = region_token_mask.to(dtype=projected.dtype)  # [B, R, P]
        denom = mask.sum(dim=-1).clamp_min(1e-6)  # [B, R]
        pooled = torch.einsum("brp,bpd->brd", mask, projected.float()) / denom.unsqueeze(-1)
        pooled = F.normalize(pooled, dim=-1)
        valid = region_token_mask.any(dim=-1)  # [B, R]
        return pooled, valid

    def _region_infonce(
        self,
        projected: torch.Tensor,
        region_contact: torch.Tensor,
        region_token_mask: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """InfoNCE per hand: + = contacting region; hard − = same-hand no-contact.

        Region index layout is ``hand * num_regions_per_hand + local``. Hard
        negatives stay inside the same hand so the model cannot shortcut to
        ``hand vs background`` (§2.5).
        """
        B = projected.shape[0]
        R = self.num_regions
        R_hand = self.num_regions_per_hand
        contact = region_contact.to(device=projected.device, dtype=torch.float32)
        if contact.shape[-1] != R:
            raise ValueError(
                f"region_contact last dim {contact.shape[-1]} != num_regions={R}"
            )
        pooled, valid = self._masked_pool(projected, region_token_mask)
        losses = []
        for b in range(B):
            if weight[b] <= 0:
                continue
            hand_losses = []
            for hand in range(self.num_hands):
                sl = slice(hand * R_hand, (hand + 1) * R_hand)
                c = contact[b, sl] > 0.5
                v = valid[b, sl]
                pos = c & v
                neg = (~c) & v
                if not bool(pos.any()) or not bool(neg.any()):
                    continue
                q = pooled[b, sl][pos]  # [Qp, D]
                keys = pooled[b, sl][v]  # [K, D]
                key_is_pos = c[v]
                logits = (q @ keys.T) / self.temperature
                pos_mask = key_is_pos.unsqueeze(0).expand(q.shape[0], -1)
                neg_inf = torch.finfo(logits.dtype).min
                pos_logits = logits.masked_fill(~pos_mask, neg_inf)
                l_pos = torch.logsumexp(pos_logits, dim=-1)
                l_all = torch.logsumexp(logits, dim=-1)
                hand_losses.append(-(l_pos - l_all).mean())
            if hand_losses:
                losses.append(torch.stack(hand_losses).mean() * weight[b])
            else:
                losses.append(pooled[b].sum() * 0)
        if not losses:
            return projected.float().sum() * 0
        return torch.stack(losses).sum() / weight.sum().clamp_min(1e-6)

    def _pressure_bin_loss(
        self,
        projected: torch.Tensor,
        region_contact: torch.Tensor,
        region_bin: torch.Tensor,
        region_token_mask: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        pooled, valid = self._masked_pool(projected, region_token_mask)
        logits = self.bin_head(pooled.float())  # [B, R, n_bins]
        target = region_bin.to(device=projected.device, dtype=torch.long)
        contact = region_contact.to(device=projected.device, dtype=torch.float32) > 0.5
        # Supervise contacting regions (bin encodes force); non-contact → bin 0
        # is already set by target builders — include them so "none" is learned.
        mask = valid & (weight.view(-1, 1) > 0)
        if not bool(mask.any()):
            return logits.sum() * 0
        loss = F.cross_entropy(
            logits[mask], target[mask], reduction="mean"
        )
        # Keep contact in the graph so empty-contact batches still differ
        # from a pure constant (harmless).
        return loss + contact.float().sum() * 0

    def heatmap_to_contact_map(
        self,
        heatmap: torch.Tensor,
        student_grid: Tuple[int, int],
        *,
        obs_hw: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """HxW heatmap → token-grid soft labels (uses module heatmap knobs)."""
        return prepare_heatmap_contact_map(
            heatmap,
            student_grid,
            obs_hw=obs_hw if obs_hw is not None else self.heatmap_obs_hw,
            normalize=self.heatmap_normalize,
        )

    def build_targets_from_examples(
        self,
        examples: Sequence[dict],
        *,
        student_grid: Tuple[int, int],
        device: torch.device,
        obs_hw: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """Pack dataloader fields into tensors for ``forward``.

        Recognized per-example keys (any subset):
          * ``present2d`` (bool) — master skip flag
          * ``align_weight`` (float)
          * ``tactile_align_targets`` (dict of tensors, already on region/grid)
          * ``tactile_heatmap`` / ``tactile_2d`` — dense HxW heatmap → contact_map
          * ``heatmap_obs_hw`` / ``obs_hw`` — optional per-row ``[H,W]`` for
            bilinear resize onto the VLM observation plane (falls back to
            ``framework.tactile_align.heatmap.obs_hw``)

        Missing alignable signal ⇒ ``present2d=False`` for that row.

        When the sparse contact gate is on (``skip.min_contact_frac > 0``),
        rows carrying a raw heatmap are judged BEFORE targets are built;
        rejected rows behave as "no alignable signal". The packed dict carries
        ``sparse_keep = (kept, total)`` over heatmap rows for the
        ``contact_align_sparse_keep_frac`` metric.
        """
        B = len(examples)
        P = student_grid[0] * student_grid[1]
        present = []
        weights = []
        cmaps = []
        region_contacts: List[Optional[torch.Tensor]] = []
        region_bins: List[Optional[torch.Tensor]] = []
        region_masks: List[Optional[torch.Tensor]] = []
        any_region = False
        default_obs = obs_hw if obs_hw is not None else self.heatmap_obs_hw
        gate_on = self.min_contact_frac > 0.0
        sparse_kept = 0
        sparse_total = 0

        for ex in examples:
            p = bool(ex.get("present2d", False))
            w = float(ex.get("align_weight", 1.0 if p else 0.0))
            targets = ex.get("tactile_align_targets")
            # Primary: HxW heatmap (aliases for LeRobot / offline tools).
            heatmap = ex.get("tactile_heatmap", ex.get("tactile_2d"))

            cmap = None
            rc = rb = rm = None
            if isinstance(targets, dict):
                if "contact_map" in targets:
                    cmap = torch.as_tensor(targets["contact_map"], dtype=torch.float32)
                if "region_contact" in targets and "region_token_mask" in targets:
                    any_region = True
                    rc = torch.as_tensor(targets["region_contact"], dtype=torch.float32)
                    rb = torch.as_tensor(
                        targets.get(
                            "region_bin",
                            torch.zeros(self.num_regions, dtype=torch.long),
                        ),
                        dtype=torch.long,
                    )
                    rm = torch.as_tensor(targets["region_token_mask"], dtype=torch.bool)
            if cmap is None and heatmap is not None:
                row_obs = ex.get("heatmap_obs_hw", ex.get("obs_hw", default_obs))
                if row_obs is not None:
                    row_obs = (int(row_obs[0]), int(row_obs[1]))
                gate_rejected = False
                if gate_on:
                    # Judge the raw heatmap (physical / log1p units), never
                    # the max-normalized map. Precomputed contact_map targets
                    # bypass the gate — there is no raw heatmap to judge.
                    sparse_total += 1
                    if self._sparse_gate_keep(heatmap):
                        sparse_kept += 1
                    else:
                        gate_rejected = True
                        p = False
                        w = 0.0
                # present2d defaults True when a heatmap is provided (unless
                # the sparse gate already rejected this row).
                if not p and not gate_rejected and ex.get("present2d") is None:
                    p = True
                    w = float(ex.get("align_weight", 1.0))
                if p:
                    built = targets_from_saliency(
                        torch.as_tensor(heatmap, dtype=torch.float32),
                        student_grid,
                        obs_hw=row_obs,
                        normalize=self.heatmap_normalize,
                    )
                    cmap = built["contact_map"]

            if cmap is None and rc is None:
                p = False
                w = 0.0
            if cmap is None:
                cmap = torch.zeros(P, dtype=torch.float32)
            present.append(p)
            weights.append(w)
            cmaps.append(cmap)
            region_contacts.append(rc)
            region_bins.append(rb)
            region_masks.append(rm)

        out: Dict[str, Any] = {
            "present2d": torch.tensor(present, dtype=torch.bool, device=device),
            "align_weight": torch.tensor(weights, dtype=torch.float32, device=device),
            "contact_map": torch.stack(cmaps, dim=0).to(device=device),
            "sparse_keep": (sparse_kept, sparse_total),
        }
        if any_region:
            filled_c, filled_b, filled_m = [], [], []
            for rc, rb, rm in zip(region_contacts, region_bins, region_masks):
                if rc is None:
                    rc = torch.zeros(self.num_regions, dtype=torch.float32)
                    rb = torch.zeros(self.num_regions, dtype=torch.long)
                    rm = torch.zeros(self.num_regions, P, dtype=torch.bool)
                filled_c.append(rc)
                filled_b.append(rb)
                filled_m.append(rm)
            out["region_contact"] = torch.stack(filled_c, dim=0).to(device=device)
            out["region_bin"] = torch.stack(filled_b, dim=0).to(device=device)
            out["region_token_mask"] = torch.stack(filled_m, dim=0).to(device=device)
        return out


# Re-export helper for callers that already have grid_thw rows.
__all__ = [
    "ContactRegionAlignment",
    "get_tactile_align",
    "student_grid_hw",
    "prepare_heatmap_contact_map",
]
