# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
# Local Spark-0 addition (NOT upstream starVLA).
"""
BFA++-style hierarchical visual-token pruning for QwenPI frameworks.

What this implements
────────────────────
A post-train token-budget knob inspired by BFA++ (arXiv:2602.20566): score
every visual token of the VLM, then keep only the most useful ones before the
layer-wise action DiT cross-attends to them. Unlike the paper's
prune-before-LLM design, this implementation prunes the VLM's *output* hidden
states (after the full Qwen3-VL forward), so it composes cleanly with the
existing stack:

  - works with flash-attention and KV-cache-free training (no mid-LLM hook),
  - leaves DA3 spatial alignment untouched (it supervises the full token set),
  - is bit-identical to the baseline when disabled (no extra parameters).

Pruning is hierarchical, matching the paper's two-level idea:

  1. **Inter-view** (`view_alphas`): a *relative* per-camera budget share that
     splits the global token budget across views. Cameras absent under
     ``flexible_views`` simply contribute no tokens.
  2. **Intra-view** (global ``beta``): the fraction of ALL visual tokens that
     survive; within each view's share the scorer's ranking decides which
     patches are dropped first.

Budget arithmetic (see ``_view_keep_counts``)::

    view_alphas = null  ->  every view keeps  beta * n_tokens(view)
    view_alphas = [..]  ->  view v keeps      beta * n_tokens(total) * a_v / sum(a)

so ``beta`` is always the *overall* keep ratio and ``view_alphas`` only shifts
budget between cameras. A ``min_keep_ratio`` floor guarantees no view is ever
pruned to zero (the paper's "禁止剪空" constraint) and can push the realised
ratio above ``beta``.

Scorer
──────
``BFAPP`` is a lightweight MLP (LayerNorm -> Linear -> GELU -> Linear -> 1)
over the (projected) VLM hidden states of one layer. It is trained
*implicitly* by the action flow-matching loss. Top-k selection is not
differentiable, so the gradient reaches the scorer through a
straight-through gate (``apply_score_gate``): every kept visual token is
multiplied by ``1 + (sigmoid(score) - sigmoid(score).detach())``, which is
exactly ``1.0`` in the forward pass (bit-identical activations) but routes
``dL/dtoken · token`` into the score. That product is the first-order Taylor
estimate of how much the loss would rise if the token were removed
(Molchanov's pruning criterion), so gradient descent raises the score of
tokens the action head cannot afford to lose. There is no separate BCE
annotation stage in this MVP (that is the P2 roadmap item).

Known limitation: dropped tokens receive no gradient, so the scorer only
refines the ranking among tokens it already keeps. Offline inter/intra
annotation (P2) is what makes the ranking globally correct.

Ranking scale
─────────────
Raw scores start near zero (``fc2`` is initialised at std 1e-3) and grow only
as fast as the action loss pushes them, so comparing them against an ABSOLUTE
noise scale makes the exploration/exploitation balance depend on an unknown
quantity — with the original 0.5 Gumbel noise the ranking was ~99% noise for
most of a run, i.e. random pruning. ``ranking_scores`` therefore standardises
each sample's visual scores to zero mean / unit variance before adding noise.
Standardisation is monotonic per sample, so it never changes a noise-free
top-k; what it buys is that ``explore_noise`` is now a *signal-to-noise
ratio* (0.3 => noise std 0.38 against signal std 1.0) instead of an absolute
number, and it stays meaningful from step 0 to convergence.

Config (all keys live under `framework.token_prune` in the train YAML)
────────────────────────────────────────────────────────────────────────────
  enable: false               # master switch; false == baseline behaviour
  score_layer_index: 25       # VLM hidden_states index whose tokens are
                              # scored (same convention as DA3 align_layer_index;
                              # negative counts from the end). Under
                              # interleave_self_attention pick an ODD layer.
  view_alphas: null           # relative per-view budget share, packed-view
                              # order (e.g. [0.3, 0.2, 0.2]); null -> every
                              # view keeps the same fraction (beta)
  beta: 0.7                   # overall visual-token keep fraction (0, 1]
  min_keep_ratio: 0.25        # never keep fewer than this fraction of a view
  explore_noise: 0.3          # Gumbel noise / standardised-score ratio (train only)
  explore_noise_final: 0.0    # noise at the end of the anneal
  explore_noise_decay_steps: 0.7   # linear anneal length. A FLOAT in (0, 1]
                              # is a fraction of the run's total forward calls
                              # (max_train_steps x grad-accum, resolved by the
                              # framework); an INT > 1 is an absolute forward
                              # count; null -> constant explore_noise
  detach_scorer_input: true   # cut the scorer's gradient path into the VLM
  hidden_dim: null            # scorer MLP hidden width; default in_dim//4
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


class BFAPP(nn.Module):
    """Lightweight per-token scorer for BFA++-style pruning.

    Only this MLP holds trainable parameters and enters the checkpoint. The
    pruning itself is index bookkeeping on the VLM hidden states, so disabling
    the module reproduces the baseline model bit-for-bit.

    ``in_dim`` is the width of the tensor the scorer actually reads. The
    framework scores the PROJECTED layer-wise states, i.e.
    ``action_dit_hidden_dim`` — not ``vl_hidden_dim``, which differs whenever
    ``project_layers`` is not an Identity.
    """

    def __init__(
        self, in_dim: int, tp_cfg, total_training_forwards: Optional[int] = None
    ) -> None:
        """``total_training_forwards`` is the run's forward-call budget
        (max_train_steps x gradient_accumulation_steps); it is only required
        when ``explore_noise_decay_steps`` is given as a fraction of the run.
        """
        super().__init__()
        self.score_layer_index = int(tp_cfg.get("score_layer_index", 25))
        self.beta = float(tp_cfg.get("beta", 0.7))
        self.min_keep_ratio = float(tp_cfg.get("min_keep_ratio", 0.25))
        if not (0.0 < self.beta <= 1.0):
            raise ValueError(f"token_prune.beta must be in (0, 1], got {self.beta}.")
        if not (0.0 < self.min_keep_ratio <= 1.0):
            raise ValueError(
                f"token_prune.min_keep_ratio must be in (0, 1], got {self.min_keep_ratio}."
            )

        # Top-k gives gradient only to tokens it already keeps, so without
        # exploration the initial ranking would freeze permanently. Additive
        # Gumbel noise (TRAINING ONLY) lets near-threshold tokens re-enter the
        # kept set and earn a gradient. 0.0 disables it. The scale is relative
        # to STANDARDISED scores (see ranking_scores), so it reads as a
        # noise-to-signal ratio rather than an absolute score offset.
        self.explore_noise = float(tp_cfg.get("explore_noise", 0.3))
        self.explore_noise_final = float(tp_cfg.get("explore_noise_final", 0.0))
        for name, value in (
            ("explore_noise", self.explore_noise),
            ("explore_noise_final", self.explore_noise_final),
        ):
            if value < 0.0:
                raise ValueError(f"token_prune.{name} must be >= 0, got {value}.")
        # Inference is deterministic top-k, so leaving the noise on until the
        # last step trains a ranking the deployed model never uses. Annealing
        # to explore_noise_final closes that train/test gap.
        decay_steps = tp_cfg.get("explore_noise_decay_steps", None)
        if isinstance(decay_steps, float) and 0.0 < decay_steps <= 1.0:
            # Fraction-of-run form: an absolute count hardcoded in a template
            # silently stops annealing (or never finishes) whenever the run
            # length changes; a fraction tracks max_train_steps automatically.
            if not total_training_forwards or total_training_forwards <= 0:
                raise ValueError(
                    f"token_prune.explore_noise_decay_steps={decay_steps} is a fraction "
                    "of the run, which requires trainer.max_train_steps to be set "
                    "(the framework passes max_train_steps x grad-accum here)."
                )
            self.explore_noise_decay_steps = int(
                math.ceil(decay_steps * total_training_forwards)
            )
        else:
            self.explore_noise_decay_steps = int(decay_steps) if decay_steps else 0
        if self.explore_noise_decay_steps < 0:
            raise ValueError(
                "token_prune.explore_noise_decay_steps must be >= 0, got "
                f"{self.explore_noise_decay_steps}."
            )
        # Counts FORWARD CALLS on this rank (micro-batches, not optimizer
        # steps): with gradient accumulation the anneal runs
        # gradient_accumulation_steps times faster than the step counter.
        self.register_buffer("explore_step", torch.zeros((), dtype=torch.long))
        # Host mirror so the schedule never has to sync the device per step.
        self._explore_step_host = 0

        # The scorer reads a VLM hidden state, so the straight-through gate
        # also back-propagates into the VLM ("make the tokens you want kept
        # score high"). That is a second, uncontrolled change on top of token
        # selection; detaching keeps the VLM's gradient identical to a pure
        # selection ablation while the scorer still trains normally.
        self.detach_scorer_input = bool(tp_cfg.get("detach_scorer_input", True))

        view_alphas = tp_cfg.get("view_alphas", None)
        if view_alphas is None:
            self.view_alphas: Optional[List[float]] = None
        else:
            self.view_alphas = [float(a) for a in view_alphas]
            if not self.view_alphas or any(a <= 0 for a in self.view_alphas):
                raise ValueError(
                    f"token_prune.view_alphas must be a non-empty list of positive "
                    f"fractions, got {view_alphas!r}."
                )

        hidden_dim = tp_cfg.get("hidden_dim", None) or max(64, in_dim // 4)
        self.in_dim = int(in_dim)
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, int(hidden_dim))
        self.fc2 = nn.Linear(int(hidden_dim), 1)
        # Near-zero output init: scores start at ~0 so the exploration noise
        # (not an arbitrary random projection) decides the early ranking, and
        # the gate starts at sigmoid(0)=0.5 where its derivative is largest.
        nn.init.normal_(self.fc2.weight, std=1e-3)
        nn.init.zeros_(self.fc2.bias)

    # ──────────────────────────────────────────────────────────────────
    #  Scoring
    # ──────────────────────────────────────────────────────────────────

    def score(self, tokens: torch.Tensor) -> torch.Tensor:
        """`[B, L, C]` VLM tokens -> per-token scores `[B, L]` (higher = keep)."""
        if tokens.shape[-1] != self.in_dim:
            raise ValueError(
                f"token_prune scorer was built for width {self.in_dim} but got "
                f"{tokens.shape[-1]}. The scorer reads the PROJECTED layer-wise "
                "states, so it must be constructed with action_dit_hidden_dim."
            )
        dtype = self.fc1.weight.dtype
        hidden = torch.nn.functional.gelu(self.fc1(self.norm(tokens.to(dtype))))
        return self.fc2(hidden).squeeze(-1)

    def score_tokens(self, hidden: torch.Tensor) -> torch.Tensor:
        """``score`` with the configured gradient policy toward the VLM.

        With ``detach_scorer_input`` (the default) the scorer is a read-only
        probe of the VLM: it still trains, but the backbone's gradient is
        exactly the one it would get from an equivalent hard token selection,
        which is what makes an A/B against the baseline interpretable.
        """
        return self.score(hidden.detach() if self.detach_scorer_input else hidden)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        # Keep the host mirror in step with the checkpointed buffer so a
        # resumed run continues the anneal instead of restarting it.
        key = prefix + "explore_step"
        if key in state_dict:
            self._explore_step_host = int(state_dict[key].reshape(-1)[0])

    def current_explore_noise(self) -> float:
        """Noise scale for this step under the linear anneal."""
        if self.explore_noise_decay_steps <= 0:
            return self.explore_noise
        progress = min(1.0, self._explore_step_host / self.explore_noise_decay_steps)
        return self.explore_noise + (self.explore_noise_final - self.explore_noise) * progress

    def _advance_explore_if_training(self) -> None:
        """One anneal tick per keep-mask build, not per layout group."""
        if not self.training or not torch.is_grad_enabled():
            return
        self._explore_step_host += 1
        self.explore_step.add_(1)

    def ranking_scores(
        self, scores: torch.Tensor, *, advance: bool = True
    ) -> torch.Tensor:
        """Scores used for top-k, with training-time exploration noise.

        ``scores`` is the `[B, n_visual]` block of VISUAL scores only — text
        positions never compete for a budget, and including them would skew
        the standardisation.

        Always float32: with bf16 parameters the raw scores (and a bf16
        ``rand_like``) resolve to ~256 distinct values in [0, 1), which would
        make both the noise and the top-k tie-breaking essentially arbitrary.

        ``advance=False`` applies the current noise scale without ticking
        the anneal — mixed-layout groups share one tick per forward.
        """
        ranking = scores.float()
        # ``is_grad_enabled`` as well as ``training``: predict_action runs under
        # inference_mode, so a deployed model left in train() mode must not get
        # stochastic pruning.
        if not self.training or not torch.is_grad_enabled():
            return ranking
        if advance:
            self._advance_explore_if_training()
        noise_scale = self.current_explore_noise()
        if noise_scale <= 0.0:
            return ranking
        # Standardise per sample so `explore_noise` is a noise-to-SIGNAL ratio.
        # Monotonic per row, so it cannot change the noise-free ranking.
        # (A single-token row has no unbiased std — nothing to rank either.)
        if ranking.shape[1] > 1:
            mean = ranking.mean(dim=1, keepdim=True)
            std = ranking.std(dim=1, keepdim=True).clamp_min(1e-6)
            ranking = (ranking - mean) / std
        # Gumbel(0,1) noise: with a softmax view this is exactly Gumbel-top-k
        # sampling from the score distribution, i.e. unbiased exploration
        # rather than a symmetric jitter.
        uniform = torch.rand_like(ranking).clamp_(min=1e-9, max=1.0 - 1e-7)
        gumbel = -torch.log(-torch.log(uniform))
        return ranking + noise_scale * gumbel

    def score_gate_factor(
        self, scores: torch.Tensor, gate_mask: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Straight-through gate so the action loss can train the scorer.

        ``torch.topk`` returns indices, which carry no gradient — selecting
        tokens by index alone leaves the scorer's parameters with ``grad=None``
        forever (and, under DDP, in the unused-parameter set). Multiplying the
        kept visual tokens by ``1 + (g - g.detach())`` with ``g = sigmoid(s)``
        is an exact no-op numerically (the factor is literally ``1.0``) while
        back-propagating ``dL/dtoken · token · sigmoid'(s)`` into the score:
        tokens the action head wants amplified rank higher next step.

        Returns the `[B, L]` factor, or None when no gradient is being
        recorded (inference), where the gate is pure overhead.
        """
        if not torch.is_grad_enabled() or not scores.requires_grad:
            return None
        gate = torch.sigmoid(scores)
        straight_through = 1.0 + (gate - gate.detach())
        return torch.where(gate_mask, straight_through, torch.ones_like(straight_through))

    def apply_score_gate(
        self, hidden: torch.Tensor, scores: torch.Tensor, gate_mask: torch.Tensor
    ) -> torch.Tensor:
        """``score_gate_factor`` applied to a full-length `[B, L, C]` sequence.

        The pruning path folds the factor into the gather instead (one pass
        over the SHORTER sequence); this stays for callers that gate without
        pruning.
        """
        factor = self.score_gate_factor(scores, gate_mask)
        if factor is None:
            return hidden
        return hidden * factor.unsqueeze(-1).to(hidden.dtype)

    # ──────────────────────────────────────────────────────────────────
    #  Pruning geometry
    # ──────────────────────────────────────────────────────────────────

    def resolve_score_layer(self, num_vl_layers: int) -> int:
        """Resolve `score_layer_index` (1-based, negative = from end) to a
        valid hidden_states index, mirroring DA3's align_layer_index rules."""
        requested = self.score_layer_index
        resolved = requested + num_vl_layers + 1 if requested < 0 else requested
        if not (1 <= resolved <= num_vl_layers):
            raise ValueError(
                f"token_prune.score_layer_index={requested} resolves to layer "
                f"{resolved}, outside [1, {num_vl_layers}] for this VLM."
            )
        return resolved

    def _view_keep_counts(self, tokens_per_view: Sequence[int]) -> List[int]:
        """Per-view keep counts: ``beta`` is the OVERALL keep ratio,
        ``view_alphas`` only redistributes that budget between cameras.

        ``view_alphas=null``      -> view v keeps ``beta * n_v``.
        ``view_alphas=[a_0, ...]`` -> view v gets ``beta * sum(n) * a_v/sum(a)``,
        water-filled: a share larger than the view itself is capped at ``n_v``
        and the leftover is redistributed over the remaining views by their
        alphas. Without that redistribution a lopsided alpha silently spent the
        budget on a view too small to hold it (``beta=0.9``, ``alphas=[9, 1]``,
        ``n=[4, 16]`` kept 6/20 tokens instead of 18/20).

        Both forms are then clamped into
        ``[ceil(min_keep_ratio * n_v), n_v]``, so a small view is never emptied
        (which can push the realised ratio above ``beta``).
        """
        n_views = len(tokens_per_view)
        n_total = sum(tokens_per_view)
        if self.view_alphas is None:
            targets = [self.beta * n_tokens for n_tokens in tokens_per_view]
        else:
            # Packed-index prefix under flexible_views: a 3-cam alpha list
            # still budgets a head-only group with a_0 / sum(a_0). Extra
            # trailing cameras are absent, not missing shares. Fewer alphas
            # than packed views cannot invent cameras.
            n_alphas = len(self.view_alphas)
            if n_alphas < n_views:
                raise ValueError(
                    f"token_prune.view_alphas has {n_alphas} entries but the "
                    f"batch packs {n_views} view(s). Under flexible_views, "
                    "view_alphas is packed-view order (or leave it null)."
                )
            alphas = self.view_alphas[:n_views]
            targets = [0.0] * n_views
            budget = self.beta * n_total
            pending = list(range(n_views))
            while pending and budget > 0:
                total_alpha = sum(alphas[i] for i in pending)
                shares = {i: budget * alphas[i] / total_alpha for i in pending}
                overflow = [i for i in pending if shares[i] >= tokens_per_view[i]]
                if not overflow:
                    for i in pending:
                        targets[i] = shares[i]
                    break
                for i in overflow:
                    targets[i] = float(tokens_per_view[i])
                    budget -= tokens_per_view[i]
                    pending.remove(i)

        keep = []
        for target, n_tokens in zip(targets, tokens_per_view):
            floor = min(n_tokens, max(1, int(math.ceil(n_tokens * self.min_keep_ratio))))
            keep.append(min(n_tokens, max(floor, int(math.ceil(target)))))
        return keep

    def _topk_visual_keep(
        self,
        ranking: torch.Tensor,
        tokens_per_view: Sequence[int],
        keep_counts: Sequence[int],
    ) -> torch.Tensor:
        """`[B, n_visual]` ranking -> `[B, n_visual]` bool keep mask.

        Batched per-view top-k. Qwen lays each image's merged tokens out in
        raster order and images in message order, so the visual positions of
        every sample split by `tokens_per_view` recover one block per camera
        (the same assumption DA3 makes). Ranking them with one topk per VIEW
        instead of one per (sample, view) keeps the batch loop — and its
        per-sample `nonzero` device sync — out of the training step.
        """
        keep = torch.zeros_like(ranking, dtype=torch.bool)
        offset = 0
        for n_tokens, n_keep in zip(tokens_per_view, keep_counts):
            view_slice = slice(offset, offset + n_tokens)
            top = ranking[:, view_slice].topk(min(n_keep, n_tokens), dim=1).indices
            keep[:, view_slice].scatter_(1, top, True)
            offset += n_tokens
        return keep

    @staticmethod
    def _normalize_tokens_per_view(
        tokens_per_view: Sequence,
        batch_size: int,
    ) -> List[List[int]]:
        """Uniform ``[n0, n1, ...]`` or per-sample ``[[...], ...]`` → B lists."""
        if not tokens_per_view:
            raise ValueError("token_prune tokens_per_view is empty.")
        first = tokens_per_view[0]
        if isinstance(first, (list, tuple)):
            if len(tokens_per_view) != batch_size:
                raise ValueError(
                    "per-sample tokens_per_view has "
                    f"{len(tokens_per_view)} row(s) but the batch is {batch_size}."
                )
            return [list(int(n) for n in row) for row in tokens_per_view]
        uniform = [int(n) for n in tokens_per_view]
        return [list(uniform) for _ in range(batch_size)]

    def build_keep_mask(
        self,
        scores: torch.Tensor,
        visual_mask: torch.Tensor,
        tokens_per_view: Sequence,
        attention_mask: Optional[torch.Tensor] = None,
        collect_stats: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Top-k keep mask over visual tokens.

        Args:
            scores: `[B, L]` per-token scores (only visual positions are used).
            visual_mask: `[B, L]` bool, True at visual-token positions.
            tokens_per_view: per-view token counts. A single list is shared
                by every sample; a list-of-lists is per sample (mixed
                ``flexible_views`` layouts). Each row must sum to that
                sample's visual-token count.
            attention_mask: `[B, L]` padding mask from the Qwen processor.
                Padding positions are DROPPED, not kept: the Qwen tokenizer
                pads on the left, so treating pads as "text" would re-pack them
                into the DiT's encoder sequence and mark them attendable — the
                baseline masks them out.
            collect_stats: also return the scorer diagnostics (one extra top-k
                over the noise-free ranking). Without them there is no way to
                tell a learned ranking from random pruning.

        Returns:
            (`[B, L]` bool keep mask — True for real non-visual positions and
            for the top-scoring visual tokens of each view — , stats or None).
        """
        batch_size = scores.shape[0]
        per_sample = self._normalize_tokens_per_view(tokens_per_view, batch_size)

        keep = ~visual_mask  # real text / special tokens always survive
        if attention_mask is not None:
            keep = keep & attention_mask.to(dtype=torch.bool, device=keep.device)

        # One anneal tick for the whole batch, even when layouts split.
        if self.training and torch.is_grad_enabled():
            self._advance_explore_if_training()

        groups: dict = {}
        order: List[tuple] = []
        for row, tpv in enumerate(per_sample):
            key = tuple(tpv)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(row)

        score_chunks: List[torch.Tensor] = []
        agree = None
        kept = None
        if collect_stats:
            agree = torch.zeros((), device=scores.device, dtype=torch.float32)
            kept = torch.zeros((), device=scores.device, dtype=torch.float32)

        for key in order:
            rows = groups[key]
            tpv = list(key)
            n_visual = sum(tpv)
            idx = torch.tensor(rows, dtype=torch.long, device=scores.device)
            vm = visual_mask.index_select(0, idx)
            sc = scores.index_select(0, idx)
            counts = vm.sum(dim=1)
            if not bool(torch.all(counts == n_visual)):
                raise ValueError(
                    f"token_prune expected {n_visual} visual tokens per sample "
                    f"({tpv} per view), found {counts.tolist()} "
                    f"(batch rows {rows})."
                )
            visual_scores = sc[vm].view(len(rows), n_visual)
            visual_keep = self._topk_visual_keep(
                self.ranking_scores(visual_scores, advance=False),
                tpv,
                self._view_keep_counts(tpv),
            )
            visual_positions = vm.nonzero(as_tuple=True)[1].view(len(rows), n_visual)
            keep_rows = keep.index_select(0, idx)
            keep_rows.scatter_(1, visual_positions, visual_keep)
            keep.index_copy_(0, idx, keep_rows)
            score_chunks.append(visual_scores.detach().float().reshape(-1))
            if collect_stats:
                with torch.no_grad():
                    clean = self._topk_visual_keep(
                        visual_scores.detach().float(),
                        tpv,
                        self._view_keep_counts(tpv),
                    )
                    agree = agree + (visual_keep & clean).sum().float()
                    kept = kept + visual_keep.sum().float()

        stats = None
        if collect_stats:
            all_scores = (
                torch.cat(score_chunks, dim=0)
                if score_chunks
                else torch.zeros((), device=scores.device)
            )
            stats = {
                "score_std": all_scores.std()
                if all_scores.numel() > 1
                else torch.zeros((), device=scores.device),
                "score_mean": all_scores.mean()
                if all_scores.numel()
                else torch.zeros((), device=scores.device),
                "topk_agreement": agree / kept.clamp(min=1.0),
            }
        return keep, stats

    @staticmethod
    def build_gather_index(keep_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """`[B, L]` keep mask -> (`[B, L']` gather index, `[B, L']` bool mask).

        Computed ONCE per forward and reused for every DiT layer: a per-layer
        boolean index would launch `B x num_layers` variable-size kernels and
        sync the device on each one, which costs more than the cross-attention
        the pruning saves.

        The index is a stable argsort of ``~keep_mask``, so kept positions come
        first in their original order and the trailing slots (padding) point at
        arbitrary dropped positions — masked out by the returned mask and
        zeroed by ``prune_with_index``.
        """
        kept_lengths = keep_mask.sum(dim=1)
        max_len = int(kept_lengths.max().item()) if keep_mask.numel() else 0
        # int32 rather than bool: sort kernels are not implemented for bool.
        order = torch.argsort((~keep_mask).to(torch.int32), dim=1, stable=True)
        index = order[:, :max_len]
        positions = torch.arange(max_len, device=keep_mask.device)
        # Always bool: the DiT treats 1/True = attend, 0/False = mask; an
        # int-typed mask would silently zero the padded rows on the training
        # path (which never casts to bool), masking real tokens whenever
        # samples keep unequal counts.
        new_attn = positions.unsqueeze(0) < kept_lengths.unsqueeze(1)
        return index, new_attn

    @staticmethod
    def gather_scale(
        index: torch.Tensor,
        new_attn: torch.Tensor,
        gate_factor: Optional[torch.Tensor],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """`[B, L', 1]` scale for the gathered sequence: padding-zeroing and
        the straight-through gate folded into ONE multiply, computed once and
        reused by every layer."""
        scale = new_attn.to(dtype)
        if gate_factor is not None:
            scale = scale * torch.gather(gate_factor, 1, index).to(dtype)
        return scale.unsqueeze(-1)

    @staticmethod
    def prune_with_index(
        hidden: torch.Tensor,
        index: torch.Tensor,
        new_attn: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Left-pack a `[B, L, C]` sequence with a precomputed gather index.

        ``scale`` (from ``gather_scale``) is applied in the same pass; without
        it the padded rows are simply zeroed.
        """
        channels = hidden.shape[-1]
        gathered = torch.gather(hidden, 1, index.unsqueeze(-1).expand(-1, -1, channels))
        if scale is None:
            scale = new_attn.unsqueeze(-1).to(hidden.dtype)
        return gathered * scale

    def prune_sequence(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        keep_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Drop pruned positions from a `[B, L, C]` sequence (left-packed).

        Samples may keep different numbers of tokens, so the result is
        re-padded to the batch max length and the attention mask is rebuilt.
        A mask is ALWAYS returned (even when the caller had none), because
        unequal keep counts make one mandatory after pruning.
        """
        index, new_attn = self.build_gather_index(keep_mask)
        return self.prune_with_index(hidden, index, new_attn), new_attn

    # ──────────────────────────────────────────────────────────────────
    #  Visual-token geometry (shared with DA3's layout derivation)
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def tokens_per_view_per_sample(
        image_grid_thw: torch.Tensor,
        n_views_per_sample: Sequence[int],
        merge_size: int,
    ) -> List[List[int]]:
        """Per-sample per-view merged-token counts (mixed layouts allowed)."""
        from starVLA.model.modules.spatial_alignment.layout_groups import (
            packed_sample_grids,
        )

        sample_grids = packed_sample_grids(
            n_views_per_sample, image_grid_thw, caller="token_prune"
        )
        out: List[List[int]] = []
        for sample_idx, grid in enumerate(sample_grids):
            tokens_per_view: List[int] = []
            for view_idx in range(int(grid.shape[0])):
                t, h, w = (int(v) for v in grid[view_idx])
                if t != 1:
                    raise ValueError(
                        f"token_prune does not support temporal image grids "
                        f"(sample {sample_idx} view {view_idx} has t={t})."
                    )
                tokens_per_view.append((h // merge_size) * (w // merge_size))
            out.append(tokens_per_view)
        return out

    @staticmethod
    def tokens_per_view_from_grid(
        image_grid_thw: torch.Tensor, batch_size: int, merge_size: int
    ) -> List[int]:
        """Per-view merged-token counts from a *uniform* `image_grid_thw`.

        Every sample must share one camera layout. Mixed packed counts /
        resolutions use ``tokens_per_view_per_sample`` instead. Temporal
        grids (t != 1) are rejected.
        """
        if image_grid_thw is None:
            raise ValueError("token_prune requires image inputs (image_grid_thw missing).")
        if image_grid_thw.shape[0] % batch_size != 0:
            raise ValueError(
                f"image count {image_grid_thw.shape[0]} is not a multiple of batch size {batch_size}."
            )
        views_per_sample = image_grid_thw.shape[0] // batch_size
        per_sample = BFAPP.tokens_per_view_per_sample(
            image_grid_thw,
            [views_per_sample] * int(batch_size),
            merge_size,
        )
        if any(row != per_sample[0] for row in per_sample):
            raise ValueError(
                f"token_prune expects every sample to share one camera layout, "
                f"got {per_sample}."
            )
        return per_sample[0]
