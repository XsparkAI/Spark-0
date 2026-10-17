# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by Jinhui YE / HKUST University] in [2025].
"""
QwenPI_v3 Framework
A Qwen2.5-VL / Qwen3-VL + layer-wise cross-DiT flow-matching action head.

Released checkpoint
─────────────────────────────
- Qwen3-VL-4B + Bridge V2 + RT-1 (OXE) co-training, 69.8% avg success on
  SimplerEnv WidowX:
  https://huggingface.co/StarVLA/Qwen3VL-PI_v3-Bridge-RT_1

Key improvements over QwenPI
─────────────────────────────
1. **Compressed Action DiT via per-layer projectors**
   Each of the N VLM hidden-state layers is passed through a dedicated
   LayerNorm + Linear projector (`project_layers`) that maps the VLM hidden
   dimension (e.g. 2560) down to a smaller Action DiT latent dimension
   (e.g. 1024, controlled by `action_dit_hidden_dim`).  This reduces the
   action head parameter count by ~(vl_hidden / dit_hidden)² while keeping
   the full layer-wise cross-attention structure.

2. **Discretised-state language injection** (`add_discretized_state_to_instruction`)
   Proprioceptive state is quantised into 256 bins and appended to the
   language instruction as plain tokens (``[STATE] <bins> [ACTION]``),
   following the π₀.5 design.  This lets the VLM attend to state without
   any extra encoder module.

Together these two features bring QwenPI_v3 close to all the core
capabilities of π₀.5 within a single open-weight VLM framework.

Parameter breakdown (Qwen3-VL-4B + action_dit_hidden_dim=1024)
═══════════════════════════════════════════════════════════════
  Module                               Params        %
  ───────────────────────────────────────────────────
  qwen_vl_interface         4,437,815,808   87.5%
  action_model                538,678,305   10.6%
  project_layers               94,593,024    1.9%
  ───────────────────────────────────────────────────
  TOTAL                     5,071,087,137  100.0%
═══════════════════════════════════════════════════════════════
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from starVLA.model.framework.action_contract import (
    assert_packed_action_state,
    assert_packed_states,
)
from PIL import Image

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config, populate_layerwise_dit_cfg
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import LayerwiseFlowmatchingActionHead, get_action_model
from starVLA.model.modules.spatial_alignment.layout_groups import (
    gather_student_views,
    group_samples_by_image_layout,
    normalized_segment_weights,
)
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resolve_predict_resize_targets, resize_images

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


def _resolve_rtc_weight_bounds(
    action_horizon: int,
    leftover_len: int,
    inference_delay: int,
    execution_horizon: Optional[int],
) -> Optional[Tuple[int, int]]:
    """RTC prefix-weight boundaries for one guided call.

    Returns ``(delay_eff, suffix_length)`` such that the head's weight
    schedule is 1 on ``[0, delay_eff)``, decays on ``[delay_eff, end)`` and is
    0 on ``[end, action_horizon)`` with ``end = action_horizon - suffix_length``.

    Paper convention (Kinetix eval loop): the leftover is the previous chunk
    shifted past its consumed rows and right-padded with zeros, and the
    weights hit zero EXACTLY where the real leftover data ends
    (``end = leftover_len``) — guidance never pulls toward padding.
    ``execution_horizon`` (lerobot naming) optionally caps how far into the
    leftover the model may attend. ``delay_eff`` is clamped to ``end`` so the
    frozen prefix never covers padding either.

    Returns ``None`` when there is nothing to guide with (empty leftover).
    """
    end = min(int(leftover_len), int(action_horizon))
    if execution_horizon is not None:
        end = min(end, int(execution_horizon))
    if end <= 0:
        return None
    delay_eff = min(int(inference_delay), end)
    suffix_length = int(action_horizon) - end
    return delay_eff, suffix_length


def validate_rtc_deploy_config(rtc_cfg) -> None:
    """Deployment-time gate for a ``framework.action_model.rtc`` block.

    Only ΠGDM is servable: it is training-free, so any checkpoint may flip
    ``rtc.enable`` at eval. ``simulated_delay`` requires a checkpoint trained
    with the simulated-delay loss, which this repo does not implement —
    serving it would silently degrade to naive prefix pinning instead of real
    RTC guidance. The schedule is validated here too: the action head only
    raises on an unknown schedule at the first guided call, i.e. mid-episode.
    ``execution_horizon`` is checked as well: with ``end <= 0`` every prefix
    weight is zero, so the guided call silently degrades to the plain path
    while the client keeps prefetching and splicing — the latency hiding still
    works but the boundary guidance is gone, with no error anywhere.
    ``inference_delay <= 0`` and a non-positive / non-finite
    ``max_guidance_weight`` are gated for the same reason: the first routes
    every guided call to the plain path, the second zeroes the ΠGDM step or —
    worse, for a negative cap — flips its sign, since
    ``gw = min(c * inv_r2, max_guidance_weight)`` with ``c * inv_r2 >= 0``
    then pushes the chunk AWAY from the leftover. Called by the policy server
    at startup (``PolicyServerWrapper``) so a bad block fails before the first
    request is served. No-op when RTC is off.
    """
    if not rtc_cfg or not bool(rtc_cfg.get("enable", False)):
        return
    mode = str(rtc_cfg.get("mode", "pigdm"))
    if mode != "pigdm":
        raise ValueError(
            f"framework.action_model.rtc.mode={mode!r} cannot be served: only 'pigdm' is "
            "training-free; 'simulated_delay' requires a checkpoint trained with the "
            "simulated-delay loss, which this repo does not implement. Serve with "
            "rtc.mode=pigdm or disable RTC."
        )
    schedule = str(rtc_cfg.get("prefix_attention_schedule", "exp"))
    if schedule not in ("exp", "linear", "ones", "zeros"):
        raise ValueError(
            f"framework.action_model.rtc.prefix_attention_schedule={schedule!r} must be one of "
            "exp | linear | ones | zeros."
        )
    execution_horizon = rtc_cfg.get("execution_horizon", None)
    if execution_horizon is not None:
        try:
            execution_horizon = int(execution_horizon)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"framework.action_model.rtc.execution_horizon={execution_horizon!r} "
                "must be a positive integer or null."
            ) from exc
        if execution_horizon < 1:
            raise ValueError(
                f"framework.action_model.rtc.execution_horizon={execution_horizon} must be a "
                "positive integer or null: end <= 0 gives every prefix row zero weight, "
                "silently disabling the guidance while the async client loop stays on."
            )
    inference_delay = rtc_cfg.get("inference_delay", None)
    if inference_delay is not None:
        try:
            inference_delay = int(inference_delay)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"framework.action_model.rtc.inference_delay={inference_delay!r} "
                "must be a positive integer or null."
            ) from exc
        if inference_delay < 1:
            raise ValueError(
                f"framework.action_model.rtc.inference_delay={inference_delay} must be a "
                "positive integer or null: delay <= 0 routes every guided call to the "
                "plain path, silently disabling the guidance while the async client "
                "loop stays on."
            )
    max_guidance_weight = rtc_cfg.get("max_guidance_weight", None)
    if max_guidance_weight is not None:
        try:
            max_guidance_weight = float(max_guidance_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"framework.action_model.rtc.max_guidance_weight={max_guidance_weight!r} "
                "must be a positive finite number or null."
            ) from exc
        if not math.isfinite(max_guidance_weight) or max_guidance_weight <= 0:
            raise ValueError(
                f"framework.action_model.rtc.max_guidance_weight={max_guidance_weight} must "
                "be a positive finite number or null: 0 disables the ΠGDM guidance and a "
                "negative value inverts it (the correction is pushed away from the "
                "leftover), both silently; an infinite cap lets the t=0 step explode."
            )

####################################################
# ⚠️ Warning: This framework has been restructured and is NOT compatible with checkpoints created before 2025-10-20.
####################################################


# ──────────────────────────────────────────────────────────────────────
#  Default Config for QwenPI_v3
#  - Same shape as QwenPIDefaultConfig (see QwenPI.py) but introduces the
#    optional `action_dit_hidden_dim` knob inside diffusion_model_cfg.
#  - Setting action_dit_hidden_dim to a value smaller than the VLM hidden
#    size lets the Action DiT run at a "compressed" latent dim while
#    `Qwen_PI_v3.project_layers` does the LayerNorm+Linear compression of
#    each VL hidden state to that dim.
#  - Leaving it None (or omitting it from YAML) reproduces the QwenPI
#    behaviour: DiT hidden = VLM hidden, projection becomes nn.Identity().
# ──────────────────────────────────────────────────────────────────────
@dataclass
class QwenPI_v3DefaultConfig:
    """QwenPI_v3 framework default parameters.

    See ``starVLA/model/framework/VLM4A/diffusion_model_cfg.md`` for the
    relationship between vl_hidden_dim, action_dit_hidden_dim and
    cross_attention_dim.
    """

    name: str = "QwenPI_v3"

    # === VLM backbone (Qwen2.5-VL / Qwen3-VL) ===
    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            "attn_implementation": "flash_attention_2",
            "vl_hidden_dim": 2048,  # auto-overridden at runtime from the loaded VLM
            "num_vl_layers": 36,  # auto-overridden at runtime from the loaded VLM
        }
    )

    # === Spatial alignment (OPTIONAL, Spark-0 local addition) ===
    # Spatial-Forcing-style distillation: align intermediate Qwen3-VL visual
    # tokens with frozen Depth-Anything-3 features during TRAINING only.
    # Default is disabled — the model then builds/behaves exactly like the
    # baseline (no extra parameters, no extra forward work), which keeps
    # controlled A/B experiments trivial. All tunable keys and their meaning
    # are documented in modules/spatial_alignment/da3_alignment.py; YAML only
    # needs to set the keys it wants to override.
    spatial_alignment: dict = field(default_factory=lambda: {"enable": False})

    # === Visual token pruning (OPTIONAL, Spark-0 local addition) ===
    # BFA++-style hierarchical pruning of the VLM's visual tokens before the
    # action DiT cross-attends to them. Default disabled == baseline
    # bit-for-bit. Keys documented in modules/token_prune/bfa_pp.py.
    token_prune: dict = field(default_factory=lambda: {"enable": False})

    # === Tactile fusion encoder (OPTIONAL, Spark-0 local addition) ===
    # Dual temporal streams (region_pressure + taxel_pressure_3d) → fixed
    # token sequence for later Action-DiT fusion. Default disabled ==
    # baseline bit-for-bit (no params). Recipes opt in with top-level
    # `tactile_fusion: <stem>` (null = off); train.sh merges the stem and
    # writes enable / action_fusion.enable. Contact-region visual alignment
    # is a separate stem (`tactile_align:` → framework.tactile_align).
    # Keys in modules/tactile/tactile_encoder.py.
    tactile: dict = field(default_factory=lambda: {"enable": False})

    # === Contact-region visual alignment (OPTIONAL, Spark-0 local) ===
    # Train-only attention regularizer: 2D contact → Qwen visual tokens.
    # Orthogonal to spatial_alignment (DA3) and tactile fusion. Default
    # disabled == baseline bit-for-bit. Stem: `tactile_align: <stem>`
    # (null = off; train.sh writes enable). Keys in contact_align.py.
    tactile_align: dict = field(default_factory=lambda: {"enable": False})

    # === VLM-side action queries (OPTIONAL, Spark-0 0821 plan change 2) ===
    # K learnable query embeddings appended at the END of the Qwen input
    # sequence (dummy token ids whose embeddings are spliced by a forward
    # hook — the vocab is NOT resized, so pre-action-query checkpoints still
    # load). Causal masking gives the V-L isolation for free: visual /
    # language tokens never see the queries, so DA3 / heatmap alignment on
    # visual positions is untouched, while the queries aggregate V+L semantics
    # through all 36 VLM layers. The layerwise DiT reads the queries'
    # per-layer hidden states instead of (query_only) or in addition to
    # (query_plus_visual) the raw V-L tokens; in query_only mode the action
    # loss reaches the VLM only through the query positions (no per-layer
    # direct pressure on visual slices). Default disabled == baseline
    # bit-for-bit. Keys: num_query (64), kv_mode (query_only |
    # query_plus_visual), init_std (0.02), dummy_token ("🔍").
    action_query: dict = field(default_factory=lambda: {"enable": False})

    # === Action head (Layer-wise Flow-matching / cross-DiT) ===
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "LayerwiseFM",
            "action_dim": 7,
            "state_dim": 7,
            # Feed the NORMALIZED continuous state to the action head's
            # state_encoder (in addition to the discretised [STATE] prompt
            # tokens). False reproduces the π0.5-style discrete-only behaviour
            # (state_encoder stays dormant), which every pre-existing
            # checkpoint was trained with.
            "continuous_state": False,
            # Canonical chunk length (number of action steps the head predicts).
            # Legacy YAMLs may use future_action_window_size = action_horizon - 1;
            # apply_config_compat normalises both directions.
            "action_horizon": 16,
            "repeated_diffusion_steps": 2,
            "num_inference_timesteps": 4,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "num_target_vision_tokens": 32,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            # Train-only shortcut mask (0821 plan §5): per-sample probability
            # of zeroing the action expert's cross-attention output, forcing
            # the VL read through the tactile joint stream. Requires
            # framework.tactile.action_fusion.mode: joint_stream. 0.0 = off.
            "shortcut_mask_prob": 0.0,
            # Optimizer steps over which shortcut_mask_prob ramps 0 -> target.
            # 0 = no ramp (constant p from step 0). A fresh run should ramp:
            # at step 0 the tactile encoder is random, so forcing the VL read
            # through it just regresses the action chunk against noise.
            "shortcut_mask_warmup_steps": 0,
            # Real-time chunking (RTC, arXiv:2506.07339) — INFERENCE-TIME only,
            # training is unaffected. enable=False (default) keeps the plain
            # synchronous predict_action path byte-identical. See
            # predict_action_realtime for the semantics of each knob.
            "rtc": {
                "enable": False,
                "inference_delay": 4,
                "execution_horizon": None,
                "prefix_attention_schedule": "exp",
                "max_guidance_weight": 10.0,
                "mode": "pigdm",
            },
            "diffusion_model_cfg": {
                # When set (e.g. 1024), DiT internal hidden = action_dit_hidden_dim
                # and Qwen_PI_v3.project_layers compress VL hidden to this dim.
                # When None, DiT internal hidden = vl_hidden_dim (== QwenPI behaviour).
                "action_dit_hidden_dim": 1024,
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "positional_embeddings": None,
                "attention_head_dim": 64,
            },
        }
    )


@FRAMEWORK_REGISTRY.register("QwenPI_v3")
class Qwen_PI_v3(baseframework):
    """
    Multimodal vision-language-action model (QwenPI_v3 variant).

    Architecture
    ────────────
    - Qwen2.5-VL / Qwen3-VL backbone for fused language / vision token embeddings.
    - Per-layer projectors (``project_layers``): one LayerNorm + Linear per VLM
      layer that compresses VLM hidden states from ``vl_hidden_dim`` down to
      ``action_dit_hidden_dim`` before feeding the Action DiT.
    - Layer-wise cross-DiT flow-matching action head that attends to every
      selected VLM layer in parallel.

    Focus: predict a future action chunk conditioned on multi-view images
    and a natural-language instruction (with optional discretised state prefix).
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """

        super().__init__()
        # Merge framework defaults with YAML config (YAML wins on conflicts).
        self.config = merge_framework_config(QwenPI_v3DefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # Read the actual hidden size and layer count from the loaded VLM.
        # `output_hidden_states=True` returns (num_hidden_layers + 1) tensors
        # (embedding output + every layer's output), and we keep the last
        # `num_hidden_layers` of them for layer-wise cross-attn — so the DiT
        # depth and project_layers count must match `num_hidden_layers` exactly.
        # Qwen3-VL stores num_hidden_layers under text_config; Qwen2.5-VL puts it
        # on the top-level config.  getattr(..., vlm_hf_cfg) handles both cases.
        vlm_hf_cfg = self.qwen_vl_interface.model.config
        text_cfg = getattr(vlm_hf_cfg, "text_config", vlm_hf_cfg)
        num_vl_layers = int(text_cfg.num_hidden_layers)
        llm_hidden_size = int(vlm_hf_cfg.hidden_size)
        self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
        self.config.framework.qwenvl.num_vl_layers = num_vl_layers

        # Resolve the Action DiT hidden dim BEFORE building the action head,
        # so that LayerwiseFlowmatchingActionHead constructs DiT at the right size.
        # If the user did not specify it, fall back to the LLM hidden size
        # (i.e. behave like QwenPI: project_layers becomes nn.Identity()).
        #
        # NOTE: `action_dit_hidden_dim` is a framework-side hint only — it is
        # NOT a DiT constructor kwarg, so we keep it out of diffusion_model_cfg
        # and instead pass it through `populate_layerwise_dit_cfg`, which writes
        # the canonical DiT-shape fields (input_embedding_dim, cross_attention_dim,
        # num_attention_heads).
        diffusion_model_cfg = self.config.framework.action_model.diffusion_model_cfg
        action_dit_hidden_dim = diffusion_model_cfg.get("action_dit_hidden_dim", None)
        if action_dit_hidden_dim is None:
            action_dit_hidden_dim = llm_hidden_size
        self.action_dit_hidden_dim = int(action_dit_hidden_dim)

        # Push the resolved DiT shape into diffusion_model_cfg.  The action head
        # is intentionally agnostic of qwenvl.* — it only consumes this dict.
        populate_layerwise_dit_cfg(
            self.config,
            dit_hidden_dim=self.action_dit_hidden_dim,
            num_dit_layers=num_vl_layers,
        )

        self.action_model: LayerwiseFlowmatchingActionHead = get_action_model(config=self.config)
        self.num_action_dit_layers = len(self.action_model.model.transformer_blocks)

        # Layer-wise projector: map each selected VL hidden to Action DiT hidden space.
        # This explicitly decouples VL representation size from action DiT latent size.
        self.project_layers = nn.ModuleList(
            [
                (
                    nn.Identity()
                    if llm_hidden_size == self.action_dit_hidden_dim
                    else nn.Sequential(
                        nn.LayerNorm(llm_hidden_size),
                        nn.Linear(llm_hidden_size, self.action_dit_hidden_dim),
                    )
                )
                for _ in range(self.num_action_dit_layers)
            ]
        )

        # `action_horizon` is the single source of truth for chunk length.
        # Legacy aliases (`future_action_window_size`, `past_action_window_size`)
        # are normalised upstream by `share_tools.apply_config_compat`, so we
        # only ever read `action_horizon` here.
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self.action_dim = int(self.config.framework.action_model.action_dim)
        self.state_dim = int(self.config.framework.action_model.get("state_dim") or 0)
        vla_cfg = None
        if self.config is not None and getattr(self.config, "datasets", None) is not None:
            datasets_cfg = self.config.datasets
            vla_cfg = (
                datasets_cfg.get("vla_data")
                if hasattr(datasets_cfg, "get")
                else getattr(datasets_cfg, "vla_data", None)
            )
        history = 1
        if vla_cfg is not None:
            history = int(
                (vla_cfg.get("state_history_frames") if hasattr(vla_cfg, "get") else 1)
                or 1
            )
        if history < 1:
            raise ValueError(
                f"datasets.vla_data.state_history_frames must be >= 1, got {history}"
            )
        self.state_history_frames = history

        # Optional continuous-state path into the action head (see the
        # `continuous_state` key in QwenPI_v3DefaultConfig). The discretised
        # [STATE] prompt injection is unaffected; this only controls whether
        # the raw normalized state ALSO reaches action_model.state_encoder.
        self.continuous_state = bool(
            self.config.framework.action_model.get("continuous_state", False)
        )

        # ── Optional spatial alignment (Spark-0 local addition) ──────────
        # When `framework.spatial_alignment.enable` is false (the default),
        # NOTHING is constructed here: no new parameters, no new state_dict
        # keys, no optimizer-group changes — the model is byte-identical to
        # the baseline. When enabled, only the small trainable projector is
        # registered; the frozen DA3 teacher is lazy-loaded inside the first
        # TRAINING forward (then a dist.barrier so NFS I/O cannot desync
        # ranks), so checkpoints stay portable and the inference /
        # policy-server path (predict_action) never touches DA3.
        sa_cfg = self.config.framework.get("spatial_alignment", None)
        if sa_cfg is not None and bool(sa_cfg.get("enable", False)):
            from starVLA.model.modules.spatial_alignment import DA3SpatialAlignment

            # hidden_states index to supervise: hidden_states[0] is the token
            # embedding, hidden_states[k] the output of decoder layer k. SF
            # finds "deep but not deepest" (~2/3 depth) best -> ~24 of 36.
            # Negative indices follow Python indexing over the L+1 hidden
            # states (-1 == the last layer), so a config like -12 keeps the
            # same relative depth (25/36 here) across differently sized VLMs.
            requested_layer = int(sa_cfg.get("align_layer_index", 25))
            self.spatial_align_layer = (
                requested_layer + num_vl_layers + 1 if requested_layer < 0 else requested_layer
            )
            if not (1 <= self.spatial_align_layer <= num_vl_layers):
                raise ValueError(
                    f"spatial_alignment.align_layer_index={requested_layer} resolves to layer "
                    f"{self.spatial_align_layer}, outside [1, {num_vl_layers}] for this VLM."
                )
            # Consumption check: the layerwise action DiT feeds
            # hidden_states[k] (= vl_embs_list[k-1]) to DiT block k-1, and
            # with interleave_self_attention + canonical forward every ODD
            # block is self-attention-only (never cross-attends its VL
            # layer). Supervising an even layer k (odd block k-1) therefore
            # aligns features the action expert never directly reads — the
            # signal only leaks through the VLM residual stream. Odd layers
            # (23, 25, ...) map to even, cross-attending blocks.
            dit_cfg = self.config.framework.action_model.diffusion_model_cfg
            if (
                bool(dit_cfg.get("interleave_self_attention", False))
                and bool(dit_cfg.get("use_canonical_forward", True))
                and self.num_action_dit_layers == num_vl_layers
                and self.spatial_align_layer % 2 == 0
            ):
                logger.warning(
                    f"[QwenPI_v3] spatial_alignment.align_layer_index={self.spatial_align_layer} "
                    f"feeds DiT block {self.spatial_align_layer - 1}, which is SELF-ATTENTION-ONLY "
                    "under interleave_self_attention: the action expert never cross-attends the "
                    "supervised VLM layer. Pick an ODD layer (e.g. 23 or 25 for a 36-layer VLM) "
                    "or disable interleave_self_attention."
                )
            # Visual tokens are located via the image placeholder id; the
            # per-image token grid comes from image_grid_thw / merge_size, so
            # any camera count and any obs_image_size work without changes.
            self.spatial_align_image_token_id = int(getattr(vlm_hf_cfg, "image_token_id", 151655))
            image_processor = getattr(self.qwen_vl_interface.processor, "image_processor", None)
            self.spatial_align_merge_size = int(getattr(image_processor, "merge_size", 2))
            self.spatial_align = DA3SpatialAlignment(llm_hidden_size, sa_cfg)
            logger.info(
                f"[QwenPI_v3] spatial alignment ENABLED: teacher={self.spatial_align.teacher_model_path}, "
                f"align_layer={self.spatial_align_layer}/{num_vl_layers}, "
                f"loss_coeff={self.spatial_align.loss_coeff}"
            )
        else:
            self.spatial_align = None

        # ── Optional BFA++ visual token pruning (Spark-0 local addition) ──
        # Same zero-intrusion contract as spatial_alignment: disabled means no
        # new parameters / state_dict keys / forward work. When enabled, the
        # scorer MLP is the only trainable piece; pruning rewrites the DiT's
        # encoder sequence after the full VLM forward, so flash-attn and DA3
        # are unaffected. The image-token id / merge size are resolved
        # unconditionally so pruning works whether or not DA3 is also on.
        tp_cfg = self.config.framework.get("token_prune", None)
        if tp_cfg is not None and bool(tp_cfg.get("enable", False)):
            from starVLA.model.modules.token_prune import BFAPP

            # A fractional explore_noise_decay_steps is a share of the run's
            # forward calls; resolve the run length here, where the trainer
            # config is visible (max_train_steps counts OPTIMIZER steps, each
            # consuming gradient_accumulation_steps forwards per rank).
            trainer_cfg = self.config.trainer if self.config else None
            total_training_forwards = None
            if trainer_cfg:
                max_train_steps = trainer_cfg.get("max_train_steps", None)
                if max_train_steps:
                    total_training_forwards = int(max_train_steps) * int(
                        trainer_cfg.get("gradient_accumulation_steps", 1) or 1
                    )
            # The scorer reads the PROJECTED layer-wise states (the same tensor
            # the DiT cross-attends), so its width is action_dit_hidden_dim —
            # NOT llm_hidden_size, which differs whenever project_layers is not
            # an Identity.
            self.token_prune = BFAPP(
                self.action_dit_hidden_dim, tp_cfg, total_training_forwards
            )
            self.token_prune_layer = self.token_prune.resolve_score_layer(num_vl_layers)
            # The scorer reads vl_embs_list[k-1], which is hidden_states[k]
            # only because the DiT is exactly as deep as the VLM (enforced by
            # populate_layerwise_dit_cfg above). DA3 indexes hidden_states
            # absolutely, so if that invariant ever breaks the two features
            # would silently supervise/score different layers.
            if self.num_action_dit_layers != num_vl_layers:
                raise ValueError(
                    f"token_prune requires the layerwise DiT depth ({self.num_action_dit_layers}) "
                    f"to equal the VLM depth ({num_vl_layers}); score_layer_index maps to a "
                    "hidden_states index through that identity."
                )
            self.token_prune_image_token_id = int(getattr(vlm_hf_cfg, "image_token_id", 151655))
            image_processor = getattr(self.qwen_vl_interface.processor, "image_processor", None)
            self.token_prune_merge_size = int(getattr(image_processor, "merge_size", 2))
            if (
                self.config.framework.action_model.diffusion_model_cfg.get(
                    "interleave_self_attention", False
                )
                and (self.token_prune_layer - 1) % 2 == 1
            ):
                # Hard error, not a warning (unlike DA3's analogous check):
                # a scorer trained on a representation no cross-attention
                # block ever reads makes the whole feature meaningless, and a
                # warning line is far too easy to miss on a long run.
                raise ValueError(
                    f"token_prune.score_layer_index={self.token_prune_layer} feeds DiT "
                    f"block {self.token_prune_layer - 1}, which is SELF-ATTENTION-ONLY under "
                    "interleave_self_attention: the scored representation is never cross-attended, "
                    "so the scorer ranks tokens the action expert does not read. Pick an ODD layer "
                    "(e.g. 23 or 25 for a 36-layer VLM)."
                )
            logger.info(
                f"[QwenPI_v3] token pruning ENABLED: score_layer={self.token_prune_layer}/{num_vl_layers}, "
                f"beta={self.token_prune.beta}, min_keep_ratio={self.token_prune.min_keep_ratio}, "
                f"view_alphas={self.token_prune.view_alphas}, "
                f"explore_noise={self.token_prune.explore_noise}"
                f"->{self.token_prune.explore_noise_final} over "
                f"{self.token_prune.explore_noise_decay_steps or 'inf'} forwards, "
                f"detach_scorer_input={self.token_prune.detach_scorer_input}"
            )
        else:
            self.token_prune = None

        # ── Optional tactile encoder (Spark-0 local addition) ────────────
        # Zero-intrusion contract (same as spatial_alignment / token_prune):
        # `framework.tactile.enable=false` (default; `tactile_fusion: null`)
        # constructs NOTHING — no parameters, no state_dict keys, and
        # forward / predict_action never touch tactile, so the run is
        # bit-identical to the pre-tactile baseline. Attribute is always
        # set (to None when disabled) so `trainer.learning_rate.tactile_encoder`
        # resolves cleanly and is skipped, matching `spatial_align`.
        #
        # A selected `tactile_fusion:` stem makes train.sh write enable and
        # action_fusion.enable. `encode_tactile(...)` is the shared
        # train/infer entry for dual streams. `action_fusion.enable` is P1:
        # slow (3D) tokens append_context, region window → AdaLN extra_temb
        # (both gates 0-init). Recipe-inline `action_fusion.enable: false`
        # keeps the encoder in the graph via `_zero_tactile_fusion_touch`
        # (DDP/ZeRO) while actions stay V-only.
        tac_cfg = self.config.framework.get("tactile", None)
        if tac_cfg is not None and bool(tac_cfg.get("enable", False)):
            from starVLA.model.modules.tactile import get_tactile_encoder

            self.tactile_encoder = get_tactile_encoder(tac_cfg)
            logger.info(
                f"[QwenPI_v3] tactile encoder ENABLED: type="
                f"{self.tactile_encoder.encoder_type}, "
                f"tokens={self.tactile_encoder.num_tokens} "
                f"(region={self.tactile_encoder.region_num_tokens}, "
                f"taxel={self.tactile_encoder.taxel_num_tokens}), "
                f"fast_enable={self.tactile_encoder.fast_enable}, "
                f"slow_enable={self.tactile_encoder.slow_enable}, "
                f"output_dim={self.tactile_encoder.output_dim}, "
                f"modality_dropout="
                f"{self.tactile_encoder.modality_dropout_prob}"
            )
        else:
            self.tactile_encoder = None

        # ── Optional tactile → Action-DiT fusion (P1) ──────────────────
        # Requires the encoder. Disabled (default) means the DiT context is
        # byte-for-byte the V-only sequence. Enabled:
        #   * 3D taxel tokens → append_context (`tactile_gate`)
        #   * region tokens → append_context (`tactile_patch_gate`; falls
        #     back to tactile_gate when patch is off)
        #   * region window → flatten → PatchToTemb → AdaLN extra_temb
        #     (UNGATED — `PatchToTemb.fc2` is zero-init and is that path's
        #     own scale; sharing `tactile_patch_gate` with the LayerNormed
        #     token path made one scalar mean two things. Skipped when the
        #     region stream is off.)
        # Both gates apply only to rows that carry real tactile; an absent
        # row's `no_tactile` placeholder passes through ungated.
        # Under `mode: joint_stream` the first two instead join the DiT hidden
        # stream; `joint_gate_mode: value` (default) keeps the same gate-scales-
        # the-value contract there, so the gates get a real gradient.
        fusion_cfg = (
            tac_cfg.get("action_fusion", None)
            if self.tactile_encoder is not None
            else None
        )
        fusion_enabled = bool(fusion_cfg.get("enable", False)) if fusion_cfg else False
        self._tactile_extra_temb = None
        if fusion_enabled:
            from starVLA.model.modules.tactile import get_tactile_action_fusion

            dit_cfg = self.config.framework.action_model.diffusion_model_cfg
            self.tactile_fusion = get_tactile_action_fusion(
                fusion_cfg,
                num_dit_layers=self.num_action_dit_layers,
                dit_hidden_dim=self.action_dit_hidden_dim,
                tactile_dim=self.tactile_encoder.output_dim,
                interleave_self_attention=bool(
                    dit_cfg.get("interleave_self_attention", False)
                ),
                use_canonical_forward=bool(dit_cfg.get("use_canonical_forward", True)),
                patch_cond_dim=self.tactile_encoder.patch_cond_dim,
                patch_token_start=self.tactile_encoder.region_token_span[0],
                patch_token_count=self.tactile_encoder.region_token_span[1],
            )
            if self.tactile_fusion.mode == "joint_stream" and dit_cfg.get(
                "positional_embeddings", None
            ):
                # joint_stream inserts T tactile rows BEFORE the actions, and
                # the DiT's per-block positional embedding is applied by
                # stream index — the action tokens would silently shift by T
                # positions, so a V-only checkpoint would no longer warm-start.
                # All Spark-0 recipes use positional_embeddings: null.
                raise ValueError(
                    "framework.tactile.action_fusion.mode=joint_stream requires "
                    "diffusion_model_cfg.positional_embeddings: null — a DiT-internal "
                    "positional embedding is indexed by stream position and the "
                    "tactile block would shift every action token."
                )
            refresh_mode = str(fusion_cfg.get("refresh_mode", "both")).lower()
            if refresh_mode not in ("both", "fast_cached_slow"):
                raise ValueError(
                    "framework.tactile.action_fusion.refresh_mode must be "
                    f"both|fast_cached_slow, got {refresh_mode!r}"
                )
            # Single-expert ablation keeps the full encode; cache needs both.
            if refresh_mode == "fast_cached_slow" and (
                self.tactile_encoder.fast_num_tokens < 1
                or self.tactile_encoder.slow_num_tokens < 1
            ):
                logger.info(
                    "[QwenPI_v3] action_fusion.refresh_mode=fast_cached_slow "
                    "needs both roles; falling back to both"
                )
                refresh_mode = "both"
            self._tactile_refresh_mode = refresh_mode
            self._tactile_slow_refresh_every = int(
                fusion_cfg.get("slow_refresh_every", 1)
            )
            if self._tactile_slow_refresh_every < 1:
                raise ValueError("action_fusion.slow_refresh_every must be >= 1")
            self._tactile_slow_cache = None
            patch_gate = self.tactile_fusion.patch_gate
            patch_gate_s = (
                f"{float(patch_gate.detach()):.3f}"
                if patch_gate is not None
                else "off"
            )
            if self.tactile_fusion.mode == "joint_stream":
                joint_s = (
                    f"joint_stream/{self.tactile_fusion.joint_gate_mode}-gated t2a "
                    f"(a2t bias scale {self.tactile_fusion.joint_bias_scale:g}) on "
                )
            else:
                joint_s = "append_context on "
            logger.info(
                f"[QwenPI_v3] tactile action fusion ENABLED: {joint_s}"
                f"{self.tactile_fusion.num_cross_layers}/{self.num_action_dit_layers} "
                f"cross-attn blocks, +{self.tactile_encoder.num_tokens} tokens "
                f"(fast={self.tactile_encoder.fast_num_tokens} + "
                f"slow={self.tactile_encoder.slow_num_tokens}; "
                f"region block [{self.tactile_fusion.patch_token_start}, "
                f"{self.tactile_fusion.patch_token_start + self.tactile_fusion.patch_token_count})"
                f" × tactile_patch_gate, rest × tactile_gate; present rows "
                f"only), "
                f"patch={self.tactile_fusion.patch_mode}/"
                f"{self.tactile_encoder.patch_cond_dim}d ungated, "
                f"gate_init={float(self.tactile_fusion.gate.detach()):.3f}, "
                f"patch_gate_init={patch_gate_s}, "
                f"gate_l1={self.tactile_fusion.gate_l1:g}, "
                f"refresh={self._tactile_refresh_mode}/"
                f"{self._tactile_slow_refresh_every} infer-calls "
                f"(3D period = infer_cadence × {self._tactile_slow_refresh_every} control steps)"
            )
        else:
            self.tactile_fusion = None
            self._tactile_refresh_mode = "both"
            self._tactile_slow_refresh_every = 1
            self._tactile_slow_cache = None
            if (
                self.tactile_encoder is None
                and tac_cfg is not None
                and bool((tac_cfg.get("action_fusion", None) or {}).get("enable", False))
            ):
                # Easy config trap: the fusion switch alone cannot build the
                # encoder it consumes, so the run would silently stay V-only.
                print(
                    "[starVLA][WARN] framework.tactile.action_fusion.enable=true is "
                    "ignored because framework.tactile.enable=false — there is no "
                    "encoder to fuse. Set both to train VT."
                )

        # Batch keys carrying tactile tensors. Missing keys mean "this row
        # has no tactile" (present=False → learned no_tactile token), so a
        # fusion-enabled run on non-tactile data trains like V plus the
        # placeholder. Datasets that declare a tactile block (egopro) pack
        # ``region_pressure`` / ``taxel_pressure_3d`` (and ``tactile_2d`` for
        # align) in ``_pack_sample``.
        if self.tactile_encoder is not None:
            from starVLA.model.modules.tactile.batch_packing import (
                resolve_tactile_batch_keys,
            )

            self._tactile_batch_keys = resolve_tactile_batch_keys(tac_cfg)
        else:
            self._tactile_batch_keys = None
        self._warned_tactile_missing = False
        self._tactile_last_tokens = None
        self._tactile_joint_cond = None

        # ── Optional VLM-side action queries (Spark-0 0821 plan, change 2) ──
        # Zero-intrusion: `action_query.enable=false` (default) constructs
        # nothing and `_encode_vl_hidden_states` stays bit-identical. When
        # enabled, K learnable query embeddings are appended at the end of
        # the Qwen sequence via a dummy-token + embedding-hook splice (the
        # vocab is never resized, so pre-action-query checkpoints load with
        # only the new query rows missing — strict=False keeps them at init).
        aq_cfg = self.config.framework.get("action_query", None)
        aq_on = aq_cfg is not None and bool(aq_cfg.get("enable", False))
        self.action_query_kv_mode = str(aq_cfg.get("kv_mode", "query_only")) if aq_on else None
        if aq_on:
            if self.action_query_kv_mode not in ("query_only", "query_plus_visual"):
                raise ValueError(
                    f"framework.action_query.kv_mode={self.action_query_kv_mode!r} "
                    "must be 'query_only' or 'query_plus_visual'."
                )
            self.action_query_num = int(aq_cfg.get("num_query", 64))
            if self.action_query_num < 1:
                raise ValueError(
                    f"framework.action_query.num_query must be >= 1, got {self.action_query_num}"
                )
            init_std = float(aq_cfg.get("init_std", 0.02))
            # A module (not a bare Parameter) so ZeRO-3 fetches the weight
            # through its forward when the splice hook runs inside the VLM
            # forward — reading a partitioned Parameter directly there would
            # see a shard.
            self.action_query_embed = nn.Embedding(self.action_query_num, llm_hidden_size)
            nn.init.normal_(self.action_query_embed.weight, mean=0.0, std=init_std)
            dummy_token = str(aq_cfg.get("dummy_token", "🔍"))
            dummy_ids = self.qwen_vl_interface.processor.tokenizer(
                dummy_token, add_special_tokens=False
            )["input_ids"]
            if len(dummy_ids) != 1:
                raise ValueError(
                    f"framework.action_query.dummy_token={dummy_token!r} must map to "
                    f"exactly one token id, got {dummy_ids}."
                )
            self.action_query_dummy_id = int(dummy_ids[0])
            if (
                self.action_query_kv_mode == "query_only"
                and self.token_prune is not None
            ):
                # The DiT reads only the query slice; ranking visual tokens
                # buys nothing and the pruner's visual-count assert would
                # fail on the sliced sequence. Skipped at runtime instead.
                print(
                    "[starVLA][WARN] action_query.kv_mode=query_only makes "
                    "token_prune a no-op for the action DiT (visual tokens are "
                    "no longer in the cross-attn KV); pruning is skipped."
                )
            logger.info(
                f"[QwenPI_v3] action query ENABLED: K={self.action_query_num}, "
                f"kv_mode={self.action_query_kv_mode}, dummy_id={self.action_query_dummy_id}"
            )
        else:
            self.action_query_embed = None
            self.action_query_num = 0
            self.action_query_dummy_id = None

        # Shortcut mask (plan §5) needs the tactile joint stream as the
        # alternative VL path; without it a masked sample has no context at
        # all. The head validates at forward too (defense in depth).
        if float(self.config.framework.action_model.get("shortcut_mask_prob", 0.0) or 0.0) > 0.0:
            joint_on = (
                self.tactile_fusion is not None
                and getattr(self.tactile_fusion, "mode", "append_context") == "joint_stream"
            )
            if not joint_on:
                raise ValueError(
                    "framework.action_model.shortcut_mask_prob > 0 requires "
                    "framework.tactile.action_fusion.mode: joint_stream — the "
                    "masked action expert needs the tactile stream as its "
                    "alternative VL path."
                )

        # ── Optional contact-region visual alignment (Spark-0 local) ───
        # Zero-intrusion contract (same as spatial_alignment / tactile):
        # `framework.tactile_align.enable=false` (default; `tactile_align:
        # null`) constructs NOTHING — no parameters, no state_dict keys,
        # and forward / predict_action never touch this path. Attribute is
        # always set (None when disabled) so `trainer.learning_rate.tactile_align`
        # resolves and is skipped. A selected stem makes train.sh write enable.
        #
        # When enabled, training may emit `contact_align_loss` (already
        # coeff-weighted). Rows with present2d=False (no 2D contact / non-
        # tactile data) contribute a zero loss with param touch so mixed
        # batches and multi-GPU reduces stay safe. Inference never runs it.
        ta_cfg = self.config.framework.get("tactile_align", None)
        if ta_cfg is not None and bool(ta_cfg.get("enable", False)):
            from starVLA.model.modules.tactile_align import get_tactile_align

            requested_layer = int(ta_cfg.get("align_layer_index", 25))
            self.tactile_align_layer = (
                requested_layer + num_vl_layers + 1 if requested_layer < 0 else requested_layer
            )
            if not (1 <= self.tactile_align_layer <= num_vl_layers):
                raise ValueError(
                    f"tactile_align.align_layer_index={requested_layer} resolves to layer "
                    f"{self.tactile_align_layer}, outside [1, {num_vl_layers}] for this VLM."
                )
            dit_cfg = self.config.framework.action_model.diffusion_model_cfg
            if (
                bool(dit_cfg.get("interleave_self_attention", False))
                and bool(dit_cfg.get("use_canonical_forward", True))
                and self.num_action_dit_layers == num_vl_layers
                and self.tactile_align_layer % 2 == 0
            ):
                logger.warning(
                    f"[QwenPI_v3] tactile_align.align_layer_index={self.tactile_align_layer} "
                    f"feeds DiT block {self.tactile_align_layer - 1}, which is SELF-ATTENTION-ONLY "
                    "under interleave_self_attention: pick an ODD layer (e.g. 23 or 25)."
                )
            self.tactile_align_image_token_id = int(getattr(vlm_hf_cfg, "image_token_id", 151655))
            image_processor = getattr(self.qwen_vl_interface.processor, "image_processor", None)
            self.tactile_align_merge_size = int(getattr(image_processor, "merge_size", 2))
            self.tactile_align = get_tactile_align(ta_cfg, llm_hidden_size)
            logger.info(
                f"[QwenPI_v3] tactile_align ENABLED: align_layer="
                f"{self.tactile_align_layer}/{num_vl_layers}, "
                f"views={self.tactile_align.align_view_indices}, "
                f"infonce={self.tactile_align.infonce_coeff}, "
                f"contact_map={self.tactile_align.contact_map_coeff}, "
                f"pressure_bin={self.tactile_align.pressure_bin_coeff}"
            )
        else:
            self.tactile_align = None

        # ── Optional future-contact aux head (variant A, train-only) ──
        # Needs the region expert tokens. Inference never calls it.
        # Attribute is always set so trainer.learning_rate.tactile_future_contact
        # resolves (None → skip).
        fc_cfg = {}
        if tac_cfg is not None:
            expert = tac_cfg.get("tactile_expert") or {}
            fc_cfg = expert.get("future_contact") or {} if isinstance(expert, dict) else {}
        if fc_cfg and bool(fc_cfg.get("enable", False)):
            if self.tactile_encoder is None:
                raise ValueError(
                    "tactile_expert.future_contact.enable=true needs "
                    "framework.tactile.enable and a fusion stem — there is "
                    "no region encoder to hang the aux head on."
                )
            if not self.tactile_encoder.region_enabled:
                raise ValueError(
                    "future_contact variant A reads region expert tokens; "
                    "enable streams.region_pressure (fingertip_only / robust). "
                    "The 3D-only default stem has nothing to predict from."
                )
            from starVLA.model.modules.tactile import get_future_contact

            self.tactile_future_contact = get_future_contact(
                fc_cfg,
                token_dim=self.tactile_encoder.output_dim,
                num_regions=self.tactile_encoder.region_num_tokens,
            )
            # The loader slices the exported window to
            # `tactile_future_horizon`; a hand-edited value that disagrees
            # with the head would otherwise raise on the first batch.
            packed_horizon = self._vla_data_value(self.config, "tactile_future_horizon")
            if packed_horizon not in (None, "", "null") and int(packed_horizon) != int(
                self.tactile_future_contact.horizon
            ):
                raise ValueError(
                    f"datasets.vla_data.tactile_future_horizon={packed_horizon} "
                    f"!= tactile_expert.future_contact.horizon="
                    f"{self.tactile_future_contact.horizon}. Leave the dataset "
                    "key unset (sync_tactile_pack_flags / train.sh write it "
                    "from the stem) or make the two agree."
                )
            logger.info(
                f"[QwenPI_v3] future_contact ENABLED: variant=aux_head "
                f"horizon={self.tactile_future_contact.horizon} glove frames "
                f"K={self.tactile_future_contact.num_subwindows} "
                f"R={self.tactile_future_contact.num_regions} "
                f"loss_coeff={self.tactile_future_contact.loss_coeff}"
            )
        else:
            self.tactile_future_contact = None

        self._warn_flexible_views_cotrain_gates()

    @staticmethod
    def _vla_data_value(config, key: str, default=None):
        """``config.datasets.vla_data[key]`` on dict or OmegaConf, else default."""
        datasets = getattr(config, "datasets", None)
        if datasets is None:
            return default
        vla = (
            datasets.get("vla_data")
            if hasattr(datasets, "get")
            else getattr(datasets, "vla_data", None)
        )
        if vla is None or not hasattr(vla, "get"):
            return default
        return vla.get(key, default)

    @staticmethod
    def _vla_flexible_views_enabled(config) -> bool:
        """True when ``datasets.vla_data.flexible_views`` is on (string ``\"false\"`` stays off)."""
        if config is None:
            return False
        datasets = getattr(config, "datasets", None)
        if datasets is None:
            return False
        vla = datasets.get("vla_data") if hasattr(datasets, "get") else getattr(datasets, "vla_data", None)
        if vla is None:
            return False
        raw = vla.get("flexible_views", False) if hasattr(vla, "get") else False
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, int) and raw in (0, 1):
            return bool(raw)
        if isinstance(raw, str):
            return raw.strip().lower() in {"true", "1", "yes", "on"}
        return bool(raw)

    def _warn_flexible_views_cotrain_gates(self) -> None:
        """Construction-time footguns when mixed-mask batches are legal.

        DA3, token prune, and tactile_align split mixed layouts inside the
        loss / prune path. ``align_view_indices`` past a group's packed
        count skip that group and re-normalize the rest to weight 1. Warn
        here so a long run does not hide the skip in a one-line mid-loop
        message.
        """
        if not self._vla_flexible_views_enabled(self.config):
            return
        if self.spatial_align is not None:
            indices = self.spatial_align.align_view_indices
            if indices is not None and any(int(idx) > 0 for idx in indices):
                logger.warning(
                    "[QwenPI_v3] flexible_views is on and spatial_alignment.align_view_indices="
                    f"{indices} indexes PACKED views, not catalog slots. A head-only layout "
                    "group cannot satisfy an index past its packed count and is skipped; "
                    "the remaining samples' DA3 loss is re-normalized to 1 (full loss_coeff). "
                    "Use null (every present view) or [0] (first packed camera, usually head)."
                )
        if self.token_prune is not None:
            alphas = getattr(self.token_prune, "view_alphas", None)
            if alphas is not None:
                logger.warning(
                    "[QwenPI_v3] token_prune.view_alphas="
                    f"{alphas} is packed-view order under flexible_views: a group "
                    "with fewer cameras uses the prefix (head-only keeps a_0). "
                    "Leave view_alphas=null unless you intend that split."
                )
        if self.tactile_align is not None:
            indices = self.tactile_align.align_view_indices
            if indices is not None and any(int(idx) > 0 for idx in indices):
                logger.warning(
                    "[QwenPI_v3] tactile_align.align_view_indices="
                    f"{indices} indexes PACKED views. A layout group whose packed "
                    "count cannot satisfy the index is skipped; remaining groups "
                    "still train contact-align. Use [0] for head-camera heatmaps."
                )

    def _pack_tactile_encode_kwargs(
        self, examples: List[dict], device: torch.device, dtype: torch.dtype
    ) -> dict:
        """Build ``TactileEncoder.encode`` kwargs from the packed batch.

        Every tensor is optional: an all-absent batch resolves to the learned
        ``no_tactile`` tokens, which is exactly the V-mode fallback. Warns once
        when the batch carries no tactile column at all, so a run that thinks
        it is training VT but is really training V-plus-placeholder is visible
        in the log directly, rather than only as a flat
        ``tactile_extra_temb_rel_norm``.
        """
        from starVLA.model.modules.tactile.batch_packing import (
            pack_tactile_encode_kwargs,
        )

        kwargs, found_any = pack_tactile_encode_kwargs(
            examples, self._tactile_batch_keys, device, dtype
        )
        if self.tactile_encoder is not None:
            self.tactile_encoder.assert_encode_kwargs(kwargs)
        if not found_any and not self._warned_tactile_missing:
            print(
                "[starVLA][WARN] tactile fusion is enabled but no tactile keys were "
                f"found in the batch (looked for {self._tactile_batch_keys['fast']} / "
                f"{self._tactile_batch_keys['slow']}); every row falls back to the "
                "learned no_tactile token. Fusion needs region_pressure / "
                "taxel_pressure_3d on the sample (dataloader packs them when "
                "meta/modality.json has a tactile block)."
            )
            self._warned_tactile_missing = True
        return kwargs

    def _append_tactile_context(
        self,
        vl_embs_list: List[torch.Tensor],
        backbone_attention_mask: Optional[torch.Tensor],
        examples: List[dict],
        *,
        apply_modality_dropout: Optional[bool] = None,
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Fuse tactile into the DiT: full token hierarchy + region extra_temb.

        No-op (returns the inputs unchanged) unless BOTH the encoder and
        ``action_fusion`` are enabled, so a V-only run never pays for this.
        Must run AFTER token pruning — pruning indexes VISUAL tokens and would
        otherwise rank / drop tactile ones.

        The WHOLE laid-out sequence (fast region tokens + slow 3D tokens) is
        appended. The region block rides ``tactile_patch_gate``, the 3D taxel
        tokens ``tactile_gate`` — separate so opening the trusted force
        expert cannot inject the untrusted camera-frame cloud — and both only
        apply to rows whose stream is really present. The FiLM path below
        (raw window → PatchToTemb → AdaLN) is UNGATED, so force enters as
        both per-region tokens (spatial, temporal) and a global modulation.

        Stores ``self._tactile_extra_temb`` (``[B, dit_hidden]`` or None) for
        the subsequent ``action_model`` / ``predict_action`` call.
        """
        self._tactile_extra_temb = None
        self._tactile_joint_cond = None
        if self.tactile_encoder is None or self.tactile_fusion is None:
            return vl_embs_list, backbone_attention_mask, None
        device = vl_embs_list[0].device
        # `encode` adopts the dtype of whatever tensor it is handed, so the
        # batch must be packed in the encoder's own parameter dtype — under
        # DeepSpeed bf16 the trunk weights are bf16 and an fp32 batch would
        # raise inside the first Linear.
        param = next(self.tactile_encoder.parameters(), None)
        dtype = param.dtype if param is not None else vl_embs_list[0].dtype
        kwargs = self._pack_tactile_encode_kwargs(examples, device, dtype)
        if apply_modality_dropout is not None:
            kwargs["apply_modality_dropout"] = apply_modality_dropout
        from starVLA.model.modules.tactile.tactile_encoder import (
            tactile_infer_cache_id,
        )

        cache_id = tactile_infer_cache_id(examples)
        tactile_tokens, tactile_mask = self._encode_tactile_for_fusion(
            kwargs, cache_id=cache_id
        )
        # Stash for the future-contact head (variant A) so the aux loss does
        # not encode a second time. Only under grad and only when the head
        # exists: keeping a graph-attached tensor on `self` past the backward
        # would retain one step's tactile subgraph on every fusion run.
        self._tactile_last_tokens = (
            tactile_tokens
            if (self.tactile_future_contact is not None and torch.is_grad_enabled())
            else None
        )
        enc = self.tactile_encoder
        # Post-dropout / post-quality present flags, per stream. The fusion
        # gates apply ONLY to rows that carry real tactile: an absent row's
        # slots hold the learned `no_tactile` placeholder, and gating that
        # would feed the gates a gradient from rows with no tactile signal
        # (the majority of a mixed batch) through an over-parameterised
        # `gate * no_tactile` product.
        present_region = enc.region_present_mask()
        present_3d = enc.taxel_present_mask()
        if getattr(self.tactile_fusion, "mode", "append_context") == "joint_stream":
            # Joint mode: the tactile tokens enter the DiT HIDDEN stream
            # (co-evolving expert) instead of the per-layer context. The
            # vl_embs_list / mask are untouched; the head receives the bundle
            # via ``joint_cond``. Under the default `joint_gate_mode: value`
            # the tokens are already gate-scaled here and the head's bias only
            # carries a2t; `bias` restores the legacy all-attention gate.
            # Placeholder ``no_tactile`` slots keep T = enc.num_tokens > 0
            # even on all-absent rows, so every encoder / fusion parameter
            # sees a real autograd edge through the bias.
            self._tactile_joint_cond = self.tactile_fusion.build_joint_cond(
                tactile_tokens, present_region, present_3d
            )
        else:
            # Append the full laid-out hierarchy. Placeholder ``no_tactile``
            # slots keep T = enc.num_tokens > 0 even on all-absent rows, so every
            # encoder / fusion parameter sees a real autograd edge — no zero-touch
            # is needed anywhere on this path.
            vl_embs_list, backbone_attention_mask = self.tactile_fusion(
                vl_embs_list,
                tactile_tokens,
                tactile_mask=tactile_mask,
                encoder_attention_mask=backbone_attention_mask,
                present_region=present_region,
                present_3d=present_3d,
            )

        patch_cond = enc.flatten_patch_cond(
            kwargs.get("region_pressure"),
            present_fast=present_region,
            batch_size=int(tactile_tokens.shape[0]),
            device=tactile_tokens.device,
            dtype=tactile_tokens.dtype,
        )
        extra_temb = self.tactile_fusion.patch_to_temb(
            patch_cond, present=present_region
        )
        self._tactile_extra_temb = extra_temb
        return vl_embs_list, backbone_attention_mask, self.tactile_fusion.gate.detach()

    def _take_tactile_extra_temb(self) -> Optional[torch.Tensor]:
        """Pop the P1 AdaLN addend so ``self`` does not keep the train graph."""
        extra_temb = getattr(self, "_tactile_extra_temb", None)
        self._tactile_extra_temb = None
        return extra_temb

    def _take_tactile_joint_cond(self) -> Optional[dict]:
        """Pop the joint_stream bundle so ``self`` does not keep the graph."""
        cond = getattr(self, "_tactile_joint_cond", None)
        self._tactile_joint_cond = None
        return cond

    def _encode_tactile_for_fusion(self, kwargs: dict, cache_id=None):
        """Train = ``encode(both)``. Infer may refresh only the fast patch.

        See ``encode_with_slow_refresh``: ``fast_cached_slow`` reuses the
        last slow 3D tokens and re-encodes ``region_pressure``.
        """
        from starVLA.model.modules.tactile.tactile_encoder import (
            encode_with_slow_refresh,
        )

        tokens, mask, self._tactile_slow_cache = encode_with_slow_refresh(
            self.tactile_encoder,
            kwargs,
            training=bool(self.training),
            refresh_mode=self._tactile_refresh_mode,
            slow_refresh_every=self._tactile_slow_refresh_every,
            cache=self._tactile_slow_cache,
            cache_id=cache_id,
        )
        return tokens, mask

    def encode_tactile(self, *args, **kwargs):
        """Encode tactile → ``(tokens, token_mask)``, or ``(None, None)`` if off.

        When ``framework.tactile.enable=false`` this is a no-op that returns
        ``(None, None)`` so callers (future fusion in ``forward`` /
        ``predict_action``) can branch without changing the baseline path —
        bit-identical to a build with no tactile module.

        When enabled, forwards to ``TactileEncoder.encode``:
        - preferred inputs: ``region_pressure`` / ``taxel_pressure_3d``
          (temporal); legacy ``points`` still accepted when history_len=1;
        - rows with ``present=False`` (no glove / non-tactile dataset) and
          training-time modality drops both use the learned ``no_tactile``
          placeholder — mixed V/VT batches are valid;
        - ``apply_modality_dropout`` defaults to ``self.training`` (off at
          eval / ``predict_action``).
        """
        if self.tactile_encoder is None:
            return None, None
        return self.tactile_encoder.encode(*args, **kwargs)

    def _zero_token_prune_touch(
        self, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Zero scalar from one dummy scorer forward (ZeRO-3 collective-safe).

        ``param.sum() * 0`` does not all-gather partitioned weights. A rank
        that skipped while another called ``score_tokens`` would deadlock
        under ZeRO-3. Same contract as ``DA3SpatialAlignment.align_segments``.
        Also ticks the explore-noise anneal so skip ranks stay on schedule.
        """
        if self.token_prune is None:
            raise RuntimeError("_zero_token_prune_touch requires token_prune.")
        dummy = torch.zeros(
            1, 1, self.token_prune.in_dim, device=device, dtype=dtype
        )
        scores = self.token_prune.score_tokens(dummy)
        self.token_prune._advance_explore_if_training()
        return scores.float().sum() * 0

    # Trainer reduce is one collective per key. A skip that emits only
    # ``contact_align_loss`` while another rank emits the sub-losses hangs
    # multi-GPU (mixed flexible_views batches). Keep this list identical to
    # the keys copied out of a real ``ContactRegionAlignment`` forward.
    _TACTILE_ALIGN_METRIC_KEYS = (
        "contact_map_loss",
        "dice_loss",
        "infonce_loss",
        "pressure_bin_loss",
        "contact_align_present_frac",
    )
    _FUTURE_CONTACT_METRIC_KEYS = (
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

    def _zero_tactile_align_outputs(
        self, *, device: torch.device, dtype: torch.dtype, batch_size: int = 1
    ) -> Tuple[torch.Tensor, dict]:
        """Zero ``contact_align_loss`` via one ``forward_groups([])`` call.

        ZeRO-3 all-gathers on module forward, not on ``param.sum()``. Empty
        segments still project a dummy row — same rank-invariant contract as
        ``DA3SpatialAlignment.align_segments``.
        """
        if self.tactile_align is None:
            raise RuntimeError("_zero_tactile_align_outputs requires tactile_align.")
        metrics = self.tactile_align.forward_groups(
            [],
            batch_size=max(1, int(batch_size)),
            device=device,
            dtype=dtype,
        )
        return metrics["loss"], metrics

    def _zero_tactile_fusion_touch(
        self, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Zero scalar from one dummy encoder forward (ZeRO-3 collective-safe).

        Only for ``framework.tactile.enable=true`` WITHOUT
        ``action_fusion.enable``. ``param.sum() * 0`` would leave the
        partitioned experts un-gathered while a future path that did encode
        would all-gather them.
        """
        if self.tactile_encoder is None:
            raise RuntimeError("_zero_tactile_fusion_touch requires tactile_encoder.")
        tokens, mask = self.tactile_encoder.encode(
            batch_size=1,
            device=device,
            dtype=dtype,
            apply_modality_dropout=False,
        )
        return tokens.float().sum() * 0 + mask.float().sum() * 0

    def _compute_future_contact_loss(self, examples: List[dict]) -> dict:
        """Variant A aux loss. Always runs the head (ZeRO-3).

        Reuses tokens from ``_append_tactile_context`` when fusion encoded
        this batch; otherwise encodes now (encoder-only + future_contact
        probe). ``predict_action`` never calls this.
        """
        if self.tactile_future_contact is None:
            raise RuntimeError("_compute_future_contact_loss requires the head.")
        if self.tactile_encoder is None:
            raise RuntimeError("future_contact needs tactile_encoder.")
        device = next(self.tactile_future_contact.parameters()).device
        param = next(self.tactile_encoder.parameters(), None)
        dtype = param.dtype if param is not None else torch.float32
        tokens = self._tactile_last_tokens
        if tokens is None:
            if self._tactile_batch_keys is None:
                raise RuntimeError("future_contact encode needs tactile batch keys.")
            kwargs = self._pack_tactile_encode_kwargs(examples, device, dtype)
            tokens, _ = self.tactile_encoder.encode(**kwargs)
            self._tactile_last_tokens = tokens
        start, count = self.tactile_encoder.region_token_span
        if count < 1:
            raise RuntimeError("future_contact region_token_span is empty.")
        region_tokens = tokens[:, start : start + count]
        from starVLA.model.modules.tactile.batch_packing import (
            DEFAULT_FAST_KEYS,
            pack_future_contact_tensors,
            stack_optional_stream,
        )

        fast_keys = (
            self._tactile_batch_keys["fast"]
            if self._tactile_batch_keys is not None
            else DEFAULT_FAST_KEYS
        )
        history, _ = stack_optional_stream(examples, fast_keys, device, dtype)
        future, future_valid = pack_future_contact_tensors(examples, device, dtype)
        self.tactile_encoder.assert_future_contact_tensors(
            future, future_valid, history=history
        )
        if future is None and not getattr(self, "_warned_future_contact_missing", False):
            print(
                "[starVLA][WARN] future_contact is enabled but the batch has no "
                "region_pressure_future (set tactile_future_horizon via "
                "sync_tactile_pack_flags / a region-packing dataset); emitting "
                "zero future_contact_loss."
            )
            self._warned_future_contact_missing = True
        present = self.tactile_encoder.region_present_mask()
        metrics = self.tactile_future_contact(
            region_tokens,
            history=history,
            future=future,
            future_valid=future_valid,
            present=present,
        )
        self._tactile_last_tokens = None
        return metrics

    def _spatial_align_group_pairs(
        self,
        input_ids: torch.Tensor,
        hidden_state: torch.Tensor,
        grid_thw: torch.Tensor,
        batch_images: List,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """``(student, teacher)`` token pairs for ONE homogeneous-layout group.

        Only gathers tokens and runs the frozen teacher — no trainable
        parameter is touched, so this may run once per layout group without
        breaking the rank-invariant projector call count (see
        ``DA3SpatialAlignment.align_segments``).
        """
        if grid_thw is None:
            raise ValueError("spatial_alignment requires image inputs (image_grid_thw missing).")

        student_views, student_grids = gather_student_views(
            hidden_state,
            input_ids,
            grid_thw,
            image_token_id=self.spatial_align_image_token_id,
            merge_size=self.spatial_align_merge_size,
        )
        view_indices, targets = self.spatial_align.teacher_targets(
            batch_images, student_grids, hidden_state.device
        )
        return [(student_views[i], targets[i]) for i in view_indices]

    def _compute_spatial_align_loss(
        self, align_ctx: dict, batch_images: List
    ) -> Tuple[torch.Tensor, int]:
        """DA3 loss for a (possibly mixed-layout) batch, plus the aligned count.

        The training batch stays mixed: it is split by packed image layout
        only to gather tokens and teacher targets, then EVERY group's pairs
        go through one projector call (``align_segments``) so the ZeRO-3
        collective count does not depend on the local batch's layout.
        Groups whose packed view count cannot satisfy ``align_view_indices``
        contribute nothing; when that leaves no group at all the returned
        loss is a 0 that still touches the projector, and the returned count
        is 0 so the caller can warn.
        """
        input_ids = align_ctx["input_ids"]  # [B, L]
        hidden_state = align_ctx["hidden_state"]  # [B, L, C]
        grid_thw = align_ctx["image_grid_thw"]  # [num_images, 3]
        if grid_thw is None:
            raise ValueError("spatial_alignment requires image inputs (image_grid_thw missing).")
        # Rank-invariant first load + barrier, even when this batch aligns
        # no group (otherwise a skip rank never opens the teacher while
        # others do NFS I/O and then hit NCCL).
        self.spatial_align.ensure_teacher_loaded(hidden_state.device)

        align_view_indices = self.spatial_align.align_view_indices
        pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        group_shapes: List[Tuple[int, int]] = []
        n_groups = 0
        n_skipped = 0
        for indices, group_images, group_grid in group_samples_by_image_layout(
            batch_images, grid_thw
        ):
            n_groups += 1
            n_packed = len(group_images[0])
            if align_view_indices is not None and any(
                not 0 <= int(idx) < n_packed for idx in align_view_indices
            ):
                n_skipped += 1
                continue
            selector = torch.as_tensor(indices, device=hidden_state.device, dtype=torch.long)
            group_pairs = self._spatial_align_group_pairs(
                input_ids=input_ids.index_select(0, selector),
                hidden_state=hidden_state.index_select(0, selector),
                grid_thw=group_grid,
                batch_images=group_images,
            )
            pairs.extend(group_pairs)
            group_shapes.append((len(indices), len(group_pairs)))
        if n_skipped and not getattr(self, "_warned_flex_align_group_skip", False):
            print(
                f"[starVLA][WARN] spatial_alignment skipped {n_skipped}/{n_groups} "
                f"layout group(s) whose packed view count cannot satisfy "
                f"align_view_indices={align_view_indices}; those samples get zero "
                "DA3 gradient and the remaining groups' loss is re-normalized to 1. "
                "Use align_view_indices=null or [0] under flexible_views."
            )
            self._warned_flex_align_group_skip = True

        weights = normalized_segment_weights(group_shapes)
        if len(pairs) != len(weights):
            raise ValueError(
                f"DA3 segment/weight length mismatch: {len(pairs)} pairs vs "
                f"{len(weights)} weights (group_shapes={list(group_shapes)})."
            )
        segments = [
            (student, target, weight) for (student, target), weight in zip(pairs, weights)
        ]
        aligned_samples = sum(n_samples for n_samples, _ in group_shapes)
        align_cosine = self.spatial_align.align_segments(
            segments, device=hidden_state.device
        )
        return align_cosine, aligned_samples

    def _gather_visual_tokens_for_align(
        self,
        *,
        hidden_state: torch.Tensor,
        input_ids: torch.Tensor,
        grid_thw: torch.Tensor,
        image_token_id: int,
        merge_size: int,
        view_indices: List[int],
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Gather selected packed views' visual tokens → ``[B, P, C]`` + grid.

        Shared by contact-region alignment (and usable by future aligners).
        Every sample must share one camera layout; selected views are
        concatenated in ``view_indices`` order. When a single view is selected
        the returned grid is that view's ``(gh, gw)``; multi-view concat uses
        the first selected view's grid only for saliency resize metadata and
        requires callers to supply precomputed targets with matching ``P``.
        """
        batch_size = hidden_state.shape[0]
        if grid_thw is None:
            raise ValueError("tactile_align requires image inputs (image_grid_thw missing).")
        if grid_thw.shape[0] % batch_size != 0:
            raise ValueError(
                f"image count {grid_thw.shape[0]} is not a multiple of batch size {batch_size}."
            )
        views_per_sample = grid_thw.shape[0] // batch_size
        grid_thw = grid_thw.view(batch_size, views_per_sample, 3)
        if not torch.all(grid_thw == grid_thw[0]):
            raise ValueError(
                f"tactile_align expects every sample to share one camera layout, got {grid_thw.tolist()}."
            )

        from starVLA.model.modules.tactile_align import student_grid_hw

        student_grids: List[Tuple[int, int]] = []
        for view_idx in range(views_per_sample):
            student_grids.append(student_grid_hw(grid_thw[0, view_idx].tolist(), merge_size))
        tokens_per_view = [gh * gw for gh, gw in student_grids]
        expected = sum(tokens_per_view)
        visual_mask = input_ids == image_token_id
        counts = visual_mask.sum(dim=1)
        if not torch.all(counts == expected):
            raise ValueError(
                f"Expected {expected} visual tokens per sample "
                f"({views_per_sample} views, {tokens_per_view} tokens each), got {counts.tolist()}."
            )
        student_tokens = hidden_state[visual_mask].view(
            batch_size, expected, hidden_state.shape[-1]
        )
        student_views = list(torch.split(student_tokens, tokens_per_view, dim=1))
        if len(view_indices) != 1:
            # Heatmap / region targets are built for a single student_grid.
            # Concatenating multiple views would make P = Σ P_v while
            # contact_map stays P_0 — reject until multi-view targets exist.
            raise ValueError(
                "tactile_align currently supports exactly one supervised view "
                f"(align_view_indices length 1); got {list(view_indices)}. "
                "Use [0] for head-camera heatmap alignment."
            )
        selected = student_views[view_indices[0]]
        meta_grid = student_grids[view_indices[0]]
        return selected, meta_grid

    @staticmethod
    def _image_hw(image) -> Optional[Tuple[int, int]]:
        """H×W of a PIL / CHW / HWC image (the plane the VLM actually saw)."""
        size = getattr(image, "size", None)  # PIL: (W, H)
        if size is not None and len(size) == 2:
            w, h = int(size[0]), int(size[1])
            if h > 0 and w > 0:
                return h, w
        if isinstance(image, torch.Tensor):
            if image.ndim == 3 and image.shape[0] in (1, 3, 4):
                return int(image.shape[1]), int(image.shape[2])
            if image.ndim == 3:
                return int(image.shape[0]), int(image.shape[1])
        shape = getattr(image, "shape", None)
        if shape is not None and len(shape) >= 2:
            # numpy HWC
            return int(shape[0]), int(shape[1])
        return None

    def _compute_tactile_align_loss(
        self, align_ctx: dict, examples: List[dict], batch_images: List
    ) -> dict:
        """Run contact-region alignment on mixed-layout batches.

        Splits by packed image layout (same groups as DA3). Each group
        gathers its own student grid; tokens are flattened through one
        ``forward_groups`` projector call. Groups whose packed count cannot
        satisfy ``align_view_indices`` contribute nothing. ``obs_hw`` is
        per-group from the supervised view's actual image size.
        """
        hidden_state = align_ctx["tactile_align_hidden_state"]
        input_ids = align_ctx["input_ids"]
        grid_thw = align_ctx["image_grid_thw"]
        if grid_thw is None:
            raise ValueError("tactile_align requires image inputs (image_grid_thw missing).")
        batch_size = hidden_state.shape[0]
        align_view_indices = self.tactile_align.align_view_indices
        segments = []
        n_groups = 0
        n_skipped = 0
        for indices, group_images, group_grid in group_samples_by_image_layout(
            batch_images, grid_thw
        ):
            n_groups += 1
            n_packed = len(group_images[0])
            if align_view_indices is not None and any(
                not 0 <= int(idx) < n_packed for idx in align_view_indices
            ):
                n_skipped += 1
                continue
            view_indices = self.tactile_align.resolve_view_indices(n_packed)
            selector = torch.as_tensor(
                indices, device=hidden_state.device, dtype=torch.long
            )
            tokens, student_grid = self._gather_visual_tokens_for_align(
                hidden_state=hidden_state.index_select(0, selector),
                input_ids=input_ids.index_select(0, selector),
                grid_thw=group_grid,
                image_token_id=self.tactile_align_image_token_id,
                merge_size=self.tactile_align_merge_size,
                view_indices=view_indices,
            )
            obs_hw = None
            view_idx = view_indices[0]
            if group_images and len(group_images[0]) > view_idx:
                obs_hw = self._image_hw(group_images[0][view_idx])
            packed = self.tactile_align.build_targets_from_examples(
                [examples[i] for i in indices],
                student_grid=student_grid,
                device=tokens.device,
                obs_hw=obs_hw,
            )
            segments.append((tokens, packed))
        if n_skipped and not getattr(self, "_warned_tactile_align_group_skip", False):
            print(
                f"[starVLA][WARN] tactile_align skipped {n_skipped}/{n_groups} "
                f"layout group(s) whose packed view count cannot satisfy "
                f"align_view_indices={align_view_indices}; those samples get "
                "zero contact-align gradient. Use [0] under flexible_views."
            )
            self._warned_tactile_align_group_skip = True
        return self.tactile_align.forward_groups(
            segments,
            batch_size=batch_size,
            device=hidden_state.device,
            dtype=hidden_state.dtype,
        )

    def _project_vl_hidden_for_action(self, vl_embs_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """Project layer-wise VL hidden states to the hidden space expected by Action DiT."""
        if len(vl_embs_list) != len(self.project_layers):
            raise ValueError(
                f"Layer number mismatch: got {len(vl_embs_list)} VL layers, "
                f"but project_layers has {len(self.project_layers)} layers."
            )
        return [proj(vl_h) for proj, vl_h in zip(self.project_layers, vl_embs_list)]

    def _encode_vl_hidden_states(
        self,
        batch_images: List,
        instructions: List[str],
        collect_align_ctx: bool = False,
        collect_tactile_align_ctx: bool = False,
    ) -> tuple:
        """Run QwenVL, project hidden states, and return (layer-wise embeddings, attention_mask, align_ctx, qwen_inputs) for the Action DiT.

        ``align_ctx`` is None unless a train-only aligner needs it (DA3 and/or
        contact-region tactile_align). It carries the raw (un-projected) hidden
        state(s) of the supervised layer(s) plus the token-geometry tensors
        needed to slice out the visual tokens. ``qwen_inputs`` is returned so
        token pruning can locate visual tokens (`input_ids`) and derive
        per-view grids (`image_grid_thw`). The baseline compute path is
        untouched when both collect flags are false.
        """
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )
        attention_mask = qwen_inputs.get("attention_mask", None)
        query_hook = None
        if self.action_query_embed is not None:
            qwen_inputs, query_hook = self._inject_action_queries(qwen_inputs)
            # The DiT's cross-attn mask must cover the appended queries.
            attention_mask = qwen_inputs.get("attention_mask", None)
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                qwenvl_outputs = self.qwen_vl_interface(
                    **qwen_inputs,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
        finally:
            if query_hook is not None:
                query_hook["handle"].remove()
        # Check AFTER the finally so a VLM-side exception is never masked by
        # this guard's own raise.
        if query_hook is not None and not query_hook["fired"]:
            raise RuntimeError(
                "action_query embedding splice hook never fired — the "
                "queries would be the dummy token's embedding. Check "
                "that the VLM input-embedding layer path still resolves."
            )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vl_embs_list = list(qwenvl_outputs.hidden_states[-self.num_action_dit_layers:])
            if self.action_query_kv_mode == "query_only":
                # Both experts read ONLY the K query summaries; the raw V-L
                # tokens stay out of the DiT's cross-attn KV entirely, so the
                # per-layer projectors no longer back-propagate the action
                # loss into visual slices — the only action-gradient path
                # into the VLM runs through the query tokens (which attend
                # V+L internally). DA3 / heatmap alignment on visual
                # positions no longer fight direct action gradients.
                K = self.action_query_num
                vl_embs_list = [h[:, -K:, :] for h in vl_embs_list]
                attention_mask = None  # queries are never padding
            align_ctx = None
            if collect_align_ctx or collect_tactile_align_ctx:
                align_ctx = {
                    "input_ids": qwen_inputs["input_ids"],
                    # [num_images_in_batch, 3] = (t, h, w) in 16px patch units,
                    # BEFORE the 2x2 spatial merge.
                    "image_grid_thw": qwen_inputs.get("image_grid_thw", None),
                }
                if collect_align_ctx:
                    # Keep the autograd graph: the alignment gradient must flow
                    # back into the VLM through this tensor.
                    align_ctx["hidden_state"] = qwenvl_outputs.hidden_states[
                        self.spatial_align_layer
                    ]
                if collect_tactile_align_ctx:
                    align_ctx["tactile_align_hidden_state"] = qwenvl_outputs.hidden_states[
                        self.tactile_align_layer
                    ]
            vl_embs_list = self._project_vl_hidden_for_action(vl_embs_list)
        return vl_embs_list, attention_mask, align_ctx, qwen_inputs

    def _inject_action_queries(self, qwen_inputs: dict):
        """Append K action-query slots at the END of the Qwen sequence.

        The input_ids get K copies of a dummy token and the attention mask K
        ones; a forward hook on the VLM's input-embedding layer then splices
        the learnable query embeddings over the dummy rows. Appending (not
        vocab resize) keeps every pre-action-query checkpoint loadable, and
        the causal mask gives the V-L isolation for free: visual / language
        tokens never attend to the queries, so their hidden states — and the
        DA3 / heatmap alignment reading them — are bit-identical to a
        query-free forward. Left padding keeps the query block at ``[-K:]``
        for every row.

        Returns ``(qwen_inputs, hook_state)``; the caller must remove
        ``hook_state["handle"]`` after the forward and check
        ``hook_state["fired"]``.
        """
        K = self.action_query_num
        input_ids = qwen_inputs["input_ids"]
        B, L = input_ids.shape
        device = input_ids.device
        dummy = torch.full((B, K), self.action_query_dummy_id, dtype=input_ids.dtype, device=device)
        qwen_inputs["input_ids"] = torch.cat([input_ids, dummy], dim=1)
        mask = qwen_inputs.get("attention_mask", None)
        if mask is not None:
            qwen_inputs["attention_mask"] = torch.cat(
                [mask, torch.ones(B, K, dtype=mask.dtype, device=device)], dim=1
            )

        hook_state = {"fired": False, "handle": None}

        def _splice_hook(module, args, output):
            # output: [B, L+K, hidden] text embeddings (image embeds merge
            # later, at image token positions — the query block survives).
            hook_state["fired"] = True
            # Call the module (not `.weight`) so ZeRO-3 fetches the parameter
            # through its forward hooks even inside the VLM's forward.
            idx = torch.arange(K, device=output.device)
            query = self.action_query_embed(idx).to(dtype=output.dtype)
            output = output.clone()  # never write in-place into the VLM graph
            output[:, L:, :] = query.unsqueeze(0)
            return output

        embed_layer = self.qwen_vl_interface.model.model.get_input_embeddings()
        hook_state["handle"] = embed_layer.register_forward_hook(_splice_hook)
        return qwen_inputs, hook_state

    def _token_prune_layout_ok(
        self, qwen_inputs: dict, n_views_per_sample: Sequence[int]
    ) -> bool:
        """Can this batch's image grid be split per camera (mixed layouts OK)?

        Temporal grids (t != 1) and a missing / mismatched ``image_grid_thw``
        still skip. Mixed packed counts / resolutions are grouped inside
        ``_apply_token_pruning`` the same way DA3 groups DA3 loss.
        """
        grid = qwen_inputs.get("image_grid_thw", None)
        if grid is None or not n_views_per_sample:
            return False
        try:
            self.token_prune.tokens_per_view_per_sample(
                grid, n_views_per_sample, self.token_prune_merge_size
            )
        except (ValueError, AttributeError):
            return False
        return True

    def _apply_token_pruning(
        self,
        vl_embs_list: List[torch.Tensor],
        backbone_attention_mask: Optional[torch.Tensor],
        qwen_inputs: dict,
        n_views_per_sample: Sequence[int],
        collect_stats: bool = False,
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor], torch.Tensor, Optional[dict]]:
        """BFA++-style pruning of visual tokens across every DiT layer.

        Scores come from the PROJECTED hidden state of ``token_prune_layer``
        (so the scorer and the DiT read the same representation); the same
        keep mask is then applied to every layer, keeping the layer-wise
        cross-attention aligned. Real text / special tokens are never pruned,
        while padding positions are dropped along with the low-scoring patches.

        Mixed packed layouts (head-only + 3-view, or same mask / different
        ``obs_image_size``) are scored in one scorer forward, then top-k'd
        per layout group so the rectangular per-view split stays well-defined.

        Returns ``(pruned_layers, attention_mask, visual_keep_ratio, stats)``;
        the ratio is a detached scalar tensor for logging (no device sync) and
        ``stats`` carries the scorer diagnostics when ``collect_stats``.
        """
        input_ids = qwen_inputs["input_ids"]
        grid_thw = qwen_inputs.get("image_grid_thw", None)
        tokens_per_view = self.token_prune.tokens_per_view_per_sample(
            grid_thw, n_views_per_sample, self.token_prune_merge_size
        )
        visual_mask = input_ids == self.token_prune_image_token_id  # [B, L]
        counts = visual_mask.sum(dim=1)
        expected = torch.tensor(
            [sum(row) for row in tokens_per_view],
            device=counts.device,
            dtype=counts.dtype,
        )
        if not torch.equal(counts, expected):
            raise ValueError(
                f"token_prune expected visual tokens {expected.tolist()} "
                f"(per-sample views {tokens_per_view}), got {counts.tolist()}."
            )
        # Score on the projected scoring layer (index into the LAST
        # num_action_dit_layers hidden states: hidden_states[k] -> k-1).
        score_hidden = vl_embs_list[self.token_prune_layer - 1]
        scores = self.token_prune.score_tokens(score_hidden)
        keep_mask, stats = self.token_prune.build_keep_mask(
            scores,
            visual_mask,
            tokens_per_view,
            attention_mask=backbone_attention_mask,
            collect_stats=collect_stats,
        )
        # Gate only the visual tokens that survive: pruned ones are gone and
        # text tokens are not the scorer's business.
        gate_mask = keep_mask & visual_mask
        keep_ratio = (
            gate_mask.sum().float() / visual_mask.sum().clamp(min=1).float()
        ).detach()

        # One gather index and one scale tensor for all layers, so each layer
        # costs a single gather + multiply over the SHORTENED sequence.
        index, new_attention_mask = self.token_prune.build_gather_index(keep_mask)
        scale = self.token_prune.gather_scale(
            index,
            new_attention_mask,
            self.token_prune.score_gate_factor(scores, gate_mask),
            vl_embs_list[0].dtype,
        )
        pruned_layers = [
            self.token_prune.prune_with_index(hidden, index, new_attention_mask, scale)
            for hidden in vl_embs_list
        ]
        return pruned_layers, new_attention_mask, keep_ratio, stats

    def forward(
        self,
        examples: List[dict] = None,
        vlm_lm_inputs: Optional[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
            vlm_lm_inputs: Optional packed Qwen LM batch (``input_ids``,
                ``labels``, …). Co-train only. Must be consumed *inside*
                this forward so DeepSpeed ZeRO-3 all-gathers the partitioned
                Qwen weights; calling ``unwrap_model(...).qwen_vl_interface``
                outside the engine sees empty shards. The trainer scales
                the emitted ``vlm_lm_loss`` by ``loss_scale.vlm``.
        Returns:
            dict:
                action_loss (torch.Tensor): Scalar diffusion noise prediction loss.
        """
        batch_images = [example["image"] for example in examples]  # List[List[PIL.Image]], length B
        instructions = [example["lang"] for example in examples]  # List[str], length B
        actions = [example["action"] for example in examples]  # List[ndarray (T, action_dim)]
        state = (
            [example["state"] for example in examples] if "state" in examples[0] else None
        )  # List[ndarray (H_state, state_dim)] or None
        assert_packed_action_state(
            actions,
            state,
            action_horizon=self.action_horizon,
            action_dim=self.action_dim,
            state_dim=self.state_dim,
            state_history_frames=self.state_history_frames,
            require_state=self.continuous_state,
        )

        # Prepend discretised proprioceptive state to each instruction string.
        instructions = (
            self.add_discretized_state_to_instruction(instructions, state) if state is not None else instructions
        )
        if not self.continuous_state:
            state = None  # state reaches the model only via the instruction tokens

        # flexible_views: samples may carry different present-camera counts.
        # The VLM / action loss already handle that. DA3, token prune, and
        # tactile_align split mixed batches by packed image layout inside
        # their gather (sample-weighted). Trainable projectors / the scorer
        # still run exactly once, which keeps ZeRO-3 collectives
        # rank-invariant. Always collect the supervised hidden state when
        # the module is on.
        n_views_per_sample = [len(imgs) for imgs in batch_images]

        collect_align = self.spatial_align is not None
        # Always collect when the module is on (DA3 contract). A data-dependent
        # skip used to call ``_zero_tactile_align_outputs`` (param.sum) on
        # V-only ranks while others ran ``forward_groups`` — ZeRO-3 deadlock
        # on mixed V/VT + 1-view/3-view mixtures. Groups that cannot satisfy
        # align_view_indices still skip *inside* the loss; empty segments
        # still project once.
        collect_tactile_align = self.tactile_align is not None

        def _example_may_tactile_align(ex: dict) -> bool:
            if not isinstance(ex, dict):
                return False
            if ex.get("tactile_align_targets") is not None:
                return True
            if ex.get("tactile_heatmap") is not None or ex.get("tactile_2d") is not None:
                return True
            return bool(ex.get("present2d", False))

        # Step 1: encode through QwenVL
        vl_embs_list, backbone_attention_mask, align_ctx, qwen_inputs = self._encode_vl_hidden_states(
            batch_images,
            instructions,
            collect_align_ctx=collect_align,
            collect_tactile_align_ctx=collect_tactile_align,
        )
        base_hidden = vl_embs_list[-1]
        n_views_ok = n_views_per_sample
        # query_only: the DiT KV is the query slice — no visual tokens to
        # prune (construction already warned). Skip so the scorer's
        # visual-count assert cannot fire on the sliced sequence.
        prune_layout_ok = (
            self.token_prune is not None
            and self.action_query_kv_mode != "query_only"
            and self._token_prune_layout_ok(qwen_inputs, n_views_ok)
        )

        # Step 1.5 (optional): spatial alignment loss on the raw hidden states
        # of the supervised layer. Computed ONCE per VLM forward — deliberately
        # BEFORE the repeated_diffusion_steps repeat below, so its cost does
        # not scale with the diffusion-noise repeat factor.
        align_loss, align_cosine = None, None
        if self.spatial_align is not None:
            align_cosine, aligned_samples = self._compute_spatial_align_loss(
                align_ctx, batch_images
            )
            if aligned_samples == 0 and not getattr(self, "_warned_flex_align_skip", False):
                print(
                    "[starVLA][WARN] spatial_alignment aligned no sample on this batch "
                    "(no layout group could satisfy align_view_indices); emitting zero "
                    "align_loss so multi-GPU metric reduce stays collective-safe."
                )
                self._warned_flex_align_skip = True
            align_loss = self.spatial_align.loss_coeff * align_cosine

        # Step 1.6 (optional): contact-region → visual-token alignment. Also
        # once per VLM forward, on the full token set (before prune). Mixed
        # layouts are grouped like DA3; rows without present2d / 2D targets
        # contribute zero via the module's own skip path. Always
        # ``_compute_tactile_align_loss`` → one ``forward_groups`` (empty
        # segments still project) so ZeRO-3 collectives stay rank-invariant.
        contact_align_loss = None
        contact_align_metrics = None
        if self.tactile_align is not None:
            contact_align_metrics = self._compute_tactile_align_loss(
                align_ctx, examples, batch_images
            )
            contact_align_loss = contact_align_metrics["loss"]
            if (
                not any(_example_may_tactile_align(ex) for ex in examples)
                and not getattr(self, "_warned_tactile_align_skip", False)
            ):
                print(
                    "[starVLA][WARN] tactile_align has no present2d / tactile_2d "
                    "/ tactile_align_targets in this batch; emitting zero "
                    "contact_align_loss via forward_groups so ZeRO-3 / metric "
                    "reduce stay collective-safe."
                )
                self._warned_tactile_align_skip = True

        # Step 1.75 (optional): BFA++ visual token pruning. Runs AFTER the DA3
        # alignment loss so the teacher still supervises the full token set,
        # and BEFORE the repeated_diffusion_steps repeat so its cost does not
        # scale with the diffusion-noise repeat factor. Pruning rewrites the
        # DiT encoder sequence only; the action loss back-propagates through
        # the kept tokens into the scorer. Mixed layouts are grouped the same
        # way as DA3 (one scorer forward, per-layout top-k). Temporal / missing
        # grids still skip rather than failing.
        #
        # ``prune_keep_ratio`` is emitted whenever pruning is CONFIGURED (1.0
        # when a batch is skipped), never conditionally on the data: the
        # trainer reduces metric dicts across ranks, so a key that appears on
        # some ranks only would desynchronise the collective.
        prune_keep_ratio = None
        prune_param_touch = None
        prune_stats = None
        if self.token_prune is not None:
            if prune_layout_ok:
                (
                    vl_embs_list,
                    backbone_attention_mask,
                    prune_keep_ratio,
                    prune_stats,
                ) = self._apply_token_pruning(
                    vl_embs_list,
                    backbone_attention_mask,
                    qwen_inputs,
                    n_views_ok,
                    collect_stats=True,
                )
            else:
                prune_keep_ratio = torch.ones((), device=base_hidden.device)
                # Same keys on every rank (see the note above): a skipped batch
                # reports "no exploration divergence, no score spread yet".
                prune_stats = {
                    "score_std": torch.zeros((), device=base_hidden.device),
                    "score_mean": torch.zeros((), device=base_hidden.device),
                    "topk_agreement": torch.ones((), device=base_hidden.device),
                }
                # The scorer would otherwise receive no gradient on this rank
                # while other ranks train it, which desynchronises DDP/ZeRO's
                # parameter set (see _zero_token_prune_touch).
                prune_param_touch = self._zero_token_prune_touch(
                    device=base_hidden.device, dtype=torch.float32
                )
                if not getattr(self, "_warned_prune_layout_skip", False):
                    print(
                        "[starVLA][WARN] token_prune skipped on a batch with "
                        "temporal or missing image_grid_thw; the action loss "
                        "still runs. Mixed packed layouts are grouped, not skipped."
                    )
                    self._warned_prune_layout_skip = True

        # Step 1.9 (optional): append tactile tokens to the DiT context. AFTER
        # pruning (which ranks VISUAL tokens only) and BEFORE the
        # repeated_diffusion_steps repeat, so the repeat below carries the
        # tactile slots and the extended mask for free.
        vl_embs_list, backbone_attention_mask, tactile_gate = (
            self._append_tactile_context(
                vl_embs_list, backbone_attention_mask, examples
            )
        )

        # Step 1.95 (optional): future-contact variant A. After the encoder
        # ran (fusion path stashes tokens; encoder-only encodes here) and
        # before the diffusion repeat. Train-only; predict_action skips it.
        future_contact_loss = None
        future_contact_metrics = None
        if self.tactile_future_contact is not None:
            future_contact_metrics = self._compute_future_contact_loss(examples)
            future_contact_loss = future_contact_metrics["loss"]

        # Step 2: compute flow-matching loss over the action chunk
        with torch.autocast("cuda", dtype=torch.float32):
            # Keep the regression targets in float32: casting to
            # base_hidden.dtype (bf16 under DeepSpeed bf16 training) put the
            # flow-matching targets on the bf16 grid (~4e-3 resolution in
            # [-1, 1]) for no benefit. The action head casts its DiT inputs
            # to the module dtype itself and computes noise/velocity in fp32.
            # T == action_horizon is asserted above; do not silently slice.
            actions_target = torch.tensor(
                np.array(actions), device=base_hidden.device, dtype=torch.float32
            )  # [B, action_horizon, action_dim]

            # Resolution order: explicit trainer.repeated_diffusion_steps, else
            # framework.action_model.repeated_diffusion_steps (class default 2).
            # Upstream read only the trainer key with a silent default of 16,
            # which multiplies the DiT batch (and 36 layers of repeated VL
            # hidden states) 16x and OOMs 80GB GPUs with horizon-50 chunks.
            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", None)
                if self.config and self.config.trainer
                else None
            )
            if repeated_diffusion_steps is None:
                repeated_diffusion_steps = self.config.framework.action_model.get("repeated_diffusion_steps", 2)
            repeated_diffusion_steps = int(repeated_diffusion_steps)

            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            # Repeat every VLM layer embedding to match the duplicated action batch.
            vl_embs_list_repeated = [h.repeat(repeated_diffusion_steps, 1, 1) for h in vl_embs_list]
            if backbone_attention_mask is not None:
                backbone_attention_mask = backbone_attention_mask.repeat(repeated_diffusion_steps, 1).to(
                    dtype=torch.bool
                )

            state_repeated = None
            if state is not None:
                state = torch.tensor(np.array(state), device=base_hidden.device, dtype=base_hidden.dtype)
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            extra_temb = self._take_tactile_extra_temb()
            if extra_temb is not None:
                extra_temb = extra_temb.repeat(repeated_diffusion_steps, 1)

            action_kwargs = {}
            if extra_temb is not None:
                action_kwargs["extra_temb"] = extra_temb
            joint_cond = self._take_tactile_joint_cond()
            if joint_cond is not None:
                # The batch is repeated for repeated_diffusion_steps; the
                # stream tokens repeat with it (gates are scalars, shared).
                # `present` is per-ROW, so it has to repeat too or the head's
                # shortcut mask sees a [B] flag against a [B*reps] batch.
                joint_present = joint_cond.get("present", None)
                joint_cond = {
                    **joint_cond,
                    "tokens": joint_cond["tokens"].repeat(repeated_diffusion_steps, 1, 1),
                    "present": (
                        None
                        if joint_present is None
                        else joint_present.repeat(repeated_diffusion_steps)
                    ),
                }
                action_kwargs["joint_cond"] = joint_cond
            action_loss = self.action_model(
                vl_embs_list_repeated,
                actions_target_repeated,
                state_repeated,
                encoder_attention_mask=backbone_attention_mask,
                **action_kwargs,
            )

        if prune_param_touch is not None:
            action_loss = action_loss + prune_param_touch.to(action_loss.dtype)

        # tactile_encoder exists but action_fusion is off: the encoder gets no
        # gradient path, so keep its param set in the backward graph for
        # DDP/ZeRO safety. With fusion on, the encoder is in the real graph
        # (the gate zeroes the VALUE, not the autograd edge).
        if (
            self.tactile_encoder is not None
            and self.tactile_fusion is None
            and self.tactile_future_contact is None
        ):
            action_loss = action_loss + self._zero_tactile_fusion_touch(
                device=action_loss.device, dtype=action_loss.dtype
            )

        output = {"action_loss": action_loss}
        if self.tactile_fusion is not None and self.tactile_fusion.gate_l1 > 0.0:
            gate_pen = self.tactile_fusion.gate_l1_penalty()
            action_loss = action_loss + gate_pen.to(dtype=action_loss.dtype)
            output["action_loss"] = action_loss
            output["tactile_gate_l1"] = gate_pen.detach().float()
        if tactile_gate is not None:
            # Learned SWITCH, not a mixing weight: under joint_stream the DiT
            # LayerNorms the stream tokens, so anything above
            # |gate|*std ~ sqrt(norm_eps) = 3.2e-3 is full amplitude and the
            # number stops moving. Read it as "did this path leave the eps
            # hole". A zero-mean oscillation well under the knee means the
            # stream is OFF — check `gate_l1` first (it pins the gate), then
            # `joint_gate_mode` (legacy `bias` cannot move at all in bf16).
            output["tactile_gate"] = tactile_gate.detach().float()
        if (
            self.tactile_fusion is not None
            and getattr(self.tactile_fusion, "patch_gate", None) is not None
        ):
            output["tactile_patch_gate"] = (
                self.tactile_fusion.patch_gate.detach().float()
            )
        if (
            self.tactile_fusion is not None
            and getattr(self.tactile_fusion, "joint_a2t_gate", None) is not None
        ):
            # Joint-stream health (plan §5): all three gates flat at 0 means
            # the two experts never opened to each other.
            output["tactile_joint_a2t_gate"] = (
                self.tactile_fusion.joint_a2t_gate.detach().float()
            )
        if (
            self.tactile_fusion is not None
            and getattr(self.tactile_fusion, "patch_mlp", None) is not None
        ):
            # THE openness metric for the region FiLM path: the gate scalars
            # cannot report it (this path is ungated, and the token path is
            # scale-invariant). Emitted whenever `patch: temb` is CONFIGURED —
            # a config property, identical on every rank, so the trainer's
            # metric reduce stays collective even if a rank saw no glove.
            rel_norm = getattr(self.action_model.model, "_extra_temb_rel_norm", None)
            output["tactile_extra_temb_rel_norm"] = (
                torch.zeros((), device=action_loss.device)
                if rel_norm is None
                else rel_norm.detach().float()
            )
        if prune_keep_ratio is not None:
            # Fraction of VISUAL tokens that reached the DiT (1.0 = no pruning
            # happened on this batch). Emitted on every rank whenever pruning
            # is configured, so the trainer's metric reduce stays collective.
            output["prune_keep_ratio"] = prune_keep_ratio.detach().float()
        if prune_stats is not None:
            # Scorer health, same collective contract. `prune_score_std` stuck
            # near its 1e-3 init and `prune_topk_agreement` stuck near
            # `beta` both mean the ranking is still noise, i.e. the run is
            # doing random token dropout rather than learned pruning.
            # `prune_score_mean` drifting to large |values| means the sigmoid
            # gate has saturated and the scorer has stopped learning.
            output["prune_score_std"] = prune_stats["score_std"].detach().float()
            output["prune_score_mean"] = prune_stats["score_mean"].detach().float()
            output["prune_topk_agreement"] = prune_stats["topk_agreement"].detach().float()
        if align_loss is not None:
            # Already weighted by loss_coeff; the trainer simply adds it to
            # the total. Baseline runs never see this key. `align_cosine` is
            # the unweighted `1 - cos` distance, reported separately so the
            # logs stay comparable across loss_coeff sweeps.
            output["align_loss"] = align_loss
            output["align_cosine"] = align_cosine.detach()
        if contact_align_loss is not None:
            # Already coeff-weighted inside ContactRegionAlignment. Baseline
            # runs (enable=false) never emit this key.
            output["contact_align_loss"] = contact_align_loss
            if contact_align_metrics is not None:
                for key in (
                    "contact_map_loss",
                    "dice_loss",
                    "infonce_loss",
                    "pressure_bin_loss",
                    "contact_align_present_frac",
                ):
                    if key in contact_align_metrics:
                        output[key] = contact_align_metrics[key].detach()
        if future_contact_loss is not None:
            # Already coeff-weighted inside FutureContactHead. Off → no key.
            output["future_contact_loss"] = future_contact_loss
            if future_contact_metrics is not None:
                for key in self._FUTURE_CONTACT_METRIC_KEYS:
                    if key in future_contact_metrics:
                        output[key] = future_contact_metrics[key].detach()
        if vlm_lm_inputs is not None:
            # Second Qwen forward on the VLM-only batch. Same module as the
            # VLA encode above, so ZeRO-3's per-module gather fires again
            # with a rank-invariant call count (every rank that entered
            # this forward takes this branch).
            vlm_output = self.qwen_vl_interface(**vlm_lm_inputs)
            if getattr(vlm_output, "loss", None) is None:
                raise ValueError(
                    "co-train vlm_lm_inputs produced no LM loss; the VLM "
                    "batch must include labels."
                )
            output["vlm_lm_loss"] = vlm_output.loss
        return output

    def _prepare_action_inference_inputs(self, examples: List[dict]):
        """Shared observation preprocessing for ``predict_action`` and
        ``predict_action_realtime``.

        Single source of truth for the inference-time observation contract so
        the RTC path cannot drift from the plain path (the v2 wrapper did):
          1. Discretised ``[STATE]`` prompt injection (always, when state is
             present); the raw normalized state is kept only when
             ``continuous_state`` is on.
          2. Re-resize to the training observation resolution via
             ``resolve_predict_resize_targets`` (uniform or per-camera).
          3. Encode through QwenVL and cast the backbone mask to bool.

        Returns:
            (vl_embs_list, backbone_attention_mask, state_tensor_or_None)
        """
        batch_images = [to_pil_preserve(example["image"]) for example in examples]  # List[List[PIL.Image]]
        instructions = [example["lang"] for example in examples]  # List[str]
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # List[ndarray] or None
        if examples:
            assert_packed_states(
                state,
                batch_size=len(examples),
                state_dim=self.state_dim,
                state_history_frames=self.state_history_frames,
                require_state=self.continuous_state,
            )

        # Encode proprioceptive state into the instruction string; the raw
        # normalized state is kept only when continuous_state is on.
        instructions = (
            self.add_discretized_state_to_instruction(instructions, state) if state is not None else instructions
        )
        if not self.continuous_state:
            state = None

        # Optionally resize images to the resolution used during training.
        # `image_size` is the alias the dataloader also honours, so a config
        # written with it still re-resizes here instead of silently keeping
        # whatever resolution the eval client happened to send. Per-camera
        # mappings expand in embodiment video_keys order (same as the
        # dataloader / deploy client). With flexible_views, each example may
        # carry its own ``view_mask`` (multi-env batch with different optional
        # wrists) — resize per sample so head-only requests still get the
        # head size rather than a wrongly truncated catalog prefix.
        if examples:
            resized = []
            for example, images in zip(examples, batch_images):
                view_mask = example.get("view_mask") if isinstance(example, dict) else None
                n_views = len(images)
                target_sizes = resolve_predict_resize_targets(
                    self.config, n_views, view_mask=view_mask
                )
                if target_sizes:
                    resized.append(resize_images(images, target_size=target_sizes))
                else:
                    resized.append(images)
            batch_images = resized
        else:
            n_views = len(batch_images[0]) if batch_images else 0
            target_sizes = resolve_predict_resize_targets(self.config, n_views)
            if target_sizes:
                batch_images = resize_images(batch_images, target_size=target_sizes)

        # Encode through QwenVL (spatial alignment is train-only, so no align
        # context is collected here and DA3 is never loaded).
        vl_embs_list, backbone_attention_mask, _, qwen_inputs = self._encode_vl_hidden_states(batch_images, instructions)
        base_hidden = vl_embs_list[-1]
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)

        # Optional BFA++ token pruning (inference uses the same scorer the
        # action loss trained; no teacher / annotation is involved). Mixed
        # per-sample layouts (multi-env flexible_views) are grouped the same
        # way as training. Temporal / missing grids skip rather than failing.
        if self.token_prune is not None and self.action_query_kv_mode != "query_only":
            n_views_per_sample = [len(imgs) for imgs in batch_images]
            if self._token_prune_layout_ok(qwen_inputs, n_views_per_sample):
                vl_embs_list, backbone_attention_mask, _, _ = self._apply_token_pruning(
                    vl_embs_list,
                    backbone_attention_mask,
                    qwen_inputs,
                    n_views_per_sample,
                )
            elif not getattr(self, "_warned_prune_layout_skip", False):
                print(
                    "[starVLA][WARN] token_prune skipped: this batch has a "
                    "temporal or missing image_grid_thw; running unpruned "
                    "for this call. Mixed packed layouts are grouped, not skipped."
                )
                self._warned_prune_layout_skip = True

        # Same fusion the action loss trained on, in the same position (after
        # pruning). Modality dropout is forced off: dropping the glove at
        # deploy time would be pure noise. A request without tactile keys
        # resolves to the learned no_tactile token, i.e. graceful V-only.
        vl_embs_list, backbone_attention_mask, _ = self._append_tactile_context(
            vl_embs_list,
            backbone_attention_mask,
            examples,
            apply_modality_dropout=False,
        )

        state = (
            torch.from_numpy(np.array(state)).to(base_hidden.device, dtype=base_hidden.dtype)
            if state is not None
            else None
        )
        return vl_embs_list, backbone_attention_mask, state

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        Run inference and return a denoised action trajectory.

        Steps:
          1. Optionally resize images to the training observation resolution.
          2. Encode images + instruction (with discretised state prefix) through QwenVL.
          3. Project layer-wise VLM hidden states to the Action DiT latent space.
          4. Run the flow-matching sampler to produce the action chunk.

        Args:
            examples: List[dict], each entry requires:
                - image: List[PIL.Image] (multi-view)
                - lang:  str instruction
                - state: np.ndarray shaped (1, state_dim), optional

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape (B, action_horizon, action_dim),
                    denoised actions in the normalised action space.
        """
        vl_embs_list, backbone_attention_mask, state = self._prepare_action_inference_inputs(examples)

        # Run the flow-matching sampler to produce the denoised action chunk.
        extra_temb = self._take_tactile_extra_temb()
        predict_kwargs = {}
        if extra_temb is not None:
            predict_kwargs["extra_temb"] = extra_temb
        joint_cond = self._take_tactile_joint_cond()
        if joint_cond is not None:
            predict_kwargs["joint_cond"] = joint_cond
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                vl_embs_list,
                state,
                encoder_attention_mask=backbone_attention_mask,
                **predict_kwargs,
            )  # (B, action_horizon, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    @torch.no_grad()
    def predict_action_realtime(
        self,
        examples: List[dict] = None,
        prev_action_chunk_normalized: Optional[np.ndarray] = None,
        inference_delay: Optional[int] = None,
        **kwargs,
    ) -> dict:
        """RTC-aware inference: real-time chunking (arXiv:2506.07339).

        Guides the new chunk with the previous chunk's un-executed tail so
        chunks splice smoothly under inference latency (ΠGDM by default — no
        retraining needed). ALL observation preprocessing is shared with
        ``predict_action`` via ``_prepare_action_inference_inputs``, so the
        only difference from the plain path is the guided sampling itself.

        Gated by ``framework.action_model.rtc.enable`` (default False): when
        disabled — or when no usable previous chunk / delay is supplied —
        this IS ``predict_action`` (identical code path, identical output).

        Batched parallel-env RTC: one call guides a whole batch, with
        PER-SAMPLE leftovers and delays (envs fire with different measured
        latencies and ragged leftover lengths). A sample whose leftover is
        empty or whose delay is non-positive gets an all-zero weight schedule
        — unguided sampling for that sample alone; only when NO sample is
        guided does the whole call fall back to ``predict_action``.

        NOTE: decorated with ``torch.no_grad()``, NOT ``torch.inference_mode()``:
        ΠGDM guidance re-enables grad locally for the vector-Jacobian product,
        and inference-mode tensors cannot require grad (the v2 wrapper is
        broken in exactly this way).

        Args:
            examples: same shape as ``predict_action``.
            prev_action_chunk_normalized: normalized actions = the previous
                chunk's un-executed tail ``prev[consumed:]`` at inference
                start (the same convention as lerobot's
                ``ActionQueue.get_left_over`` and the Kinetix reference).
                Either a uniform (B, T_prev, action_dim) / (T_prev,
                action_dim) array, or a RAGGED list of B per-sample
                (T_prev_i, action_dim) arrays (batched fires). Rows beyond a
                sample's own T_prev are zero-padded and MUST land in the
                zero-weight region — see below. The caller owns the frame
                contract: these rows must be expressed in the SAME space this
                call predicts in, i.e. re-encoded against THIS observation's
                reference pose and normalized with row indices starting at 0.
                Representations that are relative to a reference pose
                (``delta_xyz_rot6d``) or normalized per horizon row
                (``hold_centered_perrow``) are not portable across chunks;
                ``PolicyServerWrapper`` prepares them.
            inference_delay: rows guaranteed to execute from the previous
                chunk while inference runs (guidance weight 1) — a scalar, or
                one entry per batch sample. ``None`` falls back to
                ``rtc.inference_delay`` from the config; the deploy client
                passes its measured latency each round.

        Config (``framework.action_model.rtc``):
            enable: master switch (default False).
            inference_delay: fallback frozen-prefix length (default 4).
            execution_horizon: cap on how many leading rows attend to the
                previous chunk. Default None = the whole leftover (the paper's
                ``end = H - consumed`` convention: weights hit zero exactly at
                the zero-padded rows).
            prefix_attention_schedule: exp | linear | ones | zeros.
            max_guidance_weight: ΠGDM guidance-weight cap (default 10.0).
            mode: pigdm (training-free) | simulated_delay (requires the
                simulated-delay training loss, not implemented here).

        Returns:
            dict with ``normalized_actions`` -> (B, action_horizon, action_dim).
        """
        rtc_cfg = self.config.framework.action_model.get("rtc", None)
        if rtc_cfg is None or not bool(rtc_cfg.get("enable", False)):
            return self.predict_action(examples)

        if prev_action_chunk_normalized is None:
            return self.predict_action(examples)

        batch_size = len(examples)

        # Per-sample leftovers: a ragged list of (T_i, D) arrays keeps each
        # env's own leftover length; a uniform array broadcasts its T_prev.
        if isinstance(prev_action_chunk_normalized, (list, tuple)):
            prev_rows = [np.asarray(p, dtype=np.float32) for p in prev_action_chunk_normalized]
        else:
            prev = np.asarray(prev_action_chunk_normalized, dtype=np.float32)
            if prev.ndim == 2:
                prev = prev[None]
            if prev.ndim != 3:
                raise ValueError(
                    f"prev_action_chunk_normalized must have shape (B, T_prev, action_dim), "
                    f"got {prev.shape}."
                )
            prev_rows = [prev[b] for b in range(prev.shape[0])]
        if len(prev_rows) != batch_size:
            raise ValueError(
                f"prev_action_chunk_normalized must carry one leftover per example "
                f"(B={batch_size}), got {len(prev_rows)}."
            )
        for rows in prev_rows:
            if rows.ndim != 2 and rows.size > 0:
                raise ValueError(
                    f"each leftover must be (T_prev_i, action_dim), got {rows.shape}."
                )
        # Never attend beyond the new chunk's own length.
        leftover_lens = [min(int(rows.shape[0]), self.action_horizon) for rows in prev_rows]

        # Per-sample delays: a scalar broadcasts; a list carries each env's
        # own measured latency.
        delay_fallback = rtc_cfg.get("inference_delay", None)
        if inference_delay is None:
            delays = [int(delay_fallback) if delay_fallback is not None else 0] * batch_size
        elif isinstance(inference_delay, (list, tuple, np.ndarray)):
            delays = [int(d) for d in inference_delay]
            if len(delays) != batch_size:
                raise ValueError(
                    f"inference_delay must be a scalar or carry one entry per example "
                    f"(B={batch_size}), got {len(delays)}."
                )
        else:
            delays = [int(inference_delay)] * batch_size

        # Per-sample weight-schedule boundaries (paper convention): the
        # weights decay to zero at `end`, so the zero-padded rows of a short
        # leftover must sit at indices >= end. With the default
        # execution_horizon=None the whole leftover is attended and the
        # padding rows always get zero weight. Computed BEFORE the VLM
        # encoding so the no-guidance fallback does not pay for an encode it
        # throws away. An unguided sample (empty leftover / delay <= 0) gets
        # end=0 — an all-zero schedule, i.e. plain sampling for that sample.
        execution_horizon = rtc_cfg.get("execution_horizon", None)
        bounds = [
            (
                _resolve_rtc_weight_bounds(
                    self.action_horizon,
                    leftover_len=leftover_len,
                    inference_delay=delay,
                    execution_horizon=execution_horizon,
                )
                if delay > 0
                else None
            )
            for leftover_len, delay in zip(leftover_lens, delays)
        ]
        if all(b is None for b in bounds):
            return self.predict_action(examples)
        delay_effects = [b[0] if b is not None else 0 for b in bounds]
        suffix_lengths = [b[1] if b is not None else self.action_horizon for b in bounds]

        vl_embs_list, backbone_attention_mask, state = self._prepare_action_inference_inputs(examples)

        # Zero-pad ragged leftovers to a uniform tensor; rows past a sample's
        # own leftover length sit at indices >= its end and get zero weight.
        t_max = max(leftover_lens)
        leftover_dims = {
            int(rows.shape[1])
            for rows, length in zip(prev_rows, leftover_lens)
            if length > 0
        }
        if len(leftover_dims) != 1:
            raise ValueError(
                f"leftover action_dim must be the same on every guided sample, "
                f"got {sorted(leftover_dims)}."
            )
        action_dim = leftover_dims.pop()
        expected_dim = getattr(self, "action_dim", None)
        if expected_dim is not None and action_dim != int(expected_dim):
            raise ValueError(
                f"leftover action_dim {action_dim} != model action_dim {int(expected_dim)}."
            )
        prev_padded = np.zeros((batch_size, t_max, action_dim), dtype=np.float32)
        for b, (rows, length) in enumerate(zip(prev_rows, leftover_lens)):
            if length > 0:
                prev_padded[b, :length] = rows[:length]

        prev_chunk_t = torch.from_numpy(prev_padded).to(vl_embs_list[-1].device)
        extra_temb = self._take_tactile_extra_temb()
        rtc_kwargs = {}
        if extra_temb is not None:
            rtc_kwargs["extra_temb"] = extra_temb
        joint_cond = self._take_tactile_joint_cond()
        if joint_cond is not None:
            rtc_kwargs["joint_cond"] = joint_cond
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action_realtime(
                vl_embs_list,
                state,
                prev_action_chunk=prev_chunk_t,
                inference_delay=delay_effects,
                mode=str(rtc_cfg.get("mode", "pigdm")),
                suffix_length=suffix_lengths,
                prefix_attention_schedule=str(rtc_cfg.get("prefix_attention_schedule", "exp")),
                max_guidance_weight=float(rtc_cfg.get("max_guidance_weight", 10.0)),
                encoder_attention_mask=backbone_attention_mask,
                **rtc_kwargs,
            )  # (B, action_horizon, action_dim)

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    def state2str_transform(self, state: np.ndarray) -> str:
        """Quantise a state vector into 256 uniform bins and return it as a space-separated token string.

        Follows the π₀.5 convention: bins span [-1, 1] uniformly.
        Example: [-0.5, 0.1, 0.8] -> "95 133 203"
        """
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        return " ".join(map(str, discretized_state))

    def add_discretized_state_to_instruction(self, instructions: List[str], states: List[np.ndarray]) -> List[str]:
        """Append discretised proprioceptive state tokens to each instruction.

        Format: ``<original instruction> [STATE] <bin indices> [ACTION]``
        This lets the VLM attend to the robot state purely through its
        existing text-token pathway — no extra encoder required.
        """
        updated_instructions = []
        for instr, state in zip(instructions, states):
            # state is (T, state_dim); with state history (T > 1) the CURRENT
            # frame is the last row. T == 1 keeps the legacy behaviour.
            state_str = self.state2str_transform(np.asarray(state)[-1])
            updated_instructions.append(f"{instr} [STATE] {state_str} [ACTION]")
        return updated_instructions


if __name__ == "__main__":
    import argparse

    import debugpy
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/simBenchmarks/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()
    import os

    if os.getenv("DEBUG_MODE", "0") == "1":
        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()
    args.config_yaml = "examples/simBenchmarks/LIBERO/train_files/starvla_cotrain_libero.yaml"
    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"

    model = Qwen_PI_v3(cfg)
    # ckpt="/mnt/petrelfs/yejinhui/Projects/llavavla/results/Checkpoints/1011_qwenpi/checkpoints/need_steps_10000_pytorch_model.pt"
    # model = Qwen_PI.from_pretrained(ckpt)
    print(model)

    def print_model_size(m: nn.Module, depth: int = 1):
        """Print parameter counts for each top-level submodule (depth=1)."""
        total = sum(p.numel() for p in m.parameters())
        print(f"\n{'='*55}")
        print(f"{'Module':<35} {'Params':>12}  {'%':>6}")
        print(f"{'-'*55}")
        for name, child in m.named_children():
            n = sum(p.numel() for p in child.parameters())
            print(f"  {name:<33} {n:>12,}  {100*n/total:>5.1f}%")
        print(f"{'-'*55}")
        print(f"  {'TOTAL':<33} {total:>12,}  100.0%")
        print(f"{'='*55}\n")

    print_model_size(model, depth=1)

    # fake sample
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),  # action_chunk, action_dim
        "image": [image, image],  # two views
        "lang": "This is a fake instruction for testing.",
        "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16),  # chunk, state_dim
    }

    batch = [sample, sample]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output["action_loss"]
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action([sample])
    normalized_actions = predict_output["normalized_actions"]
    print(f"Unnormalized Action: {normalized_actions}")

    # # # Advance: try forward model with dataloader
    # # # can be fake sample, but here get from dataloader for simpler
    # from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    # vla_dataset_cfg = cfg.datasets.vla_data
    # vla_dataset_cfg.include_state = True

    # dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    # from torch.utils.data import DataLoader

    # train_dataloader = DataLoader(
    #     dataset,
    #     batch_size=2,
    #     num_workers=1,  # For Debug
    #     collate_fn=collate_fn,
    # )
    # #
    # for batch in tqdm(train_dataloader, desc="Processing Batches"):
    #     batch
    #     break

    # # try get model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # model(batch)

    # action = model.predict_action(batch)
