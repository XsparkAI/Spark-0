# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Jinhui YE/ HKUST] in [2026].
# Modification: [rm and add some connect adapter to match with starVLA, e.g., "rm "].


import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from starVLA.model.modules.action_model.flow_matching_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)
from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT

# TODO try to meger DiT Modules with follow_match_head, they are just the same arch, but diff loss, use diffusers package will be simple


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        # import ipdb; ipdb.set_trace()
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, output_dim=2048):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size=1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,) so we can replicate across T.")

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.layer1(actions)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then layer2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.layer2(x))

        # 5) Finally W3 => (B, T, w)
        x = self.layer3(x)
        return x


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size=1024, num_embodiments=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,) so we can replicate across T.")

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(default=True, metadata={"help": "Whether to add positional embedding"})
    diffusion_model_cfg: dict = field(default=None, metadata={"help": "Diffusion model configuration."})
    input_embedding_dim: int = field(default=1536, metadata={"help": "Input embedding channel dimension."})

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(default=0.999, metadata={"help": "Flow matching noise Beta distribution s."})
    num_timestep_buckets: int = field(default=1000, metadata={"help": "Number of timestep discretization buckets."})
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(default=True, metadata={"help": "Whether to tune the diffusion model."})
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(default=32, metadata={"help": "Number of target vision tokens."})
    action_loss_weight: dict = field(
        default=None,
        metadata={
            "help": (
                "Per-element weights for the flow-matching MSE "
                "{horizon_power, exec_rows, exec_weight, dims}; "
                "None/all-default = plain mean (see build_action_loss_weight)."
            )
        },
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


DiTConfig = {
    "num_layers": 36,
    "input_embedding_dim": 2048,
    "attention_head_dim": 64,
    "num_attention_heads": 32,
}  # default for qwen2.5-vl


def build_action_loss_weight(spec, action_horizon: int, action_dim: int):
    """``(1, H, D)`` weights for the flow-matching MSE, or ``None`` for uniform.

    The plain ``.mean()`` splits the gradient budget by how large each target
    happens to be. Two known imbalances, both configurable and both disabled by
    default (``None`` return == the original ``.mean()`` call):

    ``horizon_power``
        Chunk-wise delta targets grow with the row index (the near rows are a
        fraction of a step of motion, the far rows a whole chunk), so under one
        pooled q01/q99 the reducible loss of the last row is orders of
        magnitude above the first — yet only the first ``execute_horizon`` rows
        are ever sent to the robot. ``w_h = (h + 1) ** -horizon_power``.
    ``exec_rows`` / ``exec_weight``
        Multiply the prefix the deployment actually executes before it
        re-infers, independent of the power law.
    ``dims``
        Per-dimension weights (train.sh expands the per-key map). Without them
        a 58-d dual-arm vector spends 40/58 of the budget on the two hand
        blocks and under a third on the arm poses that carry the EE trajectory.

    Weights are rescaled to mean 1, so the loss magnitude (and therefore the
    effective learning rate) does not change when the shape does.
    """
    if spec is None:
        return None
    get = spec.get if hasattr(spec, "get") else (lambda k, d=None: getattr(spec, k, d))

    def _num(key, default):
        value = get(key, default)
        return default if value is None else value

    horizon_power = float(_num("horizon_power", 0.0))
    exec_rows = int(_num("exec_rows", 0))
    exec_weight = float(_num("exec_weight", 1.0))
    dims = get("dims", None)

    if exec_rows < 0 or exec_rows > action_horizon:
        raise ValueError(
            f"action_loss_weight.exec_rows must be in [0, {action_horizon}], got {exec_rows}"
        )
    if exec_weight <= 0:
        raise ValueError(f"action_loss_weight.exec_weight must be > 0, got {exec_weight}")

    row_w = torch.ones(action_horizon, dtype=torch.float32)
    if horizon_power:
        row_w = (torch.arange(action_horizon, dtype=torch.float32) + 1.0) ** (-horizon_power)
    if exec_rows and exec_weight != 1.0:
        row_w = row_w.clone()
        row_w[:exec_rows] *= exec_weight

    if dims is None:
        dim_w = torch.ones(action_dim, dtype=torch.float32)
    else:
        dim_w = torch.tensor([float(v) for v in dims], dtype=torch.float32)
        if dim_w.numel() != action_dim:
            raise ValueError(
                f"action_loss_weight.dims has {dim_w.numel()} entries, expected {action_dim}"
            )
        if torch.any(dim_w <= 0):
            raise ValueError("action_loss_weight.dims entries must be > 0")

    weight = row_w[:, None] * dim_w[None, :]
    if torch.allclose(weight, torch.ones_like(weight)):
        return None
    return (weight / weight.mean())[None]


def get_rtc_prefix_weights(start: int, end: int, total: int, schedule: str, device=None) -> torch.Tensor:
    """ΠGDM prefix-attention weight schedule across the action horizon.

    Matches Physical Intelligence's reference ``get_prefix_weights``
    (real-time-chunking-kinetix/src/model.py) and lerobot's
    ``RTCProcessor.get_prefix_weights`` bit-for-bit:

      - weights are 1 on ``[0, start)`` (the guaranteed-executed prefix),
      - decay on ``[start, end)`` per the schedule (``ones``/``linear``/``exp``),
      - 0 on ``[end, total)`` (the zero-padded / free-generation region).

    ``zeros`` collapses to a hard prefix mask (1 on ``[0, start)`` only);
    ``ones`` attends uniformly up to ``end``. ``end`` takes precedence over
    ``start`` (``start`` is clamped down to ``end``).
    """
    start = min(start, end)
    positions = torch.arange(total, device=device, dtype=torch.float32)
    if schedule == "ones":
        w = torch.ones(total, device=device)
    elif schedule == "zeros":
        w = (positions < start).float()
    elif schedule in ("linear", "exp"):
        w = ((start - 1 - positions) / (end - start + 1) + 1).clamp(0, 1)
        if schedule == "exp":
            w = w * torch.expm1(w) / (math.e - 1)
    else:
        raise ValueError(f"Invalid schedule: {schedule}")
    return torch.where(positions >= end, torch.zeros_like(w), w)


def _rtc_per_sample_ints(value, batch_size: int, name: str) -> list:
    """Broadcast a scalar RTC knob to one entry per batch sample, or validate
    an already per-sample sequence.

    Batched parallel-env RTC fires one guided call for several envs whose
    measured delays (and leftover lengths, via ``suffix_length``) generally
    differ, so every schedule boundary is per-sample. Scalars keep the
    single-env / direct-caller contract.
    """
    if isinstance(value, (int, float, bool)):
        return [int(value)] * batch_size
    values = [int(v) for v in value]  # list / tuple / ndarray / 1-d tensor
    if len(values) != batch_size:
        raise ValueError(
            f"{name} must be a scalar or carry one entry per batch sample "
            f"(B={batch_size}), got {len(values)}."
        )
    return values


class LayerwiseFlowmatchingActionHead(nn.Module):
    """
    Layer-wise cross-attention DiT action head.

    NOTE on configuration boundary:
        This module is intentionally decoupled from any specific VLM backbone.
        It ONLY reads from ``global_config.framework.action_model`` (and its
        ``diffusion_model_cfg`` sub-tree).  The framework (e.g. ``Qwen_PI`` /
        ``Qwen_PI_v3``) is responsible for populating ``diffusion_model_cfg``
        with the correct DiT shape **before** calling ``get_action_model``:

            diffusion_model_cfg.num_layers           = <DiT depth>
            diffusion_model_cfg.input_embedding_dim  = <DiT internal hidden>
            diffusion_model_cfg.cross_attention_dim  = <DiT internal hidden>
            diffusion_model_cfg.num_attention_heads  = input_embedding_dim // attention_head_dim

        The head no longer reads ``framework.qwenvl.*`` directly.  See
        ``starVLA/model/framework/VLM4A/diffusion_model_cfg.md`` for details.
    """

    def __init__(
        self,
        global_config,
        **kwargs,
    ):
        super().__init__()
        action_config = global_config.framework.action_model
        diffusion_model_cfg = action_config.diffusion_model_cfg

        # ------------------------------------------------------------------
        # Pure consumer: trust diffusion_model_cfg.  Apply DiTConfig only as
        # a fallback for keys the framework forgot to set.  This keeps the
        # head free of any VLM-specific knowledge.
        # ------------------------------------------------------------------
        for k, v in DiTConfig.items():
            if diffusion_model_cfg.get(k, None) is None:
                diffusion_model_cfg[k] = v

        # Build a plain dict view and drop framework-side hints that are NOT
        # DiT constructor kwargs (e.g. `action_dit_hidden_dim`).  This keeps the
        # head agnostic of any framework convention while still being safe
        # against accidentally-leaked hint keys.
        _DIT_NON_KWARGS = {"action_dit_hidden_dim"}
        diffusion_model_cfg_kwargs = {k: v for k, v in diffusion_model_cfg.items() if k not in _DIT_NON_KWARGS}

        self.input_embedding_dim = diffusion_model_cfg_kwargs["input_embedding_dim"]
        self.model = DiT(**diffusion_model_cfg_kwargs)  # TODO: ideally copy LLM init from VLM
        self.dit_out_hidden_size = self.input_embedding_dim
        self.action_dim = action_config.action_dim
        # `action_horizon` is the canonical chunk length.  Legacy YAMLs are
        # normalised by share_tools.apply_config_compat upstream, so this
        # head never reads `future_action_window_size`.
        self.action_horizon = int(action_config.action_horizon)
        self.num_inference_timesteps = action_config.num_inference_timesteps

        self.state_encoder = (
            MLP(
                input_dim=action_config.state_dim,
                output_dim=self.input_embedding_dim,
            )
            if action_config.state_dim
            else None
        )

        self.action_encoder = ActionEncoder(
            action_dim=action_config.action_dim,
            hidden_size=self.input_embedding_dim,
        )
        self.action_decoder = MLP(
            input_dim=self.input_embedding_dim,
            hidden_dim=1024,
            output_dim=self.action_dim,
        )
        # 0 = no learnable future slots: the noisy action tokens cross-attend
        # the VL context directly (sa_embs = [state? | action]). None (not a
        # 0-row embedding) keeps the parameter out of the state_dict / DDP.
        n_future = int(action_config.num_target_vision_tokens)
        if n_future < 0:
            raise ValueError(
                f"action_model.num_target_vision_tokens must be >= 0, got {n_future}"
            )
        self.future_tokens = (
            nn.Embedding(n_future, self.input_embedding_dim) if n_future > 0 else None
        )
        if self.future_tokens is not None:
            nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        if action_config.add_pos_embed:
            self.position_embedding = nn.Embedding(action_config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(action_config.noise_beta_alpha, action_config.noise_beta_beta)
        self.num_timestep_buckets = action_config.num_timestep_buckets
        self.config = action_config

        # Train-only shortcut mask (plan §5): with this probability per
        # sample, the action-expert rows' cross-attention OUTPUT is zeroed at
        # every cross block, so those samples must read VL semantics through
        # the tactile joint stream. Requires the framework to run the tactile
        # fusion in joint_stream mode (validated there). 0.0 = off = baseline.
        self.shortcut_mask_prob = float(action_config.get("shortcut_mask_prob", 0.0) or 0.0)
        if not (0.0 <= self.shortcut_mask_prob <= 1.0):
            raise ValueError(
                f"action_model.shortcut_mask_prob must be in [0, 1], got {self.shortcut_mask_prob}"
            )
        # Ramp the mask in linearly over the first N optimizer steps: at step 0
        # the tactile encoder and the DiT are both random, so forcing the VL
        # read through the tactile bottleneck immediately just trains the
        # action expert against noise. 0 = no ramp = constant probability
        # (the pre-0902 behaviour, and what the head-level tests assume).
        self.shortcut_mask_warmup_steps = int(
            action_config.get("shortcut_mask_warmup_steps", 0) or 0
        )
        if self.shortcut_mask_warmup_steps < 0:
            raise ValueError(
                "action_model.shortcut_mask_warmup_steps must be >= 0, got "
                f"{self.shortcut_mask_warmup_steps}"
            )
        # Plain int, not a buffer: reading a device buffer here would cost a
        # device->host sync on every train step, and the counter is derived
        # from the trainer's global step (restored on resume), never learned.
        self._train_step = 0

        # Non-persistent: never written to / read from a checkpoint, so it can
        # be changed between runs without breaking weight loading.
        loss_weight_cfg = (
            action_config.get("action_loss_weight", None)
            if hasattr(action_config, "get")
            else getattr(action_config, "action_loss_weight", None)
        )
        self.register_buffer(
            "action_loss_weight",
            build_action_loss_weight(loss_weight_cfg, self.action_horizon, int(self.action_dim)),
            persistent=False,
        )

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def _future_token_embs(self, batch_size: int):
        """``[B, n_future, D]`` learnable slots, or None when disabled (n=0)."""
        if self.future_tokens is None:
            return None
        return self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)

    @staticmethod
    def _concat_sa_embs(state_features, future_tokens, action_features, joint_tokens=None):
        """Stream layout: ``[state? | future? | tactile(joint)? | actions]``.

        Tactile joint tokens sit BEFORE the actions so the decode slice
        ``pred[:, -H:]`` keeps working; they carry no positional embedding
        (their content identity comes from the tactile encoder).
        """
        parts = [
            t
            for t in (state_features, future_tokens, joint_tokens, action_features)
            if t is not None
        ]
        return torch.cat(parts, dim=1)

    def _joint_prefix_len(self, state_features) -> int:
        """Stream positions before the tactile block: state + future slots.

        The state block is the packed history (``state_history_frames`` rows,
        3 in the shipped recipes), NOT a single token — read the real length
        off the tensor or the tactile span would be mis-indexed.
        """
        n = int(state_features.shape[1]) if state_features is not None else 0
        if self.future_tokens is not None:
            n += self.future_tokens.weight.shape[0]
        return n

    def _build_joint_self_attn_bias(
        self, joint_cond: dict, batch_size: int, seq_len: int, n_prefix: int, device, dtype
    ) -> torch.Tensor:
        """Additive self-attn bias implementing the joint-expert block mask.

        The a2t direction (tactile queries reading state / action / future
        rows) always rides the additive gate ``-scale·(1-σ(gate_a2t))``: it
        gates QUERIES, so there is no value to scale instead.

        The t2a direction depends on ``joint_cond["gate_mode"]``:

        ``value`` (default)
            ``TactileActionFusion.build_joint_cond`` already scaled the
            stream tokens by ``gate`` / ``patch_gate``, so t2a attention is
            free (bias 0) and the gate gradient is the append path's
            ``<grad_tokens, tokens>``.
        ``bias``
            Legacy: ``-scale·(1-σ(gate))`` per segment (region block on
            ``gate_region``, 3D rest on ``gate_3d``). At gate 0 and the old
            ``scale=20`` that is -10, a 4.5e-5 relative leak — below bf16
            resolution, so the gate gets rounding noise, not gradient.

        At eval a closed 3D gate (|gate| ≤ eps) hard-masks that segment in
        both modes, mirroring ``_mask_untrusted_if_gate_closed``.
        """
        T = int(joint_cond["tokens"].shape[1])
        t0, t1 = n_prefix, n_prefix + T
        rs, rc = (int(v) for v in joint_cond["region_span"])
        gate_region = joint_cond["gate_region"].float()
        gate_3d = joint_cond["gate_3d"].float()
        gate_a2t = joint_cond["gate_a2t"].float()
        gate_mode = str(joint_cond.get("gate_mode", "value"))
        if gate_mode not in ("value", "bias"):
            raise ValueError(
                f"joint_cond['gate_mode']={gate_mode!r} must be 'value' or 'bias'"
            )
        scale = float(joint_cond.get("bias_scale", 20.0))
        if gate_mode == "value":
            # Free t2a: the gates live on the token values. new_zeros keeps
            # device/dtype and, being leaf-free, adds no autograd edge — the
            # gates are reached through build_joint_cond instead.
            b_region = gate_region.new_zeros(())
            b_3d = gate_3d.new_zeros(())
        else:
            b_region = -scale * (1.0 - torch.sigmoid(gate_region))
            b_3d = -scale * (1.0 - torch.sigmoid(gate_3d))
        b_a2t = -scale * (1.0 - torch.sigmoid(gate_a2t))
        if not self.training and float(gate_3d.detach().abs()) <= float(joint_cond["gate_mask_eps"]):
            b_3d = gate_3d.new_tensor(-1e4)

        bias = torch.zeros(1, 1, seq_len, seq_len, dtype=torch.float32, device=device)
        nontac = torch.cat(
            [torch.arange(0, t0, device=device), torch.arange(t1, seq_len, device=device)]
        )
        cols_3d = torch.cat(
            [
                torch.arange(t0, t0 + rs, device=device),
                torch.arange(t0 + rs + rc, t1, device=device),
            ]
        )
        if cols_3d.numel():
            bias[:, :, nontac.view(-1, 1), cols_3d.view(1, -1)] = b_3d
        if rc:
            cols_region = torch.arange(t0 + rs, t0 + rs + rc, device=device)
            bias[:, :, nontac.view(-1, 1), cols_region.view(1, -1)] = b_region
        rows_tac = torch.arange(t0, t1, device=device)
        bias[:, :, rows_tac.view(-1, 1), nontac.view(1, -1)] = b_a2t
        return bias.to(dtype).expand(batch_size, 1, seq_len, seq_len)

    def set_train_step(self, step: int) -> None:
        """Trainer hook: current OPTIMIZER step, for step-dependent schedules.

        Only the shortcut-mask ramp reads it today. Driven from outside
        rather than self-incremented because gradient accumulation and
        ``repeated_diffusion_steps`` decouple forward calls from optimizer
        steps, and a resume must not restart the schedule.
        """
        self._train_step = int(step)

    @property
    def shortcut_mask_prob_now(self) -> float:
        """``shortcut_mask_prob`` after the linear warm-in ramp."""
        if self.shortcut_mask_prob <= 0.0 or self.shortcut_mask_warmup_steps <= 0:
            return self.shortcut_mask_prob
        frac = self._train_step / float(self.shortcut_mask_warmup_steps)
        return self.shortcut_mask_prob * min(max(frac, 0.0), 1.0)

    def _shortcut_row_mask(
        self, batch_size: int, seq_len: int, joint_cond, n_prefix: int, device, dtype
    ):
        """``[B, S, 1]`` multiplier zeroing cross-attn output rows for samples
        whose action→VL/AQ read is masked this step (train-only). Tactile
        rows keep reading — the semantic path then runs VLM → tactile stream
        → action. Built without any data-dependent early return so the train
        step stays host-sync free; the only branches are on config and on the
        step counter, which every rank agrees on.

        A sample only fires when it actually CARRIES tactile
        (``joint_cond["present"]``). Masking VL on a row whose tactile slots
        hold the ``no_tactile`` placeholder would leave the action expert
        with no context at all and regress a 50-step chunk from noise — on
        the shipped mixtures that is more than half the batch. ``present``
        missing / None means "assume present" (head-level tests, and any
        caller that builds the bundle by hand).
        """
        if not (self.training and self.shortcut_mask_prob > 0.0):
            return None
        if joint_cond is None:
            # A config error, so it must surface even at ramp step 0.
            raise ValueError(
                "action_model.shortcut_mask_prob > 0 requires the tactile joint "
                "stream (framework.tactile.action_fusion.mode: joint_stream)."
            )
        prob = self.shortcut_mask_prob_now
        if prob <= 0.0:  # still at the start of the ramp
            return None
        T = int(joint_cond["tokens"].shape[1])
        t1 = n_prefix + T
        rows = torch.cat(
            [torch.arange(0, n_prefix, device=device), torch.arange(t1, seq_len, device=device)]
        )
        fire = torch.rand(batch_size, device=device) < prob
        present = joint_cond.get("present", None)
        if present is not None:
            present = present.to(device=device, dtype=torch.bool)
            if present.shape != (batch_size,):
                raise ValueError(
                    f"joint_cond['present'] must be [B]=({batch_size},), "
                    f"got {tuple(present.shape)}"
                )
            fire = fire & present
        keep = torch.where(
            fire,
            torch.zeros((), device=device, dtype=dtype),
            torch.ones((), device=device, dtype=dtype),
        )
        mask = torch.ones(batch_size, seq_len, 1, device=device, dtype=dtype)
        mask[:, rows, :] = keep.view(batch_size, 1, 1)
        return mask

    def _prepare_joint(
        self, joint_cond, state_features, batch_size: int, device, dtype, action_len: int
    ):
        """Shared joint/shortcut plumbing for every sampling branch.

        Returns ``(joint_tokens_or_None, self_attn_bias, row_mask)``. The bias
        carries the a2t gate (and, under ``gate_mode: bias``, the t2a gates);
        the t2a gates otherwise ride the token values built by
        ``build_joint_cond``. Either way the action loss reaches them.
        """
        if joint_cond is None:
            if self.training and self.shortcut_mask_prob > 0.0:
                raise ValueError(
                    "action_model.shortcut_mask_prob > 0 requires the tactile "
                    "joint stream (joint_cond is None)."
                )
            return None, None, None
        joint_tokens = joint_cond["tokens"].to(device=device, dtype=dtype)
        n_prefix = self._joint_prefix_len(state_features)
        seq_len = n_prefix + int(joint_tokens.shape[1]) + int(action_len)
        bias = self._build_joint_self_attn_bias(
            joint_cond, batch_size, seq_len, n_prefix, device, dtype
        )
        row_mask = self._shortcut_row_mask(
            batch_size, seq_len, joint_cond, n_prefix, device, dtype
        )
        return joint_tokens, bias, row_mask

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def forward(
        self,
        vl_embs_list: list,
        actions: torch.Tensor,
        state: torch.Tensor = None,
        encoder_attention_mask=None,
        extra_temb: torch.Tensor = None,
        joint_cond: dict = None,
    ):
        """
        vl_embs: list of torch.Tensor, each shape (B, seq_length, feature_dim)
        actions: shape (B, action_horizon, D_action)
        encoder_attention_mask: optional (B, seq_length) bool/int
        joint_cond: optional tactile joint-stream bundle from
            ``TactileActionFusion.build_joint_cond`` (tokens + gate tensors).
        """
        device = actions.device
        num_layers = len(vl_embs_list)
        B, L, D = vl_embs_list[0].shape
        # Compute the flow-matching targets in float32 regardless of the
        # module dtype: with bf16 weights (DeepSpeed bf16 training) the old
        # actions.dtype path put noise/velocity on the bf16 grid (~4e-3 in
        # [-1, 1]). Only the DiT INPUT is cast down to the module dtype; the
        # regression target keeps full precision.
        model_dtype = self.dtype
        actions = actions.float()
        # Embed noised action trajectory.
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed. sample_time can return
        # t == 1.0 exactly (Beta sample underflows to 0.0), which would index
        # one past the ada_norm embedding table; negative t truncates to 0.
        t_discretized = (
            (t[:, 0, 0] * self.num_timestep_buckets)
            .long()
            .clamp_(0, self.num_timestep_buckets - 1)
        )
        action_features = self.action_encoder(noisy_trajectory.to(model_dtype), t_discretized)

        # Embed state
        state_features = self.state_encoder(state) if state is not None else None

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # state and action embedding along sequence dimension.
        joint_tokens, self_attn_bias, row_mask = self._prepare_joint(
            joint_cond, state_features, B, device, model_dtype, action_features.shape[1]
        )
        sa_embs = self._concat_sa_embs(
            state_features, self._future_token_embs(B), action_features, joint_tokens
        )

        # Layer-wise DiT forward. DiT handles cross/self-attention interleaving.
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs_list,
            timestep=t_discretized,
            encoder_attention_mask=encoder_attention_mask,
            return_pre_output=True,
            extra_temb=extra_temb,
            self_attn_bias=self_attn_bias,
            cross_attn_out_row_mask=row_mask,
        )

        # Decode only the action-token positions.
        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target. Upcast the
        # prediction so the loss is computed against the fp32 target.
        squared_error = (pred_actions.float() - velocity) ** 2
        if self.action_loss_weight is None:
            loss = squared_error.mean()
        else:
            loss = (squared_error * self.action_loss_weight.to(squared_error.dtype)).mean()
        return loss

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs_list: list,
        state: torch.Tensor = None,
        encoder_attention_mask=None,
        extra_temb: torch.Tensor = None,
        joint_cond: dict = None,
    ) -> torch.Tensor:
        # Set initial actions as the sampled noise. The Euler accumulator is
        # kept in float32 so integration error does not compound on the bf16
        # grid across steps; the DiT input is cast to the module dtype per
        # step (the network itself still runs at its own precision).
        batch_size = vl_embs_list[0].shape[0]
        device = vl_embs_list[0].device
        model_dtype = vl_embs_list[0].dtype
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=torch.float32,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        state_features = self.state_encoder(state) if state is not None else None

        # Hoisted out of the denoising loop: the joint tokens and the gate
        # bias do not depend on t, and rebuilding them per step would also
        # repeat the eval hard-mask's device->host sync once per step — pure
        # latency on the RTC / deploy path.
        joint_tokens, self_attn_bias, _ = self._prepare_joint(
            joint_cond, state_features, batch_size, device, model_dtype, actions.shape[1]
        )

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized_int = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized_int, device=device, dtype=torch.long
            )

            # Embed current action trajectory with timestep
            action_features = self.action_encoder(actions.to(model_dtype), timesteps_tensor)

            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            sa_embs = self._concat_sa_embs(
                state_features, self._future_token_embs(batch_size), action_features, joint_tokens
            )

            # Layer-wise DiT forward. DiT handles cross/self-attention interleaving.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs_list,
                timestep=timesteps_tensor,
                encoder_attention_mask=encoder_attention_mask,
                return_pre_output=True,
                extra_temb=extra_temb,
                self_attn_bias=self_attn_bias,
            )
            # Decode only the action-token positions.
            pred = self.action_decoder(model_output)
            pred_velocity = pred[:, -self.action_horizon :]

            # Euler integration (fp32 accumulator).
            actions = actions + dt * pred_velocity.float()
        return actions

    def predict_action_realtime(
        self,
        vl_embs_list: list,
        state: torch.Tensor = None,
        prev_action_chunk: torch.Tensor = None,
        inference_delay: int | list = 1,
        mode: str = "pigdm",
        suffix_length: int | list | None = None,
        prefix_attention_schedule: str = "exp",
        max_guidance_weight: float = 10.0,
        encoder_attention_mask=None,
        extra_temb: torch.Tensor = None,
        joint_cond: dict = None,
    ) -> torch.Tensor:
        """RTC-aware flow-matching inference.

        Conditions sampling on the previous chunk's un-executed actions as a
        known prefix, so chunks splice smoothly under execution latency.
        Mirrors the official reference implementation (Kinetix,
        arXiv:2506.07339): the ΠGDM step formulas and the prefix-weight
        schedules are line-by-line ports.

        Args:
            vl_embs_list: layer-wise VL embeddings, list of (B, seq, D).
            state: optional state tensor (B, 1, state_dim).
            prev_action_chunk: (B, action_horizon, action_dim) normalised
                actions from the previous prediction.
            inference_delay: number of leading timesteps to fix as prefix (d).
                A scalar, or one entry per batch sample — batched parallel-env
                RTC fires one guided call for envs whose measured delays
                generally differ.
            mode: "pigdm" (default, VJP-guided; works without retraining,
                ~2x cost per step) or "simulated_delay" (naive
                replace-and-denoise with per-position timesteps; requires
                model trained with simulated-delay loss).
            suffix_length: ΠGDM only. Number of trailing timesteps with
                weight 0 — scalar or per-sample (a batched call's leftovers
                are generally ragged). Default ``None`` follows the paper
                convention (Kinetix eval loop): the weights hit zero exactly
                where the real leftover data ends, i.e.
                ``action_horizon - T_prev`` — the zero-padded rows of a short
                leftover must never carry weight. (The framework in QwenPI_v3
                always passes it explicitly; the default only serves direct
                callers.)
            prefix_attention_schedule: ΠGDM only. Weight schedule for the
                transition zone — "exp", "linear", "ones", or "zeros".
            max_guidance_weight: ΠGDM only. Cap on the guidance weight.
            encoder_attention_mask: optional (B, seq) mask for the VL tokens,
                same contract as ``predict_action``. Required with
                ``interleave_self_attention`` recipes, where padded VL tokens
                would otherwise leak into even-block cross-attention.

        Returns:
            actions: (B, action_horizon, action_dim) predicted actions.

        NOTE: callers must invoke this under ``torch.no_grad()`` (or no
        decorator at all), NOT ``torch.inference_mode()`` — the ΠGDM step
        re-enables grad locally for the vector-Jacobian product, and
        inference-mode tensors cannot require grad.
        """
        batch_size = vl_embs_list[0].shape[0]
        device = vl_embs_list[0].device
        model_dtype = vl_embs_list[0].dtype

        delays = _rtc_per_sample_ints(inference_delay, batch_size, "inference_delay")
        if prev_action_chunk is None or all(d <= 0 for d in delays):
            # Unguided fallback — taken for the FIRST chunk of every episode
            # (no previous chunk) and for any zero-delay call. joint_cond must
            # ride along: dropping it would make exactly that first chunk
            # V-only while every later chunk sees the tactile stream.
            return self.predict_action(
                vl_embs_list,
                state,
                encoder_attention_mask=encoder_attention_mask,
                extra_temb=extra_temb,
                joint_cond=joint_cond,
            )

        def _forward_velocity(actions, timesteps, state_features, joint_tokens, self_attn_bias):
            """Single DiT forward pass returning predicted velocity (fp32).

            Same precision contract as ``predict_action``: the Euler
            accumulator stays in float32, only the DiT input is cast to the
            module dtype, and the mask reaches the cross-attention layers.
            The joint tokens / gate bias are built once by the caller: they do
            not depend on the step, and the eval hard-mask costs a device->host
            sync that must not run once per ΠGDM step.
            """
            B = actions.shape[0]
            action_features = self.action_encoder(actions.to(model_dtype), timesteps)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs
            sa_embs = self._concat_sa_embs(
                state_features, self._future_token_embs(B), action_features, joint_tokens
            )
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs_list,
                timestep=timesteps,
                encoder_attention_mask=encoder_attention_mask,
                return_pre_output=True,
                extra_temb=extra_temb,
                self_attn_bias=self_attn_bias,
            )
            pred = self.action_decoder(model_output)
            return pred[:, -self.action_horizon :].float()

        # Clamp to the chunk's own length (the framework already truncates;
        # this keeps direct callers safe), then zero-pad to the horizon. The
        # padded rows must land in the zero-weight region — see suffix_length.
        T_prev = min(int(prev_action_chunk.shape[1]), self.action_horizon)
        prev_action_chunk = prev_action_chunk[:, :T_prev]
        if T_prev < self.action_horizon:
            pad = torch.zeros(
                batch_size,
                self.action_horizon - T_prev,
                prev_action_chunk.shape[2],
                device=device,
                dtype=prev_action_chunk.dtype,
            )
            prev_action_chunk = torch.cat([prev_action_chunk, pad], dim=1)

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # fp32 Euler accumulator, mirroring ``predict_action``: integrating on
        # the bf16 grid compounds rounding error across steps.
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=torch.float32,
            device=device,
        )

        state_features = self.state_encoder(state) if state is not None else None

        # Step-independent, so built once for BOTH branches (see
        # _forward_velocity). Built here under the caller's no_grad: the ΠGDM
        # VJP only differentiates w.r.t. x_t, so the gates never need to be in
        # that graph.
        joint_tokens, self_attn_bias, _ = self._prepare_joint(
            joint_cond, state_features, batch_size, device, model_dtype, self.action_horizon
        )

        if mode == "pigdm":
            # ── ΠGDM: pseudo-inverse guided diffusion ──
            if suffix_length is None:
                # Paper convention: end = T_prev, so the weights hit zero
                # exactly where the real leftover data ends. Defaulting to
                # inference_delay instead would put non-zero weight on the
                # zero padding of a short leftover (end = H - delay > T_prev),
                # pulling those rows toward zero.
                suffixes = [self.action_horizon - T_prev] * batch_size
            else:
                suffixes = _rtc_per_sample_ints(suffix_length, batch_size, "suffix_length")
            # (B, H): one weight schedule per sample. Scalar callers stack B
            # identical rows, so the single-env path is bit-identical to the
            # pre-batch broadcast.
            weights = torch.stack(
                [
                    get_rtc_prefix_weights(
                        delay,
                        self.action_horizon - suffix,
                        self.action_horizon,
                        prefix_attention_schedule,
                        device=device,
                    )
                    for delay, suffix in zip(delays, suffixes)
                ]
            )
            prev = prev_action_chunk.float()

            for t_step in range(num_steps):
                t_cont = t_step / float(num_steps)
                t_bucket = int(t_cont * self.num_timestep_buckets)
                timesteps_tensor = torch.full(
                    (batch_size,),
                    t_bucket,
                    device=device,
                    dtype=torch.long,
                )

                with torch.enable_grad():
                    x_t = actions.detach().requires_grad_(True)
                    v_t = _forward_velocity(
                        x_t, timesteps_tensor, state_features, joint_tokens, self_attn_bias
                    )
                    x_1_hat = x_t + v_t * (1.0 - t_cont)
                    error = (prev - x_1_hat) * weights[:, :, None]
                    pinv_correction = torch.autograd.grad(
                        x_1_hat,
                        x_t,
                        grad_outputs=error,
                    )[0]

                # Guidance weight schedule from ΠGDM paper
                t = t_cont
                inv_r2 = (t**2 + (1 - t) ** 2) / ((1 - t) ** 2 + 1e-8)
                c = (1 - t) / (t + 1e-8)
                gw = min(c * inv_r2, max_guidance_weight)

                v_corrected = v_t.detach() + gw * pinv_correction.detach()
                actions = actions.detach() + dt * v_corrected

            return actions

        # ── simulated_delay mode ──
        # NOTE: this is NOT a faithful Kinetix port and must not be treated as
        # one. There the per-position time (prefix rows pinned at t=1)
        # conditions the WHOLE network; here only the action encoder sees the
        # per-position timesteps while the DiT's adaLN timestep stays the
        # scalar noisy t for every row. It also requires the simulated-delay
        # training loss this repo does not implement — which is why the deploy
        # gate (validate_rtc_deploy_config) refuses to serve it. Kept for
        # head-level experiments only.
        with torch.no_grad():
            # (B, H, 1): per-sample frozen prefix (delays differ across a
            # batched parallel-env call).
            positions = torch.arange(self.action_horizon, device=device)
            prefix_mask = (
                positions[None, :] < torch.tensor(delays, device=device)[:, None]
            )[:, :, None]
            prev = prev_action_chunk.float()

            for t_step in range(num_steps):
                t_cont = t_step / float(num_steps)
                actions = torch.where(prefix_mask, prev, actions)

                t_bucket = int(t_cont * self.num_timestep_buckets)
                t_bucket_prefix = int(1.0 * self.num_timestep_buckets) - 1
                timesteps_tensor = torch.full(
                    (batch_size, self.action_horizon),
                    t_bucket,
                    device=device,
                    dtype=torch.long,
                )
                timesteps_tensor[prefix_mask[..., 0]] = t_bucket_prefix

                # Inline ActionEncoder with per-position timesteps
                a_emb = self.action_encoder.layer1(actions.to(model_dtype))
                tau_emb = self.action_encoder.pos_encoding(timesteps_tensor).to(dtype=a_emb.dtype)
                x_enc = torch.cat([a_emb, tau_emb], dim=-1)
                x_enc = swish(self.action_encoder.layer2(x_enc))
                action_features = self.action_encoder.layer3(x_enc)

                if self.config.add_pos_embed:
                    pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                    pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                    action_features = action_features + pos_embs

                sa_embs = self._concat_sa_embs(
                    state_features, self._future_token_embs(batch_size), action_features, joint_tokens
                )

                temb_tensor = torch.full(
                    (batch_size,),
                    t_bucket,
                    device=device,
                    dtype=torch.long,
                )
                model_output = self.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embs_list,
                    timestep=temb_tensor,
                    encoder_attention_mask=encoder_attention_mask,
                    return_pre_output=True,
                    extra_temb=extra_temb,
                    self_attn_bias=self_attn_bias,
                )

                pred = self.action_decoder(model_output)
                pred_velocity = pred[:, -self.action_horizon :]
                actions = actions + dt * pred_velocity.float()
                actions = torch.where(prefix_mask, prev, actions)

            return actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


def get_action_model(config=None):
    """
    Factory: build FlowmatchingActionHead from global framework config.

    Args:
        config: Global config (expects config.framework.action_model namespace).

    Returns:
        FlowmatchingActionHead: Initialized FlowMatchingActionHead.
    """
    return LayerwiseFlowmatchingActionHead(global_config=config)


if __name__ == "__main__":
    # TODO make each backbone.py can be debug independently

    pass
