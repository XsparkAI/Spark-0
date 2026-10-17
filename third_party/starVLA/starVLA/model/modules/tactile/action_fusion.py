# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Local Spark-0 addition (NOT upstream starVLA).
"""
Tactile → Action-DiT fusion (P1).

The full encoder hierarchy — fast region tokens AND slow 3D taxel tokens —
stays ``append_context``:

    context_layer = concat(projected_vl_context_layer,
                           split_gate * proj(tactile_tokens) + layer_embedding)

``split_gate`` is ``patch_gate`` on the region block and ``gate`` on the
3D rest, so opening the trusted force path does not inject the
camera-frame cloud. The block is the encoder's ``region_token_span`` —
keyed on the region STREAM, not the fast role, because a role swap
(``region_pressure.role: slow``) would otherwise gate the cloud with the
patch scalar. When ``patch_gate`` is absent (``patch: off``) the region
block falls back to ``gate``.

Region-pressure patches are a short force window (code default 2 hands
× 5 regions × 5 frames; ``fingertip_only`` / ``robust`` ship 20).
They ALSO join that sequence as per-region
TemporalConv tokens (spatial routing + temporal inductive bias); on top of
that, the raw window is flattened → MLP → added to every DiT block's AdaLN
``temb`` (every denoising step, every layer, including self-attn-only
blocks) as a dense global modulation. The MLP's last layer starts at 0, so
the first step's temb is unchanged.

Only the DiT blocks that actually cross-attend get a layer embedding. Under
``interleave_self_attention`` + the canonical forward the odd blocks are
handed ``encoder_hidden_states=None``, so building 36 embeddings for an
18-cross-attn DiT would leave half of them without gradient (a DDP
unused-parameter hang). Tokens are still appended to every layer so
``vl_embs_list`` stays homogeneous and the single shared
``encoder_attention_mask`` is length-consistent wherever it is used.

``gate`` (3D taxel tokens, logged ``tactile_gate``) and ``patch_gate``
(region tokens, logged ``tactile_patch_gate``) are independent scalars.
Opening the trusted force path therefore does not inject the untrusted
camera-frame cloud. At **train**, ``gate=0`` zeroes that slice's value but
the keys still enter softmax (warm start / DDP value-path). At **eval**,
``|gate| <= gate_mask_eps`` masks the 3D slice out of
``encoder_attention_mask`` so those keys do not compete. Bit-identity is
the ``enable=false`` contract. ``gate_l1`` is an optional L1 on the 3D
gate only (0 = off; see the warning under "gate semantics" below).

Both gates are applied ONLY to rows that carry real tactile
(``present_region`` / ``present_3d``, post-dropout and post-quality-gate).
Absent rows hold the learned ``no_tactile`` placeholder and pass through
ungated. Gating them would give the gate a gradient from rows with no
tactile information — the majority on the shipped mixtures — and
``gate * no_tactile`` is an over-parameterised product anyway, since the
placeholder is a free learnable vector that can absorb any scale.

gate semantics under ``joint_stream``
─────────────────────────────────────
The stream tokens hit the DiT's per-token LayerNorm (``AdaLayerNorm``,
``norm_eps`` 1e-5) before anything reads them, so the block output is
scale-invariant above ``|gate| * std(tokens) ~ sqrt(norm_eps) = 3.2e-3``
and collapses below it. Consequences, all of them load-bearing:

* the gate is a SWITCH (left the eps hole or not), never a mixing weight;
* ``gate_l1`` cannot regularise information flow — it only wins the
  gradient race and pins the logged number inside the hole, i.e. it turns
  "weak" into "off". Prefer ``slow_dropout_prob`` plus the encoder's
  ``quality_*`` gates for an untrusted stream. Measured on the 0901
  pretrain: ``patch_gate`` (no L1) climbed to 0.105 while ``gate``
  (``gate_l1: 0.01``) stayed a zero-mean ±3e-5 oscillation, 100x under the
  knee, with everything else in the run identical;
* ``gate_init`` should start ABOVE the knee (the shipped stems use 0.1,
  33x). 0-init parks the parameter at the one point where the LayerNorm
  Jacobian blows up as ``1/sqrt(eps)``, and a token-only gate has no
  un-normalised path to pull it back out.

The AdaLN ``temb`` path is the exception: it is added to an unnormalised
embedding, so magnitude IS linear there. It therefore does NOT share
``patch_gate`` (``PatchToTemb.fc2`` is zero-init and is its own scale).

Config: ``framework.tactile.action_fusion`` (via ``tactile_fusion: <stem>``).
``patch: temb`` (default when the region stream is on) builds PatchToTemb;
``patch: off`` skips it (3D-only stems).

``mode: joint_stream`` (Spark-0 0821 plan, change 1)
────────────────────────────────────────────────────
Instead of appending static tokens to every layer's context, the tactile
tokens join the DiT HIDDEN stream (``[state | future? | tactile | actions]``)
and co-evolve through the blocks: even blocks cross-attend the per-layer
VL/AQ context (both experts read it), odd blocks run joint self-attention
where the two experts attend each other. ``layer_embedding`` is not
constructed in this mode. Requires the interleaved canonical DiT forward.

``joint_gate_mode`` decides HOW the tactile→action (t2a) direction is gated:

``value`` (default)
    The tokens handed to the head are already scaled by ``_gate_tokens``
    (region block × ``patch_gate``, 3D rest × ``gate``) and the t2a
    self-attention is left FREE. This is the append_context contract moved
    into the joint stream: at gate 0 the tactile rows carry no information,
    but ``dL/dgate = <grad_tokens, tokens>`` is a full-strength polynomial
    signal, so the action loss can actually open the gate.
``bias``
    The original formulation: t2a rides an additive self-attn bias
    ``-joint_bias_scale·(1-σ(gate))``. At gate 0 that is -10, i.e. the
    tactile keys reach ``e^-10 ≈ 4.5e-5`` of the attention mass. Under bf16
    training (relative resolution ~3.9e-3) that perturbation is rounded out
    of the block output, so the forward is unchanged and the gate receives
    rounding noise instead of gradient — with ``gate_l1`` on top it can
    never leave 0. Kept only to reproduce pre-fix runs.

Either way the parameter set is identical (no state_dict change), and the
a2t direction (``joint_a2t_gate``: what the tactile rows read FROM the
action stream) stays on the additive bias — it gates queries, not values,
so there is nothing to scale. ``joint_bias_scale`` therefore matters in
both modes: the shipped stems lower it from 20 to 6 so a2t starts at -3
(≈5% leak, above the bf16 floor) instead of -10.

At eval a closed 3D gate (``|gate| <= gate_mask_eps``) still hard-masks the
3D columns out of the joint attention, mirroring
``_mask_untrusted_if_gate_closed`` on the append path.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


def _as_dict(cfg: Any) -> dict:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    if hasattr(cfg, "to_dict"):
        return dict(cfg.to_dict())
    return dict(cfg)


def resolve_cross_attention_layers(
    num_layers: int,
    *,
    interleave_self_attention: bool,
    use_canonical_forward: bool = True,
) -> List[int]:
    """DiT block indices that consume ``encoder_hidden_states``.

    Mirrors ``cross_attention_dit.DiT.forward``. ``use_canonical_forward``
    matters: a block has ONE ``attn1`` doing double duty, so an odd block only
    stops reading context because the canonical forward explicitly hands it
    ``encoder_hidden_states=None``. Under the legacy forward every block is
    still fed ``encoder_hidden_states[idx]`` and therefore consumes it.
    """
    if interleave_self_attention and use_canonical_forward:
        return [i for i in range(num_layers) if i % 2 == 0]
    return list(range(num_layers))


def flatten_region_patch(
    region_pressure: Optional[torch.Tensor],
    *,
    num_hands: int,
    num_regions: int,
    history_len: int,
    channel_dim: int,
    present: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """``[B, H, R, T, C]`` → ``[B, H*R*history_len*C]`` (last ``history_len``).

    Packed tensors already match the encoder window. This helper is still
    lenient on ``T``: take the newest frames, left-pad with zeros if short.
    ``H`` / ``R`` / ``C`` must match the encoder. Absent rows
    (``present=False``) are zeroed so a missing glove cannot leak force.
    Non-finite values are zeroed so a NaN pad cannot FiLM the whole DiT.
    """
    h, r, t_need, c = int(num_hands), int(num_regions), int(history_len), int(channel_dim)
    if min(h, r, t_need, c) < 1:
        raise ValueError(
            "flatten_region_patch needs num_hands/num_regions/"
            "history_len/channel_dim >= 1"
        )
    dim = h * r * t_need * c
    if region_pressure is None:
        if batch_size is None:
            raise ValueError("flatten_region_patch needs batch_size when region_pressure is None")
        B = int(batch_size)
        if device is None:
            device = present.device if present is not None else torch.device("cpu")
        if dtype is None:
            dtype = torch.float32
        out = torch.zeros(B, dim, device=device, dtype=dtype)
    else:
        if not isinstance(region_pressure, torch.Tensor):
            raise TypeError(
                f"region_pressure must be a Tensor, got {type(region_pressure)}"
            )
        if region_pressure.ndim != 5:
            raise ValueError(
                f"region_pressure must be [B,H,R,T,C], got {tuple(region_pressure.shape)}"
            )
        B, H, R, T, C = region_pressure.shape
        if (H, R, C) != (h, r, c):
            raise ValueError(
                f"region_pressure [B,{H},{R},T,{C}] != encoder "
                f"[B,{h},{r},T,{c}]"
            )
        if batch_size is not None and int(batch_size) != B:
            raise ValueError(
                f"region_pressure B={B} != batch_size={int(batch_size)}"
            )
        if T >= t_need:
            window = region_pressure[:, :, :, T - t_need :, :]
        else:
            pad = region_pressure.new_zeros(B, H, R, t_need - T, C)
            window = torch.cat([pad, region_pressure], dim=3)
        out = window.reshape(B, dim)
        out = out.masked_fill(~torch.isfinite(out), 0.0)
        if device is not None or dtype is not None:
            out = out.to(
                device=device if device is not None else out.device,
                dtype=dtype if dtype is not None else out.dtype,
            )
    if present is not None:
        present = present.to(device=out.device, dtype=torch.bool)
        if present.shape != (out.shape[0],):
            raise ValueError(
                f"present must be [B]=({out.shape[0]},), got {tuple(present.shape)}"
            )
        out = out * present.to(dtype=out.dtype).view(out.shape[0], 1)
    return out


class PatchToTemb(nn.Module):
    """Flattened region window → DiT AdaLN ``temb`` addend.

    Last Linear is zero-init (weight and bias) so ``temb`` is unchanged at
    step 0. The first layer stays in the graph (DDP / ZeRO unused-param).
    ``fc2`` is also the ONLY learnable scale on this path — see
    ``TactileActionFusion.patch_to_temb`` for why ``patch_gate`` must not
    multiply it as well.
    """

    def __init__(self, patch_dim: int, temb_dim: int, hidden_dim: int = 256):
        super().__init__()
        if patch_dim < 1 or temb_dim < 1 or hidden_dim < 1:
            raise ValueError("PatchToTemb dims must be >= 1")
        self.fc1 = nn.Linear(patch_dim, hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, temb_dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, patch_cond: torch.Tensor) -> torch.Tensor:
        if patch_cond.ndim != 2:
            raise ValueError(
                f"patch_cond must be [B, D], got {tuple(patch_cond.shape)}"
            )
        return self.fc2(self.act(self.fc1(patch_cond)))


class TactileActionFusion(nn.Module):
    """Full-hierarchy ``append_context`` + optional region-pressure ``PatchToTemb``.

    Region tokens (``patch_token_start`` / ``patch_token_count``, from the
    encoder's ``region_token_span``) ride ``patch_gate`` so the trusted force
    expert can open without injecting the 3D cloud; every other token uses
    ``gate``. ``patch_token_count=0`` (the default) puts the whole sequence
    on ``gate`` — the 3D-only stems. The AdaLN temb path is ungated (see
    ``patch_to_temb``), and both gates only apply to rows whose tactile is
    really present (see ``_gate_tokens``).
    """

    def __init__(
        self,
        cfg: Any,
        *,
        num_dit_layers: int,
        dit_hidden_dim: int,
        tactile_dim: int,
        interleave_self_attention: bool,
        use_canonical_forward: bool = True,
        patch_cond_dim: int = 0,
        patch_token_start: int = 0,
        patch_token_count: int = 0,
    ):
        super().__init__()
        cfg = _as_dict(cfg)
        if not bool(cfg.get("enable", False)):
            raise ValueError(
                "TactileActionFusion requires action_fusion.enable=true; the "
                "framework must skip construction otherwise."
            )

        mode = str(cfg.get("mode", "append_context"))
        if mode not in ("append_context", "joint_stream"):
            raise ValueError(
                f"framework.tactile.action_fusion.mode={mode!r} is not implemented; "
                "use 'append_context' (static per-layer context append) or "
                "'joint_stream' (tactile tokens join the DiT hidden stream and "
                "co-evolve via gated joint attention)."
            )
        if mode == "joint_stream" and not (interleave_self_attention and use_canonical_forward):
            raise ValueError(
                "action_fusion.mode=joint_stream requires the interleaved canonical "
                "DiT forward (interleave_self_attention=true, use_canonical_forward=true): "
                "the joint bias lives on the odd self-attention blocks."
            )
        self.mode = mode

        # joint_stream t2a gating. See the module docstring: `bias` is the
        # original attention-bias gate, unusable under bf16 at gate 0.
        joint_gate_mode = str(cfg.get("joint_gate_mode", "value")).lower()
        if joint_gate_mode not in ("value", "bias"):
            raise ValueError(
                f"framework.tactile.action_fusion.joint_gate_mode={joint_gate_mode!r} "
                "must be 'value' (gates scale the stream tokens, free t2a "
                "attention) or 'bias' (legacy additive self-attn gate)."
            )
        joint_bias_scale = float(cfg.get("joint_bias_scale", 20.0))
        if not joint_bias_scale > 0.0:
            raise ValueError(
                "framework.tactile.action_fusion.joint_bias_scale must be > 0, "
                f"got {joint_bias_scale}"
            )
        # Both keys are validated in either mode but only read under
        # joint_stream — append_context always gates the appended VALUES.
        # No raise: the shipped stems carry them, and flipping the one `mode`
        # line to reproduce an old append_context checkpoint must keep working.
        self.joint_gate_mode = joint_gate_mode
        self.joint_bias_scale = joint_bias_scale

        layer_selection = str(cfg.get("layer_selection", "even_all"))
        if layer_selection not in ("even_all", "all"):
            raise ValueError(
                f"framework.tactile.action_fusion.layer_selection={layer_selection!r} "
                "must be 'even_all' (cross-attending blocks) or 'all'."
            )

        # Both settings resolve to the same set: only cross-attending blocks
        # can read context at all. 'all' is accepted so a non-interleaved DiT
        # config reads naturally.
        cross_layers = resolve_cross_attention_layers(
            num_dit_layers,
            interleave_self_attention=interleave_self_attention,
            use_canonical_forward=use_canonical_forward,
        )
        if not cross_layers:
            raise ValueError(
                f"no cross-attending DiT block among {num_dit_layers} layers; "
                "tactile append_context has nowhere to go."
            )

        self.num_dit_layers = int(num_dit_layers)
        self.cross_layers = cross_layers
        self._cross_slot = {idx: slot for slot, idx in enumerate(cross_layers)}
        self.dit_hidden_dim = int(dit_hidden_dim)
        self.tactile_dim = int(tactile_dim)
        self.patch_cond_dim = int(patch_cond_dim)
        # The region block of the laid-out token sequence (encoder
        # `region_token_span`). Everything outside it is the 3D cloud.
        self.patch_token_start = int(patch_token_start)
        self.patch_token_count = int(patch_token_count)
        if self.patch_token_start < 0 or self.patch_token_count < 0:
            raise ValueError("patch_token_start / patch_token_count must be >= 0")

        # Adapter only when the encoder width differs from the DiT context.
        if self.tactile_dim != self.dit_hidden_dim:
            self.in_proj: nn.Module = nn.Linear(self.tactile_dim, self.dit_hidden_dim)
        else:
            self.in_proj = nn.Identity()

        # Zero-init so the appended block is exactly the zero vector at step 0.
        # Meaningless in joint_stream mode (stream tokens co-evolve through the
        # blocks instead of being re-appended per layer) — not constructed
        # there so no dead parameter enters DDP.
        if self.mode == "append_context":
            self.layer_embedding = nn.Embedding(len(cross_layers), self.dit_hidden_dim)
            nn.init.zeros_(self.layer_embedding.weight)
        else:
            self.layer_embedding = None

        gate_init = float(cfg.get("gate_init", 0.0))
        self.gate = nn.Parameter(torch.tensor(gate_init, dtype=torch.float32))
        # joint_stream only: action→tactile read gate (the t2a direction
        # reuses `gate` / `patch_gate` per segment). 0-init = both directions
        # ~closed at start (bias -10, leak ≈ 4.5e-5/key) = warm-start baseline.
        self.joint_a2t_gate = (
            nn.Parameter(
                torch.tensor(float(cfg.get("joint_a2t_gate_init", 0.0)), dtype=torch.float32)
            )
            if self.mode == "joint_stream"
            else None
        )
        self.gate_l1 = float(cfg.get("gate_l1", 0.0))
        if self.gate_l1 < 0.0:
            raise ValueError(
                f"framework.tactile.action_fusion.gate_l1 must be >= 0, "
                f"got {self.gate_l1}"
            )
        if self.gate_l1 > 0.0 and self.mode == "joint_stream":
            # The L1 subgradient is a CONSTANT gate_l1; AdamW normalises it to
            # a full lr step, while the data gradient vanishes above the
            # LayerNorm knee. The gate therefore oscillates inside the eps
            # hole (measured: ±3e-5 vs a 3.2e-3 knee) and the 3D stream is
            # off, not weak. Kept configurable only to reproduce old runs.
            print(
                "[starVLA][WARN] action_fusion.gate_l1="
                f"{self.gate_l1:g} under mode=joint_stream: the stream tokens "
                "are LayerNormed, so this L1 does not regularise information "
                "flow — it pins tactile_gate at ~0 and switches the 3D stream "
                "OFF. Use slow_dropout_prob / the encoder quality gates "
                "instead, and set gate_l1: 0."
            )
        self.gate_mask_eps = float(cfg.get("gate_mask_eps", 1e-6))
        if self.gate_mask_eps < 0.0:
            raise ValueError(
                f"framework.tactile.action_fusion.gate_mask_eps must be >= 0, "
                f"got {self.gate_mask_eps}"
            )

        self.vt_to_v_distill_coeff = float(cfg.get("vt_to_v_distill_coeff", 0.0))

        # Region → AdaLN temb. Default temb when the encoder has a region
        # window; 3D-only stems (patch_cond_dim=0) stay off.
        default_patch = "temb" if self.patch_cond_dim > 0 else "off"
        patch_mode = str(cfg.get("patch", default_patch)).lower()
        if patch_mode not in ("temb", "off"):
            raise ValueError(
                f"framework.tactile.action_fusion.patch={patch_mode!r} "
                "must be 'temb' or 'off'."
            )
        if patch_mode == "temb" and self.patch_cond_dim < 1:
            raise ValueError(
                "action_fusion.patch=temb needs an enabled region_pressure "
                "stream (patch_cond_dim >= 1)."
            )
        self.patch_mode = patch_mode
        if patch_mode == "temb":
            hidden = int(cfg.get("patch_hidden", 256))
            self.patch_mlp = PatchToTemb(
                self.patch_cond_dim, self.dit_hidden_dim, hidden_dim=hidden
            )
            patch_gate_init = float(cfg.get("patch_gate_init", 0.0))
            self.patch_gate = nn.Parameter(
                torch.tensor(patch_gate_init, dtype=torch.float32)
            )
        else:
            self.patch_mlp = None
            self.register_parameter("patch_gate", None)

    @property
    def num_cross_layers(self) -> int:
        return len(self.cross_layers)

    def extra_repr(self) -> str:
        joint = (
            f", joint_gate_mode={self.joint_gate_mode}, "
            f"joint_bias_scale={self.joint_bias_scale:g}"
            if self.mode == "joint_stream"
            else ""
        )
        return (
            f"cross_layers={len(self.cross_layers)}/{self.num_dit_layers}, "
            f"tactile_dim={self.tactile_dim}, dit_hidden_dim={self.dit_hidden_dim}, "
            f"patch={self.patch_mode}, patch_cond_dim={self.patch_cond_dim}, "
            f"region_tokens=[{self.patch_token_start}, "
            f"{self.patch_token_start + self.patch_token_count}), "
            f"gate_l1={self.gate_l1:g}, gate_mask_eps={self.gate_mask_eps:g}"
            f"{joint}"
        )

    def gate_l1_penalty(self) -> torch.Tensor:
        """L1 on the 3D ``gate`` only (0 when ``gate_l1`` is off)."""
        return self.gate.abs() * self.gate_l1

    def _mask_untrusted_if_gate_closed(self, tac_mask: torch.Tensor) -> torch.Tensor:
        """Eval-only: drop 3D keys from softmax when ``|gate|`` is ~0.

        Train leaves the keys in so the value path can open the gate.
        Region tokens (``patch_token_span``) stay on ``patch_gate``.
        """
        if self.training:
            return tac_mask
        if float(self.gate.detach().abs()) > self.gate_mask_eps:
            return tac_mask
        start, count = self.patch_token_start, self.patch_token_count
        T = tac_mask.shape[1]
        if start + count > T:
            raise ValueError(
                f"region token span [{start}, {start + count}) does not fit the "
                f"tactile sequence T={T}; encoder and fusion disagree on the layout"
            )
        if count <= 0:
            return torch.zeros_like(tac_mask)
        keep = torch.zeros(T, dtype=torch.bool, device=tac_mask.device)
        keep[start : start + count] = True
        return tac_mask & keep.view(1, T)

    def _unused_append_touch(self, ref: torch.Tensor) -> torch.Tensor:
        """Keep append params in the graph when T=0 (region-only / no 3D)."""
        touch = self.gate.to(dtype=ref.dtype) * 0
        if self.patch_gate is not None:
            touch = touch + self.patch_gate.to(dtype=ref.dtype) * 0
        touch = touch + self.layer_embedding.weight.to(dtype=ref.dtype).sum() * 0
        if isinstance(self.in_proj, nn.Linear):
            touch = touch + self.in_proj.weight.to(dtype=ref.dtype).sum() * 0
            if self.in_proj.bias is not None:
                touch = touch + self.in_proj.bias.to(dtype=ref.dtype).sum() * 0
        return touch

    def _gate_tokens(
        self,
        ctx: torch.Tensor,
        present_region: Optional[torch.Tensor] = None,
        present_3d: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Scale the region block by ``patch_gate``, the 3D rest by ``gate``.

        Shared by the append path and by ``joint_gate_mode: value``.
        Built as one ``[B, T, 1]`` gate vector rather than in-place slice
        writes: both scalars stay differentiable and no uninitialised
        buffer can survive a layout change.

        ``present_region`` / ``present_3d`` are the encoder's post-dropout,
        post-quality-gate ``[B]`` flags. Rows they mark absent hold the
        learned ``no_tactile`` placeholder and are passed through UNGATED
        (scale 1) instead of being scaled by the gate: see the "gate applies
        only to real tactile" paragraph in the module docstring. ``None``
        means "every row is present" (bit-identical to the pre-0902 path).
        """
        B, T = int(ctx.shape[0]), int(ctx.shape[1])
        start, count = self.patch_token_start, self.patch_token_count
        if start + count > T:
            raise ValueError(
                f"region token span [{start}, {start + count}) does not fit the "
                f"tactile sequence T={T}; encoder and fusion disagree on the layout"
            )
        gate = self.gate.to(dtype=ctx.dtype)
        has_patch_gate = self.patch_gate is not None
        patch_gate = self.patch_gate.to(dtype=ctx.dtype) if has_patch_gate else gate
        one = torch.ones((), device=ctx.device, dtype=ctx.dtype)

        def _block(scalar: torch.Tensor, n: int, present: Optional[torch.Tensor]):
            if present is None:
                return scalar.view(1, 1, 1).expand(B, n, 1)
            present = present.to(device=ctx.device, dtype=torch.bool)
            if present.shape != (B,):
                raise ValueError(
                    f"tactile present mask must be [B]=({B},), "
                    f"got {tuple(present.shape)}"
                )
            return torch.where(
                present.view(B, 1, 1), scalar.view(1, 1, 1), one
            ).expand(B, n, 1)

        pieces: List[torch.Tensor] = [
            _block(scalar, n, present)
            for scalar, n, present in (
                (gate, start, present_3d),
                (patch_gate, count, present_region),
                (gate, T - start - count, present_3d),
            )
            if n
        ]
        out = torch.cat(pieces, dim=1) * ctx if pieces else ctx

        # DDP / ZeRO: both scalars must keep an autograd edge even when their
        # block is empty (layout) or every row in it is absent (all-V batch).
        touch = gate * 0
        if has_patch_gate:
            touch = touch + patch_gate * 0
        return out + touch

    def patch_to_temb(
        self,
        patch_cond: Optional[torch.Tensor],
        present: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """``[B, patch_cond_dim]`` → ``[B, dit_hidden]`` AdaLN addend, or None.

        NOT multiplied by ``patch_gate``. That scalar also scales the region
        STREAM tokens, which the DiT immediately LayerNorms — there its
        magnitude is swallowed above the ``sqrt(norm_eps)`` knee. The AdaLN
        addend is added to an UNNORMALISED ``temb``, so the same scalar is a
        linear attenuation here. One parameter cannot serve both semantics;
        it was the reason the FiLM path trained as a two-zero-init-factor
        product (``patch_gate * fc2``) and needed ~14k steps to bootstrap.
        ``fc2`` is zero-init, which keeps the step-0 "temb unchanged"
        contract on its own, and is the natural scale for this path.

        ``present=False`` zeros the row *after* the MLP so a learned bias
        cannot FiLM an absent glove. At init the last Linear is 0, so the
        addend is ~0 even when present.
        """
        if self.patch_mlp is None:
            return None
        if patch_cond is None:
            raise ValueError(
                "action_fusion.patch=temb requires patch_cond "
                f"[B, {self.patch_cond_dim}]"
            )
        if patch_cond.ndim != 2 or patch_cond.shape[-1] != self.patch_cond_dim:
            raise ValueError(
                f"patch_cond must be [B, {self.patch_cond_dim}], "
                f"got {tuple(patch_cond.shape)}"
            )
        x = patch_cond.to(dtype=self.patch_mlp.fc1.weight.dtype)
        temb = self.patch_mlp(x)
        if present is not None:
            present = present.to(device=temb.device, dtype=torch.bool)
            if present.shape != (temb.shape[0],):
                raise ValueError(
                    f"present must be [B]=({temb.shape[0]},), "
                    f"got {tuple(present.shape)}"
                )
            temb = temb * present.to(dtype=temb.dtype).view(temb.shape[0], 1)
        return temb

    @staticmethod
    def _present_any(
        present_region: Optional[torch.Tensor],
        present_3d: Optional[torch.Tensor],
        *,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """``present_region | present_3d``, or None when neither is known."""
        parts = [
            p.to(device=device, dtype=torch.bool)
            for p in (present_region, present_3d)
            if p is not None
        ]
        if not parts:
            return None
        out = parts[0]
        for extra in parts[1:]:
            out = out | extra
        return out

    def build_joint_cond(
        self,
        tactile_tokens: torch.Tensor,
        present_region: Optional[torch.Tensor] = None,
        present_3d: Optional[torch.Tensor] = None,
    ) -> Optional[dict]:
        """``joint_stream`` mode: tactile tokens for the DiT hidden stream.

        Returns the bundle the action head's ``joint_cond`` argument expects,
        or None when the laid-out sequence is empty (T=0).

        Under ``joint_gate_mode: value`` (default) the projected tokens are
        scaled here by the same per-segment ``_gate_tokens`` the append path
        uses, and the head leaves t2a attention free — the gate gradient is
        then ``<grad_tokens, tokens>``, not a 4.5e-5-suppressed attention
        term. Under ``bias`` the tokens go out unscaled and the head rebuilds
        the legacy additive gate. Either way the gate tensors are passed by
        reference: ``joint_a2t_gate`` is always read by the head's bias, and
        ``gate`` is still needed there for the eval hard-mask.
        """
        if self.mode != "joint_stream":
            raise ValueError("build_joint_cond is only valid in joint_stream mode")
        if tactile_tokens.ndim != 3:
            raise ValueError(
                f"tactile_tokens must be [B,T,D], got {tuple(tactile_tokens.shape)}"
            )
        B, T, D = tactile_tokens.shape
        if D != self.tactile_dim:
            raise ValueError(
                f"tactile_tokens dim {D} != encoder output_dim {self.tactile_dim}"
            )
        if T == 0:
            return None
        tokens = self.in_proj(tactile_tokens.to(self.gate.dtype))
        if self.joint_gate_mode == "value":
            tokens = self._gate_tokens(tokens, present_region, present_3d)
        return {
            "tokens": tokens,
            "region_span": (self.patch_token_start, self.patch_token_count),
            "gate_region": self.patch_gate if self.patch_gate is not None else self.gate,
            "gate_3d": self.gate,
            "gate_a2t": self.joint_a2t_gate,
            "gate_mask_eps": self.gate_mask_eps,
            "gate_mode": self.joint_gate_mode,
            "bias_scale": self.joint_bias_scale,
            # `[B]` "this row carries real tactile", or None when unknown.
            # The head's shortcut mask needs it: firing the VL mask on a row
            # with no tactile leaves the action expert with no context at all.
            "present": self._present_any(
                present_region, present_3d, device=tokens.device
            ),
        }

    def forward(
        self,
        vl_embs_list: Sequence[torch.Tensor],
        tactile_tokens: torch.Tensor,
        tactile_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        present_region: Optional[torch.Tensor] = None,
        present_3d: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Return ``(vl_embs_list_with_tactile, encoder_attention_mask)``.

        ``append_context`` mode only; joint_stream goes through
        ``build_joint_cond`` and leaves ``vl_embs_list`` untouched.
        ``present_*`` follow ``_gate_tokens``: absent rows keep their
        ``no_tactile`` placeholder ungated.
        """
        if self.mode != "append_context":
            raise ValueError(
                f"TactileActionFusion.forward is append_context-only; mode={self.mode!r} "
                "must use build_joint_cond."
            )
        if len(vl_embs_list) != self.num_dit_layers:
            raise ValueError(
                f"expected {self.num_dit_layers} context layers, got {len(vl_embs_list)}"
            )
        if tactile_tokens.ndim != 3:
            raise ValueError(
                f"tactile_tokens must be [B,T,D], got {tuple(tactile_tokens.shape)}"
            )
        B, T, D = tactile_tokens.shape
        if D != self.tactile_dim:
            raise ValueError(
                f"tactile_tokens dim {D} != encoder output_dim {self.tactile_dim}"
            )
        ref = vl_embs_list[0]
        if ref.shape[0] != B:
            raise ValueError(
                f"batch mismatch: VL context B={ref.shape[0]} vs tactile B={B}"
            )

        if T == 0:
            touch = self._unused_append_touch(ref)
            fused = [layer + touch.to(dtype=layer.dtype) for layer in vl_embs_list]
            L = ref.shape[1]
            if encoder_attention_mask is None:
                out_mask = torch.ones(B, L, dtype=torch.bool, device=ref.device)
            else:
                out_mask = encoder_attention_mask.to(
                    device=ref.device, dtype=torch.bool
                )
                if out_mask.shape != (B, L):
                    raise ValueError(
                        f"encoder_attention_mask must be [B,L]=({B},{L}), "
                        f"got {tuple(out_mask.shape)}"
                    )
            return fused, out_mask

        ctx = self.in_proj(tactile_tokens.to(self.gate.dtype))
        ctx = self._gate_tokens(ctx, present_region, present_3d)  # [B, T, H]

        fused: List[torch.Tensor] = []
        for idx, layer in enumerate(vl_embs_list):
            slot = self._cross_slot.get(idx, None)
            if slot is None:
                # Canonical forward hands this block encoder_hidden_states=None,
                # so the append is dropped; keep it anyway so every layer has
                # the same length as the shared mask.
                block = ctx
            else:
                block = ctx + self.layer_embedding.weight[slot].view(1, 1, -1)
            fused.append(torch.cat([layer, block.to(layer.dtype)], dim=1))

        L = ref.shape[1]
        device = ref.device
        if encoder_attention_mask is None:
            vl_mask = torch.ones(B, L, dtype=torch.bool, device=device)
        else:
            vl_mask = encoder_attention_mask.to(device=device, dtype=torch.bool)
            if vl_mask.shape != (B, L):
                raise ValueError(
                    f"encoder_attention_mask must be [B,L]=({B},{L}), "
                    f"got {tuple(vl_mask.shape)}"
                )
        if tactile_mask is None:
            tac_mask = torch.ones(B, T, dtype=torch.bool, device=device)
        else:
            tac_mask = tactile_mask.to(device=device, dtype=torch.bool)
            if tac_mask.shape != (B, T):
                raise ValueError(
                    f"tactile token_mask must be [B,T]=({B},{T}), "
                    f"got {tuple(tac_mask.shape)}"
                )
        tac_mask = self._mask_untrusted_if_gate_closed(tac_mask)
        return fused, torch.cat([vl_mask, tac_mask], dim=1)


def get_tactile_action_fusion(
    cfg: Any,
    *,
    num_dit_layers: int,
    dit_hidden_dim: int,
    tactile_dim: int,
    interleave_self_attention: bool,
    use_canonical_forward: bool = True,
    patch_cond_dim: int = 0,
    patch_token_start: int = 0,
    patch_token_count: int = 0,
) -> TactileActionFusion:
    return TactileActionFusion(
        cfg,
        num_dit_layers=num_dit_layers,
        dit_hidden_dim=dit_hidden_dim,
        tactile_dim=tactile_dim,
        interleave_self_attention=interleave_self_attention,
        use_canonical_forward=use_canonical_forward,
        patch_cond_dim=patch_cond_dim,
        patch_token_start=patch_token_start,
        patch_token_count=patch_token_count,
    )
