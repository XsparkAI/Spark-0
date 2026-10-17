# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Local Spark-0 addition (NOT upstream starVLA).
"""
DA3 spatial representation alignment (Spatial-Forcing style) for QwenPI frameworks.

What this implements
────────────────────
The "Spatial Forcing" recipe (arXiv:2510.12276, ICLR 2026; reference impl:
openpi-SF / XPolicyLab policy/Spatial_Forcing), with Depth-Anything-3 (DA3,
arXiv:2511.10647) as the frozen 3D teacher instead of VGGT:

  1. The SAME training images that go into the VLM are also fed to a frozen
     DA3 backbone (DINOv2-style ViT with cross-view attention, trained with
     depth-ray supervision). Its intermediate patch tokens carry explicit
     3D-geometry information.
  2. A small trainable projector (LayerNorm -> Linear -> GELU -> Linear) maps
     the VLM's intermediate-layer *visual* tokens into the teacher feature
     space, and a per-token cosine-distance loss pulls them together.
  3. `total_loss = action_loss + loss_coeff * align_loss` — the VLM is thereby
     "forced" to encode spatial structure in exactly the hidden states that
     the layer-wise action DiT cross-attends to.

At inference NOTHING changes: the teacher is never loaded (lazy-loaded on the
first *training* forward only, then `dist.barrier` so NFS I/O skew cannot
desync ranks) and the projector is unused by `predict_action`. Disabling
`spatial_alignment.enable` reproduces the baseline model bit-for-bit (no
extra parameters / state_dict keys). `train.sh` also runs
`scripts/check_da3_teacher.sh` so a missing teacher dies at launch.

Camera / resolution generality
──────────────────────────────
Every view is handled on its own terms: per-view source resolution, per-view
VLM token grid (`image_grid_thw` may differ between cameras) and per-view
teacher input size. One teacher/student gather still needs a shared layout
across the samples it sees; mixed-mask training batches are split by layout
in `QwenPI_v3` (`layout_groups.py`) and the per-group losses are
sample-weighted. Views in one sample do not have to match each other.

`teacher_view_mode` decides how the teacher sees a sample's cameras:

  joint        all selected views enter one DA3 forward, so its cross-view
               attention can reason about them as one scene. Correct for
               genuinely overlapping, time-synchronised cameras. DA3 batches
               views as one tensor, so they must share an input size; when the
               per-view aspect ratios disagree the configured
               `teacher_image_size` is used verbatim for all of them.
  independent  every view is a separate single-view DA3 forward. Correct for
               wrist/ego cameras that share no field of view, and for history
               frames stacked as extra "views" — joint mode would otherwise
               let DA3 fuse unrelated scenes into each view's features.

Teacher-order caveat (multi-view, `joint` only)
───────────────────────────────────────────────
When >= 3 views are passed jointly, DA3 internally re-orders views around an
auto-selected reference view, and the exported aux features are NOT restored
to the input order (upstream behaviour). We therefore default to
`ref_view_strategy="first"`, which selects view 0 and keeps the input order —
safe for any camera count.

Config (all keys live under `framework.spatial_alignment` in the train YAML)
────────────────────────────────────────────────────────────────────────────
  enable: false                  # master switch; false == baseline behaviour
  teacher_model_path: ...        # local dir (offline) or HF repo id for
                                 # `DepthAnything3.from_pretrained`
  teacher_feature_dim: 1024      # backbone embed dim (vits 384 / vitb 768 /
                                 # vitl 1024 / vitg 1536); checked at runtime
  teacher_feat_layer: 23         # backbone block index to export (vitl: 0-23;
                                 # SF uses the LAST teacher layer). Negative
                                 # indices count from the end (-1 == last).
  teacher_image_size: 392        # int or [height, width]; each dim MUST be a
                                 # multiple of DA3's patch size 14
  teacher_preserve_aspect: true  # derive the teacher input from each view's
                                 # own aspect ratio (same token budget as
                                 # `teacher_image_size`, snapped to /14).
                                 # The "source" is the frame the VLM sees, i.e.
                                 # already resized to `obs_image_size`, so this
                                 # is a no-op unless obs_image_size is
                                 # non-square.
  teacher_view_mode: independent # `joint` | `independent` (see above)
  teacher_ref_view_strategy: first  # DA3 reference-view pick in `joint` mode
  teacher_chunk_size: 0          # >0 caps how many samples enter one teacher
                                 # forward (bounds peak VRAM on many cameras)
  teacher_dtype: bfloat16        # teacher compute dtype (float32 on CPU)
  align_view_indices: null       # null = every camera; else a list of unique
                                 # non-negative packed view indices (e.g. [0]
                                 # = first packed camera). Negatives / dups
                                 # are rejected at construct.
  align_layer_index: 25          # Qwen3-VL hidden_states index to supervise
                                 # (1-based layer output; hidden_states[0] is
                                 # the embedding). Negative indices follow
                                 # Python indexing over the L+1 hidden states
                                 # (-1 == last layer), so -12 ports the same
                                 # relative depth across VLM sizes. SF finds
                                 # ~2/3 depth best: 25/36 for Qwen3-VL-4B.
                                 # With interleave_self_attention the layerwise
                                 # DiT only cross-attends hidden_states[k] at
                                 # block k-1 when k-1 is EVEN, so pick an ODD
                                 # layer (23, 25, ...) — an even one (e.g. 24)
                                 # supervises features the DiT never reads.
  loss_coeff: 0.2                # weight of the alignment loss (openpi-SF
                                 # real-robot default; paper sweeps 0.02-0.5)
  use_vlm_norm: true             # LayerNorm on student tokens before the MLP
  target_spatial_zscore: false   # per-channel z-score of teacher tokens over
                                 # the token axis (openpi-SF `iSF_zscore`)
  add_target_pos_embed: true     # add 2D sin-cos PE to teacher features so
                                 # the supervised tokens keep positional order
                                 # (important per the SF ablation)
  target_pos_embed_scale: 0.1    # PE magnitude (openpi-SF `ratio=0.1`)
  projector_hidden_dim: null     # MLP hidden width; default 2*teacher dim
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.modules.spatial_alignment.layout_groups import parse_align_view_indices
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# ImageNet statistics — DA3's InputProcessor normalises with these.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# DA3 backbones are ViT/14: teacher input H/W must be multiples of this.
_DA3_PATCH_SIZE = 14


def _parse_hw(value, key: str) -> Tuple[int, int]:
    """Accept `392` or `[height, width]` and return a (height, width) pair."""
    if isinstance(value, (int, np.integer)):
        return (int(value), int(value))
    if isinstance(value, Sequence) and not isinstance(value, str) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"spatial_alignment.{key} must be an int or a [height, width] pair, got {value!r}.")


def _snap_to_patch(value: float) -> int:
    """Round to the nearest positive multiple of the DA3 patch size.

    Each dimension snaps independently, so aspect preservation is only exact
    up to one patch per side — negligible at the default 392 budget (28x28
    patches), visible only if `teacher_image_size` is dialled down to a few
    patches.
    """
    return max(_DA3_PATCH_SIZE, int(round(value / _DA3_PATCH_SIZE)) * _DA3_PATCH_SIZE)


def _build_2d_sincos_pos_embed(
    dim: int, grid_h: int, grid_w: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Standard 2D sin-cos positional embedding, shape [grid_h * grid_w, dim].

    Half of the channels encode the y coordinate, half the x coordinate.
    Serves the same purpose as openpi-SF's `_apply_pos_embed` on VGGT
    features: the cosine target must retain token order information, since
    the supervised VLM tokens live in an ordered sequence.
    """
    if dim % 4 != 0:
        raise ValueError(f"pos-embed dim must be divisible by 4, got {dim}.")

    def _embed_1d(d: int, positions: torch.Tensor) -> torch.Tensor:
        omega = torch.arange(d // 2, device=device, dtype=torch.float32) / (d // 2)
        omega = 1.0 / (10000.0**omega)  # [d/2]
        angles = positions.float()[:, None] * omega[None, :]  # [n, d/2]
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # [n, d]

    emb_y = _embed_1d(dim // 2, torch.arange(grid_h, device=device))  # [gh, dim/2]
    emb_x = _embed_1d(dim // 2, torch.arange(grid_w, device=device))  # [gw, dim/2]
    emb = torch.cat(
        [
            emb_y[:, None, :].expand(grid_h, grid_w, dim // 2),
            emb_x[None, :, :].expand(grid_h, grid_w, dim // 2),
        ],
        dim=-1,
    )  # [gh, gw, dim]
    return emb.reshape(grid_h * grid_w, dim).to(dtype)


def _image_hw(image) -> Tuple[int, int]:
    """(height, width) of a supported image object, without moving any pixels.

    Mirrors the layout rules of `_to_chw_float`, so probing a batch for its
    source resolution can never disagree with the actual conversion.
    """
    if isinstance(image, torch.Tensor):
        shape = tuple(int(v) for v in image.shape)
        if len(shape) == 2:
            return shape[0], shape[1]
        if len(shape) != 3:
            raise ValueError(f"spatial_alignment expects a CHW image tensor, got shape {shape}.")
        if shape[0] not in (1, 3) and shape[-1] in (1, 3, 4):
            return shape[0], shape[1]  # HWC tensor, as tolerated below
        return shape[1], shape[2]

    size = getattr(image, "size", None)
    if isinstance(size, tuple) and len(size) == 2:  # PIL reports (width, height)
        return int(size[1]), int(size[0])
    shape = np.asarray(image).shape
    if len(shape) not in (2, 3):
        raise ValueError(f"spatial_alignment expects an HWC image, got shape {shape}.")
    return int(shape[0]), int(shape[1])


def _to_chw_float(image) -> torch.Tensor:
    """Any supported image object -> CPU [3, H, W] float tensor in [0, 1].

    Accepts PIL images (any mode), HWC / HW numpy arrays (uint8 or float) and
    CHW tensors, so the teacher path does not silently depend on the exact
    object the dataloader or the caller happens to produce.

    The [0, 255] -> [0, 1] rescale is decided by the SOURCE DTYPE, not by the
    observed maximum: a legitimately dark uint8 frame has a max below 1 and a
    value-based test would leave it 255x too bright. Conversion stays on the
    CPU so the caller can stack the whole view and pay a single host-to-device
    copy per teacher forward.
    """
    if isinstance(image, torch.Tensor):
        pixels = image.detach().cpu()
        if pixels.ndim == 2:
            pixels = pixels[None]
        elif pixels.ndim != 3:
            raise ValueError(f"spatial_alignment expects a CHW image tensor, got shape {tuple(image.shape)}.")
        if pixels.shape[0] not in (1, 3) and pixels.shape[-1] in (1, 3, 4):
            pixels = pixels.permute(2, 0, 1)  # tolerate HWC tensors
        is_integer = not pixels.is_floating_point() and pixels.dtype is not torch.bool
        pixels = pixels.float()
    else:
        if hasattr(image, "convert"):  # PIL: normalise palette / RGBA / L to RGB
            image = image.convert("RGB")
        array = np.asarray(image)
        if array.ndim == 2:
            array = array[:, :, None]
        elif array.ndim != 3:
            raise ValueError(f"spatial_alignment expects an HWC image, got shape {array.shape}.")
        is_integer = np.issubdtype(array.dtype, np.integer)
        pixels = torch.from_numpy(np.ascontiguousarray(array)).float().permute(2, 0, 1)

    if pixels.shape[0] == 4:  # drop alpha
        pixels = pixels[:3]
    if pixels.shape[0] == 1:
        pixels = pixels.expand(3, -1, -1)
    if pixels.shape[0] != 3:
        raise ValueError(f"spatial_alignment expects 1/3/4 image channels, got {pixels.shape[0]}.")

    if is_integer:
        return pixels / 255.0
    # Float inputs are contracted to be already in [0, 1]. Rescaling them by a
    # value-based guess is what produced the dark-frame bug above, so an
    # out-of-range float is an error rather than a silent 255x mis-scale.
    if pixels.numel() and float(pixels.max()) > 1.5:
        raise ValueError(
            "spatial_alignment expects float images already normalised to [0, 1] "
            f"(got max {float(pixels.max()):.3f}); pass uint8 pixels instead."
        )
    return pixels


class DA3SpatialAlignment(nn.Module):
    """Trainable student projector + frozen DA3 teacher + cosine alignment loss.

    Only the projector (`vlm_norm`/`fc1`/`fc2`) holds trainable parameters and
    enters the checkpoint. The DA3 teacher is intentionally kept OUT of the
    nn.Module tree (plain list attribute) so that:
      - it never appears in `state_dict()` (checkpoints stay portable),
      - it never enters optimizer param groups / DDP buckets,
      - it is lazy-loaded on the first training forward only — policy servers
        that merely call `predict_action` never pay for it.
    """

    def __init__(self, vl_hidden_dim: int, sa_cfg) -> None:
        super().__init__()
        self.teacher_model_path = str(sa_cfg.get("teacher_model_path", "depth-anything/DA3-LARGE"))
        self.teacher_feature_dim = int(sa_cfg.get("teacher_feature_dim", 1024))
        self.teacher_feat_layer = int(sa_cfg.get("teacher_feat_layer", 23))
        self.teacher_image_size = _parse_hw(sa_cfg.get("teacher_image_size", 392), "teacher_image_size")
        self.teacher_preserve_aspect = bool(sa_cfg.get("teacher_preserve_aspect", True))
        self.teacher_ref_view_strategy = str(sa_cfg.get("teacher_ref_view_strategy", "first"))
        self.teacher_chunk_size = int(sa_cfg.get("teacher_chunk_size", 0) or 0)
        self.loss_coeff = float(sa_cfg.get("loss_coeff", 0.2))
        self.target_spatial_zscore = bool(sa_cfg.get("target_spatial_zscore", False))
        self.add_target_pos_embed = bool(sa_cfg.get("add_target_pos_embed", True))
        self.target_pos_embed_scale = float(sa_cfg.get("target_pos_embed_scale", 0.1))

        view_mode = str(sa_cfg.get("teacher_view_mode", "independent")).lower()
        if view_mode not in ("joint", "independent"):
            raise ValueError(
                f"spatial_alignment.teacher_view_mode must be 'joint' or 'independent', got {view_mode!r}."
            )
        self.teacher_view_mode = view_mode

        for dim, name in zip(self.teacher_image_size, ("height", "width")):
            if dim % _DA3_PATCH_SIZE != 0:
                raise ValueError(
                    f"spatial_alignment.teacher_image_size {name}={dim} must be a multiple of the "
                    f"DA3 patch size {_DA3_PATCH_SIZE} (e.g. 378 or 392)."
                )

        self.align_view_indices = parse_align_view_indices(
            sa_cfg.get("align_view_indices", None),
            field="spatial_alignment.align_view_indices",
        )

        dtype_name = str(sa_cfg.get("teacher_dtype", "bfloat16"))
        if not hasattr(torch, dtype_name) or not isinstance(getattr(torch, dtype_name), torch.dtype):
            raise ValueError(f"spatial_alignment.teacher_dtype={dtype_name!r} is not a torch dtype.")
        self.teacher_dtype = getattr(torch, dtype_name)

        # Student -> teacher-space projector, mirroring openpi-SF's
        # AlignProjector (LayerNorm -> fc1 -> GELU -> fc2). The teacher side
        # stays untouched: SF supervises the student, never adapts the target.
        projector_hidden_dim = sa_cfg.get("projector_hidden_dim", None) or 2 * self.teacher_feature_dim
        use_vlm_norm = bool(sa_cfg.get("use_vlm_norm", True))
        self.vlm_norm = nn.LayerNorm(vl_hidden_dim) if use_vlm_norm else nn.Identity()
        self.fc1 = nn.Linear(vl_hidden_dim, int(projector_hidden_dim))
        self.fc2 = nn.Linear(int(projector_hidden_dim), self.teacher_feature_dim)

        # Frozen teacher, lazily constructed; a plain list dodges nn.Module
        # registration (see class docstring for why that matters).
        self._teacher_holder: list = []
        self._warned: set = set()

    def _warn_once(self, key: str, message: str) -> None:
        """Log a per-condition warning once — these fire inside the train loop."""
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(message)

    # ──────────────────────────────────────────────────────────────────
    #  Frozen teacher
    # ──────────────────────────────────────────────────────────────────

    def _distributed_barrier_after_teacher_load(self) -> None:
        """Sync ranks after the first NFS / ``from_pretrained`` load.

        A rank that skipped every layout group used to never call
        ``_get_teacher`` while others did I/O and then hit NCCL on
        backward. Even when every rank loads, NFS skew on the first step
        is a hang. Single-process / uninitialized dist is a no-op.
        """
        if not dist.is_available() or not dist.is_initialized():
            return
        if dist.get_world_size() < 2:
            return
        dist.barrier()

    def ensure_teacher_loaded(self, device: torch.device) -> nn.Module:
        """Load the teacher on this rank, then barrier on the first load.

        Call this at the start of every training DA3 forward — including
        batches that align no group — so the load is rank-invariant.
        """
        first = not self._teacher_holder
        teacher = self._get_teacher(device)
        if first:
            self._distributed_barrier_after_teacher_load()
        return teacher

    def _get_teacher(self, device: torch.device) -> nn.Module:
        """Load the DA3 backbone once per process (training ranks only)."""
        if self._teacher_holder:
            # Returned as-is: a rank's device never changes, and `.to()` would
            # walk every parameter of the ViT on each training step.
            return self._teacher_holder[0]

        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError as e:  # pragma: no cover - depends on training env
            raise ImportError(
                "framework.spatial_alignment.enable=true requires the `depth_anything_3` "
                "package in the TRAINING environment (inference never needs it). Install "
                "with: pip install 'git+https://github.com/ByteDance-Seed/Depth-Anything-3' "
                "and download the teacher weights to spatial_alignment.teacher_model_path."
            ) from e

        # DA3 logs an INFO line per forward when it selects a reference view
        # (>= 3 joint views), which floods training logs. Downgrade its custom
        # logger to WARN unless the user explicitly chose a level.
        if "DA3_LOG_LEVEL" not in os.environ:
            try:
                from depth_anything_3.utils.logger import LOG_LEVELS, logger as da3_logger

                da3_logger.level = LOG_LEVELS["WARN"]
            except Exception:
                pass

        da3 = DepthAnything3.from_pretrained(self.teacher_model_path)
        net = da3.model
        # Nested presets (da3nested-*) wrap the any-view net; the metric branch
        # is irrelevant for feature distillation.
        if not hasattr(net, "backbone") and hasattr(net, "da3"):
            net = net.da3
        if not hasattr(net, "backbone"):
            raise ValueError(
                f"Unsupported DA3 preset at {self.teacher_model_path!r}: "
                "expected a DepthAnything3Net with a `.backbone` attribute."
            )
        # Keep ONLY the ViT backbone; DPT / camera heads are dead weight here.
        teacher = net.backbone
        teacher.eval()
        teacher.requires_grad_(False)

        # bf16 is a CUDA-only assumption; CPU debug runs fall back to fp32.
        if device.type != "cuda" and self.teacher_dtype is not torch.float32:
            logger.warning(
                f"[spatial_alignment] teacher_dtype={self.teacher_dtype} is unsupported on "
                f"{device.type}; falling back to float32."
            )
            self.teacher_dtype = torch.float32
        teacher.to(device=device, dtype=self.teacher_dtype)

        num_blocks = len(teacher.pretrained.blocks)
        if self.teacher_feat_layer < 0:
            self.teacher_feat_layer += num_blocks
        if not (0 <= self.teacher_feat_layer < num_blocks):
            raise ValueError(
                f"spatial_alignment.teacher_feat_layer={self.teacher_feat_layer} out of range "
                f"for a {num_blocks}-block DA3 backbone."
            )

        self._teacher_holder.append(teacher)
        return teacher

    # ──────────────────────────────────────────────────────────────────
    #  Teacher inputs
    # ──────────────────────────────────────────────────────────────────

    def _teacher_input_size(self, source_hw: Tuple[int, int]) -> Tuple[int, int]:
        """Teacher resolution for a source image, snapped to the patch grid.

        With `teacher_preserve_aspect` the configured size only fixes the
        token BUDGET: a 3:4 source becomes 336x448 instead of a squashed
        392x392, so DA3 sees the scene at its true aspect ratio. Square
        sources reproduce `teacher_image_size` exactly.

        Note the source here is what the VLM was fed, i.e. the dataloader's
        `obs_image_size` frame, not the raw camera — this only does anything
        when `obs_image_size` itself is non-square.
        """
        ref_h, ref_w = self.teacher_image_size
        if not self.teacher_preserve_aspect:
            return ref_h, ref_w
        src_h, src_w = source_hw
        if src_h <= 0 or src_w <= 0:
            return ref_h, ref_w
        scale = math.sqrt((ref_h * ref_w) / float(src_h * src_w))
        return _snap_to_patch(src_h * scale), _snap_to_patch(src_w * scale)

    def _prepare_view_pixels(
        self, batch_images: Sequence[Sequence], view_idx: int, target_hw: Tuple[int, int], device: torch.device
    ) -> torch.Tensor:
        """One camera across the batch -> normalized [B, 3, Ht, Wt].

        Samples usually share a resolution, which lets us stack on the host and
        pay a single transfer plus one interpolate call for the whole view;
        heterogeneous samples are resized individually (still device-side)
        instead of failing to stack.
        """
        images = [_to_chw_float(sample[view_idx]) for sample in batch_images]
        shapes = {tuple(img.shape) for img in images}
        if len(shapes) == 1:
            pixels = torch.stack(images, dim=0).to(device=device, non_blocking=True)
            if pixels.shape[-2:] != target_hw:
                pixels = F.interpolate(pixels, size=target_hw, mode="bilinear", align_corners=False)
        else:
            pixels = torch.stack(
                [
                    img
                    if img.shape[-2:] == target_hw
                    else F.interpolate(img[None], size=target_hw, mode="bilinear", align_corners=False)[0]
                    for img in (image.to(device=device, non_blocking=True) for image in images)
                ],
                dim=0,
            )

        mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
        return (pixels - mean) / std

    def _resolve_view_indices(self, views_per_sample: int) -> List[int]:
        if self.align_view_indices is None:
            return list(range(views_per_sample))
        out_of_range = [i for i in self.align_view_indices if not (0 <= i < views_per_sample)]
        if out_of_range:
            raise ValueError(
                f"spatial_alignment.align_view_indices={self.align_view_indices} references view(s) "
                f"{out_of_range}, but the batch only provides {views_per_sample} camera(s)."
            )
        return list(self.align_view_indices)

    def _run_teacher(self, teacher: nn.Module, pixels: torch.Tensor) -> torch.Tensor:
        """Frozen DA3 forward. [N, S, 3, H, W] -> aux patch features [N, S, P, C] (fp32)."""
        device_type = pixels.device.type
        chunk = self.teacher_chunk_size if self.teacher_chunk_size > 0 else pixels.shape[0]
        outputs = []
        for start in range(0, pixels.shape[0], chunk):
            part = pixels[start : start + chunk].to(self.teacher_dtype)
            with torch.autocast(device_type, dtype=self.teacher_dtype, enabled=device_type == "cuda"):
                # get_intermediate_layers returns (main outputs, aux outputs);
                # aux = LayerNorm'ed patch tokens of `export_feat_layers` blocks
                # with cls/register tokens already stripped.
                # We call the inner ViT directly instead of DinoV2.forward so that
                # `n` selects a single block: the wrapper would build main outputs
                # (2x-width cat_token tensors + LayerNorm) for all of DA3's DPT
                # `out_layers`, which we would just throw away.
                _, aux_features = teacher.pretrained.get_intermediate_layers(
                    part,
                    n=[self.teacher_feat_layer],
                    export_feat_layers=[self.teacher_feat_layer],
                    ref_view_strategy=self.teacher_ref_view_strategy,
                )
            outputs.append(aux_features[0].float())
        if not outputs:
            raise ValueError("spatial_alignment received an empty image batch.")
        return torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]

    def _resample_to_student_grid(
        self, patches: torch.Tensor, patch_hw: Tuple[int, int], student_grid: Tuple[int, int]
    ) -> torch.Tensor:
        """[N, P, C] teacher patches -> [N, gh*gw, C] on the student's token grid."""
        patch_h, patch_w = patch_hw
        grid_h, grid_w = student_grid
        n, num_patches, channels = patches.shape
        if num_patches != patch_h * patch_w:
            raise ValueError(
                f"DA3 returned {num_patches} patches but the teacher grid is {patch_h}x{patch_w}; "
                "the teacher input size and its patch grid disagree."
            )
        target = patches.reshape(n, patch_h, patch_w, channels).permute(0, 3, 1, 2)  # [N, C, ph, pw]

        if self.add_target_pos_embed:
            pos = _build_2d_sincos_pos_embed(channels, patch_h, patch_w, target.device, target.dtype)
            pos = pos.reshape(patch_h, patch_w, channels).permute(2, 0, 1)[None]
            target = target + self.target_pos_embed_scale * pos

        # Teacher patch grid -> student token grid (row-major both sides: DA3
        # patch tokens and Qwen3-VL merged visual tokens are raster order).
        # align_corners=False keeps the cell-centre convention used when the
        # pixels were resized, so patch i maps to the same image region on
        # both sides.
        if (patch_h, patch_w) != (grid_h, grid_w):
            target = F.interpolate(target, size=(grid_h, grid_w), mode="bilinear", align_corners=False)
        return target.flatten(2).permute(0, 2, 1)  # [N, gh*gw, C]

    @torch.no_grad()
    def _extract_teacher_targets(
        self,
        batch_images: Sequence[Sequence],
        student_grids: Sequence[Tuple[int, int]],
        view_indices: Sequence[int],
        device: torch.device,
    ) -> Dict[int, torch.Tensor]:
        """Frozen DA3 forward -> per-view targets {view_idx: [B, gh*gw, C]} (fp32)."""
        teacher = self._get_teacher(device)

        # The teacher input size follows each camera's aspect ratio, so it is
        # resolved from the WHOLE batch: one sample cannot speak for the rest.
        # Samples of one camera that disagree have no shared aspect ratio to
        # preserve, so that camera falls back to the configured size.
        teacher_hw = {}
        for view_idx in view_indices:
            source_hw = {_image_hw(sample[view_idx]) for sample in batch_images}
            # Compare the RESOLVED sizes: samples of one camera that differ in
            # resolution but share an aspect ratio still agree here, and only
            # a genuine aspect-ratio conflict forces the square fallback.
            resolved = {self._teacher_input_size(hw) for hw in source_hw}
            if len(resolved) > 1:
                self._warn_once(
                    f"mixed_source_{view_idx}",
                    f"[spatial_alignment] view {view_idx} arrives at mixed aspect ratios across the "
                    f"batch ({sorted(source_hw)}); no single one can be preserved, so it uses "
                    f"teacher_image_size={self.teacher_image_size}.",
                )
                teacher_hw[view_idx] = self.teacher_image_size
            else:
                teacher_hw[view_idx] = next(iter(resolved))

        # DA3 batches a sample's views into one tensor, so a joint forward
        # needs one common input size. Mixed aspect ratios fall back to the
        # configured size rather than silently cropping one camera's geometry.
        if self.teacher_view_mode == "joint" and len(set(teacher_hw.values())) > 1:
            self._warn_once(
                "mixed_aspect",
                f"[spatial_alignment] cameras need different teacher input sizes ({teacher_hw}); "
                f"joint DA3 forwards need one input size, so all views use teacher_image_size="
                f"{self.teacher_image_size}. Set teacher_view_mode=independent to keep each "
                "camera's aspect ratio.",
            )
            teacher_hw = {view_idx: self.teacher_image_size for view_idx in view_indices}

        # `joint` puts every view of a sample in one forward (cross-view
        # attention); `independent` gives each view its own single-view group.
        groups = [list(view_indices)] if self.teacher_view_mode == "joint" else [[i] for i in view_indices]

        targets: Dict[int, torch.Tensor] = {}
        for group in groups:
            group_hw = teacher_hw[group[0]]
            pixels = torch.stack(
                [self._prepare_view_pixels(batch_images, view_idx, group_hw, device) for view_idx in group],
                dim=1,
            )  # [B, S_group, 3, Ht, Wt]
            features = self._run_teacher(teacher, pixels)  # [B, S_group, P, C]

            if features.shape[-1] != self.teacher_feature_dim:
                raise ValueError(
                    f"DA3 teacher produced {features.shape[-1]}-dim features but "
                    f"spatial_alignment.teacher_feature_dim={self.teacher_feature_dim}; "
                    "fix the config to match the loaded backbone size."
                )

            patch_hw = (group_hw[0] // _DA3_PATCH_SIZE, group_hw[1] // _DA3_PATCH_SIZE)
            for position, view_idx in enumerate(group):
                student_grid = student_grids[view_idx]
                target = self._resample_to_student_grid(features[:, position], patch_hw, student_grid)
                if self.target_spatial_zscore:
                    # Per-sample, per-channel standardization over the token
                    # axis (openpi-SF `spatial_zscore`).
                    target = (target - target.mean(dim=1, keepdim=True)) / (target.std(dim=1, keepdim=True) + 1e-6)
                targets[view_idx] = target
        return targets

    # ──────────────────────────────────────────────────────────────────
    #  Loss
    # ──────────────────────────────────────────────────────────────────

    def teacher_targets(
        self,
        batch_images: Sequence[Sequence],
        student_grids: Sequence[Tuple[int, int]],
        device: torch.device,
    ) -> Tuple[List[int], Dict[int, torch.Tensor]]:
        """Frozen-teacher targets for ONE homogeneous-layout group.

        Returns ``(view_indices, {view_idx: [B, gh*gw, C]})``. The teacher is
        outside the nn.Module tree and runs under ``no_grad``, so this touches
        no trainable parameter and a rank MAY call it a data-dependent number
        of times (once per layout group). The projector may not — see
        ``align_segments``.
        """
        views_per_sample = len(batch_images[0])
        if any(len(images) != views_per_sample for images in batch_images):
            raise ValueError("spatial_alignment expects the same camera count for every batch sample.")
        if len(student_grids) != views_per_sample:
            raise ValueError(
                f"Group provides {views_per_sample} camera(s) but {len(student_grids)} token grid(s)."
            )
        view_indices = self._resolve_view_indices(views_per_sample)
        targets = self._extract_teacher_targets(batch_images, student_grids, view_indices, device)
        return view_indices, targets

    def align_segments(
        self,
        segments: Sequence[Tuple[torch.Tensor, torch.Tensor, float]],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Weighted `1 - cosine` over pre-paired (student, target, weight) segments.

        Runs the trainable projector EXACTLY ONCE, whatever the batch's view
        layout. That invariant is load-bearing under DeepSpeed ZeRO-3: a
        module forward all-gathers the partitioned weights, which is a
        collective, so a projector call count that depends on the local
        batch's layout (per view, per layout group, or skip-vs-run) makes
        ranks issue different numbers of collectives and the job deadlocks.
        `fc1`/`fc2` are both well past `stage3_param_persistence_threshold`,
        so they really are re-gathered per forward.

        The projector is token-wise (LayerNorm -> Linear -> GELU -> Linear),
        so concatenating every segment's tokens along the token axis and
        splitting the result back is mathematically identical to projecting
        each segment on its own.

        An empty ``segments`` still projects a zero row so the call count and
        the DDP/ZeRO parameter set stay identical to a batch that aligns; the
        returned loss is 0 and its gradient is 0.

        Args:
            segments: ``(student [B, P, C], target [B, P, D], weight)`` triples.
                Weights are expected to sum to 1 (see
                ``layout_groups.normalized_segment_weights``).
        """
        if not segments:
            dummy = torch.zeros(
                1, self.fc1.in_features, device=device, dtype=self.fc1.weight.dtype
            )
            projected = self.fc2(F.gelu(self.fc1(self.vlm_norm(dummy))))
            return projected.float().sum() * 0.0

        students, targets, sizes = [], [], []
        for position, (student, target, _) in enumerate(segments):
            if student.shape[:2] != target.shape[:2]:
                raise ValueError(
                    f"Segment {position}: student/teacher token count mismatch — student "
                    f"{tuple(student.shape[:2])} vs teacher {tuple(target.shape[:2])}."
                )
            students.append(student.reshape(-1, student.shape[-1]))
            targets.append(target.reshape(-1, target.shape[-1]))
            sizes.append(students[-1].shape[0])

        # Match the projector's parameter dtype so the module also works
        # outside an autocast context (e.g. debug __main__ runs); under the
        # trainer's bf16 autocast this cast is effectively free.
        flat_students = torch.cat(students, dim=0).to(self.fc1.weight.dtype)
        projected = self.fc2(F.gelu(self.fc1(self.vlm_norm(flat_students))))
        flat_targets = torch.cat(targets, dim=0)

        # Cosine in fp32 for numerical stability (ambient autocast is bf16).
        with torch.autocast(device.type, enabled=False):
            projected = F.normalize(projected.float(), dim=-1, eps=1e-6)
            flat_targets = F.normalize(flat_targets.float(), dim=-1, eps=1e-6)
            per_token = 1.0 - (projected * flat_targets).sum(dim=-1)  # [N]
            segment_means = torch.stack(
                [part.mean() for part in torch.split(per_token, sizes)]
            )
            weights = torch.tensor(
                [float(weight) for _, _, weight in segments],
                device=segment_means.device,
                dtype=segment_means.dtype,
            )
            return (segment_means * weights).sum()

    def forward(
        self,
        student_views: Sequence[torch.Tensor],
        batch_images: Sequence[Sequence],
        student_grids: Sequence[Tuple[int, int]],
    ) -> torch.Tensor:
        """Unweighted alignment loss (mean of per-token `1 - cosine`).

        Single-layout convenience wrapper over ``teacher_targets`` +
        ``align_segments``; mixed-layout training batches go through
        ``QwenPI_v3`` instead, which pairs every layout group before making
        the one projector call.

        Averaging is per view first, then across views, so cameras with
        different token grids contribute equally instead of being weighted by
        resolution. Identical to a flat token mean when all views share a grid.

        Args:
            student_views: one `[B, gh_i * gw_i, vl_hidden_dim]` tensor per
                camera — the VLM's visual tokens at the supervised layer, in
                raster order (gradients DO flow into the VLM through these).
            batch_images: per-sample image lists, same order as the VLM input.
            student_grids: per-camera `(gh, gw)` token grid of the VLM.
        """
        if len(student_views) != len(student_grids):
            raise ValueError(
                f"Got {len(student_views)} student view tensors but {len(student_grids)} token grids."
            )
        views_per_sample = len(batch_images[0])
        if views_per_sample != len(student_views):
            raise ValueError(
                f"Batch provides {views_per_sample} camera(s) but the VLM produced "
                f"{len(student_views)} visual token group(s)."
            )

        device = student_views[0].device
        view_indices, targets = self.teacher_targets(batch_images, student_grids, device)
        weight = 1.0 / len(view_indices)
        segments = [(student_views[i], targets[i], weight) for i in view_indices]
        return self.align_segments(segments, device=device)
