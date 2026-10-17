# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).
"""

# Standard Library
import argparse
import json
import os
import re
import time
from datetime import timedelta
from pathlib import Path
from typing import Tuple

# Third-Party Libraries
import numpy as np
import torch
import torch.distributed as dist

# NPU support: import torch_npu and enable automatic CUDA→NPU mapping.
# On GPU-only environments this is a no-op (ImportError is silently ignored).
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    pass

import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import GradientAccumulationPlugin, InitProcessGroupKwargs, set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

# Local Modules
from starVLA.dataloader import build_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.model.framework.action_contract import (
    assert_pred_matches_targets,
    stacked_action_targets,
)
from starVLA.training.trainer_utils.nan_fuse import assert_identical_rank_keys, grads_are_finite
from starVLA.training.trainer_utils.trainer_tools import (
    CHECKPOINT_FILENAME_RE,
    TrainerUtils,
    normalize_dotlist_args,
    resolve_resume_checkpoint,
    setup_optimizer_and_scheduler,
)

deepspeed_plugin = DeepSpeedPlugin()
# This trainer manually recreates its dataloader iterator and is step-based
# rather than epoch-based. Do not let Accelerate force an early synchronization
# on a short final dataloader batch: accumulation groups must continue across
# iterator resets so every optimizer step contains exactly the configured
# number of micro-batches. The launch wrapper exports the resolved YAML value.
gradient_accumulation_plugin = GradientAccumulationPlugin(
    num_steps=int(os.environ.get("ACCELERATE_GRADIENT_ACCUMULATION_STEPS", "1")),
    sync_with_dataloader=False,
)
# Accelerate defaults NCCL process-group timeout to 600s; rank-0 dataset cache
# builds (e.g. converted pose stats) can exceed that. Honor NCCL_TIMEOUT (seconds).
try:
    _ddp_timeout_s = int(os.environ.get("NCCL_TIMEOUT", "600"))
except ValueError as exc:
    raise ValueError("NCCL_TIMEOUT must be a positive integer number of seconds") from exc
if _ddp_timeout_s <= 0:
    raise ValueError(
        f"NCCL_TIMEOUT must be a positive integer number of seconds, got {_ddp_timeout_s}"
    )
accelerator = Accelerator(
    deepspeed_plugin=deepspeed_plugin,
    gradient_accumulation_plugin=gradient_accumulation_plugin,
    kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=_ddp_timeout_s))],
)
accelerator.print(accelerator.state)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
logger = get_logger(__name__)


def load_fast_tokenizer():
    return AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> DataLoader:
    """Prepare VLA training data."""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    if dist.is_initialized():
        dist.barrier()
    return vla_train_dataloader


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()
        self._step_metric_sums = {}
        self._step_metric_count = 0

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # Save config snapshots upfront so that even if a later setup step
        # (ckpt load / DeepSpeed init / dataloader build) crashes, the
        # produced run dir is still introspectable / from_pretrained-able.
        self._save_initial_configs()

        self._init_checkpointing()
        self._adjust_lr_scheduler_for_resume()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
        )

        self._init_wandb()

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """Initialize Weights & Biases (best-effort; must not block training)."""
        self._wandb_enabled = False
        if os.environ.get("WANDB_MODE") == "disabled" or os.environ.get("WANDB_DISABLED", "").lower() in {
            "1",
            "true",
            "yes",
        }:
            self.accelerator.wait_for_everyone()
            return
        if self.accelerator.is_main_process:
            try:
                wandb.init(
                    name=self.config.run_id,
                    dir=os.path.join(self.config.output_dir, "wandb"),
                    project=self.config.wandb_project,
                    entity=self.config.wandb_entity,
                    group="vla-train",
                )
                self._wandb_enabled = True
            except Exception as exc:
                logger.warning(f"W&B init failed; continuing without W&B: {exc}")
                self._wandb_enabled = False
        # Rendezvous after rank-0 W&B init. Otherwise a slow or failing init on
        # rank 0 lets the other ranks reach the first collective alone and
        # eventually hit an NCCL watchdog timeout.
        self.accelerator.wait_for_everyone()

    def _save_initial_configs(self):
        """Save full config and training script at the very start of training."""
        if not self.accelerator.is_main_process:
            return

        output_dir = Path(self.config.output_dir)

        # 1. Save config.full.yaml — the complete merged config (all parameters)
        if isinstance(self.config, AccessTrackedConfig):
            full_cfg = self.config.unwrap()
        else:
            full_cfg = self.config
        full_yaml_path = output_dir / "config.full.yaml"
        OmegaConf.save(full_cfg, full_yaml_path, resolve=True)
        logger.info(f"📝 Full config saved at {full_yaml_path}")

        # 2. Save config.yaml — accessed-only snapshot (will be updated at checkpoints)
        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
            logger.info(f"📊 Accessed config snapshot saved at {output_dir / 'config.yaml'}")

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        resume_from = getattr(self.config.trainer, "resume_from", None)
        self.resume_from_checkpoint = pretrained_checkpoint

        # An explicit `trainer.resume_from` continues a previous run (possibly
        # from another run directory): weights plus the step counter, so the LR
        # schedule picks up where it stopped instead of re-warming up. It wins
        # over `is_resume` (same-run auto-restart) and `pretrained_checkpoint`
        # (warm start, step counter reset to 0).
        if resume_from:
            self.resume_from_checkpoint, self.completed_steps = resolve_resume_checkpoint(resume_from)
            max_train_steps = self.config.trainer.max_train_steps
            if self.completed_steps >= max_train_steps:
                raise ValueError(
                    f"trainer.resume_from is at step {self.completed_steps}, at or past "
                    f"trainer.max_train_steps={max_train_steps}: nothing left to train. "
                    "Raise max_train_steps."
                )
            self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
            logger.info(
                f"Continuing training from {self.resume_from_checkpoint}, steps: {self.completed_steps}"
                f"/{max_train_steps}. Optimizer moments are NOT restored (checkpoints hold "
                "model weights only); only the LR schedule is fast-forwarded."
            )
            return

        if is_resume:
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                logger.info(
                    f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}"
                )
                return

            logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
            self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    def _adjust_lr_scheduler_for_resume(self):
        """Adjust LR scheduler state after resuming from non-zero steps."""
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(
                f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _save_checkpoint(self):
        """Save current training state."""
        # get_state_dict is a COLLECTIVE under DeepSpeed ZeRO-3 (it all-gathers
        # the partitioned parameters), so every rank must call it. Guarding it
        # behind is_main_process deadlocks: rank 0 blocks in the gather while
        # the other ranks wait at the barrier below. Under ZeRO-2/DDP this is
        # just a local state_dict copy on each rank.
        ds = getattr(self.accelerator.state, "deepspeed_plugin", None)
        is_zero3 = ds is not None and ds.zero_stage == 3

        state_dict = (
            self.accelerator.get_state_dict(self.model)
            if is_zero3 or self.accelerator.is_main_process
            else None
        )
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")

            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, checkpoint_path + "_model.safetensors")
            elif save_format == "pt":
                torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")

            if isinstance(self.config, AccessTrackedConfig):
                logger.info("📊 Saving accessed configuration...")
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
                logger.info("✅ Configuration files saved")

        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """Record training metrics."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        if self.completed_steps % self.config.trainer.logging_frequency == 0 and rank == 0:
            last_lrs = self.lr_scheduler.get_last_lr()
            for i, group in enumerate(self.optimizer.param_groups):
                group_name = group.get("name", str(i))
                metrics[f"learning_rate/{group_name}"] = last_lrs[i] if i < len(last_lrs) else last_lrs[-1]
            # completed_steps counts OPTIMIZER steps; each one consumes
            # gradient_accumulation_steps micro-batches from the dataloader.
            micro_batches_consumed = self.completed_steps * self.accelerator.gradient_accumulation_steps
            metrics["epoch"] = round(micro_batches_consumed / len(self.vla_train_dataloader), 2)
            if getattr(self, "_wandb_enabled", False):
                try:
                    wandb.log(metrics, step=self.completed_steps)
                except Exception as exc:
                    self._wandb_enabled = False
                    logger.warning(f"W&B log failed; disabling W&B: {exc}")
            logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """Create data iterators."""
        self.vla_iter = iter(self.vla_train_dataloader)

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """Execute training loop."""
        self._log_training_config()
        self._create_data_iterators()
        progress_bar = tqdm(
            total=self.config.trainer.max_train_steps,
            initial=self.completed_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()
            optimizer_stepped = step_metrics.pop("_optimizer_step")

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            # Everything below is keyed off `completed_steps` (an OPTIMIZER-step
            # counter). With gradient_accumulation_steps>1 a single optimizer
            # step spans several micro-step iterations, and `completed_steps`
            # only changes on the update micro-step. Guard the whole block with
            # the actual optimizer boundary (Accelerate for DDP, the engine for
            # DeepSpeed) so eval/logging/checkpoint fire exactly once per
            # optimizer step (unguarded, they would re-fire on every
            # mid-accumulation micro-step that shares the same completed_steps —
            # e.g. eval even at step 0 — wasting compute and consuming extra
            # training batches). At GA==1 every micro-step is a sync step, so
            # this is a no-op relative to the previous behaviour.
            if optimizer_stepped:
                progress_bar.update(1)
                self.completed_steps += 1

                if self.completed_steps % self.config.trainer.eval_interval == 0:
                    step_metrics = self.eval_action_model(step_metrics)

                step_metrics["timing/data"] = t_end_data - t_start_data
                step_metrics["timing/model"] = t_end_model - t_start_model
                self._log_metrics(step_metrics)

                if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                    self._save_checkpoint()

                if self.completed_steps >= self.config.trainer.max_train_steps:
                    break

        self._finalize_training()

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """Periodic action eval on a FIXED reference batch of training samples.

        The batch is captured once (at the first eval) and reused afterwards,
        so the curve is comparable across steps and eval stops consuming a
        fresh training batch every `eval_interval`. Caveats when reading the
        metric: the samples come from the TRAINING stream (no held-out split),
        and the sampler is the checkpoint's own (`num_inference_timesteps`),
        so this measures fit to the training distribution, not generalization
        or sample quality under a different step budget.
        """
        # ZeRO-3 partitions parameters across ranks: calling the unwrapped
        # module outside the engine forward sees empty (0-sized) weights and
        # crashes mid-run. Skip this informational eval instead of killing a
        # long training job; ZeRO-2/DDP keep full weights per rank and are fine.
        ds_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        if ds_plugin is not None and ds_plugin.zero_stage == 3:
            if not getattr(self, "_warned_zero3_eval", False):
                logger.warning(
                    "Skipping mid-training action eval: predict_action on the unwrapped "
                    "module is incompatible with DeepSpeed ZeRO-3 partitioned parameters."
                )
                self._warned_zero3_eval = True
            return step_metrics

        if getattr(self, "_eval_examples", None) is None:
            self._eval_examples = self._get_next_batch()
        examples = self._eval_examples
        actions = [example["action"] for example in examples]
        unwrapped = self.accelerator.unwrap_model(self.model)
        # The DiT has dropout (diffusion_model_cfg.dropout / final_dropout);
        # sampling in train mode would add dropout noise to the metric.
        was_training = unwrapped.training
        unwrapped.eval()
        output_dict = unwrapped.predict_action(examples=examples)
        if was_training:
            unwrapped.train()

        targets = stacked_action_targets(
            actions,
            action_horizon=int(unwrapped.action_horizon),
            action_dim=int(unwrapped.action_dim),
        )
        if self.accelerator.is_main_process:
            normalized_actions = output_dict["normalized_actions"]
            assert_pred_matches_targets(normalized_actions, targets)
            num_pots = np.prod(targets.shape)
            score = TrainerUtils.euclidean_distance(normalized_actions, targets)
            step_metrics["eval_train/fixed_batch_mse"] = score / num_pots

        if dist.is_initialized():
            dist.barrier()
        return step_metrics

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """Execute single training step.

        DDP accumulation is driven by `accelerator.accumulate`: the wrapped
        optimizer makes `step()`/`zero_grad()` no-ops on intermediate
        micro-steps and Accelerate scales the loss by the accumulation count.
        DeepSpeed ZeRO cannot use `accumulate()` because its `no_sync()` is
        incompatible with partitioned gradients, so its engine drives
        backward, stepping, loss scaling, clipping, and zeroing directly.

        Non-finite `action_loss` / `align_loss` / `contact_align_loss` / `future_contact_loss` /
        `total_loss` abort every rank before `backward`; non-finite local
        grads abort before the optimizer step. There is no skip-and-continue
        path — skipping backward on one rank hangs DDP.
        """
        self._sync_train_step_counter()
        is_deepspeed = hasattr(self.model, "is_gradient_accumulation_boundary") and hasattr(self.model, "backward")

        if is_deepspeed:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                total_loss = action_loss
                align_loss = output_dict.get("align_loss", None)
                if align_loss is not None:
                    total_loss = total_loss + align_loss
                contact_align_loss = output_dict.get("contact_align_loss", None)
                if contact_align_loss is not None:
                    total_loss = total_loss + contact_align_loss
                future_contact_loss = output_dict.get("future_contact_loss", None)
                if future_contact_loss is not None:
                    total_loss = total_loss + future_contact_loss

            loss_parts = {
                "action_loss": action_loss,
                "align_loss": align_loss,
                "contact_align_loss": contact_align_loss,
                "future_contact_loss": future_contact_loss,
                "total_loss": total_loss,
            }
            self._nan_fuse(loss_parts)
            self.model.backward(total_loss)
            self._nan_fuse(loss_parts, grads_ok=grads_are_finite(self.model))
            optimizer_stepped = bool(self.model.is_gradient_accumulation_boundary())
            self.model.step()
            if optimizer_stepped:
                self.lr_scheduler.step()
        else:
            with self.accelerator.accumulate(self.model):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output_dict = self.model.forward(batch_vla)
                    action_loss = output_dict["action_loss"]
                    total_loss = action_loss
                    # Optional spatial-alignment loss (QwenPI_v3 with
                    # framework.spatial_alignment.enable=true). The framework
                    # returns it ALREADY weighted by loss_coeff; baseline runs
                    # never emit this key, so their loss is exactly
                    # `action_loss` as before.
                    align_loss = output_dict.get("align_loss", None)
                    if align_loss is not None:
                        total_loss = total_loss + align_loss
                    # Optional contact-region visual alignment
                    # (framework.tactile_align.enable=true); already
                    # coeff-weighted. Absent when the module is off.
                    contact_align_loss = output_dict.get("contact_align_loss", None)
                    if contact_align_loss is not None:
                        total_loss = total_loss + contact_align_loss
                    future_contact_loss = output_dict.get("future_contact_loss", None)
                    if future_contact_loss is not None:
                        total_loss = total_loss + future_contact_loss

                loss_parts = {
                    "action_loss": action_loss,
                    "align_loss": align_loss,
                    "contact_align_loss": contact_align_loss,
                    "future_contact_loss": future_contact_loss,
                    "total_loss": total_loss,
                }
                self._nan_fuse(loss_parts)
                self.accelerator.backward(total_loss)
                self._nan_fuse(loss_parts, grads_ok=grads_are_finite(self.model))

                # Clip only once the full accumulation group's gradients exist.
                if self.config.trainer.gradient_clipping is not None and self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

                self.optimizer.step()
                optimizer_stepped = self.accelerator.sync_gradients
                if optimizer_stepped:
                    self.lr_scheduler.step()
                # The wrapped optimizer only clears on the sync micro-step, so
                # zeroing after step preserves earlier accumulated gradients.
                self.optimizer.zero_grad()

        # Accumulate detached metrics over the same micro-batches as the
        # gradients. On the optimizer boundary, reduce their local mean across
        # every process so logs describe the effective global optimizer step,
        # not just rank 0's final micro-batch.
        local_metrics = {"action_dit_loss": action_loss.detach().float()}
        # BFA++ pruning telemetry (only emitted when
        # framework.token_prune.enable=true, and then on every rank):
        # the surviving-token fraction plus three scorer-health gauges —
        # `prune_score_std` pinned near its ~1e-3 init or
        # `prune_topk_agreement` pinned near `beta` means the ranking is still
        # noise and the run is doing random token dropout, while
        # `prune_score_mean` drifting far from 0 (|mean| >~ 4) means the
        # sigmoid gate has saturated and the scorer has stopped learning.
        for key in (
            "prune_keep_ratio",
            "prune_score_std",
            "prune_score_mean",
            "prune_topk_agreement",
            # Tactile fusion gates (only emitted when
            # framework.tactile.action_fusion.enable=true, then on every
            # rank). tactile_gate = 3D taxel tokens, tactile_patch_gate =
            # region tokens, tactile_joint_a2t_gate = what the tactile rows
            # read back from the action stream. Under joint_stream these are
            # SWITCHES, not mixing weights (the DiT LayerNorms the stream, so
            # the loss stops caring above |gate|*std ~ 3.2e-3): read them as
            # "left the eps hole or not". The FiLM path's real openness is
            # tactile_extra_temb_rel_norm = ||extra_temb|| / ||timestep_emb||.
            "tactile_gate",
            "tactile_patch_gate",
            "tactile_joint_a2t_gate",
            "tactile_extra_temb_rel_norm",
        ):
            value = output_dict.get(key, None)
            if value is not None:
                local_metrics[key] = value.detach().float()
        if align_loss is not None:
            # `spatial_align_loss` is the weighted contribution to the total
            # (loss_coeff already applied), `spatial_align_cosine` the raw
            # `1 - cos` distance — the one to watch across loss_coeff sweeps.
            local_metrics["spatial_align_loss"] = align_loss.detach().float()
            align_cosine = output_dict.get("align_cosine", None)
            if align_cosine is not None:
                local_metrics["spatial_align_cosine"] = align_cosine.detach().float()
        if contact_align_loss is not None:
            local_metrics["contact_align_loss"] = contact_align_loss.detach().float()
            for key in (
                "contact_map_loss",
                "dice_loss",
                "infonce_loss",
                "pressure_bin_loss",
                "contact_align_present_frac",
            ):
                value = output_dict.get(key, None)
                if value is not None:
                    local_metrics[key] = value.detach().float()
        if future_contact_loss is not None:
            local_metrics["future_contact_loss"] = future_contact_loss.detach().float()
            for key in (
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
            ):
                value = output_dict.get(key, None)
                if value is not None:
                    local_metrics[key] = value.detach().float()

        for name, value in local_metrics.items():
            if name not in self._step_metric_sums:
                self._step_metric_sums[name] = torch.zeros_like(value)
            self._step_metric_sums[name] += value
        self._step_metric_count += 1

        if not optimizer_stepped:
            return {"_optimizer_step": False}

        if self._step_metric_sums:
            device = next(iter(self._step_metric_sums.values())).device
            assert_identical_rank_keys(
                self._step_metric_sums.keys(),
                device=device,
                step=self.completed_steps,
            )
        step_metrics = {"_optimizer_step": True}
        # Sorted: assert_identical_rank_keys hashes the set, not dict order.
        for name in sorted(self._step_metric_sums):
            value_sum = self._step_metric_sums[name]
            local_mean = value_sum / self._step_metric_count
            global_mean = self.accelerator.reduce(local_mean, reduction="mean")
            step_metrics[name] = global_mean.item()
        self._step_metric_sums.clear()
        self._step_metric_count = 0
        return step_metrics

    def _finalize_training(self):
        """Training end processing."""
        # Collective under ZeRO-3; must run on every rank (see _save_checkpoint).
        ds = getattr(self.accelerator.state, "deepspeed_plugin", None)
        is_zero3 = ds is not None and ds.zero_stage == 3

        state_dict = (
            self.accelerator.get_state_dict(self.model)
            if is_zero3 or self.accelerator.is_main_process
            else None
        )
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")

        if self.accelerator.is_main_process and getattr(self, "_wandb_enabled", False):
            try:
                wandb.finish()
            except Exception:
                pass

        self.accelerator.wait_for_everyone()


def validate_gradient_accumulation(cfg) -> None:
    """Fail fast when trainer.gradient_accumulation_steps did not reach accelerate.

    The YAML key is only effective if the launch environment exported it
    (train.sh sets ACCELERATE_GRADIENT_ACCUMULATION_STEPS from the YAML before
    `accelerate launch`). A DeepSpeed config file with a hard-coded
    gradient_accumulation_steps silently wins over that env var, so a mismatch
    here means the run would train with a different effective batch than the
    config snapshot claims — abort instead.
    """
    expected = int(OmegaConf.select(cfg, "trainer.gradient_accumulation_steps", default=1) or 1)
    actual = int(accelerator.gradient_accumulation_steps)
    ds_plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    ds_value = (
        ds_plugin.deepspeed_config.get("gradient_accumulation_steps", "auto") if ds_plugin is not None else actual
    )
    # DeepSpeed/accelerate may leave the value as "auto", an int, or a numeric
    # string ("4"); treat those as matching `expected` so a quoted JSON entry
    # does not false-positive abort a correctly configured run.
    if ds_value == "auto":
        ds_ok = True
    else:
        try:
            ds_ok = int(ds_value) == expected
        except (TypeError, ValueError):
            ds_ok = False
    if actual != expected or not ds_ok:
        raise ValueError(
            f"trainer.gradient_accumulation_steps={expected} but the launch environment resolved "
            f"gradient accumulation to {actual} (DeepSpeed config value: {ds_value!r}). "
            "Launch via train.sh (it exports ACCELERATE_GRADIENT_ACCUMULATION_STEPS from the training "
            "YAML), or make ACCELERATE_GRADIENT_ACCUMULATION_STEPS / the DeepSpeed config file's "
            "gradient_accumulation_steps ('auto' recommended) agree with the YAML."
        )

    # The per-rank check above cannot see CROSS-NODE mismatches: each node
    # resolves env > CLI > YAML on its own, and an
    # ACCELERATE_GRADIENT_ACCUMULATION_STEPS override exported on only some
    # nodes passes per-node validation while making ranks disagree on the
    # accumulation boundary (mismatched collectives -> wrong gradients or a
    # hang). Compare the resolved value across the whole world and abort early.
    if dist.is_initialized() and dist.get_world_size() > 1:
        value = torch.tensor([expected], dtype=torch.int64, device=accelerator.device)
        low, high = value.clone(), value.clone()
        dist.all_reduce(low, op=dist.ReduceOp.MIN)
        dist.all_reduce(high, op=dist.ReduceOp.MAX)
        if low.item() != high.item():
            raise ValueError(
                f"gradient_accumulation_steps differs across ranks (min={low.item()}, max={high.item()}, "
                f"this rank={expected}). Set ACCELERATE_GRADIENT_ACCUMULATION_STEPS identically on every "
                "node, or unset it everywhere so the shared training YAML decides."
            )


def sync_deepspeed_gradient_clipping(cfg) -> None:
    """Make trainer.gradient_clipping effective under DeepSpeed.

    Every GPU run of this trainer resolves to DeepSpeed (a DeepSpeedPlugin is
    always passed to Accelerator), and the engine only applies the clipping
    value carried by its own config: `accelerator.clip_grad_norm_` (the DDP
    fallback branch) is a no-op there, and accelerate fills a leftover "auto"
    with its hardcoded 1.0 default rather than the YAML value. Write the YAML
    value into the DeepSpeed config before `prepare()` so the training YAML is
    the single source of truth (0.0 = DeepSpeed's "no clipping").
    """
    ds_plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    if ds_plugin is None:
        return
    clip = getattr(cfg.trainer, "gradient_clipping", None)
    resolved = float(clip) if clip is not None else 0.0
    previous = ds_plugin.deepspeed_config.get("gradient_clipping", "auto")
    if previous not in ("auto", resolved):
        logger.warning(
            f"Overriding DeepSpeed config gradient_clipping={previous!r} with "
            f"trainer.gradient_clipping={resolved} (the training YAML wins)."
        )
    ds_plugin.deepspeed_config["gradient_clipping"] = resolved


def main(cfg) -> None:
    logger.info("VLA Training :: Warming Up")

    validate_gradient_accumulation(cfg)
    cfg = wrap_config(cfg)
    sync_deepspeed_gradient_clipping(cfg)
    logger.info("✅ Configuration wrapped for access tracking")

    output_dir = setup_directories(cfg=cfg)
    vla = build_framework(cfg)
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    trainer.prepare_training()
    trainer.train()

    logger.info("... and that's all, folks!")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/simBenchmarks/SimplerEnv/train_files/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # Normalise legacy YAML keys into the current `version_id == "0.21"` schema.
    # This is idempotent and does not modify framework class signatures.
    # See bar/config_tighten.md for the rationale.
    cfg = apply_config_compat(cfg)

    # Store source config path for later copying to output dir
    cfg.config_yaml = args.config_yaml

    # The process group is created by `Accelerator()` inside main(), so
    # `dist.is_initialized()` is still False here and the old guard made this
    # hook dead code. Gate on the launcher's RANK env instead. The other ranks
    # then block in the `Accelerator()` rendezvous until rank 0 resumes, which
    # is bounded by NCCL_TIMEOUT — raise it when you expect a long debug pause.
    if cfg.is_debug and int(os.environ.get("RANK", "0")) == 0:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
