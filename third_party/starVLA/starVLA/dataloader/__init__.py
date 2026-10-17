import json
import os
from accelerate.logging import get_logger
import numpy as np
from torch.utils.data import DataLoader
import torch.distributed as dist
from pathlib import Path
from starVLA.dataloader.vlm_datasets import make_vlm_dataloader

logger = get_logger(__name__)

def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")



def apply_pretrained_statistics(dataset, checkpoint, cfg):
    """Apply the checkpoint run's statistics to the current data transforms."""
    import yaml
    from starVLA.dataloader.gr00t_lerobot.datasets import get_used_modality_keys

    checkpoint = Path(checkpoint)
    if checkpoint.parent.name not in {"checkpoints", "final_model"}:
        raise ValueError("pretrained_checkpoint must be a file inside <run>/checkpoints/ or <run>/final_model/")
    stats_path = checkpoint.parent.parent / "dataset_statistics.json"
    with (stats_path.parent / "config.full.yaml").open("r", encoding="utf-8") as stream:
        source_config = yaml.safe_load(stream)
    source_spec = source_config["embodiment_spec"]
    for field in ("embodiment_tag", "state_keys", "action_keys", "pose_representation",
                  "action_horizon", "state_normalization", "action_normalization"):
        current_value = cfg.embodiment_spec.get(field)
        source_value = source_spec.get(field)
        if field in ("state_keys", "action_keys"):
            matches = list(current_value.items()) == list(source_value.items())
        else:
            matches = current_value == source_value
        if not matches:
            raise ValueError(f"checkpoint and current embodiment_spec.{field} differ")
    source_data = source_config["datasets"]["vla_data"]
    for field in ("action_mode", "action_mode_apply_keys", "action_mode_state_map"):
        if cfg.datasets.vla_data.get(field) != source_data.get(field):
            raise ValueError(f"checkpoint and current datasets.vla_data.{field} differ")
    with stats_path.open("r", encoding="utf-8") as stream:
        pretrained = json.load(stream)

    used_keys = {"action": {}, "state": {}}
    for member in dataset.datasets:
        action_keys, state_keys = get_used_modality_keys(member.modality_keys)
        used_keys["action"].update(dict.fromkeys(action_keys))
        used_keys["state"].update(dict.fromkeys(state_keys))

    fields = ("mean", "std", "min", "max", "q01", "q99")
    if set(pretrained) != set(dataset.merged_metadata):
        raise ValueError("checkpoint statistics embodiment tags differ from the current dataset")
    for tag, metadata in dataset.merged_metadata.items():
        for modality in ("action", "state"):
            current = getattr(metadata.statistics, modality)
            keys = [key for key in used_keys[modality] if key in current]
            widths = [len(current[key].mean) for key in keys]
            boundaries = np.cumsum(widths)[:-1]
            expected = sum(widths)
            source = pretrained[tag].get(modality)
            if not isinstance(source, dict):
                raise ValueError(f"checkpoint statistics are missing {tag}.{modality}")
            for field in fields:
                try:
                    values = np.asarray(source.get(field), dtype=float)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"checkpoint {tag}.{modality}.{field} is not numeric") from exc
                if values.ndim != 1 or values.size != expected or not np.isfinite(values).all():
                    raise ValueError(f"checkpoint {tag}.{modality}.{field} has invalid values or dimension")
                for key, chunk in zip(keys, np.split(values, boundaries)):
                    setattr(current[key], field, chunk)

            per_row = pretrained[tag].get(f"{modality}_per_row", {})
            if not isinstance(per_row, dict):
                raise ValueError(f"checkpoint {tag}.{modality}_per_row is malformed")
            for key in keys:
                row_stats = per_row.get(key, {})
                if not isinstance(row_stats, dict):
                    raise ValueError(f"checkpoint {tag}.{modality}_per_row.{key} is malformed")
                for field in ("q01_per_row", "q99_per_row"):
                    values = row_stats.get(field)
                    existing = getattr(current[key], field)
                    if values is None and existing is None:
                        continue
                    if values is None:
                        raise ValueError(f"checkpoint is missing {tag}.{modality}.{key}.{field}")
                    try:
                        rows = np.asarray(values, dtype=float)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"checkpoint {tag}.{modality}.{key}.{field} is not numeric") from exc
                    if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] != len(current[key].mean) or not np.isfinite(rows).all():
                        raise ValueError(f"checkpoint {tag}.{modality}.{key}.{field} has invalid values or width")
                    if existing is not None and rows.shape != np.asarray(existing).shape:
                        raise ValueError(f"checkpoint {tag}.{modality}.{key}.{field} has an incompatible horizon")
                    setattr(current[key], field, rows)

    for member in dataset.datasets:
        member.set_transforms_metadata(dataset.merged_metadata[member.tag])
    logger.info(f"Using pretrained normalization statistics from {stats_path}")


def build_dataloader(cfg, dataset_py="lerobot_datasets_oxe"): # TODO now here only is get dataset, we need mv dataloader to here

    if dataset_py == "lerobot_datasets":
        from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
        vla_dataset_cfg = cfg.datasets.vla_data

        # The mixture's per-index sampling RNG must follow the run seed:
        # without this every run draws the SAME (dataset, trajectory, frame)
        # sequence regardless of `seed`, so multi-seed runs are not
        # independent resamples of the data. NB: `seed: 0` is a legitimate
        # seed (all shipped recipes use it) — only fall back to 42 when the
        # key is absent/None, never on falsiness.
        run_seed = getattr(cfg, "seed", None)
        vla_dataset = get_vla_dataset(
            data_cfg=vla_dataset_cfg,
            balance_dataset_weights=vla_dataset_cfg.get("balance_dataset_weights", False),
            balance_trajectory_weights=vla_dataset_cfg.get("balance_trajectory_weights", False),
            seed=42 if run_seed is None else int(run_seed),
            framework=getattr(cfg, "framework", None),
        )
        if bool(getattr(cfg.trainer, "use_pretrained_stats", False)):
            checkpoint = getattr(cfg.trainer, "pretrained_checkpoint", None)
            if not checkpoint:
                raise ValueError("trainer.use_pretrained_stats requires trainer.pretrained_checkpoint")
            apply_pretrained_statistics(vla_dataset, checkpoint, cfg)
        num_workers = int(vla_dataset_cfg.get("num_workers", 4))
        dataloader_kwargs = {
            "batch_size": cfg.datasets.vla_data.per_device_batch_size,
            "collate_fn": collate_fn,
            "num_workers": num_workers,
            "pin_memory": bool(vla_dataset_cfg.get("pin_memory", True)),
            # shuffle=True: accelerate installs DistributedSampler so DDP
            # ranks see a per-epoch permutation. The mixture's (epoch, index)
            # RNG also needs the new epoch: set_epoch writes a share_memory
            # buffer persistent workers can read (a plain attribute stays 0).
            "shuffle": True,
        }
        if num_workers > 0:
            if not getattr(vla_dataset, "_epoch_shared_ok", True):
                raise RuntimeError(
                    "LeRobotMixtureDataset._epoch_shared.share_memory_() failed; "
                    "DataLoader workers would redraw the same epoch-0 sequence. "
                    "Set datasets.vla_data.num_workers=0 or use a start method "
                    "that can share tensors."
                )
            dataloader_kwargs["persistent_workers"] = bool(vla_dataset_cfg.get("persistent_workers", True))
            dataloader_kwargs["prefetch_factor"] = int(vla_dataset_cfg.get("prefetch_factor", 2))

        vla_train_dataloader = DataLoader(
            vla_dataset,
            **dataloader_kwargs,
        )
        if not dist.is_initialized() or dist.get_rank() == 0:
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
    elif dataset_py == "vlm_datasets":
        vlm_data_module = make_vlm_dataloader(cfg)
        vlm_train_dataloader = vlm_data_module["train_dataloader"]
        
        return vlm_train_dataloader
