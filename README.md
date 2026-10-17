<h1 align="center">Spark-0: Scaling Spatially Aligned Tactile Pretraining for Robot Dexterous Manipulation</h1>

<p align="center">
<a href="https://arxiv.org/abs/2610.22342">[Paper]</a>
<a href="https://xsparkai.com/Spark-0/">[Project Page]</a>
<a href="https://huggingface.co/TianxingChen/Spark-0">[Checkpoint]</a>
<a href="https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/Spark_0">[Code (XPolicyLab)]</a>
</p>

<p align="center">
Tianxing Chen<sup>*,§</sup>, Wenwei Lin<sup>*</sup>, Chuqiao Lyu<sup>*,§</sup>, Peicheng Xiang<sup>*</sup>, Junyuan Tang<sup>*</sup>, Tian Nian<sup>*</sup>, Zijian Cai, Kehe Ye, Chenze Yu, Wenxuan Zhu, Eric J. Chen, Yan Qin, Yue Chen, Tongyu Wu, Junhao Gong, Kai-Chong Lei, Jiahao Zhang, Kailun Su, Qiwei Liang, Renjing Xu, Ping Luo<sup>†</sup>, Wenbo Ding<sup>†</sup>
</p>

<p align="center">
<sup>*</sup>Equal contribution &nbsp; <sup>§</sup>Project leads &nbsp; <sup>†</sup>Corresponding authors
</p>

<p align="center"><i>Teaching vision where contact happens, and actions how it feels.</i></p>

Spark-0 is a spatially aligned vision-tactile-language-action (VTLA) policy for bimanual dexterous hands. Touch plays two roles in one pretraining run:

- **Where contact occurs.** Image-space contact maps and a frozen Depth-Anything-3 teacher supervise the VLM's visual tokens. This supervision is train-only and removed at inference.
- **How contact evolves.** A tactile encoder conditions the action expert.

![](https://cdn.xsparkai.com/spark-0/assets/figures/teaser-opt.webp)

One checkpoint therefore serves both vision-only and tactile-conditioned deployment. The model is built on a patched copy of [StarVLA](https://github.com/starVLA/starVLA) (Qwen3-VL-4B + flow-matching action expert).

> Spark-0 is also integrated in [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab/tree/main/policy/Spark_0) for benchmarking against other policies. This repository is the standalone release; checkpoints are interchangeable between the two.

## Contents

- [Installation](#installation)
- [Model Zoo](#model-zoo)
- [Quick Start: Serve a Checkpoint](#quick-start-serve-a-checkpoint)
- [Data](#data)
- [Training](#training)
- [Deployment](#deployment)
- [Configuration](#configuration)
- [Repository Structure](#repository-structure)

## Installation

```bash
git clone https://github.com/XsparkAI/Spark-0.git && cd Spark-0
conda create -n spark0 python=3.10 -y && conda activate spark0
bash install.sh          # torch 2.6, flash-attn, vendored StarVLA, spark0
bash install.sh --data   # additionally: data conversion / retargeting tools (lerobot>=0.4, mcap, ...)
```

A robot-side client needs only `pip install -e .` (numpy, msgpack, websockets, opencv); it does not need torch.

## Model Zoo

| Checkpoint | Description | Download |
| --- | --- | --- |
| `pretrain_spark_0` | Visual + tactile pretraining on retargeted human and robot data (Tianji Marvin arms + Wuji Hand 2) | [TianxingChen/Spark-0](https://huggingface.co/TianxingChen/Spark-0) |

```bash
bash scripts/download_pretrained_ckpt.sh   # base VLM + DA3 teacher + Spark-0 checkpoint; HF_ENDPOINT=https://hf-mirror.com for a mirror
```

<details>
<summary>Where each asset lands</summary>

| Asset | Location | Script |
| --- | --- | --- |
| Qwen3-VL-4B-Instruct | `third_party/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct/` | `scripts/download_qwen3_vl_4b_instruct.sh` |
| DA3-LARGE (train-only) | `third_party/starVLA/playground/Pretrained_models/DA3-LARGE/` | `scripts/download_da3_large.sh` |
| Spark-0 pretrained checkpoint | `checkpoints/pretrain_spark_0/` | `scripts/download_spark0_pretrained_ckpt.sh` |

</details>

## Quick Start: Serve a Checkpoint

Start the policy server on a GPU machine:

```bash
bash scripts/serve.sh pretrain_spark_0 --gpu 0 --port 8000
```

Then drive it from any machine. `examples/client_example.py` runs a synthetic environment end to end:

```bash
python examples/client_example.py --url ws://<gpu-host>:8000 --steps 100
```

In your own control loop:

```python
from spark0.serving.client import SparkClient

with SparkClient("ws://gpu-host:8000") as client:
    client.reset()
    while not done:
        client.update_obs(obs)              # format: see Deployment
        for action in client.get_action():  # one action row per call
            robot.apply(action)
```

`spark0.serving.client.run_episode(env, client)` implements this loop for any object with `is_episode_end()`, `get_obs()`, and `take_action(action)`; `run_episode_batch` does the same for vectorized environments.

## Data

Training reads **LeRobot v3** datasets in a Spark-specific layout (EE poses + Wuji hand joints as state/action, cameras declared in `meta/modality.json`, tactile streams as extra columns). Datasets from the official LeRobot converters cannot be fed to the trainer directly. Everything lives under `data/` (or `$SPARK_0_DATA_ROOT`):

```text
data/
├── raw/<task>/**/*.hdf5          # Spark HDF5 episodes recorded on the robot
├── retarget_src/<source>/        # human / other-robot datasets to retarget
├── retargeted/<source>/**/*.hdf5 # retargeting output (Spark HDF5)
└── lerobot/<dataset>/            # LeRobot v3 training datasets (what train.sh reads)
```

### Spark HDF5 to LeRobot v3

```bash
# One or more input directories (searched recursively) -> data/lerobot/<dataset>/
bash scripts/process_data.sh pour_water data/raw/pour_water
bash scripts/process_data.sh kitchen_mix pour_water stack_bowls --max_episode 50   # bare names resolve under data/raw/
```

The converter requires `vision/cam_head/extrinsics` in every episode, exports tactile columns when present, and renders the `tactile.2d` contact heatmap from the stored MANO mesh (pass `--mano-assets <mano_v1_2>`).

### Cross-embodiment retargeting

Pretraining data is not recorded on Tianji + Wuji. Every source is first retargeted into the same 54-d layout (Tianji EE poses + Wuji Hand 2 joints), then exported to LeRobot v3:

```bash
bash scripts/run_retarget_batch.sh openaoe                                  # data/retarget_src/OpenAoE-2000h -> data/retargeted/openaoe
bash scripts/run_retarget_batch.sh human_touch --shard X001 --mano-assets /path/to/mano_v1_2
bash scripts/process_data.sh openaoe data/retargeted/openaoe
```

Run configs live in `scripts/retarget_configs/`; any extra flag overrides the config (e.g. `--limit 2 --dry-run`). MANO and the Wuji / Inspire / Fourier URDFs are **not bundled**: obtain MANO from [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de) and place the files under `data_tools/data_alignment/assets/` or pass `--mano-assets` / `--mano-model`. See [data_tools/README.md](data_tools/README.md) for every source and the HDF5 layout.

<details>
<summary>How retargeting works</summary>

1. **Arm / EE.** Human wrists are re-expressed into the Spark head-camera mount frame (`camera_alignment.py`): the initial camera window is placed at the world origin with the mount orientation from `spark_camera_alignment.json` (RPY 30/0/0), and a fixed per-hand orientation offset matches the initial palm to `target_ee_orientation_wxyz`. Training reads EE poses in the head-camera frame, so this placement does not change model inputs. Robot datasets keep their own base frame.
2. **Hand.** 5×5 finger keypoints are morphology-normalized and solved to Wuji Hand 2 joints by IK (`retargeting/wuji2.py`). Every source writes the same 20-d Wuji Hand 2 order (Isaac order, EE = `Link7`).

| Source | Hand input | Hand path |
| --- | --- | --- |
| `human_touch`, `openaoe`, `egodex`, `egoverse` | Human hand keypoints / MANO | Camera alignment + Wuji IK |
| `fourier_actionnet`, `agibot_world_beta` | Fourier / Inspire hand joints | Robot-hand FK → keypoints → Wuji IK |
| `egovla_bench` | Inspire 12-d joints | Direct joint mapping (`datasets/egovla_bench/hand_mapping.json`), no IK |

</details>

<details>
<summary>Packed streams (what the model sees)</summary>

| Stream | Packed shape | Source column |
| --- | --- | --- |
| State / action | `(58,)` after pose conversion; 54-d on disk | `[L_ee_pose(7), L_hand(20), R_ee_pose(7), R_hand(20)]`, poses `[x, y, z, qw, qx, qy, qz]` |
| Cameras | RGB, resized to `obs_image_size` | `video.cam_high`, `video.cam_left_wrist`, `video.cam_right_wrist` |
| Region force | `[hands, regions, history, 1]` | `tactile.region_pressure_vector.{left,right}_ee` |
| 3D taxel cloud | `[hands, taxels, T, 4]` (xyz + pressure) | `taxel_pressure_3D.{left,right}_ee` or converter `3d_left` / `3d_right` |
| Contact heatmap | `[H, W]` | `tactile.2d` (train-only, for `tactile_align`) |

Tactile columns are packed only when a tactile stem is on. Images are RGB end to end: read and write stored frames with `decode_image_bit` / `encode_image_bit` from `spark0.utils.image_codec` and never add a `BGR2RGB` swap after decoding. `python data_tools/repack_lerobot_row_groups.py <dataset_dir>` repacks exports written with one parquet row group per file, for fast random frame reads.

</details>

## Training

```bash
bash scripts/train.sh <recipe> [--data NAME] [--run-name NAME] [--gpus IDS] [--seed N] [overrides...]
```

Normalization statistics must be precomputed first (CPU only, once per change to the data or a stats-relevant knob):

```bash
bash scripts/precompute_stats.sh sft_spark_0 --data pour_water
bash scripts/train.sh sft_spark_0 --data pour_water --gpus 0,1,2,3 \
    trainer.pretrained_checkpoint=$PWD/checkpoints/pretrain_spark_0/<path/to/steps_N_pytorch_model.pt>
```

The run is written to `checkpoints/<run-name>-seed<seed>/` (`<run-name>` defaults to the recipe stem). Training runs with its working directory inside `third_party/starVLA/`, so pass absolute paths in overrides. `PER_DEVICE_BS=4` overrides the per-GPU batch size; any `key=value` argument is an OmegaConf override.

| Recipe | Purpose | Schedule |
| --- | --- | --- |
| `pretrain_spark_0` | Visual + tactile pretraining, `sample_without_replacement: true`, DA3 teacher at 336 px | 500k steps, 20k warmup, save every 50k |
| `sft_spark_0` | Finetune from the pretrain checkpoint, `use_pretrained_stats: true`, `rtc: default`, DA3 teacher at 392 px | 80k steps, 8k warmup, save every 10k |

Both recipes enable DA3 alignment, flexible views, action queries, `tactile_fusion: default`, and `tactile_align: default`. To co-train on several datasets, set the recipe's `data_mixture` (`- {dataset: <name under data/lerobot>, weight: <prob>}`) and drop `--data`.

<details>
<summary>Training notes: statistics, resume, multi-node, DeepSpeed, monitoring</summary>

- **Statistics.** Training requires precomputed stats by default (`SPARK_0_REQUIRE_PRECOMPUTED_STATS=1`). Every rank reads the content-addressed caches `<dataset>/meta/stats_xpolicy*__<hash>.json`; a missing cache fails the launch with its path. Changing the data, pose formats, `state_history_frames`, `action_horizon`, `exclude_episodes`, or the embodiment produces a new hash.
- **DA3 teacher.** The launch checks that the teacher weights exist (`scripts/check_da3_teacher.sh`); `SKIP_DA3_CHECK=1` skips the check. The teacher is never loaded at inference.
- **Warm start vs. resume.** `trainer.pretrained_checkpoint` loads weights with a fresh step counter; `trainer.resume_from` (or `SPARK_0_RESUME_FROM`) also continues the step counter. Optimizer moments are not saved.
- **Multi-node.** Run the same command on every node with a unique `SPARK_0_MACHINE_RANK` and the same `SPARK_0_NUM_MACHINES`, `SPARK_0_MAIN_PROCESS_IP`, `SPARK_0_MAIN_PROCESS_PORT`. Set `SPARK_0_GPUS_PER_NODE` so a GPU-count mismatch fails at launch instead of hanging.
- **DeepSpeed.** `SPARK_0_ACCELERATE_CONFIG` takes an accelerate YAML (`starVLA/config/deepseeds/zero2.yaml` or `accelerate_zero3.yaml`, not the raw `zero3.yaml`).
- **NaN fuse.** Training aborts every rank on a non-finite loss or gradient before the optimizer step; reload the last saved checkpoint.
- **Monitoring.** `eval_train/fixed_batch_mse` is the fit on a fixed training batch, not a held-out metric. `tactile_gate` / `tactile_patch_gate` are on/off switches and `tactile_extra_temb_rel_norm` the force-path amplitude. Whether touch helps needs a paired evaluation of one checkpoint with and without tactile (`--set tactile=false`).

</details>

## Deployment

```bash
bash scripts/serve.sh <ckpt> [--gpu ID] [--port 8000] [--config configs/serve.yaml] \
    [--model-config <recipe>] [--execute-horizon N] [--set KEY=VALUE ...] \
    [--backend-override KEY=VALUE ...] [--dry-run]
```

`<ckpt>` is a weight file, a run directory (newest weights are used), or a run name under `checkpoints/`. The server is one process on one GPU: it loads the checkpoint in-process, wraps it in `spark0.policy.SparkPolicy` (observation preprocessing, pose conversion, tactile packing, RTC, action decoding), and serves it over websocket. It replays the checkpoint's saved training config — embodiment, pose formats, normalization, cameras, state history, tactile layout — so keep `auto` in `configs/serve.yaml` unless the difference is the experiment. `--dry-run` prints the resolved plan without loading the model.

### Observation format

```python
obs = {
    "env_idx": 0,                                   # optional; required for batched / multi-env use
    "instruction": "pour water into the cup",
    "vision": {
        "cam_head":       {"color": rgb_uint8_hwc,  # or encode_image_bit(rgb) JPEG bytes (~20x smaller)
                           "extrinsics": [x, y, z, qw, qx, qy, qz]},  # world_from_camera; or pin ee_camera_pose
        "cam_left_wrist":  {"color": ...},          # optional if the checkpoint trained with flexible_views
        "cam_right_wrist": {"color": ...},
    },
    "state": {
        "left_ee_pose":  [x, y, z, qw, qx, qy, qz], "left_ee_joint_state":  wuji_20,
        "right_ee_pose": [x, y, z, qw, qx, qy, qz], "right_ee_joint_state": wuji_20,
    },
    "tactile": {                                    # optional; omitted streams use the learned no-tactile token
        "region_pressure":   ...,                   # [hands, regions, history], raw sensor units, oldest -> current
        "taxel_pressure_3d": ...,                   # [hands, taxels, T, 4]: xyz (m, training frame) + pressure
    },
}
```

An action has the same keys as `state`: `left_ee_pose`, `left_ee_joint_state`, `right_ee_pose`, `right_ee_joint_state`.

<details>
<summary>Wire protocol and serving contract</summary>

- **On connect**, the server sends a metadata dict: `robot`, `action_type`, `arm_dim`, `ee_dim`, `video_keys`, `image_sizes`, `action_chunk_size`, `state_history_frames`, `rtc_enabled`, `obs_period`, `tactile`, `checkpoint`.
- **Each request** is `{"method": ..., "kwargs": {...}}` (msgpack-numpy) and gets `{"ok": true, "result": ...}` or `{"ok": false, "error": ..., "traceback": ...}`.

| Method | kwargs | Result |
| --- | --- | --- |
| `reset` | none | Clears history, cached chunks, and RTC queues for every env |
| `update_obs` / `update_obs_batch` | `obs` / `obs_list` | none |
| `get_action` / `get_action_batch` | none / `env_idx_list` | `[action]` / `[[action], ...]` |
| `ping` | none | `"pong"` |

- **One row per call.** `get_action` returns one action row per call. The policy infers once per chunk and serves the other rows from a cache, which keeps the state-history frames consecutive as in training.
- **Fast observations.** On steps between inferences the observation is not read. `run_episode(..., fast_obs=True)` (the default) uploads a full observation every `obs_period` steps and keeps the state-history ring buffer on the client; its actions are identical to uploading every step. `obs_period` is `execute_horizon` on the sync path, or `action_chunk_size - fire threshold` under RTC (16 with `rtc: default`).
- **Single tenant.** Per-episode buffers are keyed by `env_idx` only. Run one server per client, or give each client a disjoint `env_idx` range.

</details>

## Configuration

A run is one recipe YAML plus shared stems. `scripts/train.sh` merges these and freezes them with the checkpoint.

```text
configs/<recipe>.yaml
├─ embodiment: tianji_marvin_wuji  -> configs/embodiments/     (robot keys, dims, camera catalog)
├─ execute_horizon: null | N       deploy-time
├─ rtc: null | <stem>              -> configs/rtc/             deploy-time
├─ tactile_fusion: null | <stem>   -> configs/tactile_fusion/  (encoder + action-expert fusion)
└─ tactile_align: null | <stem>    -> configs/tactile_align/   (train-only contact alignment)
configs/serve.yaml                 policy-server defaults (all `auto` = replay the checkpoint)
```

**Deploy-time knobs** (`execute_horizon`, `rtc`) are read from the **live** recipe, not the checkpoint, so they can change without retraining. The server finds it via `--model-config`, then `configs/<ckpt name>.yaml`, then the recipe stem recorded in the checkpoint. `execute_horizon: null` re-plans once per 50-row chunk; `rtc: default` prefetches the next chunk asynchronously, re-planning every 16 steps (`action_horizon` 50 minus `fire_at_remaining` 34), and overrides `execute_horizon`.

<details>
<summary>Model switches</summary>

| Switch | Key | Spark-0 recipes |
| --- | --- | --- |
| Tactile encoder + fusion | `tactile_fusion` | `default` (320-taxel full-hand glove); `fingertip_only` for 80-taxel fingertip exports; `null` = off, identical to a vision-only model |
| Contact-heatmap alignment | `tactile_align` | `default` (head camera); `null` = off |
| DA3 geometry distillation | `framework.spatial_alignment.enable` | `true`, `align_layer_index: 25` (must be odd) |
| VLM action queries | `framework.action_query` | 64 queries, `kv_mode: query_plus_visual` |
| Shortcut mask | `framework.action_model.shortcut_mask_prob` | `0.1`: hides the VLM from the action expert on some tactile rows so touch is not ignored |
| Flexible camera sets | `datasets.vla_data.flexible_views` | `true`: head-only, one-wrist, and two-wrist datasets co-train |
| State history | `datasets.vla_data.state_history_frames` | `3` |
| Pose formats | `datasets.vla_data.{state,action}_pose_format` | `cam_abs_xyz_rot6d` / `delta_xyz_rot6d` |
| Pose normalization | `datasets.vla_data.pose_normalization` | `hold_centered_perrow` |

For a vision-only run, set both tactile stems to `null` and `shortcut_mask_prob: 0`. In `data_mixture`, weights are sampling probabilities; to leave a dataset out, delete its entry (`weight: 0` raises an error). Do not set the stems' `enable` flags or `run_id` in a recipe.

</details>

<details>
<summary>configs/serve.yaml keys (override with --set KEY=VALUE)</summary>

| Key | Default | Purpose |
| --- | --- | --- |
| `robot`, `action_type` | `tianji_marvin_wuji`, `ee` | Arm / hand dims and action space |
| `unnorm_key`, `image_size`, `include_state`, `state_history_frames`, `ee_in_camera_frame`, `max_infer_batch` | `auto` | Replayed from the checkpoint |
| `flexible_views` / `required_views` | `auto` | Missing wrists are fine if training used `flexible_views` |
| `ee_camera_pose`, `ee_camera_extrinsics_semantics`, `ee_camera_axes` | `auto` | Observation camera convention |
| `tactile` | `auto` | `false` disables the glove to A/B the same checkpoint |
| `tactile_fast` / `tactile_slow` | `auto` | `false` drops the region-force / 3D-taxel stream |

</details>

<details>
<summary>Environment variables</summary>

| Variable | Notes |
| --- | --- |
| `SPARK_0_DATA_ROOT` / `SPARK_0_CKPT_ROOT` | Data (`data/`) and checkpoint (`checkpoints/`) roots |
| `SPARK_0_RESUME_FROM` | Override `trainer.resume_from` |
| `SPARK_0_ACCELERATE_CONFIG` | Accelerate YAML for DeepSpeed ZeRO-2 / ZeRO-3 |
| `SPARK_0_NUM_MACHINES`, `SPARK_0_MACHINE_RANK`, `SPARK_0_MAIN_PROCESS_IP`, `SPARK_0_MAIN_PROCESS_PORT`, `SPARK_0_GPUS_PER_NODE` | Multi-node launch |
| `NCCL_TIMEOUT` | Collective watchdog in seconds (default 600) |
| `SPARK_0_REQUIRE_PRECOMPUTED_STATS` | `1` (default) requires stats caches; `0` computes them on rank 0 at launch |
| `SPARK_0_STATS_MODE` | `auto` / `memory` / `streaming` (streaming keeps RAM near 1 GB) |
| `SPARK_0_TACTILE_STATS_READ_WORKERS` | Read threads for tactile statistics (default 32) |
| `SPARK_0_WANDB_ENTITY` / `SPARK_0_WANDB_PROJECT` / `WANDB_MODE` | Weights & Biases (offline by default) |
| `SPARK_0_BASE_VLM` | Override `framework.qwenvl.base_vlm` |
| `SKIP_DA3_CHECK` | `1` skips the DA3 teacher check |
| `SPARK_WORKSPACE_ROOT` | Root for relative paths in retarget run configs (default: repository root) |

Some internal identifiers keep their XPolicyLab-era names so that published checkpoints and stats caches keep resolving: the `SPARK_0_XPOLICY_*` registry variables, the `xpolicylab_runtime` data mix, and the `stats_xpolicy*` cache files.

</details>

## Repository Structure

| Path | Purpose |
| --- | --- |
| `spark0/policy.py` | `SparkPolicy`: observation → model input → decoded action rows, RTC, tactile packing |
| `spark0/serving/` | Policy server, client SDK (`client.py`), launch resolution from the checkpoint (`launch.py`) |
| `spark0/action_codec.py`, `ee_delta_action.py`, `relative_pose.py` | Env-space ↔ model-space action decoding and pose math |
| `spark0/runtime_config.py` | Resolves `auto` knobs from the checkpoint config so training and serving match |
| `spark0/utils/` | RGB image codec and robot-state packing |
| `configs/` | Recipes, `serve.yaml`, and the `embodiments/`, `rtc/`, `tactile_fusion/`, `tactile_align/` stems |
| `scripts/` | `train.sh`, `precompute_stats.sh`, `process_data.sh`, `serve.sh`, retargeting, downloads |
| `data_tools/` | Dataset readers, camera alignment, Wuji / MANO retargeting, Spark HDF5 and LeRobot v3 writers |
| `examples/` | Client example with a synthetic environment |
| `third_party/starVLA/` | Vendored, patched StarVLA (model, training loop, inference wrapper, Spark-0 data registry) |

<details>
<summary>Patches to the vendored StarVLA</summary>

`third_party/starVLA/` tracks [starVLA/starVLA](https://github.com/starVLA/starVLA) at commit `c00b8e1` (2026-07-20). Local patches add the tactile encoder and fusion, contact alignment, DA3 alignment, action queries, flexible views, RTC, precomputed / streaming statistics and mixture sampling fixes, ZeRO-3 safety, and the NaN fuse. Unused upstream frameworks, examples, and deployment servers were removed; diff against that commit when re-syncing.

</details>

## Acknowledgements

Built on [StarVLA](https://github.com/starVLA/starVLA), [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL), [Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3), [LeRobot](https://github.com/huggingface/lerobot), and [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab), whose image codec and state packing are adapted in `spark0/utils/` (Apache-2.0).

## Citation

```bibtex
@misc{chen2026spark0scalingspatiallyaligned,
      title={Spark-0: Scaling Spatially Aligned Tactile Pretraining for Robot Dexterous Manipulation}, 
      author={Tianxing Chen and Wenwei Lin and Chuqiao Lyu and Peicheng Xiang and Junyuan Tang and Tian Nian and Zijian Cai and Kehe Ye and Chenze Yu and Wenxuan Zhu and Eric J. Chen and Yan Qin and Yue Chen and Tongyu Wu and Junhao Gong and Kai-Chong Lei and Jiahao Zhang and Kailun Su and Qiwei Liang and Renjing Xu and Ping Luo and Wenbo Ding},
      year={2026},
      eprint={2610.22342},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2610.22342}, 
}
```

## License

Released under the [MIT License](LICENSE). `third_party/starVLA/` keeps its own MIT license. Code adapted from XPolicyLab is Apache-2.0, as noted in each file header. MANO is not included and is governed by its own license.

Contact: Tianxing Chen (chentianxing2002@gmail.com).
