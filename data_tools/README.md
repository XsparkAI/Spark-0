# Spark Data Alignment and HDF5 Conversion

This directory contains the Spark data conversion entry points. Shared readers, retargeting code, camera alignment, and storage writers live under `data_tools/data_alignment/`.

Run the commands below from the repository root; all paths are relative to it. Install the extra dependencies with `bash install.sh --data` (or `pip install -e ".[data]"`). The usual flow is retargeting into `data/retargeted/<source>/` (via `scripts/run_retarget_batch.sh` or the commands below), then `scripts/process_data.sh` into `data/lerobot/<dataset>/`.

## Supported dataset names

The public dataset names accepted by the command line tools are:

- **human_touch** — HumanTouch1/EgoPro, HumanTouch2, and HumanTouch2 real-robot MCAP data. The reader inspects MCAP channel keys and selects the matching internal format automatically.
- **openaoe** — OpenAOE samples containing hands.npz, camera trajectories, and video.
- **egodex** — EgoDex HDF5 episodes with their camera/video data.
- **fourier_actionnet** — Fourier ActionNet hand data and RGB video.
- **agibot_world_beta** — AgiBot World Beta hand/proprioception data.
- **egovla_bench** — EgoVLA Benchmark humanoid-simulation episodes.
- **egoverse** — EgoVerse zarr episodes and Lightwheel capture directories.
- **auto** — Detects the dataset from the source layout. Batch runs every family's episode discovery and fails if the input directory holds more than one family; this walks the tree once per family, so pass an explicit name for large roots.

The old names egopro, humantouch2, humantouch2_real_robot, and agibot_world are not public command-line aliases. HumanTouch format detection is internal and is recorded in the output provenance.

## Batch conversion

The unified batch entry point is:

~~~bash
python data_tools/batch_retarget_camera_aligned.py \
  --input-dir ./data/retarget_src/<source> \
  --output-dir ./data/retargeted/<source> \
  --dataset-name auto
~~~

Use an explicit dataset name when the input directory contains mixed layouts or when you want the run to fail instead of guessing:

~~~bash
python data_tools/batch_retarget_camera_aligned.py \
  --input-dir ./data/retarget_src/HumanTouch1 \
  --output-dir ./data/retargeted/human_touch \
  --dataset-name human_touch \
  --camera-config data_tools/data_alignment/assets/spark_camera_alignment.json \
  --mano-assets ./mano_v1_2
~~~

Useful batch options:

- workers N controls concurrent episode workers.
- frames N converts only the first N source frames for a smoke test.
- fps 30 selects the output frequency. Spark HDF5 output uses an integer frequency.
- limit N converts at most N discovered episodes.
- overwrite regenerates existing outputs.
- retry-failed retries sources recorded as failed in the batch report.
- dry-run discovers inputs and prints the planned work without writing HDF5.
- config scripts/retarget_configs/openaoe.json loads settings from JSON; explicit command-line flags take precedence.
- instruction TEXT supplies one instruction for every selected episode.
- instruction-map ./instructions.json supplies task instructions from a mapping file.
- report-path ./data/retargeted/batch_report.jsonl changes the resumable batch report location.

The batch runner discovers episode-level inputs, writes each episode to a temporary partial HDF5, validates the result, and atomically publishes the final file. Existing valid outputs are skipped unless overwrite is set. Outputs whose `metadata/source` ends in `_retargeted_to_wuji` (the retired Wuji v1 hand, different joint order) fail validation and must be regenerated with `--overwrite`.

For a quick discovery check:

~~~bash
python data_tools/batch_retarget_camera_aligned.py \
  --input-dir ./data/retarget_src/<source> \
  --output-dir ./data/retargeted/dry_run \
  --dataset-name auto \
  --dry-run
~~~

## Single-episode conversion

The single-episode command is useful when debugging a reader or checking one output.

### Camera-aligned datasets

Use this command for HumanTouch, OpenAOE, EgoDex, and EgoVerse:

~~~bash
python -m data_tools.data_alignment.commands.retarget retarget-camera-aligned \
  --input ./data/episode.mcap \
  --dataset-name human_touch \
  --camera-config data_tools/data_alignment/assets/spark_camera_alignment.json \
  --mano-assets ./mano_v1_2 \
  --output ./data/retargeted/episode.hdf5 \
  --fps 30
~~~

For OpenAOE, use the sample directory and provide a `mano_right.npz` converted from the official MANO release (not bundled; see [Required assets](#required-assets)):

~~~bash
python -m data_tools.data_alignment.commands.retarget retarget-camera-aligned \
  --input ./data/openaoe_episode \
  --dataset-name openaoe \
  --camera-config data_tools/data_alignment/assets/spark_camera_alignment.json \
  --mano-model data_tools/data_alignment/assets/mano_right.npz \
  --output ./data/retargeted/openaoe_episode.hdf5 \
  --fps 30
~~~

### Source-robot datasets

Fourier ActionNet and AgiBot World Beta are converted in their source robot coordinate frame and retargeted to Wuji2. EgoVLA Benchmark uses its reviewed direct Inspire12-to-Wuji20 index/sign mapping:

~~~bash
python -m data_tools.data_alignment.commands.retarget retarget-source-robot \
  --input ./data/fourier_episode.hdf5 \
  --dataset-name fourier_actionnet \
  --output ./data/retargeted/fourier_episode.hdf5 \
  --fps 30
~~~

~~~bash
python -m data_tools.data_alignment.commands.retarget retarget-source-robot \
  --input ./data/proprio_stats.h5 \
  --dataset-name agibot_world_beta \
  --output ./data/retargeted/agibot_episode.hdf5 \
  --fps 30
~~~

~~~bash
python -m data_tools.data_alignment.commands.retarget retarget-source-robot \
  --input ./data/egovla_episode.hdf5 \
  --dataset-name egovla_bench \
  --output ./data/retargeted/egovla_episode.hdf5 \
  --fps 30
~~~

The source-robot command uses the dataset-specific hand path. EgoVLA uses the direct mapping table and does not run Inspire/Wuji FK or IK; Fourier and AgiBot use their URDF FK plus Wuji Hand 2 IK. No camera alignment JSON is required.

AgiBot World Beta keeps the `state/end` flange translation as the EE position. EE orientation and the Inspire FK parent right-multiply the built-in mount rotation `INSPIRE_MOUNT_ROTATION` in `data_alignment/datasets/agibot_world/reader.py`; the mount translation is ignored. The calibration is baked into the source as two 3x3 matrices (manual fingertip PnP, 9.53 px RMS left / 26.89 px RMS right), so no asset file or CLI flag is involved. Recalibrating means editing that constant.

`spark_camera_alignment.json` describes the Spark camera mount orientation (`camera.rotation_rpy_deg`) and the fixed EE orientation anchors (`workspace.target_ee_orientation_wxyz`). Camera-aligned conversion places the episode's initial head camera at the world origin with that orientation and keeps that one world transform for the whole trajectory. No translation is applied. Axes follow the mount orientation and all positions are in meters. Existing HDF5 written with a mount position or workspace translation stays self-consistent: its EE poses, extrinsics, and tactile/MANO maps share that shift, so camera-frame training inputs match a fresh conversion. Regenerate with `--overwrite` when world-frame values must match.

| Key | Meaning |
|---|---|
| `camera.rotation_rpy_deg` | Mount orientation in the world frame (roll/pitch/yaw, deg). The mount sits at the origin. |
| `camera.mount_from_optical` | Rotation from the camera optical frame to the mount frame; must be a proper rotation (det = +1). |
| `camera.initial_window_s` | Window (s) after the first frame with both hands valid. Initial camera poses are averaged over it to fix the world transform, and initial palm orientations are averaged over it for the orientation offset. |
| `workspace.target_ee_orientation_wxyz.{left,right}` | Unit quaternions in the world frame. Each hand gets one fixed rotation offset that maps its initial mean palm onto this orientation. |

Every path writes the same 20-d Wuji Hand 2 joint order (`isaac_wujihand2`), so all sources can be mixed in one training run.

## RGB and camera conventions

- Images are RGB throughout the pipeline. Raw source JPEGs and videos are decoded with OpenCV and converted to RGB once, right at that decode.
- Spark HDF5 frames are written with `encode_image_bit` and read with `decode_image_bit` from `spark0.utils.image_codec`; cameras declare `color_order=xpl_rgb1`. Older outputs labelled `spark_native_jpeg` / `rgb` (bare `cv2.imencode` on RGB) still decode correctly and pass batch validation.
- The standard Spark camera output is 640 x 480. Wide source images are center-cropped to 4:3 before resizing.
- EgoVerse follows the standard 4:3-to-640x480 rule.
- AgiBot World Beta keeps its native 640 x 480 head stream.
- EgoVLA Benchmark keeps its native 384 x 384 RGB images; it is not center-cropped or resized.
- Camera intrinsics are scaled together with the image crop and resize. Camera extrinsics are stored as [x, y, z, qw, qx, qy, qz].

## Output layout

Spark HDF5 outputs contain the following main groups:

- state — current EE poses, Wuji hand joints, and optional arm/tactile fields.
- action — future-state actions; the final action_steps frames are truncated.
- validity — validity masks for state and action fields.
- aligned — camera-aligned hand geometry, confidence, and retargeting error.
- vision — fixed-width JPEG datasets and camera calibration.
- metadata — source provenance, conversion settings, and camera alignment details.

The standard hand state is 54 dimensions: left EE pose (7), left Wuji hand (20), right EE pose (7), and right Wuji hand (20).

## Spark HDF5 to LeRobot v3

~~~bash
bash scripts/process_data.sh <dataset> data/retargeted/<source> --mano-assets ./mano_v1_2
# equivalent to:
python data_tools/transform_lerobot_v30_format_tianji_ee.py \
  --input_dir data/retargeted/<source> \
  --repo_id <dataset> \
  --output_root data/lerobot \
  --mano-assets ./mano_v1_2
~~~

`--input_dir` accepts several directories; their episodes are merged into one dataset.

The converter requires `lerobot>=0.4` (LeRobot v3.0); `--help` works without it. It also renders the `tactile.2d` head-camera contact heatmap from the stored MANO mesh and taxel pressure (`data_alignment/tactile/contact_map.py`).

## Repository layout

- `batch_retarget_camera_aligned.py` — unified batch entry point.
- `data_alignment/commands/` — single-episode and batch command-line implementations.
- `data_alignment/datasets/` — adapter contracts, registry, format readers, and dataset-specific code.
- `data_alignment/retargeting/` — Wuji Hand 2 (`wuji2.py`) and MANO models, forward kinematics, and inverse kinematics.
- `data_alignment/storage/` — Spark HDF5 writers (`spark_hdf5.py` camera-aligned, `source_robot_hdf5.py` Fourier / AgiBot, `egovla_hdf5.py`), shared writer code (`writer_common.py`), the HDF5 reader, and the LeRobot v3 converter.
- `data_alignment/tactile/` — tactile collection, taxel geometry, MANO proximity, and the `tactile.2d` contact map.
- `data_alignment/schema.py` — the single in-memory motion/geometry schema plus alignment and retargeting summaries.
- `data_alignment/readers.py` — HumanTouch and OpenAOE parsing, dataset-name detection, and instruction lookup.
- `data_alignment/pipeline.py`, `camera_alignment.py`, `transforms.py`, and `resampling.py` — shared conversion stages.

For HumanTouch, only human_touch should be supplied to the command line; the reader records the detected internal variant in metadata/source_format.

## Required assets

The repository tracks only `data_alignment/assets/spark_camera_alignment.json` (Spark camera mount pose and EE orientation anchors).

MANO is distributed under its own license and must be downloaded from [mano.is.tue.mpg.de](https://mano.is.tue.mpg.de). OpenAOE FK reads `data_alignment/assets/mano_right.npz` (or `MANO_RIGHT.pkl`) converted from that release. Official MANO `MANO_LEFT.pkl` and `MANO_RIGHT.pkl`, and the Wuji, Inspire, and Fourier URDFs, are not bundled. Place them under `data_tools/data_alignment/assets/`; the source-robot readers load those default paths. Pass `--mano-assets` or `--mano-model` where the command accepts them.
