"""Batch CLI configuration for HumanTouch/OpenAOE conversion."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


# Run settings, in precedence order: DEFAULTS < --config JSON < explicit CLI flags.
DEFAULTS: dict[str, object] = {
    "input_dir": None,
    "input_json": None,
    "output_dir": None,
    "camera_config": None,
    "shard": None,
    "workers": 2,
    "threads_per_worker": 1,
    "dataset_name": "auto",
    "frames": None,
    "fps": 30.0,
    "action_steps": 1,
    "max_rms_error_m": 0.03,
    "ik_solver": "analytic",
    "ik_stride": 2,
    "ik_max_iterations": 35,
    "no_trim_invalid": False,
    "allow_missing_jpeg": False,
    "mano_model": None,
    "mano_assets": None,
    "mano_flat_hand_mean": True,
    "mano_left_convention": "mirror_model",
    "instruction": None,
    "instruction_map": None,
    "annotations_root": None,
    "task_name_map": None,
    "limit": None,
    "skip": [],
    "skip_file": None,
    "stop_on_error": False,
    "retry_failed": False,
    "overwrite": False,
    "dry_run": False,
    "report_path": None,
}
CONFIG_COMMENT_KEYS = frozenset({"name", "description", "notes"})
CONFIG_PATH_KEYS = (
    "input_dir",
    "input_json",
    "output_dir",
    "camera_config",
    "mano_model",
    "mano_assets",
    "skip_file",
    "annotations_root",
    "task_name_map",
    "instruction_map",
    "report_path",
)
SHARD_PLACEHOLDER = "{shard}"


def _workspace_root() -> Path:
    """Root that relative config paths are resolved against.

    Defaults to the repository root, so ``data/...`` and ``data_tools/...`` in
    a run config resolve from any cwd. ``SPARK_WORKSPACE_ROOT`` overrides it.
    """
    override = os.environ.get("SPARK_WORKSPACE_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[3]


def _load_config(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Run config must be a JSON object: {path}")
    unknown = set(payload) - set(DEFAULTS) - CONFIG_COMMENT_KEYS
    if unknown:
        raise ValueError(f"Unknown run config keys in {path}: {sorted(unknown)}")
    return {key: value for key, value in payload.items() if key in DEFAULTS}


def _apply_shard(values: dict[str, object], shard: str | None) -> None:
    """Substitute ``{shard}`` in config paths, e.g. ``data/ego_pro/demo/{shard}``."""
    for key in CONFIG_PATH_KEYS:
        value = values.get(key)
        if not isinstance(value, str) or SHARD_PLACEHOLDER not in value:
            continue
        if not shard:
            raise ValueError(f"Run config {key} needs --shard: {value}")
        values[key] = value.replace(SHARD_PLACEHOLDER, shard)


def resolve_settings(args: argparse.Namespace) -> argparse.Namespace:
    """Merge defaults, an optional run config, and explicit CLI flags.

    Relative paths from the config are anchored at the workspace root so a run
    config is independent of the caller's working directory. Relative paths
    passed on the command line keep their usual working-directory meaning.
    """
    values = dict(DEFAULTS)
    if args.config is not None:
        config_values = _load_config(args.config)
        _apply_shard(config_values, args.shard or config_values.get("shard"))
        root = _workspace_root()
        for key in CONFIG_PATH_KEYS:
            value = config_values.get(key)
            if isinstance(value, str) and value.strip():
                config_values[key] = str(root / value)
        values.update(config_values)
    for key, value in vars(args).items():
        if key == "config" or value is None:
            continue
        values[key] = value
    required = ["output_dir"]
    if not values.get("input_json"):
        required.append("input_dir")
    missing = [key for key in required if not values[key]]
    if missing:
        raise ValueError(
            "Missing required settings (pass --config or the matching flags): "
            + ", ".join(sorted(missing))
        )
    return argparse.Namespace(**values)


def build_parser() -> argparse.ArgumentParser:
    """Flags default to ``None`` so unset ones fall back to the run config."""
    parser = argparse.ArgumentParser(
        description="Batch conversion for human_touch, openaoe, egodex, fourier_actionnet, agibot_world_beta, egovla_bench, and egoverse with atomic resumable outputs."
    )
    parser.add_argument(
        "--config",
        help=(
            "JSON run config; keys match these flag names in snake_case. "
            "Relative paths inside it are resolved against the workspace root."
        ),
    )
    parser.add_argument(
        "--shard",
        help="Value substituted for {shard} in run config paths, e.g. X001",
    )
    parser.add_argument("--input-dir")
    parser.add_argument(
        "--input-json",
        help="JSON manifest with a 'files' list of MCAP paths to convert "
        "(alternative to --input-dir)",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--camera-config")
    parser.add_argument(
        "--dataset-name",
        help=(
            "Dataset name human_touch, openaoe, egodex, fourier_actionnet, "
            "agibot_world_beta, egovla_bench, or egoverse (default: auto-detect)"
        ),
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument(
        "--frames",
        type=int,
        help="Convert only the first N source frames (useful for a smoke test)",
    )
    parser.add_argument("--fps", type=float)
    parser.add_argument("--action-steps", type=int)
    parser.add_argument("--max-rms-error-m", type=float)
    parser.add_argument("--ik-solver", choices=("analytic", "finite_difference"))
    parser.add_argument("--ik-stride", type=int)
    parser.add_argument("--ik-max-iterations", type=int)
    parser.add_argument("--threads-per-worker", type=int)
    parser.add_argument(
        "--no-trim-invalid",
        action="store_true",
        default=None,
        help="Keep leading/trailing invalid frames (passed through to the CLI)",
    )
    parser.add_argument(
        "--allow-missing-jpeg",
        action="store_true",
        default=None,
        help=(
            "Pass through the explicit nearest-valid-frame fallback for selected "
            "source messages without JPEG payloads"
        ),
    )
    parser.add_argument(
        "--mano-model",
        help="Path to mano_right.npz for OpenAoE FK only "
        "(default: data_tools/data_alignment/assets/mano_right.npz, not bundled)",
    )
    parser.add_argument(
        "--mano-assets",
        help=(
            "Official mano_v1_2 directory with models/MANO_LEFT.pkl and "
            "MANO_RIGHT.pkl. Required for EgoPro 21-keypoint MANO IK."
        ),
    )
    parser.add_argument(
        "--mano-flat-hand-mean",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "OpenAoE only: hand_pose is absolute, do not add hands_mean "
            "(default); --no-mano-flat-hand-mean adds it"
        ),
    )
    parser.add_argument(
        "--mano-left-convention",
        choices=("mirror_model", "flip_aa", "right_then_mirror"),
        help="OpenAoE only: left-hand MANO parameter convention",
    )
    parser.add_argument("--instruction")
    parser.add_argument(
        "--instruction-map",
        "--instruction-json",
        dest="instruction_map",
        help=(
            "HumanTouch2 JSON mapping task_id/blockIndex to an English task "
            "instruction. HumanTouch task_id_to_instruction JSON files are "
            "accepted. --instruction-json is an alias of --instruction-map. "
            "May be omitted or left blank; instruction is then empty."
        ),
    )
    parser.add_argument(
        "--annotations-root",
        help=(
            "Override sibling <dataset>/annotations. Default is resolved "
            "from <dataset>/mcap/... ."
        ),
    )
    parser.add_argument(
        "--task-name-map",
        help=(
            "JSON map of task_id → task_name. Used as the episode instruction "
            "when the matching semantic_review.json is missing. Also reads "
            "HDF5_TASK_NAME_MAP."
        ),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--skip",
        action="append",
        help="Skip one episode by relative path, filename, or stem (repeatable)",
    )
    parser.add_argument(
        "--skip-file",
        help="Text file of episodes to skip; one relative path, filename, or stem per line",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        default=None,
        help="Stop the whole batch when one episode fails (default: skip and continue)",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        default=None,
        help="Retry episodes previously recorded as failed in batch_report.jsonl",
    )
    parser.add_argument("--overwrite", action="store_true", default=None)
    parser.add_argument("--dry-run", action="store_true", default=None)
    parser.add_argument(
        "--report-path",
        help="Batch JSONL report path. Default: <output-dir>/batch_report.jsonl. "
        "Use a per-shard file so multi-host runs do not share one NFS jsonl.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return resolve_settings(build_parser().parse_args(argv))




# Output checks used by the batch command.

TACTILE_HDF5_KEYS = (
    "tactile/taxel_pressure_3D/left_ee",
    "tactile/taxel_pressure_3D/right_ee",
    "tactile/region_pressure_vector/left_ee",
    "tactile/region_pressure_vector/right_ee",
    "raw/manus/left_ee",
    "raw/manus/right_ee",
)
RAW_60HZ_HDF5_KEYS = (
    "raw/tactile_raw_60hz/left_ee",
    "raw/tactile_raw_60hz/right_ee",
    "raw/tactile_patch_60hz/left_ee",
    "raw/tactile_patch_60hz/right_ee",
)
MANO_HDF5_KEYS = (
    "mano/state/left_ee_poses",
    "mano/state/left_ee_joint_states",
    "mano/state/left_hand_betas",
    "mano/state/right_ee_poses",
    "mano/state/right_ee_joint_states",
    "mano/state/right_hand_betas",
    "mano/action/left_ee_poses",
    "mano/action/left_ee_joint_states",
    "mano/action/left_hand_betas",
    "mano/action/right_ee_poses",
    "mano/action/right_ee_joint_states",
    "mano/action/right_hand_betas",
)


def _validate_tactile_output(file: Any, frames: int) -> None:
    import numpy as np

    missing = [key for key in TACTILE_HDF5_KEYS if key not in file]
    if missing:
        raise ValueError(f"Missing EgoPro tactile HDF5 keys: {missing}")
    for key in TACTILE_HDF5_KEYS:
        if int(file[key].shape[0]) != frames:
            raise ValueError(f"Length mismatch for {key}")
    for key in (
        "tactile/taxel_pressure_3D/left_ee",
        "tactile/taxel_pressure_3D/right_ee",
    ):
        if file[key].shape[1:] != (320, 4):
            raise ValueError(f"{key} must have shape [T,320,4]")
        if file[key].dtype != np.dtype(np.float32):
            raise ValueError(f"{key} must use float32")
        for start in range(0, frames, 1024):
            stop = min(start + 1024, frames)
            if not np.all(np.isfinite(file[key][start:stop])):
                raise ValueError(f"{key} contains non-finite values")
    for key in (
        "tactile/region_pressure_vector/left_ee",
        "tactile/region_pressure_vector/right_ee",
    ):
        if file[key].shape[1:] != (50, 6):
            raise ValueError(f"{key} must have shape [T,50,6]")
        if file[key].dtype != np.dtype(np.float32):
            raise ValueError(f"{key} must use float32")
        if not np.all(np.isfinite(file[key][()])):
            raise ValueError(f"{key} contains non-finite values")
    for key in ("raw/manus/left_ee", "raw/manus/right_ee"):
        if file[key].shape[1:] != (21, 3):
            raise ValueError(f"{key} must have shape [T,21,3]")
        if file[key].dtype != np.dtype(np.float32):
            raise ValueError(f"{key} must use float32")
    missing_raw = [key for key in RAW_60HZ_HDF5_KEYS if key not in file]
    if missing_raw:
        raise ValueError(f"Missing EgoPro raw 60 Hz HDF5 keys: {missing_raw}")
    for key in (
        "raw/tactile_raw_60hz/left_ee",
        "raw/tactile_raw_60hz/right_ee",
    ):
        if file[key].ndim != 2 or file[key].shape[1] != 460:
            raise ValueError(f"{key} must have shape [N,460]")
        if file[key].dtype != np.dtype(np.float32):
            raise ValueError(f"{key} must use float32")
    for key in (
        "raw/tactile_patch_60hz/left_ee",
        "raw/tactile_patch_60hz/right_ee",
    ):
        if file[key].ndim != 2 or file[key].shape[1] != 6:
            raise ValueError(f"{key} must have shape [N,6]")
        if file[key].dtype != np.dtype(np.float32):
            raise ValueError(f"{key} must use float32")
    missing_mano = [key for key in MANO_HDF5_KEYS if key not in file]
    if missing_mano:
        raise ValueError(f"Missing EgoPro MANO HDF5 keys: {missing_mano}")
    for key in MANO_HDF5_KEYS:
        if int(file[key].shape[0]) != frames:
            raise ValueError(f"Length mismatch for {key}")
        if key.endswith("_ee_poses"):
            expected = (7,)
        elif key.endswith("_hand_betas"):
            expected = (10,)
        else:
            expected = (45,)
        if file[key].shape[1:] != expected:
            raise ValueError(f"{key} must have shape [T,{expected[0]}]")


def _validate_output(path: Path) -> dict[str, object]:
    try:
        import h5py
        import numpy as np
    except ImportError as error:
        raise ImportError("Batch validation requires h5py and numpy") from error
    from ..storage.writer_common import IMAGE_BITS_COLOR_ORDER

    required = (
        "instruction",
        "additional_info/frequency",
        "state/left_ee_poses",
        "state/right_ee_poses",
        "state/left_ee_joint_states",
        "state/right_ee_joint_states",
        "action/left_ee_poses",
        "action/right_ee_poses",
        "action/left_ee_joint_states",
        "action/right_ee_joint_states",
        "vision/cam_head/colors",
        "vision/cam_head/shape",
        "vision/cam_head/extrinsics",
        "vision/cam_head/intrinsics",
        "validity/state_54",
        "validity/cam_head_pose",
        "metadata/camera_alignment_json",
        "metadata/episode_truncation",
        "metadata/source",
        "intrinsics",
    )
    with h5py.File(path, "r") as file:
        missing = [key for key in required if key not in file]
        if missing:
            raise ValueError(f"Missing HDF5 keys: {missing}")
        frames = int(file["state/left_ee_poses"].shape[0])
        if frames < 1:
            raise ValueError("Output contains no frames")
        truncation = int(file["metadata/episode_truncation"][()])
        if truncation < 1:
            raise ValueError(
                f"metadata/episode_truncation must be >= 1, got {truncation}"
            )
        for key in (
            "state/right_ee_poses",
            "state/left_ee_joint_states",
            "state/right_ee_joint_states",
            "action/left_ee_poses",
            "action/right_ee_poses",
            "action/left_ee_joint_states",
            "action/right_ee_joint_states",
            "vision/cam_head/colors",
            "vision/cam_head/extrinsics",
            "validity/state_54",
            "validity/cam_head_pose",
        ):
            if int(file[key].shape[0]) != frames:
                raise ValueError(f"Length mismatch for {key}")
        source_raw = file["metadata/source"][()]
        source = (
            source_raw.decode("utf-8")
            if isinstance(source_raw, bytes)
            else str(source_raw)
        )
        if source.endswith("_retargeted_to_wuji"):
            raise ValueError(
                "Output was retargeted to the Wuji v1 hand (different 20-d joint "
                "order); re-run with --overwrite to regenerate it for Wuji Hand 2"
            )
        if source.startswith(("egopro", "humantouch2")) and any(
            key in file for key in TACTILE_HDF5_KEYS
        ):
            _validate_tactile_output(file, frames)
        frequency = int(file["additional_info/frequency"][()])
        colors = file["vision/cam_head/colors"]
        if colors.dtype.kind != "S":
            raise ValueError(
                "vision/cam_head/colors must use Spark fixed-width byte strings, "
                f"got {colors.dtype}"
            )
        image_shape = [int(v) for v in file["vision/cam_head/shape"][()]]
        resize_attr = str(file["vision/cam_head"].attrs.get("resize", ""))
        source_topic = str(file["vision/cam_head"].attrs.get("source_topic", ""))
        # HumanTouch/OpenAOE/EgoDex/Fourier use the standard 640x480 Spark
        # crop.  AgiBot head_color.mp4 is already 640x480 and therefore has
        # no crop attribute.  EgoVLA benchmark images are native 384x384.
        standard_640 = image_shape == [480, 640, 3]
        native_agibot = standard_640 and source_topic == "head_color.mp4"
        egovla_native = image_shape == [384, 384, 3] and resize_attr == "native_384x384"
        if not (standard_640 or egovla_native):
            raise ValueError(
                "vision/cam_head/shape must be [480, 640, 3] or EgoVLA "
                f"[384, 384, 3], got {image_shape}"
            )
        color_order = str(file["vision/cam_head"].attrs.get("color_order", ""))
        # Legacy labels mark frames written with a bare ``cv2.imencode``;
        # ``decode_image_bit`` still reads those correctly, so they stay valid.
        if color_order not in {IMAGE_BITS_COLOR_ORDER, "spark_native_jpeg", "rgb"}:
            raise ValueError(
                f"vision/cam_head must declare color_order={IMAGE_BITS_COLOR_ORDER} "
                f"(or a legacy label), got {color_order!r}"
            )
        intrinsics = np.asarray(file["intrinsics"][()], dtype=np.float64).reshape(4)
        cam_intrinsics = np.asarray(
            file["vision/cam_head/intrinsics"][()], dtype=np.float64
        ).reshape(4)
        if not np.allclose(intrinsics, cam_intrinsics):
            raise ValueError("top-level and cam_head intrinsics disagree")
        if not np.all(np.isfinite(intrinsics)):
            raise ValueError("intrinsics must be finite [fx,fy,cx,cy]")
        _fx, _fy, cx, cy = intrinsics.tolist()
        if not (0.0 <= cx < 640.0 and 0.0 <= cy < 480.0):
            raise ValueError(
                "intrinsics principal point must lie in the 640x480 image, "
                f"got cx={cx}, cy={cy}"
            )
        if standard_640 and not native_agibot and "center_crop" not in resize_attr:
            raise ValueError(
                "vision/cam_head must use center-crop resize unless the source "
                "head stream is already 640x480; "
                f"got attrs.resize={resize_attr!r}, source_topic={source_topic!r}"
            )
        for index in {0, frames - 1}:
            jpeg = bytes(colors[index]).rstrip(b"\0")
            if not (jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9")):
                raise ValueError(f"Invalid JPEG bytes at frame {index}")
        instruction_raw = file["instruction"][()]
        instruction = (
            instruction_raw.decode("utf-8")
            if isinstance(instruction_raw, bytes)
            else str(instruction_raw)
        )
        if not instruction.strip() and not source.startswith("humantouch2"):
            raise ValueError("Output instruction is empty")
        valid_fraction = float(np.mean(file["validity/state_54"][()]))
    return {
        "frames": frames,
        "frequency": frequency,
        "episode_truncation": truncation,
        "cam_head_dtype": str(colors.dtype),
        "instruction": instruction,
        "valid_state_fraction": valid_fraction,
        "size_bytes": path.stat().st_size,
    }
