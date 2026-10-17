"""Command-line entry point for camera-aligned Spark 54-D HDF5 conversion."""

from __future__ import annotations

import argparse

from ..pipeline import command_retarget_camera_aligned, command_retarget_source_robot
from ..retargeting.mano.model import default_mano_asset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert human_touch, openaoe, egodex, fourier_actionnet, agibot_world_beta, egovla_bench, or egoverse samples into "
            "camera-aligned Spark 54-D HDF5. Source data is never modified."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser(
        "retarget-camera-aligned",
        help="Write one episode as camera-aligned Spark 54-D HDF5",
    )
    command.add_argument(
        "--input",
        required=True,
        help="Episode source for human_touch, openaoe, egodex, or egoverse",
    )
    command.add_argument("--camera-config", required=True)
    command.add_argument("--output", required=True)
    command.add_argument(
        "--dataset-name",
        default="auto",
        help="human_touch, openaoe, egodex, or egoverse; source format is auto-detected",
    )
    command.add_argument("--instruction")
    command.add_argument(
        "--instruction-map",
        help=(
            "HumanTouch2 JSON mapping from blockIndex to task description. "
            "Optional; instruction remains empty when omitted."
        ),
    )
    command.add_argument(
        "--annotations-root",
        help=(
            "Override the sibling <dataset>/annotations directory. Default "
            "is resolved from <dataset>/mcap/... . Missing review/task_name "
            "is an error unless --instruction is set."
        ),
    )
    command.add_argument("--frames", type=int)
    command.add_argument("--fps", type=float, default=25.0)
    command.add_argument("--max-gap-s", type=float, default=0.1)
    command.add_argument("--lowpass-alpha", type=float, default=0.35)
    command.add_argument("--initial-window-s", type=float)
    command.add_argument("--action-steps", type=int, default=1)
    command.add_argument(
        "--no-trim-invalid",
        action="store_true",
        help=(
            "Keep leading/trailing invalid frames. Default drops prefix/suffix "
            "outside the first..last frame where both EE and both Wuji blocks "
            "are valid (interior gaps kept as NaN)."
        ),
    )
    command.add_argument(
        "--allow-missing-jpeg",
        action="store_true",
        help=(
            "For EgoPro/HumanTouch2 real-robot streams, duplicate the nearest "
            "valid JPEG when a selected source message has no JPEG payload and "
            "mark that camera frame invalid."
        ),
    )
    command.add_argument("--max-rms-error-m", type=float, default=0.03)
    command.add_argument(
        "--ik-solver",
        choices=("analytic", "finite_difference"),
        default="analytic",
    )
    command.add_argument("--ik-stride", type=int, default=2)
    command.add_argument("--ik-max-iterations", type=int, default=35)
    command.add_argument(
        "--mano-model",
        default=str(default_mano_asset()),
        help="Path to mano_right.npz for OpenAoE FK only (default: data_tools/data_alignment/assets/mano_right.npz, not bundled)",
    )
    command.add_argument(
        "--mano-assets",
        help=(
            "Official mano_v1_2 directory with models/MANO_LEFT.pkl and "
            "MANO_RIGHT.pkl. Required for EgoPro/HumanTouch2 MANO IK and 2D."
        ),
    )
    command.add_argument(
        "--mano-flat-hand-mean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "OpenAoE hand_pose is absolute articulation, so hands_mean is not "
            "added (default). --no-mano-flat-hand-mean adds hands_mean."
        ),
    )
    command.add_argument(
        "--mano-left-convention",
        choices=("mirror_model", "flip_aa", "right_then_mirror"),
        default="mirror_model",
        help="Left-hand MANO parameter convention for OpenAoE",
    )
    command.add_argument(
        "--episode-meta-json",
        help=(
            "Optional JSON with capture_date, mcap_stem, task_id, "
            "tactile_usable, and tactile_qc (fail reasons)."
        ),
    )
    command.set_defaults(func=command_retarget_camera_aligned)

    source = subparsers.add_parser(
        "retarget-source-robot",
        help="Convert Fourier ActionNet, AgiBot World Beta, or EgoVLA Benchmark in source-robot coordinates",
    )
    source.add_argument("--input", required=True)
    source.add_argument("--output", required=True)
    source.add_argument(
        "--dataset-name",
        default="auto",
        help="auto, fourier_actionnet, agibot_world_beta, or egovla_bench",
    )
    source.add_argument("--instruction")
    source.add_argument("--frames", type=int)
    source.add_argument("--fps", type=float, default=30.0)
    source.add_argument("--max-gap-s", type=float, default=0.1)
    source.add_argument("--lowpass-alpha", type=float, default=0.35)
    source.add_argument("--action-steps", type=int, default=1)
    source.add_argument("--max-rms-error-m", type=float, default=0.03)
    source.add_argument(
        "--ik-solver", choices=("analytic", "finite_difference"), default="analytic"
    )
    source.add_argument("--ik-stride", type=int, default=2)
    source.add_argument("--ik-max-iterations", type=int, default=35)
    source.add_argument("--mano-left-convention", default="mirror_model")
    source.set_defaults(func=command_retarget_source_robot)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
