"""Unified camera-aligned HumanTouch/OpenAOE conversion pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .camera_alignment import (
    CameraAlignmentResult,
    SparkCameraConfig,
    SparkWorkspaceConfig,
    align_reference_to_spark_camera,
    slice_camera_alignment,
)
from .datasets.registry import detect_dataset, get_adapter
from .schema import (
    ReferenceFrameGeometry,
    UnifiedRobotMotion,
    bimanual_frame_valid,
    fill_hand_joint_nan_adjacent,
    leading_trailing_true_span,
    slice_robot_motion,
    pack_spark_state,
)
from .storage.spark_hdf5 import write_camera_aligned_spark_hdf5
from .camera_alignment import resample_reference_geometry
from .resampling import resample_human_motion
from .schema import summarize_camera_alignment, summarize_robot_motion
from .tactile.collector import EgoProTactileCollector
from .retargeting.wuji2 import retarget_camera_aligned_motion


def _trim_to_valid_bimanual_segment(
    robot: UnifiedRobotMotion,
    alignment: CameraAlignmentResult,
) -> tuple[UnifiedRobotMotion, CameraAlignmentResult, dict[str, int | str]]:
    """Drop leading/trailing frames outside the first..last bimanual-valid span."""
    source_frames = len(robot.timestamps)
    frame_valid = bimanual_frame_valid(robot.state_valid)
    start, end = leading_trailing_true_span(frame_valid)
    trimmed_robot = slice_robot_motion(robot, start, end)
    trimmed_alignment = slice_camera_alignment(alignment, start, end)
    kept_valid = frame_valid[start:end]
    info: dict[str, int | str] = {
        "criterion": "bimanual_ee_and_wuji_valid",
        "mode": "leading_trailing",
        "source_frames": source_frames,
        "kept_start": start,
        "kept_end": end,
        "kept_frames": end - start,
        "dropped_leading": start,
        "dropped_trailing": source_frames - end,
        "interior_invalid_frames": int(np.count_nonzero(~kept_valid)),
    }
    trimmed_robot.metadata["validity_trim"] = dict(info)
    trimmed_alignment.motion.metadata["validity_trim"] = dict(info)
    return trimmed_robot, trimmed_alignment, info


def _slice_reference(
    geometry: ReferenceFrameGeometry,
    frames: int | None,
) -> ReferenceFrameGeometry:
    if frames is None:
        return geometry
    index = slice(0, max(0, frames))
    return ReferenceFrameGeometry(
        timestamps=geometry.timestamps[index],
        wrist_poses_ref=geometry.wrist_poses_ref[index],
        wrist_valid=geometry.wrist_valid[index],
        joints_ref=geometry.joints_ref[index],
        joint_confidence=geometry.joint_confidence[index],
        joints_local=geometry.joints_local[index],
        camera_from_world=geometry.camera_from_world[index],
        camera_valid=geometry.camera_valid[index],
        intrinsics=geometry.intrinsics,
        source=geometry.source,
        instruction=geometry.instruction,
        metadata=dict(geometry.metadata),
    )


def _read_reference(
    args: argparse.Namespace,
    dataset_name: str,
) -> tuple[ReferenceFrameGeometry, EgoProTactileCollector | None]:
    adapter = get_adapter(dataset_name)
    return adapter.read_reference(
        Path(args.input),
        instruction_map=args.instruction_map,
        annotations_root=args.annotations_root,
        resolve_instruction=args.instruction is None,
        mano_model=args.mano_model,
        flat_hand_mean=bool(args.mano_flat_hand_mean),
        left_convention=args.mano_left_convention,
    )


def command_retarget_camera_aligned(args: argparse.Namespace) -> None:
    """Convert one supported episode into camera-aligned Spark 54-D HDF5."""
    output = Path(args.output)
    if output.suffix.lower() not in {".hdf5", ".h5"}:
        raise ValueError("--output must end in .hdf5 or .h5")
    dataset_name = detect_dataset(Path(args.input), args.dataset_name)
    if dataset_name == "human_touch" and not args.mano_assets:
        raise ValueError(
            "EgoPro/HumanTouch2 conversion requires --mano-assets pointing at official "
            "mano_v1_2 (models/MANO_LEFT.pkl and MANO_RIGHT.pkl)"
        )

    from .retargeting.mano.ik import limit_cpu_threads

    limit_cpu_threads()
    stage_s: dict[str, float] = {}
    payload = json.loads(Path(args.camera_config).read_text(encoding="utf-8"))
    camera = SparkCameraConfig.from_mapping(payload)
    workspace = SparkWorkspaceConfig.from_mapping(payload)
    if args.initial_window_s is not None:
        camera = SparkCameraConfig(
            rotation_rpy_deg=camera.rotation_rpy_deg,
            initial_window_s=args.initial_window_s,
            mount_from_optical=camera.mount_from_optical,
        )

    started = time.perf_counter()
    geometry, tactile_collector = _read_reference(args, dataset_name)
    geometry = _slice_reference(geometry, args.frames)
    stage_s["read_reference"] = time.perf_counter() - started

    started = time.perf_counter()
    geometry = resample_reference_geometry(
        geometry,
        fps=args.fps,
        max_gap_s=args.max_gap_s,
        lowpass_alpha=args.lowpass_alpha,
    )
    stage_s["resample"] = time.perf_counter() - started

    started = time.perf_counter()
    alignment = align_reference_to_spark_camera(
        geometry,
        camera,
        workspace,
    )
    stage_s["camera_alignment"] = time.perf_counter() - started

    started = time.perf_counter()
    robot = retarget_camera_aligned_motion(
        alignment.motion,
        tolerance_m=args.max_rms_error_m,
        solver=args.ik_solver,
        stride=args.ik_stride,
        max_iterations=args.ik_max_iterations,
    )
    stage_s["ik"] = time.perf_counter() - started
    robot.metadata["performance"] = {"stage_s": stage_s}
    if args.instruction is not None:
        robot.instruction = args.instruction

    validity_trim: dict[str, int | str] | None = None
    if not args.no_trim_invalid:
        started = time.perf_counter()
        robot, alignment, validity_trim = _trim_to_valid_bimanual_segment(
            robot, alignment
        )
        stage_s["validity_trim"] = time.perf_counter() - started
        robot.metadata["performance"] = {"stage_s": stage_s}

    started = time.perf_counter()
    robot = fill_hand_joint_nan_adjacent(robot)
    stage_s["hand_joint_nan_fill"] = time.perf_counter() - started
    robot.metadata["performance"] = {"stage_s": stage_s}
    hand_joint_nan_fill = robot.metadata.get("hand_joint_nan_fill")

    report = summarize_robot_motion(robot)
    report["camera_alignment"] = summarize_camera_alignment(alignment)
    if validity_trim is not None:
        report["validity_trim"] = validity_trim
    if hand_joint_nan_fill is not None:
        report["hand_joint_nan_fill"] = hand_joint_nan_fill
    output.parent.mkdir(parents=True, exist_ok=True)
    episode_meta = get_adapter(dataset_name).episode_metadata(
        Path(args.input), geometry
    )
    if episode_meta and args.instruction is not None:
        episode_meta["instruction"] = str(args.instruction)
        episode_meta["instruction_source"] = "cli"
    elif episode_meta and geometry.instruction:
        episode_meta.setdefault("instruction_source", "reader")
    if getattr(args, "episode_meta_json", None):
        supplied_meta = json.loads(
            Path(args.episode_meta_json).read_text(encoding="utf-8")
        )
        if not isinstance(supplied_meta, dict):
            raise ValueError("--episode-meta-json must contain a JSON object")
        episode_meta = {**(episode_meta or {}), **supplied_meta}
        robot.metadata["episode_meta"] = episode_meta
    started = time.perf_counter()
    write_camera_aligned_spark_hdf5(
        output,
        robot,
        alignment,
        args.input,
        fps=args.fps,
        action_steps=args.action_steps,
        tactile_context=(
            tactile_collector.context if tactile_collector is not None else None
        ),
        mano_assets=args.mano_assets,
        episode_meta=episode_meta,
        allow_missing_jpeg=bool(args.allow_missing_jpeg),
    )
    stage_s["hdf5_total"] = time.perf_counter() - started
    print(
        json.dumps(
            {
                "output": str(output),
                "output_format": "spark_hdf5_v1",
                "output_action_source": f"retargeted_future_state:{args.action_steps}",
                "episode_truncation": int(args.action_steps),
                "instruction": robot.instruction,
                "source": geometry.source,
                "validity_trim": validity_trim,
                "hand_joint_nan_fill": hand_joint_nan_fill,
                "validation": report,
                "performance": robot.metadata["performance"],
                "source_readonly": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )




def command_retarget_source_robot(args: argparse.Namespace) -> None:
    """Convert a Fourier ActionNet or AgiBotWorld episode in its robot frame."""
    output = Path(args.output)
    if output.suffix.lower() not in {".hdf5", ".h5"}:
        raise ValueError("--output must end in .hdf5 or .h5")
    dataset_name = detect_dataset(
        Path(args.input), str(getattr(args, "dataset_name", "auto") or "auto")
    )
    if dataset_name not in {"fourier_actionnet", "agibot_world_beta", "egovla_bench"}:
        raise ValueError(
            "retarget-source-robot supports Fourier ActionNet, AgiBot World, and EgoVLA Benchmark; "
            "use retarget-camera-aligned for EgoDex, HumanTouch, or OpenAOE"
        )
    from .retargeting.mano.ik import limit_cpu_threads
    from .retargeting.wuji2 import retarget_source_robot_motion
    from .storage.source_robot_hdf5 import (
        write_agibot_spark_hdf5,
        write_fourier_spark_hdf5,
    )
    from .storage.egovla_hdf5 import write_egovla_spark_hdf5
    limit_cpu_threads()
    adapter = get_adapter(dataset_name)
    reference_kwargs: dict[str, object] = {}
    if dataset_name == "egovla_bench":
        reference_kwargs["need_finger_chains"] = False
    geometry, _ = adapter.read_reference(Path(args.input), **reference_kwargs)
    motion = geometry.metadata.get("source_motion")
    if motion is None:
        raise ValueError(
            f"{dataset_name} reader did not return source-robot motion metadata"
        )
    if args.frames is not None:
        frames = max(0, int(args.frames))
        source_meta = dict(motion.metadata)
        for key in ("joints_world", "head_poses", "head_valid", "qpos", "left_arm_q", "right_arm_q"):
            if source_meta.get(key) is not None:
                source_meta[key] = source_meta[key][:frames]
        source_meta["video_frame_count"] = None
        motion = type(motion)(
            timestamps=motion.timestamps[:frames],
            wrist_poses=motion.wrist_poses[:frames],
            wrist_valid=motion.wrist_valid[:frames],
            joints_local=motion.joints_local[:frames],
            joint_confidence=motion.joint_confidence[:frames],
            source=motion.source,
            instruction=motion.instruction,
            metadata=source_meta,
        )
    video_frames = motion.metadata.get("video_frame_count")
    target_timestamps = None
    if video_frames is not None and int(video_frames) > 1:
        t_start = float(motion.timestamps[0])
        t_end = min(
            float(motion.timestamps[-1]),
            t_start + (int(video_frames) - 1) / float(args.fps),
        )
        count = int(np.floor((t_end - t_start) * args.fps + 1e-9)) + 1
        target_timestamps = t_start + np.arange(count, dtype=np.float64) / float(args.fps)
    motion = resample_human_motion(
        motion,
        fps=args.fps,
        max_gap_s=float(getattr(args, "max_gap_s", 0.1)),
        lowpass_alpha=float(getattr(args, "lowpass_alpha", 0.35)),
        target_timestamps=target_timestamps,
    )
    if args.instruction is not None:
        motion.instruction = args.instruction
    if dataset_name == "egovla_bench":
        robot = _retarget_egovla_table_motion(motion)
    else:
        robot = retarget_source_robot_motion(
            motion,
            tolerance_m=args.max_rms_error_m,
            solver=args.ik_solver,
            stride=args.ik_stride,
            max_iterations=args.ik_max_iterations,
            hand_variant="wujihand2",
        )
    robot.metadata["dataset_name"] = adapter.display_name
    robot.metadata["source_format"] = adapter.variant(Path(args.input))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer_kwargs = dict(
        output_path=output,
        robot=robot,
        motion=motion,
        episode_path=Path(args.input),
        fps=args.fps,
        action_steps=args.action_steps,
        mano_arrays=None,
        mano_left_convention=args.mano_left_convention,
    )
    if dataset_name == "fourier_actionnet":
        write_fourier_spark_hdf5(**writer_kwargs)
    elif dataset_name == "egovla_bench":
        write_egovla_spark_hdf5(**writer_kwargs)
    else:
        write_agibot_spark_hdf5(**writer_kwargs)
    print(
        json.dumps(
            {
                "output": str(output),
                "output_format": "spark_hdf5_v1",
                "dataset": adapter.display_name,
                "source": motion.source,
                "frames": int(len(robot.timestamps)),
                "episode_truncation": int(args.action_steps),
                "camera_alignment": {"skipped": True},
                "source_readonly": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )




def _retarget_egovla_table_motion(motion):
    """Convert EgoVLA Inspire qpos with the reviewed direct table codec."""
    from .datasets.egovla_bench.hand_mapping import (
        CODEC_ID,
        HandCodec,
        wuji20_from_qpos,
    )

    qpos = motion.metadata.get("qpos")
    if qpos is None:
        raise ValueError("EgoVLA motion is missing qpos for the hand table codec")
    q20 = wuji20_from_qpos(qpos, codec=HandCodec())
    ee = np.asarray(motion.wrist_poses, dtype=np.float64)
    if ee.shape != (len(motion.timestamps), 2, 7):
        raise ValueError(f"EgoVLA EE shape {ee.shape} does not match motion timestamps")
    state = pack_spark_state(ee[:, 0], q20[:, 0], ee[:, 1], q20[:, 1])
    state_valid = np.zeros_like(state, dtype=bool)
    wrist_valid = np.asarray(motion.wrist_valid, dtype=bool)
    hand_valid = np.all(np.isfinite(q20), axis=-1)
    state_valid[:, 0:7] = wrist_valid[:, 0, None]
    state_valid[:, 7:27] = hand_valid[:, 0, None]
    state_valid[:, 27:34] = wrist_valid[:, 1, None]
    state_valid[:, 34:54] = hand_valid[:, 1, None]
    state[~state_valid] = np.nan
    metadata = {
        key: value
        for key, value in motion.metadata.items()
        if key not in {"joints_world", "qpos", "stored_fingertips"}
    }
    metadata.update(
        {
            "hand_codec": CODEC_ID,
            "hand_pipeline": "measured_inspire12 -> pure_index_sign_unbounded_wuji20",
            "hand_ik": False,
            "retargeter": "egovla_vitra_index_sign_table",
            "retarget_error_definition": "not_applicable_no_ik",
            "wuji_order": "isaac_wujihand2",
            "hand_variant": "wujihand2",
            "coordinate_frame": "source_robot_as_is",
        }
    )
    return UnifiedRobotMotion(
        timestamps=motion.timestamps,
        state=state,
        state_valid=state_valid,
        source=f"{motion.source}:inspire12_wuji20_vitra_index_sign_unbounded_v2",
        retarget_error=np.full((len(state), 2), np.nan, dtype=np.float64),
        action_source="not_available",
        instruction=motion.instruction,
        metadata=metadata,
    )
