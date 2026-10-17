"""EgoVLA-Humanoid-Sim → Spark HDF5 writer (wuji2 54-D or --to egovla 12-D)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from spark0.utils.image_codec import encode_image_bit

from ..schema import (
    UnifiedHumanMotion,
    UnifiedRobotMotion,
    empty_tactile,
)
from ..datasets.egovla_bench.camera import (
    CAMERA_SOURCE,
    official_head_intrinsics,
    official_head_pose,
    official_head_poses,
    write_official_extrinsic_attrs,
    write_official_intrinsic_attrs,
)
from .writer_common import (
    IMAGE_BITS_COLOR_ORDER,
    StateActionBlocks,
    _write_mano_groups,
    create_fixed_width_jpeg_colors,
    source_frame_indices,
    validate_spark_write,
    write_aligned_group,
    write_common_metadata,
    write_header,
    write_state_action_groups,
    write_validity_datasets,
)

HAND_VARIANT_INSPIRE12 = "inspire12"
EGOVLA = "egovla"


def _encode_egovla_rgb_jpegs(rgb: np.ndarray, *, quality: int = 95) -> list[bytes]:
    """Encode native 384×384 RGB (Isaac) frames as Spark image bits."""
    out: list[bytes] = []
    for index, frame in enumerate(rgb):
        if frame.shape[:2] != (384, 384) or frame.shape[-1] != 3:
            raise ValueError(f"EgoVLA RGB frame {index} shape {frame.shape}")
        try:
            out.append(encode_image_bit(np.ascontiguousarray(frame), quality=quality))
        except (TypeError, ValueError) as error:
            raise ValueError(f"JPEG encode failed at EgoVLA frame {index}") from error
    return out


def write_egovla_spark_hdf5(
    output_path: str | Path,
    robot: UnifiedRobotMotion | Any,
    motion: UnifiedHumanMotion,
    episode_path: str | Path,
    fps: float,
    action_steps: int = 1,
    mano_arrays: dict[str, np.ndarray] | None = None,
    mano_left_convention: str = "mirror_model",
) -> None:
    """Write EgoVLA-Humanoid-Sim → Spark HDF5 (native 384×384 JPEG)."""
    try:
        import h5py
    except ImportError as error:
        raise ImportError("Writing Spark HDF5 requires h5py") from error

    from ..datasets.egovla_bench.reader import (
        EGOVLA_IMAGE_HW,
        INSPIRE12_ORDER,
        is_egovla_episode,
        load_egovla_rgb,
        parse_egovla_episode,
    )

    source_frames = len(robot.timestamps)
    keep = validate_spark_write(fps, action_steps, source_frames)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    episode_path = Path(episode_path)
    if not is_egovla_episode(episode_path):
        raise ValueError(f"Not an EgoVLA episode: {episode_path}")
    _root, task, ep = parse_egovla_episode(episode_path)

    blocks = StateActionBlocks.from_robot(robot, action_steps=action_steps, keep=keep)
    joints_world = np.asarray(
        motion.metadata.get(
            "joints_world",
            np.full((keep, 2, 5, 5, 3), np.nan, dtype=np.float64),
        )[:keep],
        dtype=np.float64,
    )
    remap = False
    joint_confidence = np.asarray(motion.joint_confidence[:keep], dtype=np.float64)
    tactile, tactile_valid = empty_tactile(keep)
    tactile_valid[:] = False

    # arm_q lives in motion.metadata, which resample_human_motion passes through
    # untouched. Slicing it with output-grid indices is only correct while the
    # output grid coincides with the source grid; otherwise the arm joints end up
    # silently misaligned against the EE poses with validity still all-True.
    for key in ("left_arm_q", "right_arm_q"):
        length = len(motion.metadata[key])
        if length != source_frames:
            raise ValueError(
                f"{episode_path}: metadata['{key}'] has {length} frames but the "
                f"output grid has {source_frames}; arm joints are not resampled, "
                f"so --fps {fps} must match the source rate "
                f"{motion.metadata.get('source_fps')}"
            )

    left_arm = np.asarray(motion.metadata["left_arm_q"][:keep], dtype=np.float64)
    right_arm = np.asarray(motion.metadata["right_arm_q"][:keep], dtype=np.float64)
    left_arm_action = np.asarray(
        motion.metadata["left_arm_q"][action_steps : action_steps + keep],
        dtype=np.float64,
    )
    right_arm_action = np.asarray(
        motion.metadata["right_arm_q"][action_steps : action_steps + keep],
        dtype=np.float64,
    )
    left_arm_valid = np.isfinite(left_arm)
    right_arm_valid = np.isfinite(right_arm)
    left_arm_action_valid = np.isfinite(left_arm_action)
    right_arm_action_valid = np.isfinite(right_arm_action)

    # Raw has no K/E. Write EgoVLA_Release official world-fixed main_camera
    # (stretch-scaled onto the stored 384×384 JPEG). Same for both --to paths.
    head_intrinsics = official_head_intrinsics()
    head_poses, head_pose_valid = official_head_poses(keep)
    video_fps = float(motion.metadata.get("video_fps", fps))
    frame_indices = source_frame_indices(blocks.timestamps, video_fps)
    head_rgb = load_egovla_rgb(episode_path, "main", frame_indices)
    head_jpegs = _encode_egovla_rgb_jpegs(head_rgb)
    image_keys = list(motion.metadata.get("image_keys") or ["main"])
    wrist_jpegs: dict[str, list[bytes] | None] = {"left": None, "right": None}
    for side, key in (("left", "left_hand"), ("right", "right_hand")):
        if key not in image_keys:
            continue
        wrist_jpegs[side] = _encode_egovla_rgb_jpegs(
            load_egovla_rgb(episode_path, key, frame_indices)
        )
    max_jpeg = max(len(b) for b in head_jpegs)
    for payload in wrist_jpegs.values():
        if payload:
            max_jpeg = max(max_jpeg, max(len(b) for b in payload))
    head_shape = np.asarray([EGOVLA_IMAGE_HW[0], EGOVLA_IMAGE_HW[1], 3], dtype=np.int64)

    with h5py.File(output, "w") as file:
        write_header(
            file, instruction=robot.instruction, timestamps=blocks.timestamps, fps=fps
        )

        vision = file.create_group("vision")
        cam = vision.create_group("cam_head")
        create_fixed_width_jpeg_colors(cam, head_jpegs, max_jpeg)
        hintr = cam.create_dataset("intrinsics", data=head_intrinsics)
        write_official_intrinsic_attrs(hintr)
        head_ext = cam.create_dataset("extrinsics", data=head_poses)
        write_official_extrinsic_attrs(head_ext, constant=True)
        cam.create_dataset("shape", data=head_shape)
        cam.attrs["encoding"] = "jpeg"
        cam.attrs["color_order"] = IMAGE_BITS_COLOR_ORDER
        cam.attrs["source_topic"] = "observations/images/main"
        cam.attrs["resize"] = "native_384x384"

        wrist_valid: dict[str, np.ndarray] = {}
        for side in ("left", "right"):
            payload = wrist_jpegs[side]
            if payload is None:
                wrist_valid[side] = np.zeros(keep, dtype=bool)
                continue
            wcam = vision.create_group(f"cam_{side}_wrist")
            create_fixed_width_jpeg_colors(wcam, payload, max_jpeg)
            wcam.create_dataset("shape", data=head_shape)
            wcam.attrs["encoding"] = "jpeg"
            wcam.attrs["color_order"] = IMAGE_BITS_COLOR_ORDER
            wcam.attrs["source_topic"] = f"observations/images/{side}_hand"
            wcam.attrs["resize"] = "native_384x384"
            wrist_valid[side] = np.ones(keep, dtype=bool)

        file.create_dataset("intrinsics", data=head_intrinsics)
        write_official_intrinsic_attrs(file["intrinsics"])
        fixed = file.create_dataset("extrinsics", data=official_head_pose())
        write_official_extrinsic_attrs(fixed, constant=True)

        write_state_action_groups(
            file,
            blocks,
            tactile=tactile,
            tactile_source="not_available",
            arm_state_left=left_arm,
            arm_state_right=right_arm,
            arm_action_left=left_arm_action,
            arm_action_right=right_arm_action,
        )
        if remap:
            order = ",".join(INSPIRE12_ORDER)
            for group_name in ("state", "action"):
                for side in ("left", "right"):
                    ds = file[group_name][f"{side}_ee_joint_states"]
                    ds.attrs["order"] = order
                    ds.attrs["unit"] = "radian"
                    ds.attrs["hand_variant"] = HAND_VARIANT_INSPIRE12
        _write_mano_groups(
            file, mano_arrays, action_steps, keep, mano_left_convention
        )

        validity = write_validity_datasets(
            file,
            blocks,
            tactile_valid=tactile_valid,
            arm_valid_left=left_arm_valid,
            arm_valid_right=right_arm_valid,
            arm_action_valid_left=left_arm_action_valid,
            arm_action_valid_right=right_arm_action_valid,
            state_valid_name="state_38" if remap else "state_54",
        )
        validity.create_dataset("cam_head", data=np.ones(keep, dtype=bool))
        validity.create_dataset("cam_head_pose", data=head_pose_valid)
        for side in ("left", "right"):
            validity.create_dataset(f"cam_{side}_wrist", data=wrist_valid[side])
            validity.create_dataset(
                f"cam_{side}_wrist_pose", data=np.zeros(keep, dtype=bool)
            )

        write_aligned_group(
            file,
            joints_world=joints_world,
            joint_confidence=joint_confidence,
            retarget_error=blocks.retarget_error,
        )

        provenance = {
            key: value
            for key, value in robot.metadata.items()
            if key
            not in {
                "joints_world",
                "qpos",
                "left_arm_q",
                "right_arm_q",
                "head_poses",
                "stored_fingertips",
            }
        }
        if remap:
            provenance.update(
                {
                    "converter": "spark_data.remap-egovla",
                    "source_task": task,
                    "source_episode": ep,
                    "hand_pipeline": "qpos_actuated_plus_urdf_mimic",
                    "hand_variant": HAND_VARIANT_INSPIRE12,
                    "hand_urdf": "data_tools/data_alignment/assets/inspire_hand/",
                    "inspire12_order": list(INSPIRE12_ORDER),
                    "ee_frame": robot.metadata.get("ee_frame", "egovla_inspire_ee"),
                    "coordinate_frame": "egovla_isaaclab_world",
                    "camera_status": CAMERA_SOURCE,
                    "vision_policy": "native_384x384_no_resize",
                    "wrist_sides_present": {
                        side: wrist_jpegs[side] is not None for side in ("left", "right")
                    },
                    "episode_truncation": int(action_steps),
                    "source_frame_count": int(source_frames),
                    "output_frame_count": int(keep),
                    "action_semantics": f"next_state_truncated:{action_steps}",
                    "requested_representations": [EGOVLA],
                }
            )
            source_label = f"egovla_humanoid_sim/{task}/{ep}:inspire12_remap"
            camera_alignment = {
                "ee_frame": robot.metadata.get("ee_frame", "egovla_inspire_ee"),
                "coordinate_frame": "egovla_isaaclab_world",
                "hand_variant": HAND_VARIANT_INSPIRE12,
                "skipped": True,
                "head_camera": CAMERA_SOURCE,
            }
        else:
            provenance.update(
                {
                    "converter": "spark_data.retarget-egovla",
                    "source_task": task,
                    "source_episode": ep,
                    "hand_pipeline": robot.metadata.get(
                        "hand_pipeline",
                        "measured_inspire12 -> pure_index_sign_unbounded_wuji20",
                    ),
                    "hand_codec": robot.metadata.get(
                        "hand_codec", "inspire12_wuji20_vitra_index_sign_unbounded_v2"
                    ),
                    "hand_ik": bool(robot.metadata.get("hand_ik", False)),
                    "hand_mapping": "data_tools/data_alignment/datasets/egovla_bench/hand_mapping.json",
                    "ee_frame": robot.metadata.get("ee_frame", "egovla_inspire_ee"),
                    "coordinate_frame": "egovla_isaaclab_world",
                    "camera_status": CAMERA_SOURCE,
                    "vision_policy": "native_384x384_no_resize",
                    "wrist_sides_present": {
                        side: wrist_jpegs[side] is not None for side in ("left", "right")
                    },
                    "episode_truncation": int(action_steps),
                    "source_frame_count": int(source_frames),
                    "output_frame_count": int(keep),
                    "action_semantics": f"next_state_truncated:{action_steps}",
                }
            )
            source_label = (
                f"egovla_humanoid_sim/{task}/{ep}:"
                f"{robot.metadata.get('hand_codec', 'inspire12_wuji20_vitra_index_sign_unbounded_v2')}"
            )
            camera_alignment = {
                "ee_frame": robot.metadata.get("ee_frame", "egovla_inspire_ee"),
                "coordinate_frame": "egovla_isaaclab_world",
                "hand_variant": robot.metadata.get("hand_variant", "wujihand2"),
                "wuji_order": "isaac_wujihand2",
                "hand_codec": robot.metadata.get(
                    "hand_codec", "inspire12_wuji20_vitra_index_sign_unbounded_v2"
                ),
                "skipped": True,
                "head_camera": CAMERA_SOURCE,
            }
        write_common_metadata(
            file,
            source=source_label,
            action_steps=action_steps,
            source_frames=source_frames,
            arm_state_action_source=b"egovla_h1_qpos",
            camera_alignment=camera_alignment,
            provenance=provenance,
        )
