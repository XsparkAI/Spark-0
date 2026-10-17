"""Deployment-side codec between env-space and model-space vectors.

The robot client exchanges RAW vectors (packed by
``spark0.utils.robot_state.pack_robot_state`` /
``unpack_robot_state``): EE poses are 7-d ``[xyz, quat_wxyz]``, hand/gripper
joints are absolute. The trained model instead sees the CONVERTED
representation declared by the embodiment YAML's ``pose_representation``
block (see ``third_party/starVLA/examples/Spark0/train_files/data_registry/
ee_delta_dataset.py``): e.g. 9-d ``abs_xyz_rot6d`` / ``cam_abs_xyz_rot6d``
states and 9-d chunk-wise ``delta_xyz_rot6d`` actions. ``cam_abs_*`` uses the
same encode/decode as ``abs_*``; the camera-frame rewrite is applied outside
this codec (``spark0/policy.py`` when ``ee_in_camera_frame`` is on).

This codec re-applies the exact training-time conversion on the way in
(raw obs state -> model state) and inverts it on the way out
(un-normalized model action chunk -> absolute raw actions):

  state  : per pose key, ``xyz_wxyz(7)`` -> ``abs_xyz_rot6d`` /
           ``cam_abs_xyz_rot6d`` (9) (or raw).
  action : per pose key, ``delta_xyz_rot6d(9)`` composed onto the CURRENT
           state pose (the reference frame, taken from the observation used
           for that inference): ``T_i = T_state @ T_rel_i``; or absolute
           ``abs_*`` / ``cam_abs_*`` rot6d→quat. The result is re-encoded as
           ``xyz_wxyz(7)`` so ``get_action`` outputs are directly deployable
           absolute pose targets. Output quaternions are hemisphere-aligned
           (row 0 to an anchor orientation, later rows to their predecessor)
           so the emitted stream never jumps between q and -q — the per-row
           ``w >= 0`` canonicalization of ``matrix_to_quat_wxyz`` otherwise
           flips sign mid-chunk whenever the wrist trajectory crosses
           ``w == 0``. The anchor defaults to the reference state pose, but
           callers should pass ``anchor_quats`` (the LAST EMITTED action, see
           :meth:`EmbodimentActionCodec.action_pose_quats`): the state quat
           comes from the env and many drivers re-canonicalize it to
           ``w >= 0``, so anchoring to it re-introduces exactly the q/-q jump
           at chunk boundaries that this alignment exists to remove.

Because every chunk row is relative to the SAME reference pose (the current
state; RT-relative, not frame-to-frame), rows must NOT be chained/accumulated
across steps — each row composes with the one cached reference pose. Training
encodes the delta against this same current-state pose, so row 0 is the first
action to execute and is NOT skipped (``identity_prefix_rows == 0``).

Non-pose keys (hand joints, grippers) pass through unchanged in both
directions. When the checkpoint has no embodiment spec (legacy joint
checkpoints), the codec degenerates to an exact passthrough.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ee_delta_action import (
    delta_to_absolute_single_arm,
    matrix_to_quat_wxyz,
    matrix_to_rot6d,
    quat_wxyz_to_matrix,
    rot6d_to_matrix,
    single_arm_chunk_to_delta,
)
from .relative_pose import compose_poses_a_with_rel, relative_poses_b_in_a

POSE_RAW_DIM = 7  # [xyz(3), quat_wxyz(4)]
POSE_FORMAT_DIMS = {
    "raw": POSE_RAW_DIM,
    "abs_xyz_rot6d": 9,
    "cam_abs_xyz_rot6d": 9,
    "delta_xyz_rot6d": 9,
}
STATE_POSE_FORMATS = ("raw", "abs_xyz_rot6d", "cam_abs_xyz_rot6d")
ACTION_POSE_FORMATS = ("raw", "abs_xyz_rot6d", "cam_abs_xyz_rot6d", "delta_xyz_rot6d")
# Absolute xyz+rot6d encodings (world or camera; camera via ee_in_camera_frame).
ABS_XYZ_ROT6D_FORMATS = ("abs_xyz_rot6d", "cam_abs_xyz_rot6d")
CAMERA_FRAME_POSE_FORMATS = ("cam_abs_xyz_rot6d",)


@dataclass(frozen=True)
class _KeyCodec:
    key: str
    raw_dim: int
    model_dim: int
    pose_format: str | None  # None -> passthrough (non-pose key)


def _pose_to_abs_xyz_rot6d(pose_raw: np.ndarray) -> np.ndarray:
    """(..., 7) [xyz, quat_wxyz] -> (..., 9) [xyz, rot6d]."""
    rot6d = matrix_to_rot6d(quat_wxyz_to_matrix(pose_raw[..., 3:7]))
    return np.concatenate([pose_raw[..., :3], rot6d], axis=-1)


def _abs_xyz_rot6d_to_pose(abs9: np.ndarray) -> np.ndarray:
    """(..., 9) [xyz, rot6d] -> (..., 7) [xyz, quat_wxyz]."""
    quat = matrix_to_quat_wxyz(rot6d_to_matrix(abs9[..., 3:9]))
    return np.concatenate([abs9[..., :3], quat], axis=-1)


def _hemisphere_align_quats(
    quats: np.ndarray, reference_quat: np.ndarray | None
) -> np.ndarray:
    """Make a decoded quaternion sequence hemisphere-continuous.

    ``matrix_to_quat_wxyz`` canonicalizes each row to ``w >= 0``
    independently. q and -q encode the same rotation, but the env consumes
    the chunk as a TIME SERIES: on embodiments whose wrist orientation
    crosses ``w == 0`` mid-motion (both Tianji wrists hover there), the
    canonicalized stream jumps between q and -q while the recorded robot
    data is strictly sign-continuous — a naive downstream consumer
    (per-component filtering/interpolation, quaternion-error servo) reads
    that jump as a near-360 deg rotation command.

    Row 0 is aligned with ``reference_quat`` (the caller's anchor
    orientation, ideally the last action already sent to the robot); every
    later row is aligned with its predecessor. ``None`` falls back to
    anchoring on row 0 itself, which keeps the chunk internally continuous
    but says nothing about the previous chunk.
    """
    quats = np.array(quats, copy=True)
    prev = quats[0] if reference_quat is None else np.asarray(reference_quat, dtype=np.float64)
    for i in range(quats.shape[0]):
        if np.dot(quats[i], prev) < 0.0:
            quats[i] = -quats[i]
        prev = quats[i]
    return quats


def _build_key_codecs(
    key_dims: dict, pose_keys: set[str], pose_format: str
) -> list[_KeyCodec]:
    codecs = []
    for key, dim in key_dims.items():
        key = str(key)
        if key in pose_keys:
            if int(dim) != POSE_RAW_DIM:
                raise ValueError(
                    f"pose key {key!r} must be declared with raw dim {POSE_RAW_DIM} "
                    f"(xyz + quat_wxyz), got {dim}"
                )
            codecs.append(
                _KeyCodec(key, POSE_RAW_DIM, POSE_FORMAT_DIMS[pose_format], pose_format)
            )
        else:
            codecs.append(_KeyCodec(key, int(dim), int(dim), None))
    return codecs


def _expected_raw_dims(robot_action_dim_info: dict) -> list[int]:
    """Packed env layout: [arm_0, ee_0(, arm_1, ee_1)] (see pack_robot_state)."""
    arm_dims = list(robot_action_dim_info["arm_dim"])
    ee_dims = list(robot_action_dim_info["ee_dim"])
    dims = []
    for arm_dim, ee_dim in zip(arm_dims, ee_dims):
        dims.extend([int(arm_dim), int(ee_dim)])
    return dims


class EmbodimentActionCodec:
    """Raw (env) <-> model representation codec for one embodiment schema.

    Args:
        spec: embodiment spec dict (the ``embodiment_spec`` saved in the
            checkpoint's ``config.yaml``, i.e. the training-time schema), or
            ``None`` for an exact passthrough codec.
        robot_action_dim_info: ``{"arm_dim": [...], "ee_dim": [...]}`` from
            ``spark0.utils.robot_state`` for the eval robot; used to cross-check that
            the training schema matches the env packing layout.
        action_type: ``"joint"`` or ``"ee"`` (the serve config's ``action_type``). ``"ee"``
            arm slots are 7-d poses; ``"joint"`` embodiments must not declare
            a ``pose_representation`` block.
    """

    def __init__(
        self,
        spec: dict | None,
        robot_action_dim_info: dict,
        action_type: str,
    ) -> None:
        expected_dims = _expected_raw_dims(robot_action_dim_info)

        if spec is None:
            # Passthrough codec mirroring the env packing slices exactly.
            passthrough = [
                _KeyCodec(f"slot_{i}", dim, dim, None) for i, dim in enumerate(expected_dims)
            ]
            self.state_codecs = list(passthrough)
            self.action_codecs = list(passthrough)
            self.state_pose_format = None
            self.action_pose_format = None
            self._finalize(expected_dims, action_type)
            return

        pose_repr = spec.get("pose_representation") or {}
        pose_keys = {str(key) for key in pose_repr.get("keys", [])}
        self.state_pose_format = pose_repr.get("state", "abs_xyz_rot6d") if pose_keys else None
        self.action_pose_format = pose_repr.get("action", "delta_xyz_rot6d") if pose_keys else None
        if pose_keys:
            if self.state_pose_format not in STATE_POSE_FORMATS:
                raise ValueError(
                    f"pose_representation.state={self.state_pose_format!r}; "
                    f"expected one of {STATE_POSE_FORMATS}"
                )
            if self.action_pose_format not in ACTION_POSE_FORMATS:
                raise ValueError(
                    f"pose_representation.action={self.action_pose_format!r}; "
                    f"expected one of {ACTION_POSE_FORMATS}"
                )
            if action_type != "ee":
                raise ValueError(
                    f"checkpoint embodiment declares pose_representation keys "
                    f"{sorted(pose_keys)} but eval action_type={action_type!r}; "
                    "EE-pose checkpoints must be evaluated with action_type='ee'."
                )

        self.state_codecs = _build_key_codecs(
            dict(spec["state_keys"]), pose_keys, self.state_pose_format or "raw"
        )
        self.action_codecs = _build_key_codecs(
            dict(spec["action_keys"]), pose_keys, self.action_pose_format or "raw"
        )
        self._finalize(expected_dims, action_type)

        # Delta actions need a same-named 7-d reference pose in the state block.
        for codec in self.action_codecs:
            if codec.pose_format == "delta_xyz_rot6d" and codec.key not in self._state_raw_slices:
                raise KeyError(
                    f"action pose key {codec.key!r} uses delta_xyz_rot6d but has no "
                    "matching state key to provide the reference pose"
                )

    # ------------------------------------------------------------------
    def _finalize(self, expected_dims: list[int], action_type: str) -> None:
        # The env packs/unpacks fixed slices [arm_0, ee_0(, arm_1, ee_1)];
        # every slice boundary must fall on a codec key boundary so raw keys
        # map onto whole env slices in order (same total dim is not enough —
        # a reordered key layout would silently scramble arms and hands).
        expected_bounds = set(np.cumsum(expected_dims).tolist())
        for modality, codecs in (("state", self.state_codecs), ("action", self.action_codecs)):
            raw_dims = [c.raw_dim for c in codecs]
            raw_bounds = set(np.cumsum(raw_dims).tolist())
            if sum(raw_dims) != sum(expected_dims) or not expected_bounds <= raw_bounds:
                raise ValueError(
                    f"{modality} raw layout {raw_dims} does not align with the env "
                    f"packing layout {expected_dims} for action_type={action_type!r}; "
                    "the checkpoint was trained for a different embodiment."
                )

        def _slices(codecs: list[_KeyCodec], attr: str) -> dict[str, slice]:
            out, cursor = {}, 0
            for c in codecs:
                dim = getattr(c, attr)
                out[c.key] = slice(cursor, cursor + dim)
                cursor += dim
            return out

        self._state_raw_slices = _slices(self.state_codecs, "raw_dim")
        self.raw_state_dim = sum(c.raw_dim for c in self.state_codecs)
        self.raw_action_dim = sum(c.raw_dim for c in self.action_codecs)
        self.model_state_dim = sum(c.model_dim for c in self.state_codecs)
        self.model_action_dim = sum(c.model_dim for c in self.action_codecs)

    # ------------------------------------------------------------------
    @property
    def identity_prefix_rows(self) -> int:
        """Leading no-op rows in a decoded action chunk.

        ``delta_xyz_rot6d`` is RT-relative to the CURRENT state pose (not the
        action chunk's own frame 0), so row 0 already encodes the first action
        to execute (``action[0]`` relative to ``state[0]``) and is generally
        NOT the identity. There are therefore no leading no-op rows to skip.
        """
        return 0

    def _reference_quat(
        self, key: str, reference_state_raw: np.ndarray
    ) -> np.ndarray | None:
        """Current-orientation quat for hemisphere alignment, if available.

        Returns ``None`` when the state block has no same-named pose key or
        when the reference is degenerate (e.g. the zero placeholder used for
        checkpoints that do not consume state).
        """
        state_slice = self._state_raw_slices.get(key)
        if state_slice is None or (state_slice.stop - state_slice.start) != POSE_RAW_DIM:
            return None
        quat = reference_state_raw[state_slice][3:7]
        # `not norm >= eps` (not `norm < eps`): a NaN quat — e.g. a driver
        # publishing an uninitialized pose — must also count as degenerate,
        # or it silently becomes the hemisphere anchor for the whole chunk.
        if not np.linalg.norm(quat) >= 1e-6:
            return None
        return quat

    def _anchor_quat(
        self,
        key: str,
        reference_state_raw: np.ndarray,
        anchor_quats: dict[str, np.ndarray] | None,
    ) -> np.ndarray | None:
        """Hemisphere anchor for one pose key: caller override, else the state."""
        if anchor_quats:
            quat = anchor_quats.get(key)
            if quat is not None:
                quat = np.asarray(quat, dtype=np.float64).reshape(-1)
                if quat.shape == (4,) and np.linalg.norm(quat) >= 1e-6:
                    return quat
        return self._reference_quat(key, reference_state_raw)

    def action_pose_quats(self, action_raw: np.ndarray) -> dict[str, np.ndarray]:
        """Per-pose-key quats of one raw ACTION row — the hemisphere anchor.

        Feed the LAST ROW ALREADY SENT to the robot back into
        :meth:`action_model_to_raw` / :meth:`realign_action_pose_quats_to_state`
        so the next chunk continues the same quaternion branch. Unlike the
        state pose, this value never leaves the codec, so it cannot be
        re-canonicalized to ``w >= 0`` behind our back.
        """
        row = np.asarray(action_raw, dtype=np.float64).reshape(-1)
        if row.shape != (self.raw_action_dim,):
            raise ValueError(
                f"expected a raw action row ({self.raw_action_dim},), got {row.shape}"
            )
        return {
            key: row[sl][3:7].copy()
            for key, sl in self._pose_key_slices("action").items()
        }

    def _pose_key_slices(self, modality: str) -> dict[str, slice]:
        """Raw packed slices for pose keys of ``state`` or ``action``."""
        codecs = self.state_codecs if modality == "state" else self.action_codecs
        out, cursor = {}, 0
        for codec in codecs:
            sl = slice(cursor, cursor + codec.raw_dim)
            cursor += codec.raw_dim
            if codec.pose_format is not None:
                out[codec.key] = sl
        return out

    def _transform_packed_poses(
        self,
        packed_raw: np.ndarray,
        camera_pose_xyz_wxyz: np.ndarray,
        modality: str,
        transform,
    ) -> np.ndarray:
        """Apply ``transform(camera, poses)`` to every pose key of a packed vector.

        ``transform`` is a BATCHED pose transform (see spark0.relative_pose):
        it takes the single camera pose and an ``(N, 7)`` block of poses.

        ``modality`` must be given explicitly: the state and action blocks
        always share the same total raw dim (both are validated against the
        env packing layout), so it cannot be inferred from the shape.
        """
        if modality not in ("state", "action"):
            raise ValueError(f"modality must be 'state' or 'action', got {modality!r}")
        expected_dim = self.raw_state_dim if modality == "state" else self.raw_action_dim
        packed = np.asarray(packed_raw, dtype=np.float64).copy()
        camera = np.asarray(camera_pose_xyz_wxyz, dtype=np.float64).reshape(POSE_RAW_DIM)
        # Accept (D,) or (H, D); pose transforms apply along the last axis.
        single = packed.ndim == 1
        if single:
            packed = packed[None, :]
        if packed.ndim != 2 or packed.shape[-1] != expected_dim:
            raise ValueError(
                f"expected packed raw {modality} (..., {expected_dim}), got {packed.shape}"
            )
        # One batched call per pose key: `transform` takes (N, 7) against the
        # single camera pose, so the quaternion helpers' numpy overhead and the
        # camera matrix are paid once per key instead of once per row.
        for sl in self._pose_key_slices(modality).values():
            packed[:, sl] = transform(camera, packed[:, sl])
        return packed[0] if single else packed

    def poses_world_to_camera(
        self,
        packed_raw: np.ndarray,
        camera_pose_xyz_wxyz: np.ndarray,
        modality: str = "state",
    ) -> np.ndarray:
        """EE pose keys → camera frame; hands pass through (matches training)."""
        return self._transform_packed_poses(
            packed_raw, camera_pose_xyz_wxyz, modality, relative_poses_b_in_a
        )

    def poses_camera_to_world(
        self,
        packed_raw: np.ndarray,
        camera_pose_xyz_wxyz: np.ndarray,
        modality: str = "state",
    ) -> np.ndarray:
        """Inverse of :meth:`poses_world_to_camera` (camera-frame -> world)."""
        return self._transform_packed_poses(
            packed_raw, camera_pose_xyz_wxyz, modality, compose_poses_a_with_rel
        )

    def realign_action_pose_quats_to_state(
        self,
        action_chunk_raw: np.ndarray,
        reference_state_raw: np.ndarray,
        anchor_quats: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Hemisphere-align decoded action pose quats in the world frame.

        Used after camera→world, whose per-row ``w >= 0`` canonicalization
        destroys the alignment ``action_model_to_raw`` applied in the camera
        frame. Same anchor contract as ``action_model_to_raw``.
        """
        chunk = np.asarray(action_chunk_raw, dtype=np.float64).copy()
        reference = np.asarray(reference_state_raw, dtype=np.float64)
        action_slices = self._pose_key_slices("action")
        for key, sl in action_slices.items():
            ref_quat = self._anchor_quat(key, reference, anchor_quats)
            chunk[:, sl.start + 3 : sl.stop] = _hemisphere_align_quats(
                chunk[:, sl.start + 3 : sl.stop], ref_quat
            )
        return chunk

    def state_raw_to_model(self, state_raw: np.ndarray) -> np.ndarray:
        """Raw packed state (..., raw_state_dim) -> model state (..., model_state_dim).

        ``state_raw`` must already be in the training frame (world, or camera
        when ``ee_in_camera_frame`` was enabled at train time).
        """
        state_raw = np.asarray(state_raw, dtype=np.float64)
        if state_raw.shape[-1] != self.raw_state_dim:
            raise ValueError(
                f"expected raw state dim {self.raw_state_dim}, got {state_raw.shape[-1]}"
            )
        parts, cursor = [], 0
        for codec in self.state_codecs:
            block = state_raw[..., cursor : cursor + codec.raw_dim]
            cursor += codec.raw_dim
            if codec.pose_format in (None, "raw"):
                parts.append(block)
            elif codec.pose_format in ABS_XYZ_ROT6D_FORMATS:
                parts.append(_pose_to_abs_xyz_rot6d(block))
            else:  # pragma: no cover - blocked by STATE_POSE_FORMATS check
                raise ValueError(f"invalid state pose format {codec.pose_format!r}")
        return np.concatenate(parts, axis=-1).astype(np.float32)

    def action_model_to_raw(
        self,
        chunk_model: np.ndarray,
        reference_state_raw: np.ndarray,
        anchor_quats: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Un-normalized model action chunk -> absolute raw env actions.

        Args:
            chunk_model: (H, model_action_dim) chunk from the policy server
                (already un-normalized, still in the converted representation).
            reference_state_raw: (raw_state_dim,) the RAW packed state of the
                observation used for this inference; provides the t=0
                reference pose for every delta pose key.
            anchor_quats: per-pose-key hemisphere anchor (see
                :meth:`action_pose_quats`). Defaults to the reference state's
                own quats.

        Returns:
            (H, raw_action_dim) absolute env actions, ready for
            ``unpack_robot_state``.
        """
        chunk_model = np.asarray(chunk_model, dtype=np.float64)
        if chunk_model.ndim != 2 or chunk_model.shape[-1] != self.model_action_dim:
            raise ValueError(
                f"expected action chunk (H, {self.model_action_dim}), got {chunk_model.shape}"
            )
        reference_state_raw = np.asarray(reference_state_raw, dtype=np.float64)
        if reference_state_raw.shape != (self.raw_state_dim,):
            raise ValueError(
                f"expected reference state ({self.raw_state_dim},), "
                f"got {reference_state_raw.shape}"
            )

        parts, cursor = [], 0
        for codec in self.action_codecs:
            block = chunk_model[:, cursor : cursor + codec.model_dim]
            cursor += codec.model_dim
            if codec.pose_format in (None, "raw"):
                parts.append(block)
            elif codec.pose_format in ABS_XYZ_ROT6D_FORMATS:
                pose = _abs_xyz_rot6d_to_pose(block)
                reference_quat = self._anchor_quat(
                    codec.key, reference_state_raw, anchor_quats
                )
                pose[..., 3:7] = _hemisphere_align_quats(pose[..., 3:7], reference_quat)
                parts.append(pose)
            elif codec.pose_format == "delta_xyz_rot6d":
                reference_pose = reference_state_raw[self._state_raw_slices[codec.key]]
                positions, rotations = delta_to_absolute_single_arm(block, reference_pose)
                reference_quat = self._anchor_quat(
                    codec.key, reference_state_raw, anchor_quats
                )
                quats = _hemisphere_align_quats(
                    matrix_to_quat_wxyz(rotations), reference_quat
                )
                parts.append(np.concatenate([positions, quats], axis=-1))
            else:  # pragma: no cover - blocked by ACTION_POSE_FORMATS check
                raise ValueError(f"invalid action pose format {codec.pose_format!r}")
        return np.concatenate(parts, axis=-1).astype(np.float32)

    def action_raw_to_model(
        self, chunk_raw: np.ndarray, reference_state_raw: np.ndarray
    ) -> np.ndarray:
        """Absolute raw env actions -> un-normalized model action chunk.

        Exact inverse of :meth:`action_model_to_raw`. Real-time chunking needs
        it: a ``delta_xyz_rot6d`` row only means something together with the
        state pose it was measured against, so a chunk cached from an earlier
        inference cannot be reused as a guidance target for a chunk generated
        from a LATER observation. Absolute poses are the one representation
        that survives the switch, so the leftover is kept absolute and
        re-encoded here against the reference pose of the current inference.

        Args:
            chunk_raw: (H, raw_action_dim) absolute env actions, already in
                the training frame (world, or camera when the caller applies
                ``poses_world_to_camera`` first).
            reference_state_raw: (raw_state_dim,) RAW packed state of the
                observation this chunk is being encoded FOR — the same
                reference :meth:`action_model_to_raw` would decode against.

        Returns:
            (H, model_action_dim) un-normalized model-space actions.
        """
        chunk_raw = np.asarray(chunk_raw, dtype=np.float64)
        if chunk_raw.ndim != 2 or chunk_raw.shape[-1] != self.raw_action_dim:
            raise ValueError(
                f"expected raw action chunk (H, {self.raw_action_dim}), got {chunk_raw.shape}"
            )
        reference_state_raw = np.asarray(reference_state_raw, dtype=np.float64)
        if reference_state_raw.shape != (self.raw_state_dim,):
            raise ValueError(
                f"expected reference state ({self.raw_state_dim},), "
                f"got {reference_state_raw.shape}"
            )

        parts, cursor = [], 0
        for codec in self.action_codecs:
            block = chunk_raw[:, cursor : cursor + codec.raw_dim]
            cursor += codec.raw_dim
            if codec.pose_format in (None, "raw"):
                parts.append(block)
            elif codec.pose_format in ABS_XYZ_ROT6D_FORMATS:
                parts.append(_pose_to_abs_xyz_rot6d(block))
            elif codec.pose_format == "delta_xyz_rot6d":
                reference_pose = reference_state_raw[self._state_raw_slices[codec.key]]
                parts.append(single_arm_chunk_to_delta(block, reference_pose))
            else:  # pragma: no cover - blocked by ACTION_POSE_FORMATS check
                raise ValueError(f"invalid action pose format {codec.pose_format!r}")
        return np.concatenate(parts, axis=-1).astype(np.float32)
