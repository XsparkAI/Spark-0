"""EE pose (xyz + quat wxyz) -> chunk-wise delta EE action (xyz + rot6d).

Implements the RT-relative delta-chunk action representation (reference
implementation: Hy-Embodied-0.5-VLA, ``act_type = relative_chunk_ee_RT``).

Conventions:

1.  Per-arm action step is 10-d: ``[delta_xyz(3), rel_rot6d(6), gripper(1)]``;
    dual-arm is 20-d ``[left(10), right(10)]``.

2.  "Chunk-wise delta" (RT-relative): every action frame i in a chunk is
    expressed relative to a REFERENCE frame — the current ``state`` pose
    (``state[0]``) in training/deployment — in that reference's wrist frame:

        T_rel_i = T_ref^{-1} @ T_i         (4x4 homogeneous transforms)

        so  delta_xyz_i = R_ref^T @ (p_i - p_ref)
            R_rel_i     = R_ref^T @ R_i

    It is NOT a per-step (frame-to-frame) delta. When the reference is left
    unspecified it defaults to the chunk's own frame 0, in which case row 0 is
    the identity (delta_xyz = 0, rot6d = [1, 0, 0, 0, 1, 0]).

3.  ``rot6d`` is the FIRST TWO ROWS of the rotation matrix, flattened
    row-major: ``[r00, r01, r02, r10, r11, r12]`` (matches pytorch3d
    ``matrix_to_rotation_6d``). Not the first two columns.

4.  Quaternion convention: inputs are ``wxyz`` (as loaded from our data) and
    are normalized before use; conversion to rotation matrices is equivalent
    to scipy's ``Rotation.from_quat`` on the reordered ``xyzw`` quaternion.

5.  The gripper channel stays ABSOLUTE (not a delta) and is appended per arm
    after the rot6d block.

Only numpy is required.
"""

from __future__ import annotations

import numpy as np

# rot6d of the identity rotation: first two rows of eye(3), flattened.
IDENTITY_ROT6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


def quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """Batched unit-quaternion (w, x, y, z) -> rotation matrix.

    Args:
        quat_wxyz: (..., 4) quaternions in **wxyz** order.

    Returns:
        (..., 3, 3) rotation matrices. Quaternions are normalized first, so
        the result matches ``scipy.spatial.transform.Rotation.from_quat``
        (which takes xyzw) applied to the reordered quaternion.
    """
    q = np.asarray(quat_wxyz, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    m = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    m[..., 0, 0] = 1 - 2 * (y * y + z * z)
    m[..., 0, 1] = 2 * (x * y - w * z)
    m[..., 0, 2] = 2 * (x * z + w * y)
    m[..., 1, 0] = 2 * (x * y + w * z)
    m[..., 1, 1] = 1 - 2 * (x * x + z * z)
    m[..., 1, 2] = 2 * (y * z - w * x)
    m[..., 2, 0] = 2 * (x * z - w * y)
    m[..., 2, 1] = 2 * (y * z + w * x)
    m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rot6d: first two ROWS, row-major flatten.

    (..., 3, 3) -> (..., 6) = [r00, r01, r02, r10, r11, r12].
    """
    matrix = np.asarray(matrix)
    return matrix[..., :2, :].reshape(matrix.shape[:-2] + (6,))


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Inverse of :func:`matrix_to_rot6d` with Gram-Schmidt orthonormalization.

    Rows are orthonormalized (a = row0, b = row1), the third row is their
    cross product — the standard rot6d -> SO(3) recovery (Zhou et al., 2019),
    matching how consumers of this representation rebuild the matrix.
    """
    r = np.asarray(rot6d, dtype=np.float64)
    a, b = r[..., 0:3], r[..., 3:6]
    row0 = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b - np.sum(row0 * b, axis=-1, keepdims=True) * row0
    row1 = b / np.linalg.norm(b, axis=-1, keepdims=True)
    row2 = np.cross(row0, row1)
    return np.stack([row0, row1, row2], axis=-2)


def matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Batched rotation matrix -> unit quaternion in **wxyz** order.

    Inverse of :func:`quat_wxyz_to_matrix`. Uses the numerically stable
    Shepperd branch selection (largest of trace / diagonal entries) and
    returns the sign-canonicalized quaternion with ``w >= 0``.

    Args:
        matrix: (..., 3, 3) rotation matrices.

    Returns:
        (..., 4) quaternions ``[w, x, y, z]``.
    """
    m = np.asarray(matrix, dtype=np.float64)
    assert m.shape[-2:] == (3, 3), f"expected (..., 3, 3), got {m.shape}"

    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]

    # Four squared-magnitude candidates: [w^2, x^2, y^2, z^2] * 4 - ... form.
    q_abs_sq = np.stack(
        [
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ],
        axis=-1,
    )
    q_abs = np.sqrt(np.maximum(q_abs_sq, 0.0))

    # Candidate quaternions, each valid when its pivot component dominates.
    quat_by_w = np.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], axis=-1)
    quat_by_x = np.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], axis=-1)
    quat_by_y = np.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], axis=-1)
    quat_by_z = np.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], axis=-1)
    candidates = np.stack([quat_by_w, quat_by_x, quat_by_y, quat_by_z], axis=-2)

    pivot = np.argmax(q_abs, axis=-1)
    denom = 2.0 * np.take_along_axis(q_abs, pivot[..., None], axis=-1)
    quat = np.take_along_axis(candidates, pivot[..., None, None], axis=-2)[..., 0, :] / denom

    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    # Canonical sign: w >= 0 (q and -q encode the same rotation).
    quat = np.where(quat[..., :1] < 0.0, -quat, quat)
    return quat


def single_arm_chunk_to_delta(
    pose_chunk_xyz_wxyz: np.ndarray,
    reference_pose_xyz_wxyz: np.ndarray | None = None,
) -> np.ndarray:
    """One arm's absolute EE-pose chunk -> chunk-wise (RT-relative) delta.

    Args:
        pose_chunk_xyz_wxyz: (H, 7) = ``[xyz(3), quat_wxyz(4)]`` absolute EE
            action poses of ONE chunk.
        reference_pose_xyz_wxyz: (7,) reference frame ``[xyz, quat_wxyz]``.
            In training/deployment this is the CURRENT state pose (``state[0]``).
            When ``None`` it defaults to the chunk's own frame 0 (legacy
            RT-relative-to-action[0] behaviour, kept for standalone use).

    Returns:
        (H, 9) = ``[delta_xyz(3), rel_rot6d(6)]`` where
        ``delta_xyz_i = R_ref^T @ (p_i - p_ref)`` (expressed in the reference
        wrist frame) and ``rel_rot6d_i = rot6d(R_ref^T @ R_i)``. Row 0 is the
        identity only when the reference equals ``action[0]``.
    """
    chunk = np.asarray(pose_chunk_xyz_wxyz, dtype=np.float64)
    assert chunk.ndim == 2 and chunk.shape[1] == 7, (
        f"expected (H, 7) [xyz + quat_wxyz], got {chunk.shape}"
    )
    positions = chunk[:, :3]
    rotations = quat_wxyz_to_matrix(chunk[:, 3:7])  # (H, 3, 3)

    if reference_pose_xyz_wxyz is None:
        ref_pos = positions[0]
        ref_rot = rotations[0]
    else:
        ref = np.asarray(reference_pose_xyz_wxyz, dtype=np.float64).reshape(7)
        ref_pos = ref[:3]
        ref_rot = quat_wxyz_to_matrix(ref[3:7])

    r0_t = ref_rot.T  # R_ref^{-1}
    delta_xyz = (positions - ref_pos) @ r0_t.T  # == (R_ref^T @ dp^T)^T
    rel_rot = r0_t @ rotations  # (H, 3, 3), broadcast over H
    return np.concatenate([delta_xyz, matrix_to_rot6d(rel_rot)], axis=1)


def dual_arm_chunk_to_delta(chunk_16d_wxyz: np.ndarray) -> np.ndarray:
    """Dual-arm absolute EE chunk (quat wxyz) -> 20-d delta-action chunk.

    Args:
        chunk_16d_wxyz: (H, 16) =
            ``[L_xyz(3), L_quat_wxyz(4), L_gripper(1),
               R_xyz(3), R_quat_wxyz(4), R_gripper(1)]``.

    Returns:
        (H, 20) =
            ``[L_delta_xyz(3), L_rel_rot6d(6), L_gripper(1),
               R_delta_xyz(3), R_rel_rot6d(6), R_gripper(1)]``
        (the ``relative_chunk_ee_RT`` layout). Grippers are passed through
        as ABSOLUTE values.
    """
    chunk = np.asarray(chunk_16d_wxyz, dtype=np.float64)
    assert chunk.ndim == 2 and chunk.shape[1] == 16, (
        f"expected (H, 16) dual-arm [xyz + quat_wxyz + gripper] x2, got {chunk.shape}"
    )
    left = single_arm_chunk_to_delta(chunk[:, 0:7])
    right = single_arm_chunk_to_delta(chunk[:, 8:15])
    left_grip = chunk[:, 7:8]
    right_grip = chunk[:, 15:16]
    return np.concatenate([left, left_grip, right, right_grip], axis=1)


def delta_to_absolute_single_arm(
    delta_chunk_9d: np.ndarray, reference_pose_xyz_wxyz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Invert the delta: predicted chunk + current pose -> absolute targets.

    Used at deployment: the model predicts the (H, 9) delta chunk relative to
    the CURRENT EE pose (the chunk's t=0 frame).

        T_i = T_0 @ T_rel_i
        p_i = p_0 + R_0 @ delta_xyz_i
        R_i = R_0 @ rot6d_to_matrix(rel_rot6d_i)

    Args:
        delta_chunk_9d: (H, 9) = ``[delta_xyz(3), rel_rot6d(6)]``.
        reference_pose_xyz_wxyz: (7,) current absolute pose ``[xyz, quat_wxyz]``.

    Returns:
        (positions (H, 3), rotation_matrices (H, 3, 3)) absolute targets.
    """
    delta = np.asarray(delta_chunk_9d, dtype=np.float64)
    ref = np.asarray(reference_pose_xyz_wxyz, dtype=np.float64)
    r0 = quat_wxyz_to_matrix(ref[3:7])
    positions = ref[:3] + delta[:, :3] @ r0.T
    rotations = r0 @ rot6d_to_matrix(delta[:, 3:9])
    return positions, rotations


def episode_to_delta_chunks(
    episode_16d_wxyz: np.ndarray, chunk_size: int, stride: int = 1
) -> np.ndarray:
    """Slice a dual-arm episode into overlapping delta-action chunks.

    For each start index t (step ``stride``), the chunk covers frames
    ``[t, t + chunk_size)`` and is made relative to frame t. The tail of the
    episode is padded by repeating the last frame (delta stays constant, i.e.
    "hold pose"), matching common chunked-action training pipelines.

    Args:
        episode_16d_wxyz: (T, 16) absolute dual-arm EE trajectory (quat wxyz).
        chunk_size: H, the number of action steps per chunk (e.g. 50 at 10 Hz).
        stride: start-index step between consecutive chunks.

    Returns:
        (num_chunks, H, 20) delta-action chunks, num_chunks = ceil(T / stride).
    """
    episode = np.asarray(episode_16d_wxyz, dtype=np.float64)
    assert episode.ndim == 2 and episode.shape[1] == 16
    total = episode.shape[0]
    chunks = []
    for start in range(0, total, stride):
        window = episode[start : start + chunk_size]
        if window.shape[0] < chunk_size:
            pad = np.repeat(window[-1:], chunk_size - window.shape[0], axis=0)
            window = np.concatenate([window, pad], axis=0)
        chunks.append(dual_arm_chunk_to_delta(window))
    return np.stack(chunks, axis=0)


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    def random_pose_chunk(h):
        quats = rng.normal(size=(h, 4))
        quats /= np.linalg.norm(quats, axis=1, keepdims=True)
        return np.concatenate([rng.normal(size=(h, 3)), quats], axis=1)

    # 1. Frame 0 must map to the identity delta.
    chunk = random_pose_chunk(50)
    delta = single_arm_chunk_to_delta(chunk)
    assert np.allclose(delta[0, :3], 0) and np.allclose(delta[0, 3:], IDENTITY_ROT6D)

    # 2. Round trip: delta + reference pose recovers the absolute trajectory.
    pos, rot = delta_to_absolute_single_arm(delta, chunk[0])
    assert np.allclose(pos, chunk[:, :3], atol=1e-9)
    assert np.allclose(rot, quat_wxyz_to_matrix(chunk[:, 3:7]), atol=1e-9)

    # 2b. With an EXTERNAL reference (current state pose), row 0 is no longer
    # the identity, but the round trip still recovers the absolute action.
    ref_pose = random_pose_chunk(1)[0]
    delta_ref = single_arm_chunk_to_delta(chunk, ref_pose)
    assert not np.allclose(delta_ref[0, :3], 0)  # action[0] != state[0]
    pos_ref, rot_ref = delta_to_absolute_single_arm(delta_ref, ref_pose)
    assert np.allclose(pos_ref, chunk[:, :3], atol=1e-9)
    assert np.allclose(rot_ref, quat_wxyz_to_matrix(chunk[:, 3:7]), atol=1e-9)

    # 3. rot6d recovery is orthonormal and consistent.
    m = quat_wxyz_to_matrix(rng.normal(size=(8, 4)))
    m2 = rot6d_to_matrix(matrix_to_rot6d(m))
    assert np.allclose(m, m2, atol=1e-9)

    # 3b. matrix -> quat wxyz round trip (up to sign, canonicalized w >= 0).
    q = rng.normal(size=(64, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q_rt = matrix_to_quat_wxyz(quat_wxyz_to_matrix(q))
    q_canon = np.where(q[:, :1] < 0.0, -q, q)
    assert np.allclose(q_rt, q_canon, atol=1e-9)
    # Including near-degenerate rotations (180 deg about each axis).
    for axis_quat in ([0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]):
        m_axis = quat_wxyz_to_matrix(np.array(axis_quat, dtype=np.float64))
        q_axis = matrix_to_quat_wxyz(m_axis)
        assert np.allclose(quat_wxyz_to_matrix(q_axis), m_axis, atol=1e-9)

    # 4. Dual-arm layout: grippers pass through absolutely.
    dual = np.concatenate(
        [random_pose_chunk(50), rng.uniform(size=(50, 1)),
         random_pose_chunk(50), rng.uniform(size=(50, 1))], axis=1)
    out = dual_arm_chunk_to_delta(dual)
    assert out.shape == (50, 20)
    assert np.allclose(out[:, 9], dual[:, 7]) and np.allclose(out[:, 19], dual[:, 15])

    # 5. Episode chunking with tail padding holds the last pose.
    chunks = episode_to_delta_chunks(dual[:30], chunk_size=50, stride=10)
    assert chunks.shape == (3, 50, 20)
    assert np.allclose(chunks[-1][-1], chunks[-1][20])  # padded region constant

    print("all self-tests passed")
