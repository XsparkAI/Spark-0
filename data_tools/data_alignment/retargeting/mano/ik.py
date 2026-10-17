"""EgoPro MANO IK from the stored 21 MANUS keypoints.

Loads official ``MANO_LEFT.pkl`` / ``MANO_RIGHT.pkl`` from ``--mano-assets``.
Joints are returned in SNAP / ``MANO_JOINT_ORDER``. This is not the OpenAoE
npz path in ``mano.py``.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .model import ManoModel, resolve_mano_assets
from ...camera_alignment import wxyz_to_rotvec

MANO_JOINT_ORDER = (
    "wrist",
    "thumb_mcp",
    "thumb_pip",
    "thumb_dip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
)
# ManoModel.joints21 is native 16 + tips (thumb, index, middle, ring, pinky).
# manotorch then remaps to SNAP, which matches MANO_JOINT_ORDER.
NATIVE21_TO_SNAP = np.asarray(
    [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20],
    dtype=np.int64,
)
PALM_INDICES = np.asarray([0, 5, 9, 13, 17], dtype=np.int64)

_CACHE_LOCK = threading.Lock()
_RUNTIME_CACHE: dict[tuple[str, str], Any] = {}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else int(default)


def limit_cpu_threads(n: int | None = None) -> int:
    """Pin Torch / OpenMP / OpenCV to a few threads.

    This machine defaults to 100+ intra-op and OpenCV threads. MANO FK is a
    tiny graph; that oversubscription makes the first backward take tens of
    seconds. Call before importing cv2 or running IK.
    """
    threads = max(1, n if n is not None else _env_int("MANO_IK_TORCH_THREADS", 1))
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(threads))
    try:
        import torch

        torch.set_num_threads(threads)
        try:
            torch.set_num_interop_threads(threads)
        except RuntimeError:
            pass
    except Exception:
        pass
    try:
        import cv2

        cv2.setNumThreads(threads)
    except Exception:
        pass
    return threads


def _unit(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= 1e-8:
        raise ValueError("degenerate palm geometry")
    return vector / length


def palm_basis(joints: np.ndarray) -> np.ndarray:
    wrist = joints[0]
    finger_mcp = joints[[5, 9, 13, 17]]
    across = _unit(finger_mcp[0] - finger_mcp[-1])
    forward = _unit(np.mean(finger_mcp, axis=0) - wrist)
    normal = _unit(np.cross(across, forward))
    forward = _unit(np.cross(normal, across))
    return np.stack([across, forward, normal], axis=1)


def _axis_angle_to_matrix_torch(axis_angle: Any, torch: Any) -> Any:
    aa = axis_angle.reshape(-1, 3)
    angle = torch.linalg.norm(aa, dim=1)
    matrix = (
        torch.eye(3, dtype=aa.dtype, device=aa.device)
        .unsqueeze(0)
        .repeat(aa.shape[0], 1, 1)
    )
    nonzero = angle > 1e-8
    if bool(nonzero.any()):
        axis = aa[nonzero] / angle[nonzero].unsqueeze(1)
        x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
        zeros = torch.zeros_like(x)
        skew = torch.stack(
            [zeros, -z, y, z, zeros, -x, -y, x, zeros], dim=1
        ).reshape(-1, 3, 3)
        cos = torch.cos(angle[nonzero]).reshape(-1, 1, 1)
        sin = torch.sin(angle[nonzero]).reshape(-1, 1, 1)
        outer = axis.unsqueeze(2) * axis.unsqueeze(1)
        eye = torch.eye(3, dtype=aa.dtype, device=aa.device)
        matrix[nonzero] = cos * eye + sin * skew + (1.0 - cos) * outer
    return matrix.reshape(*axis_angle.shape[:-1], 3, 3)


class LocalManoBackend:
    """Torch FK over official left/right MANO pickles (manopth assets)."""

    def __init__(self, hand: str, mano_assets: Path | str, device: str) -> None:
        import torch

        if hand not in ("left", "right"):
            raise ValueError("hand must be 'left' or 'right'")
        self.torch = torch
        self.device = torch.device(device)
        self.hand = hand
        model = ManoModel.from_official(mano_assets, hand)
        self.faces = model.faces.copy()
        mean = np.asarray(model.hands_mean, dtype=np.float64).reshape(45)
        self.hand_mean = torch.as_tensor(
            mean, dtype=torch.float32, device=self.device
        )
        self._v_template = torch.as_tensor(
            model.v_template, dtype=torch.float32, device=self.device
        )
        self._shapedirs = torch.as_tensor(
            model.shapedirs, dtype=torch.float32, device=self.device
        )
        self._posedirs = torch.as_tensor(
            model.posedirs, dtype=torch.float32, device=self.device
        )
        self._j_regressor = torch.as_tensor(
            model.j_regressor, dtype=torch.float32, device=self.device
        )
        self._weights = torch.as_tensor(
            model.weights, dtype=torch.float32, device=self.device
        )
        self._parents = np.asarray(model.parents, dtype=np.int64)
        self._tip_ids = torch.as_tensor(
            model.tip_ids, dtype=torch.long, device=self.device
        )
        self._snap = torch.as_tensor(
            NATIVE21_TO_SNAP, dtype=torch.long, device=self.device
        )
        self._tip_template = self._v_template[self._tip_ids]
        self._tip_shapedirs = self._shapedirs[self._tip_ids]
        self._tip_posedirs = self._posedirs[self._tip_ids]
        self._tip_weights = self._weights[self._tip_ids]
        self._j_shaped0 = self._j_regressor @ self._v_template
        self._j_shapedirs = torch.einsum(
            "jv,vcd->jcd", self._j_regressor, self._shapedirs
        )

    def tensor(self, value: Any, *, grad: bool = False) -> Any:
        result = self.torch.as_tensor(
            value, dtype=self.torch.float32, device=self.device
        )
        result.requires_grad_(grad)
        return result

    def _fk(
        self,
        global_orient: Any,
        hand_pose: Any,
        betas: Any,
        *,
        need_vertices: bool = True,
    ) -> tuple[Any, Any]:
        torch = self.torch
        batch = global_orient.shape[0]
        # Absolute articulation (flat_hand_mean=True): stored handPose is the
        # posed 45-D vector, matching ManoRuntime's subtract-then-readd mean.
        full_pose = torch.cat([global_orient, hand_pose], dim=1).reshape(batch, 16, 3)
        rot_mats = _axis_angle_to_matrix_torch(full_pose, torch)
        if need_vertices:
            v_shaped = self._v_template + torch.einsum(
                "vcd,bd->bvc", self._shapedirs, betas
            )
            joints = torch.einsum("jv,bvk->bjk", self._j_regressor, v_shaped)
        else:
            # Keep the IK graph on 21 joints while retaining MANO shape.  The
            # exported schema has no extra global scale, so hand size must be
            # represented by the standard 10-D betas.
            v_shaped = None
            joints = self._j_shaped0.unsqueeze(0) + torch.einsum(
                "jcd,bd->bjc", self._j_shapedirs, betas
            )
        ident = torch.eye(3, dtype=rot_mats.dtype, device=rot_mats.device)
        pose_feature = (rot_mats[:, 1:] - ident).reshape(batch, -1)

        rel_joints = joints.clone()
        parents = self._parents
        rel_joints[:, 1:] = joints[:, 1:] - joints[:, parents[1:]]
        local = rot_mats.new_zeros(batch, 16, 4, 4)
        local[:, :, :3, :3] = rot_mats
        local[:, :, :3, 3] = rel_joints
        local[:, :, 3, 3] = 1.0
        abs_transforms = [local[:, 0]]
        for joint_index in range(1, 16):
            parent = int(parents[joint_index])
            abs_transforms.append(abs_transforms[parent] @ local[:, joint_index])
        transforms_abs = torch.stack(abs_transforms, dim=1)
        # packed is [I | J; 0 1], inverse is [I | -J; 0 1]
        packed_inv = rot_mats.new_zeros(batch, 16, 4, 4)
        packed_inv[:, :, :3, :3] = ident
        packed_inv[:, :, :3, 3] = -joints
        packed_inv[:, :, 3, 3] = 1.0
        rel_transforms = transforms_abs @ packed_inv
        posed_joints = transforms_abs[:, :, :3, 3]
        if need_vertices:
            v_posed = v_shaped + torch.einsum(
                "vpc,bc->bvp", self._posedirs, pose_feature
            )
            t = torch.einsum("vj,bjkl->bvkl", self._weights, rel_transforms)
            ones = torch.ones(
                (batch, v_posed.shape[1], 1),
                dtype=v_posed.dtype,
                device=v_posed.device,
            )
            rest_hom = torch.cat([v_posed, ones], dim=-1)
            vertices = torch.einsum("bvkl,bvl->bvk", t, rest_hom)[..., :3]
            tips = vertices[:, self._tip_ids]
        else:
            vertices = None
            v_tips = self._tip_template + torch.einsum(
                "vpc,bc->bvp", self._tip_posedirs, pose_feature
            )
            t = torch.einsum("vj,bjkl->bvkl", self._tip_weights, rel_transforms)
            ones = torch.ones(
                (batch, v_tips.shape[1], 1),
                dtype=v_tips.dtype,
                device=v_tips.device,
            )
            rest_hom = torch.cat([v_tips, ones], dim=-1)
            tips = torch.einsum("bvkl,bvl->bvk", t, rest_hom)[..., :3]
        joints21 = torch.cat([posed_joints, tips], dim=1)
        return vertices, joints21

    def _apply_root(
        self, points: Any, scale: float, translation: Any
    ) -> Any:
        scale_t = (
            scale
            if self.torch.is_tensor(scale)
            else points.new_tensor(float(scale))
        )
        return points.reshape(-1, 3) * scale_t + translation.reshape(1, 3)

    def forward(
        self,
        global_orient: Any,
        hand_pose: Any,
        betas: Any,
        scale: float,
        translation: Any,
    ) -> dict[str, Any]:
        go = global_orient.reshape(1, 3)
        hp = hand_pose.reshape(1, 45)
        be = betas.reshape(1, 10)
        vertices, joints21 = self._fk(go, hp, be, need_vertices=True)
        joints = joints21[:, self._snap]
        # Scale is not a MANO parameter. Apply it after LBS, then translate.
        return {
            "vertices": self._apply_root(vertices, scale, translation),
            "joints": self._apply_root(joints, scale, translation),
        }

    def forward_joints(
        self,
        global_orient: Any,
        hand_pose: Any,
        betas: Any,
        scale: float,
        translation: Any,
    ) -> Any:
        """21 SNAP joints only. IK does not need the 778-vertex LBS graph."""
        go = global_orient.reshape(1, 3)
        hp = hand_pose.reshape(1, 45)
        be = betas.reshape(1, 10)
        _vertices, joints21 = self._fk(go, hp, be, need_vertices=False)
        return self._apply_root(joints21[:, self._snap], scale, translation)


class LocalManoRuntime:
    def __init__(self, mano_assets: Path | str, device: str) -> None:
        assets = resolve_mano_assets(mano_assets)
        self.backends = {
            "left": LocalManoBackend("left", assets, device),
            "right": LocalManoBackend("right", assets, device),
        }


def cached_runtime(*, mano_assets: Path | str | None, device: str) -> LocalManoRuntime:
    assets = str(resolve_mano_assets(mano_assets))
    key = (assets, str(device))
    runtime = _RUNTIME_CACHE.get(key)
    if runtime is not None:
        return runtime
    with _CACHE_LOCK:
        runtime = _RUNTIME_CACHE.get(key)
        if runtime is None:
            if str(device).startswith("cpu"):
                limit_cpu_threads()
            runtime = LocalManoRuntime(assets, device)
            if str(device).startswith("cpu"):
                _warmup_cpu_autograd(runtime.backends["left"])
            _RUNTIME_CACHE[key] = runtime
        return runtime


def _warmup_cpu_autograd(backend: LocalManoBackend) -> None:
    """Pay the one-time CPU oneDNN/autograd compile before the frame loop.

    The cu128 wheel's first backward on this graph is ~10s; later frames are
    a few milliseconds. Doing it here keeps ``MANO IK left 1/T`` from looking
    hung.
    """
    print("MANO IK warmup CPU autograd", flush=True)
    started = time.perf_counter()
    hp = backend.hand_mean.detach().clone().requires_grad_(True)
    zero = backend.tensor(np.zeros(3))
    betas = backend.tensor(np.zeros(10))
    joints = backend.forward_joints(zero, hp, betas, 1.0, zero)
    joints.square().mean().backward()
    print(f"MANO IK warmup done {time.perf_counter() - started:.1f}s", flush=True)


def closed_form_palm(
    target_np: np.ndarray, model_joints: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial.transform import Rotation

    rotation = palm_basis(target_np) @ palm_basis(model_joints).T
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1
    global_np = Rotation.from_matrix(rotation).as_rotvec()
    translation_np = np.average(
        target_np[PALM_INDICES] - model_joints[PALM_INDICES],
        axis=0,
        weights=[20, 1, 1, 1, 1],
    )
    return global_np.astype(np.float64), np.asarray(translation_np, dtype=np.float64)


def fit_hand_mesh_fast(
    backend: LocalManoBackend,
    target_21: np.ndarray,
    previous: Mapping[str, Any] | None,
    *,
    bootstrap_iterations: int,
    live_iterations: int,
    need_vertices: bool = True,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Closed-form palm, then Adam on standard MANO parameters.

    The exported EgoDex-compatible schema contains root pose, 45-D hand pose,
    and 10-D betas, but no non-standard global scale.  On a new track, betas
    are therefore optimized together with articulation and root; later frames
    reuse that subject shape and optimize only pose/root.  ``scale`` remains
    fixed at 1 so every stored parameter tuple reconstructs independently.
    """
    torch = backend.torch
    target_np = np.asarray(target_21, dtype=np.float64)
    if target_np.shape != (21, 3):
        raise ValueError(f"expected (21, 3) joints, got {target_np.shape}")

    new_track = previous is None
    if new_track:
        hand_pose_np = backend.hand_mean.detach().cpu().numpy().astype(np.float64)
        betas_np = np.zeros(10, dtype=np.float64)
        iters = max(0, int(bootstrap_iterations))
        step_lr = 0.03
    else:
        hand_pose_np = np.asarray(previous["handPose"], dtype=np.float64).reshape(45)
        betas_np = np.asarray(previous["betas"], dtype=np.float64).reshape(10)
        iters = max(0, int(live_iterations))
        step_lr = 0.01

    optimize_shape = new_track and iters > 0
    betas = backend.tensor(betas_np, grad=optimize_shape)
    zero = backend.tensor(np.zeros(3))
    pose_t = backend.tensor(hand_pose_np)
    with torch.no_grad():
        rest = backend.forward_joints(zero, pose_t, betas, 1.0, zero).detach().cpu().numpy()
    scale = 1.0
    global_np, translation_np = closed_form_palm(target_np, rest)
    global_orient = backend.tensor(global_np, grad=iters > 0)
    translation = backend.tensor(translation_np, grad=iters > 0)

    if iters <= 0:
        vertices = None
        if need_vertices:
            with torch.no_grad():
                vertices = backend.forward(
                    global_orient,
                    backend.tensor(hand_pose_np),
                    betas,
                    scale,
                    translation,
                )["vertices"].detach().cpu().numpy()
        return vertices, {
            "handPose": hand_pose_np,
            "betas": betas_np,
            "scale": scale,
            "translation": translation_np,
            "global_orient": global_np,
        }

    hand_pose = backend.tensor(hand_pose_np, grad=True)
    target = backend.tensor(target_np)
    parameters = [hand_pose, global_orient, translation]
    if optimize_shape:
        parameters.append(betas)
    optimizer = torch.optim.Adam(parameters, lr=step_lr)
    for _ in range(iters):
        optimizer.zero_grad(set_to_none=True)
        joints = backend.forward_joints(
            global_orient, hand_pose, betas, scale, translation
        )
        loss = torch.mean((joints - target) ** 2)
        if optimize_shape:
            # Weak Gaussian prior prevents a noisy bootstrap frame from
            # producing extreme shape coefficients while still allowing size.
            loss = loss + 1e-7 * torch.mean(betas**2)
        loss.backward()
        optimizer.step()
        if optimize_shape:
            with torch.no_grad():
                betas.clamp_(-3.0, 3.0)
    pose_out = hand_pose.detach().cpu().numpy().astype(np.float64)
    betas_out = betas.detach().cpu().numpy().astype(np.float64)
    translation_out = translation.detach().cpu().numpy().astype(np.float64).reshape(3)
    global_out = global_orient.detach().cpu().numpy().astype(np.float64).reshape(3)
    vertices = None
    if need_vertices:
        with torch.no_grad():
            vertices = backend.forward(
                global_orient, hand_pose, betas, scale, translation
            )["vertices"].detach().cpu().numpy()
    return vertices, {
        "handPose": pose_out,
        "betas": betas_out,
        "scale": scale,
        "translation": translation_out,
        "global_orient": global_out,
    }


def vertices_from_mano_params(
    backend: LocalManoBackend,
    pose7: np.ndarray,
    joints45: np.ndarray,
    scale: float,
    betas: np.ndarray | None = None,
) -> np.ndarray | None:
    pose7 = np.asarray(pose7, dtype=np.float64).reshape(7)
    joints45 = np.asarray(joints45, dtype=np.float64).reshape(45)
    if betas is None:
        betas = np.zeros(10, dtype=np.float64)
    betas = np.asarray(betas, dtype=np.float64).reshape(10)
    if not (
        np.isfinite(pose7).all()
        and np.isfinite(joints45).all()
        and np.isfinite(betas).all()
        and np.isfinite(scale)
    ):
        return None
    torch = backend.torch
    with torch.no_grad():
        return (
            backend.forward(
                backend.tensor(wxyz_to_rotvec(pose7[3:7])),
                backend.tensor(joints45),
                backend.tensor(betas),
                float(scale),
                backend.tensor(pose7[:3]),
            )["vertices"]
            .detach()
            .cpu()
            .numpy()
        )
