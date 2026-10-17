"""MANO assets and FK.

Two call paths:

* **OpenAoE / already-parameterized MANO** — ``ManoModel`` + ``forward_hand``
  evaluate stored ``global_orient / hand_pose / betas / transl``. Default asset
  is the packaged right-hand ``mano_right.npz``; left uses a convention flag.
* **EgoPro 21 keypoints** — official ``mano_v1_2`` (``MANO_LEFT.pkl`` /
  ``MANO_RIGHT.pkl``) via ``--mano-assets``. Those keypoints are retargeted
  with ``mano_ik``; do not use the packaged npz or a mirrored right hand.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np

OFFICIAL_CACHE_FORMAT = "egopro-mano-official-cache-v1"
_OFFICIAL_CACHE_ARRAYS = (
    "v_template",
    "shapedirs",
    "posedirs",
    "J_regressor",
    "weights",
    "kintree_table",
    "hands_mean",
    "f",
)

# MANO kinematic layout (joint index → role). Tips come from mesh vertices.
# 0 wrist
# 1–3 index MCP/PIP/DIP, 4–6 middle, 7–9 pinky, 10–12 ring, 13–15 thumb CMC/MCP/IP
_MANO_FINGER_JOINTS = {
    "thumb": (13, 14, 15),
    "index": (1, 2, 3),
    "middle": (4, 5, 6),
    "ring": (10, 11, 12),
    "pinky": (7, 8, 9),
}
FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")
LeftConvention = Literal["mirror_model", "flip_aa", "right_then_mirror"]
# manopth / manotorch fingertip vertices (middle differs by side).
MANO_TIP_VERTEX = {
    "left": np.asarray([745, 317, 445, 556, 673], dtype=np.int64),
    "right": np.asarray([745, 317, 444, 556, 673], dtype=np.int64),
}


def _axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Rodrigues: ``[..., 3]`` axis-angle → ``[..., 3, 3]`` rotation."""
    aa = np.asarray(axis_angle, dtype=np.float64)
    leading = aa.shape[:-1]
    flat = aa.reshape(-1, 3)
    angle = np.linalg.norm(flat, axis=1)
    matrix = np.tile(np.eye(3, dtype=np.float64), (len(flat), 1, 1))
    nonzero = angle > 1e-8
    if np.any(nonzero):
        axis = flat[nonzero] / angle[nonzero, None]
        x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
        zeros = np.zeros_like(x)
        skew = np.stack(
            [
                zeros, -z, y,
                z, zeros, -x,
                -y, x, zeros,
            ],
            axis=1,
        ).reshape(-1, 3, 3)
        cos = np.cos(angle[nonzero])[:, None, None]
        sin = np.sin(angle[nonzero])[:, None, None]
        outer = axis[:, :, None] * axis[:, None, :]
        matrix[nonzero] = (
            cos * np.eye(3)
            + sin * skew
            + (1.0 - cos) * outer
        )
    return matrix.reshape(leading + (3, 3))


def _patch_legacy_numpy_aliases() -> None:
    """Official MANO pickles need numpy aliases removed in 1.24+."""
    aliases = {
        "bool": np.bool_,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)


def resolve_mano_assets(explicit: Path | str | None) -> Path:
    """Return the directory that contains official ``MANO_{LEFT,RIGHT}.pkl``.

    Pass the ``mano_v1_2`` root or its ``models/`` folder. There is no default
    and no filesystem search — EgoPro callers must pass ``--mano-assets``.
    """
    if explicit is None or str(explicit).strip() == "":
        raise ValueError(
            "EgoPro MANO requires --mano-assets pointing at official mano_v1_2 "
            "(models/MANO_LEFT.pkl and models/MANO_RIGHT.pkl)"
        )
    root = Path(explicit).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"MANO assets directory not found: {root}")
    models = root / "models" if (root / "models").is_dir() else root
    left = models / "MANO_LEFT.pkl"
    right = models / "MANO_RIGHT.pkl"
    if not left.is_file() or not right.is_file():
        raise FileNotFoundError(
            f"Expected MANO_LEFT.pkl and MANO_RIGHT.pkl under {models}"
        )
    return models


def official_mano_pkl(assets_root: Path | str, hand: str) -> Path:
    if hand not in ("left", "right"):
        raise ValueError("hand must be 'left' or 'right'")
    return resolve_mano_assets(assets_root) / f"MANO_{hand.upper()}.pkl"


def official_mano_cache_path(pkl_path: Path | str) -> Path:
    """``mano_v1_2/models/MANO_LEFT.pkl`` → ``mano_v1_2/MANO_LEFT.cache.npz``."""
    path = Path(pkl_path)
    models_dir = path.parent
    root = models_dir.parent if models_dir.name == "models" else models_dir
    return root / f"{path.stem}.cache.npz"


def _pkl_fingerprint(pkl_path: Path) -> tuple[int, int]:
    stat = pkl_path.stat()
    return int(stat.st_mtime_ns), int(stat.st_size)


def _read_official_cache(cache_path: Path, pkl_path: Path) -> dict[str, np.ndarray] | None:
    if not cache_path.is_file():
        return None
    try:
        payload = np.load(cache_path, allow_pickle=False)
        if str(np.asarray(payload["format"]).item()) != OFFICIAL_CACHE_FORMAT:
            return None
        mtime_ns, size = _pkl_fingerprint(pkl_path)
        if (
            int(np.asarray(payload["source_mtime_ns"]).item()) != mtime_ns
            or int(np.asarray(payload["source_size"]).item()) != size
        ):
            return None
        return {
            key: np.asarray(payload[key])
            for key in _OFFICIAL_CACHE_ARRAYS
        }
    except (OSError, KeyError, ValueError):
        return None


def _write_official_cache(
    cache_path: Path, pkl_path: Path, data: dict[str, np.ndarray]
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    mtime_ns, size = _pkl_fingerprint(pkl_path)
    tmp = cache_path.with_name(f".{cache_path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(
            tmp,
            format=np.asarray(OFFICIAL_CACHE_FORMAT),
            source_name=np.asarray(pkl_path.name),
            source_mtime_ns=np.int64(mtime_ns),
            source_size=np.int64(size),
            **{key: data[key] for key in _OFFICIAL_CACHE_ARRAYS},
        )
        os.replace(tmp, cache_path)
    finally:
        tmp.unlink(missing_ok=True)


def _unpickle_official(path: Path) -> dict[str, np.ndarray]:
    import pickle

    _patch_legacy_numpy_aliases()
    with path.open("rb") as stream:
        raw = pickle.load(stream, encoding="latin1")

    def to_np(key: str) -> np.ndarray:
        value = raw[key]
        if hasattr(value, "r"):
            value = value.r
        if hasattr(value, "toarray"):
            value = value.toarray()
        return np.asarray(value)

    faces = to_np("f")
    return {
        "v_template": to_np("v_template").astype(np.float64),
        "shapedirs": to_np("shapedirs").astype(np.float64),
        "posedirs": to_np("posedirs").astype(np.float64),
        "J_regressor": to_np("J_regressor").astype(np.float64),
        "weights": to_np("weights").astype(np.float64),
        "kintree_table": to_np("kintree_table").astype(np.int64),
        "hands_mean": to_np("hands_mean").astype(np.float64).reshape(-1),
        "f": faces.astype(np.int64),
    }


def _load_official_pkl(path: Path) -> dict[str, np.ndarray]:
    cache_path = official_mano_cache_path(path)
    cached = _read_official_cache(cache_path, path)
    if cached is not None:
        return cached
    data = _unpickle_official(path)
    try:
        _write_official_cache(cache_path, path, data)
        print(f"MANO wrote official cache {cache_path}", flush=True)
    except OSError as error:
        print(f"MANO cache write skipped ({error})", flush=True)
    return data


class ManoModel:
    """One MANO hand: packaged ``.npz`` (OpenAoE) or official ``.pkl`` (EgoPro)."""

    def __init__(self, asset_path: str | Path, *, side: str | None = None) -> None:
        path = Path(asset_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"MANO model not found: {path}. The packaged right hand is "
                "data_tools/data_alignment/assets/mano_right.npz; official MANO_*.pkl "
                "files come from https://mano.is.tue.mpg.de (pass --mano-model)."
            )
        if path.suffix.lower() == ".pkl":
            data = _load_official_pkl(path)
            inferred = "left" if "LEFT" in path.name.upper() else "right"
            self.side = side or inferred
            self.v_template = data["v_template"]
            self.shapedirs = data["shapedirs"]
            self.posedirs = data["posedirs"]
            self.j_regressor = data["J_regressor"]
            self.weights = data["weights"]
            self.kintree_table = data["kintree_table"]
            self.hands_mean = data["hands_mean"]
            self.faces = data["f"]
            self.tip_ids = MANO_TIP_VERTEX[self.side].copy()
        else:
            data = np.load(path)
            self.side = side or "right"
            self.v_template = np.asarray(data["v_template"], dtype=np.float64)
            self.shapedirs = np.asarray(data["shapedirs"], dtype=np.float64)
            self.posedirs = np.asarray(data["posedirs"], dtype=np.float64)
            self.j_regressor = np.asarray(data["J_regressor"], dtype=np.float64)
            self.weights = np.asarray(data["weights"], dtype=np.float64)
            self.kintree_table = np.asarray(data["kintree_table"], dtype=np.int64)
            self.hands_mean = np.asarray(data["hands_mean"], dtype=np.float64)
            self.tip_ids = np.asarray(data["tip_ids"], dtype=np.int64)
            if "f" in data.files:
                self.faces = np.asarray(data["f"], dtype=np.int64)
            elif "faces" in data.files:
                self.faces = np.asarray(data["faces"], dtype=np.int64)
            else:
                raise ValueError(f"MANO asset {path} has no faces ('f')")
        if self.shapedirs.shape[-1] < 10:
            raise ValueError(f"Expected >=10 shape components, got {self.shapedirs.shape}")
        self.shapedirs = self.shapedirs[..., :10]
        parents = self.kintree_table[0].copy()
        parents[0] = -1
        self.parents = parents
        self.asset_path = path

    @classmethod
    def from_official(cls, assets_root: Path | str, hand: str) -> "ManoModel":
        """Load native ``MANO_{LEFT,RIGHT}.pkl`` from a ``mano_v1_2`` tree."""
        return cls(official_mano_pkl(assets_root, hand), side=hand)

    def forward(
        self,
        global_orient: np.ndarray,
        hand_pose: np.ndarray,
        betas: np.ndarray,
        transl: np.ndarray,
        *,
        flat_hand_mean: bool = False,
        return_vertices: bool = False,
    ) -> dict[str, np.ndarray]:
        """Run MANO FK for a batch of right hands.

        Parameters use OpenAoE / smplx conventions:
        ``global_orient`` ``[B,3]``, ``hand_pose`` ``[B,45]``, ``betas`` ``[B,10]``,
        ``transl`` ``[B,3]`` (meters).
        """
        global_orient = np.asarray(global_orient, dtype=np.float64).reshape(-1, 3)
        hand_pose = np.asarray(hand_pose, dtype=np.float64).reshape(-1, 45)
        betas = np.asarray(betas, dtype=np.float64).reshape(-1, 10)
        transl = np.asarray(transl, dtype=np.float64).reshape(-1, 3)
        batch = global_orient.shape[0]
        if not (
            hand_pose.shape[0] == batch
            and betas.shape[0] == batch
            and transl.shape[0] == batch
        ):
            raise ValueError("MANO batch dimensions must match")

        pose_hand = hand_pose if flat_hand_mean else hand_pose + self.hands_mean
        full_pose = np.concatenate([global_orient, pose_hand], axis=1).reshape(batch, 16, 3)
        rot_mats = _axis_angle_to_matrix(full_pose)

        v_shaped = self.v_template + np.einsum("vcd,bd->bvc", self.shapedirs, betas)
        joints = np.einsum("jv,bvk->bjk", self.j_regressor, v_shaped)

        ident = np.eye(3, dtype=np.float64)
        pose_feature = (rot_mats[:, 1:] - ident).reshape(batch, -1)
        v_posed = v_shaped + np.einsum("vpc,bc->bvp", self.posedirs, pose_feature)

        transforms = self._batch_rigid_transform(rot_mats, joints)
        posed_joints = transforms[:, :, :3, 3]

        rel_transforms = transforms.copy()
        packed = np.zeros((batch, 16, 4, 4), dtype=np.float64)
        packed[:, :, :3, :3] = ident
        packed[:, :, :3, 3] = joints
        packed[:, :, 3, 3] = 1.0
        rel_transforms = transforms @ np.linalg.inv(packed)

        # Vertices are required for fingertip landmarks.
        t = np.einsum("vj,bjkl->bvkl", self.weights, rel_transforms)
        rest_hom = np.concatenate(
            [v_posed, np.ones((batch, v_posed.shape[1], 1), dtype=np.float64)],
            axis=-1,
        )
        vertices = np.einsum("bvkl,bvl->bvk", t, rest_hom)[..., :3]

        tips = vertices[:, self.tip_ids]
        # tip order in asset: thumb, index, middle, ring, pinky
        joints21 = np.concatenate([posed_joints, tips], axis=1)
        joints21 = joints21 + transl[:, None, :]
        vertices = vertices + transl[:, None, :]

        result = {
            "joints16": posed_joints + transl[:, None, :],
            "joints21": joints21,
        }
        if return_vertices:
            result["vertices"] = vertices
        return result

    def _batch_rigid_transform(
        self,
        rot_mats: np.ndarray,
        joints: np.ndarray,
    ) -> np.ndarray:
        """Accumulate absolute joint transforms ``[B,16,4,4]``."""
        batch = rot_mats.shape[0]
        parents = self.parents
        rel_joints = joints.copy()
        rel_joints[:, 1:] = joints[:, 1:] - joints[:, parents[1:]]

        transforms_abs = np.zeros((batch, 16, 4, 4), dtype=np.float64)
        for joint_index in range(16):
            transform = np.zeros((batch, 4, 4), dtype=np.float64)
            transform[:, :3, :3] = rot_mats[:, joint_index]
            transform[:, :3, 3] = rel_joints[:, joint_index]
            transform[:, 3, 3] = 1.0
            parent = int(parents[joint_index])
            if parent < 0:
                transforms_abs[:, joint_index] = transform
            else:
                transforms_abs[:, joint_index] = transforms_abs[:, parent] @ transform
        return transforms_abs


def _mirror_axis_angle(axis_angle: np.ndarray) -> np.ndarray:
    """Map right-hand axis-angle to left via ``(x,y,z) → (x,-y,-z)``."""
    aa = np.array(axis_angle, dtype=np.float64, copy=True)
    aa[..., 1] *= -1.0
    aa[..., 2] *= -1.0
    return aa


def forward_hand(
    model: ManoModel,
    *,
    is_right: bool,
    global_orient: np.ndarray,
    hand_pose: np.ndarray,
    betas: np.ndarray,
    transl: np.ndarray,
    flat_hand_mean: bool = False,
    left_convention: LeftConvention = "mirror_model",
) -> np.ndarray:
    """Return ``[B,21,3]`` joints (16 MANO + 5 tips) in the source world frame."""
    global_orient = np.asarray(global_orient, dtype=np.float64).reshape(-1, 3)
    hand_pose = np.asarray(hand_pose, dtype=np.float64).reshape(-1, 45)
    betas = np.asarray(betas, dtype=np.float64).reshape(-1, 10)
    transl = np.asarray(transl, dtype=np.float64).reshape(-1, 3)

    if is_right:
        return model.forward(
            global_orient,
            hand_pose,
            betas,
            transl,
            flat_hand_mean=flat_hand_mean,
        )["joints21"]

    if left_convention == "right_then_mirror":
        out = model.forward(
            global_orient,
            hand_pose,
            betas,
            transl,
            flat_hand_mean=flat_hand_mean,
        )["joints21"]
        out = np.array(out, copy=True)
        out[..., 0] *= -1.0
        return out

    if left_convention == "flip_aa":
        go = _mirror_axis_angle(global_orient)
        hp = _mirror_axis_angle(hand_pose.reshape(-1, 15, 3)).reshape(-1, 45)
        tr = np.array(transl, copy=True)
        tr[..., 0] *= -1.0
        out = model.forward(go, hp, betas, tr, flat_hand_mean=flat_hand_mean)["joints21"]
        out = np.array(out, copy=True)
        out[..., 0] *= -1.0
        return out

    # mirror_model: feed left params after (x,-y,-z) on rotations and -x on transl,
    # then flip resulting joints back (standard MANO_LEFT-from-RIGHT recipe).
    go = _mirror_axis_angle(global_orient)
    hp = _mirror_axis_angle(hand_pose.reshape(-1, 15, 3)).reshape(-1, 45)
    tr = np.array(transl, copy=True)
    tr[..., 0] *= -1.0
    # shapedirs[:,0,:] left/right bug compensation is applied by mirroring betas[0]
    betas_m = np.array(betas, copy=True)
    betas_m[..., 0] *= -1.0
    out = model.forward(go, hp, betas_m, tr, flat_hand_mean=flat_hand_mean)["joints21"]
    out = np.array(out, copy=True)
    out[..., 0] *= -1.0
    return out


def joints21_to_chains(
    joints21: np.ndarray,
    *,
    metacarpal_weight: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    """Build anatomical 5×5 chains from MANO 21 joints.

    Returns ``(points [5,5,3], weights [5,5])``. Non-thumb finger root (j=0) is a
    wrist→MCP interpolation so camera-alignment palm frames (which read j=0) and
    ``canonicalize_hand_points(..., knuckle_index=1)`` both have valid anchors.
    """
    joints21 = np.asarray(joints21, dtype=np.float64)
    if joints21.shape != (21, 3):
        raise ValueError(f"Expected joints21 [21,3], got {joints21.shape}")
    wrist = joints21[0]
    tips = {
        "thumb": joints21[16],
        "index": joints21[17],
        "middle": joints21[18],
        "ring": joints21[19],
        "pinky": joints21[20],
    }
    points = np.full((5, 5, 3), np.nan, dtype=np.float64)
    weights = np.zeros((5, 5), dtype=np.float64)
    for finger_index, name in enumerate(FINGER_ORDER):
        j0, j1, j2 = _MANO_FINGER_JOINTS[name]
        mcp, pip, dip = joints21[j0], joints21[j1], joints21[j2]
        tip = tips[name]
        if name == "thumb":
            # CMC / MCP / IP / tip; chain root is wrist (matches EgoDex thumb).
            points[finger_index] = np.stack([wrist, mcp, pip, dip, tip], axis=0)
            weights[finger_index] = 1.0
        else:
            metacarpal = wrist + 0.35 * (mcp - wrist)
            points[finger_index] = np.stack([metacarpal, mcp, pip, dip, tip], axis=0)
            weights[finger_index] = np.array(
                [metacarpal_weight, 1.0, 1.0, 1.0, 1.0], dtype=np.float64
            )
    return points, weights


def default_mano_asset() -> Path:
    return Path(__file__).resolve().parents[2] / "assets" / "mano_right.npz"
