# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Local Spark-0 addition (NOT upstream starVLA).
"""
Hierarchical dual-stream tactile → fixed token sequence for Action-DiT fusion.

Streams (data keys under ``tactile/.../{left,right}_ee``, or converter
``3d_left`` / ``3d_right`` packed as ``taxel_pressure_3d``):
  * **fast** ``region_pressure`` — fingertip / patch force history
    ``[B, H, R, T_r, C_r]`` → 1 token / region
  * **slow** ``taxel_pressure_3d`` — 3D taxel cloud
    ``[B, H, N, T_x, 4]`` (xyz + pressure; default N=320, T=1 — every
    shipped stem is single-frame). Opt into ``T>1`` and the last-frame
    xyz is kept while pressure goes through a thin per-taxel TemporalConv
    (``streams.taxel_pressure_3d.temporal_layers``, default 1; not
    ``tactile_expert.temporal_layers``, which sizes the trusted region encoder)

Hierarchy (``hierarchy.layout`` default ``fast_then_slow``):
  token layout ``[fast patch tokens | slow 3D tokens]``. ``encode`` accepts
  ``stream_mode`` ∈ {``both``, ``slow``, ``fast``}:
  * ``both`` — full encode (training / RTC replan)
  * ``slow`` — encode 3D cloud; fill fast slots with ``no_tactile``
  * ``fast`` — encode patch vectors; splice ``cached_slow_*`` (or
    ``no_tactile``) into slow slots for cheap high-rate refresh.
    Infer wiring: ``encode_with_slow_refresh`` (used by ``QwenPI_v3``);
    the slow-3D cache is per-``env_idx``, so RTC subset fires and
    ``max_infer_batch`` splits still hit.

Config: ``framework.tactile`` (usually via ``tactile_fusion: <stem>`` →
``configs/tactile_fusion/<stem>.yaml``). A stream is built iff
``streams.<name>.enable`` and ``fast_enable`` / ``slow_enable`` for its
``role`` (both flags default on, so they are a no-op on existing stems).

Disable: ``enable=false`` → do not construct (baseline bit-identical;
``QwenPI_v3`` sets ``tactile_encoder=None`` and ``encode_tactile`` returns
``(None, None)``). Missing / train dropout: ``present=False`` → learned
``no_tactile`` tokens. DiT ``append_context`` is gated by
``action_fusion.enable`` (train + ``predict_action`` + RTC). P1: the DiT
appends the full laid-out hierarchy (fast region tokens + slow 3D slice);
the raw region window is additionally flattened into AdaLN ``temb``.

Legacy single-frame ``points [B,H,N,C]`` / ``lattice`` configs without a
``streams:`` block remain supported (taxel stream with ``history_len=1``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from starVLA.model.modules.tactile.action_fusion import flatten_region_patch
from starVLA.model.modules.tactile.region_encoder import RegionLatticeCNN, RegionSetEncoder
from starVLA.model.modules.tactile.temporal import TemporalConvEncoder
from omegaconf import ListConfig
logger = logging.getLogger(__name__)


def _as_role_flag(value: Any, default: bool = True) -> bool:
    """``fast_enable`` / ``slow_enable``: missing / auto → ``default``."""
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    key = str(value).strip().lower()
    if key in ("", "auto", "none", "null"):
        return bool(default)
    if key in ("true", "1", "yes", "on"):
        return True
    if key in ("false", "0", "no", "off"):
        return False
    raise ValueError(f"fast_enable/slow_enable must be a bool, got {value!r}")


def _stream_role(name: str, stream_cfg: Dict[str, Any]) -> str:
    default = "fast" if name == "region_pressure" else "slow"
    return str(stream_cfg.get("role", default)).lower()


def stream_is_active(
    name: str,
    stream_cfg: Any,
    *,
    fast_enable: Any = True,
    slow_enable: Any = True,
    default_enable: bool = False,
) -> bool:
    """Stream is on iff ``enable`` and the matching role flag.

    Precedence (all must be true):
      ``framework.tactile.enable`` (caller) × ``streams.<name>.enable``
      × ``fast_enable`` / ``slow_enable`` for that stream's ``role``.

    ``default_enable`` is False at construct (a stub / missing stream does
    not build an expert — same as before the role flags) and True when the
    stream is *listed* in a packing or deploy spec (legacy ckpts omit
    ``enable``). Missing role flags default on, so ``fast_enable: true``
    is a no-op on every existing stem.
    """
    if stream_cfg is None or not hasattr(stream_cfg, "get"):
        return False
    if not bool(stream_cfg.get("enable", default_enable)):
        return False
    role = _stream_role(name, stream_cfg)
    if role == "fast":
        return _as_role_flag(fast_enable, True)
    if role == "slow":
        return _as_role_flag(slow_enable, True)
    return False


def _as_dict(cfg: Any) -> Dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    to_container = getattr(cfg, "to_container", None)
    if callable(to_container):
        return dict(to_container(resolve=True))
    if hasattr(cfg, "items"):
        return {str(k): v for k, v in cfg.items()}
    return dict(cfg)


def _uniform_region_map(num_taxels: int, num_regions: int) -> torch.Tensor:
    """Contiguous equal-as-possible blocks ``[0, 0, …, R-1, R-1]``.

    Remainder taxels go to the first regions (320 / 8 is even; 320 / 6
    → two×54 + four×53). Finger-pad exports that are ``N % R == 0``
    (80/5, 360/6) stay even.
    """
    n = int(num_taxels)
    r = int(num_regions)
    if n < 1 or r < 1:
        raise ValueError("uniform region_map needs num_taxels and num_regions >= 1")
    if n < r:
        raise ValueError(
            f"uniform region_map: num_taxels={n} < num_region_tokens={r}"
        )
    base, rem = divmod(n, r)
    sizes = [base + (1 if i < rem else 0) for i in range(r)]
    return torch.repeat_interleave(
        torch.arange(r, dtype=torch.long),
        torch.tensor(sizes, dtype=torch.long),
    )


def _load_region_map(path_or_list: Any, num_taxels: int) -> Optional[torch.Tensor]:
    """Static per-taxel region ids → ``LongTensor [num_taxels]``."""
    if path_or_list is None:
        return None
    to_dict = getattr(path_or_list, "to_dict", None)
    if callable(to_dict):
        path_or_list = to_dict(resolve=True)
    if isinstance(path_or_list, torch.Tensor):
        ids = path_or_list.long().reshape(-1)
    elif isinstance(path_or_list, (list, tuple, ListConfig)):
        ids = torch.tensor(list(path_or_list), dtype=torch.long)
    else:
        path = Path(str(path_or_list)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"tactile region_map not found: {path}")
        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix in {".yaml", ".yml"}:
            import yaml

            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            ids = torch.as_tensor(torch.load(path, map_location="cpu")).long().reshape(-1)
            data = None
        if data is not None:
            if isinstance(data, dict):
                data = data.get("region_ids", data.get("ids"))
                if data is None:
                    raise ValueError(
                        f"region_map {path} needs a list or keys region_ids/ids"
                    )
            ids = torch.tensor(list(data), dtype=torch.long)
    if ids.numel() != num_taxels:
        raise ValueError(
            f"region_map length {ids.numel()} != num_taxels={num_taxels}"
        )
    return ids


def _require_tensor(
    name: str,
    x: Any,
    *,
    ndim: int,
    tail: Sequence[int],
) -> torch.Tensor:
    """Require ``x`` to be a tensor with ``shape[1:] == tail`` (batch free)."""
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a Tensor, got {type(x)}")
    if x.ndim != ndim:
        raise ValueError(
            f"{name} must have ndim={ndim}, got shape {tuple(x.shape)}"
        )
    got_tail = tuple(x.shape[1:])
    exp_tail = tuple(tail)
    if got_tail != exp_tail:
        raise ValueError(
            f"{name} shape {tuple(x.shape)} mismatch: expected "
            f"[B, {', '.join(map(str, exp_tail))}]"
        )
    return x


def _normalize_streams(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return a ``streams`` mapping; synthesize from legacy ``input`` if needed."""
    streams = _as_dict(cfg.get("streams"))
    if streams:
        return streams

    # Legacy: single taxel / lattice cloud under top-level ``input`` + ``tactile_expert``.
    input_cfg = _as_dict(cfg.get("input"))
    expert_cfg = _as_dict(cfg.get("tactile_expert"))
    if not input_cfg and not expert_cfg:
        raise ValueError(
            "tactile config needs ``streams:`` (region_pressure / "
            "taxel_pressure_3d) or legacy ``input`` + ``tactile_expert``."
        )
    return {
        "region_pressure": {"enable": False},
        "taxel_pressure_3d": {
            "enable": True,
            "num_hands": int(input_cfg.get("num_hands", 2)),
            "num_taxels": int(input_cfg.get("num_taxels", 320)),
            "history_len": 1,
            "point_dim": int(input_cfg.get("point_dim", 4)),
            "xyz_frame": str(expert_cfg.get("xyz_frame", "ee")),
            "geometry_carrier": str(input_cfg.get("geometry_carrier", "point_set")),
            "region_map": input_cfg.get("region_map"),
            "keep_zero_pressure_points": bool(
                input_cfg.get("keep_zero_pressure_points", True)
            ),
            "pool_mode": str(input_cfg.get("pool_mode", "softmax")),
            "trim_frac": float(input_cfg.get("trim_frac", 0.1)),
            "drop_stuck_taxels": bool(input_cfg.get("drop_stuck_taxels", False)),
            "stuck_std": float(input_cfg.get("stuck_std", 1.0e-4)),
            "stuck_mean": float(input_cfg.get("stuck_mean", 0.05)),
            "stuck_abs": float(input_cfg.get("stuck_abs", 0.0)),
            "token_mixer_layers": int(input_cfg.get("token_mixer_layers", 0)),
            "type": str(expert_cfg.get("type", "region_set_encoder")),
            "num_region_tokens": int(expert_cfg.get("num_region_tokens", 8)),
            "num_point_tokens": int(expert_cfg.get("num_point_tokens", 0)),
            "use_fourier_features": bool(expert_cfg.get("use_fourier_features", True)),
            "fourier_bands": int(expert_cfg.get("fourier_bands", 4)),
        },
    }


class TactileEncoder(nn.Module):
    """Dual-hand dual-stream tactile → ``[B, num_tokens, output_dim]`` + mask."""

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        cfg = _as_dict(cfg)
        expert_cfg = _as_dict(cfg.get("tactile_expert"))
        fusion_cfg = _as_dict(cfg.get("action_fusion"))
        streams = _normalize_streams(cfg)

        if not bool(cfg.get("enable", False)):
            raise ValueError(
                "TactileEncoder constructed with enable=false; skip construction."
            )
        if not bool(expert_cfg.get("enable", True)):
            raise ValueError(
                "tactile.enable=true but tactile_expert.enable=false; "
                "disable the top-level switch instead."
            )

        self.hidden_dim = int(expert_cfg.get("hidden_dim", 256))
        self.num_layers = int(expert_cfg.get("num_layers", 3))
        self.output_dim = int(expert_cfg.get("output_dim", 1024))
        self.temporal_layers = int(expert_cfg.get("temporal_layers", 2))
        self.temporal_kernel = int(expert_cfg.get("temporal_kernel", 3))
        self.use_no_tactile_token = bool(
            expert_cfg.get("no_tactile_token", fusion_cfg.get("no_tactile_token", True))
        )
        self.modality_dropout_prob = float(
            fusion_cfg.get("modality_dropout_prob", cfg.get("modality_dropout_prob", 0.0))
        )
        # Independent of the region draw. Unset → same p as modality_dropout,
        # but a separate coin flip so dual-stream can train "force without 3D".
        if "slow_dropout_prob" in fusion_cfg:
            self.slow_dropout_prob = float(fusion_cfg.get("slow_dropout_prob"))
        else:
            self.slow_dropout_prob = self.modality_dropout_prob
        if self.hidden_dim < 1 or self.output_dim < 1 or self.num_layers < 1:
            raise ValueError("hidden_dim / output_dim / num_layers must be >= 1")
        if not (0.0 <= self.modality_dropout_prob <= 1.0):
            raise ValueError(
                f"modality_dropout_prob must be in [0, 1], got {self.modality_dropout_prob}"
            )
        if not (0.0 <= self.slow_dropout_prob <= 1.0):
            raise ValueError(
                f"slow_dropout_prob must be in [0, 1], got {self.slow_dropout_prob}"
            )
        if (
            self.modality_dropout_prob > 0.0 or self.slow_dropout_prob > 0.0
        ) and not self.use_no_tactile_token:
            raise ValueError(
                "modality_dropout_prob / slow_dropout_prob > 0 requires "
                "no_tactile_token=true"
            )

        region_cfg = _as_dict(streams.get("region_pressure"))
        taxel_cfg = _as_dict(streams.get("taxel_pressure_3d"))
        self.fast_enable = _as_role_flag(cfg.get("fast_enable", True))
        self.slow_enable = _as_role_flag(cfg.get("slow_enable", True))
        self.region_role = _stream_role("region_pressure", region_cfg)
        self.taxel_role = _stream_role("taxel_pressure_3d", taxel_cfg)
        if bool(region_cfg.get("enable", False)) and self.region_role not in (
            "fast",
            "slow",
        ):
            raise ValueError(
                f"streams.region_pressure.role must be fast|slow, got {self.region_role!r}"
            )
        if bool(taxel_cfg.get("enable", False)) and self.taxel_role not in (
            "fast",
            "slow",
        ):
            raise ValueError(
                f"streams.taxel_pressure_3d.role must be fast|slow, got {self.taxel_role!r}"
            )
        self.region_enabled = stream_is_active(
            "region_pressure",
            region_cfg,
            fast_enable=self.fast_enable,
            slow_enable=self.slow_enable,
            default_enable=False,
        )
        self.taxel_enabled = stream_is_active(
            "taxel_pressure_3d",
            taxel_cfg,
            fast_enable=self.fast_enable,
            slow_enable=self.slow_enable,
            default_enable=False,
        )
        if not self.region_enabled and not self.taxel_enabled:
            raise ValueError(
                "At least one tactile role must be on: set streams.*.enable "
                "and fast_enable/slow_enable so one of {fast, slow} survives "
                "(ablation may turn exactly one off)."
            )

        # Hierarchy roles: fast = patch vectors, slow = 3D temporal cloud.
        # When BOTH streams are on they MUST claim distinct roles — otherwise
        # ``_encode_role_hidden`` would silently drop one stream while
        # ``num_tokens`` still counted both (DiT length mismatch).
        if self.region_enabled and self.taxel_enabled:
            if self.region_role == self.taxel_role:
                raise ValueError(
                    "streams.region_pressure.role and streams.taxel_pressure_3d.role "
                    f"must differ (got both {self.region_role!r}); assign one "
                    "'fast' (patch) and one 'slow' (3D cloud)."
                )
            if {self.region_role, self.taxel_role} != {"fast", "slow"}:
                raise ValueError(
                    "dual-stream hierarchy requires roles {fast, slow}, got "
                    f"{{{self.region_role}, {self.taxel_role}}}"
                )
        hier = _as_dict(cfg.get("hierarchy"))
        self.hierarchy_layout = str(hier.get("layout", "fast_then_slow")).lower()
        if self.hierarchy_layout not in ("fast_then_slow", "slow_then_fast"):
            raise ValueError(
                f"hierarchy.layout must be fast_then_slow|slow_then_fast, "
                f"got {self.hierarchy_layout!r}"
            )
        self.hierarchy_fusion = str(hier.get("fusion", "concat")).lower()
        if self.hierarchy_fusion != "concat":
            raise ValueError(
                f"hierarchy.fusion={self.hierarchy_fusion!r} unsupported; "
                "only concat is implemented (fixed DiT context length)."
            )
        if (
            self.region_enabled
            and self.taxel_enabled
            and int(region_cfg.get("num_hands", 2)) != int(taxel_cfg.get("num_hands", 2))
        ):
            raise ValueError(
                f"streams disagree on num_hands: region_pressure="
                f"{region_cfg.get('num_hands', 2)} vs taxel_pressure_3d="
                f"{taxel_cfg.get('num_hands', 2)}"
            )

        self.region_cfg = region_cfg
        self.taxel_cfg = taxel_cfg
        self._init_region_stream(region_cfg)
        self._init_taxel_stream(taxel_cfg)

        # Token layout follows hierarchy.layout (default fast_then_slow =
        # [patch | 3D]). Stream embeds keep the two branches distinguishable.
        self.num_tokens = self.region_num_tokens + self.taxel_num_tokens
        n_streams = int(self.region_enabled) + int(self.taxel_enabled)
        self.stream_embed = (
            nn.Embedding(n_streams, self.hidden_dim) if n_streams > 1 else None
        )
        # stream_embed index: 0 = fast branch, 1 = slow branch when both on.
        self._fast_stream_idx = 0
        self._slow_stream_idx = 1 if n_streams > 1 else 0
        self.out_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.output_dim),
        )
        if self.use_no_tactile_token:
            self.no_tactile = nn.Parameter(torch.zeros(1, 1, self.output_dim))
            nn.init.normal_(self.no_tactile, std=0.02)
        else:
            self.register_parameter("no_tactile", None)

        # Compatibility fields used by QwenPI_v3 logging / older tests.
        kinds = []
        if self.region_enabled:
            kinds.append(f"region_pressure[{self.region_role}]")
        if self.taxel_enabled:
            kinds.append(f"{self.taxel_encoder_type}[{self.taxel_role}]")
        self.encoder_type = "+".join(kinds)
        self.num_hands = (
            self.region_num_hands if self.region_enabled else self.taxel_num_hands
        )
        self.num_taxels = self.taxel_num_taxels if self.taxel_enabled else 0
        self.point_dim = self.taxel_point_dim if self.taxel_enabled else 0
        self.geometry_carrier = (
            self.taxel_geometry_carrier if self.taxel_enabled else "none"
        )
        self.num_region_tokens = (
            self.taxel_tokens_per_hand if self.taxel_enabled else self.region_num_regions
        )
        self.use_lattice = self.taxel_enabled and self.taxel_encoder_type == "region_lattice_cnn"
        # Slice helpers for hierarchical splice (after layout assemble).
        self.fast_num_tokens = self._tokens_for_role("fast")
        self.slow_num_tokens = self._tokens_for_role("slow")
        self.quality_min_valid_frac = float(expert_cfg.get("quality_min_valid_frac", 0.0))
        self.quality_max_sat_frac = float(expert_cfg.get("quality_max_sat_frac", 1.0))
        self.quality_sat_abs = float(expert_cfg.get("quality_sat_abs", 0.0))
        if not (0.0 <= self.quality_min_valid_frac <= 1.0):
            raise ValueError("quality_min_valid_frac must be in [0, 1]")
        if not (0.0 <= self.quality_max_sat_frac <= 1.0):
            raise ValueError("quality_max_sat_frac must be in [0, 1]")

        logger.info(
            "[TactileEncoder] %s layout=%s tokens=%d (fast=%d slow=%d) → %d "
            "dropout=%.2f slow_dropout=%.2f",
            self.encoder_type,
            self.hierarchy_layout,
            self.num_tokens,
            self.fast_num_tokens,
            self.slow_num_tokens,
            self.output_dim,
            self.modality_dropout_prob,
            self.slow_dropout_prob,
        )

    @property
    def region_token_span(self) -> Tuple[int, int]:
        """``(start, count)`` of the region block in the laid-out sequence.

        Keyed on the region STREAM, not the fast role. Default stems are
        region=fast so the two coincide, but a role swap must still hand
        the trusted force block to ``tactile_patch_gate`` and leave the
        taxel cloud on ``tactile_gate``. Both streams cannot share a role
        (checked at construct), so the block is a prefix or a suffix.
        """
        if not self.region_enabled:
            return 0, 0
        first_role = "fast" if self.hierarchy_layout == "fast_then_slow" else "slow"
        start = 0 if self.region_role == first_role else self._tokens_for_role(first_role)
        return start, self.region_num_tokens

    def _tokens_for_role(self, role: str) -> int:
        n = 0
        if self.region_enabled and self.region_role == role:
            n += self.region_num_tokens
        if self.taxel_enabled and self.taxel_role == role:
            n += self.taxel_num_tokens
        return n

    def _init_region_stream(self, cfg: Dict[str, Any]) -> None:
        if not self.region_enabled:
            self.region_num_hands = 0
            self.region_num_regions = 0
            self.region_history_len = 0
            self.region_channel_dim = 0
            self.region_num_tokens = 0
            self.region_temporal = None
            self.region_hand_embed = None
            self.region_slot_embed = None
            return

        self.region_num_hands = int(cfg.get("num_hands", 2))
        self.region_num_regions = int(cfg.get("num_regions", 5))
        self.region_history_len = int(cfg.get("history_len", 5))
        self.region_channel_dim = int(cfg.get("channel_dim", 1))
        if min(
            self.region_num_hands,
            self.region_num_regions,
            self.region_history_len,
            self.region_channel_dim,
        ) < 1:
            raise ValueError(
                "region_pressure num_hands/num_regions/history_len/channel_dim "
                "must be >= 1"
            )
        self.region_num_tokens = self.region_num_hands * self.region_num_regions
        self.region_temporal = TemporalConvEncoder(
            self.region_channel_dim,
            self.hidden_dim,
            num_layers=self.temporal_layers,
            kernel_size=self.temporal_kernel,
        )
        self.region_hand_embed = nn.Embedding(self.region_num_hands, self.hidden_dim)
        self.region_slot_embed = nn.Embedding(self.region_num_regions, self.hidden_dim)

    def _init_taxel_stream(self, cfg: Dict[str, Any]) -> None:
        if not self.taxel_enabled:
            self.taxel_num_hands = 0
            self.taxel_num_taxels = 0
            self.taxel_history_len = 0
            self.taxel_point_dim = 0
            self.taxel_num_tokens = 0
            self.taxel_tokens_per_hand = 0
            self.taxel_encoder_type = "none"
            self.taxel_geometry_carrier = "none"
            self.taxel_backbone = None
            self.taxel_temporal = None
            self.taxel_temporal_layers = 0
            self.taxel_hand_embed = None
            self.taxel_slot_embed = None
            self.region_map = None
            self.keep_zero_pressure_points = True
            self.pool_mode = "softmax"
            self.trim_frac = 0.1
            self.drop_stuck_taxels = False
            self.stuck_abs = 0.0
            self.taxel_token_mixer = None
            return

        self.taxel_num_hands = int(cfg.get("num_hands", 2))
        self.taxel_num_taxels = int(cfg.get("num_taxels", 320))
        self.taxel_history_len = int(cfg.get("history_len", 1))
        self.taxel_point_dim = int(cfg.get("point_dim", 4))
        self.taxel_xyz_frame = str(cfg.get("xyz_frame", "head_camera"))
        self.taxel_geometry_carrier = str(cfg.get("geometry_carrier", "point_set"))
        self.taxel_encoder_type = str(cfg.get("type", "region_set_encoder"))
        self.taxel_tokens_per_hand = int(cfg.get("num_region_tokens", 8))
        self.num_point_tokens = int(cfg.get("num_point_tokens", 0))
        self.use_fourier_features = bool(cfg.get("use_fourier_features", True))
        self.fourier_bands = int(cfg.get("fourier_bands", 4))
        self.keep_zero_pressure_points = bool(
            cfg.get("keep_zero_pressure_points", True)
        )
        self.pool_mode = str(cfg.get("pool_mode", "softmax")).lower()
        self.trim_frac = float(cfg.get("trim_frac", 0.1))
        self.drop_stuck_taxels = bool(cfg.get("drop_stuck_taxels", False))
        self.stuck_std = float(cfg.get("stuck_std", 1.0e-4))
        self.stuck_mean = float(cfg.get("stuck_mean", 0.05))
        self.stuck_abs = float(cfg.get("stuck_abs", 0.0))
        self.token_mixer_layers = int(cfg.get("token_mixer_layers", 0))
        if min(
            self.taxel_num_hands,
            self.taxel_num_taxels,
            self.taxel_history_len,
            self.taxel_point_dim,
        ) < 1:
            raise ValueError(
                "taxel_pressure_3d num_hands/num_taxels/history_len/point_dim "
                "must be >= 1"
            )
        if self.taxel_geometry_carrier not in ("point_set", "uv_lattice"):
            raise ValueError(
                f"geometry_carrier={self.taxel_geometry_carrier!r}; "
                "expected point_set | uv_lattice"
            )

        region_map_cfg = cfg.get("region_map")
        if isinstance(region_map_cfg, str) and region_map_cfg.lower() in {
            "uniform",
            "contiguous",
            "blocks",
        }:
            region_map = _uniform_region_map(
                self.taxel_num_taxels, self.taxel_tokens_per_hand
            )
        else:
            region_map = _load_region_map(region_map_cfg, self.taxel_num_taxels)
        if region_map is not None:
            n_regions = int(region_map.max().item()) + 1
            if self.taxel_tokens_per_hand < n_regions:
                raise ValueError(
                    f"num_region_tokens={self.taxel_tokens_per_hand} < "
                    f"region_map regions ({n_regions})"
                )
            self.register_buffer("region_map", region_map, persistent=True)
        else:
            self.region_map = None

        self.taxel_backbone, self.taxel_tokens_per_hand = self._build_taxel_backbone()
        self.taxel_num_tokens = self.taxel_num_hands * self.taxel_tokens_per_hand

        # Temporal pressure → hidden-space per-point features, fused into the
        # backbone's point MLP output (NOT squeezed back to point_dim, which
        # would corrupt the Fourier geometry encoding). history_len==1 skips
        # this path entirely so legacy single-frame configs stay bit-identical.
        # Default 1 layer: a short pressure summary. Inheriting
        # expert.temporal_layers (3 on EgoPro stems) would twin the trusted
        # region encoder onto the untrusted cloud (~1.6M extra params).
        self.taxel_temporal_layers = int(cfg.get("temporal_layers", 1))
        if self.taxel_temporal_layers < 1:
            raise ValueError(
                "taxel_pressure_3d.temporal_layers must be >= 1, "
                f"got {self.taxel_temporal_layers}"
            )
        is_lattice = self.taxel_encoder_type == "region_lattice_cnn"
        if self.taxel_history_len > 1 and not is_lattice:
            self.taxel_temporal = TemporalConvEncoder(
                1,
                self.hidden_dim,
                num_layers=self.taxel_temporal_layers,
                kernel_size=self.temporal_kernel,
            )
        else:
            self.taxel_temporal = None
        self.taxel_hand_embed = nn.Embedding(self.taxel_num_hands, self.hidden_dim)
        self.taxel_slot_embed = nn.Embedding(self.taxel_tokens_per_hand, self.hidden_dim)
        if self.token_mixer_layers > 0:
            nhead = 8
            while nhead > 1 and self.hidden_dim % nhead != 0:
                nhead //= 2
            mixer_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=nhead,
                dim_feedforward=self.hidden_dim * 2,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.taxel_token_mixer = nn.TransformerEncoder(
                mixer_layer, num_layers=self.token_mixer_layers
            )
        else:
            self.taxel_token_mixer = None

    def _build_taxel_backbone(self) -> Tuple[nn.Module, int]:
        n_tokens = self.taxel_tokens_per_hand
        pdim = self.taxel_point_dim
        if self.taxel_encoder_type == "region_set_encoder":
            backbone = RegionSetEncoder(
                point_dim=pdim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                num_region_tokens=n_tokens,
                use_fourier_features=self.use_fourier_features,
                fourier_bands=self.fourier_bands,
                pool_mode=self.pool_mode,
                trim_frac=self.trim_frac,
            )
        elif self.taxel_encoder_type == "region_lattice_cnn":
            if self.taxel_geometry_carrier != "uv_lattice":
                raise ValueError(
                    "region_lattice_cnn requires geometry_carrier=uv_lattice"
                )
            backbone = RegionLatticeCNN(
                in_channels=pdim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                num_region_tokens=n_tokens,
            )
        elif self.taxel_encoder_type == "point_transformer":
            if self.num_point_tokens < 1:
                raise ValueError(
                    "point_transformer requires taxel_pressure_3d.num_point_tokens > 0"
                )
            n_tokens = self.num_point_tokens
            backbone = RegionSetEncoder(
                point_dim=pdim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                num_region_tokens=n_tokens,
                use_fourier_features=self.use_fourier_features,
                fourier_bands=self.fourier_bands,
                pool_mode=self.pool_mode,
                trim_frac=self.trim_frac,
            )
        else:
            raise ValueError(
                f"Unknown taxel type={self.taxel_encoder_type!r}; expected "
                "region_set_encoder | region_lattice_cnn | point_transformer"
            )
        if n_tokens < 1:
            raise ValueError("num_region_tokens / num_point_tokens must be >= 1")
        return backbone, n_tokens

    @property
    def patch_cond_dim(self) -> int:
        """Flattened region window for DiT AdaLN ``temb`` (0 if region off)."""
        if not self.region_enabled:
            return 0
        return (
            self.region_num_hands
            * self.region_num_regions
            * self.region_history_len
            * self.region_channel_dim
        )

    def flatten_patch_cond(
        self,
        region_pressure: Optional[torch.Tensor],
        *,
        present_fast: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Optional[torch.Tensor]:
        """Raw ``region_pressure`` → ``[B, patch_cond_dim]``, or None if off.

        Uses the same window the encoder was built for. Prefer the
        post-dropout region-stream present mask (``region_present_mask``)
        so a dropped row cannot FiLM the DiT.
        """
        if not self.region_enabled:
            return None
        if present_fast is None:
            present_fast = self.region_present_mask()
        if batch_size is None:
            if region_pressure is not None:
                batch_size = int(region_pressure.shape[0])
            elif present_fast is not None:
                batch_size = int(present_fast.shape[0])
        return flatten_region_patch(
            region_pressure,
            num_hands=self.region_num_hands,
            num_regions=self.region_num_regions,
            history_len=self.region_history_len,
            channel_dim=self.region_channel_dim,
            present=present_fast,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

    def region_present_mask(self) -> Optional[torch.Tensor]:
        """Post-dropout present flag for the *region* stream (not the fast role).

        Packing's ``present_fast`` is region-present; the encoder's
        ``present_fast`` is the fast *role*. They match on the default
        ``region=fast`` stems; this helper stays correct if the roles swap.
        """
        if not self.region_enabled:
            return None
        if self.region_role == "fast":
            return getattr(self, "_last_present_fast", None)
        return getattr(self, "_last_present_slow", None)

    def taxel_present_mask(self) -> Optional[torch.Tensor]:
        """Post-dropout, post-quality-gate present flag for the *3D* stream.

        Mirror of ``region_present_mask`` for ``taxel_pressure_3d``, keyed on
        the STREAM rather than the slow role so a role swap stays correct.
        The fusion gates read this so a row whose cloud was dropped (slow
        dropout) or rejected (``quality_*``) does not vote on ``tactile_gate``.
        """
        if not self.taxel_enabled:
            return None
        if self.taxel_role == "slow":
            return getattr(self, "_last_present_slow", None)
        return getattr(self, "_last_present_fast", None)

    @property
    def expected_region_pressure_shape(
        self,
    ) -> Optional[Tuple[Optional[int], int, int, int, int]]:
        if not self.region_enabled:
            return None
        return (
            None,
            self.region_num_hands,
            self.region_num_regions,
            self.region_history_len,
            self.region_channel_dim,
        )

    @property
    def expected_taxel_pressure_3d_shape(
        self,
    ) -> Optional[Tuple[Optional[int], int, int, int, int]]:
        if not self.taxel_enabled or self.use_lattice:
            return None
        return (
            None,
            self.taxel_num_hands,
            self.taxel_num_taxels,
            self.taxel_history_len,
            self.taxel_point_dim,
        )

    @property
    def expected_points_shape(self) -> Tuple[Optional[int], int, int, int]:
        """Legacy single-frame taxel shape ``[B,H,N,C]`` (history_len must be 1)."""
        if not self.taxel_enabled or self.use_lattice:
            return (None, 0, 0, 0)
        return (None, self.taxel_num_hands, self.taxel_num_taxels, self.taxel_point_dim)

    def assert_encode_kwargs(self, kwargs: dict) -> None:
        """Check packed ``encode()`` kwargs against this encoder's layout.

        Called from the training / infer pack path so an 80-vs-320 clash
        dies with the expected ``[B, H, N, T, C]`` before the first Linear,
        not as a mid-step CUDA error. Missing tensors are legal (those
        rows become ``no_tactile``); only a present tensor with the wrong
        tail is an error. ``encode`` still re-checks via ``_validate_inputs``.
        """
        if not isinstance(kwargs, dict):
            raise TypeError(f"encode kwargs must be a dict, got {type(kwargs)}")
        rp = kwargs.get("region_pressure")
        if rp is not None and self.region_enabled:
            expected = self.expected_region_pressure_shape
            _require_tensor(
                "region_pressure", rp, ndim=5, tail=expected[1:]
            )
        tp = kwargs.get("taxel_pressure_3d")
        if tp is not None and self.taxel_enabled and not self.use_lattice:
            expected = self.expected_taxel_pressure_3d_shape
            _require_tensor(
                "taxel_pressure_3d", tp, ndim=5, tail=expected[1:]
            )
        mask = kwargs.get("taxel_valid_mask")
        if mask is not None and self.taxel_enabled and not self.use_lattice:
            if mask.ndim == 3:
                _require_tensor(
                    "taxel_valid_mask",
                    mask,
                    ndim=3,
                    tail=(self.taxel_num_hands, self.taxel_num_taxels),
                )
            else:
                _require_tensor(
                    "taxel_valid_mask",
                    mask,
                    ndim=4,
                    tail=(
                        self.taxel_num_hands,
                        self.taxel_num_taxels,
                        self.taxel_history_len,
                    ),
                )
        tensors = [t for t in (rp, tp) if t is not None]
        if (
            len(tensors) == 2
            and int(tensors[0].shape[0]) != int(tensors[1].shape[0])
        ):
            raise ValueError(
                f"region_pressure B={tensors[0].shape[0]} != "
                f"taxel_pressure_3d B={tensors[1].shape[0]}"
            )

    def assert_future_contact_tensors(
        self,
        future: Optional[torch.Tensor],
        future_valid: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
    ) -> None:
        """Check variant-A future-window tensors against the region layout."""
        if not self.region_enabled:
            if future is not None:
                raise ValueError(
                    "region_pressure_future was packed but the region stream "
                    "is off; future_contact needs streams.region_pressure"
                )
            return
        if history is not None:
            expected = self.expected_region_pressure_shape
            _require_tensor("region_pressure", history, ndim=5, tail=expected[1:])
        if future is None:
            return
        if not isinstance(future, torch.Tensor):
            raise TypeError(
                f"region_pressure_future must be a Tensor, got {type(future)}"
            )
        if future.ndim != 5:
            raise ValueError(
                f"region_pressure_future must be [B, H, R, horizon, 1], "
                f"got {tuple(future.shape)}"
            )
        got = tuple(int(x) for x in future.shape)
        if (
            got[1] != self.region_num_hands
            or got[2] != self.region_num_regions
            or got[4] != 1
        ):
            raise ValueError(
                f"region_pressure_future {got} != "
                f"[B, {self.region_num_hands}, {self.region_num_regions}, "
                "horizon, 1]"
            )
        if future_valid is not None:
            if not isinstance(future_valid, torch.Tensor):
                raise TypeError(
                    f"region_pressure_future_valid must be a Tensor, "
                    f"got {type(future_valid)}"
                )
            exp_valid = (got[0], got[3])
            if tuple(int(x) for x in future_valid.shape) != exp_valid:
                raise ValueError(
                    f"region_pressure_future_valid {tuple(future_valid.shape)} "
                    f"!= {exp_valid}"
                )

    def encode_fast(
        self,
        region_pressure: Optional[torch.Tensor] = None,
        *,
        present: Optional[torch.Tensor] = None,
        cached_slow_tokens: Optional[torch.Tensor] = None,
        cached_slow_mask: Optional[torch.Tensor] = None,
        apply_modality_dropout: Optional[bool] = None,
        batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fast path: patch / region-pressure vectors (+ cached slow splice)."""
        return self.encode(
            region_pressure=region_pressure,
            present=present,
            stream_mode="fast",
            cached_slow_tokens=cached_slow_tokens,
            cached_slow_mask=cached_slow_mask,
            apply_modality_dropout=apply_modality_dropout,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

    def encode_slow(
        self,
        taxel_pressure_3d: Optional[torch.Tensor] = None,
        taxel_valid_mask: Optional[torch.Tensor] = None,
        *,
        points: Optional[torch.Tensor] = None,
        lattice: Optional[torch.Tensor] = None,
        region_valid: Optional[torch.Tensor] = None,
        present: Optional[torch.Tensor] = None,
        apply_modality_dropout: Optional[bool] = None,
        batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Slow path: 3D temporal taxel cloud (fast slots → no_tactile)."""
        return self.encode(
            taxel_pressure_3d=taxel_pressure_3d,
            taxel_valid_mask=taxel_valid_mask,
            points=points,
            lattice=lattice,
            region_valid=region_valid,
            present=present,
            stream_mode="slow",
            apply_modality_dropout=apply_modality_dropout,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

    def encode(
        self,
        region_pressure: Optional[torch.Tensor] = None,
        taxel_pressure_3d: Optional[torch.Tensor] = None,
        taxel_valid_mask: Optional[torch.Tensor] = None,
        present: Optional[torch.Tensor] = None,
        *,
        stream_mode: str = "both",
        present_fast: Optional[torch.Tensor] = None,
        present_slow: Optional[torch.Tensor] = None,
        cached_slow_tokens: Optional[torch.Tensor] = None,
        cached_slow_mask: Optional[torch.Tensor] = None,
        # Legacy aliases -------------------------------------------------
        points: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        lattice: Optional[torch.Tensor] = None,
        region_valid: Optional[torch.Tensor] = None,
        apply_modality_dropout: Optional[bool] = None,
        batch_size: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(tokens [B,T,D], token_mask [B,T])``.

        Preferred kwargs: ``region_pressure`` (fast), ``taxel_pressure_3d`` (slow).
        ``stream_mode``: ``both`` | ``slow`` | ``fast`` (see module docstring).
        Legacy: ``points`` (``[B,H,N,C]`` → taxel with T=1) or ``lattice``.
        """
        mode = str(stream_mode).lower()
        if mode not in ("both", "slow", "fast"):
            raise ValueError(
                f"stream_mode must be both|slow|fast, got {stream_mode!r}"
            )

        if points is not None:
            if taxel_pressure_3d is not None:
                raise ValueError("pass only one of taxel_pressure_3d= or points=")
            taxel_pressure_3d = self._points_to_taxel(points)
        if valid_mask is not None:
            if taxel_valid_mask is not None:
                raise ValueError("pass only one of taxel_valid_mask= or valid_mask=")
            taxel_valid_mask = valid_mask

        # Resolve B/device/dtype from ANY provided tensor first (including
        # tensors for roles we may later skip), so present_* can be checked
        # before we decide which streams are live.
        ref = None
        for candidate in (
            region_pressure,
            taxel_pressure_3d,
            lattice,
            cached_slow_tokens,
        ):
            if candidate is not None:
                ref = candidate
                break
        if ref is not None:
            B, device, dtype = int(ref.shape[0]), ref.device, ref.dtype
        else:
            if batch_size is None:
                raise ValueError(
                    "batch_size required when region_pressure / "
                    "taxel_pressure_3d / lattice / cached_slow are all None"
                )
            B = int(batch_size)
            device = device or self.out_proj[1].weight.device
            dtype = dtype or self.out_proj[1].weight.dtype

        if present_fast is not None:
            present_fast = present_fast.to(device=device, dtype=torch.bool)
            if present_fast.shape != (B,):
                raise ValueError(
                    f"present_fast must be [B]=({B},), got {tuple(present_fast.shape)}"
                )
        if present_slow is not None:
            present_slow = present_slow.to(device=device, dtype=torch.bool)
            if present_slow.shape != (B,):
                raise ValueError(
                    f"present_slow must be [B]=({B},), got {tuple(present_slow.shape)}"
                )

        if present is None:
            if present_fast is not None or present_slow is not None:
                pf = present_fast if present_fast is not None else torch.zeros(
                    B, dtype=torch.bool, device=device
                )
                ps = present_slow if present_slow is not None else torch.zeros(
                    B, dtype=torch.bool, device=device
                )
                present = pf | ps
            else:
                # Infer presence from tensors that this mode could consume.
                has_fast_input = (
                    (self.region_enabled and self.region_role == "fast" and region_pressure is not None)
                    or (self.taxel_enabled and self.taxel_role == "fast" and (
                        taxel_pressure_3d is not None or lattice is not None
                    ))
                )
                has_slow_input = (
                    (self.region_enabled and self.region_role == "slow" and region_pressure is not None)
                    or (self.taxel_enabled and self.taxel_role == "slow" and (
                        taxel_pressure_3d is not None or lattice is not None
                    ))
                )
                if mode == "fast":
                    inferred = has_fast_input or cached_slow_tokens is not None
                elif mode == "slow":
                    inferred = has_slow_input
                else:
                    inferred = has_fast_input or has_slow_input
                present = torch.full((B,), inferred, dtype=torch.bool, device=device)
        else:
            present = present.to(device=device, dtype=torch.bool)
            if present.shape != (B,):
                raise ValueError(
                    f"present must be [B]=({B},), got {tuple(present.shape)}"
                )

        # Role flags are independent. Legacy ``present=`` (or inferred presence)
        # fills BOTH when neither role flag was passed. If only one role flag
        # is explicit, the omitted role defaults to False — never to the OR'd
        # ``present`` (that previously made ``present_slow=…`` alone also gate
        # the fast stream).
        explicit_fast = present_fast is not None
        explicit_slow = present_slow is not None
        if present_fast is None and present_slow is None:
            present_fast = present
            present_slow = present
        elif present_fast is None:
            present_fast = torch.zeros(B, dtype=torch.bool, device=device)
        elif present_slow is None:
            present_slow = torch.zeros(B, dtype=torch.bool, device=device)
        # Reconcile batch-level present with the resolved role masks when the
        # caller only supplied per-role flags (present was pf|ps already).
        if explicit_fast or explicit_slow:
            present = present_fast | present_slow

        if apply_modality_dropout is None:
            apply_modality_dropout = bool(self.training)
        if (
            apply_modality_dropout
            and (self.modality_dropout_prob > 0.0 or self.slow_dropout_prob > 0.0)
            and bool(present.any())
        ):
            # Independent coins: a shared keep used to drop both streams
            # together and almost never trained "region force without 3D".
            if self.modality_dropout_prob > 0.0:
                keep_fast = torch.rand(B, device=device) >= self.modality_dropout_prob
                present_fast = present_fast & keep_fast
            if self.slow_dropout_prob > 0.0:
                keep_slow = torch.rand(B, device=device) >= self.slow_dropout_prob
                present_slow = present_slow & keep_slow
            present = present_fast | present_slow

        present_slow = self._apply_quality_gate(
            present_slow, taxel_pressure_3d, taxel_valid_mask
        )
        present = present_fast | present_slow

        # Live-encode every role this stream_mode owns — do not gate on
        # ``present_*.any()``. ZeRO-3 all-gathers on module.forward; a
        # V-only rank that skipped region/taxel/out_proj while another
        # encoded gloves would deadlock. Missing tensors become zeros;
        # ``_apply_role_present`` / ``_apply_present`` still replace
        # absent rows with ``no_tactile`` (output matches the old
        # all-absent path). Infer ``mode=fast`` still does not live-encode
        # slow (cached / placeholder slots).
        need_fast = mode in ("both", "fast") and self.fast_num_tokens > 0
        need_slow_live = mode in ("both", "slow") and self.slow_num_tokens > 0

        rp = region_pressure
        tp = taxel_pressure_3d
        tv = taxel_valid_mask
        lat = lattice
        rv = region_valid
        # Drop tensors for roles this stream_mode does not encode so
        # shape checks don't spuriously require them.
        if not need_fast:
            if self.region_role == "fast":
                rp = None
            if self.taxel_role == "fast":
                tp, tv, lat, rv = None, None, None, None
        if not need_slow_live:
            if self.region_role == "slow":
                rp = None
            if self.taxel_role == "slow":
                tp, tv, lat, rv = None, None, None, None

        # Reject malformed packed tensors even when every row is absent.
        self._validate_inputs(rp, tp, tv, lat, rv)
        rp, tp, tv, lat, rv = self._fill_missing_stream_tensors(
            rp,
            tp,
            tv,
            lat,
            rv,
            need_fast=need_fast,
            need_slow_live=need_slow_live,
            B=B,
            device=device,
            dtype=dtype,
        )

        # Encode live streams in hidden space, project, then splice any
        # cached / placeholder slots in output space (fixed DiT length).
        fast_h = fast_msk = None
        slow_h = slow_msk = None

        if need_fast:
            fast_h, fast_msk = self._encode_role_hidden(
                "fast", rp, tp, tv, lat, rv
            )
        if need_slow_live:
            slow_h, slow_msk = self._encode_role_hidden(
                "slow", rp, tp, tv, lat, rv
            )

        # Project whatever we encoded; missing roles filled after proj.
        if fast_h is not None:
            fast_out = self.out_proj(fast_h)
        else:
            fast_out = None
        if slow_h is not None:
            slow_out = self.out_proj(slow_h)
        else:
            slow_out = None

        if self.fast_num_tokens > 0 and fast_out is None:
            fast_out, fast_msk = self._no_tactile_slots(
                B, self.fast_num_tokens, device=device, dtype=dtype
            )
        if self.slow_num_tokens > 0 and slow_out is None:
            if mode == "fast" and cached_slow_tokens is not None:
                slow_out, slow_msk = self._splice_cached_slow(
                    cached_slow_tokens,
                    cached_slow_mask,
                    B=B,
                    device=device,
                    dtype=dtype,
                )
            else:
                slow_out, slow_msk = self._no_tactile_slots(
                    B, self.slow_num_tokens, device=device, dtype=dtype
                )

        tokens, token_mask = self._assemble_hierarchy(
            fast_out, fast_msk, slow_out, slow_msk
        )
        tokens, token_mask = self._apply_role_present(
            tokens, token_mask, present_fast, present_slow
        )
        tokens, token_mask = self._apply_present(tokens, token_mask, present)
        if tokens.shape != (B, self.num_tokens, self.output_dim):
            raise RuntimeError(
                f"TactileEncoder.encode produced {tuple(tokens.shape)}, "
                f"expected [{B}, {self.num_tokens}, {self.output_dim}]"
            )
        if token_mask.shape != (B, self.num_tokens):
            raise RuntimeError(
                f"TactileEncoder.encode mask {tuple(token_mask.shape)} != "
                f"[{B}, {self.num_tokens}]"
            )
        # Post-dropout / quality flags for P1 flatten (train must not FiLM
        # a row encode() just replaced with no_tactile).
        self._last_present_fast = present_fast
        self._last_present_slow = present_slow
        return tokens, token_mask

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encode(*args, **kwargs)

    def _add_stream_embed(self, tok: torch.Tensor, idx: int) -> torch.Tensor:
        if self.stream_embed is None:
            return tok
        return tok + self.stream_embed.weight[idx].to(dtype=tok.dtype)

    def _encode_role_hidden(
        self,
        role: str,
        region_pressure: Optional[torch.Tensor],
        taxel_pressure_3d: Optional[torch.Tensor],
        taxel_valid_mask: Optional[torch.Tensor],
        lattice: Optional[torch.Tensor],
        region_valid: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode the stream assigned to ``role`` → hidden tokens + mask."""
        stream_idx = (
            self._fast_stream_idx if role == "fast" else self._slow_stream_idx
        )
        if self.region_enabled and self.region_role == role:
            tok, msk = self._encode_region(region_pressure)
            return self._add_stream_embed(tok, stream_idx), msk
        if self.taxel_enabled and self.taxel_role == role:
            if self.use_lattice:
                tok, msk = self._encode_lattice(lattice, region_valid)
            else:
                tok, msk = self._encode_taxel(taxel_pressure_3d, taxel_valid_mask)
            return self._add_stream_embed(tok, stream_idx), msk
        raise RuntimeError(f"no stream assigned to role={role!r}")

    def _fill_missing_stream_tensors(
        self,
        region_pressure: Optional[torch.Tensor],
        taxel_pressure_3d: Optional[torch.Tensor],
        taxel_valid_mask: Optional[torch.Tensor],
        lattice: Optional[torch.Tensor],
        region_valid: Optional[torch.Tensor],
        *,
        need_fast: bool,
        need_slow_live: bool,
        B: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Zeros for a needed stream the caller omitted (V-only / dummy encode)."""

        def _need_role(role: str) -> bool:
            if role == "fast":
                return need_fast
            if role == "slow":
                return need_slow_live
            return False

        if (
            self.region_enabled
            and region_pressure is None
            and _need_role(self.region_role)
        ):
            region_pressure = torch.zeros(
                (
                    B,
                    self.region_num_hands,
                    self.region_num_regions,
                    self.region_history_len,
                    self.region_channel_dim,
                ),
                device=device,
                dtype=dtype,
            )
        if self.taxel_enabled and _need_role(self.taxel_role):
            if self.use_lattice:
                if lattice is None:
                    lattice = torch.zeros(
                        (
                            B,
                            self.taxel_num_hands,
                            self.taxel_tokens_per_hand,
                            self.taxel_point_dim,
                            1,
                            1,
                        ),
                        device=device,
                        dtype=dtype,
                    )
            elif taxel_pressure_3d is None:
                taxel_pressure_3d = torch.zeros(
                    (
                        B,
                        self.taxel_num_hands,
                        self.taxel_num_taxels,
                        self.taxel_history_len,
                        self.taxel_point_dim,
                    ),
                    device=device,
                    dtype=dtype,
                )
        return region_pressure, taxel_pressure_3d, taxel_valid_mask, lattice, region_valid

    def _no_tactile_slots(
        self,
        B: int,
        n_tokens: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.no_tactile is None:
            raise RuntimeError(
                "Hierarchical placeholder slots require no_tactile_token=true"
            )
        tok = self.no_tactile.to(device=device, dtype=dtype).expand(
            B, n_tokens, self.output_dim
        )
        # Touch params so DDP sees the same set when a role is skipped.
        touch = tok.new_zeros(())
        for param in self.parameters():
            if param.requires_grad:
                touch = touch + param.to(dtype=dtype).sum() * 0
        tok = tok + touch
        msk = torch.ones(B, n_tokens, dtype=torch.bool, device=device)
        return tok, msk

    def _splice_cached_slow(
        self,
        cached_slow_tokens: torch.Tensor,
        cached_slow_mask: Optional[torch.Tensor],
        *,
        B: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Accept full ``[B,T,D]`` cache or slow-only ``[B,S,D]`` slice."""
        if not isinstance(cached_slow_tokens, torch.Tensor):
            raise TypeError("cached_slow_tokens must be a Tensor")
        cached_slow_tokens = cached_slow_tokens.to(device=device, dtype=dtype)
        if cached_slow_tokens.ndim != 3:
            raise ValueError(
                f"cached_slow_tokens must be [B,T,D], got {tuple(cached_slow_tokens.shape)}"
            )
        if cached_slow_tokens.shape[0] != B:
            raise ValueError(
                f"cached_slow_tokens B={cached_slow_tokens.shape[0]} != batch {B}"
            )
        if cached_slow_tokens.shape[-1] != self.output_dim:
            raise ValueError(
                f"cached_slow_tokens D={cached_slow_tokens.shape[-1]} != "
                f"output_dim={self.output_dim} (pass post-proj tokens)"
            )
        T = cached_slow_tokens.shape[1]
        if T == self.num_tokens:
            slow = self._slice_role(cached_slow_tokens, "slow")
            if cached_slow_mask is not None:
                msk = self._slice_role(
                    cached_slow_mask.to(device=device, dtype=torch.bool), "slow"
                )
            else:
                msk = torch.ones(
                    B, self.slow_num_tokens, dtype=torch.bool, device=device
                )
            return slow, msk
        if T == self.slow_num_tokens:
            if cached_slow_mask is None:
                msk = torch.ones(B, T, dtype=torch.bool, device=device)
            else:
                msk = cached_slow_mask.to(device=device, dtype=torch.bool)
                if msk.shape != (B, T):
                    raise ValueError(
                        f"cached_slow_mask shape {tuple(msk.shape)} != {(B, T)}"
                    )
            return cached_slow_tokens, msk
        raise ValueError(
            f"cached_slow_tokens T={T} must be num_tokens={self.num_tokens} "
            f"or slow_num_tokens={self.slow_num_tokens}"
        )

    def _assemble_hierarchy(
        self,
        fast_tok: Optional[torch.Tensor],
        fast_msk: Optional[torch.Tensor],
        slow_tok: Optional[torch.Tensor],
        slow_msk: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.hierarchy_layout == "fast_then_slow":
            order = [(fast_tok, fast_msk), (slow_tok, slow_msk)]
        else:
            order = [(slow_tok, slow_msk), (fast_tok, fast_msk)]
        parts_t: List[torch.Tensor] = []
        parts_m: List[torch.Tensor] = []
        for tok, msk in order:
            if tok is None:
                continue
            parts_t.append(tok)
            parts_m.append(msk)
        if not parts_t:
            raise RuntimeError("hierarchy assemble received no stream tensors")
        return torch.cat(parts_t, dim=1), torch.cat(parts_m, dim=1)

    def _slice_role(self, tokens: torch.Tensor, role: str) -> torch.Tensor:
        """Slice fast/slow block out of a full laid-out sequence."""
        n_fast, n_slow = self.fast_num_tokens, self.slow_num_tokens
        if self.hierarchy_layout == "fast_then_slow":
            if role == "fast":
                return tokens[:, :n_fast]
            return tokens[:, n_fast : n_fast + n_slow]
        if role == "slow":
            return tokens[:, :n_slow]
        return tokens[:, n_slow : n_slow + n_fast]

    def _role_slot_present(
        self,
        B: int,
        T: int,
        present_fast: torch.Tensor,
        present_slow: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """``[B, T]`` — True where that role slot's row is present."""
        m = torch.zeros(B, T, dtype=torch.bool, device=device)
        n_fast, n_slow = self.fast_num_tokens, self.slow_num_tokens
        if self.hierarchy_layout == "fast_then_slow":
            if n_fast > 0:
                m[:, :n_fast] = present_fast.view(B, 1)
            if n_slow > 0:
                m[:, n_fast : n_fast + n_slow] = present_slow.view(B, 1)
        else:
            if n_slow > 0:
                m[:, :n_slow] = present_slow.view(B, 1)
            if n_fast > 0:
                m[:, n_slow : n_slow + n_fast] = present_fast.view(B, 1)
        return m

    def _apply_role_present(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        present_fast: torch.Tensor,
        present_slow: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Replace absent role slots with ``no_tactile`` (keeps length fixed)."""
        if bool(present_fast.all()) and bool(present_slow.all()):
            return tokens, token_mask
        B, T, D = tokens.shape
        slot_ok = self._role_slot_present(
            B, T, present_fast, present_slow, tokens.device
        )
        if self.no_tactile is None:
            return tokens * slot_ok.unsqueeze(-1).to(tokens.dtype), token_mask & slot_ok
        filler = self.no_tactile.to(dtype=tokens.dtype).expand(B, T, D)
        tokens = torch.where(slot_ok.unsqueeze(-1), tokens, filler)
        # Absent-role slots stay attendable via the learned placeholder.
        return tokens, token_mask | (~slot_ok)

    def _points_to_taxel(self, points: torch.Tensor) -> torch.Tensor:
        if not self.taxel_enabled or self.use_lattice:
            raise ValueError("points= requires an enabled point_set taxel stream")
        if not isinstance(points, torch.Tensor):
            raise TypeError(f"points must be a Tensor, got {type(points)}")
        if points.ndim == 4:
            # [B, H, N, C] → [B, H, N, 1, C]
            if self.taxel_history_len != 1:
                raise ValueError(
                    f"points [B,H,N,C] only valid when taxel history_len=1 "
                    f"(got history_len={self.taxel_history_len}); pass "
                    f"taxel_pressure_3d [B,H,N,T,C] instead"
                )
            return points.unsqueeze(3)
        if points.ndim == 5:
            return points
        raise ValueError(
            f"points must be [B,H,N,C] or [B,H,N,T,C], got shape {tuple(points.shape)}"
        )

    def _validate_inputs(
        self,
        region_pressure: Optional[torch.Tensor],
        taxel_pressure_3d: Optional[torch.Tensor],
        taxel_valid_mask: Optional[torch.Tensor],
        lattice: Optional[torch.Tensor],
        region_valid: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        ref: Optional[torch.Tensor] = None

        if self.region_enabled:
            if region_pressure is not None:
                region_pressure = _require_tensor(
                    "region_pressure",
                    region_pressure,
                    ndim=5,
                    tail=(
                        self.region_num_hands,
                        self.region_num_regions,
                        self.region_history_len,
                        self.region_channel_dim,
                    ),
                )
                ref = region_pressure
        elif region_pressure is not None:
            # Ablation / leftover column: ignore, do not require the stream.
            region_pressure = None

        # Cross-stream batch consistency (B must agree when both are given).
        if (
            ref is not None
            and taxel_pressure_3d is not None
            and int(taxel_pressure_3d.shape[0]) != int(ref.shape[0])
        ):
            raise ValueError(
                f"region_pressure B={ref.shape[0]} != "
                f"taxel_pressure_3d B={taxel_pressure_3d.shape[0]}"
            )
        if (
            ref is not None
            and lattice is not None
            and int(lattice.shape[0]) != int(ref.shape[0])
        ):
            raise ValueError(
                f"region_pressure B={ref.shape[0]} != lattice B={lattice.shape[0]}"
            )

        if self.taxel_enabled and self.use_lattice:
            if taxel_pressure_3d is not None:
                raise ValueError("region_lattice_cnn expects lattice=, not taxel_pressure_3d=")
            if lattice is None:
                return ref
            R = self.taxel_tokens_per_hand
            if not isinstance(lattice, torch.Tensor):
                raise TypeError(f"lattice must be a Tensor, got {type(lattice)}")
            if lattice.ndim != 6:
                raise ValueError(
                    f"lattice must be [B, {self.taxel_num_hands}, {R}, "
                    f"{self.taxel_point_dim}, Hr, Wr], got shape {tuple(lattice.shape)}"
                )
            B, H, R_in, C, Hr, Wr = lattice.shape
            if (
                (H, R_in, C) != (self.taxel_num_hands, R, self.taxel_point_dim)
                or Hr < 1
                or Wr < 1
            ):
                raise ValueError(
                    f"lattice shape {tuple(lattice.shape)} != "
                    f"[B, {self.taxel_num_hands}, {R}, {self.taxel_point_dim}, "
                    "Hr>=1, Wr>=1]"
                )
            if region_valid is not None:
                _require_tensor(
                    "region_valid", region_valid, ndim=3, tail=(H, R)
                )
                if region_valid.shape[0] != B:
                    raise ValueError(
                        f"region_valid B={region_valid.shape[0]} != lattice B={B}"
                    )
            return lattice if ref is None else ref

        if lattice is not None:
            raise ValueError(
                f"{self.taxel_encoder_type} expects taxel_pressure_3d / points, "
                "not lattice="
            )

        if self.taxel_enabled:
            if taxel_pressure_3d is not None:
                taxel_pressure_3d = _require_tensor(
                    "taxel_pressure_3d",
                    taxel_pressure_3d,
                    ndim=5,
                    tail=(
                        self.taxel_num_hands,
                        self.taxel_num_taxels,
                        self.taxel_history_len,
                        self.taxel_point_dim,
                    ),
                )
                if taxel_valid_mask is not None:
                    # valid over taxels (shared across time) OR full [B,H,N,T]
                    if taxel_valid_mask.ndim == 3:
                        _require_tensor(
                            "taxel_valid_mask",
                            taxel_valid_mask,
                            ndim=3,
                            tail=(self.taxel_num_hands, self.taxel_num_taxels),
                        )
                    else:
                        _require_tensor(
                            "taxel_valid_mask",
                            taxel_valid_mask,
                            ndim=4,
                            tail=(
                                self.taxel_num_hands,
                                self.taxel_num_taxels,
                                self.taxel_history_len,
                            ),
                        )
                    if taxel_valid_mask.shape[0] != taxel_pressure_3d.shape[0]:
                        raise ValueError(
                            f"taxel_valid_mask B={taxel_valid_mask.shape[0]} != "
                            f"taxel_pressure_3d B={taxel_pressure_3d.shape[0]}"
                        )
                ref = taxel_pressure_3d if ref is None else ref
        elif taxel_pressure_3d is not None:
            taxel_pressure_3d = None
            taxel_valid_mask = None

        return ref

    def _no_tactile_batch(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Unused by ``encode``: an all-absent batch still runs the
        # stream_mode experts (zeros + ``no_tactile`` splice) so ZeRO-3
        # collectives match a gloved rank. Kept for callers that only
        # need the placeholder tensor.
        if self.no_tactile is None:
            raise RuntimeError(
                "Absent tactile batch requires no_tactile_token=true"
            )
        tokens = self.no_tactile.to(device=device, dtype=dtype).expand(
            batch_size, self.num_tokens, self.output_dim
        )
        # Keep DDP/ZeRO param set identical to the VT path.
        touch = tokens.new_zeros(())
        for param in self.parameters():
            if param.requires_grad:
                touch = touch + param.to(dtype=dtype).sum() * 0
        tokens = tokens + touch
        mask = torch.ones(batch_size, self.num_tokens, dtype=torch.bool, device=device)
        return tokens, mask

    def _encode_region(
        self, region_pressure: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if region_pressure is None:
            raise ValueError(
                "region_pressure required when any sample is present "
                f"(pack [B, {self.region_num_hands}, {self.region_num_regions}, "
                f"{self.region_history_len}, {self.region_channel_dim}])"
            )
        B, H, R, T, C = region_pressure.shape
        # Same contract as `_encode_taxel`: a NaN/Inf frame must not reach
        # TemporalConv (one bad scalar would poison that region token and,
        # via flatten_region_patch, the global AdaLN temb). Fully-nonfinite
        # regions stay in the sequence as a computed token of zeros but are
        # masked out of cross-attn. ``masked_fill`` (not ``nan_to_num``)
        # maps Inf to 0 too. Clean data is all-finite, so this is identity.
        finite = torch.isfinite(region_pressure)
        region_ok = finite.reshape(B, H, R, -1).any(dim=-1)  # [B, H, R]
        x = region_pressure.masked_fill(~finite, 0.0)
        x = x.reshape(B * H * R, T, C)
        h = self.region_temporal(x).view(B, H, R, self.hidden_dim)
        h = h.masked_fill(~torch.isfinite(h), 0.0)
        hand_tokens = []
        hand_masks = []
        for hand in range(H):
            tok = h[:, hand]  # [B, R, D]
            tok = tok + self.region_hand_embed.weight[hand].to(dtype=tok.dtype)
            tok = tok + self.region_slot_embed.weight.to(dtype=tok.dtype).unsqueeze(0)
            hand_tokens.append(tok)
            hand_masks.append(region_ok[:, hand])
        return torch.cat(hand_tokens, dim=1), torch.cat(hand_masks, dim=1)

    def _fuse_taxel_points(
        self, taxel: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """``[B,H,N,T,C]`` → last-frame points + optional hidden temporal feats."""
        B, H, N, T, C = taxel.shape
        last = taxel[:, :, :, -1, :]  # [B,H,N,C]
        if self.taxel_temporal is None:
            return last, None
        pressure = taxel[..., -1].reshape(B * H * N, T, 1)
        temporal = self.taxel_temporal(pressure).view(B, H, N, self.hidden_dim)
        return last, temporal

    def _encode_taxel(
        self,
        taxel_pressure_3d: Optional[torch.Tensor],
        valid_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if taxel_pressure_3d is None:
            raise ValueError(
                "taxel_pressure_3d required when any sample is present "
                f"(pack [B, {self.taxel_num_hands}, {self.taxel_num_taxels}, "
                f"{self.taxel_history_len}, {self.taxel_point_dim}])"
            )
        points, temporal_feats = self._fuse_taxel_points(taxel_pressure_3d)
        B, H, N, C = points.shape
        # Non-finite pads must never reach the point MLP. The EgoPro export
        # writes NaN xyz/pressure whenever a hand's geometry could not be
        # placed (skeleton drop / off-camera), and nothing downstream
        # sanitises it. A single non-finite pad would poison EVERY region
        # token: the pooling einsum does ``0 * nan = nan`` / ``0 * inf = nan``
        # even for a masked pad. Mark those pads invalid and zero their values
        # so masked positions stay finite. ``masked_fill`` (not ``nan_to_num``)
        # maps BOTH NaN and Inf to 0 — ``nan_to_num`` turns Inf into ~3.4e38,
        # which can still overflow a Linear to Inf and re-poison the einsum.
        # Clean data is all-finite, so this is identity.
        finite = torch.isfinite(points).all(dim=-1)  # [B, H, N]
        points = points.masked_fill(~finite.unsqueeze(-1), 0.0)
        if temporal_feats is not None:
            # Same sanitisation contract as `points` above: Inf must become
            # 0 too, not ~3.4e38 (the default posinf fill can overflow a
            # later Linear back to Inf).
            temporal_feats = torch.nan_to_num(
                temporal_feats, nan=0.0, posinf=0.0, neginf=0.0
            )
        if valid_mask is None:
            valid_mask = finite
        else:
            valid_mask = valid_mask.to(device=points.device, dtype=torch.bool)
            if valid_mask.ndim == 4:
                valid_mask = valid_mask.any(dim=-1)
            valid_mask = valid_mask & finite
        if not self.keep_zero_pressure_points and taxel_pressure_3d.shape[-1] >= 4:
            valid_mask = valid_mask & (taxel_pressure_3d[..., -1, 3].abs() > 1e-6)
        if self.drop_stuck_taxels:
            stuck = self._stuck_taxel_mask(taxel_pressure_3d)
            if stuck is not None:
                valid_mask = valid_mask & ~stuck

        hand_tokens, hand_masks = [], []
        for h in range(H):
            tok, msk = self.taxel_backbone(
                points[:, h],
                valid_mask[:, h],
                region_ids=self.region_map,
                point_features=None if temporal_feats is None else temporal_feats[:, h],
            )
            if self.taxel_token_mixer is not None:
                tok = tok + self._mix_hand_tokens(tok, msk)
            tok = tok + self.taxel_hand_embed.weight[h].to(dtype=tok.dtype)
            tok = tok + self.taxel_slot_embed.weight.to(dtype=tok.dtype).unsqueeze(0)
            hand_tokens.append(tok)
            hand_masks.append(msk)
        return torch.cat(hand_tokens, dim=1), torch.cat(hand_masks, dim=1)

    def _encode_lattice(
        self,
        lattice: Optional[torch.Tensor],
        region_valid: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if lattice is None:
            raise ValueError(
                "lattice required when any sample is present "
                f"(pack [B, {self.taxel_num_hands}, {self.taxel_tokens_per_hand}, "
                f"{self.taxel_point_dim}, Hr, Wr])"
            )
        hand_tokens, hand_masks = [], []
        for h in range(lattice.shape[1]):
            rv = None if region_valid is None else region_valid[:, h]
            tok, msk = self.taxel_backbone(lattice[:, h], region_valid=rv)
            tok = tok + self.taxel_hand_embed.weight[h].to(dtype=tok.dtype)
            tok = tok + self.taxel_slot_embed.weight.to(dtype=tok.dtype).unsqueeze(0)
            hand_tokens.append(tok)
            hand_masks.append(msk)
        return torch.cat(hand_tokens, dim=1), torch.cat(hand_masks, dim=1)

    def _stuck_taxel_mask(
        self, taxel_pressure_3d: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """``[B, H, N]`` taxels to drop, or ``None`` when the heuristic is off.

        ``stuck_abs > 0`` (shipped stems): last-frame saturation valve, and
        the only test that can fire at ``T=1``. It also wins at ``T>1``: a
        short hold looks "stuck" under temporal variance and would drop real
        fingertip contact. Temporal std/mean is only the fallback when
        ``stuck_abs`` is off and ``T>1`` (long-window stuck-ON sensors).
        """
        if taxel_pressure_3d.shape[-1] < 4:
            return None
        pressure = taxel_pressure_3d[..., 3]
        if self.stuck_abs > 0.0:
            return pressure[..., -1].abs() > self.stuck_abs
        if pressure.shape[-1] > 1:
            return (pressure.std(dim=-1) < self.stuck_std) & (
                pressure.mean(dim=-1).abs() > self.stuck_mean
            )
        return None

    def _mix_hand_tokens(
        self, tok: torch.Tensor, msk: torch.Tensor
    ) -> torch.Tensor:
        """Per-hand mixer; padded slots do not attend. Fully-invalid rows
        keep a dummy unmasked position so softmax stays finite.

        The mixer is ALWAYS forwarded, and slot 0 is unmasked branch-free.
        An early return on ``pad.all()`` used to skip the module entirely,
        which under ZeRO-3 is a missing all-gather: a rank whose microbatch
        happens to be fully invalid (V-only rows zero-filled with
        ``keep_zero_pressure_points=False``, an all-false validity column, or
        the stuck-taxel heuristic dropping everything) would issue one
        collective fewer than a rank with live taxels and the job deadlocks.
        The ``bool(...)`` guards also forced a host sync on every call.
        """
        pad = ~msk
        fully = pad.all(dim=-1, keepdim=True)
        slot0 = torch.zeros_like(pad)
        slot0[..., 0] = True
        pad = pad & ~(fully & slot0)
        return self.taxel_token_mixer(tok, src_key_padding_mask=pad)

    def _content_valid_mask(
        self,
        taxel_pressure_3d: torch.Tensor,
        taxel_valid_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Per-taxel content validity for the quality gate.

        An exporter mask (Tianji ``validity_*``) is ANDed with finite
        geometry (it cannot resurrect a NaN pad). EgoPro has no mask and
        publishes real camera-frame xyz for every placed pad — including
        zero-pressure pads, whose hand pose is still meaningful — and NaN
        for pads it could not place. So "valid" is *finite content*, NOT
        contact: a broken / half-placed cloud reads as thin while a
        not-touching hand stays fully valid. Keying this off pressure
        (``|p| > 0``) instead would gate every no-contact frame and make
        ``no_tactile`` a pre-grasp shortcut. Pressure *saturation* is a
        separate ``quality_sat_abs`` check, but pressure *finiteness* is
        part of validity here: a NaN-pressure pad carries no force reading,
        and ``_encode_taxel`` drops it from the encode anyway — counting it
        as valid would keep an all-NaN-pressure cloud live instead of
        falling back to ``no_tactile``.
        """
        last = taxel_pressure_3d[..., -1, :]
        finite = torch.isfinite(last).all(dim=-1)
        if taxel_valid_mask is not None:
            valid = taxel_valid_mask.to(
                device=taxel_pressure_3d.device, dtype=torch.bool
            )
            if valid.ndim == 4:
                valid = valid.any(dim=-1)
            # AND with finite geometry: a mask can be present on a mixed batch
            # where some rows (e.g. EgoPro) never opted into validity and were
            # filled all-True by packing. Without this, a NaN cloud on those
            # rows would read as fully valid and bypass the geometry gate.
            return valid & finite
        return finite

    def _apply_quality_gate(
        self,
        present_slow: torch.Tensor,
        taxel_pressure_3d: Optional[torch.Tensor],
        taxel_valid_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Mark a row absent when the taxel cloud is too thin or saturated.

        Defaults (min_valid=0, max_sat=1) are a no-op. ``min_valid`` gates on
        the fraction of pads with *finite content* (see ``_content_valid_mask``):
        a hand whose export collapsed to NaN, or a dirty glove that publishes
        garbage, falls back to ``no_tactile`` — the same contract as a missing
        column — while a not-touching-but-placed hand stays live. ``max_sat`` /
        ``quality_sat_abs`` separately drop clouds that are mostly saturated.
        """
        if not self.taxel_enabled or taxel_pressure_3d is None:
            return present_slow
        if self.quality_min_valid_frac <= 0.0 and self.quality_max_sat_frac >= 1.0:
            return present_slow
        B = present_slow.shape[0]
        valid = self._content_valid_mask(taxel_pressure_3d, taxel_valid_mask)
        valid_frac = valid.reshape(B, -1).float().mean(dim=-1)
        keep = present_slow & (valid_frac >= self.quality_min_valid_frac)
        if self.quality_sat_abs > 0.0 and taxel_pressure_3d.shape[-1] >= 4:
            pressure = taxel_pressure_3d[..., -1, 3].abs()
            sat_frac = (pressure > self.quality_sat_abs).float().reshape(B, -1).mean(dim=-1)
            keep = keep & (sat_frac <= self.quality_max_sat_frac)
        return keep

    def _apply_present(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        present: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if bool(present.all()):
            return tokens, token_mask
        B, T, D = tokens.shape
        present_b1 = present.view(B, 1)
        token_mask = token_mask & present_b1
        if self.no_tactile is None:
            return tokens * present.view(B, 1, 1).to(tokens.dtype), token_mask
        filler = self.no_tactile.to(dtype=tokens.dtype).expand(B, T, D)
        tokens = torch.where(present.view(B, 1, 1), tokens, filler)
        return tokens, token_mask | (~present_b1)


def tactile_infer_cache_id(examples) -> Optional[Tuple[Tuple[int, int], ...]]:
    """Per-sample ``(env_idx, tactile_cache_gen)`` keys for the slow-3D cache.

    ``encode_with_slow_refresh`` stores one slot per ``env_idx`` so RTC
    subset fires and ``max_infer_batch`` splits can still hit. A multi-env
    batch that omits ``env_idx`` returns ``None`` so nothing is reused —
    same-B batches of different env sets would otherwise share one slot
    keyed only by generation. A single-env call without ``env_idx`` still
    caches (``env_idx=-1``) so unit tests and the first unguided infer work.
    Duplicate ``env_idx`` in one batch is rejected: two rows cannot own
    one per-env slot.
    """
    ids: List[Tuple[int, int]] = []
    missing_env = False
    for ex in examples:
        if not isinstance(ex, dict):
            ids.append((-1, 0))
            missing_env = True
            continue
        env = ex.get("env_idx")
        gen = int(ex.get("tactile_cache_gen", 0) or 0)
        if env is None:
            missing_env = True
            ids.append((-1, gen))
        else:
            ids.append((int(env), gen))
    if missing_env and len(ids) > 1:
        return None
    envs = [env for env, _ in ids]
    if len(envs) != len(set(envs)):
        raise ValueError(
            f"tactile slow cache requires unique env_idx per row, got {ids}"
        )
    return tuple(ids)


def _regularize_cached_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Detach and clone so a ``predict_action`` ``inference_mode`` tensor
    can be reused under RTC's ``no_grad`` + ΠGDM ``enable_grad``.

    ``clone()`` inside ``inference_mode`` still yields an inference tensor —
    leave that context first.
    """
    with torch.inference_mode(False):
        return tensor.detach().clone()


def slow_refresh_control_period(
    infer_period_steps: int, slow_refresh_every: int
) -> int:
    """Control steps between full 3D encodes.

    ``slow_refresh_every`` counts **infer calls**, not control steps. A
    3D encode can only run at an infer, so the control-step period is
    always ``infer_period_steps * slow_refresh_every``. ``N=1`` refreshes
    3D on every infer (every RTC fire). Two configs that both say ``8``
    (RTC period 8 steps, ``slow_refresh_every: 8``) mean 64 control steps.
    """
    period = int(infer_period_steps)
    every = int(slow_refresh_every)
    if period < 1 or every < 1:
        raise ValueError(
            "infer_period_steps and slow_refresh_every must be >= 1 "
            f"(got period={period}, every={every})"
        )
    return period * every


def _normalize_slow_cache_keys(cache_id: Any) -> Optional[List[Tuple[int, int]]]:
    """``((env, gen), ...)`` from ``tactile_infer_cache_id``.

    A legacy whole-batch id like ``(0,)`` is treated as ``[(0, 0)]`` so
    older unit tests keep their meaning (different id → different env).
    """
    if cache_id is None:
        return None
    if not isinstance(cache_id, (tuple, list)) or not cache_id:
        return None
    keys: List[Tuple[int, int]] = []
    for item in cache_id:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            keys.append((int(item[0]), int(item[1])))
        else:
            keys.append((int(item), 0))
    envs = [env for env, _ in keys]
    if len(envs) != len(set(envs)):
        raise ValueError(
            f"tactile slow cache requires unique env_idx per row, got {keys}"
        )
    return keys


def _as_env_slow_cache(cache: Optional[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Accept the per-env dict; drop the legacy whole-batch slot."""
    if not cache:
        return {}
    if "tokens" in cache and "id" in cache:
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    for key, entry in cache.items():
        if not isinstance(entry, dict) or "tokens" not in entry:
            continue
        out[int(key)] = entry
    return out


def _batch_size_from_encode_kwargs(kwargs: Dict[str, Any]) -> Optional[int]:
    for key in (
        "region_pressure",
        "taxel_pressure_3d",
        "points",
        "present",
        "present_fast",
        "present_slow",
    ):
        tensor = kwargs.get(key)
        if isinstance(tensor, torch.Tensor) and tensor.ndim >= 1:
            return int(tensor.shape[0])
    batch = kwargs.get("batch_size")
    return int(batch) if batch is not None else None


def _slice_encode_kwargs(
    kwargs: Dict[str, Any], index: torch.Tensor, batch_size: int
) -> Dict[str, Any]:
    """Row-select every batched tensor; keep scalars (device / dropout)."""
    n = int(index.numel())
    out: Dict[str, Any] = {}
    for key, value in kwargs.items():
        if key == "batch_size":
            out[key] = n
        elif (
            isinstance(value, torch.Tensor)
            and value.ndim >= 1
            and value.shape[0] == batch_size
        ):
            out[key] = value.index_select(0, index.to(device=value.device))
        else:
            out[key] = value
    return out


def _stack_cached_slow(
    cache: Dict[int, Dict[str, Any]],
    keys: Sequence[Tuple[int, int]],
    rows: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.cat([cache[keys[i][0]]["tokens"] for i in rows], dim=0)
    mask = torch.cat([cache[keys[i][0]]["mask"] for i in rows], dim=0)
    return tokens, mask


def _live_slow_row_mask(
    encoder: TactileEncoder,
    kwargs: Dict[str, Any],
    batch: int,
) -> Optional[torch.Tensor]:
    """Rows whose full encode produced live slow (3D) tokens.

    V-only / no_tactile batches are ``None`` so a good per-env 3D slot is
    not overwritten with filler. ``present_slow`` (else ``present``) gates
    mixed batches; omitted flags mean "tensor present ⇒ every row live".

    The packed flags are pre-gate, so the encoder's quality gate is replayed
    here: a cloud ``encode`` will replace with ``no_tactile`` is not live and
    must not land in the cache (nor force a filler-producing miss).
    """
    if int(getattr(encoder, "slow_num_tokens", 0) or 0) <= 0:
        return None
    has_slow = False
    if getattr(encoder, "taxel_enabled", False) and getattr(
        encoder, "taxel_role", None
    ) == "slow":
        has_slow = any(
            kwargs.get(key) is not None
            for key in ("taxel_pressure_3d", "points", "lattice")
        )
    if getattr(encoder, "region_enabled", False) and getattr(
        encoder, "region_role", None
    ) == "slow":
        has_slow = has_slow or kwargs.get("region_pressure") is not None
    if not has_slow:
        return None
    taxel = kwargs.get("taxel_pressure_3d")
    taxel_valid = kwargs.get("taxel_valid_mask")
    if taxel is None and kwargs.get("points") is not None:
        taxel = encoder._points_to_taxel(kwargs["points"])
    if taxel_valid is None:
        # encode() applies the legacy valid_mask alias unconditionally, not
        # only on the points= path; the gate replay must see the same mask.
        taxel_valid = kwargs.get("valid_mask")

    # Mirror encode()'s role resolution: an explicit present_slow wins; a
    # present_fast-only call leaves the slow role absent (not inherited from
    # fast / present), so no slow row is live; a bare present (both roles
    # omitted) fills both; otherwise a bare tensor is inferred all-live.
    present_slow = kwargs.get("present_slow")
    present_fast = kwargs.get("present_fast")
    present = kwargs.get("present")
    if present_slow is not None:
        mask = present_slow
    elif present_fast is not None:
        return None
    elif present is not None:
        mask = present
    elif isinstance(taxel, torch.Tensor):
        mask = torch.ones(batch, dtype=torch.bool, device=taxel.device)
    else:
        return torch.ones(batch, dtype=torch.bool)
    if not isinstance(mask, torch.Tensor) or tuple(mask.shape) != (batch,):
        return None
    mask = mask.to(dtype=torch.bool)
    if isinstance(taxel, torch.Tensor):
        mask = encoder._apply_quality_gate(
            mask.to(device=taxel.device), taxel, taxel_valid
        )
    if not bool(mask.any()):
        return None
    return mask


def _store_live_slow_rows(
    encoder: TactileEncoder,
    kwargs: Dict[str, Any],
    tokens: torch.Tensor,
    mask: torch.Tensor,
    env_cache: Dict[int, Dict[str, Any]],
    keys: Sequence[Tuple[int, int]],
) -> None:
    """Write live slow rows from a full-encode into per-env slots.

    ``keys`` is already aligned with ``tokens`` (full batch or a miss slice).
    V-only / no_tactile / ``present_slow=False`` must not overwrite a good
    slot — including the ``use_fast`` all-miss / mixed-miss path, where
    ``region_pressure`` is present so the encode still runs.
    """
    batch = int(tokens.shape[0])
    if len(keys) != batch:
        raise ValueError(
            f"cache keys has {len(keys)} entries but the tactile batch is {batch}"
        )
    live = _live_slow_row_mask(encoder, kwargs, batch)
    if live is None:
        return
    for row, (env, gen) in enumerate(keys):
        if not bool(live[row].item()):
            continue
        _store_slow_cache_entry(
            env_cache,
            env,
            gen,
            tokens[row : row + 1],
            mask[row : row + 1],
        )


def _store_infer_slow_from_full_encode(
    encoder: TactileEncoder,
    kwargs: Dict[str, Any],
    tokens: torch.Tensor,
    mask: torch.Tensor,
    env_cache: Dict[int, Dict[str, Any]],
    cache_id: Any,
) -> None:
    """Write live slow tokens after an infer full-encode (region-less / N<=1).

    A later ``fast_cached_slow`` hit would otherwise splice whatever 3D was
    already in the slot — including a previous episode's cloud.
    """
    keys = _normalize_slow_cache_keys(cache_id)
    if keys is None:
        return
    batch = _batch_size_from_encode_kwargs(kwargs)
    if batch is None:
        return
    if len(keys) != batch:
        raise ValueError(
            f"cache_id has {len(keys)} keys but the tactile batch is {batch}"
        )
    _store_live_slow_rows(encoder, kwargs, tokens, mask, env_cache, keys)


def _store_slow_cache_entry(
    cache: Dict[int, Dict[str, Any]],
    env: int,
    gen: int,
    tokens: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    if tokens.ndim != 3 or tokens.shape[0] != 1:
        raise ValueError(
            f"per-env tactile cache stores [1, T, D], got {tuple(tokens.shape)}"
        )
    if mask.shape[:2] != tokens.shape[:2]:
        raise ValueError(
            f"cached mask {tuple(mask.shape)} does not match tokens {tuple(tokens.shape)}"
        )
    cache[int(env)] = {
        "gen": int(gen),
        "tokens": _regularize_cached_tensor(tokens),
        "mask": _regularize_cached_tensor(mask),
        "age": 1,
    }


def _fast_encode_with_cached_slow(
    encoder: TactileEncoder,
    kwargs: Dict[str, Any],
    cached_tokens: torch.Tensor,
    cached_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = int(cached_tokens.shape[0])
    region = kwargs.get("region_pressure")
    if isinstance(region, torch.Tensor) and int(region.shape[0]) != n:
        raise ValueError(
            "fast-repair cached slow tokens are not aligned with region_pressure: "
            f"cache B={n} vs region B={int(region.shape[0])}"
        )
    present_fast = kwargs.get("present_fast")
    if present_fast is None:
        present_fast = kwargs.get("present")
    if isinstance(present_fast, torch.Tensor) and int(present_fast.shape[0]) != n:
        raise ValueError(
            "fast-repair cached slow tokens are not aligned with present_fast: "
            f"cache B={n} vs present_fast B={int(present_fast.shape[0])}"
        )
    device = cached_tokens.device
    present_slow = torch.ones(n, dtype=torch.bool, device=device)
    # An explicit present_slow (needed to keep the cached 3D slots)
    # would otherwise default the omitted present_fast to False and
    # skip the live region encode. Infer presence from the caller,
    # then from the region tensor itself.
    if present_fast is None:
        present_fast = torch.ones(n, dtype=torch.bool, device=device)
    return encoder.encode(
        region_pressure=kwargs.get("region_pressure"),
        present=kwargs.get("present"),
        present_fast=present_fast,
        present_slow=present_slow,
        stream_mode="fast",
        cached_slow_tokens=cached_tokens,
        cached_slow_mask=cached_mask,
        apply_modality_dropout=False,
    )


_WARNED_MULTI_RANK_FAST_REPAIR = False


def _multi_rank_infer() -> bool:
    """True when >1 rank shares this process group (disables fast-repair).

    The fast-repair branch runs a data-dependent NUMBER of encoder forwards:
    all-hit is one ``stream_mode="fast"`` encode (the taxel experts are never
    forwarded), all-miss is one full encode, and a mixed batch is both. Under
    ZeRO-3 a module forward all-gathers partitioned weights, so ranks whose
    per-env cache states differ would issue different collective counts and
    deadlock. Cache hits depend on each rank's own env ids / ages, which
    nothing keeps in sync.

    The production policy server is single-process, so this only gives up the
    cache in a multi-rank inference setup (e.g. a distributed eval harness),
    where every rank then takes the plain full-encode path — rank-uniform by
    construction. Warned once so the lost optimization is visible.
    """
    global _WARNED_MULTI_RANK_FAST_REPAIR
    if not dist.is_available() or not dist.is_initialized():
        return False
    if dist.get_world_size() < 2:
        return False
    if not _WARNED_MULTI_RANK_FAST_REPAIR:
        _WARNED_MULTI_RANK_FAST_REPAIR = True
        logger.warning(
            "[tactile] action_fusion.refresh_mode=fast_cached_slow is disabled "
            f"because this process group has {dist.get_world_size()} ranks. The "
            "fast-repair branch runs a cache-dependent number of encoder "
            "forwards, which desynchronises ZeRO-3 all-gathers across ranks. "
            "Every call falls back to a full encode (correct, just slower); "
            "single-process serving keeps the cache."
        )
    return True


def encode_with_slow_refresh(
    encoder: TactileEncoder,
    kwargs: Dict[str, Any],
    *,
    training: bool,
    refresh_mode: str,
    slow_refresh_every: int,
    cache: Optional[Dict[str, Any]],
    cache_id: Any = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
    """Train = full ``encode``. Infer may refresh only the fast patch.

    ``refresh_mode=fast_cached_slow`` keeps the last slow (3D) tokens and
    re-encodes ``region_pressure`` so a chunk can be repaired from the
    cheap region-force stream without rebuilding the point cloud. Falls
    back to a full encode when there is no per-env entry, ``cache_id`` is
    ``None``, the fast tensor is missing, ``slow_refresh_every <= 1``, or the
    process group has more than one rank (see ``_multi_rank_infer``).
    ``slow_refresh_every`` is an infer-call count per env (see
    ``slow_refresh_control_period``). Training never *reads* or *writes*
    the cache — the action loss must see a live slow encode, and a train
    step must not poison the infer slot. A full-encode fallback (missing
    ``region_pressure``, ``N<=1``, single-expert ablation) still returns
    the incoming per-env dict. Assigning ``None`` would wipe every env's
    3D slot on a V-only / no-region batch (RTC subset fire,
    ``max_infer_batch`` split, bench-omitted region). Infer + live slow
    (taxel / points present, not all-absent) + a valid ``cache_id`` also
    writes those tokens so a later fast-repair cannot splice stale 3D —
    the same ``_live_slow_row_mask`` gate covers the ``use_fast`` all-miss
    / mixed-miss full-encode, where ``region_pressure`` is present but
    ``present_slow=False`` must still not overwrite a good slot. A miss
    that is not live but already has a matching-gen slot is reclassified
    as a hit (reuse the last cloud, advance age) so a refresh boundary
    cannot spin on full-encodes. Reclassified hits are appended, then
    ``hit_rows`` is sorted so the all-hit fast-repair stacks cache in
    batch order (unsorted ``[later_hit, earlier_reclass]`` would splice
    env A's 3D onto env B's region). V-only / no_tactile does not
    overwrite a good slot.

    The cache is a dict keyed by ``env_idx`` (not one slot for the whole
    batch). RTC subset fires and ``max_infer_batch`` splits therefore
    still hit. A mixed hit/miss batch fast-repairs the hits and
    full-encodes the misses. On a cache hit, ``present_slow`` is forced
    on so a missing taxel column on the current frame cannot wipe the
    stored 3D slots. ``tactile_cache_gen`` must match or that env is
    refreshed — a ``reset()`` cannot keep the previous episode's 3D.
    """
    # Cache is a dual-stream infer trick. A single-expert ablation
    # (fast_enable / slow_enable / streams.*.enable) must keep the old
    # full-encode path — including robust.yaml's refresh_mode default.
    use_fast = (
        not bool(training)
        and str(refresh_mode).lower() == "fast_cached_slow"
        and encoder.fast_num_tokens > 0
        and encoder.slow_num_tokens > 0
        and kwargs.get("region_pressure") is not None
        and int(slow_refresh_every) > 1
        and not _multi_rank_infer()
    )
    if not use_fast:
        tokens, mask = encoder.encode(**kwargs)
        # Preserve other envs. Training never writes. Infer + live slow
        # refreshes that env's slot; V-only / no_tactile must not.
        env_cache = _as_env_slow_cache(cache)
        if not training:
            _store_infer_slow_from_full_encode(
                encoder, kwargs, tokens, mask, env_cache, cache_id
            )
        return tokens, mask, env_cache or None

    keys = _normalize_slow_cache_keys(cache_id)
    env_cache = _as_env_slow_cache(cache)
    if keys is None:
        tokens, mask = encoder.encode(**kwargs)
        return tokens, mask, env_cache or None

    batch = _batch_size_from_encode_kwargs(kwargs)
    if batch is None:
        raise ValueError(
            "encode_with_slow_refresh needs a batched tactile tensor "
            "(region_pressure / taxel_pressure_3d / present_*)."
        )
    if len(keys) != batch:
        raise ValueError(
            f"cache_id has {len(keys)} keys but the tactile batch is {batch}"
        )

    every = int(slow_refresh_every)
    hit_rows: List[int] = []
    miss_rows: List[int] = []
    for row, (env, gen) in enumerate(keys):
        entry = env_cache.get(int(env))
        if (
            entry is not None
            and int(entry.get("gen", -1)) == int(gen)
            and int(entry.get("age", 0)) % every != 0
        ):
            hit_rows.append(row)
        else:
            miss_rows.append(row)

    # Refresh-due with no live 3D must not full-encode filler (that used
    # to poison the slot). A matching-gen slot is a hit: reuse the last
    # cloud and advance age so we do not spin on the same boundary. A
    # gen mismatch (new episode) stays a miss and does not write.
    if miss_rows:
        live = _live_slow_row_mask(encoder, kwargs, batch)
        kept_miss: List[int] = []
        for row in miss_rows:
            env, gen = keys[row]
            entry = env_cache.get(int(env))
            has_slot = (
                entry is not None and int(entry.get("gen", -1)) == int(gen)
            )
            row_live = live is not None and bool(live[row].item())
            if has_slot and not row_live:
                hit_rows.append(row)
            else:
                kept_miss.append(row)
        miss_rows = kept_miss

    hit_rows.sort()
    miss_rows.sort()

    if not miss_rows:
        if hit_rows != list(range(batch)):
            raise ValueError(
                "tactile fast-repair all-hit path expects hit_rows == "
                f"0..{batch - 1} in order (cache stacked against full-batch "
                f"kwargs); got {hit_rows}"
            )
        cached_tokens, cached_mask = _stack_cached_slow(env_cache, keys, hit_rows)
        tokens, mask = _fast_encode_with_cached_slow(
            encoder, kwargs, cached_tokens, cached_mask
        )
        for row in hit_rows:
            env = keys[row][0]
            env_cache[env]["age"] = int(env_cache[env]["age"]) + 1
        return tokens, mask, env_cache

    if not hit_rows:
        tokens, mask = encoder.encode(**kwargs)
        _store_live_slow_rows(encoder, kwargs, tokens, mask, env_cache, keys)
        return tokens, mask, env_cache

    device = None
    for key in ("region_pressure", "taxel_pressure_3d", "points"):
        tensor = kwargs.get(key)
        if isinstance(tensor, torch.Tensor):
            device = tensor.device
            break
    if device is None:
        device = torch.device("cpu")
    hit_index = torch.tensor(hit_rows, dtype=torch.long, device=device)
    miss_index = torch.tensor(miss_rows, dtype=torch.long, device=device)

    miss_kwargs = _slice_encode_kwargs(kwargs, miss_index, batch)
    miss_tokens, miss_mask = encoder.encode(**miss_kwargs)
    miss_present_fast = getattr(encoder, "_last_present_fast", None)
    miss_present_slow = getattr(encoder, "_last_present_slow", None)
    _store_live_slow_rows(
        encoder,
        miss_kwargs,
        miss_tokens,
        miss_mask,
        env_cache,
        [keys[row] for row in miss_rows],
    )

    cached_tokens, cached_mask = _stack_cached_slow(env_cache, keys, hit_rows)
    hit_tokens, hit_mask = _fast_encode_with_cached_slow(
        encoder,
        _slice_encode_kwargs(kwargs, hit_index, batch),
        cached_tokens,
        cached_mask,
    )
    hit_present_fast = getattr(encoder, "_last_present_fast", None)
    hit_present_slow = getattr(encoder, "_last_present_slow", None)
    if hit_tokens.shape[1:] != miss_tokens.shape[1:]:
        raise ValueError(
            "tactile fast-repair / full-encode token shapes disagree: "
            f"hit {tuple(hit_tokens.shape)} vs miss {tuple(miss_tokens.shape)}"
        )
    tokens = miss_tokens.new_empty((batch,) + tuple(miss_tokens.shape[1:]))
    mask = miss_mask.new_empty((batch,) + tuple(miss_mask.shape[1:]))
    tokens.index_copy_(0, miss_index, miss_tokens)
    mask.index_copy_(0, miss_index, miss_mask)
    tokens.index_copy_(0, hit_index.to(device=tokens.device), hit_tokens)
    mask.index_copy_(0, hit_index.to(device=mask.device), hit_mask)
    for row in hit_rows:
        env = keys[row][0]
        env_cache[env]["age"] = int(env_cache[env]["age"]) + 1
    def _restack_present(miss_flag, hit_flag):
        if not (
            isinstance(miss_flag, torch.Tensor) and isinstance(hit_flag, torch.Tensor)
        ):
            return None
        stacked = miss_flag.new_empty((batch,), dtype=torch.bool)
        stacked.index_copy_(0, miss_index.to(device=stacked.device), miss_flag)
        stacked.index_copy_(0, hit_index.to(device=stacked.device), hit_flag)
        return stacked

    restacked_fast = _restack_present(miss_present_fast, hit_present_fast)
    restacked_slow = _restack_present(miss_present_slow, hit_present_slow)
    if restacked_fast is None or restacked_slow is None:
        raise RuntimeError(
            "mixed hit/miss tactile refresh could not restack present_* "
            "to the full batch; leaving the last subset's flags would make "
            "region FiLM see the wrong B"
        )
    encoder._last_present_fast = restacked_fast
    encoder._last_present_slow = restacked_slow
    return tokens, mask, env_cache


def get_tactile_encoder(cfg: Any) -> TactileEncoder:
    return TactileEncoder(cfg)


__all__ = [
    "TactileEncoder",
    "encode_with_slow_refresh",
    "get_tactile_encoder",
    "slow_refresh_control_period",
    "stream_is_active",
    "tactile_infer_cache_id",
]
