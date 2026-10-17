#!/usr/bin/env python
"""InspireHand FK for AgiBotWorld-Beta(hand) dexhand (dex-urdf assets).

AgiBot per-hand 6-D (rad, calibrated) → URDF actuated joints → 5×5 anatomical
chains in ``hand_base_link``. Mimic joints are applied from the URDF.
"""

from __future__ import annotations

import numpy as np

from ..common import (
    DEFAULT_INSPIRE_ASSET_DIR as DEFAULT_ASSET_DIR,
    INSPIRE_ACTUATED as ACTUATED,
    INSPIRE_ANATOMICAL_CHAIN as ANATOMICAL_CHAIN,
    INSPIRE_THUMB_LINK_CHAIN as THUMB_LINK_CHAIN,
    InspireHandKinematics,
)

# AgiBot disk order (left 0..5 / right 6..11): calibrated 2026-08-06
#   0 thumb_swing, 1 index, 2 middle, 3 ring, 4 pinky, 5 thumb_flex (frozen)
AGIBOT_SWING_LO, AGIBOT_SWING_HI = -0.30, 0.28
AGIBOT_FLEX_LO, AGIBOT_FLEX_HI = -0.165, 1.06


def agibot6_to_inspire_q(agibot6: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Map one hand (T,6) or (6,) AgiBot rad → Inspire actuated q in ``ACTUATED`` order.

    Flexion dims are treated as already-near-Inspire radians: shift so the
    open resting value (``AGIBOT_FLEX_LO``) lands on URDF lower, then clamp —
    do **not** stretch AgiBot's ~1.22 rad span onto the full 1.47 URDF span.
    Thumb yaw still uses normalized swing (open → lower, opposed → upper).
    Thumb pitch uses 4-finger grasp synergy because AgiBot dim5 is frozen.
    """
    a = np.asarray(agibot6, dtype=np.float64)
    single = a.ndim == 1
    if single:
        a = a.reshape(1, 6)
    if a.ndim != 2 or a.shape[1] != 6:
        raise ValueError(f"expected (T,6), got {a.shape}")
    sN = np.clip((a[:, 0] - AGIBOT_SWING_LO) / (AGIBOT_SWING_HI - AGIBOT_SWING_LO), 0, 1)
    # shift open→0, keep rad magnitude (no full-range stretch)
    flex = np.clip(a[:, 1:5] - AGIBOT_FLEX_LO, 0.0, None)
    g = np.clip(flex.mean(axis=1) / (AGIBOT_FLEX_HI - AGIBOT_FLEX_LO), 0.0, 1.0)
    yaw = lower[0] + (1.0 - sN) * (upper[0] - lower[0])
    index = np.clip(lower[1] + flex[:, 0], lower[1], upper[1])
    middle = np.clip(lower[2] + flex[:, 1], lower[2], upper[2])
    ring = np.clip(lower[3] + flex[:, 2], lower[3], upper[3])
    pinky = np.clip(lower[4] + flex[:, 3], lower[4], upper[4])
    pitch = lower[5] + g * (upper[5] - lower[5])
    q = np.stack([yaw, index, middle, ring, pinky, pitch], axis=-1)
    return q[0] if single else q


class InspireHandModel(InspireHandKinematics):
    """Kinematic FK for one AgiBot Inspire hand."""

    def agibot_to_q(self, agibot6: np.ndarray) -> np.ndarray:
        return agibot6_to_inspire_q(agibot6, self.lower, self.upper)
