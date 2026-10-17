"""Experimental VITRA-style sparse index/sign mapping for simulator Inspire.

NOT the official VITRA XHand correspondence table. The same index/sign idea is
adapted to the actual EgoVLA Inspire USD and Wuji hand2 URDF joint axes.
No fitted matrix, gain, bias, angle splitting, IK, MANO mesh or clipping in
EITHER direction. Mechanical limits are deliberately not part of this codec.
"""
from __future__ import annotations

import numpy as np

CANONICAL_NAMES = (
    'thumb_proximal_yaw_joint', 'index_proximal_joint',
    'middle_proximal_joint', 'ring_proximal_joint', 'pinky_proximal_joint',
    'thumb_proximal_pitch_joint', 'thumb_intermediate_joint', 'thumb_distal_joint',
    'index_intermediate_joint', 'middle_intermediate_joint',
    'ring_intermediate_joint', 'pinky_intermediate_joint',
)
BENCH_TO_CANONICAL = np.array([8, 0, 2, 6, 4, 9, 10, 11, 1, 3, 7, 5])
CANONICAL_TO_BENCH = np.argsort(BENCH_TO_CANONICAL)
# One unique Wuji channel for each canonical Inspire component.
CANONICAL_TO_WUJI = np.array([4, 0, 1, 3, 2, 9, 14, 19, 10, 11, 13, 12])
UNMAPPED_WUJI = np.array([5, 6, 7, 8, 15, 16, 17, 18])
CODEC_ID = 'inspire12_wuji20_vitra_index_sign_unbounded_v2'


def as_angles(values, width):
    a = np.asarray(values)
    if a.ndim < 1 or a.shape[-1] != width:
        raise ValueError(f'Expected (...,{width}), got {a.shape}')
    if a.dtype.kind not in 'fiu':
        raise ValueError('Expected real numeric joint angles')
    if a.dtype.kind != 'f':
        a = a.astype(np.float64)
    elif a.dtype.itemsize < 4:
        a = a.astype(np.float32)
    if not np.all(np.isfinite(a)):
        raise ValueError('Joint angles must be finite')
    return a


class VitraStyleHandMapping:
    """A single hand; signed index copy with no mechanical limit enforcement.

Both actual hands use +1 for the twelve selected pairs. A sign vector is
accepted only to test the general signed-permutation invariant, not to fit.
Public methods default to BENCH12 order; use order='canonical' for raw audits.
"""
    def __init__(self, signs=None):
        self.signs = np.ones(12) if signs is None else np.asarray(signs)
        if self.signs.shape != (12,) or not np.all(np.isin(self.signs, [-1, 1])):
            raise ValueError('Only exactly +1/-1 signs are permitted')
        self.mapped_mask = np.zeros(20, dtype=bool)
        self.mapped_mask[CANONICAL_TO_WUJI] = True

    @staticmethod
    def _check_order(order):
        if order not in ('bench', 'canonical'):
            raise ValueError('order must be bench or canonical')

    def inspire_to_wuji(self, q12, *, order='bench'):
        self._check_order(order)
        q = as_angles(q12, 12)
        if order == 'bench':
            q = q[..., BENCH_TO_CANONICAL]
        w = np.zeros(q.shape[:-1]+(20,), dtype=q.dtype)
        w[..., CANONICAL_TO_WUJI] = q*self.signs.astype(q.dtype)
        return w  # Deliberately unbounded, even if original Inspire is illegal.

    def wuji_to_inspire(self, w20, *, order='bench'):
        self._check_order(order)
        w = as_angles(w20, 20)
        q = w[..., CANONICAL_TO_WUJI]*self.signs.astype(w.dtype)
        return q[..., CANONICAL_TO_BENCH] if order == 'bench' else q
