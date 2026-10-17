"""Raw EgoVLA dual-hand adapter for the strictly unbounded index/sign codec."""
import json
from pathlib import Path
import numpy as np
try:
    from .vitra_joint_mapping import VitraStyleHandMapping, CODEC_ID, CANONICAL_TO_WUJI, UNMAPPED_WUJI
except ImportError:
    from vitra_joint_mapping import VitraStyleHandMapping, CODEC_ID, CANONICAL_TO_WUJI, UNMAPPED_WUJI

DEFAULT_MAPPING = Path(__file__).with_name('hand_mapping.json')
RAW_QPOS_INDICES = np.array([[30,26,27,29,28,40,46,48,36,37,39,38],
                            [35,31,32,34,33,45,47,49,41,42,44,43]])


def wuji_order(side):
    p = 'l' if side == 'left' else 'r'
    fingers = ('index_finger','middle_finger','pinky','ring_finger','thumb')
    return [f'{p}_{f}_{t if f != "thumb" else tt}'
            for t,tt in [('mcp_flex','cmc_flex'),('mcp_abd','cmc_abd'),('pip','mcp'),('dip','ip')]
            for f in fingers]


class HandCodec:
    def __init__(self, mapping=DEFAULT_MAPPING):
        self.calibration = json.loads(Path(mapping).read_text())
        c = self.calibration
        if (c['codec'] != CODEC_ID or c['canonical_to_wuji'] != CANONICAL_TO_WUJI.tolist()
                or c['unmapped_wuji_zero'] != UNMAPPED_WUJI.tolist()
                or c['forward_clip'] is not False or c['reverse_clip'] is not False
                or c['gain'] is not None or c['offset'] is not None):
            raise ValueError('Mapping table does not describe the approved unbounded codec')
        if c['signs'] != [1]*12:
            raise ValueError('Reviewed simulator mapping uses +1 for all selected pairs')
        self.codec = VitraStyleHandMapping(signs=c['signs'])
        c['sides'] = {s: {'wuji20_order': wuji_order(s)} for s in ('left','right')}

    def inspire12_to_wuji20(self, q):
        q = np.asarray(q)
        if q.shape[-2:] != (2,12):
            raise ValueError('Expected (...,2,12) canonical dual-hand angles')
        return self.codec.inspire_to_wuji(q, order='canonical')

    def wuji20_to_inspire12(self, w):
        w = np.asarray(w)
        if w.shape[-2:] != (2,20):
            raise ValueError('Expected (...,2,20) dual-hand Wuji angles')
        return self.codec.wuji_to_inspire(w, order='canonical')


def inspire12_from_qpos(qpos):
    """Extract canonical left/right Inspire12 angles from EgoVLA qpos."""
    qpos = np.asarray(qpos)
    if qpos.ndim != 2 or qpos.shape[1] != 50:
        raise ValueError(f"Expected qpos (T,50), got {qpos.shape}")
    if qpos.dtype.kind not in "fiu":
        raise ValueError("Expected numeric qpos")
    return qpos[:, RAW_QPOS_INDICES]


def wuji20_from_qpos(qpos, *, codec=None):
    """Apply the approved table map directly: EgoVLA Inspire12 -> Wuji20."""
    codec = codec or HandCodec()
    return codec.inspire12_to_wuji20(inspire12_from_qpos(qpos))
