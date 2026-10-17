# Local Spark-0 addition (NOT upstream starVLA): dual-stream tactile encoder
# (region_pressure + taxel_pressure_3d) → fixed tokens for Action-DiT fusion.
from starVLA.model.modules.tactile.action_fusion import (
    PatchToTemb,
    TactileActionFusion,
    flatten_region_patch,
    get_tactile_action_fusion,
    resolve_cross_attention_layers,
)
from starVLA.model.modules.tactile.tactile_encoder import (
    TactileEncoder,
    encode_with_slow_refresh,
    get_tactile_encoder,
    slow_refresh_control_period,
    tactile_infer_cache_id,
)
from starVLA.model.modules.tactile.future_contact import (
    DEFAULT_FUTURE_HORIZON,
    FUTURE_CONTACT_METRIC_KEYS,
    FutureContactHead,
    get_future_contact,
)
from starVLA.model.modules.tactile.temporal import TemporalConvEncoder

__all__ = [
    "DEFAULT_FUTURE_HORIZON",
    "FUTURE_CONTACT_METRIC_KEYS",
    "FutureContactHead",
    "PatchToTemb",
    "TactileActionFusion",
    "TactileEncoder",
    "TemporalConvEncoder",
    "encode_with_slow_refresh",
    "flatten_region_patch",
    "get_future_contact",
    "get_tactile_action_fusion",
    "get_tactile_encoder",
    "resolve_cross_attention_layers",
    "slow_refresh_control_period",
    "tactile_infer_cache_id",
]
