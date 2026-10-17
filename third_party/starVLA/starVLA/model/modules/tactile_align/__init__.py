# Local Spark-0 addition (NOT upstream starVLA): train-only contact-region
# → Qwen visual-token alignment (orthogonal to DA3 and tactile fusion).
from starVLA.model.modules.tactile_align.contact_align import (
    ContactRegionAlignment,
    get_tactile_align,
)
from starVLA.model.modules.tactile_align.region_targets import (
    aggregate_points_to_regions,
    targets_from_saliency,
)
from starVLA.model.modules.tactile_align.token_map import (
    prepare_heatmap_contact_map,
    saliency_to_contact_map,
    student_grid_hw,
    uv_to_token_index,
)

__all__ = [
    "ContactRegionAlignment",
    "get_tactile_align",
    "aggregate_points_to_regions",
    "targets_from_saliency",
    "prepare_heatmap_contact_map",
    "saliency_to_contact_map",
    "student_grid_hw",
    "uv_to_token_index",
]
