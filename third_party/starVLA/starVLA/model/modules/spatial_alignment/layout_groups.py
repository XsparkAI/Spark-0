"""Split a (possibly mixed) image batch into homogeneous layouts.

Qwen packs every sample's cameras into one flat ``image_grid_thw``
``[sum_i n_views_i, 3]`` in sample order. Rectangular gathers
(``view(B, V, 3)`` + ``hidden[visual].view(B, expected, C)``) need one
``V`` / ``expected`` for the whole tensor, so mixed head-only + 3-view
batches cannot share a gather. Grouping here lets DA3 / token prune /
tactile_align gather each layout separately. The training batch itself
is unchanged.

Trainable projectors are deliberately NOT run per group — see
``DA3SpatialAlignment.align_segments`` (and the matching tactile_align
``forward_groups``) for why the projector forward count must stay
rank-invariant under ZeRO-3.

Importable with torch only (no DA3 / Accelerate / VLM).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch

LayoutGroup = Tuple[List[int], List, torch.Tensor]


def parse_align_view_indices(
    view_indices,
    *,
    field: str,
) -> Optional[List[int]]:
    """``null`` → ``None``; otherwise a non-empty list of unique packed indices.

    Packed indices are non-negative. A negative value (e.g. ``[-1]``) would
    make every layout group look out-of-range and silently zero the loss.
    Duplicates would supervise the same camera twice.
    """
    if view_indices is None:
        return None
    parsed = [int(i) for i in view_indices]
    if not parsed:
        raise ValueError(f"{field} must be null or a non-empty list.")
    if any(index < 0 for index in parsed):
        raise ValueError(
            f"{field} must be non-negative packed view indices, got {parsed}."
        )
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{field} must not contain duplicates, got {parsed}.")
    return parsed


def packed_sample_grids(
    n_views_per_sample: Sequence[int],
    grid_thw: torch.Tensor,
    *,
    caller: str = "packed_sample_grids",
) -> List[torch.Tensor]:
    """Split flat ``image_grid_thw`` into one ``[n_views_i, 3]`` per sample."""
    if grid_thw is None:
        raise ValueError(f"{caller} requires image_grid_thw.")
    if grid_thw.ndim != 2 or grid_thw.shape[-1] != 3:
        raise ValueError(
            f"image_grid_thw must be [num_images, 3], got shape {tuple(grid_thw.shape)}."
        )
    n_per = [int(n) for n in n_views_per_sample]
    if any(n <= 0 for n in n_per):
        raise ValueError(f"{caller} cannot group a sample with zero packed images.")
    total = int(sum(n_per))
    if total != int(grid_thw.shape[0]):
        raise ValueError(
            f"image_grid_thw has {int(grid_thw.shape[0])} row(s) but the batch "
            f"packs {total} image(s) ({n_per})."
        )
    sample_grids: List[torch.Tensor] = []
    offset = 0
    for n_views in n_per:
        sample_grids.append(grid_thw[offset : offset + n_views])
        offset += n_views
    return sample_grids


def group_indices_by_image_layout(
    n_views_per_sample: Sequence[int],
    grid_thw: torch.Tensor,
    *,
    caller: str = "group_indices_by_image_layout",
) -> List[Tuple[List[int], torch.Tensor]]:
    """Group sample indices that share one packed camera layout.

    A layout key is ``(n_views, flattened per-view grid_thw)``. First-seen
    order is preserved. Returns ``(sample_indices, group_grid_thw)``.
    """
    sample_grids = packed_sample_grids(
        n_views_per_sample, grid_thw, caller=caller
    )
    n_per = [int(n) for n in n_views_per_sample]
    groups: dict = {}
    order: List[tuple] = []
    for index, (n_views, grid) in enumerate(zip(n_per, sample_grids)):
        key = (n_views, tuple(int(v) for v in grid.reshape(-1).tolist()))
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(index)

    out: List[Tuple[List[int], torch.Tensor]] = []
    for key in order:
        indices = groups[key]
        group_grid = torch.cat([sample_grids[i] for i in indices], dim=0)
        out.append((indices, group_grid))
    return out


def group_samples_by_image_layout(
    batch_images: Sequence[Sequence],
    grid_thw: torch.Tensor,
) -> List[LayoutGroup]:
    """Group samples that share one packed camera layout.

    A layout key is ``(n_views, flattened per-view grid_thw)`` — the exact
    geometry the rectangular gather needs, which is stricter than the
    ``view_mask`` alone (two samples with equal masks but different
    ``obs_image_size`` grids must not share a gather). First-seen order is
    preserved.

    Returns ``(sample_indices, group_images, group_grid_thw)`` for each
    layout. ``group_grid_thw`` concatenates those samples' rows in the
    same order as ``sample_indices``.
    """
    n_per = [len(images) for images in batch_images]
    out: List[LayoutGroup] = []
    for indices, group_grid in group_indices_by_image_layout(
        n_per, grid_thw, caller="group_samples_by_image_layout"
    ):
        group_images = [batch_images[i] for i in indices]
        out.append((indices, group_images, group_grid))
    return out


def gather_student_views(
    hidden_state: torch.Tensor,
    input_ids: torch.Tensor,
    grid_thw: torch.Tensor,
    *,
    image_token_id: int,
    merge_size: int,
) -> Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
    """Visual tokens of ONE homogeneous-layout group, split per camera.

    Token geometry is derived at runtime, so any camera count /
    ``obs_image_size`` works, including cameras that differ from each other
    in resolution or aspect ratio:
      - visual token positions   <- ``input_ids == image_token_id``
      - per-image token grid     <- ``grid_thw / merge_size``
    Qwen lays each image's merged visual tokens out in raster order (and
    images in message order), and a boolean gather preserves that order, so
    splitting by per-view token count recovers one tensor per camera.

    Returns ``([B, P_v, C] per view], [(gh, gw) per view])``.
    """
    batch_size = hidden_state.shape[0]
    if grid_thw.shape[0] % batch_size != 0:
        raise ValueError(
            f"image count {grid_thw.shape[0]} is not a multiple of batch size {batch_size}."
        )
    views_per_sample = grid_thw.shape[0] // batch_size

    # Cameras may differ from each other, but every SAMPLE in this group must
    # present the same camera layout — otherwise the per-view split below
    # would silently mix cameras across the group.
    grid_thw = grid_thw.view(batch_size, views_per_sample, 3)
    if not torch.all(grid_thw == grid_thw[0]):
        raise ValueError(
            f"spatial_alignment expects every sample to share one camera layout, got {grid_thw.tolist()}."
        )

    student_grids: List[Tuple[int, int]] = []
    for view_idx in range(views_per_sample):
        t, h, w = (int(v) for v in grid_thw[0, view_idx])
        if t != 1:
            # t > 1 means temporal (video) tokens; the teacher only ever sees
            # one frame per view, so there is no correspondence.
            raise ValueError(
                f"spatial_alignment does not support temporal image grids (view {view_idx} has t={t}); "
                "pass frames as separate views and set teacher_view_mode=independent."
            )
        student_grids.append((h // merge_size, w // merge_size))

    tokens_per_view = [grid_h * grid_w for grid_h, grid_w in student_grids]
    expected = sum(tokens_per_view)
    visual_mask = input_ids == image_token_id  # [B, L]
    counts = visual_mask.sum(dim=1)
    if not torch.all(counts == expected):
        raise ValueError(
            f"Expected {expected} visual tokens per sample "
            f"({views_per_sample} views, {tokens_per_view} tokens each), got {counts.tolist()}."
        )
    student_tokens = hidden_state[visual_mask].view(
        batch_size, expected, hidden_state.shape[-1]
    )
    return list(torch.split(student_tokens, tokens_per_view, dim=1)), student_grids


def normalized_segment_weights(
    group_shapes: Sequence[Tuple[int, int]]
) -> List[float]:
    """Per-``(group, view)`` weights for the flattened segment list.

    ``group_shapes`` is ``[(n_samples, n_views), ...]`` in segment order.
    The weights make the total

        sum_g (n_g / N) * (1 / V_g) * sum_v cos[g][v]

    i.e. each group keeps DA3's "mean per view, then across views"
    reduction and groups are combined by SAMPLE count, so every sample
    carries the same weight regardless of how many cameras it packs.
    Sums to 1; empty input gives an empty list.
    """
    if not group_shapes:
        return []
    for n_samples, n_views in group_shapes:
        if int(n_samples) <= 0 or int(n_views) <= 0:
            raise ValueError(
                f"normalized_segment_weights needs positive (samples, views), got {list(group_shapes)}."
            )
    total = float(sum(int(n) for n, _ in group_shapes))
    weights: List[float] = []
    for n_samples, n_views in group_shapes:
        weights.extend([int(n_samples) / (total * int(n_views))] * int(n_views))
    return weights
