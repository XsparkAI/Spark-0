"""Dataset adapter registry for the unified retarget pipeline."""
from __future__ import annotations

from pathlib import Path

from .base import DatasetAdapter
from .human_touch.adapter import HumanTouchAdapter
from .openaoe.adapter import OpenAOEAdapter
from .egodex.adapter import EgoDexAdapter
from .fourier_actionnet.adapter import FourierActionNetAdapter
from .agibot_world.adapter import AgiBotWorldAdapter
from .egovla_bench.adapter import EgoVLABenchAdapter
from .egoverse.adapter import EgoVerseAdapter

HUMAN_TOUCH = "human_touch"
OPENAOE = "openaoe"
EGODEX = "egodex"
FOURIER_ACTIONNET = "fourier_actionnet"
AGIBOT_WORLD_BETA = "agibot_world_beta"
EGOVLA_BENCH = "egovla_bench"
EGOVERSE = "egoverse"
AUTO = "auto"

ADAPTERS: tuple[DatasetAdapter, ...] = (
    HumanTouchAdapter(), OpenAOEAdapter(), EgoDexAdapter(),
    FourierActionNetAdapter(), AgiBotWorldAdapter(),
    EgoVLABenchAdapter(), EgoVerseAdapter(),
)

def canonical_dataset_name(value: str | None) -> str:
    normalized = str(value or AUTO).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"", AUTO}:
        return AUTO
    valid = {HUMAN_TOUCH, OPENAOE, EGODEX, FOURIER_ACTIONNET,
             AGIBOT_WORLD_BETA, EGOVLA_BENCH, EGOVERSE}
    if normalized not in valid:
        raise ValueError(
            f"Unknown dataset name {value!r}; choose human_touch, openaoe, "
            "egodex, fourier_actionnet, agibot_world_beta, egovla_bench, or egoverse"
        )
    return normalized

def display_dataset_name(value: str | None) -> str:
    canonical = canonical_dataset_name(value)
    if canonical == AUTO:
        return "auto"
    return get_adapter(canonical).display_name

def get_adapter(value: str | None) -> DatasetAdapter:
    canonical = canonical_dataset_name(value)
    if canonical == AUTO:
        raise ValueError("Dataset adapter is ambiguous for --dataset-name auto")
    for adapter in ADAPTERS:
        if adapter.canonical_name == canonical:
            return adapter
    raise AssertionError(canonical)

def detect_dataset(path: str | Path, requested: str | None = AUTO) -> str:
    canonical = canonical_dataset_name(requested)
    source = Path(path)
    if canonical != AUTO:
        return canonical
    if source.is_dir():
        lowered = source.name.lower()
        for name, tokens in (
            (FOURIER_ACTIONNET, ("fourier", "actionnet")),
            (AGIBOT_WORLD_BETA, ("agibot",)),
            (EGOVLA_BENCH, ("egovla", "benchmark")),
            (EGOVERSE, ("egoverse",)),
            (EGODEX, ("egodex",)),
        ):
            if any(token in lowered for token in tokens):
                return name
    for adapter in ADAPTERS:
        if adapter.can_handle(source):
            return adapter.canonical_name
    raise ValueError(f"Cannot detect dataset adapter for source: {source}")
