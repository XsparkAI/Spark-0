"""EgoVLA benchmark dataset adapter."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from ...schema import ReferenceFrameGeometry
from ...schema import UnifiedHumanMotion
from ..base import DatasetAdapter
from ..common import source_motion_to_reference
from .reader import is_egovla_episode, read_egovla_source

class EgoVLABenchAdapter(DatasetAdapter):
    canonical_name = "egovla_bench"
    display_name = "EgoVLA Benchmark"

    def can_handle(self, path: str | Path) -> bool:
        source = Path(path)
        if source.is_file():
            return is_egovla_episode(source)
        if not source.is_dir():
            return False
        return any(
            is_egovla_episode(candidate)
            for candidate in source.rglob("*.hdf5")
        )

    def discover(self, root: str | Path) -> list[Path]:
        source = Path(root)
        if source.is_file():
            return [source.resolve()] if is_egovla_episode(source) else []
        if not source.is_dir():
            raise FileNotFoundError(source)
        return sorted(
            p.resolve() for p in source.rglob("*.hdf5") if is_egovla_episode(p)
        )

    def variant(self, path: str | Path) -> str:
        return "egovla_bench"

    def read_motion(self, path: str | Path, **kwargs: Any) -> UnifiedHumanMotion:
        return read_egovla_source(
            path, instruction=kwargs.get("instruction"),
            need_finger_chains=bool(kwargs.get("need_finger_chains", True))
        )

    def read_reference(
        self, path: str | Path, **kwargs: Any
    ) -> tuple[ReferenceFrameGeometry, None]:
        motion = self.read_motion(path, **kwargs)
        geometry = source_motion_to_reference(motion)
        geometry.metadata = {
            **geometry.metadata,
            "dataset_name": self.display_name,
            "source_format": self.variant(path),
            "source_motion": motion,
        }
        return geometry, None

    def episode_metadata(
        self, path: str | Path, geometry: ReferenceFrameGeometry
    ) -> dict[str, Any]:
        source = Path(path)
        return {
            "dataset_name": self.display_name,
            "source_format": self.variant(path),
            "episode_id": geometry.metadata.get("episode", source.stem),
            "task": geometry.metadata.get("task", geometry.instruction),
            "source_file": source.name,
        }
