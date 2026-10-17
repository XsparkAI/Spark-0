"""EgoDex dataset adapter."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from ...schema import ReferenceFrameGeometry
from ..base import DatasetAdapter
from .reader import is_egodex_episode, read_egodex_reference

class EgoDexAdapter(DatasetAdapter):
    canonical_name = "egodex"
    display_name = "EgoDex"
    def can_handle(self, path: str | Path) -> bool:
        source = Path(path)
        if source.is_file():
            return is_egodex_episode(source)
        return source.is_dir() and ("egodex" in source.name.lower() or any(p.with_suffix(".mp4").is_file() for p in source.rglob("*.hdf5")))
    def discover(self, root: str | Path) -> list[Path]:
        source = Path(root)
        if source.is_file():
            return [source.resolve()] if is_egodex_episode(source) else []
        if not source.is_dir():
            raise FileNotFoundError(source)
        return sorted(p.resolve() for p in source.rglob("*.hdf5") if is_egodex_episode(p))
    def variant(self, path: str | Path) -> str:
        return "egodex"
    def read_reference(self, path: str | Path, **kwargs: Any) -> tuple[ReferenceFrameGeometry, None]:
        del kwargs
        geometry = read_egodex_reference(path)
        geometry.metadata = {**geometry.metadata, "dataset_name": self.display_name, "source_format": "egodex"}
        return geometry, None
    def episode_metadata(self, path: str | Path, geometry: ReferenceFrameGeometry) -> dict[str, Any]:
        source = Path(path)
        return {"dataset_name": self.display_name, "source_format": "egodex",
                "episode_id": source.stem, "task": geometry.metadata.get("task", source.parent.name),
                "mcap_name": source.name}

