"""AgiBot World (Beta hand) dataset adapter."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from ...schema import ReferenceFrameGeometry
from ...schema import UnifiedHumanMotion
from ..base import DatasetAdapter
from ..common import source_motion_to_reference
from .reader import is_agibot_episode, read_agibot_source

class AgiBotWorldAdapter(DatasetAdapter):
    canonical_name = "agibot_world_beta"
    display_name = "AgiBot World Beta"
    def can_handle(self, path: str | Path) -> bool:
        source = Path(path)
        if source.is_file():
            return is_agibot_episode(source)
        return source.is_dir() and ("agibot" in source.name.lower() or (source / "proprio_stats").is_dir())
    def discover(self, root: str | Path) -> list[Path]:
        source = Path(root)
        if source.is_file():
            return [source.resolve()] if is_agibot_episode(source) else []
        if not source.is_dir():
            raise FileNotFoundError(source)
        return sorted(p.resolve() for p in source.rglob("proprio_stats.h5") if is_agibot_episode(p))
    def variant(self, path: str | Path) -> str:
        return "agibot_world_beta"
    def read_motion(self, path: str | Path, **kwargs: Any) -> UnifiedHumanMotion:
        return read_agibot_source(path)

    def read_reference(self, path: str | Path, **kwargs: Any) -> tuple[ReferenceFrameGeometry, None]:
        geometry = source_motion_to_reference(self.read_motion(path, **kwargs))
        geometry.metadata = {**geometry.metadata, "dataset_name": self.display_name, "source_format": "agibot_world_beta"}
        return geometry, None
    def episode_metadata(self, path: str | Path, geometry: ReferenceFrameGeometry) -> dict[str, Any]:
        source = Path(path)
        return {"dataset_name": self.display_name, "source_format": "agibot_world_beta",
                "episode_id": geometry.metadata.get("episode", source.parent.name),
                "task": geometry.instruction, "mcap_name": source.name}

