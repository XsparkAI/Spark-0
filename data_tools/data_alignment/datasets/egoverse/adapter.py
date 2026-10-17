"""EgoVerse human-hand dataset adapter."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from ...schema import ReferenceFrameGeometry
from ..base import DatasetAdapter
from .reader import is_egoverse_episode, is_lightwheel_episode, read_egoverse_reference

class EgoVerseAdapter(DatasetAdapter):
    canonical_name = "egoverse"
    display_name = "EgoVerse"

    def can_handle(self, path: str | Path) -> bool:
        source = Path(path)
        if is_egoverse_episode(source) or is_lightwheel_episode(source):
            return True
        if source.is_dir():
            if source.name.lower() in {"egoverse", "lightwheel"} or "egoverse" in source.name.lower():
                return True
            return any(is_egoverse_episode(p) or is_lightwheel_episode(p) for p in source.rglob("*.zarr"))
        return False

    def discover(self, root: str | Path) -> list[Path]:
        source = Path(root)
        if source.is_file():
            return [source.resolve()] if self.can_handle(source) else []
        if not source.is_dir():
            raise FileNotFoundError(source)
        # Zarr stores are episode directories. Avoid Path.rglob, which would
        # descend into every chunk under each store and is prohibitively slow.
        candidates: list[Path] = []
        if is_lightwheel_episode(source):
            candidates.append(source)
            return [source.resolve()]
        direct = source if source.name.endswith(".zarr") else None
        if direct is not None and is_egoverse_episode(direct):
            candidates.append(direct)
        else:
            for group in source.iterdir():
                if not group.is_dir():
                    continue
                if group.name.endswith(".zarr"):
                    if is_egoverse_episode(group):
                        candidates.append(group)
                    continue
                for episode in group.iterdir():
                    if episode.is_dir() and episode.name.endswith(".zarr") and is_egoverse_episode(episode):
                        candidates.append(episode)
                    elif episode.is_dir() and is_lightwheel_episode(episode):
                        candidates.append(episode)
        return sorted(set(p.resolve() for p in candidates))

    def variant(self, path: str | Path) -> str:
        return "egoverse_lightwheel" if is_lightwheel_episode(path) else "egoverse_zarr"

    def read_reference(self, path: str | Path, **kwargs: Any) -> tuple[ReferenceFrameGeometry, None]:
        del kwargs
        geometry = read_egoverse_reference(path)
        geometry.metadata = {
            **geometry.metadata,
            "dataset_name": self.display_name,
            "source_format": self.variant(path),
        }
        return geometry, None

    def episode_metadata(self, path: str | Path, geometry: ReferenceFrameGeometry) -> dict[str, Any]:
        source = Path(path)
        return {
            "dataset_name": self.display_name,
            "source_format": self.variant(source),
            "episode_id": source.stem,
            "task": geometry.instruction,
            "mcap_name": source.name,
        }