"""Fourier ActionNet dataset adapter."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from ...schema import ReferenceFrameGeometry
from ...schema import UnifiedHumanMotion
from ..base import DatasetAdapter
from ..common import source_motion_to_reference
from .reader import is_fourier_episode, read_fourier_source

class FourierActionNetAdapter(DatasetAdapter):
    canonical_name = "fourier_actionnet"
    display_name = "Fourier ActionNet"
    def can_handle(self, path: str | Path) -> bool:
        source = Path(path)
        if source.is_file():
            return is_fourier_episode(source)
        return source.is_dir() and ("fourier" in source.name.lower() or any(p.name == "rgb.mp4" and p.parent.name == "top" for p in source.rglob("rgb.mp4")))
    def discover(self, root: str | Path) -> list[Path]:
        source = Path(root)
        if source.is_file():
            return [source.resolve()] if is_fourier_episode(source) else []
        if not source.is_dir():
            raise FileNotFoundError(source)
        return sorted(p.resolve() for p in source.rglob("*.hdf5") if is_fourier_episode(p))
    def variant(self, path: str | Path) -> str:
        return "fourier_actionnet"
    def read_motion(self, path: str | Path, **kwargs: Any) -> UnifiedHumanMotion:
        return read_fourier_source(path)

    def read_reference(self, path: str | Path, **kwargs: Any) -> tuple[ReferenceFrameGeometry, None]:
        geometry = source_motion_to_reference(self.read_motion(path, **kwargs))
        geometry.metadata = {**geometry.metadata, "dataset_name": self.display_name, "source_format": "fourier_actionnet"}
        return geometry, None
    def episode_metadata(self, path: str | Path, geometry: ReferenceFrameGeometry) -> dict[str, Any]:
        source = Path(path)
        return {"dataset_name": self.display_name, "source_format": "fourier_actionnet",
                "episode_id": source.stem, "task": geometry.instruction, "mcap_name": source.name}

