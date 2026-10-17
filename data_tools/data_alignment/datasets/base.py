"""Common dataset adapter contract for camera-aligned Spark conversion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..schema import ReferenceFrameGeometry


class DatasetAdapter(ABC):
    """Small source adapter used by both the one-shot CLI and batch runner.

    Adapters stop at :class:`ReferenceFrameGeometry`. Resampling, camera
    alignment, Wuji IK, tactile/MANO writing, and HDF5 serialization remain in
    the shared pipeline so adding a dataset only requires a reader adapter.
    """

    canonical_name: str
    display_name: str

    @abstractmethod
    def can_handle(self, path: str | Path) -> bool:
        """Return whether ``path`` can belong to this dataset family."""

    @abstractmethod
    def discover(self, root: str | Path) -> list[Path]:
        """Discover episode-level inputs below ``root``."""

    @abstractmethod
    def read_reference(self, path: str | Path, **kwargs: Any) -> tuple[
        ReferenceFrameGeometry, Any
    ]:
        """Read one source into canonical reference geometry plus optional tactile data."""

    def variant(self, path: str | Path) -> str:
        """Return a source-format variant label for provenance."""
        return self.canonical_name

    def episode_metadata(
        self, path: str | Path, geometry: ReferenceFrameGeometry
    ) -> dict[str, Any]:
        """Return optional identity metadata consumed by the HDF5 writer."""
        del path
        return {
            "dataset_name": self.display_name,
            "source_format": geometry.metadata.get("source_format", self.canonical_name),
        }
