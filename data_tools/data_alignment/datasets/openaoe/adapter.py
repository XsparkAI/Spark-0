"""OpenAoE dataset adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..base import DatasetAdapter
from .reader import is_openaoe_sample, read_openaoe_reference
from ...schema import ReferenceFrameGeometry


class OpenAOEAdapter(DatasetAdapter):
    canonical_name = "openaoe"
    display_name = "OpenAOE"
    marker = Path("ego_process/ego_hands_reconstruction/hands.npz")

    def can_handle(self, path: str | Path) -> bool:
        source = Path(path)
        if is_openaoe_sample(source):
            return True
        if not source.is_dir():
            return False
        # Dataset roots contain one or more sample directories. Discovery does
        # the authoritative marker scan; this cheap name probe prevents an
        # automatic HumanTouch walk through unrelated directories.
        return "openaoe" in source.name.lower() or "open_aoe" in source.name.lower()

    def discover(self, root: str | Path) -> list[Path]:
        source = Path(root)
        if is_openaoe_sample(source):
            return [source.resolve()]
        if not source.is_dir():
            raise FileNotFoundError(source)
        found = {
            marker.parents[2].resolve()
            for marker in source.rglob("hands.npz")
            if marker.parent.name == "ego_hands_reconstruction"
            and marker.parent.parent.name == "ego_process"
        }
        return sorted(found)

    def read_reference(self, path: str | Path, **kwargs: Any) -> tuple[
        ReferenceFrameGeometry, None
    ]:
        geometry = read_openaoe_reference(
            path,
            mano_model=kwargs.get("mano_model"),
            flat_hand_mean=bool(kwargs.get("flat_hand_mean", True)),
            left_convention=str(kwargs.get("left_convention", "mirror_model")),
        )
        geometry.metadata = {
            **geometry.metadata,
            "dataset_name": self.display_name,
            "source_format": "openaoe",
        }
        return geometry, None
