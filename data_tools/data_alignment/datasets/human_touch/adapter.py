"""HumanTouch source adapter.

HumanTouch1/EgoPro, canonical HumanTouch2, and real-robot recordings share
one public dataset name but publish different MCAP topic layouts. Detection is
therefore topic-first; path names are only used to label the variant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..base import DatasetAdapter
from .reader import (
    read_egopro_reference,
    read_humantouch2_real_robot_reference,
    read_humantouch2_reference,
)
from ...readers import (
    humantouch2_block_index_from_path,
)
from ...schema import ReferenceFrameGeometry
from ...tactile.collector import EgoProTactileCollector


HUMANTOUCH_DISPLAY_NAME = "HumanTouch"
def _is_real_robot_path(path: Path) -> bool:
    return any(
        "real_robot" in part.lower() or part.lower().startswith("realrobot")
        for part in path.parts
    )


def _topic_set(path: Path) -> set[str]:
    """Read MCAP channel keys without decoding full message payloads."""
    if not path.is_file() or path.suffix.lower() != ".mcap":
        return set()
    try:
        from mcap.reader import make_reader
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError("HumanTouch MCAP detection requires the `mcap` package") from error
    topics: set[str] = set()
    with path.open("rb") as stream:
        reader = make_reader(stream)
        summary = reader.get_summary()
        if summary is not None:
            topics.update(channel.topic for channel in summary.channels.values())
        if topics:
            return topics
        # Some MCAPs do not carry summary indexes. A bounded scan is enough to
        # distinguish the two HumanTouch schemas and avoids reading video data.
        for _schema, channel, _message in reader.iter_messages():
            topics.add(channel.topic)
            if (
                any(topic.startswith("/human/hands/") for topic in topics)
                or "/manus/left" in topics
                or "/tracking/vive_tracker_manus" in topics
            ):
                break
    return topics


def detect_humantouch_variant(path: str | Path) -> str:
    """Detect the concrete HumanTouch transport from MCAP keys first."""
    source = Path(path)
    topics = _topic_set(source)
    if _is_real_robot_path(source):
        return "humantouch2_real_robot"
    if any(topic.startswith("/human/hands/") for topic in topics) or "/views/head/image" in topics:
        return "humantouch2"
    if (
        any(topic.startswith("/manus/") for topic in topics)
        or any(topic.startswith("/tracking/") for topic in topics)
        or any(topic.startswith("/ego_video/") for topic in topics)
        or any(topic.startswith("/hand_skeleton/") for topic in topics)
        or any(topic.startswith("/pose/") for topic in topics)
    ):
        # HumanTouch2 has a legacy transport layout that uses EgoPro-v2
        # topics, but it remains the HumanTouch2 source format.
        if any(part.lower() == "humantouch2" for part in source.parts):
            return "humantouch2"
        return "egopro"
    # A path hint is only a fallback for malformed/summary-less files.
    if any(part.lower() == "humantouch2" for part in source.parts):
        return "humantouch2"
    if any(part.lower() == "humantouch1" for part in source.parts):
        return "egopro"
    raise ValueError(f"Could not detect HumanTouch MCAP topic layout: {source}")


def _episode_level_mcap(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    if parts & {".convert_tmp", ".staged", "ego_video", "ego_camera_head", "ego_camera_left", "ego_camera_right"}:
        return False
    name = path.name.lower()
    if name in {"head.mcap", "left.mcap", "right.mcap", "ego_head.mcap", "ego_left.mcap", "ego_right.mcap"}:
        return False
    return name.startswith("data_raw_") or name.endswith(".annotated.mcap")


class HumanTouchAdapter(DatasetAdapter):
    canonical_name = "human_touch"
    display_name = HUMANTOUCH_DISPLAY_NAME
    def can_handle(self, path: str | Path) -> bool:
        source = Path(path)
        if source.is_file():
            return source.suffix.lower() == ".mcap"
        if not source.exists():
            return False
        return any(
            token in source.name.lower()
            for token in ("humantouch", "egopro", "mcap", "real_robot")
        ) or any(source.rglob("data_raw_*.mcap"))

    def discover(self, root: str | Path) -> list[Path]:
        source = Path(root)
        if source.is_file():
            if not _episode_level_mcap(source):
                raise ValueError(f"Not an episode-level HumanTouch MCAP: {source}")
            return [source]
        if not source.is_dir():
            raise FileNotFoundError(source)
        found = {
            path.resolve()
            for path in source.rglob("*.mcap")
            if _episode_level_mcap(path)
        }
        return sorted(found)

    def variant(self, path: str | Path) -> str:
        return detect_humantouch_variant(path)

    def read_reference(self, path: str | Path, **kwargs: Any) -> tuple[
        ReferenceFrameGeometry, EgoProTactileCollector
    ]:
        source = Path(path)
        variant = self.variant(source)
        topics = _topic_set(source)
        canonical_h2_layout = (
            variant == "humantouch2"
            and (
                any(topic.startswith("/human/hands/") for topic in topics)
                or "/views/head/image" in topics
            )
        )
        collector = kwargs.get("tactile_collector") or EgoProTactileCollector()
        instruction_map = kwargs.get("instruction_map")
        annotations_root = kwargs.get("annotations_root")
        resolve_instruction = bool(kwargs.get("resolve_instruction", True))
        if canonical_h2_layout:
            geometry = read_humantouch2_reference(
                source,
                tactile_collector=collector,
                instruction_map=instruction_map,
            )
        elif variant == "humantouch2_real_robot":
            geometry = read_humantouch2_real_robot_reference(
                source,
                tactile_collector=collector,
            )
        else:
            geometry = read_egopro_reference(
                source,
                tactile_collector=collector,
                annotations_root=annotations_root,
                resolve_instruction=resolve_instruction,
                source_dataset=(
                    "humantouch2" if variant == "humantouch2" else "egopro"
                ),
            )
            # Transport-compatible HumanTouch2 streams use the EgoPro video
            # writer while retaining HumanTouch/source_format metadata.
            if variant == "humantouch2" and not canonical_h2_layout:
                geometry.source = "egopro"
        geometry.metadata = {
            **geometry.metadata,
            "dataset_name": HUMANTOUCH_DISPLAY_NAME,
            "source_format": variant,
        }
        return geometry, collector

    def episode_metadata(
        self, path: str | Path, geometry: ReferenceFrameGeometry
    ) -> dict[str, Any]:
        source = Path(path)
        variant = str(geometry.metadata.get("source_format") or self.variant(source))
        metadata: dict[str, Any] = {
            "dataset_name": HUMANTOUCH_DISPLAY_NAME,
            "source_format": variant,
            "mcap_name": source.name,
            "mcap_stem": source.stem,
            "tactile_usable": True,
        }
        if variant.startswith("humantouch2"):
            block_index = str(
                geometry.metadata.get("block_index")
                or humantouch2_block_index_from_path(source)
                or ""
            ).strip().upper()
            metadata.update(
                {
                    "block_index": block_index,
                    "task_id": block_index,
                    "task_id_full": block_index,
                    "block_id": source.parent.name,
                    "capture_date": source.parent.name[:8],
                }
            )
        return metadata
