#!/usr/bin/env python3
"""Batch conversion for all supported datasets with atomic resumable outputs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any


# Keep direct-file invocation and existing imports of helpers working.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "data_tools"

from .data_alignment.commands.batch import parse_args
from .data_alignment.commands.batch import _validate_output


EGOPRO_SUFFIX = ".annotated.mcap"
EGOPRO_RAW_SUFFIX = ".mcap"
OPENAOE_MARKER = "ego_process/ego_hands_reconstruction/hands.npz"


def _is_openaoe_sample(path: Path) -> bool:
    """Mirror data_tools.data_alignment.readers.is_openaoe_sample without importing it."""
    return path.is_dir() and (path / OPENAOE_MARKER).is_file()


def _dataset_name_for_source(source: Path, requested: str | None) -> str:
    """Resolve the canonical adapter name (topic/marker probe for ``auto``)."""
    from .data_alignment.datasets.registry import detect_dataset

    return detect_dataset(source, requested)


@dataclass
class RunningEpisode:
    source: Path
    output: Path
    partial: Path
    log_path: Path
    log_stream: IO[str]
    process: subprocess.Popen[str]
    started_at: float
    dataset_name: str


def _output_name(source: Path) -> str:
    if _is_openaoe_sample(source):
        return f"{source.name}.hdf5"
    if source.name == "proprio_stats.h5":
        return f"{source.parent.name}.hdf5"
    if source.suffix.lower() in {".hdf5", ".h5"}:
        return f"{source.stem}.hdf5"
    if source.name.endswith(EGOPRO_SUFFIX):
        return f"{source.name[:-len(EGOPRO_SUFFIX)]}.hdf5"
    if source.name.endswith(EGOPRO_RAW_SUFFIX):
        return f"{source.name[:-len(EGOPRO_RAW_SUFFIX)]}.hdf5"
    # EgoVerse zarr stores and EgoVLA benchmark episodes are directories.
    # Preserve the episode directory name while removing only .zarr.
    if source.is_dir():
        stem = source.name[:-5] if source.name.lower().endswith(".zarr") else source.name
        return f"{stem}.hdf5"
    raise ValueError(f"Unexpected dataset filename: {source.name}")


def _discover_sources(input_dir: Path, requested: str | None = "auto") -> list[Path]:
    """Discover episode inputs through the selected canonical adapter.

    HumanTouch discovery deliberately excludes nested camera MCAPs and temporary
    conversion directories; OpenAOE discovery uses its hands.npz marker.
    """
    from .data_alignment.datasets.registry import canonical_dataset_name, get_adapter

    if input_dir.is_file():
        return [input_dir]
    canonical = canonical_dataset_name(requested)
    if canonical == "auto":
        return _discover_single_family(input_dir)
    return get_adapter(canonical).discover(input_dir)


def _discover_single_family(input_dir: Path) -> list[Path]:
    """Run every adapter's episode discovery and refuse mixed-family roots.

    EgoVLA discovery opens each HDF5, so it only probes HDF5 files that no
    other family claimed (an EgoDex root holds hundreds of thousands).
    """
    from .data_alignment.datasets.egovla_bench.reader import is_egovla_episode
    from .data_alignment.datasets.registry import ADAPTERS, EGOVLA_BENCH

    found: dict[str, list[Path]] = {}
    for adapter in ADAPTERS:
        if adapter.canonical_name == EGOVLA_BENCH:
            continue
        episodes = adapter.discover(input_dir)
        if episodes:
            found[adapter.canonical_name] = episodes
    claimed = {path.resolve() for episodes in found.values() for path in episodes}
    egovla = sorted(
        path.resolve()
        for path in input_dir.rglob("*.hdf5")
        if path.resolve() not in claimed and is_egovla_episode(path)
    )
    if egovla:
        found[EGOVLA_BENCH] = egovla
    if len(found) > 1:
        summary = ", ".join(
            f"{name} ({len(episodes)} episodes)" for name, episodes in found.items()
        )
        raise ValueError(
            f"--dataset-name auto found several dataset families under {input_dir}: "
            f"{summary}. Pass --dataset-name or split the input directory by family."
        )
    return next(iter(found.values()), [])


def _load_input_manifest(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Input manifest must be a JSON object: {path}")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"Input manifest has no 'files' list: {path}")
    return payload


def _manifest_file_entries(payload: dict[str, object]) -> list[dict[str, Any]]:
    raw = payload.get("files")
    if not isinstance(raw, list):
        return []
    entries: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            entries.append({"path": item})
        elif isinstance(item, dict) and item.get("path"):
            entries.append(dict(item))
    return entries


def _json_source_files(
    payload: dict[str, object], limit: int | None = None
) -> tuple[list[Path], dict[Path, dict[str, Any]]]:
    object_mode = any(isinstance(item, dict) for item in payload.get("files", []))
    extras: dict[Path, dict[str, Any]] = {}
    sources: list[Path] = []
    seen: set[Path] = set()
    for item in _manifest_file_entries(payload):
        path = Path(item["path"]).resolve()
        if not path.exists():
            print(f"BATCH_SKIP {path.name} reason=missing_source", flush=True)
            continue
        if path in seen:
            print(f"BATCH_SKIP {path.name} reason=duplicate_source", flush=True)
            continue
        seen.add(path)
        sources.append(path)
        extras[path] = item
        if limit is not None and len(sources) >= limit:
            break
    if not object_mode:
        sources = sorted(sources)
        extras = {path: extras[path] for path in sources}
    return sources, extras


def _source_files(
    input_dir: Path,
    requested: str | None = "auto",
    limit: int | None = None,
) -> list[Path]:
    sources = sorted(
        (
            path
            for path in _discover_sources(input_dir, requested)
            if path == input_dir
            or not any(
                part.startswith(".") for part in path.relative_to(input_dir).parts
            )
        ),
        key=lambda path: (
            "" if path == input_dir else path.relative_to(input_dir).as_posix()
        ),
    )
    return sources if limit is None else sources[:limit]


def _output_path(
    source: Path,
    input_dir: Path,
    output_dir: Path,
    extra: dict[str, Any] | None = None,
) -> Path:
    if extra and extra.get("output_relative"):
        return output_dir / str(extra["output_relative"])
    relative = source.relative_to(input_dir)
    return output_dir / relative.parent / _output_name(source)


def _existing_review_path(extra: dict[str, Any] | None) -> Path | None:
    if not extra:
        return None
    candidates: list[object] = [extra.get("annotation_json")]
    qc = extra.get("tactile_qc")
    if isinstance(qc, dict):
        candidates.append(qc.get("annotation_json"))
    seen: set[Path] = set()
    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        path = Path(candidate)
        if path in seen:
            continue
        seen.add(path)
        if path.is_file():
            return path
    return None


def _load_task_name_map(args: argparse.Namespace) -> dict[str, str]:
    raw = args.task_name_map or os.environ.get("HDF5_TASK_NAME_MAP")
    if not raw:
        return {}
    path = Path(raw)
    if not path.is_file():
        print(f"BATCH_WARN task_name_map missing: {path}", flush=True)
        return {}
    from .data_alignment.readers import load_task_name_map

    mapping = load_task_name_map(path)
    print(
        json.dumps({"task_name_map": str(path), "n_keys": len(mapping)}, ensure_ascii=False),
        flush=True,
    )
    return mapping


def _load_instruction_map(args: argparse.Namespace) -> dict[str, str]:
    raw = args.instruction_map
    if not isinstance(raw, str) or not raw.strip():
        return {}
    path = Path(raw)
    if not path.is_file():
        print(f"BATCH_WARN instruction_map missing: {path}", flush=True)
        return {}
    from .data_alignment.readers import load_block_instruction_map

    mapping = load_block_instruction_map(path)
    print(
        json.dumps(
            {"instruction_map": str(path), "n_keys": len(mapping)},
            ensure_ascii=False,
        ),
        flush=True,
    )
    return mapping


def _fill_episode_instruction(
    extra: dict[str, Any] | None,
    args: argparse.Namespace,
    mapping: dict[str, str],
    *,
    source_path: Path,
    dataset_name: str,
    instruction_mapping: dict[str, str],
) -> dict[str, Any]:
    from .data_alignment.datasets.registry import canonical_dataset_name, get_adapter

    extra = dict(extra or {})
    canonical = canonical_dataset_name(dataset_name)
    adapter = get_adapter(canonical)
    variant = adapter.variant(source_path)
    extra["dataset_name"] = canonical
    extra["source_format"] = variant
    if canonical == "openaoe":
        if args.instruction is not None:
            extra["instruction"] = str(args.instruction)
            extra["instruction_source"] = "cli"
        return extra
    if variant.startswith("humantouch2"):
        from .data_alignment.readers import (
            humantouch2_block_index_from_path,
            lookup_block_instruction,
        )

        task_id = str(
            extra.get("task_id")
            or extra.get("task_id_full")
            or extra.get("block_index")
            or extra.get("blockIndex")
            or humantouch2_block_index_from_path(source_path)
            or ""
        ).strip().upper()
        if task_id:
            extra["task_id"] = task_id
            extra.setdefault("task_id_full", task_id)
            # Preserve the legacy HumanTouch2 blockIndex metadata field: in
            # this dataset it carries the same A001/B002-style task code.
            extra["block_index"] = task_id
        extra.setdefault("block_id", source_path.parent.name)
        extra.setdefault("capture_date", source_path.parent.name[:8])
        extra.setdefault("mcap_name", source_path.name)
        extra.setdefault("mcap_stem", source_path.stem)
        extra.setdefault("tactile_usable", True)
        if args.instruction is not None:
            extra["instruction"] = str(args.instruction)
            extra["instruction_source"] = "cli"
        elif isinstance(extra.get("instruction"), str) and extra["instruction"].strip():
            extra.setdefault("instruction_source", "manifest")
        else:
            name = lookup_block_instruction(task_id, instruction_mapping)
            if name:
                extra["instruction"] = name
                extra["instruction_source"] = "task_id_instruction_json"
            else:
                extra.setdefault("instruction", "")
                extra["instruction_source"] = "pending_instruction_json"
        return extra
    from .data_alignment.readers import (
        egopro_task_code_from_text,
        find_egopro_semantic_review,
        load_review_json,
        lookup_task_name,
        short_task_id,
    )

    review = _existing_review_path(extra)
    if review is None and variant == "egopro":
        try:
            review = find_egopro_semantic_review(
                source_path, annotations_root=getattr(args, "annotations_root", None)
            )
        except ValueError:
            # A custom source tree may have no sibling annotations directory.
            # Explicit/manifest instructions and task maps still work there.
            review = None
    payload = load_review_json(review) if review is not None else None
    if review is not None:
        extra["annotation_json"] = str(review)
    if variant == "egopro":
        review_task_id = str((payload or {}).get("task_id") or "").strip()
        if review_task_id:
            extra.setdefault("task_id_full", review_task_id)
        task_id = short_task_id(extra.get("task_id_full")) or egopro_task_code_from_text(
            str(source_path)
        )
        if task_id:
            extra.setdefault("task_id", task_id)
    if args.instruction is not None:
        extra["instruction"] = str(args.instruction)
        extra["instruction_source"] = "cli"
        return extra
    if isinstance(extra.get("instruction"), str) and extra["instruction"].strip():
        extra.setdefault("instruction_source", "manifest")
        return extra

    name = lookup_task_name(
        extra.get("task_id_full") if isinstance(extra.get("task_id_full"), str) else None,
        extra.get("task_id") if isinstance(extra.get("task_id"), str) else None,
        mapping=mapping,
    )
    source = "task_id_map"
    if not name and payload is not None:
        name = str(payload.get("task_name") or "").strip() or None
        if name:
            source = "annotation_json"
    if name:
        extra["instruction"] = name
        extra["instruction_source"] = source
    return extra


def _skip_patterns(args: argparse.Namespace) -> set[str]:
    patterns = {value.strip() for value in args.skip if value.strip()}
    if args.skip_file is not None:
        for line in Path(args.skip_file).read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if value and not value.startswith("#"):
                patterns.add(value)
    return patterns


def _is_skipped(source: Path, input_dir: Path, skip_patterns: set[str]) -> bool:
    if not skip_patterns:
        return False
    relative = source.relative_to(input_dir).as_posix()
    candidates = {relative, source.name, source.stem}
    return any(pattern in skip_patterns for pattern in candidates)


def _command(
    package_root: Path,
    source: Path,
    camera_config: Path,
    partial: Path,
    args: argparse.Namespace,
    extra: dict[str, Any] | None = None,
) -> list[str]:
    dataset_name = str((extra or {}).get("dataset_name") or args.dataset_name)
    from .data_alignment.datasets.registry import canonical_dataset_name
    canonical = canonical_dataset_name(dataset_name)
    if canonical in {"fourier_actionnet", "agibot_world_beta", "egovla_bench"}:
        command = [
            sys.executable, "-m", "data_tools.data_alignment.commands.retarget",
            "retarget-source-robot", "--input", str(source),
            "--output", str(partial), "--dataset-name", canonical,
            "--fps", str(args.fps), "--action-steps", str(args.action_steps),
            "--max-rms-error-m", str(args.max_rms_error_m),
            "--ik-solver", args.ik_solver, "--ik-stride", str(args.ik_stride),
            "--ik-max-iterations", str(args.ik_max_iterations),
        ]
        if args.frames is not None:
            command.extend(["--frames", str(args.frames)])
        instruction = args.instruction
        if instruction is None and extra:
            value = extra.get("instruction")
            if isinstance(value, str) and value.strip():
                instruction = value.strip()
        if instruction is not None:
            command.extend(["--instruction", instruction])
        return command
    command = [
        sys.executable, "-m", "data_tools.data_alignment.commands.retarget",
        "retarget-camera-aligned", "--input", str(source),
        "--camera-config", str(camera_config), "--output", str(partial),
        "--dataset-name", canonical, "--fps", str(args.fps),
        "--action-steps", str(args.action_steps),
        "--max-rms-error-m", str(args.max_rms_error_m),
        "--ik-solver", args.ik_solver, "--ik-stride", str(args.ik_stride),
        "--ik-max-iterations", str(args.ik_max_iterations),
    ]
    if args.frames is not None:
        command.extend(["--frames", str(args.frames)])
    instruction = args.instruction
    if instruction is None and extra:
        value = extra.get("instruction")
        if isinstance(value, str) and value.strip():
            instruction = value.strip()
    if instruction is not None:
        command.extend(["--instruction", instruction])
    if isinstance(args.instruction_map, str) and args.instruction_map.strip():
        command.extend(["--instruction-map", args.instruction_map])
    if args.annotations_root:
        command.extend(["--annotations-root", args.annotations_root])
    if args.no_trim_invalid:
        command.append("--no-trim-invalid")
    if args.allow_missing_jpeg:
        command.append("--allow-missing-jpeg")
    if args.mano_model is not None:
        command.extend(["--mano-model", args.mano_model])
    if args.mano_assets is not None:
        command.extend(["--mano-assets", args.mano_assets])
    elif canonical == "human_touch":
        raise ValueError(
            "HumanTouch EgoPro/HumanTouch2 episodes require --mano-assets "
            "(official mano_v1_2 directory)"
        )
    if _is_openaoe_sample(source):
        command.extend(["--mano-left-convention", args.mano_left_convention])
        command.append(
            "--mano-flat-hand-mean"
            if args.mano_flat_hand_mean
            else "--no-mano-flat-hand-mean"
        )
    meta_path = _write_episode_meta_json(partial, extra)
    if meta_path is not None:
        command.extend(["--episode-meta-json", str(meta_path)])
    return command


def _write_episode_meta_json(partial: Path, extra: dict[str, Any] | None) -> Path | None:
    if not extra:
        return None
    path = Path(str(partial) + ".episode_meta.json")
    path.write_text(json.dumps(extra, ensure_ascii=False), encoding="utf-8")
    return path


def _unlink_episode_meta_json(partial: Path) -> None:
    Path(str(partial) + ".episode_meta.json").unlink(missing_ok=True)


def _failed_sources(report_path: Path) -> set[str]:
    if not report_path.exists():
        return set()
    failed: set[str] = set()
    try:
        with report_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue
                relative = record.get("source_relative")
                if isinstance(relative, str) and relative:
                    if record.get("status") == "failed":
                        failed.add(relative)
                    elif record.get("status") in {"ok", "completed"}:
                        failed.discard(relative)
    except OSError as exc:
        print(f"BATCH_WARN cannot read {report_path}: {exc}", flush=True)
        return set()
    return failed


def _append_report(report_path: Path, record: dict[str, object]) -> None:
    with report_path.open("a", encoding="utf-8") as report:
        report.write(json.dumps(record, ensure_ascii=False) + "\n")


def _record_episode_failure(
    report_path: Path,
    episode: RunningEpisode,
    input_dir: Path,
    output_dir: Path,
    elapsed_s: float,
    reason: str,
    detail: str,
) -> None:
    _append_report(
        report_path,
        {
            "source": str(episode.source),
            "source_relative": episode.source.relative_to(input_dir).as_posix(),
            "output_relative": episode.output.relative_to(output_dir).as_posix(),
            "status": "failed",
            "dataset_name": episode.dataset_name,
            "reason": reason,
            "detail": detail,
            "log_path": str(episode.log_path),
            "elapsed_s": elapsed_s,
        },
    )
    print(
        f"BATCH_FAIL {episode.source.name} reason={reason} log={episode.log_path}",
        flush=True,
    )


def _worker_environment(threads_per_worker: int) -> dict[str, str]:
    if threads_per_worker < 1:
        raise ValueError("--threads-per-worker must be positive")
    environment = os.environ.copy()
    thread_count = str(threads_per_worker)
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OPENCV_FOR_THREADS_NUM",
    ):
        environment[name] = thread_count
    return environment


def _terminate_all(running: dict[int, RunningEpisode]) -> None:
    for episode in running.values():
        if episode.process.poll() is None:
            episode.process.terminate()
    deadline = time.monotonic() + 10.0
    for episode in running.values():
        remaining = max(0.0, deadline - time.monotonic())
        try:
            episode.process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            episode.process.kill()
            episode.process.wait()
        episode.log_stream.close()
        episode.partial.unlink(missing_ok=True)
        _unlink_episode_meta_json(episode.partial)


def _start_episode(
    source: Path,
    output: Path,
    output_dir: Path,
    log_dir: Path,
    package_root: Path,
    camera_config: Path,
    args: argparse.Namespace,
    extra: dict[str, Any] | None,
    environment: dict[str, str],
) -> RunningEpisode:
    """Launch one worker, cleaning temporary files if launch fails."""
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.stem}.partial.hdf5")
    partial.unlink(missing_ok=True)
    relative_output = output.relative_to(output_dir)
    log_path = log_dir / relative_output.parent / f"{output.stem}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_stream = log_path.open("w", encoding="utf-8")
    try:
        command = _command(
            package_root, source, camera_config, partial, args, extra
        )
        process = subprocess.Popen(
            command,
            cwd=package_root,
            env=environment,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except BaseException:
        log_stream.close()
        partial.unlink(missing_ok=True)
        _unlink_episode_meta_json(partial)
        raise
    return RunningEpisode(
        source=source,
        output=output,
        partial=partial,
        log_path=log_path,
        log_stream=log_stream,
        process=process,
        started_at=time.monotonic(),
        dataset_name=str((extra or {}).get("dataset_name") or args.dataset_name),
    )


def run_batch(args: argparse.Namespace) -> int:
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.threads_per_worker < 1:
        raise ValueError("--threads-per-worker must be positive")
    if args.ik_stride < 1 or args.ik_max_iterations < 1:
        raise ValueError("--ik-stride and --ik-max-iterations must be positive")
    if args.frames is not None and args.frames <= args.action_steps:
        raise ValueError("--frames must be greater than --action-steps")
    script_path = Path(__file__).resolve()
    # This file lives at Spark-0/data_tools/; package root is one level up.
    package_root = script_path.parents[1]
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    if args.input_json is not None:
        manifest = _load_input_manifest(args.input_json)
        root = manifest.get("source_root")
        input_dir = (
            Path(root).resolve()
            if isinstance(root, str) and Path(root).is_dir()
            else None
        )
    else:
        input_dir = Path(args.input_dir).resolve() if args.input_dir else None
    output_dir = Path(args.output_dir).resolve()
    camera_config = (
        Path(args.camera_config).resolve()
        if args.camera_config
        else package_root / "tools" / "data_alignment" / "assets" / "spark_camera_alignment.json"
    )
    checks: list[tuple[Path | None, str]] = [
        (input_dir, "input directory"),
        (camera_config, "camera config"),
        (package_root, "Spark package root"),
    ]
    for path, label in checks:
        if path is None:
            continue
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    extras: dict[Path, dict[str, Any]] = {}
    if args.input_json is not None:
        sources, extras = _json_source_files(manifest, args.limit)
        if input_dir is None and sources:
            # commonpath of a single file is that file, not its directory.
            input_dir = Path(os.path.commonpath([str(path.parent) for path in sources]))
    else:
        sources = _source_files(input_dir, args.dataset_name, args.limit)
    if not sources:
        raise ValueError(
            "No input episodes found (recursive dataset episodes (MCAP, HDF5, Zarr, or OpenAoE directories) "
            "or --input-json manifest files)"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        report_path = output_dir / "batch_report.jsonl"
    skip_patterns = _skip_patterns(args)
    previously_failed: set[str] = set()
    if args.overwrite:
        print(
            f"BATCH_INFO overwrite=1 skip previous-failure scan report={report_path}",
            flush=True,
        )
    else:
        previously_failed = _failed_sources(report_path)
    task_name_map = _load_task_name_map(args)
    instruction_map = _load_instruction_map(args)

    pending: list[tuple[Path, Path]] = []
    dataset_counts: dict[str, int] = {}
    skipped = 0
    skipped_explicit = 0
    skipped_failed = 0
    skipped_no_instruction = 0
    for source in sources:
        manifest_extra = extras.get(source) or {}
        dataset_name = _dataset_name_for_source(
            source,
            str(manifest_extra.get("dataset_name") or args.dataset_name),
        )
        from .data_alignment.datasets.registry import display_dataset_name

        dataset_label = display_dataset_name(dataset_name)
        dataset_counts[dataset_label] = dataset_counts.get(dataset_label, 0) + 1
        relative = source.relative_to(input_dir).as_posix()
        if _is_skipped(source, input_dir, skip_patterns):
            skipped_explicit += 1
            _append_report(
                report_path,
                {
                    "source": str(source),
                    "source_relative": relative,
                    "status": "skipped",
                    "reason": "explicit_skip",
                },
            )
            print(f"BATCH_SKIP {source.name} reason=explicit_skip", flush=True)
            continue
        if relative in previously_failed and not args.retry_failed:
            skipped_failed += 1
            print(
                f"BATCH_SKIP {source.name} reason=previous_failure "
                "(use --retry-failed to retry)",
                flush=True,
            )
            continue
        extra = _fill_episode_instruction(
            extras.get(source),
            args,
            task_name_map,
            source_path=source,
            dataset_name=dataset_name,
            instruction_mapping=instruction_map,
        )
        extras[source] = extra
        if (
            dataset_name == "human_touch"
            and str(extra.get("source_format")) == "egopro"
            and not extra.get("instruction")
        ):
            skipped_no_instruction += 1
            _append_report(
                report_path,
                {
                    "source": str(source),
                    "source_relative": relative,
                    "status": "skipped",
                    "reason": "no_instruction",
                    "task_id": extra.get("task_id"),
                    "task_id_full": extra.get("task_id_full"),
                    "annotation_json": extra.get("annotation_json"),
                },
            )
            print(
                f"BATCH_SKIP {source.name} reason=no_instruction "
                f"task_id={extra.get('task_id_full') or extra.get('task_id')}",
                flush=True,
            )
            continue
        output = _output_path(source, input_dir, output_dir, extra)
        if output.exists() and not args.overwrite:
            try:
                _validate_output(output)
            except Exception as error:
                raise ValueError(
                    f"Existing output is invalid; inspect or use --overwrite: {output}"
                ) from error
            skipped += 1
            continue
        pending.append((source, output))

    print(
        json.dumps(
            {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "selected": len(sources),
                "pending": len(pending),
                "skipped_valid": skipped,
                "skipped_explicit": skipped_explicit,
                "skipped_failed": skipped_failed,
                "skipped_no_instruction": skipped_no_instruction,
                "datasets": dataset_counts,
                "workers": args.workers,
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.dry_run:
        return 0

    running: dict[int, RunningEpisode] = {}
    completed = 0
    failed = 0
    pending_index = 0
    environment = _worker_environment(args.threads_per_worker)
    try:
        while pending_index < len(pending) or running:
            while pending_index < len(pending) and len(running) < args.workers:
                source, output = pending[pending_index]
                pending_index += 1
                episode = _start_episode(
                    source,
                    output,
                    output_dir,
                    log_dir,
                    package_root,
                    camera_config,
                    args,
                    extras.get(source),
                    environment,
                )
                running[episode.process.pid] = episode
                print(f"BATCH_START {source.name} pid={episode.process.pid}", flush=True)

            finished_pid = next(
                (pid for pid, item in running.items() if item.process.poll() is not None),
                None,
            )
            if finished_pid is None:
                time.sleep(0.5)
                continue
            episode = running.pop(finished_pid)
            return_code = episode.process.returncode
            episode.log_stream.close()
            elapsed_s = time.monotonic() - episode.started_at
            _unlink_episode_meta_json(episode.partial)
            if return_code != 0:
                episode.partial.unlink(missing_ok=True)
                _record_episode_failure(
                    report_path,
                    episode,
                    input_dir,
                    output_dir,
                    elapsed_s,
                    reason="process_exit",
                    detail=f"exit_code={return_code}",
                )
                failed += 1
                if args.stop_on_error:
                    raise RuntimeError(
                        f"{episode.source.name} failed with exit code {return_code}; "
                        f"see {episode.log_path}"
                    )
                continue
            try:
                validation = _validate_output(episode.partial)
            except Exception as error:
                episode.partial.unlink(missing_ok=True)
                _record_episode_failure(
                    report_path,
                    episode,
                    input_dir,
                    output_dir,
                    elapsed_s,
                    reason="validation_failed",
                    detail=str(error),
                )
                failed += 1
                if args.stop_on_error:
                    raise
                continue
            os.replace(episode.partial, episode.output)
            completed += 1
            record = {
                "source": str(episode.source),
                "source_relative": episode.source.relative_to(input_dir).as_posix(),
                "output": str(episode.output),
                "output_relative": episode.output.relative_to(output_dir).as_posix(),
                "status": "ok",
                "dataset_name": episode.dataset_name,
                "elapsed_s": elapsed_s,
                **validation,
            }
            _append_report(report_path, record)
            print(
                f"BATCH_PROGRESS completed={completed} failed={failed} skipped={skipped} "
                f"remaining={len(pending) - completed - failed} episode={episode.output.name} "
                f"elapsed_s={elapsed_s:.1f}",
                flush=True,
            )
    except BaseException:
        _terminate_all(running)
        raise

    print(
        f"BATCH_COMPLETE completed={completed} failed={failed} skipped={skipped} "
        f"skipped_explicit={skipped_explicit} skipped_failed={skipped_failed} "
        f"total={len(sources)}",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(run_batch(parse_args()))
