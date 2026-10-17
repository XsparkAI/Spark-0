#!/usr/bin/env python3
"""Rewrite a LeRobot dataset's parquet files with small row groups.

Why this exists: parquet's decode unit is the ROW GROUP, and LeRobot's writer
emits one row group per data file. For a dataset with tactile columns that is
fatal to random access — the egopro export packs ~110k rows and 1.1 GB into a
single group, so reading one frame's taxel cloud decodes the whole file
(measured 3.7 s per sample; the dataloader's one-frame reader is then useless).
Repacking into ~1k-row groups takes that to ~26 ms with no schema change: the
columns, the values, the row order and the file names are all identical, so
every stats cache, episode index and video path stays valid.

This is worth running on any dataset with large per-frame columns; a dataset of
plain state/action vectors is fast either way.

    python data_tools/repack_lerobot_row_groups.py data/lerobot/egopro_tactile_10h
    python data_tools/repack_lerobot_row_groups.py data/lerobot/egopro_tactile_10h --dry-run
    python data_tools/repack_lerobot_row_groups.py data/lerobot/egopro_tactile_10h --rows 2000

Each file is rewritten through a temporary file in the same directory and moved
into place only after it verifies (row count, schema and a sampled row match
the original), so an interrupted run leaves the dataset untouched.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_ROWS_PER_GROUP = 1000
# Rows per group above this are what makes random access slow.
REPACK_THRESHOLD_ROWS = 20000


def _sample_row_indices(num_rows: int, count: int = 5) -> list[int]:
    if num_rows <= count:
        return list(range(num_rows))
    step = max(1, num_rows // count)
    return [min(i * step, num_rows - 1) for i in range(count)]


def _verify(source: pa.Table, repacked: Path) -> None:
    """Check the rewritten file against the in-memory table it was written from.

    Sampled rows are read the way the dataloader will read them (one row group
    at a time), so this also proves the row-group addressing lines up.
    """
    handle = pq.ParquetFile(repacked)
    if handle.metadata.num_rows != source.num_rows:
        raise RuntimeError(
            f"{repacked.name}: {handle.metadata.num_rows} rows != {source.num_rows}"
        )
    if handle.schema_arrow != source.schema:
        raise RuntimeError(f"{repacked.name}: schema changed")

    group_starts = []
    cursor = 0
    for index in range(handle.metadata.num_row_groups):
        group_starts.append(cursor)
        cursor += handle.metadata.row_group(index).num_rows

    for row in _sample_row_indices(source.num_rows):
        group = max(i for i, start in enumerate(group_starts) if start <= row)
        offset = row - group_starts[group]
        actual = handle.read_row_group(group).slice(offset, 1)
        expected = source.slice(row, 1)
        for column in source.schema.names:
            if actual.column(column).to_pylist() != expected.column(column).to_pylist():
                raise RuntimeError(f"{repacked.name}: row {row} column {column} differs")


def repack_file(path: Path, rows_per_group: int, compression: str) -> tuple[int, int]:
    """Rewrite ``path`` in place. Returns ``(old_groups, new_groups)``."""
    old_groups = pq.ParquetFile(path).metadata.num_row_groups
    table = pq.read_table(path)
    tmp = path.with_suffix(path.suffix + ".repack.tmp")
    try:
        pq.write_table(
            table,
            tmp,
            row_group_size=rows_per_group,
            compression=compression,
        )
        _verify(table, tmp)
        del table
        new_groups = pq.ParquetFile(tmp).metadata.num_row_groups
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return old_groups, new_groups


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path, help="LeRobot dataset root (contains data/ and meta/)")
    parser.add_argument(
        "--rows",
        type=int,
        default=DEFAULT_ROWS_PER_GROUP,
        help=f"rows per row group (default {DEFAULT_ROWS_PER_GROUP})",
    )
    parser.add_argument(
        "--compression",
        default="zstd",
        help="parquet compression codec for the rewritten files (default zstd)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="repack every file, including ones already small enough",
    )
    parser.add_argument("--dry-run", action="store_true", help="report what would change")
    args = parser.parse_args()

    files = sorted((args.dataset / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        print(f"[repack] no parquet files under {args.dataset / 'data'}", file=sys.stderr)
        return 1
    if args.rows < 1:
        print("[repack] --rows must be >= 1", file=sys.stderr)
        return 1

    todo = []
    for path in files:
        metadata = pq.ParquetFile(path).metadata
        largest = max(
            metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)
        )
        needs = args.force or largest > max(REPACK_THRESHOLD_ROWS, args.rows)
        print(
            f"[repack] {path.relative_to(args.dataset)}: {metadata.num_rows} rows in "
            f"{metadata.num_row_groups} group(s), largest {largest} "
            f"-> {'REPACK' if needs else 'skip'}"
        )
        if needs:
            todo.append(path)

    if args.dry_run or not todo:
        print(f"[repack] {len(todo)} file(s) would be rewritten")
        return 0

    for path in todo:
        old_groups, new_groups = repack_file(path, args.rows, args.compression)
        print(
            f"[repack] {path.relative_to(args.dataset)}: "
            f"{old_groups} -> {new_groups} row groups"
        )
    print(f"[repack] rewrote {len(todo)} file(s); schema and row order unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
