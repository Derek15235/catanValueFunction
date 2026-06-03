"""flatten.py: Flatten per-game parquet files into a single dataset file.

Reads data/raw/<game_id>.parquet files, concatenates them into a single
parquet file at out_path, and generates splits.json (80/10/10 by game_id,
reproducible).

Public API:
    SPLIT_SEED              int  — fixed seed for deterministic splits
    build_splits(game_ids)  list[str] -> dict[str,list[str]] — 80/10/10 by game_id
    assert_no_overlap(splits) — raises AssertionError if any id is in two splits
    flatten_run(...)        — orchestrates the full flatten pipeline; returns summary
"""

import json
import random as _rnd
import shutil
import sys
from pathlib import Path

import pyarrow.dataset as ds
import pyarrow.parquet as pq

from schema import SCHEMA

# Module-level constants

SPLIT_SEED = 0xCA7A_5EED


# Partition game_ids into train/val/test lists (80/10/10).
def build_splits(game_ids: list[str]):
    sorted_ids = sorted(game_ids)
    rng = _rnd.Random(SPLIT_SEED)
    rng.shuffle(sorted_ids)

    n = len(sorted_ids)
    n_val = max(1, round(n * 0.10)) if n >= 2 else 0
    n_test = max(1, round(n * 0.10)) if n >= 2 else 0

    if n == 100:
        n_val = 10
        n_test = 10

    train = sorted_ids[: n - n_val - n_test]
    val = sorted_ids[n - n_val - n_test : n - n_test]
    test = sorted_ids[n - n_test :]

    return {"train": train, "val": val, "test": test}


# Raise AssertionError if any game_id appears in more than one split.
def assert_no_overlap(splits: dict[str, list[str]]):
    seen: set[str] = set()
    for split_name, ids in splits.items():
        overlap = seen & set(ids)
        assert not overlap, (
            f"game_ids appear in multiple splits: {overlap} (found in '{split_name}')"
        )
        seen |= set(ids)


# Flatten per-game parquet files into a single output parquet file.
# AssertionError if integrity checks fail.
def flatten_run(
    raw_dir: Path,
    out_path: Path,
    data_dir: Path | None = None,
):
    raw_dir = Path(raw_dir)
    out_path = Path(out_path)
    if data_dir is None:
        data_dir = out_path.parent

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Read raw dataset (validates against SCHEMA on read)
    parquet_files = list(raw_dir.glob("*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"no parquet files under {raw_dir}")

    raw_dataset = ds.dataset(str(raw_dir), schema=SCHEMA, format="parquet")
    table = raw_dataset.to_table()
    game_ids_col = table.column("game_id").to_pylist()

    # If a directory exists at out_path (legacy partitioned layout), remove it
    # so pq.write_table can write a single file at that path.
    if out_path.is_dir():
        shutil.rmtree(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write single parquet file
    pq.write_table(table, str(out_path), compression="snappy")

    # Build splits and write splits.json
    unique_ids = sorted(set(game_ids_col))
    splits = build_splits(unique_ids)
    assert_no_overlap(splits)

    splits_path = data_dir / "splits.json"
    splits_path.write_text(
        json.dumps(splits, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Compute totals and extras
    total_games = len(unique_ids)
    total_snapshots = table.num_rows

    # truncation_rate: fraction of games that did NOT complete within TURNS_LIMIT
    game_complete_col = table.column("game_complete").to_pylist()
    game_complete_map: dict[str, bool] = {}
    for gid, gc in zip(game_ids_col, game_complete_col):
        game_complete_map[gid] = game_complete_map.get(gid, False) or bool(gc)

    n_complete_games = sum(1 for v in game_complete_map.values() if v)
    truncation_rate = (
        (total_games - n_complete_games) / total_games if total_games else 0.0
    )

    # per_pairing_snapshot_counts: dict["{p1}|{p2}", int]
    p1_col = table.column("p1_bot").to_pylist()
    p2_col = table.column("p2_bot").to_pylist()
    per_pairing: dict[str, int] = {}
    for p1, p2 in zip(p1_col, p2_col):
        key = f"{p1}|{p2}"
        per_pairing[key] = per_pairing.get(key, 0) + 1

    n_failed_games = 0  # collect.py excludes failed games (no parquet written)

    # Integrity assertions
    tmp_survivors = list(raw_dir.glob("*.tmp"))
    assert not tmp_survivors, f".tmp survivors in {raw_dir}: {tmp_survivors}"

    # Output schema equals SCHEMA
    out_schema = pq.read_schema(str(out_path))
    assert out_schema.equals(SCHEMA, check_metadata=False), (
        f"output schema drifted from SCHEMA.\n"
        f"Got:      {out_schema}\n"
        f"Expected: {SCHEMA}"
    )

    # No duplicates within a split; no overlap across splits
    seen: set[str] = set()
    for split_name, ids in splits.items():
        assert len(ids) == len(set(ids)), f"Duplicate game_ids within split '{split_name}'"
        overlap = seen & set(ids)
        assert not overlap, f"game_ids overlap between splits: {overlap}"
        seen |= set(ids)

    assert seen == set(unique_ids), (
        f"splits do not cover all game_ids.\n"
        f"Missing: {set(unique_ids) - seen}\n"
        f"Extra:   {seen - set(unique_ids)}"
    )

    return {
        "total_games": total_games,
        "total_snapshots": total_snapshots,
        "truncation_rate": truncation_rate,
        "n_failed_games": n_failed_games,
        "out_path": str(out_path),
        "splits_path": str(splits_path),
    }


# CLI entry point
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Flatten raw per-game parquet files into a single dataset file.")
    # Directory of per-game parquet files (default: data/raw)
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("data/raw")
    )
    # Output parquet file (default: data/snapshots.parquet)
    parser.add_argument(
        "--out-file", type=Path, default=Path("data/snapshots.parquet")
    )
    # Where to write splits.json (default: out_file.parent)
    parser.add_argument(
        "--data-dir", type=Path, default=None
    )
    args = parser.parse_args()
    result = flatten_run(
        raw_dir=args.raw_dir,
        out_path=args.out_file,
        data_dir=args.data_dir,
    )
    print(json.dumps(result, indent=2))
