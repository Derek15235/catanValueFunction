"""Locked pyarrow schema for catan_vf Parquet I/O.

Column order:
  11 metadata columns (canonical, see field table below)
  + 92 feature columns (alphabetical order from data/feature_ordering.json)
  = 103 fields total.
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa

_FEATURE_ORDERING_PATH = Path(__file__).parent / "data" / "feature_ordering.json"

FEATURE_ORDERING: list[str] = json.loads(_FEATURE_ORDERING_PATH.read_text())

_COLOR_DICT = pa.dictionary(pa.int8(), pa.string())

_METADATA_FIELDS: list[pa.Field] = [
    pa.field("game_id", pa.string(), nullable=False),
    pa.field("snapshot_idx", pa.uint32(), nullable=False),
    pa.field("turn", pa.uint32(), nullable=False),
    pa.field("current_color", _COLOR_DICT, nullable=False),
    pa.field("pov_color", _COLOR_DICT, nullable=False),
    pa.field("max_vp", pa.uint8(), nullable=False),
    pa.field("label", pa.uint8(), nullable=False),
    pa.field("game_complete", pa.bool_(), nullable=False),
    pa.field("p1_bot", pa.string(), nullable=False),
    pa.field("p2_bot", pa.string(), nullable=False),
    pa.field("seed", pa.uint64(), nullable=False),
]

_FEATURE_FIELDS: list[pa.Field] = [
    pa.field(name, pa.float32(), nullable=False) for name in FEATURE_ORDERING
]

SCHEMA: pa.Schema = pa.schema(_METADATA_FIELDS + _FEATURE_FIELDS).with_metadata({
    b"catan_vf.schema_version": b"1",
    b"catan_vf.num_features": str(len(FEATURE_ORDERING)).encode(),
})

__all__ = ["SCHEMA", "FEATURE_ORDERING"]
