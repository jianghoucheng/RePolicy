#!/usr/bin/env python3
"""Convert the RL/eval JSONL splits to parquet.

Two reasons the published dataset uses parquet throughout:

* A repo mixing ``.jsonl`` and ``.parquet`` makes ``load_dataset`` infer a single
  parquet builder for every config, so the JSONL configs fail to load.
* Parquet enables the Hub dataset viewer.

verl reads either format, so training is unaffected either way.

Dynamic-key maps inside ``extra_info`` are stored as JSON strings, keeping the
arrow schema identical across rows (the local->global id maps have different
keys in every example). ``tools_kwargs`` stays a real dict because verl indexes
into it during rollout.

    python -m repolicy.to_parquet
"""

from __future__ import annotations

import argparse
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq


# extra_info fields whose keys vary per example; stored as JSON strings so the
# arrow schema stays identical across rows.
#
# tools_kwargs is deliberately NOT in this list: verl indexes it as a dict
# (`tools_kwargs.get(tool_name)` in tool_agent_loop), so stringifying it would
# break rollout. Its nested policy_library has a fixed schema, so arrow handles
# it directly.
JSON_STRING_FIELDS = [
    "local_to_global_policy",
    "local_to_global_clause",
    "global_to_local_policy",
    "global_to_local_clause",
]

DEFAULT_PAIRS = [
    ("data/rl/train.jsonl", "data/rl/train.parquet"),
    ("data/rl/val.jsonl", "data/rl/val.parquet"),
    ("data/eval/benchmark_val.jsonl", "data/eval/benchmark_val.parquet"),
]


def normalize(row: dict) -> dict:
    extra = row.get("extra_info")
    if isinstance(extra, dict):
        for key in JSON_STRING_FIELDS:
            if isinstance(extra.get(key), (dict, list)):
                extra[key] = json.dumps(extra[key], ensure_ascii=False)
    return row


def convert(src: str, dst: str, batch_size: int = 500) -> int:
    writer: pq.ParquetWriter | None = None
    rows: list[dict] = []
    total = 0

    def write(rows: list[dict], writer: pq.ParquetWriter | None) -> pq.ParquetWriter:
        table = pa.Table.from_pylist(rows)
        if writer is None:
            writer = pq.ParquetWriter(dst, table.schema, compression="zstd")
        writer.write_table(table.cast(writer.schema))
        return writer

    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(normalize(json.loads(line)))
            total += 1
            if len(rows) >= batch_size:
                writer = write(rows, writer)
                rows = []
    if rows:
        writer = write(rows, writer)
    if writer is not None:
        writer.close()
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    for src, dst in DEFAULT_PAIRS:
        if not os.path.exists(src):
            print(f"skip (missing): {src}")
            continue
        n = convert(src, dst, args.batch_size)
        print(
            f"{dst}: {n} rows, {os.path.getsize(dst) / 1e6:.1f} MB "
            f"(from {os.path.getsize(src) / 1e6:.1f} MB jsonl)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
