"""Build the PolicyTraj-20K SFT and RL splits from annotated trajectories."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from repolicy.policy_context import (
    build_context,
    build_prompt,
    build_rl_extra_info,
    build_sft_messages,
    build_target,
    load_policy_library,
    policy_lookup_tool_schema,
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_parquet(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


class JsonlParquetWriter:
    def __init__(self, jsonl_path: Path, parquet_path: Path | None = None, batch_size: int = 128):
        self.jsonl_path = jsonl_path
        self.parquet_path = parquet_path
        self.batch_size = batch_size
        self.batch: list[dict[str, Any]] = []
        self.parquet_writer: pq.ParquetWriter | None = None
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.jsonl_path.open("w", encoding="utf-8")

    def write(self, row: dict[str, Any]) -> None:
        self.file.write(json.dumps(row, ensure_ascii=False) + "\n")
        if self.parquet_path is not None:
            self.batch.append(row)
            if len(self.batch) >= self.batch_size:
                self.flush()

    def flush(self) -> None:
        if not self.batch or self.parquet_path is None:
            return
        self.parquet_path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(self.batch)
        if self.parquet_writer is None:
            self.parquet_writer = pq.ParquetWriter(self.parquet_path, table.schema)
        table = table.cast(self.parquet_writer.schema)
        self.parquet_writer.write_table(table)
        self.batch.clear()

    def close(self) -> None:
        self.flush()
        self.file.close()
        if self.parquet_writer is not None:
            self.parquet_writer.close()


def make_one_record(
    sample: dict[str, Any], policy_library: list[dict[str, Any]], seed: int, idx: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = random.Random(seed + idx * 9973)
    context = build_context(sample, policy_library, rng)
    prompt = build_prompt(sample, context)
    target = build_target(sample, context)
    sft_row = {
        "messages": build_sft_messages(sample, context, target),
        "tools": [policy_lookup_tool_schema()],
        "enable_thinking": False,
        "extra_info": {"index": idx},
    }
    extra_info = build_rl_extra_info(sample, context, idx)
    rl_row = {
        "data_source": "agent_guard",
        "agent_name": "tool_agent",
        "prompt": prompt,
        "ability": "agent_safety_guard",
        "reward_model": {"style": "rule", "ground_truth": target},
        "extra_info": extra_info,
    }
    return sft_row, rl_row


def make_records(
    samples: list[dict[str, Any]], policy_library: list[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sft_rows = []
    rl_rows = []
    for idx, sample in enumerate(samples):
        sft_row, rl_row = make_one_record(sample, policy_library, seed, idx)
        sft_rows.append(sft_row)
        rl_rows.append(rl_row)
    return sft_rows, rl_rows


def split_indices(
    count: int, sft_train_size: int, sft_val_size: int, rl_val_size: int, seed: int
) -> dict[int, str]:
    indices = list(range(count))
    rng = random.Random(seed)
    rng.shuffle(indices)

    assignment: dict[int, str] = {}
    pos = 0
    for idx in indices[pos : pos + min(sft_val_size, count - pos)]:
        assignment[idx] = "sft_val"
    pos += min(sft_val_size, count - pos)

    for idx in indices[pos : pos + min(sft_train_size, count - pos)]:
        assignment[idx] = "sft_train"
    pos += min(sft_train_size, count - pos)

    remaining = indices[pos:]
    for idx in remaining[: min(rl_val_size, len(remaining))]:
        assignment[idx] = "rl_val"
    for idx in remaining[min(rl_val_size, len(remaining)) :]:
        assignment[idx] = "rl_train"
    return assignment


def build_and_write_streaming(
    samples: list[dict[str, Any]],
    policy_library: list[dict[str, Any]],
    seed: int,
    sft_dir: Path,
    rl_dir: Path,
    sft_train_size: int,
    sft_val_size: int,
    rl_val_size: int,
    write_rl_parquet: bool = False,
) -> dict[str, Any]:
    """Write data without materializing the full RL prompt set in memory."""

    assignment = split_indices(len(samples), sft_train_size, sft_val_size, rl_val_size, seed)
    rl_train_rows_for_parquet: list[dict[str, Any]] = []
    rl_val_rows_for_parquet: list[dict[str, Any]] = []

    sft_dir.mkdir(parents=True, exist_ok=True)
    rl_dir.mkdir(parents=True, exist_ok=True)
    counts = {"sft_train": 0, "sft_val": 0, "rl_train": 0, "rl_val": 0}

    sft_train_writer = JsonlParquetWriter(sft_dir / "train.jsonl", sft_dir / "train.parquet")
    sft_val_writer = JsonlParquetWriter(sft_dir / "val.jsonl", sft_dir / "val.parquet")
    try:
        rl_train_f = (rl_dir / "train.jsonl").open("w", encoding="utf-8")
        rl_val_f = (rl_dir / "val.jsonl").open("w", encoding="utf-8")
        try:
            for idx, sample in enumerate(samples):
                sft_row, rl_row = make_one_record(sample, policy_library, seed, idx)
                split = assignment[idx]
                counts[split] += 1
                if split == "sft_train":
                    sft_train_writer.write(sft_row)
                elif split == "sft_val":
                    sft_val_writer.write(sft_row)
                elif split == "rl_val":
                    rl_val_f.write(json.dumps(rl_row, ensure_ascii=False) + "\n")
                    if write_rl_parquet:
                        rl_val_rows_for_parquet.append(rl_row)
                else:
                    rl_train_f.write(json.dumps(rl_row, ensure_ascii=False) + "\n")
                    if write_rl_parquet:
                        rl_train_rows_for_parquet.append(rl_row)
        finally:
            rl_train_f.close()
            rl_val_f.close()
    finally:
        sft_train_writer.close()
        sft_val_writer.close()

    if write_rl_parquet:
        write_parquet(rl_dir / "train.parquet", rl_train_rows_for_parquet)
        write_parquet(rl_dir / "val.parquet", rl_val_rows_for_parquet)

    return {
        "input_count": len(samples),
        "sft_train": counts["sft_train"],
        "sft_val": counts["sft_val"],
        "rl_train": counts["rl_train"],
        "rl_val": counts["rl_val"],
        "sft_dir": str(sft_dir),
        "rl_dir": str(rl_dir),
        "rl_parquet_written": write_rl_parquet,
    }


def split_rows(rows: list[dict[str, Any]], train_size: int, val_size: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    val = rows[: min(val_size, len(rows))]
    remaining = rows[len(val) :]
    train = remaining[: min(train_size, len(remaining))] if train_size > 0 else remaining
    rest = remaining[len(train) :]
    return train, val, rest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", default="data/generated/train.jsonl")
    parser.add_argument("--policy-library", default="repolicy/assets/policies/policy_library.json")
    parser.add_argument("--sft-dir", default="data/sft")
    parser.add_argument("--rl-dir", default="data/rl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sft-train-size", type=int, default=5000)
    parser.add_argument("--sft-val-size", type=int, default=500)
    parser.add_argument("--rl-val-size", type=int, default=500)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--write-rl-parquet", action="store_true")
    args = parser.parse_args()

    samples = read_jsonl(args.input_jsonl)
    if args.limit:
        samples = samples[: args.limit]
    policy_library = load_policy_library(args.policy_library)
    sft_dir = Path(args.sft_dir)
    rl_dir = Path(args.rl_dir)
    summary = build_and_write_streaming(
        samples=samples,
        policy_library=policy_library,
        seed=args.seed,
        sft_dir=sft_dir,
        rl_dir=rl_dir,
        sft_train_size=args.sft_train_size,
        sft_val_size=args.sft_val_size,
        rl_val_size=args.rl_val_size,
        write_rl_parquet=args.write_rl_parquet,
    )
    (sft_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (rl_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
