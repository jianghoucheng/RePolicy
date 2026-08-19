#!/usr/bin/env python3
"""Build the six-benchmark evaluation split from normalized trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from repolicy.build_dataset import write_parquet
from repolicy.policy_context import (
    build_full_library_context,
    build_prompt,
    build_rl_extra_info,
    build_target,
    load_policy_library,
)


DATA_FILES = [
    "atbench.jsonl",
    "r_judge.jsonl",
    "open_agent_safety.jsonl",
    "assebench.jsonl",
    "hintbench.jsonl",
    "agent_hazard.jsonl",
]

BENCHMARK_BY_FILE = {
    "atbench.jsonl": "atbench",
    "r_judge.jsonl": "r_judge",
    "open_agent_safety.jsonl": "open_agent_safety",
    "assebench.jsonl": "assebench",
    "hintbench.jsonl": "hintbench",
    "agent_hazard.jsonl": "agent_hazard",
}


def read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_normalized(data_dir: Path) -> dict[str, dict]:
    base_dir = data_dir / "normalized" if (data_dir / "normalized").exists() else data_dir
    out = {}
    for name in DATA_FILES:
        for row in read_jsonl(base_dir / name):
            row = dict(row)
            row["source_benchmark"] = BENCHMARK_BY_FILE[name]
            out[row["id"]] = row
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--policy-library", default="repolicy/assets/policies/policy_library.json")
    parser.add_argument("--coverage-map", default="repolicy/assets/policies/policy_coverage_map.jsonl")
    parser.add_argument("--output-dir", default="data/eval")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--write-parquet", action="store_true")
    args = parser.parse_args()

    normalized = load_normalized(Path(args.data_dir))
    coverage_rows = list(read_jsonl(Path(args.coverage_map)))
    samples = []
    for cov in coverage_rows:
        src = normalized.get(cov["sample_id"])
        if not src:
            continue
        samples.append(
            {
                "sample_id": cov["sample_id"],
                "source_benchmark": src["source_benchmark"],
                "trajectory": src["contents"],
                "selected_policies": cov["selected_policies"],
                "label": cov["label"],
                "rationale": {
                    "summary": cov.get("rationale", ""),
                    "violated_clauses": [
                        {"policy_id": cid.split(".")[0], "clause_id": cid, "text": ""}
                        for cid in cov.get("violated_clause_ids", [])
                    ],
                },
            }
        )
    if args.limit:
        samples = samples[: args.limit]
    policy_library = load_policy_library(args.policy_library)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rl_rows_for_parquet = []
    with (out / "benchmark_val.jsonl").open("w", encoding="utf-8") as f:
        for idx, sample in enumerate(samples):
            context = build_full_library_context(sample, policy_library)
            target = build_target(sample, context)
            extra_info = build_rl_extra_info(sample, context, idx)
            extra_info["sample_id"] = sample["sample_id"]
            extra_info["source_benchmark"] = sample["source_benchmark"]
            rl_row = {
                "data_source": f"agent_guard_benchmark/{sample['source_benchmark']}",
                "agent_name": "tool_agent",
                "prompt": build_prompt(sample, context),
                "ability": "agent_safety_guard",
                "reward_model": {"style": "rule", "ground_truth": target},
                "extra_info": extra_info,
            }
            f.write(json.dumps(rl_row, ensure_ascii=False) + "\n")
            if args.write_parquet:
                rl_rows_for_parquet.append(rl_row)
    if args.write_parquet:
        write_parquet(out / "benchmark_val.parquet", rl_rows_for_parquet)
    print(
        json.dumps(
            {
                "benchmark_eval_count": len(samples),
                "output_dir": str(out),
                "parquet_written": args.write_parquet,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
