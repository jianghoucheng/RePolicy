#!/usr/bin/env bash
# Upload PolicyTraj-20K to the Hugging Face Hub.
#
#   huggingface-cli login          # once, needs a write token
#   bash repolicy/scripts/upload_data.sh
#
# Uploads data/{sft,rl,eval} plus the policy library and the dataset card.
# This publishes to a public repo by default; set PRIVATE=true to stage it first.
set -xeuo pipefail

DATASET_REPO=${DATASET_REPO:-JiangHoucheng/PolicyTraj-20K}
DATA_DIR=${DATA_DIR:-data}
PRIVATE=${PRIVATE:-false}

test -f "${DATA_DIR}/sft/train.parquet" || { echo "missing ${DATA_DIR}/sft/train.parquet"; exit 1; }
test -f "${DATA_DIR}/rl/train.jsonl"    || { echo "missing ${DATA_DIR}/rl/train.jsonl"; exit 1; }
test -f "${DATA_DIR}/eval/benchmark_val.jsonl" || { echo "missing ${DATA_DIR}/eval/benchmark_val.jsonl"; exit 1; }

python3 - "$DATASET_REPO" "$DATA_DIR" "$PRIVATE" <<'PY'
import sys
from huggingface_hub import HfApi

repo_id, data_dir, private = sys.argv[1], sys.argv[2], sys.argv[3].lower() == "true"
api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)

# The dataset card lives at docs/DATASET_CARD.md in this repo; upload it as the README.
api.upload_file(
    path_or_fileobj="docs/DATASET_CARD.md",
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="dataset",
)
# Ship the policy library alongside the splits: the ids in the data refer to it.
api.upload_file(
    path_or_fileobj="repolicy/assets/policies/policy_library.json",
    path_in_repo="policies/policy_library.json",
    repo_id=repo_id,
    repo_type="dataset",
)
api.upload_folder(
    folder_path=data_dir,
    repo_id=repo_id,
    repo_type="dataset",
    allow_patterns=["sft/*", "rl/*", "eval/*"],
)
print(f"Uploaded to https://huggingface.co/datasets/{repo_id}")
PY
