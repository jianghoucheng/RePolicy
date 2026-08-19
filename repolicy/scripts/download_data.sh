#!/usr/bin/env bash
# Download PolicyTraj-20K (training/eval data) from the Hugging Face Hub into data/.
#
#   bash repolicy/scripts/download_data.sh
#
# Downloads ~2.3GB. The layout it produces is what run_sft.sh / run_grpo.sh expect.
set -xeuo pipefail

DATASET_REPO=${DATASET_REPO:-JiangHoucheng/PolicyTraj-20K}
TARGET_DIR=${TARGET_DIR:-data}

python3 - "$DATASET_REPO" "$TARGET_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id, target_dir = sys.argv[1], sys.argv[2]
path = snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=target_dir,
    allow_patterns=["sft/*", "rl/*", "eval/*"],
)
print(f"Downloaded {repo_id} -> {path}")
PY

echo "--- data layout ---"
find "${TARGET_DIR}" -type f -name "*.jsonl" -o -type f -name "*.parquet" | sort
