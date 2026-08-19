#!/usr/bin/env bash
# Upload a trained RePolicy-4B checkpoint to the Hugging Face Hub.
#
#   huggingface-cli login          # once, needs a write token
#   MODEL_DIR=release/RePolicy-4B bash repolicy/scripts/upload_model.sh
#
# MODEL_DIR must be a merged HF-format directory (config.json + *.safetensors),
# i.e. the output of export_sft_hf.sh, not raw FSDP shards.
# This publishes to a public repo by default; set PRIVATE=true to stage it first.
set -xeuo pipefail

MODEL_REPO=${MODEL_REPO:-JiangHoucheng/RePolicy-4B}
MODEL_DIR=${MODEL_DIR:-release/RePolicy-4B}
PRIVATE=${PRIVATE:-false}

test -f "${MODEL_DIR}/config.json" || { echo "missing ${MODEL_DIR}/config.json"; exit 1; }
ls "${MODEL_DIR}"/*.safetensors >/dev/null 2>&1 || { echo "no safetensors in ${MODEL_DIR}"; exit 1; }

python3 - "$MODEL_REPO" "$MODEL_DIR" "$PRIVATE" <<'PY'
import sys
from huggingface_hub import HfApi

repo_id, model_dir, private = sys.argv[1], sys.argv[2], sys.argv[3].lower() == "true"
api = HfApi()
api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

# The model card lives at docs/MODEL_CARD.md in this repo; upload it as the README.
api.upload_file(
    path_or_fileobj="docs/MODEL_CARD.md",
    path_in_repo="README.md",
    repo_id=repo_id,
)
api.upload_folder(
    folder_path=model_dir,
    repo_id=repo_id,
    ignore_patterns=["README.md", "*.pt", "optimizer*", "**/optimizer*"],
)
print(f"Uploaded to https://huggingface.co/{repo_id}")
PY
