#!/usr/bin/env bash
# Convert a verl FSDP checkpoint into a HuggingFace-format directory.
#
# Run this after run_sft.sh: GRPO loads the merged HF model, not the FSDP shards.
set -xeuo pipefail

SFT_CKPT_DIR=${SFT_CKPT_DIR:-checkpoints/repolicy_sft/global_step_78}
TARGET_DIR=${TARGET_DIR:-checkpoints/repolicy_sft_hf}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-false}

TRUST_FLAG=()
if [ "${TRUST_REMOTE_CODE}" = "true" ]; then
  TRUST_FLAG=(--trust-remote-code)
fi

python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${SFT_CKPT_DIR}" \
  --target_dir "${TARGET_DIR}" \
  "${TRUST_FLAG[@]}"
