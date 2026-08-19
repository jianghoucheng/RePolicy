#!/usr/bin/env bash
# Stage 1: cold-start SFT on PolicyTraj-20K.
#
# Teaches the full invocation-reasoning sequence: emit a get_policy tool call,
# read the returned clauses, then produce the rationale and safety judgment.
# Defaults target 8x H20 (96GB). Override any value via the environment.
set -xeuo pipefail

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}
TRAIN_FILE=${TRAIN_FILE:-data/sft/train.parquet}
VAL_FILE=${VAL_FILE:-data/sft/val.parquet}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/repolicy_sft}
NUM_GPUS=${NUM_GPUS:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
BATCH_SIZE=${BATCH_SIZE:-4}
LR=${LR:-1e-5}
EPOCHS=${EPOCHS:-2}
MAX_LENGTH=${MAX_LENGTH:-12288}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-${MAX_LENGTH}}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
USE_TORCH_COMPILE=${USE_TORCH_COMPILE:-false}
PROJECT_NAME=${PROJECT_NAME:-repolicy}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-repolicy_sft_qwen3_4b}
LOGGER=${LOGGER:-'["console","wandb"]'}

torchrun --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" \
  -m verl.trainer.sft_trainer \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.micro_batch_size_per_gpu="${BATCH_SIZE}" \
  data.messages_key=messages \
  data.tools_key=tools \
  data.enable_thinking_key=enable_thinking \
  data.ignore_input_ids_mismatch=True \
  data.max_length="${MAX_LENGTH}" \
  data.max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
  data.truncation=right \
  optim.lr="${LR}" \
  engine=fsdp \
  engine.use_torch_compile="${USE_TORCH_COMPILE}" \
  model.path="${MODEL_PATH}" \
  +model.override_config.attn_implementation="${ATTN_IMPLEMENTATION}" \
  model.use_remove_padding=true \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.logger="${LOGGER}" \
  trainer.total_epochs="${EPOCHS}" \
  "$@"
