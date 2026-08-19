#!/usr/bin/env bash
# Stage 2: GRPO on safeguard rollouts.
#
# Each rollout invokes get_policy, receives the clause text as a tool response,
# and then produces the rationale and judgment. The rule-based reward in
# repolicy/reward.py scores format, policy invocation, and safety accuracy.
# Defaults target 8x H20 (96GB). Override any value via the environment.
set -xeuo pipefail

MODEL_PATH=${MODEL_PATH:-checkpoints/repolicy_sft_hf}
TRAIN_FILE=${TRAIN_FILE:-data/rl/train.jsonl}
VAL_FILE=${VAL_FILE:-data/rl/val.jsonl}
BENCHMARK_VAL_FILE=${BENCHMARK_VAL_FILE:-data/eval/benchmark_val.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/repolicy_grpo}
NUM_GPUS=${NUM_GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-2}
LOG_PROB_MICRO_BATCH_SIZE=${LOG_PROB_MICRO_BATCH_SIZE:-4}
LR=${LR:-1e-6}
ROLLOUT_N=${ROLLOUT_N:-16}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-12288}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:-10}
PROJECT_NAME=${PROJECT_NAME:-repolicy}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-repolicy_grpo_qwen3_4b}
LOGGER=${LOGGER:-'["console","wandb"]'}
INFER_BACKEND=${INFER_BACKEND:-vllm}
ROLLOUT_TP=${ROLLOUT_TP:-1}
TOOL_CONFIG_PATH=${TOOL_CONFIG_PATH:-repolicy/config/policy_tool_config.yaml}
TOOL_FORMAT=${TOOL_FORMAT:-hermes}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-false}
USE_TORCH_COMPILE=${USE_TORCH_COMPILE:-false}
PYTHON=${PYTHON:-python3}

# Validate on the held-out RL split, plus the six-benchmark suite when present.
VAL_FILES="['${VAL_FILE}']"
if [ -n "${BENCHMARK_VAL_FILE}" ] && [ -f "${BENCHMARK_VAL_FILE}" ]; then
  VAL_FILES="['${VAL_FILE}', '${BENCHMARK_VAL_FILE}']"
fi

$PYTHON -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="['${TRAIN_FILE}']" \
  data.val_files="${VAL_FILES}" \
  data.train_batch_size="${BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.return_raw_chat=True \
  data.filter_overlong_prompts=True \
  data.truncation=right \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  +actor_rollout_ref.model.override_config.attn_implementation="${ATTN_IMPLEMENTATION}" \
  actor_rollout_ref.model.use_remove_padding="${USE_REMOVE_PADDING}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.use_remove_padding="${USE_REMOVE_PADDING}" \
  actor_rollout_ref.actor.fsdp_config.use_torch_compile="${USE_TORCH_COMPILE}" \
  actor_rollout_ref.actor.optim.lr="${LR}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name="${INFER_BACKEND}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.format="${TOOL_FORMAT}" \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG_PATH}" \
  actor_rollout_ref.rollout.multi_turn.max_user_turns=1 \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2 \
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length=12000 \
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice=True \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser="${TOOL_FORMAT}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE}" \
  reward.custom_reward_function.path=repolicy/reward.py \
  reward.custom_reward_function.name=compute_score \
  trainer.logger="${LOGGER}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${NUM_GPUS}" \
  trainer.nnodes=1 \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  "$@"
