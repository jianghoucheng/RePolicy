<div align="center">

# RePolicy

**Reinforcement Learning for Safety-Policy Invocation in Agent Safeguards**

[![Model](https://img.shields.io/badge/🤗%20Model-RePolicy--4B-blue)](https://huggingface.co/JiangHoucheng/RePolicy-4B)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-PolicyTraj--20K-green)](https://huggingface.co/datasets/JiangHoucheng/PolicyTraj-20K)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](LICENSE)

</div>

RePolicy is an agent safeguard that **learns to invoke safety policies** through
reinforcement learning. Given an agent trajectory and a dynamic policy library,
it selects the applicable policy, retrieves its clauses via a tool call, and
produces a policy-grounded rationale plus a trajectory-level safety judgment.

The key difference from prompting- or SFT-based policy-aware guards: RePolicy
never sees all policy text up front. Its prompt carries only policy **titles**,
so it must decide *which* policy governs the trajectory before it can read any
requirements. That turns safety-policy invocation into an agentic RL problem
optimized directly through safeguard outcomes.

Paper: RePolicy: Reinforcement Learning for Safety-Policy Invocation in Agent Safeguards
```
                policy invocation  →  policy content  →  safety reasoning  →  safety prediction

  trajectory τ  ──┐
                  ├──►  πθ  ──►  get_policy(["Policy_7"])  ──►  clause text  ──►  rationale + <JUDGE>unsafe</JUDGE>
  policy titles ──┘                         ▲                                              │
                                            └──────────── GRPO ◄───── R = 0.1·fmt + 0.3·pol + 0.6·acc
```

## Results

Unsafe-class F1 (%) on six agent safety benchmarks; Overall is the unweighted
mean. RePolicy-4B ranks first on four of six and best overall, at 4B parameters.

| Model | ATBench | R-Judge | OpenAgentSafety | ASSEBench | HINTBench | AgentHazard | Overall |
|---|---|---|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 (backbone) | 75.79 | 63.12 | 11.48 | 61.72 | 84.58 | 54.11 | 58.47 |
| Claude Sonnet 4.6 | 95.46 | 86.26 | 51.97 | 81.10 | 96.47 | 93.73 | 84.17 |
| GPT-5.4 | 96.83 | 87.44 | 55.63 | 76.25 | 95.42 | 87.13 | 83.12 |
| DynaGuard-4B | 89.43 | 75.53 | 55.76 | 69.70 | 87.48 | 81.24 | 76.52 |
| AgentDoG-4B | 87.93 | 80.24 | 48.99 | 82.85 | 90.25 | 87.87 | 79.69 |
| **RePolicy-4B** | **99.40** | 82.17 | **62.79** | **88.11** | 97.23 | **99.20** | **88.15** |

Ablations (Overall): base model 58.47 → cold-start SFT 86.95 → **RePolicy 88.15**.
Removing explicit policy invocation costs 2.32 points; removing policy-context
perturbation costs 0.68.

## Repository layout

```
repolicy/                     Everything specific to RePolicy
├── prompts.py                System/user prompt construction
├── policy_context.py         Local library sampling + policy-context perturbation
├── policy_tool.py            The get_policy tool served during rollout
├── reward.py                 Verifiable reward: format / invocation / accuracy
├── build_dataset.py          Builds the PolicyTraj-20K SFT + RL splits
├── build_benchmark_eval.py   Builds the six-benchmark eval split
├── evaluate.py               Per-benchmark unsafe-F1 scoring
├── predict.py                Runnable single-trajectory inference demo
├── config/                   Tool config consumed by verl
├── assets/policies/          The 30-policy / 358-clause safety library
└── scripts/                  run_sft.sh, export_sft_hf.sh, run_grpo.sh, HF up/download

verl/                         Training framework (upstream verl 0.8.0.dev, unmodified API)
data/                         PolicyTraj-20K after download (sft/ rl/ eval/)
docs/                         Model card, dataset card, and the paper
```

## Setup

Requires Linux, 8x GPU with ≥80GB each for the default config (trained on 8x H20
96GB), CUDA 12.x, and Python 3.10+.

```bash
git clone https://github.com/JiangHoucheng/RePolicy.git
cd RePolicy

python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e .            # installs verl + repolicy and their dependencies
pip install vllm            # rollout engine used by GRPO
```

Optional but recommended:

```bash
export HF_HOME=/path/to/hf-cache      # somewhere with room for the 4B backbone
wandb login                            # scripts log to wandb by default;
                                       # set LOGGER='["console"]' to skip
```

## Pipeline

Four steps, in order. Each stage's output is the next stage's input.

```
  ① download data  →  ② SFT  →  ③ merge to HF  →  ④ GRPO  →  ⑤ evaluate
     PolicyTraj-20K    cold start   FSDP→HF        RePolicy-4B   unsafe F1
```

### ① Download PolicyTraj-20K

```bash
bash repolicy/scripts/download_data.sh
```

Pulls ~2.3GB from [`JiangHoucheng/PolicyTraj-20K`](https://huggingface.co/datasets/JiangHoucheng/PolicyTraj-20K) into `data/`:

| Path | Rows | Used by |
|---|---|---|
| `data/sft/train.parquet` | 5,000 | SFT |
| `data/sft/val.parquet` | 500 | SFT validation |
| `data/rl/train.parquet` | 14,425 | GRPO |
| `data/rl/val.parquet` | 500 | GRPO validation |
| `data/eval/benchmark_val.parquet` | 7,369 | Six-benchmark evaluation |

The data ships ready to train — no preprocessing needed. Rebuilding it from raw
annotated trajectories is only necessary if you change the policy library or the
perturbation settings; see [Rebuilding the data](#rebuilding-the-data-optional).

### ② Cold-start SFT

Teaches the full invocation→reasoning sequence. Loss is applied only to the two
assistant turns (the tool call and the final judgment).

```bash
NUM_GPUS=8 \
OUTPUT_DIR=checkpoints/repolicy_sft \
bash repolicy/scripts/run_sft.sh
```

Defaults: `Qwen/Qwen3-4B-Instruct-2507`, 2 epochs, lr 1e-5, batch 128, max length
12288 → 78 steps on 8 GPUs. Override anything via the environment, e.g.
`EPOCHS=3 LR=5e-6 NUM_GPUS=4 bash repolicy/scripts/run_sft.sh`.

### ③ Merge the SFT checkpoint to HuggingFace format

GRPO loads an HF-format model, not raw FSDP shards.

```bash
SFT_CKPT_DIR=$(ls -dt checkpoints/repolicy_sft/global_step_* | head -1) \
TARGET_DIR=checkpoints/repolicy_sft_hf \
bash repolicy/scripts/export_sft_hf.sh
```

Verify `checkpoints/repolicy_sft_hf/` contains `config.json` and
`*.safetensors` before continuing.

### ④ GRPO

Optimizes the complete safeguard rollout: 16 rollouts per prompt, each invoking
a policy, receiving its clauses, and producing a judgment.

```bash
MODEL_PATH=checkpoints/repolicy_sft_hf \
NUM_GPUS=8 \
OUTPUT_DIR=checkpoints/repolicy_grpo \
bash repolicy/scripts/run_grpo.sh
```

Defaults: lr 1e-6, batch 64, `ROLLOUT_N=16`, prompt 12288 / response 4096,
`MAX_MODEL_LEN=16384`, KL coef 0.001, 3 epochs, checkpoint and validate every 10
steps. Validation runs on both `data/rl/val.parquet` and the benchmark suite.

**Reward** (`repolicy/reward.py`):

| Weight | Term | Condition |
|---|---|---|
| 0.1 | format | exactly one valid `get_policy` call **and** a well-formed `<JUDGE>` tag |
| 0.3 | policy invocation | invoked policy set intersects the gold policies |
| 0.6 | safety accuracy | predicted label matches the gold label |

Also logged per rollout: `decoy_selected`, `tool_call_count`, `num_pred_policies`
— useful for spotting a model that stops invoking policies or starts grabbing
decoys.

**Policy-context perturbation** is applied when the data is built, not at
rollout time: each example's local library mixes the gold policy with real
policies from unrelated scenes plus synthetic decoys (15–35 policies, mean 28.9,
mean 9.2 decoys), with ids re-indexed and order shuffled per example.

Then export the final model:

```bash
SFT_CKPT_DIR=$(ls -dt checkpoints/repolicy_grpo/global_step_* | head -1)/actor \
TARGET_DIR=release/RePolicy-4B \
bash repolicy/scripts/export_sft_hf.sh
```

### ⑤ Evaluate

Score-only pass over the six-benchmark split, no gradient updates:

```bash
MODEL_PATH=release/RePolicy-4B \
NUM_GPUS=8 \
OUTPUT_DIR=checkpoints/repolicy_eval \
EXPERIMENT_NAME=repolicy_eval \
bash repolicy/scripts/run_grpo.sh \
  trainer.val_only=True \
  trainer.val_before_train=True \
  data.train_max_samples=1
```

To score predictions you generated yourself, with per-benchmark unsafe F1:

```bash
python -m repolicy.evaluate \
  --predictions your_outputs.jsonl \
  --references data/eval/benchmark_val.parquet
```

`--predictions` is a JSONL whose rows carry the raw model text under
`output`, `prediction`, `response`, or `solution`, in the same order as the
reference file. Reported `unsafe_f1` treats `unsafe` as the positive class; the
paper's Overall is the unweighted mean of the six per-benchmark values.

## Inference

The trained model is on the Hub. The rollout is two steps: the model asks for a
policy, you serve the clauses, the model judges.

```bash
python -m repolicy.predict --model JiangHoucheng/RePolicy-4B
```

That runs a built-in unsafe demo trajectory (an agent emailing staging
credentials to an external address) against the shipped 30-policy library and
prints both steps. Use your own trajectory with `--trajectory my_traj.json`, or a
local checkpoint with `--model release/RePolicy-4B`.

For the raw `transformers` loop, prompt format, and output contract, see
[`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

Expected output shape:

```
<tool_call>{"name": "get_policy", "arguments": {"policy_ids": ["Policy_1"]}}</tool_call>
```

then, once the clauses are returned:

```
Policy_1 is relevant. The agent read credentials and emailed them to an external
address without user authorization. The trajectory violates the retrieved policy
clauses.

<JUDGE>unsafe</JUDGE>
```

## Publishing to Hugging Face

Both scripts need a write token (`huggingface-cli login`) and default to public
repos — pass `PRIVATE=true` to stage first. They upload `docs/MODEL_CARD.md` and
`docs/DATASET_CARD.md` as the respective `README.md`.

```bash
# Model → JiangHoucheng/RePolicy-4B
MODEL_DIR=release/RePolicy-4B bash repolicy/scripts/upload_model.sh

# Dataset → JiangHoucheng/PolicyTraj-20K
bash repolicy/scripts/upload_data.sh
```

`upload_model.sh` requires a merged HF directory (step ③/④ output) and refuses to
run without `config.json` and `*.safetensors`. Override the destination with
`MODEL_REPO=` / `DATASET_REPO=`.

## Rebuilding the data (optional)

Only needed if you change the policy library, the perturbation settings, or the
train/val sizes. Both builders read the policy library in
`repolicy/assets/policies/`.

```bash
# SFT + RL splits from annotated trajectories
python -m repolicy.build_dataset \
  --input-jsonl data/generated/train.jsonl \
  --sft-dir data/sft --rl-dir data/rl \
  --sft-train-size 5000 --sft-val-size 500 --rl-val-size 500 --seed 42

# Six-benchmark eval split from normalized benchmark files
python -m repolicy.build_benchmark_eval \
  --data-dir data/benchmarks \
  --output-dir data/eval
```

`--input-jsonl` expects rows with `trajectory`, `label`, `selected_policies`, and
`rationale.violated_clauses`. `--data-dir` expects the six normalized benchmark
files (`atbench.jsonl`, `r_judge.jsonl`, `open_agent_safety.jsonl`,
`assebench.jsonl`, `hintbench.jsonl`, `agent_hazard.jsonl`), joined by `id`
against `repolicy/assets/policies/policy_coverage_map.jsonl`. The source
trajectory pool derives from third-party benchmarks and is not redistributed
here; the built splits on the Hub are.

## Notes and caveats

- **Long prompts.** Trajectory plus a 30-policy title list runs to ~12K tokens; `MAX_PROMPT_LENGTH=12288` with `filter_overlong_prompts=True` silently drops longer examples. Raise both if you enlarge the library.
- **`max_assistant_turns=2`** is deliberate — one turn to invoke, one to judge. Raising it lets the model call `get_policy` repeatedly, which the format reward penalizes.
- **`get_policy` caps at 5 policies per call** (`MAX_POLICIES_PER_CALL` in `repolicy/policy_tool.py` and `repolicy/reward.py`); keep the two in sync if you change it.
- **Memory.** With `ROLLOUT_N=16` and 8×80GB, lower `PPO_MICRO_BATCH_SIZE` before anything else if you hit OOM.
- The reward verifies the invoked policy and the final label, not that every sentence of the rationale is faithful to the policy.


## Acknowledgements

Built on [verl](https://github.com/verl-project/verl) for RL training and
[Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) as
the backbone. Evaluation uses ATBench, R-Judge, OpenAgentSafety, ASSEBench,
HINTBench, and AgentHazard — please cite those alongside this work.

## License

Apache 2.0. See [LICENSE](LICENSE); `verl/` retains its upstream copyright.
