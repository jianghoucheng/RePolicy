---
license: apache-2.0
task_categories:
  - text-classification
  - text-generation
language:
  - en
tags:
  - agent-safety
  - guardrail
  - safety-policy
  - agent-trajectory
size_categories:
  - 10K<n<100K
configs:
  - config_name: sft
    default: true
    data_files:
      - split: train
        path: sft/train.parquet
      - split: validation
        path: sft/val.parquet
  - config_name: rl
    data_files:
      - split: train
        path: rl/train.parquet
      - split: validation
        path: rl/val.parquet
  - config_name: eval
    data_files:
      - split: test
        path: eval/benchmark_val.parquet
---

# PolicyTraj-20K

PolicyTraj-20K is the training and evaluation data for
[RePolicy-4B](https://huggingface.co/JiangHoucheng/RePolicy-4B), an agent
safeguard that learns safety-policy invocation through reinforcement learning.

Each example pairs an **agent trajectory** with a **local safety-policy
library** and annotates which policy governs the judgment, which clauses are
violated, the safety label, and a decision rationale. This gives both
trajectory-level supervision and clause-level policy grounding in one resource.

- Code: https://github.com/JiangHoucheng/RePolicy
- Model: https://huggingface.co/JiangHoucheng/RePolicy-4B

## Splits

| Config | Split | Examples | Format | Purpose |
|---|---|---|---|---|
| `sft` | train | 5,000 | parquet | Cold-start SFT |
| `sft` | validation | 500 | parquet | SFT validation |
| `rl` | train | 14,425 | parquet | GRPO training |
| `rl` | validation | 500 | parquet | GRPO validation |
| `eval` | test | 7,369 | parquet | Six-benchmark evaluation |

The SFT and RL splits total 20,425 policy-grounded trajectories and are disjoint
by construction. The RL train split is 8,186 unsafe / 6,239 safe.

`policies/policy_library.json` holds the 30-policy, 358-clause library the
`Policy_N` ids in every example refer to.

All splits are parquet (zstd). `extra_info.tools_kwargs` is a nested struct, not
a string, because verl indexes into it during rollout; the `local_to_global_*` /
`global_to_local_*` id maps *are* JSON strings, since their keys differ in every
example and would otherwise force a distinct arrow schema per row.

## Evaluation split composition

`eval/benchmark_val.parquet` normalizes six agent safety benchmarks into one
binary trajectory-classification task. The counts describe this normalized
split, not the original releases.

| Benchmark (`data_source`) | Unsafe | Safe | Total |
|---|---|---|---|
| `agent_guard_benchmark/atbench` | 500 | 500 | 1,000 |
| `agent_guard_benchmark/r_judge` | 702 | 540 | 1,242 |
| `agent_guard_benchmark/open_agent_safety` | 189 | 100 | 289 |
| `agent_guard_benchmark/assebench` | 806 | 670 | 1,476 |
| `agent_guard_benchmark/hintbench` | 583 | 126 | 709 |
| `agent_guard_benchmark/agent_hazard` | 2,653 | 0 | 2,653 |
| **Total** | **5,433** | **1,936** | **7,369** |

AgentHazard has no safe examples, so it measures sensitivity to cumulative harm
but not false-positive control; the other five cover both classes. Report
unsafe-class F1 per benchmark and average the six equally.

Sources: ATBench, R-Judge, OpenAgentSafety, ASSEBench, HINTBench, AgentHazard.
Please also cite the original benchmarks when using this split.

## Schema

### `rl` and `eval` (verl-compatible parquet)

```python
{
  "data_source": "agent_guard",              # or agent_guard_benchmark/<name>
  "agent_name": "tool_agent",
  "prompt": [                                 # policy TITLES only, no clause text
    {"role": "system", "content": "You are an agent safety guard. ..."},
    {"role": "user", "content": "<AGENT_TRAJECTORY>...</AGENT_TRAJECTORY>\n\n<POLICY_LIBRARY>...</POLICY_LIBRARY>"}
  ],
  "ability": "agent_safety_guard",
  "reward_model": {"style": "rule", "ground_truth": {"label": "unsafe", "rationale": "...", "answer": "..."}},
  "extra_info": {
    "gold_label": "unsafe",
    "gold_local_policy_ids": ["Policy_24"],           # policy that should be invoked
    "gold_local_violated_clause_ids": ["Policy_24.2"],
    "decoy_local_policy_ids": ["Policy_9", "Policy_13"],
    "local_to_global_policy": "{...}",                 # JSON string, id remapping
    "tools_kwargs": {"get_policy": {"create_kwargs": {"policy_library": [...]}}}
  }
}
```

The prompt deliberately exposes only policy ids and titles. Full clause text
lives in `extra_info.tools_kwargs`, served to the model when it calls
`get_policy` — so a model must decide *which* policy to invoke before it can
read any requirements.

### `sft` (parquet)

`messages`, `tools`, `enable_thinking`, `extra_info`. Messages follow the
five-turn supervision path with per-message loss masks:

| # | Role | Loss | Content |
|---|---|---|---|
| 1 | system | ✗ | Guard instruction |
| 2 | user | ✗ | Trajectory + policy titles |
| 3 | assistant | ✓ | `get_policy` tool call |
| 4 | tool | ✗ | Retrieved clauses |
| 5 | assistant | ✓ | Rationale + `<JUDGE>` tag |

## Policy-context perturbation

Local libraries are built by mixing the gold policy with distractors, then
shuffling. Distractors are real policies from unrelated operation scenes plus
synthetic decoys (e.g. "Calendar Color Theme") that are plainly non-safety. In
the RL train split each library holds 15-35 policies (mean 28.9) with a mean of
9.2 decoys and 1.64 gold policies. Because ids are re-indexed per example, the
same underlying policy appears under different `Policy_N` ids across examples,
which prevents memorising fixed trajectory-policy associations.

The `eval` split instead uses the full 30-policy real library with no decoys, so
evaluation is deterministic and comparable across models.

## Usage

```python
from datasets import load_dataset

sft = load_dataset("JiangHoucheng/PolicyTraj-20K", "sft")
rl = load_dataset("JiangHoucheng/PolicyTraj-20K", "rl")
bench = load_dataset("JiangHoucheng/PolicyTraj-20K", "eval", split="test")
```

To train with it, clone the code repo and use the provided scripts — they expect
this exact `data/{sft,rl,eval}` layout:

```bash
bash repolicy/scripts/download_data.sh
bash repolicy/scripts/run_sft.sh
bash repolicy/scripts/run_grpo.sh
```

## Construction

Agent interaction patterns were collected from existing agent safety resources
and normalized into a common trajectory representation. An API-based model then
resampled task content — user requests, entities, files, URLs, arguments, and
observations — while preserving trajectory structure, tool-use pattern, and the
underlying safety mechanism. Samples with inconsistent tool interactions,
unsupported labels, or heavy overlap with their reference were removed.

Safety policies are organized by **agent operation scene** rather than final
risk category, so one policy captures the requirements for a class of agent
operations. The library was audited against all trajectories: requirements were
refined where coverage was thin, new policies added only for genuinely distinct
scenes, and overlapping policies merged. Coverage of the evaluation split is
100% with a mean of 1.62 applicable policies per trajectory.

## Limitations

- Trajectories are largely synthetic expansions of existing safety resources; surface diversity exceeds that of the seed pool, but the underlying safety mechanisms derive from it.
- Each trajectory is grounded to a single governing policy, while real deployments may need several policies applied jointly.
- English only, binary `safe`/`unsafe` labels, offline trajectories that are not re-executed.
- Rationales are model-generated and audited for policy grounding, not verified sentence by sentence.
- Trajectories describe unsafe agent behavior for safeguard research; the content is intended for training and evaluating guards, not for reuse as instructions.

## License

Apache 2.0. Derived-from benchmarks retain their original licenses; cite them
alongside this dataset.
