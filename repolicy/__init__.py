"""RePolicy: reinforcement learning for safety-policy invocation in agent safeguards.

The package holds everything specific to RePolicy; the surrounding ``verl``
tree is the unmodified training framework it runs on.

Modules
-------
prompts             System/user prompt construction for the safeguard rollout.
policy_context      Local policy-library sampling, including policy-context
                    perturbation (irrelevant policies plus synthetic decoys).
policy_tool         The ``get_policy`` tool verl serves during multi-turn rollout.
reward              Rule-based reward: format, policy invocation, safety accuracy.
build_dataset       Builds the SFT and RL splits of PolicyTraj-20K.
build_benchmark_eval  Builds the six-benchmark evaluation split.
evaluate            Unsafe-F1 scoring of generated predictions.
"""

__all__ = ["__version__"]

__version__ = "1.0.0"
