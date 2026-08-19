"""Construct local policy contexts, including policy-context perturbation.

Each example gets a local library formed from the gold policy plus distractors:
real policies from unrelated operation scenes and synthetic decoy policies.
Policy ids are re-indexed to ``Policy_N`` and the order is shuffled, so the
model cannot memorise fixed trajectory-policy associations.
"""

from __future__ import annotations

import copy
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from repolicy.prompts import build_messages


DECOY_THEMES = [
    ("Decorative Color Preference", "This policy is unrelated to safety and only concerns decorative interface color preferences."),
    ("Music Playlist Style", "This policy is unrelated to safety and only concerns preferred music playlist ordering."),
    ("Fictional Character Naming", "This policy is unrelated to safety and only concerns naming fictional characters."),
    ("Desktop Wallpaper Theme", "This policy is unrelated to safety and only concerns wallpaper theme selection."),
    ("Casual Greeting Tone", "This policy is unrelated to safety and only concerns casual greeting style."),
    ("Emoji Preference", "This policy is unrelated to safety and only concerns preferred emoji usage."),
    ("Font Size Preference", "This policy is unrelated to safety and only concerns visual font size preferences."),
    ("Calendar Color Theme", "This policy is unrelated to safety and only concerns calendar color labels."),
    ("Keyboard Shortcut Preference", "This policy is unrelated to safety and only concerns preferred keyboard shortcut naming."),
    ("Recipe Serving Style", "This policy is unrelated to safety and only concerns how recipe serving suggestions are written."),
    ("Notebook Tab Color", "This policy is unrelated to safety and only concerns organizing notebook tabs by color."),
    ("Avatar Border Style", "This policy is unrelated to safety and only concerns decorative avatar border preferences."),
    ("Weather Widget Layout", "This policy is unrelated to safety and only concerns the visual layout of a weather widget."),
    ("Meeting Room Theme Name", "This policy is unrelated to safety and only concerns naming themes for meeting rooms."),
    ("Bookmark Folder Icon", "This policy is unrelated to safety and only concerns decorative icons for bookmark folders."),
    ("Coffee Order Nickname", "This policy is unrelated to safety and only concerns friendly nicknames for coffee orders."),
    ("Dashboard Accent Shape", "This policy is unrelated to safety and only concerns decorative accent shapes on dashboards."),
    ("Reading List Mood", "This policy is unrelated to safety and only concerns mood labels for reading lists."),
]


@dataclass
class PolicyContext:
    local_policy_library: list[dict[str, Any]]
    local_to_global_policy: dict[str, str]
    global_to_local_policy: dict[str, str]
    local_to_global_clause: dict[str, str]
    global_to_local_clause: dict[str, str]
    gold_local_policy_ids: list[str]
    gold_local_violated_clause_ids: list[str]
    decoy_local_policy_ids: list[str]


def load_policy_library(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def policy_by_id(policies: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {p["policy_id"]: p for p in policies}


def clause_text_by_id(policies: list[dict[str, Any]]) -> dict[str, str]:
    return {c["clause_id"]: c["text"] for p in policies for c in p.get("clauses", [])}


def local_policy_id(index: int) -> str:
    return f"Policy_{index + 1}"


def make_decoy(index: int) -> dict[str, Any]:
    title, description = DECOY_THEMES[index % len(DECOY_THEMES)]
    return {
        "policy_id": f"DECOY_{index + 1:02d}",
        "title": title,
        "description": description,
        "clauses": [],
    }


def policy_catalog(local_policy_library: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "policy_id": p["policy_id"],
            "title": p["title"],
        }
        for p in local_policy_library
    ]


def selected_policy_texts(context: PolicyContext) -> list[dict[str, Any]]:
    selected = set(context.gold_local_policy_ids)
    return [p for p in context.local_policy_library if p["policy_id"] in selected]


def policy_lookup_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "get_policy",
            "description": "Retrieve full safety policy text and clauses by local policy ids. Call once with the relevant policy ids.",
            "parameters": {
                "type": "object",
                "properties": {
                    "policy_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Local policy ids to retrieve, such as Policy_1.",
                    }
                },
                "required": ["policy_ids"],
            },
        },
    }


def strip_global_ids(text: str, global_to_local_policy: dict[str, str], global_to_local_clause: dict[str, str]) -> str:
    out = text or ""
    for gid, lid in sorted(global_to_local_clause.items(), key=lambda kv: -len(kv[0])):
        out = out.replace(gid, lid)
    for gid, lid in sorted(global_to_local_policy.items(), key=lambda kv: -len(kv[0])):
        out = out.replace(gid, lid)
    out = re.sub(r"\bP\d{2}(?:\.\d+)?\b", "the selected local policy", out)
    return out


def clean_rationale_text(text: str) -> str:
    out = text or ""
    out = re.sub(r"\bThe unsafe benchmark label is preserved with the same violated clauses as the reference\.?\s*", "", out)
    out = re.sub(r"\bThe safe benchmark label is preserved with no violated clauses\.?\s*", "", out)
    out = re.sub(r"\bThe benchmark label is preserved\.?\s*", "", out)
    out = re.sub(r"\bthe benchmark label is preserved\b", "the safety label is unsafe", out, flags=re.IGNORECASE)
    return out


def sample_training_policy_count(rng: random.Random, minimum: int, maximum: int) -> int:
    """Skew counts toward about 30 while retaining smaller policy-library cases."""

    if rng.random() < 0.8:
        lower = min(maximum, max(minimum, 27))
        return rng.randint(lower, maximum)
    upper = min(maximum, 26)
    if upper < minimum:
        return rng.randint(minimum, maximum)
    return rng.randint(minimum, upper)


def build_context(
    sample: dict[str, Any],
    policy_library: list[dict[str, Any]],
    rng: random.Random,
    min_total_policies: int = 15,
    max_total_policies: int = 35,
    min_decoys: int = 2,
) -> PolicyContext:
    by_id = policy_by_id(policy_library)
    gold_global_ids = [p["policy_id"] for p in sample.get("selected_policies", []) if p.get("policy_id") in by_id]
    if not gold_global_ids:
        raise ValueError("sample has no valid selected_policies")

    irrelevant_pool = [p for p in policy_library if p["policy_id"] not in set(gold_global_ids)]
    target_count = sample_training_policy_count(rng, max(min_total_policies, len(gold_global_ids)), max_total_policies)
    available_slots = max(0, target_count - len(gold_global_ids))
    min_decoys_for_target = min(min_decoys, available_slots)
    max_real_irrelevant = min(len(irrelevant_pool), available_slots - min_decoys_for_target)
    min_real_irrelevant = min(len(irrelevant_pool), max(0, min(available_slots, min_total_policies - len(gold_global_ids) - min_decoys)))
    if max_real_irrelevant < min_real_irrelevant:
        max_real_irrelevant = min_real_irrelevant
    n_irrelevant = rng.randint(min_real_irrelevant, max_real_irrelevant) if max_real_irrelevant > 0 else 0
    n_decoys = available_slots - n_irrelevant

    candidate_global = [copy.deepcopy(by_id[pid]) for pid in gold_global_ids]
    candidate_global += [copy.deepcopy(p) for p in rng.sample(irrelevant_pool, n_irrelevant)]
    candidate_global += [make_decoy(i) for i in range(n_decoys)]
    rng.shuffle(candidate_global)

    local_to_global_policy: dict[str, str] = {}
    global_to_local_policy: dict[str, str] = {}
    local_to_global_clause: dict[str, str] = {}
    global_to_local_clause: dict[str, str] = {}
    local_policy_library: list[dict[str, Any]] = []
    decoy_local_policy_ids: list[str] = []

    for idx, policy in enumerate(candidate_global):
        local_pid = local_policy_id(idx)
        global_pid = policy["policy_id"]
        local_to_global_policy[local_pid] = global_pid
        global_to_local_policy[global_pid] = local_pid
        local_clauses = []
        for cidx, clause in enumerate(policy.get("clauses", []), 1):
            local_cid = f"{local_pid}.{cidx}"
            global_cid = clause["clause_id"]
            local_to_global_clause[local_cid] = global_cid
            global_to_local_clause[global_cid] = local_cid
            local_clauses.append({"clause_id": local_cid, "text": clause["text"]})
        if global_pid.startswith("DECOY_"):
            decoy_local_policy_ids.append(local_pid)
        local_policy_library.append(
            {
                "policy_id": local_pid,
                "title": policy["title"],
                "description": policy["description"],
                "clauses": local_clauses,
            }
        )

    gold_local_policy_ids = [global_to_local_policy[pid] for pid in gold_global_ids]
    gold_global_violated = [
        c["clause_id"]
        for c in sample.get("rationale", {}).get("violated_clauses", [])
        if c.get("clause_id") in global_to_local_clause
    ]
    gold_local_violated_clause_ids = [global_to_local_clause[cid] for cid in gold_global_violated]
    return PolicyContext(
        local_policy_library=local_policy_library,
        local_to_global_policy=local_to_global_policy,
        global_to_local_policy=global_to_local_policy,
        local_to_global_clause=local_to_global_clause,
        global_to_local_clause=global_to_local_clause,
        gold_local_policy_ids=gold_local_policy_ids,
        gold_local_violated_clause_ids=gold_local_violated_clause_ids,
        decoy_local_policy_ids=decoy_local_policy_ids,
    )


def build_full_library_context(
    sample: dict[str, Any],
    policy_library: list[dict[str, Any]],
) -> PolicyContext:
    """Build deterministic eval context with the full real policy library and no decoys."""

    by_id = policy_by_id(policy_library)
    gold_global_ids = [p["policy_id"] for p in sample.get("selected_policies", []) if p.get("policy_id") in by_id]
    if not gold_global_ids:
        raise ValueError("sample has no valid selected_policies")

    local_to_global_policy: dict[str, str] = {}
    global_to_local_policy: dict[str, str] = {}
    local_to_global_clause: dict[str, str] = {}
    global_to_local_clause: dict[str, str] = {}
    local_policy_library: list[dict[str, Any]] = []

    for idx, policy in enumerate(policy_library):
        local_pid = local_policy_id(idx)
        global_pid = policy["policy_id"]
        local_to_global_policy[local_pid] = global_pid
        global_to_local_policy[global_pid] = local_pid
        local_clauses = []
        for cidx, clause in enumerate(policy.get("clauses", []), 1):
            local_cid = f"{local_pid}.{cidx}"
            global_cid = clause["clause_id"]
            local_to_global_clause[local_cid] = global_cid
            global_to_local_clause[global_cid] = local_cid
            local_clauses.append({"clause_id": local_cid, "text": clause["text"]})
        local_policy_library.append(
            {
                "policy_id": local_pid,
                "title": policy["title"],
                "description": policy["description"],
                "clauses": local_clauses,
            }
        )

    gold_global_violated = [
        c["clause_id"]
        for c in sample.get("rationale", {}).get("violated_clauses", [])
        if c.get("clause_id") in global_to_local_clause
    ]
    return PolicyContext(
        local_policy_library=local_policy_library,
        local_to_global_policy=local_to_global_policy,
        global_to_local_policy=global_to_local_policy,
        local_to_global_clause=local_to_global_clause,
        global_to_local_clause=global_to_local_clause,
        gold_local_policy_ids=[global_to_local_policy[pid] for pid in gold_global_ids],
        gold_local_violated_clause_ids=[global_to_local_clause[cid] for cid in gold_global_violated],
        decoy_local_policy_ids=[],
    )


def label_text(label: int) -> str:
    return "unsafe" if int(label) == 1 else "safe"


def build_target(sample: dict[str, Any], context: PolicyContext) -> dict[str, Any]:
    label = label_text(sample["label"])
    summary = sample.get("rationale", {}).get("summary", "")
    summary = strip_global_ids(summary, context.global_to_local_policy, context.global_to_local_clause)
    summary = clean_rationale_text(summary)
    relevant = ", ".join(context.gold_local_policy_ids)
    verb = "is" if len(context.gold_local_policy_ids) == 1 else "are"
    if label == "safe":
        summary = f"{relevant} {verb} relevant. {summary} The trajectory does not violate the retrieved policy clauses."
    else:
        summary = f"{relevant} {verb} relevant. {summary} The trajectory violates the retrieved policy clauses."
    summary = " ".join(summary.split())
    answer = f"{summary}\n\n<JUDGE>{label}</JUDGE>"
    return {
        "label": label,
        "rationale": summary,
        "answer": answer,
    }


def build_prompt(sample: dict[str, Any], context: PolicyContext) -> list[dict[str, str]]:
    return build_messages(sample["trajectory"], policy_catalog(context.local_policy_library))


def build_policy_tool_call(context: PolicyContext) -> str:
    return (
        "<tool_call>"
        + json.dumps(
            {
                "name": "get_policy",
                "arguments": {"policy_ids": context.gold_local_policy_ids},
            },
            ensure_ascii=False,
        )
        + "</tool_call>"
    )


def build_policy_tool_calls(context: PolicyContext) -> list[dict[str, Any]]:
    """OpenAI/Qwen chat-template compatible tool call representation."""

    return [
        {
            "id": "policy_lookup_1",
            "type": "function",
            "function": {
                "name": "get_policy",
                "arguments": json.dumps({"policy_ids": context.gold_local_policy_ids}, ensure_ascii=False),
            },
        }
    ]


def build_policy_tool_response(context: PolicyContext) -> str:
    return json.dumps({"policies": selected_policy_texts(context)}, ensure_ascii=False, indent=2)


def build_sft_messages(sample: dict[str, Any], context: PolicyContext, target: dict[str, Any]) -> list[dict[str, Any]]:
    prompt_messages = [
        {**message, "loss": False}
        for message in build_prompt(sample, context)
    ]
    return prompt_messages + [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": build_policy_tool_calls(context),
            "loss": True,
        },
        {
            "role": "tool",
            "name": "get_policy",
            "tool_call_id": "policy_lookup_1",
            "content": build_policy_tool_response(context),
            "loss": False,
        },
        {
            "role": "assistant",
            "content": target["answer"],
            "loss": True,
        },
    ]


def build_rl_extra_info(sample: dict[str, Any], context: PolicyContext, index: int) -> dict[str, Any]:
    return {
        "index": index,
        "gold_label": label_text(sample["label"]),
        "gold_local_policy_ids": context.gold_local_policy_ids,
        "gold_local_violated_clause_ids": context.gold_local_violated_clause_ids if int(sample["label"]) == 1 else [],
        "decoy_local_policy_ids": context.decoy_local_policy_ids,
        # Keep dynamic-key maps as JSON strings so datasets.load_dataset("json")
        # does not try to infer a different nested schema for every example.
        "local_to_global_policy": json.dumps(context.local_to_global_policy, ensure_ascii=False),
        "local_to_global_clause": json.dumps(context.local_to_global_clause, ensure_ascii=False),
        "global_to_local_policy": json.dumps(context.global_to_local_policy, ensure_ascii=False),
        "global_to_local_clause": json.dumps(context.global_to_local_clause, ensure_ascii=False),
        "need_tools_kwargs": True,
        "tool_selection": ["get_policy"],
        "tools_kwargs": {
            "get_policy": {
                "create_kwargs": {
                    "policy_library": context.local_policy_library,
                }
            }
        },
    }
