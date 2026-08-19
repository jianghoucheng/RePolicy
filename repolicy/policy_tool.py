"""The get_policy tool RePolicy invokes during multi-turn rollout."""

from __future__ import annotations

import json
from typing import Any, Optional
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse


MAX_POLICIES_PER_CALL = 5


class PolicyLookupTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instances: dict[str, dict[str, Any]] = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        create_kwargs = kwargs.get("create_kwargs", {}) or {}
        policies = create_kwargs.get("policy_library", [])
        self._instances[instance_id] = {p["policy_id"]: p for p in policies if "policy_id" in p}
        return instance_id, ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        policy_ids = parameters.get("policy_ids", [])
        if isinstance(policy_ids, str):
            policy_ids = [policy_ids]
        if not isinstance(policy_ids, list):
            policy_ids = []
        if len(policy_ids) > MAX_POLICIES_PER_CALL:
            payload = {
                "error": "too_many_policies",
                "message": f"Call get_policy once with at most {MAX_POLICIES_PER_CALL} relevant policy ids.",
                "requested_policy_count": len(policy_ids),
            }
            return ToolResponse(text=json.dumps(payload, ensure_ascii=False, indent=2)), 0.0, {
                "requested_policy_count": len(policy_ids),
                "returned_policy_count": 0,
            }
        library = self._instances.get(instance_id, {})
        selected = [library[pid] for pid in policy_ids if isinstance(pid, str) and pid in library]
        missing = [pid for pid in policy_ids if isinstance(pid, str) and pid not in library]
        payload = {"policies": selected}
        if missing:
            payload["missing_policy_ids"] = missing
        return ToolResponse(text=json.dumps(payload, ensure_ascii=False, indent=2)), 0.0, {
            "requested_policy_count": len(policy_ids),
            "returned_policy_count": len(selected),
        }

    async def release(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)
