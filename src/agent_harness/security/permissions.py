from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional


class ToolPermission(Enum):
    ALLOW = "allow"
    DENY = "deny"
    ASK_USER = "ask_user"


@dataclass
class PermissionPolicy:
    tool_name: str
    permission: ToolPermission = ToolPermission.ALLOW
    max_calls_per_session: int = 0
    max_output_length: int = 50000
    allowed_paths: List[str] = field(default_factory=list)
    deny_paths: List[str] = field(default_factory=list)
    require_gatekeeper: bool = False


class PermissionManager:
    """工具权限管理器 — 策略引擎 + 可选人工确认"""

    def __init__(
        self,
        ask_user_callback: Optional[Callable[[str, Dict[str, Any]], Awaitable[bool]]] = None,
    ):
        self._policies: Dict[str, PermissionPolicy] = {}
        self._call_counts: Dict[str, int] = {}
        self._ask_user = ask_user_callback
        self._default_policy = PermissionPolicy(tool_name="*", permission=ToolPermission.ALLOW)

    def set_policy(self, policy: PermissionPolicy) -> PermissionManager:
        self._policies[policy.tool_name] = policy
        return self

    def set_policies(self, policies: List[PermissionPolicy]) -> PermissionManager:
        for p in policies:
            self._policies[p.tool_name] = p
        return self

    def get_policy(self, tool_name: str) -> PermissionPolicy:
        return self._policies.get(tool_name, self._default_policy)

    async def check(self, tool_name: str, tool_input: Dict[str, Any]) -> bool:
        policy = self.get_policy(tool_name)

        if policy.permission == ToolPermission.DENY:
            return False

        if policy.max_calls_per_session > 0:
            count = self._call_counts.get(tool_name, 0)
            if count >= policy.max_calls_per_session:
                return False

        if policy.require_gatekeeper or policy.permission == ToolPermission.ASK_USER:
            if self._ask_user:
                return await self._ask_user(tool_name, tool_input)
            return policy.permission == ToolPermission.ALLOW

        return True

    async def check_path(self, tool_name: str, path: str) -> bool:
        from pathlib import Path
        policy = self.get_policy(tool_name)
        resolved = Path(path).expanduser().resolve()

        for deny in policy.deny_paths:
            deny_resolved = Path(deny).expanduser().resolve()
            if self._is_relative_to(resolved, deny_resolved):
                return False

        if policy.allowed_paths:
            for allowed in policy.allowed_paths:
                allowed_resolved = Path(allowed).expanduser().resolve()
                if self._is_relative_to(resolved, allowed_resolved):
                    return True
            return False

        return True

    def _is_relative_to(self, path, parent) -> bool:
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False

    def record_call(self, tool_name: str) -> None:
        self._call_counts[tool_name] = self._call_counts.get(tool_name, 0) + 1

    def reset_counts(self) -> None:
        self._call_counts.clear()
