from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Message:
    """记忆中的一条消息"""
    role: str  # "user" | "assistant" | "system" | "tool"
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseMemory(ABC):
    """记忆基类 — 模仿 LangChain BaseMemory"""

    @abstractmethod
    async def aload_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """加载记忆变量，注入到 agent 上下文中"""
        ...

    @abstractmethod
    async def asave_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """保存本轮交互到记忆中"""
        ...

    @abstractmethod
    async def aclear(self) -> None:
        """清空记忆"""
        ...

    @abstractmethod
    def get_messages(self) -> List[Message]:
        """获取全部消息历史"""
        ...
