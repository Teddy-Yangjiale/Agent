from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional


class AgentAction(Enum):
    """Agent 动作类型"""
    THINK = "think"
    TOOL_CALL = "tool_call"
    FINAL_ANSWER = "final_answer"
    ERROR = "error"


@dataclass
class AgentStep:
    """Agent 单步执行的记录"""
    action: AgentAction
    thought: str = ""
    tool_name: str = ""
    tool_input: Dict[str, Any] = field(default_factory=dict)
    tool_output: Any = None
    final_answer: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "thought": self.thought,
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "tool_output": str(self.tool_output)[:500] if self.tool_output else None,
            "final_answer": self.final_answer,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class AgentInput:
    """Agent 输入"""
    query: str
    context: Dict[str, Any] = field(default_factory=dict)
    max_steps: int = 10
    allowed_tools: Optional[List[str]] = None


@dataclass
class AgentOutput:
    """Agent 输出"""
    result: str
    steps: List[AgentStep] = field(default_factory=list)
    total_tokens: int = 0
    total_duration_ms: float = 0.0
    tool_calls_count: int = 0


class BaseAgent(ABC):
    """Agent 基类 — 模仿 LangChain BaseSingleActionAgent"""

    name: str = "base_agent"
    description: str = "Base agent"

    @abstractmethod
    async def plan(
        self, input_data: AgentInput, intermediate_steps: List[AgentStep]
    ) -> AgentStep:
        """根据当前输入和历史步骤，决定下一步行动"""
        ...

    @abstractmethod
    async def astream(
        self, input_data: AgentInput
    ) -> AsyncIterator[AgentStep]:
        """流式执行 agent 循环"""
        ...

    @abstractmethod
    async def arun(self, input_data: AgentInput) -> AgentOutput:
        """完整执行 agent 循环，返回最终结果"""
        ...

    def get_tool_names(self) -> List[str]:
        """返回该 agent 可用的工具名称列表"""
        return []
