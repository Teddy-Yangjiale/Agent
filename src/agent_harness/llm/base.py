from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

from pydantic import BaseModel
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential


@dataclass
class LLMResponse:
    """LLM 统一响应格式"""
    content: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: Dict[str, int] = field(default_factory=dict)
    raw: Any = None


@dataclass
class ToolCall:
    """LLM 返回的工具调用"""
    id: str
    name: str
    arguments: Dict[str, Any]


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ToolDefinition(BaseModel):
    """传给 LLM 的工具定义"""
    type: str = "function"
    function: Dict[str, Any]


@dataclass
class LLMClientConfig:
    """Shared timeout and retry policy for LLM providers."""

    timeout_seconds: float = 60.0
    max_retries: int = 3
    retry_min_seconds: float = 1.0
    retry_max_seconds: float = 30.0
    retry_exceptions: tuple[type[BaseException], ...] = (Exception,)


class BaseLLM(ABC):
    """LLM 抽象层 — 统一不同提供商的接口"""

    model: str = ""
    temperature: float = 0.0
    max_tokens: int = 4096
    client_config: LLMClientConfig = LLMClientConfig()

    async def _run_with_retries(self, operation):
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.client_config.max_retries),
            wait=wait_exponential(
                multiplier=1,
                min=self.client_config.retry_min_seconds,
                max=self.client_config.retry_max_seconds,
            ),
            retry=retry_if_exception_type(self.client_config.retry_exceptions),
            reraise=True,
        ):
            with attempt:
                return await operation()
        raise RuntimeError("LLM retry loop exited without result")

    @abstractmethod
    async def agenerate(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs,
    ) -> LLMResponse:
        """生成回复，可选工具调用"""
        ...

    @abstractmethod
    async def astream(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """流式生成"""
        ...
