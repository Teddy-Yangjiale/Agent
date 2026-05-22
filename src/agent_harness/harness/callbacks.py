from __future__ import annotations

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
import time
import uuid

from agent_harness.agents.base import AgentStep


@dataclass
class CallbackEvent:
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    event_type: str = ""
    agent_name: str = ""
    step: Optional[AgentStep] = None
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class BaseCallback:
    def __init__(self):
        self._handlers = {
            "agent_start": self.on_agent_start,
            "agent_step": self.on_agent_step,
            "tool_start": self.on_tool_start,
            "tool_end": self.on_tool_end,
            "agent_end": self.on_agent_end,
            "agent_error": self.on_agent_error,
            "llm_token": self.on_llm_token,
        }

    async def on_agent_start(self, event: CallbackEvent) -> None: pass
    async def on_agent_step(self, event: CallbackEvent) -> None: pass
    async def on_tool_start(self, event: CallbackEvent) -> None: pass
    async def on_tool_end(self, event: CallbackEvent) -> None: pass
    async def on_agent_end(self, event: CallbackEvent) -> None: pass
    async def on_agent_error(self, event: CallbackEvent) -> None: pass
    async def on_llm_token(self, event: CallbackEvent) -> None: pass


class CallbackManager:
    def __init__(self, callbacks: Optional[List[BaseCallback]] = None):
        self._callbacks: List[BaseCallback] = callbacks or []

    def add(self, callback: BaseCallback) -> None:
        self._callbacks.append(callback)

    def remove(self, callback: BaseCallback) -> None:
        self._callbacks.remove(callback)

    async def notify(self, event: CallbackEvent) -> None:
        for cb in self._callbacks:
            handler = cb._handlers.get(event.event_type)
            if handler:
                try:
                    await handler(event)
                except Exception:
                    if handler.__code__.co_name != "on_llm_token":
                        pass


class LoggingCallback(BaseCallback):
    def __init__(self, logger=None):
        super().__init__()
        import structlog
        self._logger = logger or structlog.get_logger()

    async def on_agent_start(self, event: CallbackEvent) -> None:
        self._logger.info("agent.start", agent=event.agent_name, query=event.data.get("query", ""))

    async def on_agent_step(self, event: CallbackEvent) -> None:
        if event.step:
            self._logger.info("agent.step", action=event.step.action.value,
                            thought=event.step.thought[:100] if event.step.thought else "")

    async def on_tool_start(self, event: CallbackEvent) -> None:
        self._logger.info("tool.start", tool=event.step.tool_name if event.step else "")

    async def on_tool_end(self, event: CallbackEvent) -> None:
        if event.step:
            self._logger.info("tool.end", tool=event.step.tool_name,
                            output=str(event.step.tool_output)[:200])

    async def on_agent_end(self, event: CallbackEvent) -> None:
        self._logger.info("agent.end", result=event.data.get("result", "")[:200])

    async def on_agent_error(self, event: CallbackEvent) -> None:
        self._logger.error("agent.error", error=event.data.get("error", ""))

    async def on_llm_token(self, event: CallbackEvent) -> None:
        pass


class TokenCounterCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    async def on_agent_step(self, event: CallbackEvent) -> None:
        usage = event.data.get("usage", {})
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        self.total_tokens += usage.get("total_tokens", 0)

    async def on_agent_end(self, event: CallbackEvent) -> None:
        usage = event.data.get("usage", {})
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        self.total_tokens += usage.get("total_tokens", 0)


class StreamCallback(BaseCallback):
    """流式输出回调 — token 级别实时推送"""

    def __init__(self, on_token=None):
        super().__init__()
        self._on_token = on_token

    async def on_llm_token(self, event: CallbackEvent) -> None:
        if self._on_token:
            token = event.data.get("token", "")
            if token:
                self._on_token(token)
