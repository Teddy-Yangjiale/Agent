from collections import deque
from typing import Any, Dict, List

from agent_harness.memory.base import BaseMemory, Message


class ConversationBufferMemory(BaseMemory):
    """滑动窗口记忆 — 最近 N 轮对话"""

    def __init__(self, max_messages: int = 20, max_token_limit: int = 8000):
        self._messages: deque[Message] = deque(maxlen=max_messages)
        self._max_token_limit = max_token_limit

    async def aload_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "chat_history": list(self._messages),
            "history_text": self._format_history(),
        }

    def _format_history(self) -> str:
        lines = []
        for m in self._messages:
            lines.append(f"[{m.role}]: {m.content}")
        return "\n".join(lines)

    async def asave_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        if "query" in inputs:
            self._messages.append(Message(role="user", content=inputs["query"]))
        if "result" in outputs:
            self._messages.append(Message(role="assistant", content=outputs["result"]))

    async def aclear(self) -> None:
        self._messages.clear()

    def get_messages(self) -> List[Message]:
        return list(self._messages)

    def add_message(self, role: str, content: str) -> None:
        self._messages.append(Message(role=role, content=content))


class ConversationSummaryMemory(ConversationBufferMemory):
    """带摘要压缩的记忆 — 超出 token 限制时自动压缩"""

    def __init__(
        self,
        llm=None,
        max_messages: int = 30,
        max_token_limit: int = 4000,
        summary_prompt: str = "请用简短的中文总结以下对话的要点：\n{history}",
    ):
        super().__init__(max_messages=max_messages, max_token_limit=max_token_limit)
        self._llm = llm
        self._summary_prompt = summary_prompt
        self._summary: str = ""

    async def aload_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        base = await super().aload_memory_variables(inputs)
        base["summary"] = self._summary
        return base

    async def asave_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        await super().asave_context(inputs, outputs)
        estimated_tokens = len(self._format_history()) // 2
        if estimated_tokens > self._max_token_limit and self._llm:
            await self._compress()

    async def _compress(self) -> None:
        full_history = self._format_history()
        prompt = self._summary_prompt.format(history=full_history)
        try:
            from agent_harness.llm.base import ChatMessage
            response = await self._llm.agenerate([
                ChatMessage(role="user", content=prompt)
            ])
            self._summary = response.content
            self._messages.clear()
            self._messages.append(
                Message(role="system", content=f"对话摘要: {self._summary}")
            )
        except Exception:
            pass
