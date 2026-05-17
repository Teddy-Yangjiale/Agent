from typing import Any, AsyncIterator, Dict, List, Optional

from openai import AsyncOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from agent_harness.llm.base import BaseLLM, ChatMessage, LLMResponse, ToolCall, ToolDefinition


class OpenAILLM(BaseLLM):
    """OpenAI 兼容的 LLM 实现 — 带指数退避重试"""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_retries: int = 3,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._max_retries = max_retries
        self._client = AsyncOpenAI(
            api_key=api_key or "sk-placeholder",
            base_url=base_url or "https://api.openai.com/v1",
            max_retries=0,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    async def agenerate(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs,
    ) -> LLMResponse:
        openai_messages = [m.model_dump(exclude_none=True) for m in messages]
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **kwargs,
        }
        if tools:
            params["tools"] = [t.model_dump() for t in tools]

        response = await self._client.chat.completions.create(**params)
        choice = response.choices[0]
        message = choice.message

        tool_calls = []
        if message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
                for tc in message.tool_calls
            ]

        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            raw=response,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    async def astream(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        openai_messages = [m.model_dump(exclude_none=True) for m in messages]
        params: Dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
            **kwargs,
        }
        if tools:
            params["tools"] = [t.model_dump() for t in tools]

        stream = await self._client.chat.completions.create(**params)
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content
