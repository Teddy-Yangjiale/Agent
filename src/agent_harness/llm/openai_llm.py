from typing import Any, AsyncIterator, Dict, List, Optional

from openai import AsyncOpenAI

from agent_harness.llm.base import BaseLLM, ChatMessage, LLMClientConfig, LLMResponse, ToolDefinition


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
        timeout_seconds: float = 60.0,
        client_config: Optional[LLMClientConfig] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client_config = client_config or LLMClientConfig(
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        self._client = AsyncOpenAI(
            api_key=api_key or "sk-placeholder",
            base_url=base_url or "https://api.openai.com/v1",
            max_retries=0,
            timeout=self.client_config.timeout_seconds,
        )

    async def agenerate(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs,
    ) -> LLMResponse:
        return await self._run_with_retries(
            lambda: self._agenerate_once(messages=messages, tools=tools, **kwargs)
        )

    async def _agenerate_once(
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

    async def astream(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        stream = await self._run_with_retries(
            lambda: self._create_stream(messages=messages, tools=tools, **kwargs)
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

    async def _create_stream(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs,
    ):
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

        return await self._client.chat.completions.create(**params)
