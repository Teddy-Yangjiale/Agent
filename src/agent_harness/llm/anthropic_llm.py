from typing import Any, AsyncIterator, Dict, List, Optional

from anthropic import AsyncAnthropic
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from agent_harness.llm.base import BaseLLM, ChatMessage, LLMResponse, ToolDefinition


class AnthropicLLM(BaseLLM):
    """Anthropic Claude LLM 实现"""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = AsyncAnthropic(api_key=api_key or "sk-placeholder")

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
        system_msg = ""
        user_messages = []
        for m in messages:
            if m.role == "system":
                system_msg = m.content or ""
            else:
                user_messages.append({"role": m.role, "content": m.content or ""})

        params: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": user_messages,
            **kwargs,
        }
        if system_msg:
            params["system"] = system_msg
        if tools:
            claude_tools = []
            for t in tools:
                claude_tools.append({
                    "name": t.function["name"],
                    "description": t.function.get("description", ""),
                    "input_schema": t.function["parameters"],
                })
            params["tools"] = claude_tools

        response = await self._client.messages.create(**params)

        tool_calls = []
        content_text = ""
        for block in response.content:
            if block.type == "text":
                content_text = block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })

        return LLMResponse(
            content=content_text,
            tool_calls=tool_calls,
            finish_reason=response.stop_reason or "stop",
            usage={
                "prompt_tokens": response.usage.input_tokens if response.usage else 0,
                "completion_tokens": response.usage.output_tokens if response.usage else 0,
                "total_tokens": (response.usage.input_tokens + response.usage.output_tokens) if response.usage else 0,
            },
            raw=response,
        )

    async def astream(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        system_msg = ""
        user_messages = []
        for m in messages:
            if m.role == "system":
                system_msg = m.content or ""
            else:
                user_messages.append({"role": m.role, "content": m.content or ""})

        params: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": user_messages,
            **kwargs,
        }
        if system_msg:
            params["system"] = system_msg

        async with self._client.messages.stream(**params) as stream:
            async for text in stream.text_stream:
                yield text
