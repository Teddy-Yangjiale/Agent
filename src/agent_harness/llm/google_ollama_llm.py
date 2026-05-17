from typing import Any, AsyncIterator, Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from agent_harness.llm.base import BaseLLM, ChatMessage, LLMResponse, ToolDefinition


class GoogleLLM(BaseLLM):
    """Google Gemini LLM 实现"""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._api_key = api_key or "placeholder"
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)

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
        self._ensure_client()
        from google.genai import types

        contents = []
        for m in messages:
            contents.append(types.Content(
                role="user" if m.role in ("user", "system") else "model",
                parts=[types.Part.from_text(text=m.content or "")],
            ))

        tool_list = None
        if tools:
            tool_list = []
            for t in tools:
                tool_list.append(types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name=t.function["name"],
                            description=t.function.get("description", ""),
                            parameters=t.function["parameters"],
                        )
                    ]
                ))

        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                tools=tool_list,
            ),
        )

        return LLMResponse(
            content=response.text or "",
            finish_reason="stop",
            usage={
                "prompt_tokens": response.usage_metadata.prompt_token_count if response.usage_metadata else 0,
                "completion_tokens": response.usage_metadata.candidates_token_count if response.usage_metadata else 0,
                "total_tokens": response.usage_metadata.total_token_count if response.usage_metadata else 0,
            },
            raw=response,
        )

    async def astream(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[ToolDefinition]] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        self._ensure_client()
        from google.genai import types

        contents = []
        for m in messages:
            contents.append(types.Content(
                role="user" if m.role in ("user", "system") else "model",
                parts=[types.Part.from_text(text=m.content or "")],
            ))

        response = await self._client.aio.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        )
        async for chunk in response:
            if chunk.text:
                yield chunk.text


class OllamaLLM(BaseLLM):
    """Ollama 本地模型 LLM 实现 — 兼容 OpenAI API"""

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434/v1",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        from openai import AsyncOpenAI
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = AsyncOpenAI(base_url=base_url, api_key="ollama")

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
                {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
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
