from typing import Optional

from agent_harness.llm.base import LLMClientConfig
from agent_harness.llm.openai_llm import OpenAILLM


class DeepSeekLLM(OpenAILLM):
    """DeepSeek LLM — OpenAI 兼容接口"""

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_retries: int = 3,
        timeout_seconds: float = 60.0,
        client_config: Optional[LLMClientConfig] = None,
    ):
        super().__init__(
            model=model,
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            client_config=client_config,
        )
