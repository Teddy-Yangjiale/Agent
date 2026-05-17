from typing import Optional

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
    ):
        super().__init__(
            model=model,
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
