from __future__ import annotations

import asyncio

from agent_harness.llm.base import BaseLLM, ChatMessage, LLMClientConfig, LLMResponse


class FlakyLLM(BaseLLM):
    def __init__(self):
        self.client_config = LLMClientConfig(max_retries=2, retry_min_seconds=0, retry_max_seconds=0)
        self.calls = 0

    async def agenerate(self, messages, tools=None, **kwargs):
        return await self._run_with_retries(self._once)

    async def _once(self):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary")
        return LLMResponse(content="ok")

    async def astream(self, messages, tools=None, **kwargs):
        yield "ok"


def test_llm_retry_policy_is_configurable():
    llm = FlakyLLM()

    result = asyncio.run(llm.agenerate([ChatMessage(role="user", content="hi")]))

    assert result.content == "ok"
    assert llm.calls == 2
