# -*- coding: utf-8 -*-
"""
Agent Harness 使用示例

设置环境变量:
    $env:DEEPSEEK_API_KEY = "sk-your-deepseek-key"

支持任何 OpenAI 兼容 API:
    $env:DEEPSEEK_API_KEY = "sk-xxx"
    $env:DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"  # 可选
"""

import asyncio
import os


async def demo_basic():
    from agent_harness import AgentExecutor
    from agent_harness.agents import ReActAgent
    from agent_harness.harness.agent_executor import AgentExecutorConfig
    from agent_harness.harness.callbacks import LoggingCallback, TokenCounterCallback
    from agent_harness.llm import OpenAILLM
    from agent_harness.memory import ConversationBufferMemory
    from agent_harness.tools import ToolRegistry, CalculatorTool

    api_key = os.getenv("DEEPSEEK_API_KEY", "sk-placeholder")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

    llm = OpenAILLM(
        model="deepseek-chat",
        api_key=api_key,
        base_url=base_url,
    )
    registry = ToolRegistry().register(CalculatorTool())
    memory = ConversationBufferMemory(max_messages=20)
    token_counter = TokenCounterCallback()

    agent = ReActAgent(llm=llm, tools=registry.list_all())
    executor = AgentExecutor(
        agent=agent,
        tool_registry=registry,
        memory=memory,
        callbacks=[LoggingCallback(), token_counter],
        config=AgentExecutorConfig(max_steps=5, enforce_timeout=True),
    )

    result = await executor.arun("计算 (123 + 456) * 789")
    print(f"Result: {result.result}")
    print(f"Duration: {result.total_duration_ms:.0f}ms")
    print(f"Tokens: {result.total_tokens}")
    print(f"Tool calls: {result.tool_calls_count}")


async def demo_cache():
    from agent_harness.cache import LLMCache
    from agent_harness.llm import OpenAILLM, ChatMessage

    api_key = os.getenv("DEEPSEEK_API_KEY", "sk-placeholder")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

    cache = LLMCache(max_entries=100, ttl_seconds=3600)
    llm = OpenAILLM(model="deepseek-chat", api_key=api_key, base_url=base_url)
    messages = [ChatMessage(role="user", content="What is 2+2?")]
    response = await llm.agenerate(messages)
    cache.set([m.model_dump() for m in messages], None, response)
    cached = cache.get([m.model_dump() for m in messages])
    print(f"Cache hit: {cached is not None}")


async def demo_cost():
    from agent_harness.cache import CostTracker
    tracker = CostTracker()
    cost = tracker.record("deepseek-chat", prompt_tokens=1000, completion_tokens=500)
    print(f"Cost: ${cost:.6f}")
    print(f"Summary: {tracker.summary()}")


async def demo_permissions():
    from agent_harness.security.permissions import PermissionManager, PermissionPolicy, ToolPermission
    pm = PermissionManager()
    pm.set_policy(PermissionPolicy(tool_name="write_file", permission=ToolPermission.ASK_USER))
    allowed = await pm.check("read_file", {})
    print(f"read_file allowed: {allowed}")


async def demo_streaming():
    from agent_harness import AgentExecutor
    from agent_harness.agents import ReActAgent
    from agent_harness.llm import OpenAILLM
    from agent_harness.tools import ToolRegistry, CalculatorTool

    api_key = os.getenv("DEEPSEEK_API_KEY", "sk-placeholder")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

    llm = OpenAILLM(model="deepseek-chat", api_key=api_key, base_url=base_url)
    registry = ToolRegistry().register(CalculatorTool())
    agent = ReActAgent(llm=llm, tools=registry.list_all(), max_steps=3)
    executor = AgentExecutor(agent=agent, tool_registry=registry)

    print("\nStreaming steps:")
    async for step in executor.astream("1+2+3+4+5?"):
        if step.action.value == "tool_call":
            print(f"  [tool] {step.tool_name} -> {step.tool_output}")
        elif step.action.value == "final_answer":
            print(f"  [answer] {step.final_answer}")


async def main():
    demos = [
        ("Basic Agent", demo_basic),
        ("LLM Cache", demo_cache),
        ("Cost Tracking", demo_cost),
        ("Permission System", demo_permissions),
        ("Streaming", demo_streaming),
    ]
    for name, fn in demos:
        print(f"\n{'='*50}\n  {name}\n{'='*50}")
        try:
            await fn()
        except Exception as e:
            print(f"  Skipped: {e}")


if __name__ == "__main__":
    asyncio.run(main())
