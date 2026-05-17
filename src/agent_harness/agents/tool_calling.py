import json
from typing import AsyncIterator, List, Optional

from agent_harness.agents.base import (
    AgentAction,
    AgentInput,
    AgentOutput,
    AgentStep,
    BaseAgent,
)
from agent_harness.llm.base import BaseLLM, ChatMessage, LLMResponse, ToolDefinition
from agent_harness.tools.base import BaseTool


_TOOL_CALLING_SYSTEM_PROMPT = """你是一个智能 Agent，可以使用工具来完成任务。

## 规则
1. 分析用户需求，选择合适的工具
2. 如果不需要工具，直接回答用户
3. 如果需要工具，调用正确的工具并基于结果作答
4. 优先用中文回复
"""


class ToolCallingAgent(BaseAgent):
    """原生 Function Calling Agent — 利用 LLM 的 tool_use 能力"""

    name = "tool_calling_agent"
    description = "使用 LLM 原生 function calling 的 Agent"

    def __init__(
        self,
        llm: BaseLLM,
        tools: List[BaseTool],
        system_prompt: str = "",
        max_steps: int = 10,
    ):
        self._llm = llm
        self._tools = {t.name: t for t in tools}
        self._system_prompt = system_prompt or _TOOL_CALLING_SYSTEM_PROMPT
        self._max_steps = max_steps

    def get_tool_names(self) -> List[str]:
        return list(self._tools.keys())

    async def plan(
        self, input_data: AgentInput, intermediate_steps: List[AgentStep]
    ) -> AgentStep:
        messages = self._build_messages(input_data, intermediate_steps)
        tool_defs = self._build_tool_definitions()
        response = await self._llm.agenerate(messages, tools=tool_defs)
        step = self._parse_response(response)
        step.metadata.update({
            "llm_usage": response.usage,
            "llm_finish_reason": response.finish_reason,
        })
        return step

    async def astream(self, input_data: AgentInput) -> AsyncIterator[AgentStep]:
        steps = []
        for _ in range(input_data.max_steps or self._max_steps):
            step = await self.plan(input_data, steps)
            yield step
            steps.append(step)
            if step.action == AgentAction.FINAL_ANSWER:
                break
            if step.action == AgentAction.ERROR:
                break

    async def arun(self, input_data: AgentInput) -> AgentOutput:
        steps = []
        async for step in self.astream(input_data):
            steps.append(step)
        last = steps[-1] if steps else AgentStep(action=AgentAction.ERROR, error="No steps executed")
        return AgentOutput(
            result=last.final_answer or last.error or "",
            steps=steps,
            tool_calls_count=sum(1 for s in steps if s.action == AgentAction.TOOL_CALL),
        )

    def _build_messages(
        self, input_data: AgentInput, intermediate_steps: List[AgentStep]
    ) -> List[ChatMessage]:
        messages = [ChatMessage(role="system", content=self._system_prompt)]

        for step in intermediate_steps:
            if step.action == AgentAction.THINK:
                messages.append(ChatMessage(role="assistant", content=step.thought))
            elif step.action == AgentAction.TOOL_CALL:
                messages.append(
                    ChatMessage(
                        role="assistant",
                        content=None,
                        tool_calls=[{
                            "id": f"call_{len(intermediate_steps)}",
                            "type": "function",
                            "function": {
                                "name": step.tool_name,
                                "arguments": json.dumps(step.tool_input, ensure_ascii=False),
                            },
                        }],
                    )
                )
                messages.append(
                    ChatMessage(
                        role="tool",
                        content=str(step.tool_output),
                        tool_call_id=f"call_{len(intermediate_steps)}",
                    )
                )

        messages.append(ChatMessage(role="user", content=input_data.query))
        return messages

    def _build_tool_definitions(self) -> List[ToolDefinition]:
        defs = []
        for tool in self._tools.values():
            schema = tool.get_schema()
            defs.append(ToolDefinition(
                type="function",
                function={
                    "name": schema.name,
                    "description": schema.description,
                    "parameters": schema.parameters_schema,
                },
            ))
        return defs

    def _parse_response(self, response: LLMResponse) -> AgentStep:
        if response.tool_calls:
            tc = response.tool_calls[0]
            try:
                arguments = json.loads(tc["arguments"]) if isinstance(tc["arguments"], str) else tc["arguments"]
            except json.JSONDecodeError:
                arguments = {"input": tc["arguments"]}
            return AgentStep(
                action=AgentAction.TOOL_CALL,
                thought=response.content,
                tool_name=tc["name"],
                tool_input=arguments,
            )

        return AgentStep(
            action=AgentAction.FINAL_ANSWER,
            thought="",
            final_answer=response.content,
        )
