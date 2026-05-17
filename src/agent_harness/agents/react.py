import re
from typing import AsyncIterator, List, Optional

from agent_harness.agents.base import (
    AgentAction,
    AgentInput,
    AgentOutput,
    AgentStep,
    BaseAgent,
)
from agent_harness.harness.planner import Planner
from agent_harness.llm.base import BaseLLM, ChatMessage
from agent_harness.tools.base import BaseTool


_REACT_SYSTEM_PROMPT = """你是一个智能 Agent，使用 ReAct 模式解决问题。

你可以使用以下工具：
{tool_descriptions}

## 响应格式

你必须严格按照以下格式回复：

Question: 用户的问题
Thought: 你需要思考接下来应该做什么
Action: 工具名称（必须是 [{tool_names}] 之一）
Action Input: 工具的输入参数（JSON 格式）
Observation: 工具返回的结果
... (这个 Thought/Action/Action Input/Observation 可以重复多次)
Thought: 我现在知道最终答案了
Final Answer: 对用户的最终回答

## 重要规则
1. 每次只能调用一个工具
2. Action Input 必须是有效的 JSON 格式
3. 当你有了最终答案时，必须以 "Final Answer:" 开头回复
4. 优先用中文思考和回答
"""


class ReActAgent(BaseAgent):
    """ReAct 模式 Agent — 模仿 LangChain ReActAgent"""

    name = "react_agent"
    description = "使用 Thought → Action → Observation 循环的 ReAct Agent"

    def __init__(
        self,
        llm: BaseLLM,
        tools: List[BaseTool],
        planner: Optional[Planner] = None,
        system_prompt: str = "",
        max_steps: int = 10,
    ):
        self._llm = llm
        self._tools = {t.name: t for t in tools}
        self._planner = planner
        self._system_prompt = system_prompt or _REACT_SYSTEM_PROMPT
        self._max_steps = max_steps

    def get_tool_names(self) -> List[str]:
        return list(self._tools.keys())

    async def plan(
        self, input_data: AgentInput, intermediate_steps: List[AgentStep]
    ) -> AgentStep:
        messages = self._build_messages(input_data, intermediate_steps)
        response = await self._llm.agenerate(messages)
        step = self._parse_react_output(response.content)
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
        tool_descriptions = "\n".join(
            f"- {name}: {t.description}" for name, t in self._tools.items()
        )
        system_content = self._system_prompt.format(
            tool_descriptions=tool_descriptions,
            tool_names=", ".join(self._tools.keys()),
        )
        messages = [ChatMessage(role="system", content=system_content)]

        for step in intermediate_steps:
            messages.append(
                ChatMessage(
                    role="assistant",
                    content=f"Thought: {step.thought}\nAction: {step.tool_name}\nAction Input: {step.tool_input}",
                )
            )
            messages.append(
                ChatMessage(
                    role="user",
                    content=f"Observation: {step.tool_output}",
                )
            )

        messages.append(
            ChatMessage(
                role="user",
                content=f"Question: {input_data.query}\n请开始思考。",
            )
        )
        return messages

    def _parse_react_output(self, text: str) -> AgentStep:
        final_answer_match = re.search(
            r"Final\s*Answer\s*[:：]\s*(.*?)$", text, re.DOTALL | re.IGNORECASE
        )
        if final_answer_match:
            return AgentStep(
                action=AgentAction.FINAL_ANSWER,
                thought=text,
                final_answer=final_answer_match.group(1).strip(),
            )

        thought_match = re.search(r"Thought\s*[:：]\s*(.*?)(?=Action\s*[:：]|$)", text, re.DOTALL)
        action_match = re.search(r"Action\s*[:：]\s*(\S+)", text)
        action_input_match = re.search(r"Action\s*Input\s*[:：]\s*(.*?)$", text, re.DOTALL)

        thought = thought_match.group(1).strip() if thought_match else ""
        tool_name = action_match.group(1).strip() if action_match else ""

        if tool_name and tool_name in self._tools:
            import json
            raw_input = action_input_match.group(1).strip() if action_input_match else "{}"
            try:
                tool_input = json.loads(raw_input)
            except json.JSONDecodeError:
                tool_input = {"input": raw_input}
            return AgentStep(
                action=AgentAction.TOOL_CALL,
                thought=thought,
                tool_name=tool_name,
                tool_input=tool_input,
            )

        return AgentStep(
            action=AgentAction.ERROR,
            thought=thought,
            error=f"无法解析 ReAct 输出: {text[:200]}",
        )
