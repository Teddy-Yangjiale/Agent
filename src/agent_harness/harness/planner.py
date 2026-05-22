from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from agent_harness.agents.base import AgentInput, AgentStep
from agent_harness.llm.base import ChatMessage


class Planner(ABC):
    """规划器基类 — 决定 Agent 的思考策略"""

    @abstractmethod
    async def build_prompt(
        self,
        input_data: AgentInput,
        intermediate_steps: List[AgentStep],
        tools_descriptions: str,
    ) -> List[ChatMessage]:
        """构建发给 LLM 的 prompt"""
        ...

    @abstractmethod
    async def parse_output(self, text: str) -> AgentStep:
        """解析 LLM 的输出"""
        ...


class ReActPlanner(Planner):
    """ReAct 模式的规划器"""

    SYSTEM_TEMPLATE = """你是一个使用 ReAct 模式解决问题的 Agent。

可用工具：
{tool_descriptions}

## 格式要求
Question: 用户输入的问题
Thought: 你应该思考接下来怎么做
Action: 工具名称
Action Input: 工具的 JSON 格式输入
Observation: 工具返回的结果（由系统填入）
... (可重复 Thought/Action/Action Input/Observation)
Thought: 我现在知道答案了
Final Answer: 最终回复
"""

    async def build_prompt(
        self,
        input_data: AgentInput,
        intermediate_steps: List[AgentStep],
        tools_descriptions: str,
    ) -> List[ChatMessage]:
        system_msg = self.SYSTEM_TEMPLATE.format(tool_descriptions=tools_descriptions)
        messages = [ChatMessage(role="system", content=system_msg)]

        for step in intermediate_steps:
            messages.append(ChatMessage(
                role="assistant",
                content=f"Thought: {step.thought}\nAction: {step.tool_name}\nAction Input: {step.tool_input}",
            ))
            messages.append(ChatMessage(
                role="user",
                content=f"Observation: {step.tool_output}",
            ))

        messages.append(ChatMessage(role="user", content=f"Question: {input_data.query}"))
        return messages

    async def parse_output(self, text: str) -> AgentStep:
        import re
        from agent_harness.agents.base import AgentAction, AgentStep

        m = re.search(r"Final\s*Answer\s*[:：]\s*(.*)", text, re.DOTALL | re.IGNORECASE)
        if m:
            return AgentStep(action=AgentAction.FINAL_ANSWER, thought=text, final_answer=m.group(1).strip())

        thought_m = re.search(r"Thought\s*[:：]\s*(.*?)(?=Action|$)", text, re.DOTALL)
        action_m = re.search(r"Action\s*[:：]\s*(\S+)", text)
        input_m = re.search(r"Action\s*Input\s*[:：]\s*(.*)", text, re.DOTALL)

        import json
        raw = input_m.group(1).strip() if input_m else "{}"
        try:
            tool_input = json.loads(raw)
        except json.JSONDecodeError:
            tool_input = {"input": raw}

        return AgentStep(
            action=AgentAction.TOOL_CALL,
            thought=thought_m.group(1).strip() if thought_m else "",
            tool_name=action_m.group(1).strip() if action_m else "",
            tool_input=tool_input,
        )


class ToolCallingPlanner(Planner):
    """Function Calling 模式的规划器 — 由 LLM 原生支持"""

    SYSTEM_TEMPLATE = """你是一个智能 Agent，可以使用工具来完成各种任务。
1. 如果需要使用工具，调用正确的函数
2. 如果可以自行回答，直接给出回答
3. 优先使用中文回复
"""

    async def build_prompt(
        self,
        input_data: AgentInput,
        intermediate_steps: List[AgentStep],
        tools_descriptions: str,
    ) -> List[ChatMessage]:
        messages = [ChatMessage(role="system", content=self.SYSTEM_TEMPLATE)]

        for step in intermediate_steps:
            if step.action.value == "tool_call":
                messages.append(ChatMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": step.tool_name, "arguments": str(step.tool_input)},
                    }],
                ))
                messages.append(ChatMessage(
                    role="tool",
                    content=str(step.tool_output),
                    tool_call_id="call_1",
                ))

        messages.append(ChatMessage(role="user", content=input_data.query))
        return messages

    async def parse_output(self, text: str) -> AgentStep:
        from agent_harness.agents.base import AgentAction, AgentStep
        return AgentStep(action=AgentAction.FINAL_ANSWER, final_answer=text)
