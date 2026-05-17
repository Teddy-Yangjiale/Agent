from __future__ import annotations

import asyncio

from agent_harness.agents.base import AgentAction, AgentInput, AgentOutput, AgentStep, BaseAgent
from agent_harness.harness.agent_executor import AgentExecutor
from agent_harness.security.permissions import PermissionManager, PermissionPolicy
from agent_harness.tools.base import BaseTool, ToolMetadata
from agent_harness.tools.registry import ToolRegistry


class ScriptedAgent(BaseAgent):
    def __init__(self, steps: list[AgentStep]):
        self._steps = steps
        self._idx = 0

    async def plan(self, input_data: AgentInput, intermediate_steps: list[AgentStep]) -> AgentStep:
        step = self._steps[min(self._idx, len(self._steps) - 1)]
        self._idx += 1
        return step

    async def astream(self, input_data: AgentInput):
        yield self._steps[0]

    async def arun(self, input_data: AgentInput) -> AgentOutput:
        return AgentOutput(result="")


class EchoTool(BaseTool):
    name = "echo"
    description = "echo input"

    async def _arun(self, input: str = "", **kwargs):
        return input


class WriteLikeTool(BaseTool):
    name = "write_file"
    description = "write content"
    metadata = ToolMetadata(
        name="write_file",
        description="write content",
        parameters_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
        requires_confirmation=True,
    )

    async def _arun(self, path: str = "", content: str = "", **kwargs):
        return "written"


def test_executor_accumulates_llm_usage():
    agent = ScriptedAgent([
        AgentStep(
            action=AgentAction.FINAL_ANSWER,
            final_answer="done",
            metadata={"llm_usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}},
        )
    ])
    result = asyncio.run(AgentExecutor(agent, ToolRegistry()).arun("hello"))

    assert result.result == "done"
    assert result.total_tokens == 5


def test_executor_rejects_disallowed_tool():
    agent = ScriptedAgent([AgentStep(action=AgentAction.TOOL_CALL, tool_name="echo", tool_input={"input": "x"})])
    registry = ToolRegistry().register(EchoTool())

    result = asyncio.run(AgentExecutor(agent, registry).arun("hello", allowed_tools=[]))

    assert result.steps[0].action == AgentAction.ERROR
    assert result.steps[0].error == "tool_not_allowed"


def test_executor_checks_path_before_confirmed_tool_runs(tmp_path):
    outside = tmp_path.parent / "outside.txt"
    agent = ScriptedAgent([
        AgentStep(
            action=AgentAction.TOOL_CALL,
            tool_name="write_file",
            tool_input={"path": str(outside), "content": "x"},
        )
    ])
    registry = ToolRegistry().register(WriteLikeTool())
    permissions = PermissionManager().set_policy(
        PermissionPolicy(tool_name="write_file", allowed_paths=[str(tmp_path)])
    )

    result = AgentExecutor(agent, registry, permission_manager=permissions)
    output = asyncio.run(result.arun("write"))

    assert output.steps[0].error == "path_denied"
    assert not outside.exists()
