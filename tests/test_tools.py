from __future__ import annotations

import asyncio

import pytest

from agent_harness.tools.base import BaseTool, FunctionTool, ToolMetadata, ToolValidationError


class StrictTool(BaseTool):
    name = "strict"
    description = "strict test tool"
    metadata = ToolMetadata(
        name="strict",
        description="strict test tool",
        parameters_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"},
            },
            "required": ["name", "count"],
        },
    )

    async def _arun(self, name: str = "", count: int = 0):
        return f"{name}:{count}"


def test_tool_validates_declared_schema():
    result = asyncio.run(StrictTool().arun(name="job", count=2))

    assert result == "job:2"


def test_tool_rejects_missing_required_input():
    with pytest.raises(ToolValidationError):
        asyncio.run(StrictTool().arun(name="job"))


def test_function_tool_rejects_extra_input():
    async def greet(name: str) -> str:
        return f"hello {name}"

    tool = FunctionTool(greet)

    with pytest.raises(ToolValidationError):
        asyncio.run(tool.arun(name="Ada", unexpected="x"))
