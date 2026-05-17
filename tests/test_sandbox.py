from __future__ import annotations

import asyncio

from agent_harness.security.sandbox import SandboxedExecutor, SandboxConfig


def test_restricted_sandbox_runs_simple_code():
    result = asyncio.run(SandboxedExecutor(SandboxConfig()).execute("print(1 + 2)"))

    assert result.error == ""
    assert result.output == "3"


def test_restricted_sandbox_rejects_loops():
    result = asyncio.run(SandboxedExecutor(SandboxConfig()).execute("while True:\n    pass"))

    assert "disallows While" in result.error
