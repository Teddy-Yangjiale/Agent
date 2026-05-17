from __future__ import annotations

import asyncio

from agent_harness.security.sandbox import SandboxedExecutor, SandboxConfig


def test_restricted_sandbox_runs_simple_code():
    result = asyncio.run(SandboxedExecutor(SandboxConfig(use_docker=False)).execute("print(1 + 2)"))

    assert result.error == ""
    assert result.output == "3"


def test_restricted_sandbox_rejects_loops():
    result = asyncio.run(SandboxedExecutor(SandboxConfig(use_docker=False)).execute("while True:\n    pass"))

    assert "disallows While" in result.error


def test_default_sandbox_falls_back_when_docker_is_unavailable(monkeypatch):
    async def fake_docker(self, code: str):
        from agent_harness.security.sandbox import SandboxResult

        return SandboxResult(error="Docker not found — set use_docker=False for restricted mode")

    monkeypatch.setattr(SandboxedExecutor, "_execute_docker", fake_docker)

    result = asyncio.run(SandboxedExecutor(SandboxConfig()).execute("print('fallback')"))

    assert result.error == ""
    assert result.output == "fallback"
