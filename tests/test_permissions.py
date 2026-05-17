from __future__ import annotations

import asyncio

from agent_harness.security.permissions import PermissionManager, PermissionPolicy


def test_allowed_path_uses_real_path_boundaries(tmp_path):
    allowed = tmp_path / "allowed"
    sibling = tmp_path / "allowed_sibling"
    allowed.mkdir()
    sibling.mkdir()

    manager = PermissionManager().set_policy(
        PermissionPolicy(tool_name="read_file", allowed_paths=[str(allowed)])
    )

    assert asyncio.run(manager.check_path("read_file", str(allowed / "ok.txt")))
    assert not asyncio.run(manager.check_path("read_file", str(sibling / "blocked.txt")))
