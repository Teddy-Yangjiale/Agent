from __future__ import annotations

import asyncio

from agent_harness.mcp.client import MCPClient, MCPServerConfig


def test_mcp_records_connection_errors():
    client = MCPClient([
        MCPServerConfig(name="missing", command="definitely-not-a-real-mcp-command")
    ])

    asyncio.run(client.connect())

    assert client.errors
    assert client.errors[0].server_name == "missing"
    assert client.errors[0].operation == "connect_stdio"
