from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

from agent_harness.tools.base import BaseTool, ToolMetadata


@dataclass
class MCPServerConfig:
    name: str
    command: str = ""
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    url: str = ""
    transport: str = "stdio"


@dataclass
class MCPToolInfo:
    name: str
    description: str
    input_schema: Dict[str, Any]
    server_name: str = ""


class MCPClient:
    def __init__(self, server_configs: Optional[List[MCPServerConfig]] = None):
        self._servers: Dict[str, MCPServerConfig] = {}
        self._tools: Dict[str, MCPToolInfo] = {}
        self._http_client: Optional[httpx.AsyncClient] = None
        self._stdio_processes: Dict[str, asyncio.subprocess.Process] = {}
        self._stdio_rpc_id: Dict[str, int] = {}
        self._stdio_pending: Dict[str, Dict[int, asyncio.Future]] = {}
        if server_configs:
            for cfg in server_configs:
                self.add_server(cfg)

    def add_server(self, config: MCPServerConfig) -> None:
        self._servers[config.name] = config

    def remove_server(self, name: str) -> None:
        self._servers.pop(name, None)
        proc = self._stdio_processes.pop(name, None)
        if proc:
            try:
                proc.terminate()
            except Exception:
                pass

    async def connect(self) -> None:
        if not self._http_client:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        for name, config in self._servers.items():
            if config.transport == "stdio":
                await self._connect_stdio(name, config)
            elif config.transport in ("sse", "streamable-http") and config.url:
                await self._discover_http_tools(name, config)

    def _next_id(self, server_name: str) -> int:
        self._stdio_rpc_id[server_name] = self._stdio_rpc_id.get(server_name, 0) + 1
        return self._stdio_rpc_id[server_name]

    async def _connect_stdio(self, server_name: str, config: MCPServerConfig) -> None:
        env = {**os.environ, **config.env}
        args = [config.command] + config.args if config.command else config.args
        if not args:
            return
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            self._stdio_processes[server_name] = proc
            self._stdio_rpc_id[server_name] = 0
            self._stdio_pending[server_name] = {}
            asyncio.create_task(self._read_stdio(server_name, proc))
            init_req = {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "agent-harness-mcp", "version": "0.2.0"},
                },
                "id": self._next_id(server_name),
            }
            await self._send_stdio(server_name, init_req)
            await asyncio.sleep(0.5)
            tools_req = {
                "jsonrpc": "2.0",
                "method": "tools/list",
                "id": self._next_id(server_name),
            }
            resp = await self._send_stdio_and_wait(server_name, tools_req, timeout=10)
            if resp:
                for tool_info in resp.get("result", {}).get("tools", []):
                    full_name = f"mcp_{server_name}_{tool_info['name']}"
                    self._tools[full_name] = MCPToolInfo(
                        name=tool_info["name"],
                        description=tool_info.get("description", ""),
                        input_schema=tool_info.get("inputSchema", {}),
                        server_name=server_name,
                    )
        except Exception:
            pass

    async def _read_stdio(self, server_name: str, proc: asyncio.subprocess.Process) -> None:
        try:
            while proc.stdout and not proc.stdout.at_eof():
                line = await proc.stdout.readline()
                if not line:
                    break
                try:
                    data = json.loads(line.decode())
                    req_id = data.get("id")
                    if req_id is not None and server_name in self._stdio_pending:
                        fut = self._stdio_pending[server_name].pop(req_id, None)
                        if fut and not fut.done():
                            fut.set_result(data)
                except json.JSONDecodeError:
                    pass
        except Exception:
            pass

    async def _send_stdio(self, server_name: str, message: dict) -> None:
        proc = self._stdio_processes.get(server_name)
        if proc and proc.stdin:
            raw = (json.dumps(message) + "\n").encode()
            proc.stdin.write(raw)
            await proc.stdin.drain()

    async def _send_stdio_and_wait(self, server_name: str, message: dict, timeout: float = 30) -> Optional[dict]:
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        if server_name not in self._stdio_pending:
            self._stdio_pending[server_name] = {}
        self._stdio_pending[server_name][message["id"]] = fut
        await self._send_stdio(server_name, message)
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def _discover_http_tools(self, server_name: str, config: MCPServerConfig) -> None:
        try:
            response = await self._http_client.post(
                f"{config.url}/tools/list",
                json={"jsonrpc": "2.0", "method": "tools/list", "id": 1},
            )
            if response.status_code == 200:
                data = response.json()
                for tool_info in data.get("result", {}).get("tools", []):
                    full_name = f"mcp_{server_name}_{tool_info['name']}"
                    self._tools[full_name] = MCPToolInfo(
                        name=tool_info["name"],
                        description=tool_info.get("description", ""),
                        input_schema=tool_info.get("inputSchema", {}),
                        server_name=server_name,
                    )
        except Exception:
            pass

    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
        config = self._servers.get(server_name)
        if not config:
            raise ValueError(f"Unknown MCP server: {server_name}")
        if config.transport == "stdio":
            message = {
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
                "id": self._next_id(server_name),
            }
            resp = await self._send_stdio_and_wait(server_name, message)
            if resp:
                if "error" in resp:
                    raise RuntimeError(f"MCP call error: {resp['error']}")
                content_list = resp.get("result", {}).get("content", [])
                if content_list and isinstance(content_list, list):
                    texts = [c.get("text", str(c)) for c in content_list if isinstance(c, dict)]
                    return "\n".join(texts) or content_list
                return resp.get("result", {})
            raise RuntimeError("MCP stdio timeout or no response")
        if config.transport in ("sse", "streamable-http") and config.url:
            response = await self._http_client.post(
                f"{config.url}/tools/call",
                json={
                    "jsonrpc": "2.0",
                    "method": "tools/call",
                    "params": {"name": tool_name, "arguments": arguments},
                    "id": 2,
                },
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("result", {}).get("content", data.get("result"))
            raise RuntimeError(f"MCP HTTP call failed: {response.status_code}")
        raise NotImplementedError(f"Unsupported transport: {config.transport}")

    def get_discovered_tools(self) -> List[MCPToolInfo]:
        return list(self._tools.values())

    async def close(self) -> None:
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        for name in list(self._stdio_processes):
            proc = self._stdio_processes.pop(name, None)
            if proc:
                try:
                    proc.terminate()
                except Exception:
                    pass


class MCPToolAdapter(BaseTool):
    def __init__(self, mcp_client: MCPClient, server_name: str, tool_info: MCPToolInfo):
        self._client = mcp_client
        self._server_name = server_name
        self._tool_info = tool_info
        self.name = f"mcp_{server_name}_{tool_info.name}"
        self.description = tool_info.description
        self.metadata = ToolMetadata(
            name=self.name,
            description=tool_info.description,
            parameters_schema=tool_info.input_schema,
            category="mcp",
            tags=["mcp", server_name],
        )

    async def _arun(self, **kwargs) -> Any:
        return await self._client.call_tool(self._server_name, self._tool_info.name, kwargs)


async def discover_and_load_mcp_tools(client: MCPClient, server_configs: List[MCPServerConfig]) -> List[BaseTool]:
    for config in server_configs:
        client.add_server(config)
    await client.connect()
    tools = []
    for info in client.get_discovered_tools():
        tools.append(MCPToolAdapter(client, info.server_name, info))
    return tools
