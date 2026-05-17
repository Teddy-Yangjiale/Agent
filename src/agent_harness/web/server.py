from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from agent_harness import AgentExecutor
from agent_harness.agents.base import AgentAction, AgentInput, AgentOutput, AgentStep


class AgentAPI:
    """Agent Web API Server"""

    def __init__(self, executor: AgentExecutor, title: str = "Agent Harness", api_key: str = ""):
        self._executor = executor
        self._api_key = api_key or os.getenv("AGENT_HARNESS_API_KEY", "")
        self._app = FastAPI(title=title)
        self._connections: Dict[str, WebSocket] = {}
        self._run_history: List[dict] = []
        self._setup_routes()

    @property
    def app(self) -> FastAPI:
        return self._app

    async def _require_auth(self, x_api_key: Optional[str] = Header(default=None)) -> None:
        if self._api_key and x_api_key != self._api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
            )

    async def _authorize_websocket(self, ws: WebSocket, token: Optional[str]) -> bool:
        if not self._api_key:
            return True
        header_key = ws.headers.get("x-api-key")
        if token == self._api_key or header_key == self._api_key:
            return True
        await ws.close(code=status.WS_1008_POLICY_VIOLATION)
        return False

    def _setup_routes(self):
        @self._app.get("/")
        async def dashboard(_: None = Depends(self._require_auth)):
            return HTMLResponse(content=self._dashboard_html())

        @self._app.get("/api/health")
        async def health(_: None = Depends(self._require_auth)):
            return {"status": "ok", "agent": self._executor.agent.name, "tools": len(self._executor.tools)}

        @self._app.get("/api/tools")
        async def list_tools(_: None = Depends(self._require_auth)):
            return {"tools": [{"name": t.name, "description": t.description} for t in self._executor.tools.list_all()]}

        @self._app.get("/api/history")
        async def history(_: None = Depends(self._require_auth)):
            return {"runs": self._run_history[-20:]}

        @self._app.websocket("/ws")
        async def websocket_endpoint(ws: WebSocket, token: Optional[str] = Query(default=None)):
            if not await self._authorize_websocket(ws, token):
                return
            await ws.accept()
            conn_id = uuid.uuid4().hex[:8]
            self._connections[conn_id] = ws
            try:
                while True:
                    data = await ws.receive_json()
                    query = data.get("query", "")
                    if query:
                        run_id = uuid.uuid4().hex[:8]
                        run_record = {"id": run_id, "query": query, "steps": [], "result": "", "started": time.time()}
                        self._run_history.append(run_record)
                        await ws.send_json({"type": "run_start", "run_id": run_id, "query": query})
                        steps = []
                        async for step in self._executor.astream(query):
                            await ws.send_json({"type": "step", "run_id": run_id, "step": step.to_dict()})
                            steps.append(step)
                        last = steps[-1] if steps else AgentStep(action=AgentAction.ERROR, error="No steps")
                        result = last.final_answer or last.error or ""
                        run_record["result"] = result
                        run_record["steps"] = [s.to_dict() for s in steps]
                        run_record["done"] = time.time()
                        await ws.send_json({"type": "run_end", "run_id": run_id, "result": result})
            except WebSocketDisconnect:
                self._connections.pop(conn_id, None)

    def _dashboard_html(self) -> str:
        return """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agent Harness Dashboard</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:monospace;background:#1a1a2e;color:#e0e0e0;display:flex;height:100vh}
.sidebar{width:280px;background:#16213e;padding:20px;overflow-y:auto;border-right:1px solid #0f3460}
.main{flex:1;display:flex;flex-direction:column}
.header{background:#0f3460;padding:16px 24px;font-size:18px;font-weight:bold;color:#e94560}
.chat{flex:1;overflow-y:auto;padding:20px}
.chat .msg{margin:8px 0;padding:12px 16px;border-radius:8px;max-width:85%}
.chat .user{align-self:flex-end;background:#0f3460;color:#e0e0e0}
.chat .agent{align-self:flex-start;background:#16213e;color:#e0e0e0;border:1px solid #0f3460}
.chat .tool{background:#1a1a2e;border:1px solid #e94560;color:#e94560;font-size:13px}
.input-area{padding:16px;background:#0f3460;display:flex;gap:8px}
.input-area input{flex:1;padding:12px;border:none;border-radius:4px;background:#1a1a2e;color:#fff;font-size:14px}
.input-area button{padding:12px 24px;background:#e94560;color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:14px}
.input-area button:hover{background:#c73a54}
.sidebar h3{color:#e94560;margin-bottom:12px}
.sidebar .run{cursor:pointer;padding:8px;margin:4px 0;border-radius:4px;font-size:12px}
.sidebar .run:hover{background:#0f3460}
.sidebar .run.active{background:#e94560;color:#fff}
.scroll{display:flex;flex-direction:column;gap:4px}
</style></head>
<body>
<div class="sidebar"><h3>Runs</h3><div id="runList" class="scroll"></div></div>
<div class="main">
<div class="header">Agent Harness Dashboard</div>
<div class="chat" id="chat"><div class="msg agent">Ready. Type a query to begin.</div></div>
<div class="input-area"><input id="queryInput" placeholder="Enter your query..." onkeydown="if(event.key==='Enter')send()"><button onclick="send()">Send</button></div>
</div>
<script>
const ws=new WebSocket(`ws://${location.host}/ws`);
let currentRun=null;
ws.onmessage=function(e){const d=JSON.parse(e.data);const c=document.getElementById('chat');
if(d.type==='run_start'){currentRun=d.run_id;c.innerHTML+='<div class="msg user">'+d.query+'</div>';addRun(d.run_id,d.query)}
if(d.type==='step'){const s=d.step;
if(s.tool_name)c.innerHTML+='<div class="msg tool">TOOL: '+s.tool_name+'</div>';
if(s.final_answer)c.innerHTML+='<div class="msg agent">'+s.final_answer+'</div>';
c.scrollTop=c.scrollHeight}
if(d.type==='run_end'){c.innerHTML+='<div class="msg agent"><b>Result:</b> '+d.result+'</div>';c.scrollTop=c.scrollHeight}};
function send(){const inp=document.getElementById('queryInput');const q=inp.value.trim();if(!q)return;inp.value='';ws.send(JSON.stringify({query:q}))}
function addRun(id,q){const l=document.getElementById('runList');l.innerHTML='<div class="run active">'+q.substring(0,50)+'...</div>'+l.innerHTML}
</script></body></html>"""

    def run(self, host: str = "0.0.0.0", port: int = 8080):
        import uvicorn
        uvicorn.run(self._app, host=host, port=port, log_level="info")
