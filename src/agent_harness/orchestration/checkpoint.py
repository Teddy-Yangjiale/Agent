import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_harness.agents.base import AgentAction, AgentStep


@dataclass
class Checkpoint:
    step_index: int
    agent_name: str
    step: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CheckpointManager:
    """检查点管理器 — 自动保存/恢复 Agent 状态"""

    def __init__(self, base_dir: str = "checkpoints", auto_save: bool = True):
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._auto_save = auto_save
        self._current_run: str = ""

    def start_run(self, run_id: str = "") -> str:
        self._current_run = run_id or f"run_{int(time.time())}"
        (self._base_dir / self._current_run).mkdir(parents=True, exist_ok=True)
        return self._current_run

    async def save_step(self, step: AgentStep, step_index: int) -> None:
        if not self._auto_save or not self._current_run:
            return
        cp = Checkpoint(
            step_index=step_index,
            agent_name="agent",
            step=step.to_dict(),
        )
        filepath = self._base_dir / self._current_run / f"step_{step_index:04d}.json"
        filepath.write_text(json.dumps(cp.__dict__, indent=2, ensure_ascii=False))

    async def save_query(self, query: str) -> None:
        if not self._current_run:
            return
        filepath = self._base_dir / self._current_run / "query.json"
        filepath.write_text(json.dumps({"query": query, "timestamp": time.time()}, indent=2))

    def get_latest_step(self) -> Optional[dict]:
        if not self._current_run:
            return None
        run_dir = self._base_dir / self._current_run
        if not run_dir.exists():
            return None
        files = sorted(run_dir.glob("step_*.json"))
        if not files:
            return None
        return json.loads(files[-1].read_text())

    def list_runs(self) -> List[dict]:
        runs = []
        for d in sorted(self._base_dir.glob("run_*"), reverse=True):
            query_file = d / "query.json"
            query = ""
            if query_file.exists():
                try:
                    query = json.loads(query_file.read_text()).get("query", "")
                except Exception:
                    pass
            step_count = len(list(d.glob("step_*.json")))
            runs.append({"id": d.name, "query": query[:100], "steps": step_count, "timestamp": d.stat().st_mtime})
        return runs

    def cleanup(self, max_runs: int = 50) -> None:
        runs = sorted(self._base_dir.glob("run_*"), key=lambda p: p.stat().st_mtime)
        for d in runs[:-max_runs]:
            for f in d.iterdir():
                f.unlink()
            d.rmdir()
