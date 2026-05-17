from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite

from agent_harness.memory.base import BaseMemory, Message


class PersistentMemory(BaseMemory):
    """SQLite 持久化记忆 — 会话历史永久存储"""

    def __init__(self, db_path: str = "agent_memory.db", session_id: str = ""):
        self._db_path = db_path
        self._session_id = session_id or uuid.uuid4().hex[:12]
        self._db: Optional[aiosqlite.Connection] = None

    async def _ensure_db(self):
        if self._db is None:
            self._db = await aiosqlite.connect(self._db_path)
            self._db.row_factory = aiosqlite.Row
            await self._db.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    created_at REAL NOT NULL
                )
            """)
            await self._db.execute("CREATE INDEX IF NOT EXISTS idx_session ON messages(session_id)")
            await self._db.commit()

    async def aload_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        await self._ensure_db()
        cursor = await self._db.execute(
            "SELECT role, content, metadata FROM messages WHERE session_id = ? ORDER BY id ASC LIMIT 50",
            (self._session_id,),
        )
        rows = await cursor.fetchall()
        messages = [Message(role=r[0], content=r[1], metadata=json.loads(r[2])) for r in rows]
        return {"chat_history": messages, "history_text": "\n".join(f"[{m.role}]: {m.content}" for m in messages)}

    async def asave_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        await self._ensure_db()
        now = time.time()
        if "query" in inputs:
            await self._db.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (self._session_id, "user", inputs["query"], now),
            )
        if "result" in outputs:
            await self._db.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (self._session_id, "assistant", outputs["result"], now),
            )
        await self._db.commit()

    async def aclear(self) -> None:
        await self._ensure_db()
        await self._db.execute("DELETE FROM messages WHERE session_id = ?", (self._session_id,))
        await self._db.commit()

    def get_messages(self) -> List[Message]:
        return []

    async def list_sessions(self) -> List[Dict[str, Any]]:
        await self._ensure_db()
        cursor = await self._db.execute(
            "SELECT session_id, MIN(created_at) as started, COUNT(*) as msg_count FROM messages GROUP BY session_id ORDER BY started DESC"
        )
        rows = await cursor.fetchall()
        return [{"id": r[0], "started": r[1], "messages": r[2]} for r in rows]

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None


class RAGMemory(PersistentMemory):
    """RAG 增强记忆 — SQLite + ChromaDB 向量检索"""

    def __init__(self, db_path: str = "agent_memory.db", collection_name: str = "agent_memory", session_id: str = ""):
        super().__init__(db_path, session_id)
        self._collection_name = collection_name
        self._chroma = None

    def _ensure_chroma(self):
        if self._chroma is None:
            try:
                import chromadb
                client = chromadb.PersistentClient(path=str(Path(self._db_path).parent / "chroma"))
                self._chroma = client.get_or_create_collection(self._collection_name)
            except ImportError:
                self._chroma = None
            except Exception:
                self._chroma = None

    async def asave_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        await super().asave_context(inputs, outputs)
        self._ensure_chroma()
        if self._chroma and "query" in inputs:
            try:
                content = inputs["query"][:1000]
                self._chroma.add(
                    documents=[content],
                    metadatas=[{"session_id": self._session_id, "role": "user"}],
                    ids=[f"{self._session_id}_{int(time.time() * 1000)}"],
                )
            except Exception:
                pass

    async def search_similar(self, query: str, k: int = 5) -> List[str]:
        self._ensure_chroma()
        if not self._chroma:
            return []
        try:
            results = self._chroma.query(query_texts=[query], n_results=k)
            return results.get("documents", [[]])[0] if results else []
        except Exception:
            return []

    async def aload_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        base = await super().aload_memory_variables(inputs)
        query = inputs.get("query", "")
        if query:
            similar_docs = await self.search_similar(query, k=3)
            if similar_docs:
                base["relevant_context"] = "\n".join(similar_docs)
        return base
