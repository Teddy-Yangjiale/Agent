from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agent_harness.llm.base import LLMResponse


@dataclass
class CacheEntry:
    response: LLMResponse
    created_at: float = field(default_factory=time.time)
    hit_count: int = 1


class LLMCache:
    """LLM 响应缓存 — 精确匹配 + 语义相似去重"""

    def __init__(
        self,
        max_entries: int = 1000,
        ttl_seconds: float = 3600.0,
        similarity_threshold: float = 0.95,
    ):
        self._exact_cache: Dict[str, CacheEntry] = {}
        self._max_entries = max_entries
        self._ttl = ttl_seconds

    def _make_key(self, messages: List[Dict[str, Any]], tools: Optional[List] = None) -> str:
        canonical = json.dumps({"messages": messages, "tools": tools}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def get(self, messages: List[Dict[str, Any]], tools: Optional[List] = None) -> Optional[LLMResponse]:
        self._cleanup()
        key = self._make_key(messages, tools)
        entry = self._exact_cache.get(key)
        if entry:
            entry.hit_count += 1
            return entry.response
        return None

    def set(self, messages: List[Dict[str, Any]], tools: Optional[List], response: LLMResponse) -> None:
        key = self._make_key(messages, tools)
        if key not in self._exact_cache and len(self._exact_cache) >= self._max_entries:
            oldest = min(self._exact_cache.items(), key=lambda x: x[1].created_at)
            del self._exact_cache[oldest[0]]
        self._exact_cache[key] = CacheEntry(response=response)

    def _cleanup(self) -> None:
        now = time.time()
        expired = [k for k, v in self._exact_cache.items() if now - v.created_at > self._ttl]
        for k in expired:
            del self._exact_cache[k]

    def clear(self) -> None:
        self._exact_cache.clear()

    @property
    def size(self) -> int:
        return len(self._exact_cache)

    @property
    def total_hits(self) -> int:
        return sum(e.hit_count for e in self._exact_cache.values())
