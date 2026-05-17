# Changelog
All notable changes to the Agent Harness project.

---

## [0.2.0] - 2026-05-15

### Added
- **#5 LLM重试&退避**: `tenacity` exponential backoff on all LLM calls (3 retries, 1-30s wait).
- **#2 记忆接入执行循环**: `AgentExecutor` auto-loads `aload_memory_variables()` before each run.
- **#6 结构化输出解析器**: `output/parser.py` — JSON-first parsing with regex fallback + Pydantic parser.
- **#1 安全代码沙箱**: `security/sandbox.py` — restricted Python mode (whitelist) + Docker isolation.
- **#4 工具权限+人工确认**: `security/permissions.py` — ALLOW/DENY/ASK_USER policies, path rules, call limits.
- **#7 真流式token输出**: `StreamCallback` with `on_llm_token` event for real-time token streaming.
- **#3 LLM响应缓存**: `cache/llm_cache.py` — SHA256 exact-match with TTL + LRU eviction.
- **#14 成本追踪**: `cache/cost_tracker.py` — pre-configured pricing for DeepSeek, GPT-4, Claude, Gemini.
- **#12 多LLM提供商**: `AnthropicLLM`, `GoogleLLM`, `OllamaLLM`, `DeepSeekLLM`.
- **#13 MCP stdio传输**: Full JSON-RPC 2.0 stdio + SSE + HTTP in `mcp/client.py`.
- **#8 多Agent协作**: `HierarchicalAgent` (controller/worker) + `SwarmAgent` (parallel discussion).
- **#9 持久化记忆+RAG**: `PersistentMemory` (SQLite) + `RAGMemory` (ChromaDB vector search).
- **#10 OpenTelemetry全链路追踪**: `TraceManager` with OTLP export + `@traced` decorator.
- **#11 检查点&恢复**: `CheckpointManager` — per-step JSON snapshots with run listing.
- **#15 Web可视化面板**: FastAPI + WebSocket real-time dashboard at `localhost:8080`.
- **#16 CLI交互模式**: `agent-harness chat/serve/eval/run` commands.
- **#17 评估基准**: `BenchmarkRunner` with 5 default test cases.

### Changed
- Default model switched to `deepseek-chat`, API key env var to `DEEPSEEK_API_KEY`.
- Heavy dependencies (anthropic, google-genai, chromadb, opentelemetry) moved to `[full]` extras.
- `AgentExecutor`: added `tool_retry_count`, `inject_memory`, `enforce_timeout` config.
- `PythonREPLTool`: now sandboxed by default via `SandboxedExecutor`.
- `BaseCallback`: refactored to non-ABC with no-op defaults + dispatch dict.

### Fixed
- Memory `aload_memory_variables()` not called — now injected at execution start.
- `max_execution_time` unenforced — now uses `asyncio.wait_for`.
- `TokenCounterCallback` was inert — now receives usage from executor events.
- MCP tool-to-server matching bug — fixed in stdio/HTTP implementation.
- `pyproject.toml` truncated after script section write — restored.

---

## [0.1.0] - 2026-05-14

### Added
- Initial project with LangChain-inspired architecture.
- `ReActAgent` + `ToolCallingAgent`, `AgentExecutor`, `ToolRegistry`.
- 5 built-in tools: Calculator, PythonREPL, WebSearch, FileRead, FileWrite.
- `ConversationBufferMemory` + `ConversationSummaryMemory`.
- `OpenAILLM`, MCP client (SSE/HTTP), `Planner`, `Callback` system.
- Structured logging via `structlog`.
