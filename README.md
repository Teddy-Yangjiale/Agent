# Agent Harness

Agent Harness 是一个面向生产化演进的 Python Agent 框架。它参考 LangChain `AgentExecutor` 的执行模型，提供 ReAct、原生 Tool Calling、工具注册、权限控制、沙箱执行、记忆、MCP、缓存、成本追踪、链路追踪、CLI 和 Web 面板等基础能力。

当前默认适配 DeepSeek API，也支持 OpenAI 兼容 API。

## 当前状态

这个项目已经具备 Agent 框架雏形，但仍处在生产化建设阶段。最近一轮重构重点增强了执行稳定性和安全边界：

- `AgentExecutor` 会聚合每轮 LLM 返回的 token usage。
- `allowed_tools` 现在会在普通执行和流式执行中生效。
- 带路径参数的工具会在执行前经过 `PermissionManager` 检查。
- 路径权限使用真实路径边界判断，避免 `C:\tmp` 误放行 `C:\tmp2`。
- restricted sandbox 增加 AST 预检，拒绝循环、dunder 属性等高风险结构。
- 新增离线单元测试覆盖执行器、权限和沙箱的核心行为。

## 安装

```powershell
$env:DEEPSEEK_API_KEY = "sk-your-deepseek-key"
pip install -e .

# 可选扩展
pip install -e ".[full]"    # Anthropic / Google / ChromaDB / OpenTelemetry
pip install -e ".[dev]"     # pytest / ruff / mypy
```

## 30 秒上手

```python
import asyncio

from agent_harness import AgentExecutor
from agent_harness.agents import ReActAgent
from agent_harness.llm import OpenAILLM
from agent_harness.tools import CalculatorTool, ToolRegistry


async def main():
    llm = OpenAILLM(
        model="deepseek-chat",
        api_key="sk-your-key",
        base_url="https://api.deepseek.com/v1",
    )
    registry = ToolRegistry().register(CalculatorTool())
    agent = ReActAgent(llm=llm, tools=registry.list_all())
    executor = AgentExecutor(agent=agent, tool_registry=registry)

    result = await executor.arun("计算 (3 + 5) * 10")
    print(result.result)
    print("tokens:", result.total_tokens)


asyncio.run(main())
```

## CLI

```powershell
$env:DEEPSEEK_API_KEY = "sk-your-key"

agent-harness run "计算 123 + 456 * 2"
agent-harness chat
agent-harness chat -m deepseek-reasoner
agent-harness eval
agent-harness serve
```

Web 面板默认运行在 `http://localhost:8080`。

## Agent 模式

| 模式 | 类 | 适用场景 |
|---|---|---|
| ReAct | `ReActAgent` | DeepSeek、开源模型、任意文本模型 |
| Tool Calling | `ToolCallingAgent` | 支持原生 function/tool calling 的模型 |

## 工具和权限

自定义工具只需要继承 `BaseTool`：

```python
from agent_harness.tools import BaseTool


class GreetTool(BaseTool):
    name = "greet"
    description = "向用户打招呼"

    async def _arun(self, name: str) -> str:
        return f"你好，{name}！"
```

按次限制工具白名单：

```python
result = await executor.arun(
    "只允许计算，不允许读写文件",
    allowed_tools=["calculator"],
)
```

配置路径权限：

```python
from agent_harness.security.permissions import PermissionManager, PermissionPolicy

permissions = PermissionManager().set_policy(
    PermissionPolicy(
        tool_name="write_file",
        allowed_paths=["D:/Agent/workspace"],
        deny_paths=["D:/Agent/secrets"],
    )
)
```

## 核心模块

| 能力 | 模块 | 说明 |
|---|---|---|
| 执行器 | `harness.agent_executor` | 控制 Agent 循环、工具执行、超时、回调、usage 汇总 |
| Agent | `agents` | ReAct 和 Tool Calling 两种基础 Agent |
| LLM | `llm` | OpenAI 兼容、DeepSeek、Anthropic、Google/Ollama 适配 |
| 工具 | `tools` | 工具基类、注册表、内置计算/文件/搜索/Python REPL 工具 |
| 权限 | `security.permissions` | 工具调用策略、调用次数、路径 allow/deny |
| 沙箱 | `security.sandbox` | restricted exec 和 Docker 隔离执行 |
| 记忆 | `memory` / `storage` | 短期对话记忆、SQLite 持久化、RAGMemory |
| MCP | `mcp` | MCP server 发现和工具适配 |
| 评估 | `evaluation` | 基础 benchmark runner |
| 可观测性 | `callbacks` / `tracing` | 日志、token 统计、流式输出、OpenTelemetry |

## 架构

```text
用户查询
  -> AgentExecutor
      -> Memory 加载上下文
      -> Agent 规划下一步
      -> LLM 生成决策
      -> PermissionManager 检查工具和路径权限
      -> ToolRegistry 执行工具
      -> Sandbox 隔离执行代码
      -> Callbacks / Tracing 输出事件和监控数据
      -> Memory 保存最终结果
```

## 测试

```powershell
$env:PYTHONPATH = "D:\Agent\src"
python -m compileall -q src tests
python -m pytest -q
```

当前基础测试覆盖：

- LLM usage 聚合。
- `allowed_tools` 工具白名单。
- 路径权限执行前拦截。
- 路径边界判断。
- restricted sandbox 基础执行和高风险循环拒绝。

## 生产化路线

优先级从高到低：

1. 给所有工具输入增加 Pydantic 校验和清晰错误返回。
2. 将 restricted sandbox 替换为默认 Docker/进程级隔离，避免主进程被用户代码影响。
3. MCP 层移除静默 `except Exception: pass`，补齐超时、重试、错误事件和日志。
4. 给 Web API 增加认证、限流、审计日志和多会话隔离。
5. 建立 CI：`ruff`、`mypy`、`pytest`、安全扫描。
6. 增加真实评估集和回归基准，覆盖多步工具调用、失败恢复、权限拒绝、长上下文任务。
