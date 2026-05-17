from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

from agent_harness.agents.base import (
    AgentAction,
    AgentInput,
    AgentOutput,
    AgentStep,
    BaseAgent,
)
from agent_harness.harness.callbacks import (
    BaseCallback,
    CallbackEvent,
    CallbackManager,
)
from agent_harness.memory.base import BaseMemory
from agent_harness.security.permissions import PermissionManager
from agent_harness.tools.base import BaseTool
from agent_harness.tools.registry import ToolRegistry


@dataclass
class AgentExecutorConfig:
    max_steps: int = 10
    max_execution_time: float = 60.0
    max_tool_output_length: int = 2000
    return_intermediate_steps: bool = True
    handle_parsing_errors: bool = True
    early_stopping_method: str = "generate"
    verbose: bool = False
    tool_retry_count: int = 0
    inject_memory: bool = True
    enforce_timeout: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentExecutor:
    """Agent 执行器 — 中央 Harness"""

    def __init__(
        self,
        agent: BaseAgent,
        tool_registry: ToolRegistry,
        memory: Optional[BaseMemory] = None,
        callbacks: Optional[List[BaseCallback]] = None,
        config: Optional[AgentExecutorConfig] = None,
        permission_manager: Optional[PermissionManager] = None,
    ):
        self._agent = agent
        self._tool_registry = tool_registry
        self._memory = memory
        self._callback_manager = CallbackManager(callbacks or [])
        self._config = config or AgentExecutorConfig()
        self._permission_manager = permission_manager

    @property
    def agent(self) -> BaseAgent:
        return self._agent

    @property
    def tools(self) -> ToolRegistry:
        return self._tool_registry

    @property
    def memory(self) -> Optional[BaseMemory]:
        return self._memory

    async def arun(self, query: str, **kwargs) -> AgentOutput:
        start_time = time.monotonic()
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        input_data = AgentInput(
            query=query,
            max_steps=kwargs.get("max_steps", self._config.max_steps),
            context=kwargs,
            allowed_tools=kwargs.get("allowed_tools"),
        )

        if self._config.inject_memory and self._memory:
            mem_vars = await self._memory.aload_memory_variables({"query": query})
            input_data.context["memory"] = mem_vars
            input_data.context["chat_history"] = mem_vars.get("chat_history", [])
            input_data.context["history_text"] = mem_vars.get("history_text", "")

        await self._notify("agent_start", agent_name=self._agent.name, data={
            "query": query,
            "max_steps": input_data.max_steps,
        })

        steps: List[AgentStep] = []
        final_result = ""

        try:
            coro = self._run_loop(input_data, steps, total_usage)
            if self._config.enforce_timeout and self._config.max_execution_time > 0:
                await asyncio.wait_for(coro, timeout=self._config.max_execution_time)
            else:
                await coro

            final_step = steps[-1] if steps else None
            if final_step and final_step.action == AgentAction.FINAL_ANSWER:
                final_result = final_step.final_answer
            elif final_step and final_step.action == AgentAction.ERROR:
                final_result = f"Agent 错误: {final_step.error}"
            else:
                final_result = "Agent 达到最大步数限制，未给出最终答案"

        except asyncio.TimeoutError:
            final_result = "Agent 执行超时"
            await self._notify("agent_error", agent_name=self._agent.name, data={"error": "timeout"})
        except Exception as e:
            final_result = f"Agent 执行异常: {e}"
            await self._notify("agent_error", agent_name=self._agent.name, data={"error": str(e)})

        duration_ms = (time.monotonic() - start_time) * 1000
        output = AgentOutput(
            result=final_result,
            steps=steps if self._config.return_intermediate_steps else [],
            total_tokens=total_usage["total_tokens"],
            total_duration_ms=duration_ms,
            tool_calls_count=sum(1 for s in steps if s.action == AgentAction.TOOL_CALL),
        )

        if self._memory:
            await self._memory.asave_context(
                {"query": query},
                {"result": final_result},
            )

        await self._notify("agent_end", agent_name=self._agent.name, data={
            "result": final_result,
            "steps": len(steps),
            "duration_ms": duration_ms,
            "usage": total_usage,
        })

        return output

    async def astream(self, query: str, **kwargs) -> AsyncIterator[AgentStep]:
        input_data = AgentInput(
            query=query,
            max_steps=kwargs.get("max_steps", self._config.max_steps),
            context=kwargs,
            allowed_tools=kwargs.get("allowed_tools"),
        )

        if self._config.inject_memory and self._memory:
            mem_vars = await self._memory.aload_memory_variables({"query": query})
            input_data.context["memory"] = mem_vars

        steps: List[AgentStep] = []
        for i in range(input_data.max_steps):
            step = await self._agent.plan(input_data, steps)
            if step.action == AgentAction.TOOL_CALL and step.tool_name:
                if not self._is_tool_allowed(input_data, step.tool_name):
                    step.tool_output = f"工具 '{step.tool_name}' 不在本次允许的工具列表中"
                    step.action = AgentAction.ERROR
                    step.error = "tool_not_allowed"
                elif self._permission_manager and not await self._check_tool_path(step):
                    pass
                else:
                    await self._execute_tool(step)
            yield step
            steps.append(step)
            if step.action in (AgentAction.FINAL_ANSWER, AgentAction.ERROR):
                break

    async def _run_loop(
        self,
        input_data: AgentInput,
        steps: List[AgentStep],
        usage: Dict[str, int],
    ) -> None:
        for i in range(input_data.max_steps):
            step = await self._agent.plan(input_data, steps)
            self._accumulate_usage(usage, step.metadata.get("llm_usage", {}))
            await self._notify("agent_step", agent_name=self._agent.name, step=step, data={
                "usage": usage,
            })

            if step.action == AgentAction.TOOL_CALL and step.tool_name:
                if not self._is_tool_allowed(input_data, step.tool_name):
                    step.tool_output = f"工具 '{step.tool_name}' 不在本次允许的工具列表中"
                    step.action = AgentAction.ERROR
                    step.error = "tool_not_allowed"
                    steps.append(step)
                    break

                if self._permission_manager:
                    if not await self._check_tool_path(step):
                        steps.append(step)
                        break

                    allowed = await self._permission_manager.check(
                        step.tool_name, step.tool_input
                    )
                    if not allowed:
                        step.tool_output = f"工具 '{step.tool_name}' 被权限策略拒绝"
                        step.action = AgentAction.ERROR
                        step.error = "permission_denied"
                        steps.append(step)
                        break
                    self._permission_manager.record_call(step.tool_name)

                await self._execute_tool(step)

            steps.append(step)

            if step.action == AgentAction.FINAL_ANSWER:
                return
            if step.action == AgentAction.ERROR:
                if self._config.handle_parsing_errors:
                    continue
                return

    async def _execute_tool(self, step: AgentStep) -> None:
        tool = self._tool_registry.get(step.tool_name)
        if not tool:
            step.tool_output = f"未知工具: {step.tool_name}"
            step.action = AgentAction.ERROR
            step.error = "unknown_tool"
            return

        await self._notify("tool_start", agent_name=self._agent.name, step=step)

        max_attempts = 1 + self._config.tool_retry_count
        for attempt in range(max_attempts):
            try:
                step.tool_output = await tool.arun(**step.tool_input)
                break
            except Exception as e:
                if attempt == max_attempts - 1:
                    step.tool_output = f"工具执行失败 (重试{max_attempts}次): {e}"
                    step.action = AgentAction.ERROR
                    step.error = str(e)
                else:
                    await asyncio.sleep(0.5 * (attempt + 1))

        output_str = str(step.tool_output)
        if len(output_str) > self._config.max_tool_output_length:
            step.tool_output = output_str[:self._config.max_tool_output_length] + "\n... (truncated)"

        await self._notify("tool_end", agent_name=self._agent.name, step=step)

    def _accumulate_usage(self, total: Dict[str, int], usage: Dict[str, int]) -> None:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            total[key] = total.get(key, 0) + int(usage.get(key, 0) or 0)

    def _is_tool_allowed(self, input_data: AgentInput, tool_name: str) -> bool:
        if input_data.allowed_tools is None:
            return True
        return tool_name in input_data.allowed_tools

    async def _check_tool_path(self, step: AgentStep) -> bool:
        if not self._permission_manager:
            return True

        tool = self._tool_registry.get(step.tool_name)
        meta = tool.get_schema() if tool else None
        path = step.tool_input.get("path") or step.tool_input.get("input")
        if path and meta:
            if not await self._permission_manager.check_path(step.tool_name, str(path)):
                step.tool_output = f"路径 '{path}' 不在允许范围内"
                step.action = AgentAction.ERROR
                step.error = "path_denied"
                return False
        return True

    async def _notify(self, event_type: str, **kwargs) -> None:
        event = CallbackEvent(event_type=event_type, **kwargs)
        await self._callback_manager.notify(event)
