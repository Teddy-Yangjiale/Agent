from typing import AsyncIterator, Dict, List

from agent_harness.agents.base import AgentAction, AgentInput, AgentOutput, AgentStep, BaseAgent
from agent_harness.tools.base import BaseTool


class WorkerAgent(BaseAgent):
    """轻量 Worker Agent — 执行子任务后返回结果"""

    name = "worker"
    description = "Worker agent for subtask execution"

    def __init__(self, llm, tools: List[BaseTool], name: str = "worker", max_steps: int = 5):
        self._llm = llm
        self._tools = {t.name: t for t in tools}
        self.name = name
        self._max_steps = max_steps

    def get_tool_names(self) -> List[str]:
        return list(self._tools.keys())

    async def plan(self, input_data: AgentInput, intermediate_steps: List[AgentStep]) -> AgentStep:
        from agent_harness.llm.base import ChatMessage
        tools_desc = "\n".join(f"- {n}: {t.description}" for n, t in self._tools.items())
        messages = [
            ChatMessage(role="system", content=f"You are a worker agent named '{self.name}'. Available tools:\n{tools_desc}\n\nComplete the user's task. If you have a final answer, reply with: FINAL ANSWER: <your answer>"),
            ChatMessage(role="user", content=input_data.query),
        ]
        for step in intermediate_steps:
            messages.append(ChatMessage(role="assistant", content=f"Action: {step.tool_name} -> {step.tool_output}"))
        response = await self._llm.agenerate(messages)
        text = response.content
        if "FINAL ANSWER:" in text.upper():
            answer = text.split("FINAL ANSWER:")[-1].strip()
            return AgentStep(action=AgentAction.FINAL_ANSWER, final_answer=answer, thought=text)
        if "Action:" in text:
            parts = text.split("Action:")[-1].strip().split("\n", 1)
            tool_name = parts[0].strip()
            if tool_name in self._tools:
                return AgentStep(action=AgentAction.TOOL_CALL, tool_name=tool_name, tool_input={"input": parts[1].strip() if len(parts) > 1 else ""})
        return AgentStep(action=AgentAction.FINAL_ANSWER, final_answer=text)

    async def astream(self, input_data: AgentInput) -> AsyncIterator[AgentStep]:
        async for step in self._run_loop(input_data):
            yield step

    async def _run_loop(self, input_data: AgentInput) -> AsyncIterator[AgentStep]:
        steps = []
        for _ in range(self._max_steps):
            step = await self.plan(input_data, steps)
            if step.action == AgentAction.TOOL_CALL and step.tool_name in self._tools:
                tool = self._tools[step.tool_name]
                step.tool_output = await tool.arun(**step.tool_input)
            yield step
            steps.append(step)
            if step.action == AgentAction.FINAL_ANSWER:
                break

    async def arun(self, input_data: AgentInput) -> AgentOutput:
        steps = []
        async for step in self.astream(input_data):
            steps.append(step)
        last = steps[-1] if steps else AgentStep(action=AgentAction.ERROR, error="No steps")
        return AgentOutput(result=last.final_answer or last.error or "", steps=steps)


class HierarchicalAgent(BaseAgent):
    """层级 Agent — Controller 分解任务分发给 Worker"""

    name = "hierarchical_agent"
    description = "Hierarchical multi-agent: controller decomposes and delegates to workers"

    def __init__(self, llm, workers: Dict[str, WorkerAgent], max_steps: int = 5):
        self._llm = llm
        self._workers = workers
        self._max_steps = max_steps

    def get_tool_names(self) -> List[str]:
        return list(self._workers.keys())

    async def plan(self, input_data: AgentInput, intermediate_steps: List[AgentStep]) -> AgentStep:
        from agent_harness.llm.base import ChatMessage
        worker_desc = "\n".join(f"- {n}: {w.description}" for n, w in self._workers.items())
        history = "\n".join(f"[{s.tool_name}] {s.tool_output}" for s in intermediate_steps)
        messages = [
            ChatMessage(role="system", content=f"You are a controller agent. Delegate work to these workers:\n{worker_desc}\n\nReply with DELEGATE: <worker_name>\nTASK: <task description>\n\nOr FINAL ANSWER: <answer> when done.\n\nWorker results so far:\n{history}"),
            ChatMessage(role="user", content=input_data.query),
        ]
        response = await self._llm.agenerate(messages)
        text = response.content
        if "FINAL ANSWER:" in text.upper():
            answer = text.split("FINAL ANSWER:")[-1].strip()
            return AgentStep(action=AgentAction.FINAL_ANSWER, final_answer=answer, thought=text)
        if "DELEGATE:" in text.upper():
            lines = text.split("\n")
            worker_name = ""
            task = ""
            for line in lines:
                if "DELEGATE:" in line.upper():
                    worker_name = line.split("DELEGATE:")[-1].strip()
                if "TASK:" in line.upper():
                    task = line.split("TASK:")[-1].strip()
            if worker_name in self._workers:
                worker = self._workers[worker_name]
                result = await worker.arun(AgentInput(query=task))
                return AgentStep(
                    action=AgentAction.TOOL_CALL,
                    tool_name=worker_name,
                    tool_input={"task": task},
                    tool_output=result.result,
                    thought=text,
                )
        return AgentStep(action=AgentAction.FINAL_ANSWER, final_answer=text)

    async def astream(self, input_data: AgentInput) -> AsyncIterator[AgentStep]:
        steps = []
        for _ in range(self._max_steps):
            step = await self.plan(input_data, steps)
            yield step
            steps.append(step)
            if step.action == AgentAction.FINAL_ANSWER:
                break

    async def arun(self, input_data: AgentInput) -> AgentOutput:
        steps = []
        async for step in self.astream(input_data):
            steps.append(step)
        last = steps[-1] if steps else AgentStep(action=AgentAction.ERROR, error="No steps")
        return AgentOutput(result=last.final_answer or last.error or "", steps=steps)
