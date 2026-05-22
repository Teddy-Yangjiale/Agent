import asyncio
from typing import AsyncIterator, List

from agent_harness.agents.base import AgentAction, AgentInput, AgentOutput, AgentStep, BaseAgent
from agent_harness.tools.base import BaseTool


class SwarmMember:
    def __init__(self, llm, tools: List[BaseTool], name: str = "member", role: str = ""):
        self._llm = llm
        self._tools = {t.name: t for t in tools}
        self.name = name
        self.role = role

    async def respond(self, topic: str, context: str) -> str:
        from agent_harness.llm.base import ChatMessage
        messages = [
            ChatMessage(role="system", content=f"You are '{self.name}' ({self.role}). Discussion context:\n{context}\n\nGive your opinion on the topic. Keep it concise (1-3 sentences)."),
            ChatMessage(role="user", content=topic),
        ]
        response = await self._llm.agenerate(messages)
        return response.content


class SwarmAgent(BaseAgent):
    """Swarm Agent — 多 Agent 平行讨论达成共识"""

    name = "swarm_agent"
    description = "Swarm multi-agent: parallel discussion with consensus"

    def __init__(self, llm, members: List[SwarmMember], rounds: int = 2):
        self._llm = llm
        self._members = members
        self._rounds = rounds

    def get_tool_names(self) -> List[str]:
        return [m.name for m in self._members]

    async def plan(self, input_data: AgentInput, intermediate_steps: List[AgentStep]) -> AgentStep:
        return AgentStep(action=AgentAction.FINAL_ANSWER, final_answer="")

    async def astream(self, input_data: AgentInput) -> AsyncIterator[AgentStep]:
        context = ""
        yield AgentStep(action=AgentAction.THINK, thought=f"Starting swarm discussion ({len(self._members)} members, {self._rounds} rounds)")
        for r in range(self._rounds):
            step = AgentStep(action=AgentAction.THINK, thought=f"Round {r+1}/{self._rounds}", tool_name="swarm")
            yield step
            tasks = [m.respond(input_data.query, context) for m in self._members]
            responses = await asyncio.gather(*tasks)
            round_output = ""
            for member, resp in zip(self._members, responses):
                round_output += f"[{member.name}]: {resp}\n"
                step = AgentStep(action=AgentAction.TOOL_CALL, tool_name=member.name, tool_input={"topic": input_data.query}, tool_output=resp, thought=f"{member.name} responds")
                yield step
            context += round_output + "\n"
        from agent_harness.llm.base import ChatMessage
        summary_msgs = [
            ChatMessage(role="system", content=f"Synthesize the following discussion into a final answer:\n{context}"),
            ChatMessage(role="user", content=f"Original question: {input_data.query}\n\nProvide the final answer."),
        ]
        response = await self._llm.agenerate(summary_msgs)
        yield AgentStep(action=AgentAction.FINAL_ANSWER, final_answer=response.content)

    async def arun(self, input_data: AgentInput) -> AgentOutput:
        steps = []
        async for step in self.astream(input_data):
            steps.append(step)
        last = steps[-1] if steps else AgentStep(action=AgentAction.ERROR, error="No steps")
        return AgentOutput(result=last.final_answer or "", steps=steps)
