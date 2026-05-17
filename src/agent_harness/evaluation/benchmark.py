import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agent_harness import AgentExecutor


@dataclass
class EvalCase:
    name: str
    query: str
    expected_keywords: List[str] = field(default_factory=list)
    expected_tool_calls: List[str] = field(default_factory=list)
    min_tools_used: int = 0
    max_duration_ms: float = 30000.0
    custom_validator: Optional[Callable[[str, List], bool]] = None


@dataclass
class EvalResult:
    name: str = ""
    passed: bool = False
    score: float = 0.0
    duration_ms: float = 0.0
    output: str = ""
    error: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class BenchmarkRunner:
    def __init__(self, executor: AgentExecutor):
        self._executor = executor

    def _default_cases(self) -> List[EvalCase]:
        return [
            EvalCase(name="math_calc", query="Calculate 123 + 456 * 2", expected_keywords=["1035"], expected_tool_calls=["calculator"]),
            EvalCase(name="math_power", query="What is 2 to the power of 10?", expected_keywords=["1024"], expected_tool_calls=["calculator"]),
            EvalCase(name="python_basic", query="Write Python to compute the sum of numbers 1 to 100", expected_keywords=["5050"], min_tools_used=1),
            EvalCase(name="non_tool", query="What is the capital of France?", expected_keywords=["Paris"]),
            EvalCase(name="multi_step", query="Calculate (100 + 200) * 3 / 2", expected_keywords=["450"], min_tools_used=1),
        ]

    async def run_all(self, cases: Optional[List[EvalCase]] = None) -> List[dict]:
        cases = cases or self._default_cases()
        results = []
        for case in cases:
            start = time.monotonic()
            try:
                output = await self._executor.arun(case.query)
                duration_ms = (time.monotonic() - start) * 1000
                passed = True
                score = 100.0

                if case.expected_keywords:
                    for kw in case.expected_keywords:
                        if kw.lower() not in output.result.lower():
                            passed = False
                            score -= 30

                if case.min_tools_used and output.tool_calls_count < case.min_tools_used:
                    passed = False
                    score -= 30

                if duration_ms > case.max_duration_ms:
                    score -= 10

                if case.custom_validator:
                    if not case.custom_validator(output.result, output.steps):
                        passed = False
                        score -= 30

                results.append({
                    "name": case.name, "passed": passed, "score": max(0.0, score),
                    "duration_ms": duration_ms, "output": output.result[:200],
                    "tool_calls": output.tool_calls_count,
                })
            except Exception as e:
                results.append({
                    "name": case.name, "passed": False, "score": 0.0,
                    "duration_ms": (time.monotonic() - start) * 1000,
                    "output": "", "tool_calls": 0, "error": str(e),
                })
        return results
