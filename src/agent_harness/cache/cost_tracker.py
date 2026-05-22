from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ModelPricing:
    prompt_per_1k: float = 0.0
    completion_per_1k: float = 0.0


PRICING_MAP: Dict[str, ModelPricing] = {
    "deepseek-chat": ModelPricing(prompt_per_1k=0.00014, completion_per_1k=0.00028),
    "deepseek-reasoner": ModelPricing(prompt_per_1k=0.00055, completion_per_1k=0.00219),
    "gpt-4o": ModelPricing(prompt_per_1k=0.005, completion_per_1k=0.015),
    "gpt-4o-mini": ModelPricing(prompt_per_1k=0.00015, completion_per_1k=0.0006),
    "gpt-4-turbo": ModelPricing(prompt_per_1k=0.01, completion_per_1k=0.03),
    "gpt-3.5-turbo": ModelPricing(prompt_per_1k=0.0005, completion_per_1k=0.0015),
    "claude-3-opus": ModelPricing(prompt_per_1k=0.015, completion_per_1k=0.075),
    "claude-3-sonnet": ModelPricing(prompt_per_1k=0.003, completion_per_1k=0.015),
    "claude-3-haiku": ModelPricing(prompt_per_1k=0.00025, completion_per_1k=0.00125),
    "gemini-2.0-flash": ModelPricing(prompt_per_1k=0.0001, completion_per_1k=0.0004),
    "gemini-1.5-pro": ModelPricing(prompt_per_1k=0.00125, completion_per_1k=0.005),
}


@dataclass
class RunCost:
    total_cost: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""


class CostTracker:
    def __init__(self, custom_pricing: Optional[Dict[str, ModelPricing]] = None):
        self._pricing = {**PRICING_MAP, **(custom_pricing or {})}
        self._runs: list[RunCost] = []

    def estimate(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        pricing = self._pricing.get(model)
        if not pricing:
            for prefix in self._pricing:
                if model.startswith(prefix):
                    pricing = self._pricing[prefix]
                    break
        if not pricing:
            return 0.0
        return (prompt_tokens / 1000) * pricing.prompt_per_1k + (completion_tokens / 1000) * pricing.completion_per_1k

    def record(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        cost = self.estimate(model, prompt_tokens, completion_tokens)
        self._runs.append(RunCost(total_cost=cost, prompt_tokens=prompt_tokens,
                                   completion_tokens=completion_tokens, model=model))
        return cost

    @property
    def total_cost(self) -> float:
        return sum(r.total_cost for r in self._runs)

    @property
    def total_tokens(self) -> int:
        return sum(r.prompt_tokens + r.completion_tokens for r in self._runs)

    @property
    def run_count(self) -> int:
        return len(self._runs)

    def summary(self) -> Dict[str, Any]:
        per_model = {}
        for run in self._runs:
            if run.model not in per_model:
                per_model[run.model] = {"cost": 0.0, "runs": 0, "tokens": 0}
            per_model[run.model]["cost"] += run.total_cost
            per_model[run.model]["runs"] += 1
            per_model[run.model]["tokens"] += run.prompt_tokens + run.completion_tokens
        return {
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "run_count": self.run_count,
            "per_model": per_model,
        }

    def reset(self) -> None:
        self._runs.clear()
