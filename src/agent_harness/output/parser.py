from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from pydantic import BaseModel


@dataclass
class ParsedOutput:
    action: str = ""  # "final_answer" | "tool_call" | "error"
    thought: str = ""
    tool_name: str = ""
    tool_input: Dict[str, Any] = field(default_factory=dict)
    final_answer: str = ""
    error: str = ""
    raw: str = ""


class OutputParser(ABC):
    @abstractmethod
    def parse(self, text: str) -> ParsedOutput:
        ...


class ReActOutputParser(OutputParser):
    """结构化 ReAct 输出解析器 — 先尝试 JSON，失败回退到正则"""

    def parse(self, text: str) -> ParsedOutput:
        json_result = self._try_json(text)
        if json_result:
            return json_result

        regex_result = self._try_regex(text)
        if regex_result:
            return regex_result

        return ParsedOutput(action="error", error=f"无法解析输出: {text[:300]}", raw=text)

    def _try_json(self, text: str) -> Optional[ParsedOutput]:
        json_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if not json_block:
            json_block = re.search(r"(\{[^{}]*\})", text, re.DOTALL)
        if not json_block:
            return None

        try:
            data = json.loads(json_block.group(1))
        except json.JSONDecodeError:
            return None

        if "final_answer" in data or "answer" in data:
            return ParsedOutput(
                action="final_answer",
                final_answer=data.get("final_answer") or data.get("answer", ""),
                thought=data.get("thought", ""),
                raw=text,
            )

        if "tool" in data or "action" in data:
            tool_name = data.get("tool") or data.get("action", "")
            return ParsedOutput(
                action="tool_call",
                tool_name=tool_name,
                tool_input=data.get("input") or data.get("arguments", {}),
                thought=data.get("thought", ""),
                raw=text,
            )

        return None

    def _try_regex(self, text: str) -> Optional[ParsedOutput]:
        fa_match = re.search(
            r"Final\s*Answer\s*[:：]\s*(.*?)$", text, re.DOTALL | re.IGNORECASE
        )
        if fa_match:
            return ParsedOutput(
                action="final_answer",
                final_answer=fa_match.group(1).strip(),
                thought=text,
                raw=text,
            )

        thought_m = re.search(r"Thought\s*[:：]\s*(.*?)(?=Action\s*[:：]|$)", text, re.DOTALL)
        action_m = re.search(r"Action\s*[:：]\s*(\S+)", text)
        input_m = re.search(r"Action\s*Input\s*[:：]\s*(.*?)$", text, re.DOTALL)

        if not action_m:
            return None

        tool_name = action_m.group(1).strip()
        raw_input = input_m.group(1).strip() if input_m else "{}"

        try:
            tool_input = json.loads(raw_input)
        except json.JSONDecodeError:
            tool_input = {"input": raw_input}

        return ParsedOutput(
            action="tool_call",
            thought=thought_m.group(1).strip() if thought_m else "",
            tool_name=tool_name,
            tool_input=tool_input,
            raw=text,
        )


class StructuredOutputParser(OutputParser):
    """基于 Pydantic 模型的结构化输出解析器"""

    def __init__(self, output_model: type[BaseModel]):
        self._model = output_model

    def parse(self, text: str) -> ParsedOutput:
        json_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if not json_block:
            json_block = re.search(r"(\{.*\})", text, re.DOTALL)
        if not json_block:
            return ParsedOutput(action="error", error="未找到 JSON 块", raw=text)

        try:
            data = json.loads(json_block.group(1))
            self._model.model_validate(data)
        except (json.JSONDecodeError, ValueError) as e:
            return ParsedOutput(action="error", error=str(e), raw=text)

        return self._parse_fields(data, text)

    def _parse_fields(self, data: dict, raw: str) -> ParsedOutput:
        if "result" in data:
            return ParsedOutput(action="final_answer", final_answer=data["result"], raw=raw)
        if "answer" in data:
            return ParsedOutput(action="final_answer", final_answer=data["answer"], raw=raw)
        if "tool" in data:
            return ParsedOutput(
                action="tool_call",
                tool_name=data["tool"],
                tool_input=data.get("input", {}),
                raw=raw,
            )
        return ParsedOutput(action="final_answer", final_answer=str(data), raw=raw)
