from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Type, get_type_hints

from pydantic import BaseModel, create_model


def _create_input_model(func: Callable) -> Type[BaseModel]:
    """从函数签名自动生成 Pydantic 输入模型"""
    hints = get_type_hints(func)
    params = {
        k: (v, ...)
        for k, v in hints.items()
        if k != "return"
    }
    if not params:
        params = {"input": (str, ...)}
    return create_model(f"{func.__name__}_input", **params)


@dataclass
class ToolMetadata:
    """工具元数据 — 供 LLM 理解工具用途"""
    name: str
    description: str
    parameters_schema: Dict[str, Any]
    category: str = "general"
    requires_confirmation: bool = False
    tags: list = field(default_factory=list)


class BaseTool(ABC):
    """工具基类 — 模仿 LangChain BaseTool"""

    name: str = "base_tool"
    description: str = ""
    metadata: ToolMetadata = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not cls.name or cls.name == "base_tool":
            cls.name = cls.__name__

    @abstractmethod
    async def _arun(self, **kwargs) -> Any:
        """异步执行工具逻辑"""
        ...

    async def arun(self, **kwargs) -> Any:
        """带钩子的异步执行入口"""
        try:
            return await self._arun(**kwargs)
        except Exception as e:
            raise type(e)(f"[{self.name}] {e}")

    def get_schema(self) -> ToolMetadata:
        """获取工具的 JSON Schema 描述"""
        if self.metadata:
            return self.metadata
        return ToolMetadata(
            name=self.name,
            description=self.description,
            parameters_schema={
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "工具输入"}
                },
                "required": ["input"],
            },
        )


class FunctionTool(BaseTool):
    """将任意 async 函数包装为 Tool"""

    def __init__(self, func: Callable, name: str = "", description: str = ""):
        self._func = func
        self.name = name or func.__name__
        self.description = description or func.__doc__ or ""
        self._input_model = _create_input_model(func)

    async def _arun(self, **kwargs) -> Any:
        return await self._func(**kwargs)

    def get_schema(self) -> ToolMetadata:
        model = self._input_model
        json_schema = model.model_json_schema()
        return ToolMetadata(
            name=self.name,
            description=self.description,
            parameters_schema=json_schema,
        )
