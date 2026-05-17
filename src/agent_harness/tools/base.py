from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Type, get_type_hints

from pydantic import BaseModel, ConfigDict, ValidationError, create_model


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
    return create_model(
        f"{func.__name__}_input",
        __config__=ConfigDict(extra="forbid"),
        **params,
    )


@dataclass
class ToolMetadata:
    """工具元数据 — 供 LLM 理解工具用途"""
    name: str
    description: str
    parameters_schema: Dict[str, Any]
    category: str = "general"
    requires_confirmation: bool = False
    tags: list = field(default_factory=list)
    strict_validation: bool = True


class ToolValidationError(ValueError):
    """Raised when tool inputs do not match the declared schema."""


def _schema_type_to_python(schema: Dict[str, Any]) -> type:
    schema_type = schema.get("type", "string")
    if schema_type == "integer":
        return int
    if schema_type == "number":
        return float
    if schema_type == "boolean":
        return bool
    if schema_type == "array":
        return list
    if schema_type == "object":
        return dict
    return str


def _model_from_json_schema(name: str, schema: Dict[str, Any]) -> Type[BaseModel]:
    if schema.get("type") != "object":
        return create_model(
            f"{name}_input",
            input=(str, ...),
            __config__=ConfigDict(extra="forbid"),
        )

    required = set(schema.get("required", []))
    fields: Dict[str, tuple[type, Any]] = {}
    for field_name, field_schema in schema.get("properties", {}).items():
        field_type = _schema_type_to_python(field_schema)
        default = ... if field_name in required else None
        fields[field_name] = (field_type, default)

    if not fields:
        fields = {"input": (str, ...)}

    return create_model(
        f"{name}_input",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


class BaseTool(ABC):
    """工具基类 — 模仿 LangChain BaseTool"""

    name: str = "base_tool"
    description: str = ""
    metadata: ToolMetadata = None
    _input_model: Type[BaseModel] | None = None

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
            validated_kwargs = self.validate_input(kwargs)
            return await self._arun(**validated_kwargs)
        except ToolValidationError:
            raise
        except ValidationError as e:
            raise ToolValidationError(f"[{self.name}] invalid input: {e}") from e
        except Exception as e:
            raise type(e)(f"[{self.name}] {e}")

    def validate_input(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize tool inputs before execution."""
        meta = self.get_schema()
        if not meta.strict_validation:
            return kwargs

        model = self.get_input_model()
        try:
            validated = model.model_validate(kwargs)
        except ValidationError as e:
            raise ToolValidationError(f"[{self.name}] invalid input: {e}") from e
        return validated.model_dump(exclude_none=True)

    def get_input_model(self) -> Type[BaseModel]:
        if self._input_model is None:
            meta = self.get_schema()
            self._input_model = _model_from_json_schema(meta.name, meta.parameters_schema)
        return self._input_model

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

    def get_input_model(self) -> Type[BaseModel]:
        return self._input_model

    def get_schema(self) -> ToolMetadata:
        model = self._input_model
        json_schema = model.model_json_schema()
        return ToolMetadata(
            name=self.name,
            description=self.description,
            parameters_schema=json_schema,
        )
