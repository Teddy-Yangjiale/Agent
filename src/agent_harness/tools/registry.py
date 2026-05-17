from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel

from agent_harness.tools.base import BaseTool


class ToolRegistry:
    """工具注册表 — 模仿 LangChain Tool 注册管理"""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._categories: Dict[str, List[str]] = {}

    def register(self, tool: BaseTool) -> "ToolRegistry":
        self._tools[tool.name] = tool
        category = getattr(tool, "metadata", None)
        cat_name = category.category if category else "general"
        if cat_name not in self._categories:
            self._categories[cat_name] = []
        self._categories[cat_name].append(tool.name)
        return self

    def register_many(self, tools: List[BaseTool]) -> "ToolRegistry":
        for tool in tools:
            self.register(tool)
        return self

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)
        for cat in self._categories.values():
            if name in cat:
                cat.remove(name)

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def list_all(self) -> List[BaseTool]:
        return list(self._tools.values())

    def list_by_category(self, category: str) -> List[BaseTool]:
        names = self._categories.get(category, [])
        return [self._tools[n] for n in names if n in self._tools]

    def list_names(self, categories: Optional[List[str]] = None) -> List[str]:
        if categories:
            result = []
            for cat in categories:
                result.extend(self._categories.get(cat, []))
            return list(dict.fromkeys(result))
        return list(self._tools.keys())

    def filter(self, tags: Optional[List[str]] = None, categories: Optional[List[str]] = None) -> List[BaseTool]:
        tools = self.list_all()
        if categories:
            allowed = set()
            for cat in categories:
                allowed.update(self._categories.get(cat, []))
            tools = [t for t in tools if t.name in allowed]
        if tags:
            tools = [
                t
                for t in tools
                if any(tag in (getattr(t, "metadata", None) and t.metadata.tags or []) for tag in tags)
            ]
        return tools

    def get_schemas(self, names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        tools = [self._tools[n] for n in names] if names else self.list_all()
        schemas = []
        for tool in tools:
            meta = tool.get_schema()
            schemas.append({
                "type": "function",
                "function": {
                    "name": meta.name,
                    "description": meta.description,
                    "parameters": meta.parameters_schema,
                },
            })
        return schemas

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools
