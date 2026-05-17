import ast
import io
import math
import operator as op
import sys
from pathlib import Path
from typing import Any

import httpx

from agent_harness.security.sandbox import SandboxedExecutor, SandboxConfig
from agent_harness.tools.base import BaseTool, ToolMetadata


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "执行安全的数学计算，支持 +, -, *, /, **, sqrt, sin, cos, log 等运算"

    metadata = ToolMetadata(
        name="calculator",
        description="安全的数学表达式计算器",
        parameters_schema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，例如 '2 + 3 * 4' 或 'sqrt(16)'",
                }
            },
            "required": ["expression"],
        },
        category="utility",
        tags=["math", "calculation"],
    )

    _SAFE_OPS = {
        ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
        ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg, ast.UAdd: op.pos,
    }

    _SAFE_FUNCS = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
        "tan": math.tan, "log": math.log, "log10": math.log10,
        "exp": math.exp, "pi": math.pi, "e": math.e,
        "ceil": math.ceil, "floor": math.floor,
    }

    async def _arun(self, expression: str = "", **kwargs) -> Any:
        expr_str = expression or kwargs.get("input", "")
        tree = ast.parse(expr_str, mode="eval")
        return self._eval_node(tree.body)

    def _eval_node(self, node) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.BinOp):
            left, right = self._eval_node(node.left), self._eval_node(node.right)
            op_func = self._SAFE_OPS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"不支持的操作符: {type(node.op).__name__}")
            return op_func(left, right)
        if isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op_func = self._SAFE_OPS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"不支持的一元操作符: {type(node.op).__name__}")
            return op_func(operand)
        if isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else ""
            if func_name not in self._SAFE_FUNCS:
                raise ValueError(f"不支持的函数: {func_name}")
            args = [self._eval_node(a) for a in node.args]
            return self._SAFE_FUNCS[func_name](*args)
        if isinstance(node, ast.Name):
            name = node.id
            if name in self._SAFE_FUNCS:
                val = self._SAFE_FUNCS[name]
                if callable(val):
                    raise ValueError(f"函数 '{name}' 需要参数")
                return val
            raise ValueError(f"未定义的变量: {name}")
        raise ValueError(f"不支持的 AST 节点: {type(node).__name__}")


class PythonREPLTool(BaseTool):
    name = "python_repl"
    description = "在沙箱中安全执行 Python 代码，捕获 stdout 返回结果"

    metadata = ToolMetadata(
        name="python_repl",
        description="在受限沙箱中执行 Python 代码并返回输出",
        parameters_schema={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "要执行的 Python 代码",
                }
            },
            "required": ["code"],
        },
        category="code",
        tags=["python", "repl", "execution", "sandbox"],
    )

    def __init__(self, sandbox_config: SandboxConfig = None):
        self._sandbox = SandboxedExecutor(sandbox_config or SandboxConfig())

    async def _arun(self, code: str = "", **kwargs) -> Any:
        code_str = code or kwargs.get("input", "")
        result = await self._sandbox.execute(code_str)
        if result.error:
            return f"执行错误: {result.error}"
        output = result.output
        if result.truncated:
            output += "\n... (output truncated)"
        return output


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "使用 HTTP 请求搜索网络（需要配置搜索引擎 API）"

    metadata = ToolMetadata(
        name="web_search",
        description="搜索互联网获取信息",
        parameters_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询关键字",
                }
            },
            "required": ["query"],
        },
        category="network",
        tags=["search", "web", "internet"],
    )

    def __init__(self, search_url: str = "https://api.duckduckgo.com/", timeout: float = 10.0):
        self._search_url = search_url
        self._timeout = timeout

    async def _arun(self, query: str = "", **kwargs) -> Any:
        query_str = query or kwargs.get("input", "")
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.get(
                    self._search_url,
                    params={"q": query_str, "format": "json"},
                )
                response.raise_for_status()
                data = response.json()
                results = []
                for item in data.get("RelatedTopics", [])[:5]:
                    if "Text" in item:
                        results.append(item["Text"])
                return "\n".join(results) if results else "未找到搜索结果"
            except Exception as e:
                return f"搜索失败: {e}"


class FileReadTool(BaseTool):
    name = "read_file"
    description = "读取本地文件内容"

    metadata = ToolMetadata(
        name="read_file",
        description="读取文件内容",
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "文件路径",
                }
            },
            "required": ["path"],
        },
        category="file",
        tags=["file", "read", "io"],
    )

    async def _arun(self, path: str = "", **kwargs) -> Any:
        path_str = path or kwargs.get("input", "")
        filepath = Path(path_str).expanduser().resolve()
        if not filepath.exists():
            return f"文件不存在: {filepath}"
        try:
            content = filepath.read_text(encoding="utf-8")
            if len(content) > 10000:
                content = content[:10000] + "\n... (truncated)"
            return content
        except Exception as e:
            return f"读取失败: {e}"


class FileWriteTool(BaseTool):
    name = "write_file"
    description = "写入内容到本地文件，需要用户确认"

    metadata = ToolMetadata(
        name="write_file",
        description="写入文件内容",
        parameters_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "文件路径"},
                "content": {"type": "string", "description": "要写入的内容"},
            },
            "required": ["path", "content"],
        },
        category="file",
        tags=["file", "write", "io"],
        requires_confirmation=True,
    )

    async def _arun(self, path: str = "", content: str = "", **kwargs) -> Any:
        path_str = path or kwargs.get("path", "")
        content_str = content or kwargs.get("content", "")
        filepath = Path(path_str).expanduser().resolve()
        filepath.parent.mkdir(parents=True, exist_ok=True)
        try:
            filepath.write_text(content_str, encoding="utf-8")
            return f"成功写入: {filepath} ({len(content_str)} 字符)"
        except Exception as e:
            return f"写入失败: {e}"
