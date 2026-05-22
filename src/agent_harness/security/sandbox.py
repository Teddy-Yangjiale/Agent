import asyncio
import ast
import io
import os
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SandboxConfig:
    max_memory_mb: int = 256
    max_timeout_sec: int = 30
    max_output_bytes: int = 100 * 1024
    allow_imports: list = field(default_factory=lambda: [
        "math", "json", "re", "itertools", "collections",
        "datetime", "decimal", "fractions", "random",
        "statistics", "string", "typing", "dataclasses",
        "functools", "operator", "hashlib", "base64",
        "csv", "io", "textwrap", "unicodedata",
        "copy", "enum", "numbers", "uuid", "pprint",
    ])
    allowed_builtins: list = field(default_factory=lambda: [
        "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes",
        "callable", "chr", "classmethod", "complex", "delattr", "dict",
        "dir", "divmod", "enumerate", "filter", "float", "format", "frozenset",
        "getattr", "hasattr", "hash", "hex", "id", "int", "isinstance",
        "issubclass", "iter", "len", "list", "map", "max", "min", "next",
        "object", "oct", "ord", "pow", "print", "property", "range",
        "repr", "reversed", "round", "set", "slice", "sorted", "staticmethod",
        "str", "sum", "super", "tuple", "type", "vars", "zip", "True", "False",
        "None", "Exception", "ValueError", "TypeError", "KeyError",
        "IndexError", "StopIteration", "RuntimeError", "ZeroDivisionError",
        "AssertionError", "ArithmeticError", "AttributeError", "EOFError",
        "ImportError", "KeyboardInterrupt", "LookupError", "MemoryError",
        "NameError", "NotImplementedError", "OSError", "OverflowError",
        "TabError", "TimeoutError", "UnboundLocalError", "UnicodeError",
        "UnicodeDecodeError", "UnicodeEncodeError", "UnicodeTranslateError",
    ])
    docker_image: str = "python:3.11-slim"
    use_docker: bool = True
    fallback_to_restricted: bool = True


@dataclass
class SandboxResult:
    output: str = ""
    error: str = ""
    duration_ms: float = 0
    truncated: bool = False


class SandboxedExecutor:
    """安全沙箱执行器 — 两种模式：进程内限制模式 和 Docker 隔离模式"""

    def __init__(self, config: Optional[SandboxConfig] = None):
        self._config = config or SandboxConfig()

    async def execute(self, code: str) -> SandboxResult:
        if self._config.use_docker:
            result = await self._execute_docker(code)
            if (
                result.error == "Docker not found — set use_docker=False for restricted mode"
                and self._config.fallback_to_restricted
            ):
                return await self._execute_restricted(code)
            return result
        return await self._execute_restricted(code)

    async def _execute_restricted(self, code: str) -> SandboxResult:
        import time as time_mod
        start = time_mod.monotonic()
        validation_error = self._validate_restricted_code(code)
        if validation_error:
            return SandboxResult(
                error=validation_error,
                duration_ms=(time_mod.monotonic() - start) * 1000,
            )

        safe_builtins = {
            name: getattr(__builtins__, name, None) or __builtins__.get(name)
            for name in self._config.allowed_builtins
        }
        safe_builtins["__import__"] = self._safe_import
        safe_builtins["__build_class__"] = __build_class__

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        captured_out = io.StringIO()
        captured_err = io.StringIO()
        sys.stdout = captured_out
        sys.stderr = captured_err

        result = SandboxResult()

        try:
            compiled = compile(code, f"<sandbox_{uuid.uuid4().hex[:8]}>", "exec")
            exec_globals = {"__builtins__": safe_builtins}
            exec(compiled, exec_globals)

            output = captured_out.getvalue()
            if captured_err.getvalue():
                output = captured_err.getvalue() + "\n" + output

            if len(output) > self._config.max_output_bytes:
                output = output[:self._config.max_output_bytes] + "\n... (truncated)"
                result.truncated = True

            result.output = output.strip() or "(no output)"

        except Exception as e:
            result.error = f"{type(e).__name__}: {e}"
            result.output = captured_out.getvalue() or ""

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        result.duration_ms = (time_mod.monotonic() - start) * 1000
        return result

    def _validate_restricted_code(self, code: str) -> str:
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as e:
            return f"SyntaxError: {e}"

        blocked_nodes = (
            ast.While,
            ast.For,
            ast.AsyncFor,
            ast.With,
            ast.AsyncWith,
            ast.Try,
            ast.Global,
            ast.Nonlocal,
            ast.Lambda,
        )
        for node in ast.walk(tree):
            if isinstance(node, blocked_nodes):
                return f"Restricted sandbox disallows {type(node).__name__}"
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                return "Restricted sandbox disallows dunder attribute access"
        return ""

    def _safe_import(self, name, *args, **kwargs):
        if name not in self._config.allow_imports:
            raise ImportError(f"模块 '{name}' 不在沙箱白名单中")
        return __import__(name, *args, **kwargs)

    async def _execute_docker(self, code: str) -> SandboxResult:
        import time as time_mod
        start = time_mod.monotonic()
        result = SandboxResult()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "run", "--rm",
                "--memory", f"{self._config.max_memory_mb}m",
                "--network", "none",
                "--cpus", "1",
                "-v", f"{tmp_path}:/code/user_script.py:ro",
                self._config.docker_image,
                "python", "/code/user_script.py",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self._config.max_timeout_sec,
            )
            output = (stdout or b"").decode("utf-8", errors="replace")
            err = (stderr or b"").decode("utf-8", errors="replace")
            if err:
                output = err + "\n" + output

            if len(output) > self._config.max_output_bytes:
                output = output[:self._config.max_output_bytes] + "\n... (truncated)"
                result.truncated = True

            result.output = output.strip() or "(no output)"
        except asyncio.TimeoutError:
            result.error = "Sandbox execution timeout"
        except FileNotFoundError:
            result.error = "Docker not found — set use_docker=False for restricted mode"
        except Exception as e:
            result.error = f"Docker sandbox error: {e}"
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        result.duration_ms = (time_mod.monotonic() - start) * 1000
        return result
