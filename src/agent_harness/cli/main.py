import asyncio
import os

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
import typer

app = typer.Typer(name="agent-harness", help="Agent Harness CLI")
console = Console()


def _build_executor(
    model: str = "deepseek-chat",
    api_key: str = "",
    base_url: str = "https://api.deepseek.com/v1",
):
    from agent_harness import AgentExecutor
    from agent_harness.agents import ReActAgent
    from agent_harness.harness.agent_executor import AgentExecutorConfig
    from agent_harness.llm import OpenAILLM
    from agent_harness.tools import ToolRegistry, CalculatorTool, PythonREPLTool

    llm = OpenAILLM(model=model, api_key=api_key, base_url=base_url)
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(PythonREPLTool())

    agent = ReActAgent(llm=llm, tools=registry.list_all(), max_steps=8)
    return AgentExecutor(agent=agent, tool_registry=registry, config=AgentExecutorConfig(max_steps=8))


@app.command()
def run(
    query: str = typer.Argument(None, help="Query to run"),
    model: str = typer.Option("deepseek-chat", "--model", "-m"),
    api_key: str = typer.Option("", "--api-key", "-k"),
    base_url: str = typer.Option("https://api.deepseek.com/v1", "--base-url", "-b"),
):
    """Execute a single query"""
    key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
    executor = _build_executor(model, key, base_url)
    result = asyncio.run(executor.arun(query or "Hello"))
    console.print(Panel(Markdown(result.result), title="Result", border_style="green"))


@app.command()
def chat(
    model: str = typer.Option("deepseek-chat", "--model", "-m"),
    api_key: str = typer.Option("", "--api-key", "-k"),
    base_url: str = typer.Option("https://api.deepseek.com/v1", "--base-url", "-b"),
):
    """Interactive chat mode"""
    key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
    executor = _build_executor(model, key, base_url)
    console.print(Panel(
        f"Agent Harness Chat\nModel: {model}\nType /exit to quit, /clear to reset, /tools to list tools",
        title="Chat",
    ))
    while True:
        query = console.input("[bold cyan]You:[/] ")
        if not query.strip():
            continue
        if query == "/exit":
            break
        if query == "/clear":
            console.clear()
            continue
        if query == "/tools":
            tools = executor.tools.list_all()
            table = Table(title="Tools")
            table.add_column("Name")
            table.add_column("Description")
            for t in tools:
                table.add_row(t.name, t.description[:60])
            console.print(table)
            continue
        with console.status("[bold yellow]Thinking...[/]"):
            result = asyncio.run(executor.arun(query))
        console.print(Panel(Markdown(result.result), border_style="green"))


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h"),
    port: int = typer.Option(8080, "--port", "-p"),
    model: str = typer.Option("deepseek-chat", "--model", "-m"),
    api_key: str = typer.Option("", "--api-key", "-k"),
    base_url: str = typer.Option("https://api.deepseek.com/v1", "--base-url", "-b"),
):
    """Start web dashboard"""
    from agent_harness.web.server import AgentAPI
    key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
    executor = _build_executor(model, key, base_url)
    api = AgentAPI(executor)
    console.print(f"[green]Dashboard: http://{host}:{port}[/]")
    api.run(host=host, port=port)


@app.command()
def eval(
    model: str = typer.Option("deepseek-chat", "--model", "-m"),
    api_key: str = typer.Option("", "--api-key", "-k"),
    base_url: str = typer.Option("https://api.deepseek.com/v1", "--base-url", "-b"),
):
    """Run evaluation benchmarks"""
    from agent_harness.evaluation.benchmark import BenchmarkRunner
    key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
    executor = _build_executor(model, key, base_url)
    runner = BenchmarkRunner(executor)
    results = asyncio.run(runner.run_all())
    table = Table(title="Benchmark Results")
    table.add_column("Test")
    table.add_column("Status")
    table.add_column("Duration")
    table.add_column("Score")
    for r in results:
        status = "[green]PASS" if r["passed"] else "[red]FAIL"
        table.add_row(r["name"], status, f"{r['duration_ms']:.0f}ms", f"{r['score']:.1f}")
    console.print(table)
