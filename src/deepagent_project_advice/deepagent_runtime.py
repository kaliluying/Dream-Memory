from __future__ import annotations

from pathlib import Path
from typing import Any

from deepagents import SubAgent, create_deep_agent

from .analyzer import build_project_analysis
from .tools import list_files, read_file, search_code


DEEPAGENT_SYSTEM_PROMPT = """You are DeepAgent Memory, a local assistant built on the DeepAgents framework and focused on importing, reviewing, and injecting durable agent memory.

Operate safely:
- Read and analyze the local project before suggesting changes.
- Prefer plans and patches over direct writes.
- Explain risks, verification commands, and acceptance criteria.
- Keep the user in control of applying changes.
"""


def list_project_files_tool(project: str, limit: int = 500) -> list[str]:
    """List non-ignored files in a project."""
    return list_files(Path(project), limit=limit)


def read_project_file_tool(project: str, path: str, max_chars: int = 40_000) -> str:
    """Read a project file by relative path."""
    return read_file(Path(project), path, max_chars=max_chars)


def search_project_code_tool(project: str, query: str, limit: int = 50) -> list[dict[str, object]]:
    """Search code/text files for a query."""
    return search_code(Path(project), query, limit=limit)


def analyze_project_tool(project: str, task: str) -> dict[str, object]:
    """Analyze a local project and task, returning structured metadata and a Markdown plan."""
    analysis = build_project_analysis(Path(project), task)
    return analysis.to_json_payload()


def build_deepagent_subagents() -> list[SubAgent]:
    return [
        {
            "name": "code_reader",
            "description": "Reads the project structure and identifies relevant files and implementation patterns.",
            "system_prompt": "You are Code Reader. Inspect project files, identify the stack, relevant files, and existing conventions. Return concise findings with file paths.",
            "tools": [list_project_files_tool, read_project_file_tool, search_project_code_tool, analyze_project_tool],
        },
        {
            "name": "implementation_agent",
            "description": "Turns a project analysis into an implementation plan and safe patch strategy.",
            "system_prompt": "You are Implementation Agent. Produce a concrete implementation plan, recommended files to change, patch strategy, and validation commands. Do not directly modify files unless explicitly requested.",
            "tools": [list_project_files_tool, read_project_file_tool, search_project_code_tool, analyze_project_tool],
        },
        {
            "name": "reviewer_agent",
            "description": "Reviews risk, test coverage, safety boundaries, and readiness for delivery.",
            "system_prompt": "You are Reviewer Agent. Review risks, missing tests, unsafe operations, and acceptance criteria. Be strict and evidence-based.",
            "tools": [list_project_files_tool, read_project_file_tool, search_project_code_tool, analyze_project_tool],
        },
    ]


def build_deep_agent(model: str = "anthropic:claude-sonnet-4-6"):
    return create_deep_agent(
        model=model,
        tools=[list_project_files_tool, read_project_file_tool, search_project_code_tool, analyze_project_tool],
        system_prompt=DEEPAGENT_SYSTEM_PROMPT,
        subagents=build_deepagent_subagents(),
        name="deepagent_project_advice",
    )


def run_deepagent_advice(
    project_path: Path | str,
    task: str,
    *,
    model: str = "anthropic:claude-sonnet-4-6",
    dry_run: bool = True,
) -> dict[str, Any]:
    project = Path(project_path).expanduser().resolve()
    analysis = build_project_analysis(project, task)
    agent = build_deep_agent(model=model)
    subagents = build_deepagent_subagents()
    result: dict[str, Any] = {
        "runtime": "deepagents",
        "model": model,
        "dry_run": dry_run,
        "project": str(project),
        "task": task,
        "subagents": [agent_spec["name"] for agent_spec in subagents],
        "agent_graph": type(agent).__name__,
        "report": analysis.report,
        "metadata": analysis.to_metadata(),
    }
    if dry_run:
        result["messages"] = [
            "DeepAgents official graph constructed successfully.",
            "Dry-run mode skipped model invocation; pass dry_run=False to call the configured model.",
        ]
        return result

    prompt = (
        f"Project: {project}\n"
        f"Task: {task}\n\n"
        "Use project tools and subagents to produce an implementation plan, risk review, and validation checklist."
    )
    invocation = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    result["agent_result"] = invocation
    return result
