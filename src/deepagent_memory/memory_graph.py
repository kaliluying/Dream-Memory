from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .memory_agent import build_memory_extraction_prompt, extract_json_payload, validate_agent_candidates
from .model_providers import invoke_model as invoke_model_provider


class MemoryExtractionState(TypedDict, total=False):
    events: list[dict[str, Any]]
    project: str | None
    model: str
    invoke_model: bool
    prompt: str
    raw_response: str
    candidates: list[dict[str, Any]]
    dry_run: bool


def _build_prompt_node(state: MemoryExtractionState) -> dict[str, Any]:
    return {
        "prompt": build_memory_extraction_prompt(
            list(state.get("events", [])),
            project=state.get("project"),
        )
    }


def _invoke_model_node(state: MemoryExtractionState) -> dict[str, Any]:
    prompt = str(state.get("prompt") or "")
    if not state.get("invoke_model", False):
        return {"dry_run": True, "candidates": []}
    raw_response = invoke_model_provider(prompt, model=str(state.get("model") or "anthropic:claude-sonnet-4-6"))
    return {"dry_run": False, "raw_response": raw_response}


def _validate_candidates_node(state: MemoryExtractionState) -> dict[str, Any]:
    if state.get("dry_run", False):
        return {"candidates": []}
    payload = extract_json_payload(str(state.get("raw_response") or ""))
    candidates = validate_agent_candidates(list(payload.get("candidates", [])), project=state.get("project"))
    return {"candidates": candidates}


def build_memory_extraction_graph():
    graph = StateGraph(MemoryExtractionState)
    graph.add_node("build_prompt", _build_prompt_node)
    graph.add_node("invoke_model", _invoke_model_node)
    graph.add_node("validate_candidates", _validate_candidates_node)
    graph.add_edge(START, "build_prompt")
    graph.add_edge("build_prompt", "invoke_model")
    graph.add_edge("invoke_model", "validate_candidates")
    graph.add_edge("validate_candidates", END)
    return graph.compile()


def run_memory_extraction_graph(
    events: list[dict[str, Any]],
    *,
    project: str | None,
    model: str,
    invoke_model: bool,
) -> MemoryExtractionState:
    graph = build_memory_extraction_graph()
    result = graph.invoke({
        "events": events,
        "project": project,
        "model": model,
        "invoke_model": invoke_model,
    })
    return dict(result)
