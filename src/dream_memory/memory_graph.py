from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .memory_agent import build_agent_candidates_from_payload, build_memory_extraction_prompt, extract_json_payload, _prompt_event_counts
from .model_providers import invoke_model as invoke_model_provider
from .model_providers import invoke_model_runtime


class MemoryExtractionState(TypedDict, total=False):
    events: list[dict[str, Any]]
    project: str | None
    model: str
    invoke_model: bool
    prompt: str
    raw_response: str
    atomic_facts: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    dry_run: bool
    runtime_config: dict[str, Any]
    model_runtime: dict[str, Any]
    trace_callback: Any
    input_event_count: int
    prompt_event_count: int
    filtered_prompt_event_count: int


def _build_prompt_node(state: MemoryExtractionState) -> dict[str, Any]:
    events = list(state.get("events", []))
    return {
        "prompt": build_memory_extraction_prompt(
            events,
            project=state.get("project"),
        ),
        **_prompt_event_counts(events),
    }


def _invoke_model_node(state: MemoryExtractionState) -> dict[str, Any]:
    prompt = str(state.get("prompt") or "")
    if not state.get("invoke_model", False):
        return {"dry_run": True, "atomic_facts": [], "candidates": []}
    runtime_config = state.get("runtime_config")
    if isinstance(runtime_config, dict):
        result = invoke_model_runtime(
            prompt,
            runtime_config=runtime_config,
            trace_callback=state.get("trace_callback"),
        )
        return {"dry_run": False, "raw_response": result.text, "model_runtime": result.to_dict()}
    raw_response = invoke_model_provider(prompt, model=str(state.get("model") or "anthropic:claude-sonnet-4-6"))
    return {"dry_run": False, "raw_response": raw_response}


def _validate_candidates_node(state: MemoryExtractionState) -> dict[str, Any]:
    if state.get("dry_run", False):
        return {"atomic_facts": [], "candidates": []}
    payload = extract_json_payload(str(state.get("raw_response") or ""))
    atomic_facts, candidates = build_agent_candidates_from_payload(payload, project=state.get("project"))
    return {"atomic_facts": atomic_facts, "candidates": candidates}


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
    runtime_config: dict[str, Any] | None = None,
    trace_callback: Any = None,
) -> MemoryExtractionState:
    graph = build_memory_extraction_graph()
    result = graph.invoke({
        "events": events,
        "project": project,
        "model": model,
        "invoke_model": invoke_model,
        "runtime_config": runtime_config,
        "trace_callback": trace_callback,
    })
    return dict(result)
