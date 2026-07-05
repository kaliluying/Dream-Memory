from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .memory_agent import agent_extract_memory_candidates
from .memory_config import DEFAULT_CONFIG_PATH, load_memory_config, write_default_memory_config
from .memory_dreaming import (
    apply_reviewed_memory,
    build_agent_context,
    build_review_queue,
    dream_from_events,
    render_context_markdown,
    extract_atomic_facts,
    load_events_jsonl,
    write_jsonl_records,
)
from .memory_importers import ClaudeCodeImporter, CodexImporter, NormalizedSessionEvent, write_events_jsonl
from .memory_runs import (
    append_trace,
    copy_input_events,
    create_run_state,
    list_runs,
    load_run_state,
    read_trace,
    update_run_state,
    write_candidate_traces,
)


def _default_project_roots(values: list[str] | None) -> list[Path]:
    return [Path(value).expanduser() for value in values or []]


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--codex-home", default=str(Path.home() / ".codex"))
    parser.add_argument("--claude-home", default=str(Path.home() / ".claude"))
    parser.add_argument("--claude-state", default=str(Path.home() / ".claude.json"))
    parser.add_argument("--project", action="append", default=[])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deepagent-memory", description="Scan and import Claude Code / Codex sessions into shared memory events.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Memory config JSON path")
    _add_source_args(parser)
    sub = parser.add_subparsers(dest="command", required=True)

    init_config = sub.add_parser("init-config", help="Write a default editable memory config file")
    init_config.add_argument("--output", default=str(DEFAULT_CONFIG_PATH))

    scan = sub.add_parser("scan", help="Scan available Codex/Claude sources")
    _add_source_args(scan)
    scan.add_argument("--output")

    imp = sub.add_parser("import", help="Import normalized events")
    _add_source_args(imp)
    imp.add_argument("source", choices=["codex", "claude", "all"])
    imp.add_argument("--output-dir")
    imp.add_argument("--dry-run", action="store_true")
    dream = sub.add_parser("dream", help="Run memory dreaming over normalized events")
    dream.add_argument("--input", required=True, help="Input normalized events JSONL")
    dream.add_argument("--project")
    dream.add_argument("--output-dir")
    dream.add_argument("--apply", action="store_true", help="Append promoted preview to MEMORY.md")
    dream.add_argument("--mode", choices=["ai", "rules"], help="Extraction mode: ai is default; rules is fallback/debug")
    dream.add_argument("--agent", action="store_true", help="Deprecated alias for --mode ai")
    dream.add_argument("--provider")
    dream.add_argument("--model")
    dream.add_argument("--api-key-env")
    dream.add_argument("--base-url")
    dream.add_argument("--timeout-seconds", type=int)
    dream.set_defaults(invoke_model=None)
    dream.add_argument("--dry-run", action="store_false", dest="invoke_model", help="Write the AI prompt only; do not invoke the model")
    dream.add_argument("--invoke-model", action="store_true", dest="invoke_model", help="Invoke the model; this is the default")

    extract = sub.add_parser("extract-facts", help="Extract atomic facts from normalized events")
    extract.add_argument("--input", required=True)
    extract.add_argument("--project")
    extract.add_argument("--output-dir")

    review = sub.add_parser("review", help="Build review queue items from candidates")
    review.add_argument("--candidates", required=True)
    review.add_argument("--memory-cards")
    review.add_argument("--output-dir")

    apply_cmd = sub.add_parser("apply", help="Apply reviewed memory decisions")
    apply_cmd.add_argument("--reviewed", required=True)
    apply_cmd.add_argument("--memory-cards")
    apply_cmd.add_argument("--output-dir")
    apply_cmd.add_argument("--reviewer", required=True)

    context = sub.add_parser("context", help="Render task-scoped memory context for agents")
    context.add_argument("--project")
    context.add_argument("--memory-cards")
    context.add_argument("--limit", type=int)
    context.add_argument("--format", choices=["json", "markdown"])

    pipeline = sub.add_parser("pipeline", help="Run dream and review in one step")
    pipeline.add_argument("--input", required=True)
    pipeline.add_argument("--project")
    pipeline.add_argument("--output-dir")
    pipeline.add_argument("--memory-cards")
    pipeline.add_argument("--mode", choices=["ai", "rules"])
    pipeline.add_argument("--provider")
    pipeline.add_argument("--model")
    pipeline.add_argument("--api-key-env")
    pipeline.add_argument("--base-url")
    pipeline.add_argument("--timeout-seconds", type=int)
    pipeline.set_defaults(invoke_model=None)
    pipeline.add_argument("--dry-run", action="store_false", dest="invoke_model", help="Write the AI prompt only; do not invoke the model")
    pipeline.add_argument("--invoke-model", action="store_true", dest="invoke_model", help="Invoke the model; this is the default")

    run = sub.add_parser("run", help="Create a persistent resumable Dream Memory run")
    run.add_argument("--input", required=True)
    run.add_argument("--project")
    run.add_argument("--output-dir")
    run.add_argument("--memory-cards")
    run.add_argument("--mode", choices=["ai", "rules"])
    run.add_argument("--provider")
    run.add_argument("--model")
    run.add_argument("--api-key-env")
    run.add_argument("--base-url")
    run.add_argument("--timeout-seconds", type=int)
    run.set_defaults(invoke_model=None)
    run.add_argument("--dry-run", action="store_false", dest="invoke_model")
    run.add_argument("--invoke-model", action="store_true", dest="invoke_model")

    status = sub.add_parser("status", help="Show one run state or list runs")
    status.add_argument("--run-id")
    status.add_argument("--output-dir")

    resume = sub.add_parser("resume", help="Resume a run after review decisions are available")
    resume.add_argument("--run-id", required=True)
    resume.add_argument("--output-dir")
    resume.add_argument("--reviewed")
    resume.add_argument("--memory-cards")
    resume.add_argument("--reviewer", default="user")

    trace = sub.add_parser("trace", help="Print run or candidate trace")
    trace.add_argument("--run-id", required=True)
    trace.add_argument("--candidate-id")
    trace.add_argument("--output-dir")

    return parser


def _build_importers(args: argparse.Namespace, config: dict[str, object]) -> tuple[CodexImporter, ClaudeCodeImporter]:
    codex_home = args.codex_home or str(config["codex_home"])
    claude_home = args.claude_home or str(config["claude_home"])
    claude_state = args.claude_state or str(config["claude_state"])
    codex = CodexImporter(codex_home=Path(codex_home).expanduser())
    claude = ClaudeCodeImporter(
        claude_home=Path(claude_home).expanduser(),
        global_state_path=Path(claude_state).expanduser(),
        project_roots=_default_project_roots(args.project),
    )
    return codex, claude


def _load_optional_jsonl(path_value: str | None) -> list[dict[str, object]]:
    if not path_value:
        return []
    path = Path(path_value).expanduser()
    if not path.exists():
        return []
    return load_events_jsonl(path)


def _write_report(output_dir: Path, payload: dict[str, object]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "import-report.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report_path


def _value(value: object | None, fallback: object) -> object:
    return fallback if value is None else value


def _configured_output_dir(args: argparse.Namespace, config: dict[str, object]) -> Path:
    return Path(str(_value(getattr(args, "output_dir", None), config["output_dir"]))).expanduser()


def _configured_model(args: argparse.Namespace, config: dict[str, object]) -> str:
    provider = _value(getattr(args, "provider", None), config.get("provider"))
    model = str(_value(getattr(args, "model", None), config["model"]))
    if provider and ":" not in model:
        return f"{provider}:{model}"
    return model


def _apply_provider_env(args: argparse.Namespace, config: dict[str, object]) -> None:
    import os
    api_key_env = _value(getattr(args, "api_key_env", None), config.get("api_key_env"))
    base_url = _value(getattr(args, "base_url", None), config.get("base_url"))
    timeout = _value(getattr(args, "timeout_seconds", None), config.get("timeout_seconds"))
    if api_key_env:
        os.environ["DEEPAGENT_MEMORY_API_KEY_ENV"] = str(api_key_env)
    if base_url:
        os.environ["DEEPAGENT_MEMORY_BASE_URL"] = str(base_url)
    if timeout:
        os.environ["DEEPAGENT_MEMORY_TIMEOUT_SECONDS"] = str(timeout)


def _run_dream_to_review(
    *,
    args: argparse.Namespace,
    config: dict[str, object],
    persistent: bool,
) -> tuple[dict[str, object], dict[str, object] | None]:
    events = load_events_jsonl(Path(args.input))
    output_dir = _configured_output_dir(args, config)
    mode = str(_value(args.mode, config["mode"]))
    model = _configured_model(args, config)
    invoke_model = bool(_value(args.invoke_model, config["invoke_model"]))
    _apply_provider_env(args, config)
    state: dict[str, object] | None = None
    if persistent:
        state = create_run_state(memory_dir=output_dir, project=args.project, input_path=str(args.input), mode=mode, model=model, invoke_model=invoke_model)
        events_path = copy_input_events(args.input, state)
        state = update_run_state(state, status="running", phase="extracting", artifacts={"events_path": str(events_path)})
        append_trace(state, "events_copied", {"events_path": str(events_path), "event_count": len(events)})
        working_dir = Path(str(state["run_dir"]))
    else:
        working_dir = output_dir
    if mode == "ai":
        extraction = agent_extract_memory_candidates(events, project=args.project, model=model, invoke_model=invoke_model)
        working_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = working_dir / "agent-prompt.md"
        prompt_path.write_text(str(extraction["prompt"]), encoding="utf-8")
        artifacts = {"agent_prompt_path": str(prompt_path)}
        if "raw_response" in extraction:
            raw_path = working_dir / "agent-raw-response.txt"
            raw_path.write_text(str(extraction["raw_response"]), encoding="utf-8")
            artifacts["agent_raw_response_path"] = str(raw_path)
        if state:
            state = update_run_state(state, phase="candidate_validation", artifacts=artifacts)
            append_trace(state, "ai_extraction_complete", {"dry_run": extraction["dry_run"], "candidate_count": len(extraction.get("candidates", []))})
        result = dream_from_events(events, project=args.project, output_dir=working_dir, apply=False, agent_candidates=list(extraction.get("candidates", [])), agent_mode=True)
        payload = {**result.to_dict(), "mode": "ai", "agent_dry_run": extraction["dry_run"], "agent_prompt_path": str(prompt_path)}
    else:
        result = dream_from_events(events, project=args.project, output_dir=working_dir, apply=False)
        payload = {**result.to_dict(), "mode": "rules"}
        if state:
            append_trace(state, "rules_extraction_complete", {"candidate_count": result.candidate_count})
    candidates = load_events_jsonl(Path(result.candidates_path))
    memory_cards = _load_optional_jsonl(str(_value(args.memory_cards, config["memory_cards"])))
    queue = build_review_queue(candidates, memory_cards)
    queue_path = write_jsonl_records(queue, working_dir / "review_queue.jsonl")
    payload = {**payload, "review_queue_path": str(queue_path), "review_count": len(queue)}
    if state:
        state = update_run_state(
            state,
            status="waiting_review",
            phase="review",
            artifacts={
                "candidates_path": str(result.candidates_path),
                "dreams_path": str(result.dreams_path),
                "memory_preview_path": str(result.memory_preview_path),
                "review_queue_path": str(queue_path),
            },
            counts={
                "event_count": result.event_count,
                "candidate_count": result.candidate_count,
                "review_count": len(queue),
            },
            next_actions=["review candidates", f"deepagent-memory resume --run-id {state['run_id']}"],
        )
        write_candidate_traces(state, candidates)
        append_trace(state, "waiting_review", {"review_queue_path": str(queue_path), "review_count": len(queue)})
        payload = {**payload, "run_id": state["run_id"], "run_dir": state["run_dir"], "state_path": str(Path(str(state["run_dir"])) / "state.json")}
    return payload, state


def _resume_run(*, args: argparse.Namespace, config: dict[str, object]) -> dict[str, object]:
    output_dir = _configured_output_dir(args, config)
    state = load_run_state(output_dir, args.run_id)
    reviewed_path = Path(args.reviewed).expanduser() if args.reviewed else Path(str(state["run_dir"])) / "reviewed.jsonl"
    reviewed = load_events_jsonl(reviewed_path) if reviewed_path.exists() else []
    existing_cards = _load_optional_jsonl(str(_value(args.memory_cards, config["memory_cards"])))
    cards, markdown, decisions = apply_reviewed_memory(reviewed, existing_cards, return_decisions=True)
    cards_path = write_jsonl_records(cards, output_dir / "memory_cards.jsonl")
    decisions_path = write_jsonl_records(decisions, output_dir / "review_decisions.jsonl")
    memory_path = output_dir / "MEMORY.md"
    memory_path.write_text(markdown, encoding="utf-8")
    state = update_run_state(
        state,
        status="completed",
        phase="applied",
        artifacts={
            "reviewed_path": str(reviewed_path),
            "memory_cards_path": str(cards_path),
            "review_decisions_path": str(decisions_path),
            "memory_path": str(memory_path),
        },
        counts={"review_decision_count": len(decisions), "memory_count": len(cards)},
        next_actions=["generate context", "inspect trace"],
    )
    append_trace(state, "run_completed", {"reviewed_path": str(reviewed_path), "memory_count": len(cards), "decision_count": len(decisions)})
    return state


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_memory_config(args.config)

    if args.command == "init-config":
        path = write_default_memory_config(args.output)
        print(json.dumps({"config_path": str(path)}, ensure_ascii=False, indent=2))
        return 0

    codex, claude = _build_importers(args, config)
    if args.command == "scan":
        payload = {"codex": codex.scan(), "claude": claude.scan()}
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.output:
            path = Path(args.output).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
        return 0

    if args.command == "extract-facts":
        events = load_events_jsonl(Path(args.input))
        facts = extract_atomic_facts(events, project=args.project)
        output_dir = _configured_output_dir(args, config)
        facts_path = write_jsonl_records(facts, output_dir / "facts.jsonl")
        print(json.dumps({"fact_count": len(facts), "facts_path": str(facts_path)}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "review":
        candidates = load_events_jsonl(Path(args.candidates))
        memory_cards = _load_optional_jsonl(str(_value(args.memory_cards, config["memory_cards"])))
        queue = build_review_queue(candidates, memory_cards)
        output_dir = _configured_output_dir(args, config)
        queue_path = write_jsonl_records(queue, output_dir / "review_queue.jsonl")
        print(json.dumps({"review_count": len(queue), "review_queue_path": str(queue_path)}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "apply":
        reviewed = load_events_jsonl(Path(args.reviewed))
        existing_cards = _load_optional_jsonl(str(_value(args.memory_cards, config["memory_cards"])))
        cards, markdown, decisions = apply_reviewed_memory(reviewed, existing_cards, return_decisions=True)
        output_dir = _configured_output_dir(args, config)
        output_dir.mkdir(parents=True, exist_ok=True)
        cards_path = write_jsonl_records(cards, output_dir / "memory_cards.jsonl")
        decisions_path = write_jsonl_records(decisions, output_dir / "review_decisions.jsonl")
        memory_path = output_dir / "MEMORY.md"
        memory_path.write_text(markdown, encoding="utf-8")
        print(json.dumps({"memory_count": len(cards), "memory_cards_path": str(cards_path), "review_decisions_path": str(decisions_path), "memory_path": str(memory_path), "reviewer": args.reviewer}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "context":
        memory_cards_path = str(_value(args.memory_cards, config["memory_cards"]))
        cards = load_events_jsonl(Path(memory_cards_path))
        payload = build_agent_context(cards, project=args.project, limit=int(_value(args.limit, config["context_limit"])))
        context_format = str(_value(args.format, config["context_format"]))
        if context_format == "markdown":
            print(render_context_markdown(payload), end="")
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "pipeline":
        payload, _ = _run_dream_to_review(args=args, config=config, persistent=False)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "run":
        payload, _ = _run_dream_to_review(args=args, config=config, persistent=True)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "status":
        output_dir = _configured_output_dir(args, config)
        if args.run_id:
            payload = load_run_state(output_dir, args.run_id)
        else:
            payload = {"runs": list_runs(output_dir)}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "resume":
        payload = _resume_run(args=args, config=config)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "trace":
        output_dir = _configured_output_dir(args, config)
        if args.candidate_id:
            candidate_path = output_dir / "runs" / args.run_id / "candidates" / f"{args.candidate_id}.json"
            payload = json.loads(candidate_path.read_text(encoding="utf-8")) if candidate_path.exists() else {"candidate_id": args.candidate_id, "trace": read_trace(output_dir, args.run_id, candidate_id=args.candidate_id)}
        else:
            payload = {"run_id": args.run_id, "trace": read_trace(output_dir, args.run_id)}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "dream":
        events = load_events_jsonl(Path(args.input))
        output_dir = _configured_output_dir(args, config)
        mode = "ai" if args.agent else str(_value(args.mode, config["mode"]))
        model = _configured_model(args, config)
        invoke_model = bool(_value(args.invoke_model, config["invoke_model"]))
        _apply_provider_env(args, config)
        if mode == "ai":
            extraction = agent_extract_memory_candidates(events, project=args.project, model=model, invoke_model=invoke_model)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "agent-prompt.md").write_text(str(extraction["prompt"]), encoding="utf-8")
            if "raw_response" in extraction:
                (output_dir / "agent-raw-response.txt").write_text(str(extraction["raw_response"]), encoding="utf-8")
            result = dream_from_events(
                events,
                project=args.project,
                output_dir=output_dir,
                apply=bool(args.apply),
                agent_candidates=list(extraction.get("candidates", [])),
                agent_mode=True,
            )
            payload = {**result.to_dict(), "mode": "ai", "agent": True, "agent_dry_run": extraction["dry_run"], "agent_prompt_path": str(output_dir / "agent-prompt.md")}
        else:
            result = dream_from_events(events, project=args.project, output_dir=output_dir, apply=bool(args.apply))
            payload = {**result.to_dict(), "mode": "rules"}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    output_dir = Path(str(_value(args.output_dir, config["imports_output_dir"]))).expanduser()
    events: list[NormalizedSessionEvent] = []
    written_files: list[str] = []
    if args.source in {"codex", "all"}:
        codex_events = codex.import_events()
        events.extend(codex_events)
        written_files.append(str(write_events_jsonl(codex_events, output_dir / "codex-events.jsonl")))
    if args.source in {"claude", "all"}:
        claude_events = claude.import_events()
        events.extend(claude_events)
        written_files.append(str(write_events_jsonl(claude_events, output_dir / "claude-events.jsonl")))
    combined_name = f"{args.source}-events.jsonl"
    written_files.append(str(write_events_jsonl(events, output_dir / combined_name)))
    _write_report(output_dir, {
        "source": args.source,
        "dry_run": bool(args.dry_run),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "event_count": len(events),
        "files": written_files,
        "next_steps": [
            "Review normalized events before promoting memory.",
            "Run future dreaming/consolidation over these events.",
        ],
    })
    print(json.dumps({"event_count": len(events), "output_dir": str(output_dir), "dry_run": bool(args.dry_run)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
