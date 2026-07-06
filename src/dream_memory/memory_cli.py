from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .memory_agent import agent_extract_memory_candidates
from .memory_config import DEFAULT_CONFIG_PATH, load_memory_config, write_default_memory_config
from .memory_export import render_all_projects_summary, write_marked_file
from .memory_eval import evaluate_labeled_events
from .memory_dreaming import (
    apply_reviewed_memory,
    build_agent_context,
    build_review_queue,
    dream_from_events,
    render_context_markdown,
    normalize_project_path,
    extract_atomic_facts,
    load_events_jsonl,
    write_jsonl_records,
)
from .memory_importers import ClaudeCodeImporter, CodexImporter, NormalizedSessionEvent, write_events_jsonl
from .model_providers import provider_diagnostics, runtime_parts_from_config
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


def _init_workspace(path: Path | str, *, force: bool = False) -> dict[str, object]:
    root = Path(path).expanduser()
    memory_dir = root / ".dream-memory"
    imports_dir = memory_dir / "imports"
    runs_dir = memory_dir / "runs"
    examples_dir = root / "examples"
    memory_dir.mkdir(parents=True, exist_ok=True)
    imports_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / ".gitkeep").touch()

    config_path = memory_dir / "config.json"
    if force or not config_path.exists():
        write_default_memory_config(config_path)

    sample_events = examples_dir / "sample-events.jsonl"
    if force or not sample_events.exists():
        examples_dir.mkdir(parents=True, exist_ok=True)
        sample_events.write_text(
            '{"event_id":"event_1","source":"codex","session_id":"sample","role":"user","event_type":"history_prompt","project":".","content":"用户偏好中文回答，正式记忆需要人工审核。"}\n',
            encoding="utf-8",
        )

    sample_reviewed = examples_dir / "reviewed.example.jsonl"
    if force or not sample_reviewed.exists():
        sample_reviewed.write_text(
            '{"candidate_id":"mem_example","action":"approved","edited_content":"用户偏好中文回答。","reviewer":"user","candidate":{"id":"mem_example","type":"preference","scope":"user","content":"用户偏好中文回答。","evidence":[{"event_id":"event_1"}]}}\n',
            encoding="utf-8",
        )

    return {
        "root": str(root),
        "memory_dir": str(memory_dir),
        "config_path": str(config_path),
        "imports_dir": str(imports_dir),
        "runs_dir": str(runs_dir),
        "examples": [str(sample_events), str(sample_reviewed)],
    }


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--codex-home", default=str(Path.home() / ".codex"))
    parser.add_argument("--claude-home", default=str(Path.home() / ".claude"))
    parser.add_argument("--claude-state", default=str(Path.home() / ".claude.json"))
    parser.add_argument("--project", action="append", default=[])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dream-memory", description="Scan and import Claude Code / Codex sessions into shared memory events.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Memory config JSON path")
    _add_source_args(parser)
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Initialize a .dream-memory workspace")
    init.add_argument("--path", default=".")
    init.add_argument("--force", action="store_true")

    init_config = sub.add_parser("init-config", help="Write a default editable memory config file")
    init_config.add_argument("--output", default=str(DEFAULT_CONFIG_PATH))

    check_provider = sub.add_parser("check-provider", help="Check provider config, API key env, and optionally invoke the model")
    check_provider.add_argument("--provider")
    check_provider.add_argument("--model")
    check_provider.add_argument("--api-key-env")
    check_provider.add_argument("--base-url")
    check_provider.add_argument("--timeout-seconds", type=int)
    check_provider.add_argument("--invoke", action="store_true")
    check_provider.add_argument("--all", action="store_true", help="Check all configured model profiles")
    check_provider.add_argument("--profile", help="Check one configured model profile")

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

    summary = sub.add_parser("summary", help="Render all-projects memory summary")
    summary.add_argument("--scope", choices=["all-projects"], default="all-projects")
    summary.add_argument("--memory-cards")
    summary.add_argument("--output")

    export = sub.add_parser("export", help="Export approved memory into AGENTS.md and/or CLAUDE.md")
    export.add_argument("--target", choices=["codex", "claude", "both"], default="both")
    export.add_argument("--scope", choices=["project", "global"], default="project")
    export.add_argument("--project")
    export.add_argument("--memory-cards")
    export.add_argument("--output-dir")
    export.add_argument("--limit", type=int)

    eval_cmd = sub.add_parser("eval", help="Evaluate extraction quality with labeled JSONL")
    eval_cmd.add_argument("--input", required=True)
    eval_cmd.add_argument("--project")
    eval_cmd.add_argument("--mode", choices=["rules", "ai"], default="rules")
    eval_cmd.add_argument("--output")

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
    profiles, policy = runtime_parts_from_config(config)
    default_profile = profiles[policy.default_profile].config
    provider = str(_value(getattr(args, "provider", None), default_profile.provider))
    model = str(_value(getattr(args, "model", None), default_profile.model))
    if provider and ":" not in model:
        return f"{provider}:{model}"
    return model


def _runtime_config_from_args(args: argparse.Namespace, config: dict[str, object]) -> dict[str, object]:
    runtime_config = {
        "models": config.get("models"),
        "model_policy": config.get("model_policy"),
    }
    provider_override = getattr(args, "provider", None)
    model_override = getattr(args, "model", None)
    api_key_override = getattr(args, "api_key_env", None)
    base_url_override = getattr(args, "base_url", None)
    timeout_override = getattr(args, "timeout_seconds", None)
    if any(value is not None for value in (provider_override, model_override, api_key_override, base_url_override, timeout_override)):
        profiles, policy = runtime_parts_from_config(config)
        default_profile = profiles[policy.default_profile].config
        provider = str(_value(provider_override, default_profile.provider))
        model = str(_value(model_override, default_profile.model))
        if ":" in model and provider == default_profile.provider:
            parsed_provider, parsed_model = model.split(":", 1)
            provider, model = parsed_provider, parsed_model
        runtime_config = {
            "models": {
                "override": {
                    "provider": provider,
                    "model": model,
                    "api_key_env": _value(api_key_override, default_profile.api_key_env),
                    "base_url": _value(base_url_override, default_profile.base_url),
                    "timeout_seconds": int(_value(timeout_override, default_profile.timeout_seconds)),
                }
            },
            "model_policy": {
                "default_profile": "override",
                "fallback_chain": ["override"],
                "retry": dict(config.get("model_policy", {}).get("retry", {})) if isinstance(config.get("model_policy"), dict) else {},
                "allow_rules_fallback": bool(config.get("model_policy", {}).get("allow_rules_fallback", False)) if isinstance(config.get("model_policy"), dict) else False,
            },
        }
    return runtime_config


def _model_trace_callback(state: dict[str, object] | None):
    if not state:
        return None

    def callback(event_type: str, payload: dict[str, object]) -> None:
        append_trace(state, event_type, payload)

    return callback


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
    existing_state: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object] | None]:
    events = load_events_jsonl(Path(args.input))
    output_dir = _configured_output_dir(args, config)
    mode = str(_value(args.mode, config["mode"]))
    model = _configured_model(args, config)
    invoke_model = bool(_value(args.invoke_model, config["invoke_model"]))
    runtime_config = _runtime_config_from_args(args, config)
    _apply_provider_env(args, config)
    state: dict[str, object] | None = None
    if persistent:
        state = existing_state or create_run_state(memory_dir=output_dir, project=args.project, input_path=str(args.input), mode=mode, model=model, invoke_model=invoke_model)
        events_path = copy_input_events(args.input, state)
        state = update_run_state(state, status="running", phase="extracting", artifacts={"events_path": str(events_path)})
        append_trace(state, "events_copied", {"events_path": str(events_path), "event_count": len(events)})
        working_dir = Path(str(state["run_dir"]))
    else:
        working_dir = output_dir
    if mode == "ai":
        extraction = agent_extract_memory_candidates(
            events,
            project=args.project,
            model=model,
            invoke_model=invoke_model,
            runtime_config=runtime_config,
            trace_callback=_model_trace_callback(state),
        )
        working_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = working_dir / "ai-prompt.md"
        prompt_path.write_text(str(extraction["prompt"]), encoding="utf-8")
        artifacts = {"ai_prompt_path": str(prompt_path)}
        if "raw_response" in extraction:
            raw_path = working_dir / "ai-raw-response.txt"
            raw_path.write_text(str(extraction["raw_response"]), encoding="utf-8")
            artifacts["ai_raw_response_path"] = str(raw_path)
        if state:
            state = update_run_state(state, phase="candidate_validation", artifacts=artifacts)
            append_trace(state, "ai_extraction_complete", {"dry_run": extraction["dry_run"], "candidate_count": len(extraction.get("candidates", []))})
        result = dream_from_events(events, project=args.project, output_dir=working_dir, apply=False, agent_candidates=list(extraction.get("candidates", [])), agent_mode=True)
        payload = {**result.to_dict(), "mode": "ai", "ai_dry_run": extraction["dry_run"], "ai_prompt_path": str(prompt_path)}
        if "model_runtime" in extraction:
            payload["model_runtime"] = extraction["model_runtime"]
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
            next_actions=["review candidates", f"dream-memory resume --run-id {state['run_id']}"],
        )
        write_candidate_traces(state, candidates)
        append_trace(state, "waiting_review", {"review_queue_path": str(queue_path), "review_count": len(queue)})
        payload = {**payload, "run_id": state["run_id"], "run_dir": state["run_dir"], "state_path": str(Path(str(state["run_dir"])) / "state.json")}
    return payload, state


def _memory_cards_path(args: argparse.Namespace, config: dict[str, object]) -> str:
    return str(_value(getattr(args, "memory_cards", None), config["memory_cards"]))


def _export_memory(*, args: argparse.Namespace, config: dict[str, object]) -> dict[str, object]:
    cards = load_events_jsonl(Path(_memory_cards_path(args, config)))
    output_dir = Path(str(_value(args.output_dir, args.project or "."))).expanduser()
    if args.scope == "project":
        project = normalize_project_path(args.project or str(output_dir))
        context = build_agent_context(cards, project=project, limit=int(_value(args.limit, config["context_limit"])))
    else:
        non_project_cards = [card for card in cards if card.get("scope") in {"user", "global"}]
        context = build_agent_context(non_project_cards, project=None, limit=int(_value(args.limit, config["context_limit"])))
    markdown = render_context_markdown(context)
    written: list[str] = []
    if args.target in {"codex", "both"}:
        target = output_dir / "AGENTS.md" if args.scope == "project" else Path.home() / ".codex" / "AGENTS.md"
        written.append(str(write_marked_file(target, markdown, heading="## Dream Memory Context")))
    if args.target in {"claude", "both"}:
        target = output_dir / "CLAUDE.md" if args.scope == "project" else Path.home() / ".claude" / "CLAUDE.md"
        written.append(str(write_marked_file(target, markdown, heading="## Dream Memory Context")))
    return {"target": args.target, "scope": args.scope, "project": context.get("project"), "written": written, "count": context.get("count")}


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
    if bool(config.get("auto_export")):
        export_args = argparse.Namespace(
            target=str(config.get("export_target") or "both"),
            scope=str(config.get("export_scope") or "project"),
            project=state.get("project"),
            memory_cards=str(cards_path),
            output_dir=config.get("export_output_dir") or state.get("project") or ".",
            limit=None,
        )
        auto_export_payload = _export_memory(args=export_args, config=config)
        state = update_run_state(
            state,
            artifacts={"auto_export_files": auto_export_payload.get("written", [])},
        )
        append_trace(state, "auto_export_complete", auto_export_payload)
    return state


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_memory_config(args.config)

    if args.command == "init":
        payload = _init_workspace(args.path, force=bool(args.force))
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "init-config":
        path = write_default_memory_config(args.output)
        print(json.dumps({"config_path": str(path)}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "check-provider":
        if args.all or args.profile:
            profiles, _ = runtime_parts_from_config(config)
            selected = [args.profile] if args.profile else list(profiles)
            diagnostics: dict[str, object] = {}
            ok = True
            for profile_name in selected:
                profile = profiles.get(profile_name)
                if profile is None:
                    diagnostics[profile_name] = {"ok": False, "error": f"unknown profile: {profile_name}"}
                    ok = False
                    continue
                item = provider_diagnostics(
                    provider=profile.config.provider,
                    model=profile.config.model,
                    api_key_env=profile.config.api_key_env,
                    base_url=profile.config.base_url,
                    timeout_seconds=profile.config.timeout_seconds,
                    invoke=bool(args.invoke),
                )
                diagnostics[profile_name] = item
                ok = ok and bool(item.get("ok"))
            print(json.dumps({"profiles": diagnostics, "ok": ok}, ensure_ascii=False, indent=2))
            return 0 if ok else 1
        profiles, policy = runtime_parts_from_config(_runtime_config_from_args(args, config))
        profile = profiles[policy.default_profile].config
        provider = profile.provider
        model = profile.model
        api_key_env = profile.api_key_env
        base_url = profile.base_url
        timeout_seconds = profile.timeout_seconds
        payload = provider_diagnostics(provider=provider, model=model, api_key_env=api_key_env, base_url=str(base_url) if base_url else None, timeout_seconds=timeout_seconds, invoke=bool(args.invoke))
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if payload.get("ok") else 1

    if args.command == "eval":
        payload = evaluate_labeled_events(args.input, project=args.project, mode=args.mode)
        if args.output:
            output = Path(args.output).expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"output": str(output), **{k: payload[k] for k in ("precision", "recall", "f1")}}, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
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

    if args.command == "summary":
        cards = load_events_jsonl(Path(_memory_cards_path(args, config)))
        markdown = render_all_projects_summary(cards)
        if args.output:
            output = Path(args.output).expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(markdown, encoding="utf-8")
            print(json.dumps({"output": str(output), "scope": args.scope}, ensure_ascii=False, indent=2))
        else:
            print(markdown, end="")
        return 0

    if args.command == "export":
        payload = _export_memory(args=args, config=config)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "dream":
        events = load_events_jsonl(Path(args.input))
        output_dir = _configured_output_dir(args, config)
        mode = "ai" if args.agent else str(_value(args.mode, config["mode"]))
        model = _configured_model(args, config)
        invoke_model = bool(_value(args.invoke_model, config["invoke_model"]))
        runtime_config = _runtime_config_from_args(args, config)
        _apply_provider_env(args, config)
        if mode == "ai":
            extraction = agent_extract_memory_candidates(events, project=args.project, model=model, invoke_model=invoke_model, runtime_config=runtime_config)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "ai-prompt.md").write_text(str(extraction["prompt"]), encoding="utf-8")
            if "raw_response" in extraction:
                (output_dir / "ai-raw-response.txt").write_text(str(extraction["raw_response"]), encoding="utf-8")
            result = dream_from_events(
                events,
                project=args.project,
                output_dir=output_dir,
                apply=bool(args.apply),
                agent_candidates=list(extraction.get("candidates", [])),
                agent_mode=True,
            )
            payload = {**result.to_dict(), "mode": "ai", "ai": True, "ai_dry_run": extraction["dry_run"], "ai_prompt_path": str(output_dir / "ai-prompt.md")}
            if "model_runtime" in extraction:
                payload["model_runtime"] = extraction["model_runtime"]
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
