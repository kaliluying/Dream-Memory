from __future__ import annotations

import argparse
import json
import sys
from importlib import resources
from datetime import datetime, timezone
from pathlib import Path

from .memory_agent import agent_extract_memory_candidates
from .memory_config import DEFAULT_CONFIG_PATH, DEFAULT_MEMORY_CONFIG, load_memory_config, write_default_memory_config
from .memory_export import render_all_projects_summary, write_marked_file
from .memory_eval import evaluate_labeled_events
from .memory_dreaming import (
    apply_reviewed_memory,
    build_agent_context,
    build_review_queue,
    dream_from_events,
    render_context_markdown,
    render_review_queue_memory_preview,
    normalize_project_path,
    extract_atomic_facts,
    load_events_jsonl,
    write_jsonl_records,
)
from .memory_importers import ClaudeCodeImporter, CodexImporter, NormalizedSessionEvent, import_project_instruction_events, import_project_marker_events, write_events_jsonl
from .model_providers import SUPPORTED_MODEL_PROVIDERS, provider_diagnostics, runtime_parts_from_config
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
    roots = [Path(value).expanduser() for value in values or []]
    return roots or [Path.cwd()]


def _packaged_example_text(name: str, fallback: str) -> str:
    try:
        return (resources.files("dream_memory") / "examples" / name).read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError):
        pass
    repository_sample = Path(__file__).resolve().parents[2] / "examples" / name
    if repository_sample.exists():
        return repository_sample.read_text(encoding="utf-8")
    return fallback


def _sample_labeled_events_text() -> str:
    fallback = {
        "id": "preference_language",
        "event": {"event_id": "event_1", "source": "codex", "role": "user", "event_type": "history_prompt", "content": "用户偏好中文回答。"},
        "expected": [{"content": "用户偏好中文回答", "type": "preference", "scope": "user"}],
    }
    return _packaged_example_text("labeled-events.jsonl", json.dumps(fallback, ensure_ascii=False, separators=(",", ":")) + "\n")


def _init_workspace(path: Path | str, *, force: bool = False, output_dir: Path | str | None = None) -> dict[str, object]:
    if output_dir is not None:
        memory_dir = Path(output_dir).expanduser()
        root = memory_dir
    else:
        root = Path(path).expanduser()
        memory_dir = root / ".dream-memory"
    imports_dir = memory_dir / "imports"
    runs_dir = memory_dir / "runs"
    examples_dir = root / "examples"
    memory_dir.mkdir(parents=True, exist_ok=True)
    imports_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / ".gitkeep").touch()
    memory_cards_path = memory_dir / "memory_cards.jsonl"
    if force or not memory_cards_path.exists():
        memory_cards_path.touch()

    config_path = memory_dir / "config.json"
    if force or not config_path.exists():
        write_default_memory_config(config_path)
    if output_dir is not None:
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        config_payload.update({
            "default_input": str(imports_dir / "all-events.jsonl"),
            "init_path": str(root),
            "init_config_output": str(config_path),
            "imports_output_dir": str(imports_dir),
            "extract_input": str(imports_dir / "all-events.jsonl"),
            "extract_output_dir": str(memory_dir),
            "review_candidates": str(memory_dir / "ai-candidates.jsonl"),
            "apply_reviewed": str(memory_dir / "reviewed.jsonl"),
            "eval_input": str(examples_dir / "labeled-events.jsonl"),
            "eval_project": "/tmp/project",
            "eval_output": str(memory_dir / "eval.json"),
            "output_dir": str(memory_dir),
            "memory_cards": str(memory_cards_path),
        })
        config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    sample_events = examples_dir / "sample-events.jsonl"
    if force or not sample_events.exists():
        examples_dir.mkdir(parents=True, exist_ok=True)
        sample_events.write_text(
            _packaged_example_text("sample-events.jsonl", '{"event_id":"event_1","source":"codex","session_id":"sample","role":"user","event_type":"history_prompt","project":".","content":"用户偏好中文回答，正式记忆需要人工审核。"}\n'),
            encoding="utf-8",
        )

    sample_reviewed = examples_dir / "reviewed.example.jsonl"
    if force or not sample_reviewed.exists():
        sample_reviewed.write_text(
            _packaged_example_text("reviewed.example.jsonl", '{"candidate_id":"mem_example","action":"approved","edited_content":"用户偏好中文回答。","reviewer":"user","candidate":{"id":"mem_example","type":"preference","scope":"user","content":"用户偏好中文回答。","evidence":[{"event_id":"event_1"}]}}\n'),
            encoding="utf-8",
        )

    sample_labeled = examples_dir / "labeled-events.jsonl"
    if force or not sample_labeled.exists():
        sample_labeled.write_text(_sample_labeled_events_text(), encoding="utf-8")

    return {
        "root": str(root),
        "memory_dir": str(memory_dir),
        "config_path": str(config_path),
        "imports_dir": str(imports_dir),
        "runs_dir": str(runs_dir),
        "memory_cards_path": str(memory_cards_path),
        "examples": [str(sample_events), str(sample_reviewed), str(sample_labeled)],
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
    init.add_argument("--output-dir", help="Initialize this memory directory directly instead of PATH/.dream-memory")
    init.add_argument("--force", action="store_true")

    init_config = sub.add_parser("init-config", help="Write a default editable memory config file")
    init_config.add_argument("--output", default=str(DEFAULT_CONFIG_PATH))

    check_provider = sub.add_parser("check-provider", help="Check provider config, API key, and optionally invoke the model")
    check_provider.add_argument("--provider")
    check_provider.add_argument("--model")
    check_provider.add_argument("--api-key")
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
    dream.add_argument("--input", help="Input normalized events JSONL; defaults to config default_input")
    dream.add_argument("--project")
    dream.add_argument("--output-dir")
    dream.add_argument("--apply", action="store_true", help="Append promoted preview to MEMORY.md")
    dream.add_argument("--mode", choices=["ai", "rules"], help="Extraction mode: ai is default; rules is fallback/debug")
    dream.add_argument("--agent", action="store_true", help="Deprecated alias for --mode ai")
    dream.add_argument("--provider")
    dream.add_argument("--model")
    dream.add_argument("--api-key")
    dream.add_argument("--api-key-env")
    dream.add_argument("--base-url")
    dream.add_argument("--timeout-seconds", type=int)
    dream.set_defaults(invoke_model=None)
    dream.add_argument("--dry-run", action="store_false", dest="invoke_model", help="Write the AI prompt only; do not invoke the model")
    dream.add_argument("--invoke-model", action="store_true", dest="invoke_model", help="Invoke the model; this is the default")

    extract = sub.add_parser("extract-facts", help="Extract atomic facts from normalized events")
    extract.add_argument("--input")
    extract.add_argument("--project")
    extract.add_argument("--output-dir")

    review = sub.add_parser("review", help="Build review queue items from candidates")
    review.add_argument("--candidates", required=True)
    review.add_argument("--memory-cards")
    review.add_argument("--output-dir")

    review_summary = sub.add_parser("review-summary", help="Summarize a review queue by action, type, quality, and score")
    review_summary.add_argument("--run-id", help="Run ID whose review_queue.jsonl should be summarized")
    review_summary.add_argument("--review-queue", help="Explicit review_queue.jsonl path")
    review_summary.add_argument("--output-dir")

    auto_review = sub.add_parser("auto-review", help="Write reviewed decisions for high-confidence run candidates")
    auto_review.add_argument("--run-id", required=True)
    auto_review.add_argument("--output-dir")
    auto_review.add_argument("--reviewer", default="auto-review")
    auto_review.add_argument("--min-score", type=float, default=0.7, help="Minimum dream_score for auto-approval")
    auto_review.add_argument("--review-queue", help="Override review_queue.jsonl path")
    auto_review.add_argument("--reviewed-output", help="Override reviewed.jsonl output path")
    auto_review.add_argument("--keep-review", action="store_true", help="Leave review/needs_more_evidence items undecided instead of writing decisions")
    auto_review.add_argument("--include-duplicates", action="store_true", help="Write rejected decisions for duplicate candidates instead of skipping them")
    auto_review.add_argument("--include-merges", action="store_true", help="Write merged decisions for similar-memory candidates instead of skipping them")
    auto_review.add_argument("--include-review", action="store_true", help="Also approve 'review' candidates with score >= min-score, not just 'create' ones")
    auto_review.add_argument("--force", action="store_true", help="Overwrite an existing reviewed.jsonl output")
    auto_review.add_argument("--dry-run", action="store_true", help="Preview auto-review decisions and skip reasons without writing reviewed.jsonl or mutating run state")

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
    context.add_argument("--task", help="Task text used to rank relevant memory")

    pipeline = sub.add_parser("pipeline", help="Run dream and review in one step")
    pipeline.add_argument("--input")
    pipeline.add_argument("--project")
    pipeline.add_argument("--output-dir")
    pipeline.add_argument("--memory-cards")
    pipeline.add_argument("--mode", choices=["ai", "rules"])
    pipeline.add_argument("--provider")
    pipeline.add_argument("--model")
    pipeline.add_argument("--api-key")
    pipeline.add_argument("--api-key-env")
    pipeline.add_argument("--base-url")
    pipeline.add_argument("--timeout-seconds", type=int)
    pipeline.set_defaults(invoke_model=None)
    pipeline.add_argument("--dry-run", action="store_false", dest="invoke_model", help="Write the AI prompt only; do not invoke the model")
    pipeline.add_argument("--invoke-model", action="store_true", dest="invoke_model", help="Invoke the model; this is the default")

    run = sub.add_parser("run", help="Create a persistent resumable Dream Memory run")
    run.add_argument("--input")
    run.add_argument("--project")
    run.add_argument("--output-dir")
    run.add_argument("--memory-cards")
    run.add_argument("--mode", choices=["ai", "rules"])
    run.add_argument("--provider")
    run.add_argument("--model")
    run.add_argument("--api-key")
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
    eval_cmd.add_argument("--input")
    eval_cmd.add_argument("--project")
    eval_cmd.add_argument("--mode", choices=["rules", "ai"])
    eval_cmd.add_argument("--output")
    eval_cmd.add_argument("--provider")
    eval_cmd.add_argument("--model")
    eval_cmd.add_argument("--api-key")
    eval_cmd.add_argument("--api-key-env")
    eval_cmd.add_argument("--base-url")
    eval_cmd.add_argument("--timeout-seconds", type=int)
    eval_cmd.add_argument("--max-rows", type=int, help="Evaluate only the first N rows")
    eval_cmd.add_argument("--max-attempts", type=int, help="Override model retry attempts for eval")
    eval_cmd.add_argument("--continue-on-error", action="store_true", help="Keep evaluating rows after model/provider errors")
    eval_cmd.add_argument("--fallback-rules-on-error", action="store_true", help="Use rules extraction for rows where AI/model extraction fails")
    eval_cmd.add_argument("--fallback-rules-on-empty", action="store_true", help="Use rules extraction when AI succeeds but returns no candidates")

    sync_cmd = sub.add_parser("sync", help="Import, dream, and optionally auto-apply memory in one step")
    sync_cmd.add_argument("--project", default=".")
    sync_cmd.add_argument("--auto", action="store_true", help="Auto-review and apply high-confidence candidates without manual review")
    sync_cmd.add_argument("--min-score", type=float, default=0.5, dest="min_score", help="Minimum dream_score for auto-approval (default: 0.5)")
    sync_cmd.add_argument("--output-dir")
    sync_cmd.add_argument("--memory-cards")
    sync_cmd.add_argument("--mode", choices=["ai", "rules"])
    sync_cmd.add_argument("--provider")
    sync_cmd.add_argument("--model")
    sync_cmd.add_argument("--api-key")
    sync_cmd.add_argument("--api-key-env")
    sync_cmd.add_argument("--base-url")
    sync_cmd.add_argument("--timeout-seconds", type=int)
    sync_cmd.set_defaults(invoke_model=True)
    sync_cmd.add_argument("--dry-run", action="store_false", dest="invoke_model", help="Write AI prompt only; do not invoke the model")

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
    api_key_override = getattr(args, "api_key", None)
    api_key_env_override = getattr(args, "api_key_env", None)
    base_url_override = getattr(args, "base_url", None)
    timeout_override = getattr(args, "timeout_seconds", None)
    if any(value is not None for value in (provider_override, model_override, api_key_override, api_key_env_override, base_url_override, timeout_override)):
        profiles, policy = runtime_parts_from_config(config)
        default_profile = profiles[policy.default_profile].config
        provider = str(_value(provider_override, default_profile.provider))
        model = str(_value(model_override, default_profile.model))
        if ":" in model and provider == default_profile.provider:
            parsed_provider, parsed_model = model.split(":", 1)
            if parsed_provider in SUPPORTED_MODEL_PROVIDERS:
                provider, model = parsed_provider, parsed_model
        runtime_config = {
            "models": {
                "override": {
                    "provider": provider,
                    "model": model,
                    "api_key": _value(api_key_override, default_profile.api_key),
                    "api_key_env": _value(api_key_env_override, default_profile.api_key_env),
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


def _run_progress(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[dream-memory] {message}", file=sys.stderr, flush=True)


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
    progress = bool(persistent)
    input_path = str(_value(getattr(args, "input", None), config.get("default_input")))
    if not input_path or input_path == "None":
        raise ValueError("run/pipeline requires --input or config default_input")
    if getattr(args, "input", None) is None:
        args.input = input_path
    if getattr(args, "project", None) is None and config.get("default_project") is not None:
        args.project = str(config.get("default_project"))
    _run_progress(progress, f"loading events from {input_path}")
    events = load_events_jsonl(Path(input_path))
    output_dir = _configured_output_dir(args, config)
    mode = str(_value(args.mode, config["mode"]))
    model = _configured_model(args, config)
    invoke_model = bool(_value(args.invoke_model, config["invoke_model"]))
    runtime_config = _runtime_config_from_args(args, config)
    _apply_provider_env(args, config)
    _run_progress(progress, f"loaded {len(events)} events; mode={mode}; invoke_model={str(invoke_model).lower()}")
    state: dict[str, object] | None = None
    if persistent:
        state = existing_state or create_run_state(memory_dir=output_dir, project=args.project, input_path=str(args.input), mode=mode, model=model, invoke_model=invoke_model)
        _run_progress(progress, f"created run {state['run_id']} in {state['run_dir']}")
        events_path = copy_input_events(args.input, state)
        state = update_run_state(state, status="running", phase="extracting", artifacts={"events_path": str(events_path)})
        append_trace(state, "events_copied", {"events_path": str(events_path), "event_count": len(events)})
        _run_progress(progress, f"copied input events to {events_path}")
        working_dir = Path(str(state["run_dir"]))
    else:
        working_dir = output_dir
    try:
        if mode == "ai":
            _run_progress(progress, f"extracting candidates with model {model}")
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
            prompt_count_payload = {
                key: extraction[key]
                for key in ("input_event_count", "prompt_event_count", "filtered_prompt_event_count")
                if key in extraction
            }
            artifacts = {"ai_prompt_path": str(prompt_path), **prompt_count_payload}
            if "raw_response" in extraction:
                raw_path = working_dir / "ai-raw-response.txt"
                raw_path.write_text(str(extraction["raw_response"]), encoding="utf-8")
                artifacts["ai_raw_response_path"] = str(raw_path)
            if state:
                state = update_run_state(state, phase="candidate_validation", artifacts=artifacts)
                state_counts = dict(state.get("counts", {}))
                state_counts.update(prompt_count_payload)
                state = update_run_state(state, counts=state_counts)
                append_trace(state, "ai_extraction_complete", {"dry_run": extraction["dry_run"], "candidate_count": len(extraction.get("candidates", [])), **prompt_count_payload})
                _run_progress(progress, f"model extraction complete; candidates={len(extraction.get('candidates', []))}; dry_run={str(extraction['dry_run']).lower()}")
            result = dream_from_events(events, project=args.project, output_dir=working_dir, apply=False, agent_candidates=list(extraction.get("candidates", [])), agent_mode=True)
            payload = {**result.to_dict(), "mode": "ai", "ai_dry_run": extraction["dry_run"], "ai_prompt_path": str(prompt_path), **prompt_count_payload}
            if "model_runtime" in extraction:
                payload["model_runtime"] = extraction["model_runtime"]
        else:
            _run_progress(progress, "extracting candidates with rules")
            result = dream_from_events(events, project=args.project, output_dir=working_dir, apply=False)
            payload = {**result.to_dict(), "mode": "rules"}
            if state:
                append_trace(state, "rules_extraction_complete", {"candidate_count": result.candidate_count})
                _run_progress(progress, f"rules extraction complete; candidates={result.candidate_count}")
        candidates = load_events_jsonl(Path(result.candidates_path))
        memory_cards = _load_optional_jsonl(_memory_cards_path(args, config))
        _run_progress(progress, f"building review queue from {len(candidates)} candidates")
        queue = build_review_queue(candidates, memory_cards)
        queue_path = write_jsonl_records(queue, working_dir / "review_queue.jsonl")
        preview_path = Path(result.memory_preview_path)
        preview_path.write_text(render_review_queue_memory_preview(queue), encoding="utf-8")
        payload = {**payload, "review_queue_path": str(queue_path), "review_count": len(queue)}
    except Exception as exc:
        if state:
            failed_state = update_run_state(
                state,
                status="failed",
                phase="failed",
                error=f"{exc.__class__.__name__}: {exc}",
                next_actions=["inspect trace", "fix provider/config and rerun"],
            )
            append_trace(failed_state, "run_failed", {"error_type": exc.__class__.__name__, "error": str(exc)})
            _run_progress(progress, f"run failed: {exc.__class__.__name__}: {exc}")
        raise
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
        _run_progress(progress, f"waiting for review; review_count={len(queue)}; run_id={state['run_id']}")
        _run_progress(progress, f"next: dream-memory resume --run-id {state['run_id']}")
    return payload, state


def _memory_cards_path(args: argparse.Namespace, config: dict[str, object]) -> str:
    explicit = getattr(args, "memory_cards", None)
    if explicit:
        return str(explicit)
    configured = str(config["memory_cards"])
    output_dir = getattr(args, "output_dir", None)
    if output_dir and configured == str(DEFAULT_MEMORY_CONFIG["memory_cards"]):
        return str(Path(str(output_dir)).expanduser() / "memory_cards.jsonl")
    return configured


def _export_memory(*, args: argparse.Namespace, config: dict[str, object]) -> dict[str, object]:
    cards = _load_optional_jsonl(_memory_cards_path(args, config))
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


def _summarize_review_queue(queue: list[dict[str, object]]) -> dict[str, object]:
    def bump(bucket: dict[str, int], key: object) -> None:
        name = str(key or "unknown")
        bucket[name] = bucket.get(name, 0) + 1

    by_status: dict[str, int] = {}
    by_suggested_action: dict[str, int] = {}
    by_type: dict[str, int] = {}
    by_scope: dict[str, int] = {}
    by_evidence_quality: dict[str, int] = {}
    duplicate_count = 0
    conflict_count = 0
    low_score_count = 0
    needs_manual_count = 0
    scores: list[float] = []
    for item in queue:
        if not isinstance(item, dict):
            continue
        candidate = item.get("candidate") if isinstance(item.get("candidate"), dict) else {}
        analysis = item.get("dream_analysis") if isinstance(item.get("dream_analysis"), dict) else {}
        quality = item.get("quality_signals") if isinstance(item.get("quality_signals"), dict) else {}
        action = str(item.get("suggested_action") or analysis.get("suggested_action") or "unknown")
        bump(by_status, item.get("status") or candidate.get("status"))
        bump(by_suggested_action, action)
        bump(by_type, candidate.get("type"))
        bump(by_scope, candidate.get("scope"))
        bump(by_evidence_quality, quality.get("evidence_quality"))
        duplicate_count += 1 if quality.get("duplicate") else 0
        conflict_count += len(item.get("conflicts") or []) if isinstance(item.get("conflicts"), list) else 0
        needs_manual_count += 1 if action in {"review", "needs_more_evidence"} else 0
        try:
            score = max(0.0, min(1.0, float(analysis.get("dream_score", 0.0) or 0.0)))
        except (TypeError, ValueError):
            score = 0.0
        scores.append(score)
        if action in {"create", "merge"} and score < 0.7:
            low_score_count += 1
    return {
        "total": len([item for item in queue if isinstance(item, dict)]),
        "by_status": dict(sorted(by_status.items())),
        "by_suggested_action": dict(sorted(by_suggested_action.items())),
        "by_type": dict(sorted(by_type.items())),
        "by_scope": dict(sorted(by_scope.items())),
        "by_evidence_quality": dict(sorted(by_evidence_quality.items())),
        "duplicate_count": duplicate_count,
        "conflict_count": conflict_count,
        "low_score_count": low_score_count,
        "needs_manual_count": needs_manual_count,
        "score_min": round(min(scores), 4) if scores else None,
        "score_max": round(max(scores), 4) if scores else None,
        "score_avg": round(sum(scores) / len(scores), 4) if scores else None,
    }


def _dream_score(item: dict[str, object]) -> float:
    analysis = item.get("dream_analysis") if isinstance(item.get("dream_analysis"), dict) else {}
    try:
        return max(0.0, min(1.0, float(analysis.get("dream_score", 0.0) or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _auto_review_run(*, args: argparse.Namespace, config: dict[str, object]) -> dict[str, object]:
    output_dir = _configured_output_dir(args, config)
    state = load_run_state(output_dir, args.run_id)
    artifacts = state.get("artifacts", {}) if isinstance(state.get("artifacts"), dict) else {}
    queue_path = Path(str(args.review_queue or artifacts.get("review_queue_path") or "")).expanduser()
    if not queue_path.exists():
        raise FileNotFoundError(f"review queue not found for run {args.run_id}: {queue_path}")
    queue = load_events_jsonl(queue_path)
    reviewed_path = Path(str(args.reviewed_output or Path(str(state["run_dir"])) / "reviewed.jsonl")).expanduser()
    dry_run = bool(getattr(args, "dry_run", False))
    if reviewed_path.exists() and not dry_run and not bool(getattr(args, "force", False)):
        raise FileExistsError(f"reviewed output already exists; pass --force to overwrite: {reviewed_path}")
    decisions: list[dict[str, object]] = []
    skipped = 0
    approved = 0
    rejected = 0
    needs_more_evidence = 0
    duplicate_skipped = 0
    merge_skipped = 0
    skip_reasons: dict[str, int] = {}

    def skip(reason: str) -> None:
        nonlocal skipped
        skipped += 1
        skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    for item in queue:
        if not isinstance(item, dict):
            skip("malformed_item")
            continue
        candidate = item.get("candidate") if isinstance(item.get("candidate"), dict) else {}
        if not candidate:
            skip("missing_candidate")
            continue
        quality_signals = item.get("quality_signals") if isinstance(item.get("quality_signals"), dict) else {}
        if quality_signals.get("duplicate") and not bool(getattr(args, "include_duplicates", False)):
            skip("duplicate")
            duplicate_skipped += 1
            continue
        analysis = item.get("dream_analysis") if isinstance(item.get("dream_analysis"), dict) else {}
        suggested_action = str(item.get("suggested_action") or analysis.get("suggested_action") or "review")
        score = _dream_score(item)
        review_action: str | None = None
        if suggested_action == "create" and score >= float(args.min_score):
            review_action = "approved"
            approved += 1
        elif suggested_action == "merge" and score >= float(args.min_score):
            if bool(getattr(args, "include_merges", False)):
                review_action = "merged"
                approved += 1
            else:
                skip("merge_requires_explicit_include")
                merge_skipped += 1
                continue
        elif suggested_action == "reject":
            review_action = "rejected"
            rejected += 1
        elif not args.keep_review and suggested_action == "needs_more_evidence":
            review_action = "needs_more_evidence"
            needs_more_evidence += 1
        elif suggested_action in {"create", "merge"} and score < float(args.min_score):
            skip("below_min_score")
        elif suggested_action == "review" and bool(getattr(args, "include_review", False)) and score >= float(args.min_score):
            review_action = "approved"
            approved += 1
        elif suggested_action in {"review", "needs_more_evidence"}:
            skip("requires_manual_review")
        else:
            skip("unhandled_action")
        if review_action is None:
            continue
        decisions.append({
            "candidate_id": candidate.get("id") or item.get("candidate_id"),
            "action": review_action,
            "edited_content": candidate.get("content"),
            "reviewer": str(args.reviewer or "auto-review"),
            "candidate": candidate,
            "review_note": f"Auto-review from suggested_action={suggested_action}, dream_score={score:.3f}.",
        })
    payload = {
        "run_id": state["run_id"],
        "reviewed_path": str(reviewed_path),
        "decision_count": len(decisions),
        "approved": approved,
        "rejected": rejected,
        "needs_more_evidence": needs_more_evidence,
        "skipped": skipped,
        "duplicate_skipped": duplicate_skipped,
        "merge_skipped": merge_skipped,
        "skip_reasons": dict(sorted(skip_reasons.items())),
        "min_score": float(args.min_score),
        "dry_run": dry_run,
    }
    if dry_run:
        return payload
    reviewed_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_records(decisions, reviewed_path)
    state = update_run_state(
        state,
        artifacts={"reviewed_path": str(reviewed_path)},
        counts={"auto_review_count": len(decisions)},
        next_actions=[f"dream-memory resume --run-id {state['run_id']}", "inspect reviewed decisions"],
    )
    append_trace(state, "auto_reviewed", payload)
    return payload


def _resume_run(*, args: argparse.Namespace, config: dict[str, object]) -> dict[str, object]:
    output_dir = _configured_output_dir(args, config)
    state = load_run_state(output_dir, args.run_id)
    reviewed_path = Path(args.reviewed).expanduser() if args.reviewed else Path(str(state["run_dir"])) / "reviewed.jsonl"
    reviewed = load_events_jsonl(reviewed_path) if reviewed_path.exists() else []
    existing_cards = _load_optional_jsonl(_memory_cards_path(args, config))
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


def _handle_sync(*, args: argparse.Namespace, config: dict[str, object]) -> dict[str, object]:
    """Import events, dream candidates, and optionally auto-apply memory in one step."""
    output_dir = _configured_output_dir(args, config)
    imports_dir = Path(str(config["imports_output_dir"])).expanduser()
    project_path = str(getattr(args, "project", None) or ".").strip() or "."
    auto_mode = bool(getattr(args, "auto", False))
    min_score = float(getattr(args, "min_score", 0.5))

    # Step 1: import events
    _run_progress(True, f"importing events (project={project_path})...")
    codex = CodexImporter(codex_home=Path(str(config["codex_home"])).expanduser())
    claude = ClaudeCodeImporter(
        claude_home=Path(str(config["claude_home"])).expanduser(),
        global_state_path=Path(str(config["claude_state"])).expanduser(),
        project_roots=[Path(project_path).expanduser()],
    )
    codex_events = codex.import_events()
    claude_events = claude.import_events()
    events = list(codex_events) + list(claude_events)
    project_roots = [Path(project_path).expanduser()]
    project_events = import_project_instruction_events(project_roots)
    if project_events:
        events.extend(project_events)
    project_marker_events = import_project_marker_events(project_roots)
    if project_marker_events:
        events.extend(project_marker_events)
    imports_dir.mkdir(parents=True, exist_ok=True)
    all_events_path = imports_dir / "all-events.jsonl"
    write_events_jsonl(events, all_events_path)
    _run_progress(True, f"imported {len(events)} events")

    if not events:
        return {"event_count": 0, "status": "no_events", "message": "no events found; nothing to sync"}

    # Step 2: dream (persistent run for better state tracking)
    dream_args = argparse.Namespace(
        input=str(all_events_path),
        project=normalize_project_path(project_path),
        output_dir=str(output_dir),
        memory_cards=getattr(args, "memory_cards", None),
        mode=getattr(args, "mode", None),
        provider=getattr(args, "provider", None),
        model=getattr(args, "model", None),
        api_key=getattr(args, "api_key", None),
        api_key_env=getattr(args, "api_key_env", None),
        base_url=getattr(args, "base_url", None),
        timeout_seconds=getattr(args, "timeout_seconds", None),
        invoke_model=getattr(args, "invoke_model", True),
    )
    payload, state = _run_dream_to_review(args=dream_args, config=config, persistent=True)
    if not state:
        return {**payload, "event_count": len(events)}

    run_id = str(state["run_id"])
    candidate_count = int(payload.get("candidate_count") or 0)
    _run_progress(True, f"dream complete; run_id={run_id}; candidates={candidate_count}")

    if not auto_mode:
        return {
            **payload,
            "event_count": len(events),
            "run_id": run_id,
            "auto": False,
            "next": f"dream-memory auto-review --run-id {run_id} --min-score {min_score}",
        }

    # Step 3: auto-review
    _run_progress(True, f"auto-review with min-score={min_score}...")
    auto_args = argparse.Namespace(
        run_id=run_id,
        output_dir=str(output_dir),
        reviewer="sync-auto",
        min_score=min_score,
        review_queue=None,
        reviewed_output=None,
        keep_review=False,
        include_duplicates=False,
        include_merges=False,
        include_review=True,
        force=True,
        dry_run=False,
    )
    auto_payload = _auto_review_run(args=auto_args, config=config)
    _run_progress(
        True,
        f"auto-review done: approved={auto_payload.get('approved', 0)}, "
        f"skipped={auto_payload.get('skipped', 0)}, "
        f"needs_manual={auto_payload.get('needs_more_evidence', 0)}",
    )

    if auto_payload.get("approved", 0) == 0 and auto_payload.get("needs_more_evidence", 0) == 0:
        return {
            "event_count": len(events),
            "run_id": run_id,
            "candidate_count": candidate_count,
            "auto_review": auto_payload,
            "status": "waiting_review",
            "message": "no approvable decisions; manual review required",
        }

    # Step 4: apply
    _run_progress(True, "applying memory decisions...")
    resume_args = argparse.Namespace(
        run_id=run_id,
        output_dir=str(output_dir),
        reviewed=None,
        memory_cards=getattr(args, "memory_cards", None),
        reviewer="sync-auto",
    )
    final_state = _resume_run(args=resume_args, config=config)
    counts = final_state.get("counts", {}) if isinstance(final_state.get("counts"), dict) else {}
    memory_count = counts.get("memory_count") or counts.get("review_decision_count")
    _run_progress(True, f"sync complete; memory_count={memory_count}")

    return {
        "event_count": len(events),
        "run_id": run_id,
        "candidate_count": candidate_count,
        "auto_review": auto_payload,
        "memory_count": memory_count,
        "status": "completed",
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_memory_config(args.config)

    if args.command == "init":
        payload = _init_workspace(args.path, force=bool(args.force), output_dir=getattr(args, "output_dir", None))
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
                    api_key=profile.config.api_key,
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
        api_key = profile.api_key
        base_url = profile.base_url
        timeout_seconds = profile.timeout_seconds
        payload = provider_diagnostics(
            provider=provider,
            model=model,
            api_key_env=api_key_env,
            api_key=api_key,
            base_url=str(base_url) if base_url else None,
            timeout_seconds=timeout_seconds,
            invoke=bool(args.invoke),
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0 if payload.get("ok") else 1

    if args.command == "sync":
        payload = _handle_sync(args=args, config=config)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "eval":
        eval_input = str(_value(getattr(args, "input", None), config.get("eval_input")))
        if not eval_input or eval_input == "None":
            print("error: eval requires --input or config eval_input", file=sys.stderr)
            return 2
        eval_project = _value(getattr(args, "project", None), config.get("eval_project"))
        eval_mode = str(_value(getattr(args, "mode", None), config.get("eval_mode", "rules")))
        eval_output = _value(getattr(args, "output", None), config.get("eval_output"))
        eval_model = _configured_model(args, config)
        eval_runtime_config = _runtime_config_from_args(args, config)
        eval_max_attempts = _value(getattr(args, "max_attempts", None), config.get("eval_max_attempts"))
        if eval_max_attempts is not None:
            policy = dict(eval_runtime_config.get("model_policy", {})) if isinstance(eval_runtime_config.get("model_policy"), dict) else {}
            retry = dict(policy.get("retry", {})) if isinstance(policy.get("retry"), dict) else {}
            retry["max_attempts"] = int(eval_max_attempts)
            policy["retry"] = retry
            eval_runtime_config["model_policy"] = policy
        try:
            payload = evaluate_labeled_events(
                eval_input,
                project=str(eval_project) if eval_project is not None else None,
                mode=eval_mode,
                model=eval_model,
                runtime_config=eval_runtime_config,
                invoke_model=True,
                continue_on_error=bool(getattr(args, "continue_on_error", False) or config.get("eval_continue_on_error", False)),
                max_rows=_value(getattr(args, "max_rows", None), config.get("eval_max_rows")),
                fallback_rules_on_error=bool(getattr(args, "fallback_rules_on_error", False) or config.get("eval_fallback_rules_on_error", False)),
                fallback_rules_on_empty=bool(getattr(args, "fallback_rules_on_empty", False) or config.get("eval_fallback_rules_on_empty", False)),
            )
        except FileNotFoundError as exc:
            print(f"error: eval input not found: {exc.filename or eval_input}", file=sys.stderr)
            return 2
        if eval_output:
            output = Path(str(eval_output)).expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            summary_keys = ["precision", "recall", "f1", "raw_candidate_total", "fallback_candidate_total", "scored_candidate_total", "extraction_success_count", "extraction_error_count", "fallback_count", "fallback_empty_count"]
            print(json.dumps({"output": str(output), **{k: payload[k] for k in summary_keys if k in payload}}, ensure_ascii=False, indent=2))
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
        extract_input = str(_value(getattr(args, "input", None), config.get("extract_input")))
        if not extract_input or extract_input == "None":
            print("error: extract-facts requires --input or config extract_input", file=sys.stderr)
            return 2
        try:
            events = load_events_jsonl(Path(extract_input))
        except FileNotFoundError as exc:
            print(f"error: extract-facts input not found: {exc.filename or extract_input}", file=sys.stderr)
            return 2
        extract_project = _value(getattr(args, "project", None), config.get("extract_project"))
        facts = extract_atomic_facts(events, project=str(extract_project) if extract_project is not None else None)
        output_dir = Path(str(_value(getattr(args, "output_dir", None), config.get("extract_output_dir", config["output_dir"])))).expanduser()
        facts_path = write_jsonl_records(facts, output_dir / "facts.jsonl")
        print(json.dumps({"fact_count": len(facts), "facts_path": str(facts_path)}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "review":
        candidates = load_events_jsonl(Path(args.candidates))
        memory_cards = _load_optional_jsonl(_memory_cards_path(args, config))
        queue = build_review_queue(candidates, memory_cards)
        output_dir = _configured_output_dir(args, config)
        queue_path = write_jsonl_records(queue, output_dir / "review_queue.jsonl")
        print(json.dumps({"review_count": len(queue), "review_queue_path": str(queue_path)}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "review-summary":
        output_dir = _configured_output_dir(args, config)
        if args.review_queue:
            queue_path = Path(args.review_queue).expanduser()
        elif args.run_id:
            state = load_run_state(output_dir, args.run_id)
            artifacts = state.get("artifacts", {}) if isinstance(state.get("artifacts"), dict) else {}
            queue_path = Path(str(artifacts.get("review_queue_path") or "")).expanduser()
        else:
            raise ValueError("review-summary requires --run-id or --review-queue")
        if not queue_path.exists():
            raise FileNotFoundError(f"review queue not found: {queue_path}")
        queue = load_events_jsonl(queue_path)
        payload = {"review_queue_path": str(queue_path), **_summarize_review_queue(queue)}
        if args.run_id:
            payload["run_id"] = args.run_id
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "auto-review":
        payload = _auto_review_run(args=args, config=config)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "apply":
        reviewed = load_events_jsonl(Path(args.reviewed))
        existing_cards = _load_optional_jsonl(_memory_cards_path(args, config))
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
        memory_cards_path = _memory_cards_path(args, config)
        cards = _load_optional_jsonl(memory_cards_path)
        payload = build_agent_context(cards, project=args.project, limit=int(_value(args.limit, config["context_limit"])), task=args.task)
        context_format = str(_value(args.format, config["context_format"]))
        if context_format == "markdown":
            print(render_context_markdown(payload), end="")
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "pipeline":
        try:
            payload, _ = _run_dream_to_review(args=args, config=config, persistent=False)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        except FileNotFoundError as exc:
            print(f"error: pipeline input not found: {exc.filename or getattr(args, 'input', '')}", file=sys.stderr)
            return 2
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "run":
        try:
            payload, _ = _run_dream_to_review(args=args, config=config, persistent=True)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        except FileNotFoundError as exc:
            print(f"error: run input not found: {exc.filename or getattr(args, 'input', '')}", file=sys.stderr)
            return 2
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
        cards = _load_optional_jsonl(_memory_cards_path(args, config))
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
        dream_input = str(_value(getattr(args, "input", None), config.get("default_input")))
        if not dream_input or dream_input == "None":
            print("error: dream requires --input or config default_input", file=sys.stderr)
            return 2
        try:
            events = load_events_jsonl(Path(dream_input))
        except FileNotFoundError as exc:
            print(f"error: dream input not found: {exc.filename or dream_input}", file=sys.stderr)
            return 2
        if getattr(args, "project", None) is None and config.get("default_project") is not None:
            args.project = str(config.get("default_project"))
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
            prompt_count_payload = {
                key: extraction[key]
                for key in ("input_event_count", "prompt_event_count", "filtered_prompt_event_count")
                if key in extraction
            }
            payload = {**result.to_dict(), "mode": "ai", "ai": True, "ai_dry_run": extraction["dry_run"], "ai_prompt_path": str(output_dir / "ai-prompt.md"), **prompt_count_payload}
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
    project_roots = _default_project_roots(args.project)
    project_events = import_project_instruction_events(project_roots)
    if project_events:
        events.extend(project_events)
        written_files.append(str(write_events_jsonl(project_events, output_dir / "project-instructions-events.jsonl")))
    project_marker_events = import_project_marker_events(project_roots)
    if project_marker_events:
        events.extend(project_marker_events)
        written_files.append(str(write_events_jsonl(project_marker_events, output_dir / "project-marker-events.jsonl")))
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
