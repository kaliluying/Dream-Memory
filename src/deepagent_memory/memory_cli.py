from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .memory_agent import agent_extract_memory_candidates
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


def _default_project_roots(values: list[str] | None) -> list[Path]:
    return [Path(value).expanduser() for value in values or []]


def _add_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--codex-home", default=str(Path.home() / ".codex"))
    parser.add_argument("--claude-home", default=str(Path.home() / ".claude"))
    parser.add_argument("--claude-state", default=str(Path.home() / ".claude.json"))
    parser.add_argument("--project", action="append", default=[])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deepagent-memory", description="Scan and import Claude Code / Codex sessions into shared memory events.")
    _add_source_args(parser)
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser("scan", help="Scan available Codex/Claude sources")
    _add_source_args(scan)
    scan.add_argument("--output")

    imp = sub.add_parser("import", help="Import normalized events")
    _add_source_args(imp)
    imp.add_argument("source", choices=["codex", "claude", "all"])
    imp.add_argument("--output-dir", default=".deepagent/memory/imports")
    imp.add_argument("--dry-run", action="store_true")
    dream = sub.add_parser("dream", help="Run memory dreaming over normalized events")
    dream.add_argument("--input", required=True, help="Input normalized events JSONL")
    dream.add_argument("--project")
    dream.add_argument("--output-dir", default=".deepagent/memory")
    dream.add_argument("--apply", action="store_true", help="Append promoted preview to MEMORY.md")
    dream.add_argument("--mode", choices=["ai", "rules"], default="ai", help="Extraction mode: ai is default; rules is fallback/debug")
    dream.add_argument("--agent", action="store_true", help="Deprecated alias for --mode ai")
    dream.add_argument("--model", default="anthropic:claude-sonnet-4-6")
    dream.add_argument("--invoke-model", action="store_true", help="Actually invoke the model; default writes prompt only")

    extract = sub.add_parser("extract-facts", help="Extract atomic facts from normalized events")
    extract.add_argument("--input", required=True)
    extract.add_argument("--project")
    extract.add_argument("--output-dir", default=".deepagent/memory")

    review = sub.add_parser("review", help="Build review queue items from candidates")
    review.add_argument("--candidates", required=True)
    review.add_argument("--memory-cards")
    review.add_argument("--output-dir", default=".deepagent/memory")

    apply_cmd = sub.add_parser("apply", help="Apply reviewed memory decisions")
    apply_cmd.add_argument("--reviewed", required=True)
    apply_cmd.add_argument("--memory-cards")
    apply_cmd.add_argument("--output-dir", default=".deepagent/memory")
    apply_cmd.add_argument("--reviewer", required=True)

    context = sub.add_parser("context", help="Render task-scoped memory context for agents")
    context.add_argument("--project")
    context.add_argument("--memory-cards", default=".deepagent/memory/memory_cards.jsonl")
    context.add_argument("--limit", type=int, default=12)
    context.add_argument("--format", choices=["json", "markdown"], default="json")

    pipeline = sub.add_parser("pipeline", help="Run dream and review in one step")
    pipeline.add_argument("--input", required=True)
    pipeline.add_argument("--project")
    pipeline.add_argument("--output-dir", default=".deepagent/memory")
    pipeline.add_argument("--memory-cards")
    pipeline.add_argument("--mode", choices=["ai", "rules"], default="ai")
    pipeline.add_argument("--model", default="anthropic:claude-sonnet-4-6")
    pipeline.add_argument("--invoke-model", action="store_true")

    return parser


def _build_importers(args: argparse.Namespace) -> tuple[CodexImporter, ClaudeCodeImporter]:
    codex = CodexImporter(codex_home=Path(args.codex_home))
    claude = ClaudeCodeImporter(
        claude_home=Path(args.claude_home),
        global_state_path=Path(args.claude_state),
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    codex, claude = _build_importers(args)
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
        output_dir = Path(args.output_dir).expanduser()
        facts_path = write_jsonl_records(facts, output_dir / "facts.jsonl")
        print(json.dumps({"fact_count": len(facts), "facts_path": str(facts_path)}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "review":
        candidates = load_events_jsonl(Path(args.candidates))
        memory_cards = _load_optional_jsonl(args.memory_cards)
        queue = build_review_queue(candidates, memory_cards)
        output_dir = Path(args.output_dir).expanduser()
        queue_path = write_jsonl_records(queue, output_dir / "review_queue.jsonl")
        print(json.dumps({"review_count": len(queue), "review_queue_path": str(queue_path)}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "apply":
        reviewed = load_events_jsonl(Path(args.reviewed))
        existing_cards = _load_optional_jsonl(args.memory_cards)
        cards, markdown, decisions = apply_reviewed_memory(reviewed, existing_cards, return_decisions=True)
        output_dir = Path(args.output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        cards_path = write_jsonl_records(cards, output_dir / "memory_cards.jsonl")
        decisions_path = write_jsonl_records(decisions, output_dir / "review_decisions.jsonl")
        memory_path = output_dir / "MEMORY.md"
        memory_path.write_text(markdown, encoding="utf-8")
        print(json.dumps({"memory_count": len(cards), "memory_cards_path": str(cards_path), "review_decisions_path": str(decisions_path), "memory_path": str(memory_path), "reviewer": args.reviewer}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "context":
        cards = load_events_jsonl(Path(args.memory_cards))
        payload = build_agent_context(cards, project=args.project, limit=int(args.limit))
        if args.format == "markdown":
            print(render_context_markdown(payload), end="")
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "pipeline":
        events = load_events_jsonl(Path(args.input))
        output_dir = Path(args.output_dir).expanduser()
        if args.mode == "ai":
            extraction = agent_extract_memory_candidates(events, project=args.project, model=args.model, invoke_model=bool(args.invoke_model))
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "agent-prompt.md").write_text(str(extraction["prompt"]), encoding="utf-8")
            if "raw_response" in extraction:
                (output_dir / "agent-raw-response.txt").write_text(str(extraction["raw_response"]), encoding="utf-8")
            result = dream_from_events(events, project=args.project, output_dir=output_dir, apply=False, agent_candidates=list(extraction.get("candidates", [])), agent_mode=True)
            payload = {**result.to_dict(), "mode": "ai", "agent_dry_run": extraction["dry_run"], "agent_prompt_path": str(output_dir / "agent-prompt.md")}
        else:
            result = dream_from_events(events, project=args.project, output_dir=output_dir, apply=False)
            payload = {**result.to_dict(), "mode": "rules"}
        candidates = load_events_jsonl(Path(result.candidates_path))
        memory_cards = _load_optional_jsonl(args.memory_cards)
        queue = build_review_queue(candidates, memory_cards)
        queue_path = write_jsonl_records(queue, output_dir / "review_queue.jsonl")
        print(json.dumps({**payload, "review_queue_path": str(queue_path), "review_count": len(queue)}, ensure_ascii=False, indent=2))
        return 0

    if args.command == "dream":
        events = load_events_jsonl(Path(args.input))
        output_dir = Path(args.output_dir)
        mode = "ai" if args.agent else args.mode
        if mode == "ai":
            extraction = agent_extract_memory_candidates(events, project=args.project, model=args.model, invoke_model=bool(args.invoke_model))
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

    output_dir = Path(args.output_dir).expanduser()
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
