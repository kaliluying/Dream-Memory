# Dream Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the missing half of the Dream Memory system so imported Claude Code / Codex session material can move from candidates into reviewed formal memory cards and then be injected back into agents as task-scoped context.

**Architecture:** Keep the current import and dreaming pipeline, then add a narrow persistence layer for facts, review queue items, reviewed decisions, and formal memory cards. Treat `MEMORY.md` as a human-readable projection of approved memory cards, not the source of truth, and add explicit CLI commands for review, apply, and context generation.

**Tech Stack:** Python 3.11+, argparse CLI, JSONL file storage, markdown renderers, pytest, existing dream-memory commands

## Global Constraints

- Keep the existing Claude Code / Codex import flow intact and extend it incrementally.
- Dreaming has proposal authority only; formal memory writes require explicit human review.
- Preserve evidence for every extracted fact, candidate, review item, and approved memory card.
- Default sensitive-content policy must prefer drop over redaction for promotable memory content.
- `project_state`, raw tool output, and build logs can be evidence but must not auto-promote into formal memory.
- Do not add vector storage, auto-approval, full automatic conflict resolution, or multi-user permissions in this implementation.
- Keep outputs under `.dream-memory/` and keep `MEMORY.md` as a derived, controlled artifact.
- Follow TDD for each task and keep files focused.

---

## File Map

- Modify: `src/dream_memory/memory_dreaming.py`
  - Split current event → candidate logic into event → fact → candidate helpers.
  - Add JSONL writers/loaders for facts, review queue items, reviewed decisions, and memory cards.
  - Add conflict detection helpers, formal-memory projection helpers, and context ranking helpers.
- Modify: `src/dream_memory/memory_cli.py`
  - Add `extract-facts`, `review`, `apply`, and `context` subcommands.
  - Keep `scan`, `import`, and `dream` compatible.
- Create: `src/dream_memory/memory_models.py`
  - Centralize typed dataclasses / dict builders for atomic facts, review queue items, reviewed decisions, and memory cards.
- Create: `tests/test_memory_models.py`
  - Cover model serialization and status defaults.
- Modify: `tests/test_memory_dreaming.py`
  - Add facts, review queue, reviewed memory, conflict detection, and context rendering tests.
- Modify: `tests/test_memory_cli.py`
  - Add CLI coverage for `extract-facts`, `review`, `apply`, and `context`.
- Optionally modify: `README.md`
  - Document the full memory pipeline after code is working.

---

### Task 1: Introduce formal memory data models

**Files:**
- Create: `src/dream_memory/memory_models.py`
- Test: `tests/test_memory_models.py`

**Interfaces:**
- Consumes: none
- Produces:
  - `build_atomic_fact(*, fact_type: str, statement: str, scope: str, project: str | None, source_event: dict[str, object], confidence: float, tags: list[str] | None = None) -> dict[str, object]`
  - `build_review_queue_item(*, candidate: dict[str, object], conflicts: list[dict[str, object]]) -> dict[str, object]`
  - `build_review_decision(*, candidate_id: str, status: str, reviewer: str, notes: str, memory_updates: list[dict[str, object]]) -> dict[str, object]`
  - `build_memory_card(*, memory_id: str, scope: str, project: str | None, memory_type: str, summary: str, evidence_refs: list[str], approved_by: str, approved_at: str, retrieval_hints: list[str], status: str = "active") -> dict[str, object]`

- [ ] **Step 1: Write the failing tests**

```python
from dream_memory.memory_models import (
    build_atomic_fact,
    build_memory_card,
    build_review_queue_item,
)


def test_build_atomic_fact_sets_defaults():
    fact = build_atomic_fact(
        fact_type="preference",
        statement="用户偏好中文回答。",
        scope="user",
        project=None,
        source_event={"source": "claude_code", "session_id": "s1", "event_id": "event_1"},
        confidence=0.9,
        tags=["language"],
    )

    assert fact["fact_type"] == "preference"
    assert fact["scope"] == "user"
    assert fact["status"] == "active"
    assert fact["evidence_refs"] == ["event_1"]


def test_build_review_queue_item_starts_pending():
    queue_item = build_review_queue_item(
        candidate={"id": "cand_1", "content": "项目目标是本地研发助手。"},
        conflicts=[],
    )

    assert queue_item["candidate_id"] == "cand_1"
    assert queue_item["status"] == "pending"
    assert queue_item["suggested_action"] == "review"


def test_build_memory_card_preserves_retrieval_metadata():
    card = build_memory_card(
        memory_id="mem_1",
        scope="project",
        project="/tmp/project",
        memory_type="decision",
        summary="项目目标是 Claude Code 风格的本地研发助手。",
        evidence_refs=["event_1", "event_2"],
        approved_by="user",
        approved_at="2026-07-05T00:00:00Z",
        retrieval_hints=["claude code", "local agent"],
    )

    assert card["status"] == "active"
    assert card["retrieval_hints"] == ["claude code", "local agent"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_memory_models.py -v`
Expected: FAIL with `ModuleNotFoundError` or missing symbol errors for `memory_models` builders.

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import hashlib


def _stable_id(prefix: str, raw: str) -> str:
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_atomic_fact(*, fact_type: str, statement: str, scope: str, project: str | None, source_event: dict[str, Any], confidence: float, tags: list[str] | None = None) -> dict[str, Any]:
    event_id = str(source_event.get("event_id") or "event")
    return {
        "id": _stable_id("fact", f"{scope}|{project or ''}|{fact_type}|{statement}"),
        "fact_type": fact_type,
        "statement": statement.strip(),
        "scope": scope,
        "project": project,
        "source": source_event.get("source"),
        "session_id": source_event.get("session_id"),
        "evidence_refs": [event_id],
        "confidence": round(float(confidence), 3),
        "tags": tags or [],
        "status": "active",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }


def build_review_queue_item(*, candidate: dict[str, Any], conflicts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "candidate_id": candidate["id"],
        "status": "pending",
        "suggested_action": "review",
        "candidate": candidate,
        "conflicts": conflicts,
        "created_at": _now_iso(),
    }


def build_review_decision(*, candidate_id: str, status: str, reviewer: str, notes: str, memory_updates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "status": status,
        "reviewer": reviewer,
        "notes": notes,
        "memory_updates": memory_updates,
        "reviewed_at": _now_iso(),
    }


def build_memory_card(*, memory_id: str, scope: str, project: str | None, memory_type: str, summary: str, evidence_refs: list[str], approved_by: str, approved_at: str, retrieval_hints: list[str], status: str = "active") -> dict[str, Any]:
    return {
        "id": memory_id,
        "scope": scope,
        "project": project,
        "memory_type": memory_type,
        "summary": summary.strip(),
        "evidence_refs": evidence_refs,
        "approved_by": approved_by,
        "approved_at": approved_at,
        "status": status,
        "retrieval_hints": retrieval_hints,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_memory_models.py -v`
Expected: PASS for all three tests.

- [ ] **Step 5: Commit**

```bash
git add tests/test_memory_models.py src/dream_memory/memory_models.py
git commit -m "feat: add dream memory data models"
```

### Task 2: Add atomic fact extraction and persistence

**Files:**
- Modify: `src/dream_memory/memory_dreaming.py`
- Modify: `src/dream_memory/memory_cli.py`
- Modify: `tests/test_memory_dreaming.py`
- Modify: `tests/test_memory_cli.py`

**Interfaces:**
- Consumes:
  - `build_atomic_fact(...) -> dict[str, object]`
- Produces:
  - `extract_atomic_facts(events: list[dict[str, object]], *, project: str | None) -> list[dict[str, object]]`
  - `write_jsonl_records(records: list[dict[str, object]], path: Path | str) -> Path`
  - CLI: `dream-memory extract-facts --input <events.jsonl> --project <path> --output-dir <dir>`

- [ ] **Step 1: Write the failing tests**

```python
def test_extract_atomic_facts_creates_fact_records():
    events = [
        {
            "event_id": "event_1",
            "source": "claude_code",
            "session_id": "global",
            "project": None,
            "role": "system",
            "event_type": "global_instruction",
            "content": "始终使用中文回答我",
        },
        {
            "event_id": "event_2",
            "source": "codex",
            "session_id": "s1",
            "project": "/tmp/project",
            "role": "user",
            "event_type": "history_prompt",
            "content": "这个项目需要人工审核后才能写正式记忆",
        },
    ]

    facts = extract_atomic_facts(events, project="/tmp/project")

    assert len(facts) >= 2
    assert any(fact["fact_type"] == "preference" for fact in facts)
    assert any("人工审核" in fact["statement"] for fact in facts)
```

```python
def test_extract_facts_cli_writes_facts_jsonl():
    exit_code = main([
        "extract-facts",
        "--input", str(events_path),
        "--project", str(root),
        "--output-dir", str(output_dir),
    ])

    assert exit_code == 0
    assert (output_dir / "facts.jsonl").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_memory_dreaming.py::MemoryDreamingTests::test_extract_atomic_facts_creates_fact_records tests/test_memory_cli.py::MemoryCliTests::test_extract_facts_cli_writes_facts_jsonl -v`
Expected: FAIL because `extract_atomic_facts` and the `extract-facts` subcommand do not exist.

- [ ] **Step 3: Write minimal implementation**

```python
from .memory_models import build_atomic_fact


def write_jsonl_records(records: list[dict[str, Any]], path: Path | str) -> Path:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output


def extract_atomic_facts(events: list[dict[str, Any]], *, project: str | None) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for index, event in enumerate(events, start=1):
        event = dict(event)
        event.setdefault("event_id", f"event_{index}")
        content = str(event.get("content") or "").strip()
        if not content or SENSITIVE_RE.search(content):
            continue
        lowered = content.lower()
        if event.get("event_type") == "global_instruction" or "始终" in content or "偏好" in content:
            facts.append(build_atomic_fact(
                fact_type="preference",
                statement=content,
                scope="user",
                project=None,
                source_event=event,
                confidence=0.95,
                tags=["preference"],
            ))
        if any(word in content for word in ["希望", "想", "需要", "不要", "必须"]):
            facts.append(build_atomic_fact(
                fact_type="requirement",
                statement=content,
                scope="project" if event.get("project") else "global",
                project=event.get("project"),
                source_event=event,
                confidence=0.82,
                tags=["requirement"],
            ))
        if any(token in lowered for token in ["uv", "python", "claude code", "codex", "dream"]):
            facts.append(build_atomic_fact(
                fact_type="project_fact" if event.get("project") else "global_fact",
                statement=content,
                scope="project" if event.get("project") else "global",
                project=event.get("project"),
                source_event=event,
                confidence=0.76,
                tags=[token for token in ["uv", "python", "claude-code", "codex", "dreams"] if token.replace("-", " ") in lowered or token in lowered],
            ))
    return facts
```

```python
extract = sub.add_parser("extract-facts", help="Extract atomic facts from normalized events")
extract.add_argument("--input", required=True)
extract.add_argument("--project")
extract.add_argument("--output-dir", default=".dream-memory")
```

```python
if args.command == "extract-facts":
    events = load_events_jsonl(Path(args.input))
    facts = extract_atomic_facts(events, project=args.project)
    output_dir = Path(args.output_dir).expanduser()
    facts_path = write_jsonl_records(facts, output_dir / "facts.jsonl")
    print(json.dumps({"fact_count": len(facts), "facts_path": str(facts_path)}, ensure_ascii=False, indent=2))
    return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_memory_dreaming.py::MemoryDreamingTests::test_extract_atomic_facts_creates_fact_records tests/test_memory_cli.py::MemoryCliTests::test_extract_facts_cli_writes_facts_jsonl -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_memory_dreaming.py tests/test_memory_cli.py src/dream_memory/memory_dreaming.py src/dream_memory/memory_cli.py
git commit -m "feat: extract atomic facts from memory events"
```

### Task 3: Convert facts to candidates with stricter promotion rules

**Files:**
- Modify: `src/dream_memory/memory_dreaming.py`
- Modify: `tests/test_memory_dreaming.py`

**Interfaces:**
- Consumes:
  - `extract_atomic_facts(...) -> list[dict[str, object]]`
- Produces:
  - `build_candidates_from_facts(facts: list[dict[str, object]]) -> list[dict[str, object]]`
  - Updated `dream_from_events(...) -> DreamResult` that writes `facts.jsonl` before `candidates.jsonl`

- [ ] **Step 1: Write the failing tests**

```python
def test_dream_from_events_writes_facts_before_candidates():
    result = dream_from_events(events, project="/tmp/project", output_dir=output_dir, apply=False)

    assert (output_dir / "facts.jsonl").exists()
    assert (output_dir / "candidates.jsonl").exists()
```

```python
def test_build_candidates_from_facts_keeps_project_state_as_evidence_only():
    facts = [
        {
            "id": "fact_1",
            "fact_type": "system_state",
            "statement": "Claude Code project state for /tmp/project",
            "scope": "project",
            "project": "/tmp/project",
            "evidence_refs": ["event_1"],
            "confidence": 0.6,
            "tags": ["project_state"],
        }
    ]

    candidates = build_candidates_from_facts(facts)

    assert candidates == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_memory_dreaming.py::MemoryDreamingTests::test_dream_from_events_writes_facts_before_candidates tests/test_memory_dreaming.py::MemoryDreamingTests::test_build_candidates_from_facts_keeps_project_state_as_evidence_only -v`
Expected: FAIL because facts are not persisted and candidate generation still starts from events directly.

- [ ] **Step 3: Write minimal implementation**

```python
def build_candidates_from_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for fact in facts:
        if fact["fact_type"] == "system_state" or "project_state" in fact.get("tags", []):
            continue
        content = str(fact["statement"])
        key = _candidate_id(content, str(fact["scope"]), fact.get("project"))
        candidate = candidates.setdefault(key, {
            "id": key,
            "type": fact["fact_type"],
            "scope": fact["scope"],
            "project": fact.get("project"),
            "content": content,
            "tags": list(fact.get("tags", [])),
            "evidence": [],
        })
        candidate["evidence"].extend({"event_id": ref} for ref in fact.get("evidence_refs", []))
        candidate["tags"] = sorted(set(candidate["tags"] + list(fact.get("tags", []))))
    return [score_candidate(candidate) for candidate in candidates.values()]
```

```python
facts = extract_atomic_facts(events, project=project)
write_jsonl_records(facts, output / "facts.jsonl")
candidates = build_candidates_from_facts(facts)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_memory_dreaming.py::MemoryDreamingTests::test_dream_from_events_writes_facts_before_candidates tests/test_memory_dreaming.py::MemoryDreamingTests::test_build_candidates_from_facts_keeps_project_state_as_evidence_only -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_memory_dreaming.py src/dream_memory/memory_dreaming.py
git commit -m "feat: derive dream candidates from atomic facts"
```

### Task 4: Add review queue generation and conflict detection

**Files:**
- Modify: `src/dream_memory/memory_dreaming.py`
- Modify: `src/dream_memory/memory_cli.py`
- Modify: `tests/test_memory_dreaming.py`
- Modify: `tests/test_memory_cli.py`

**Interfaces:**
- Consumes:
  - `build_review_queue_item(...) -> dict[str, object]`
- Produces:
  - `detect_candidate_conflicts(candidates: list[dict[str, object]], memory_cards: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]`
  - `build_review_queue(candidates: list[dict[str, object]], memory_cards: list[dict[str, object]]) -> list[dict[str, object]]`
  - CLI: `dream-memory review --candidates <path> --memory-cards <path> --output-dir <dir>`

- [ ] **Step 1: Write the failing tests**

```python
def test_detect_candidate_conflicts_flags_same_scope_competing_summary():
    candidates = [{
        "id": "cand_1",
        "scope": "project",
        "project": "/tmp/project",
        "type": "decision",
        "content": "项目目标是 Claude Code 风格助手。",
        "score": 0.9,
        "status": "promote",
    }]
    memory_cards = [{
        "id": "mem_1",
        "scope": "project",
        "project": "/tmp/project",
        "memory_type": "decision",
        "summary": "项目目标是通用聊天机器人。",
        "status": "active",
    }]

    conflicts = detect_candidate_conflicts(candidates, memory_cards)

    assert conflicts["cand_1"][0]["memory_id"] == "mem_1"
```

```python
def test_review_cli_writes_review_queue_jsonl():
    exit_code = main([
        "review",
        "--candidates", str(candidates_path),
        "--memory-cards", str(memory_cards_path),
        "--output-dir", str(output_dir),
    ])

    assert exit_code == 0
    assert (output_dir / "review_queue.jsonl").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_memory_dreaming.py::MemoryDreamingTests::test_detect_candidate_conflicts_flags_same_scope_competing_summary tests/test_memory_cli.py::MemoryCliTests::test_review_cli_writes_review_queue_jsonl -v`
Expected: FAIL because conflict detection and the `review` subcommand do not exist.

- [ ] **Step 3: Write minimal implementation**

```python
def detect_candidate_conflicts(candidates: list[dict[str, Any]], memory_cards: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    conflicts: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        content = str(candidate.get("content") or "")
        candidate_scope = candidate.get("scope")
        candidate_project = candidate.get("project")
        candidate_type = candidate.get("type")
        for card in memory_cards:
            if card.get("scope") != candidate_scope:
                continue
            if card.get("project") != candidate_project:
                continue
            if card.get("memory_type") != candidate_type:
                continue
            if str(card.get("summary") or "") == content:
                continue
            conflicts.setdefault(str(candidate["id"]), []).append({
                "memory_id": card.get("id"),
                "reason": "same-scope-type-different-summary",
                "summary": card.get("summary"),
            })
    return conflicts


def build_review_queue(candidates: list[dict[str, Any]], memory_cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conflict_map = detect_candidate_conflicts(candidates, memory_cards)
    queue = []
    for candidate in candidates:
        queue.append(build_review_queue_item(
            candidate=candidate,
            conflicts=conflict_map.get(str(candidate["id"]), []),
        ))
    return queue
```

```python
review = sub.add_parser("review", help="Build review queue items from candidates")
review.add_argument("--candidates", required=True)
review.add_argument("--memory-cards")
review.add_argument("--output-dir", default=".dream-memory")
```

```python
if args.command == "review":
    candidates = load_events_jsonl(Path(args.candidates))
    memory_cards = load_events_jsonl(Path(args.memory_cards)) if args.memory_cards else []
    queue = build_review_queue(candidates, memory_cards)
    output_dir = Path(args.output_dir).expanduser()
    queue_path = write_jsonl_records(queue, output_dir / "review_queue.jsonl")
    print(json.dumps({"review_count": len(queue), "review_queue_path": str(queue_path)}, ensure_ascii=False, indent=2))
    return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_memory_dreaming.py::MemoryDreamingTests::test_detect_candidate_conflicts_flags_same_scope_competing_summary tests/test_memory_cli.py::MemoryCliTests::test_review_cli_writes_review_queue_jsonl -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_memory_dreaming.py tests/test_memory_cli.py src/dream_memory/memory_dreaming.py src/dream_memory/memory_cli.py
git commit -m "feat: add review queue for dream memory"
```

### Task 5: Add apply flow for reviewed decisions and formal memory cards

**Files:**
- Modify: `src/dream_memory/memory_dreaming.py`
- Modify: `src/dream_memory/memory_cli.py`
- Modify: `tests/test_memory_dreaming.py`
- Modify: `tests/test_memory_cli.py`

**Interfaces:**
- Consumes:
  - `build_memory_card(...) -> dict[str, object]`
  - `build_review_decision(...) -> dict[str, object]`
- Produces:
  - `apply_reviewed_memory(reviewed: list[dict[str, object]], existing_cards: list[dict[str, object]]) -> tuple[list[dict[str, object]], str]`
  - CLI: `dream-memory apply --reviewed <path> --memory-cards <path> --output-dir <dir> --reviewer <name>`

- [ ] **Step 1: Write the failing tests**

```python
def test_apply_reviewed_memory_writes_memory_cards_and_markdown_projection():
    reviewed = [{
        "candidate_id": "cand_1",
        "status": "approved",
        "reviewer": "user",
        "notes": "looks good",
        "memory_updates": [{
            "id": "mem_1",
            "scope": "project",
            "project": "/tmp/project",
            "memory_type": "decision",
            "summary": "项目目标是 Claude Code 风格的本地研发助手。",
            "evidence_refs": ["event_1"],
            "approved_by": "user",
            "approved_at": "2026-07-05T00:00:00Z",
            "status": "active",
            "retrieval_hints": ["claude code"],
        }],
    }]

    cards, markdown = apply_reviewed_memory(reviewed, existing_cards=[])

    assert cards[0]["summary"] == "项目目标是 Claude Code 风格的本地研发助手。"
    assert "Claude Code 风格" in markdown
```

```python
def test_apply_cli_writes_memory_cards_and_memory_md():
    exit_code = main([
        "apply",
        "--reviewed", str(reviewed_path),
        "--output-dir", str(output_dir),
        "--reviewer", "user",
    ])

    assert exit_code == 0
    assert (output_dir / "memory_cards.jsonl").exists()
    assert (output_dir / "MEMORY.md").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_memory_dreaming.py::MemoryDreamingTests::test_apply_reviewed_memory_writes_memory_cards_and_markdown_projection tests/test_memory_cli.py::MemoryCliTests::test_apply_cli_writes_memory_cards_and_memory_md -v`
Expected: FAIL because apply helpers and the `apply` subcommand do not exist.

- [ ] **Step 3: Write minimal implementation**

```python
def render_memory_markdown(cards: list[dict[str, Any]]) -> str:
    lines = ["# MEMORY.md", "", "## Approved Memory", ""]
    for card in sorted(cards, key=lambda item: (str(item.get("scope")), str(item.get("summary")))):
        prefix = str(card.get("scope"))
        if card.get("project"):
            prefix = f"{prefix}: {card['project']}"
        lines.append(f"- **{prefix} / {card['memory_type']}**: {card['summary']}")
    return "\n".join(lines) + "\n"


def apply_reviewed_memory(reviewed: list[dict[str, Any]], existing_cards: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    cards_by_id = {str(card["id"]): dict(card) for card in existing_cards}
    for decision in reviewed:
        if decision.get("status") not in {"approved", "edited_and_approved", "merged"}:
            continue
        for update in decision.get("memory_updates", []):
            cards_by_id[str(update["id"])] = dict(update)
    cards = list(cards_by_id.values())
    return cards, render_memory_markdown(cards)
```

```python
apply_cmd = sub.add_parser("apply", help="Apply reviewed memory decisions")
apply_cmd.add_argument("--reviewed", required=True)
apply_cmd.add_argument("--memory-cards")
apply_cmd.add_argument("--output-dir", default=".dream-memory")
apply_cmd.add_argument("--reviewer", required=True)
```

```python
if args.command == "apply":
    reviewed = load_events_jsonl(Path(args.reviewed))
    existing_cards = load_events_jsonl(Path(args.memory_cards)) if args.memory_cards else []
    cards, markdown = apply_reviewed_memory(reviewed, existing_cards)
    output_dir = Path(args.output_dir).expanduser()
    cards_path = write_jsonl_records(cards, output_dir / "memory_cards.jsonl")
    memory_path = output_dir / "MEMORY.md"
    memory_path.write_text(markdown, encoding="utf-8")
    print(json.dumps({"memory_count": len(cards), "memory_cards_path": str(cards_path), "memory_path": str(memory_path)}, ensure_ascii=False, indent=2))
    return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_memory_dreaming.py::MemoryDreamingTests::test_apply_reviewed_memory_writes_memory_cards_and_markdown_projection tests/test_memory_cli.py::MemoryCliTests::test_apply_cli_writes_memory_cards_and_memory_md -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_memory_dreaming.py tests/test_memory_cli.py src/dream_memory/memory_dreaming.py src/dream_memory/memory_cli.py
git commit -m "feat: apply reviewed dream memory cards"
```

### Task 6: Add task-scoped context generation for agents

**Files:**
- Modify: `src/dream_memory/memory_dreaming.py`
- Modify: `src/dream_memory/memory_cli.py`
- Modify: `tests/test_memory_dreaming.py`
- Modify: `tests/test_memory_cli.py`

**Interfaces:**
- Consumes:
  - `memory_cards.jsonl`
- Produces:
  - `build_agent_context(memory_cards: list[dict[str, object]], *, project: str | None, limit: int = 12) -> dict[str, object]`
  - CLI: `dream-memory context --project <path> --memory-cards <path> --limit <n>`

- [ ] **Step 1: Write the failing tests**

```python
def test_build_agent_context_prioritizes_project_then_user_then_global():
    cards = [
        {"id": "mem_1", "scope": "global", "project": None, "memory_type": "workflow", "summary": "正式记忆必须人工审核。", "retrieval_hints": [], "status": "active"},
        {"id": "mem_2", "scope": "user", "project": None, "memory_type": "preference", "summary": "用户偏好中文回答。", "retrieval_hints": [], "status": "active"},
        {"id": "mem_3", "scope": "project", "project": "/tmp/project", "memory_type": "decision", "summary": "项目目标是 Claude Code 风格助手。", "retrieval_hints": [], "status": "active"},
    ]

    context = build_agent_context(cards, project="/tmp/project", limit=3)

    assert context["items"][0]["id"] == "mem_3"
    assert context["items"][1]["id"] == "mem_2"
    assert context["items"][2]["id"] == "mem_1"
```

```python
def test_context_cli_prints_ranked_context_json():
    exit_code = main([
        "context",
        "--project", "/tmp/project",
        "--memory-cards", str(cards_path),
        "--limit", "2",
    ])

    assert exit_code == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_memory_dreaming.py::MemoryDreamingTests::test_build_agent_context_prioritizes_project_then_user_then_global tests/test_memory_cli.py::MemoryCliTests::test_context_cli_prints_ranked_context_json -v`
Expected: FAIL because no context builder or `context` subcommand exists.

- [ ] **Step 3: Write minimal implementation**

```python
def build_agent_context(memory_cards: list[dict[str, Any]], *, project: str | None, limit: int = 12) -> dict[str, Any]:
    def rank(card: dict[str, Any]) -> tuple[int, str]:
        if card.get("status") != "active":
            return (99, str(card.get("summary")))
        if project and card.get("scope") == "project" and card.get("project") == project:
            return (0, str(card.get("summary")))
        if card.get("scope") == "user":
            return (1, str(card.get("summary")))
        if card.get("scope") == "global":
            return (2, str(card.get("summary")))
        if card.get("scope") == "session":
            return (3, str(card.get("summary")))
        return (4, str(card.get("summary")))

    filtered = [card for card in memory_cards if card.get("status") == "active"]
    ranked = sorted(filtered, key=rank)[:limit]
    return {
        "project": project,
        "count": len(ranked),
        "items": ranked,
    }
```

```python
context = sub.add_parser("context", help="Render task-scoped memory context for agents")
context.add_argument("--project")
context.add_argument("--memory-cards", default=".dream-memory/memory_cards.jsonl")
context.add_argument("--limit", type=int, default=12)
```

```python
if args.command == "context":
    cards = load_events_jsonl(Path(args.memory_cards))
    payload = build_agent_context(cards, project=args.project, limit=int(args.limit))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_memory_dreaming.py::MemoryDreamingTests::test_build_agent_context_prioritizes_project_then_user_then_global tests/test_memory_cli.py::MemoryCliTests::test_context_cli_prints_ranked_context_json -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_memory_dreaming.py tests/test_memory_cli.py src/dream_memory/memory_dreaming.py src/dream_memory/memory_cli.py
git commit -m "feat: generate scoped dream memory context"
```

### Task 7: Tighten safety rules and document the full pipeline

**Files:**
- Modify: `src/dream_memory/memory_dreaming.py`
- Modify: `src/dream_memory/memory_agent.py`
- Modify: `README.md`
- Modify: `tests/test_memory_dreaming.py`

**Interfaces:**
- Consumes:
  - Existing dream extraction helpers
- Produces:
  - Safer candidate filtering for secrets, project state, tool logs, and raw transcript copies
  - README section documenting `scan -> import -> extract-facts -> dream -> review -> apply -> context`

- [ ] **Step 1: Write the failing tests**

```python
def test_extract_atomic_facts_drops_secret_like_content():
    events = [{
        "event_id": "event_1",
        "source": "codex",
        "session_id": "s1",
        "project": "/tmp/project",
        "role": "user",
        "event_type": "history_prompt",
        "content": "OPENAI_API_KEY=sk-secret-value",
    }]

    facts = extract_atomic_facts(events, project="/tmp/project")

    assert facts == []
```

```python
def test_agent_prompt_rejects_project_state_as_memory_content():
    prompt = build_memory_extraction_prompt([
        {
            "source": "claude_code",
            "session_id": "s1",
            "project": "/tmp/project",
            "role": "system",
            "event_type": "project_state",
            "content": "Claude Code project state for /tmp/project",
        }
    ], project="/tmp/project")

    assert "reject it or omit it" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_memory_dreaming.py::MemoryDreamingTests::test_extract_atomic_facts_drops_secret_like_content tests/test_memory_dreaming.py::MemoryDreamingTests::test_agent_prompt_rejects_project_state_as_memory_content -v`
Expected: FAIL if secret-like content is still convertible or prompt policy is too weak.

- [ ] **Step 3: Write minimal implementation**

```python
BLOCKED_EVENT_TYPES = {"project_state", "tool_output", "build_log"}


def extract_atomic_facts(events: list[dict[str, Any]], *, project: str | None) -> list[dict[str, Any]]:
    facts = []
    for index, event in enumerate(events, start=1):
        event = dict(event)
        event.setdefault("event_id", f"event_{index}")
        if str(event.get("event_type") or "") in BLOCKED_EVENT_TYPES:
            continue
        content = str(event.get("content") or "").strip()
        if not content or SENSITIVE_RE.search(content):
            continue
        ...
```

```python
- You must reject tool state, project index records, one-off logs, temporary command output, and low-value status metadata.
- API keys, auth tokens, cookie values, passwords, and raw credential strings must be omitted entirely, not summarized.
```

```md
## Dream Memory Workflow

```bash
uv run dream-memory scan --output .dream-memory/scan.json
uv run dream-memory import all --output-dir .dream-memory/imports --dry-run
uv run dream-memory extract-facts --input .dream-memory/imports/all-events.jsonl --project . --output-dir .dream-memory
uv run dream-memory dream --input .dream-memory/imports/all-events.jsonl --project . --output-dir .dream-memory
uv run dream-memory review --candidates .dream-memory/candidates.jsonl --memory-cards .dream-memory/memory_cards.jsonl --output-dir .dream-memory
uv run dream-memory apply --reviewed .dream-memory/reviewed.jsonl --memory-cards .dream-memory/memory_cards.jsonl --output-dir .dream-memory --reviewer user
uv run dream-memory context --project . --memory-cards .dream-memory/memory_cards.jsonl --limit 12
```
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_memory_dreaming.py::MemoryDreamingTests::test_extract_atomic_facts_drops_secret_like_content tests/test_memory_dreaming.py::MemoryDreamingTests::test_agent_prompt_rejects_project_state_as_memory_content -v`
Expected: PASS.

- [ ] **Step 5: Run the focused memory test suite**

Run: `pytest tests/test_memory_models.py tests/test_memory_importers.py tests/test_memory_dreaming.py tests/test_memory_cli.py -v`
Expected: PASS for all focused memory tests.

- [ ] **Step 6: Commit**

```bash
git add README.md tests/test_memory_models.py tests/test_memory_importers.py tests/test_memory_dreaming.py tests/test_memory_cli.py src/dream_memory/memory_models.py src/dream_memory/memory_dreaming.py src/dream_memory/memory_agent.py src/dream_memory/memory_cli.py
git commit -m "docs: finalize dream memory workflow"
```

## Spec Coverage Check

- Import Claude / Codex events: covered by existing commands, preserved in Tasks 2 and 7.
- Generate candidates from imported material: covered by Tasks 2 and 3.
- Add explicit review queue: covered by Task 4.
- Require human-reviewed formal memory writes: covered by Task 5.
- Add agent context generation: covered by Task 6.
- Tighten safety and evidence rules: covered by Task 7.

## Placeholder Scan

- No `TODO`, `TBD`, or deferred implementation markers remain.
- Every task lists exact files, commands, and concrete code snippets.
- Later tasks use names defined in earlier tasks.

## Type Consistency Check

- Facts use `statement`, candidates use `content`, and formal cards use `summary` consistently.
- `candidate_id`, `memory_id`, `reviewed`, and `memory_cards` names match across tasks.
- CLI subcommands and output file names are consistent with the design and current repository layout.
