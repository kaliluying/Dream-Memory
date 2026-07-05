from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .memory_models import build_atomic_fact, build_memory_card, build_review_decision, build_review_queue_item

SENSITIVE_RE = re.compile(
    r"(sk-[a-zA-Z0-9]|api[_-]?key\s*[=:]|access[_-]?token\s*[=:]|refresh[_-]?token\s*[=:]|password\s*[=:]|secret\s*[=:]|cookie\s*[=:]|bearer\s+[a-zA-Z0-9]|-----BEGIN [A-Z ]*PRIVATE KEY-----|AKIA[0-9A-Z]{16}|eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+)",
    re.I,
)
BLOCKED_EVENT_TYPES = {"project_state", "tool_output", "build_log"}
RAW_TRANSCRIPT_RE = re.compile(r"(^|\n)\s*(user|assistant|system)\s*:", re.I)


@dataclass(frozen=True)
class DreamResult:
    event_count: int
    candidate_count: int
    promoted_count: int
    review_count: int
    rejected_count: int
    output_dir: str
    candidates_path: str
    dreams_path: str
    memory_preview_path: str
    memory_path: str | None
    applied: bool

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def load_events_jsonl(path: Path | str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                events.append(obj)
    return events



def write_jsonl_records(records: list[dict[str, Any]], path: Path | str) -> Path:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(output)
    return output


def normalize_memory_text(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value).strip().lower())
    text = re.sub(r"[。．.!！?？,，;；:：]+$", "", text)
    return text.strip()


def _content_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def _is_raw_transcript_like(content: str) -> bool:
    if len(content) > 1200:
        return True
    if content.count("\n") >= 6:
        return True
    if "```" in content:
        return True
    return bool(RAW_TRANSCRIPT_RE.search(content))


def _evidence_refs_from_candidate(candidate: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for index, evidence in enumerate(candidate.get("evidence", []), start=1):
        if isinstance(evidence, dict):
            ref = evidence.get("event_id") or evidence.get("id") or evidence.get("source_event_id")
            if ref:
                refs.append(str(ref))
            else:
                source = evidence.get("source") or "evidence"
                session = evidence.get("session_id") or index
                refs.append(f"{source}:{session}")
        elif evidence:
            refs.append(str(evidence))
    return refs or [str(candidate.get("id") or "candidate")]


def extract_atomic_facts(events: list[dict[str, Any]], *, project: str | None) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for index, raw_event in enumerate(events, start=1):
        event = dict(raw_event)
        event.setdefault("event_id", f"event_{index}")
        event_type = str(event.get("event_type") or "")
        if event_type in BLOCKED_EVENT_TYPES:
            continue
        content = str(event.get("content") or "").strip()
        if not content or SENSITIVE_RE.search(content):
            continue
        lowered = content.lower()
        event_project = event.get("project") or project

        if event_type == "global_instruction" or "始终" in content or "偏好" in content or "prefer" in lowered:
            facts.append(build_atomic_fact(
                fact_type="preference",
                statement=content,
                scope="user",
                project=None,
                source_event=event,
                confidence=0.95,
                tags=["preference"],
            ))

        if any(word in content for word in ["希望", "想", "需要", "不要", "必须", "人工审核"]):
            facts.append(build_atomic_fact(
                fact_type="requirement",
                statement=content,
                scope="project" if event_project else "global",
                project=str(event_project) if event_project else None,
                source_event=event,
                confidence=0.82,
                tags=["requirement"],
            ))

        if any(token in lowered for token in ["uv", "python", "claude code", "codex", "dream", "runtime", "patch"]):
            tags = [
                tag for tag, needle in [
                    ("uv", "uv"),
                    ("python", "python"),
                    ("dreams", "dream"),
                    ("claude-code", "claude code"),
                    ("codex", "codex"),
                    ("patch", "patch"),
                    ("runtime", "runtime"),
                ] if needle in lowered
            ]
            facts.append(build_atomic_fact(
                fact_type="project_fact" if event_project else "global_fact",
                statement=content,
                scope="project" if event_project else "global",
                project=str(event_project) if event_project else None,
                source_event=event,
                confidence=0.76,
                tags=tags,
            ))
    return facts

def _candidate_id(content: str, scope: str, project: str | None) -> str:
    raw = f"{scope}|{project or ''}|{content}".encode("utf-8")
    return "mem_" + hashlib.sha1(raw).hexdigest()[:12]


def _base_candidate(
    *,
    memory_type: str,
    scope: str,
    project: str | None,
    content: str,
    event: dict[str, Any],
    tags: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": _candidate_id(content, scope, project),
        "type": memory_type,
        "scope": scope,
        "project": project,
        "content": content.strip(),
        "tags": tags or [],
        "evidence": [
            {
                "source": event.get("source"),
                "session_id": event.get("session_id"),
                "event_type": event.get("event_type"),
                "timestamp": event.get("timestamp"),
            }
        ],
    }


def classify_event(event: dict[str, Any]) -> list[dict[str, Any]]:
    content = str(event.get("content") or "").strip()
    project = event.get("project")
    event_type = str(event.get("event_type") or "")
    role = str(event.get("role") or "")
    if not content or SENSITIVE_RE.search(content):
        return []

    candidates: list[dict[str, Any]] = []
    lowered = content.lower()

    if event_type == "global_instruction" or "始终" in content or "偏好" in content or "prefer" in lowered:
        candidates.append(_base_candidate(
            memory_type="preference",
            scope="global",
            project=None,
            content=content,
            event=event,
            tags=["preference"],
        ))

    if any(token in content for token in ["使用 uv", "uv 管理", "Python", "Dreams", "Claude Code", "Codex", "runtime", "patch"]):
        candidates.append(_base_candidate(
            memory_type="project_fact" if project else "global_fact",
            scope="project" if project else "global",
            project=project,
            content=content,
            event=event,
            tags=[tag for tag in ["uv" if "uv" in lowered else "", "python" if "python" in lowered else "", "dreams" if "dream" in lowered else "", "claude-code" if "claude code" in lowered else "", "codex" if "codex" in lowered else "", "patch" if "patch" in lowered else ""] if tag],
        ))

    if role == "user" and any(word in content for word in ["希望", "想", "需要", "不要", "必须"]):
        candidates.append(_base_candidate(
            memory_type="requirement",
            scope="project" if project else "global",
            project=project,
            content=content,
            event=event,
            tags=["requirement"],
        ))

    return candidates


def _merge_candidates(candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        key = candidate["id"]
        if key not in merged:
            merged[key] = candidate
            continue
        merged[key]["evidence"].extend(candidate.get("evidence", []))
        merged[key]["tags"] = sorted(set(merged[key].get("tags", []) + candidate.get("tags", [])))
    return list(merged.values())


def score_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    content = str(candidate.get("content") or "")
    evidence_count = len(candidate.get("evidence", []))
    tags = set(candidate.get("tags", []))
    score = 0.2
    score += min(evidence_count, 4) * 0.12
    if candidate.get("scope") == "project":
        score += 0.18
    if candidate.get("type") in {"preference", "requirement", "project_fact"}:
        score += 0.18
    if tags & {"uv", "python", "dreams", "claude-code", "codex", "patch"}:
        score += 0.12
    if len(content) >= 12:
        score += 0.08
    if SENSITIVE_RE.search(content):
        score -= 0.6
    score = max(0.0, min(1.0, round(score, 3)))
    candidate = dict(candidate)
    candidate["score"] = score
    candidate["status"] = "promote" if score >= 0.72 else "review" if score >= 0.5 else "reject"
    return candidate



def build_candidates_from_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for fact in facts:
        if fact.get("fact_type") == "system_state" or "project_state" in fact.get("tags", []):
            continue
        content = str(fact.get("statement") or "").strip()
        if not content or SENSITIVE_RE.search(content) or _is_raw_transcript_like(content):
            continue
        scope = str(fact.get("scope") or "global")
        project = fact.get("project")
        key_content = normalize_memory_text(content)
        key = _candidate_id(key_content, scope, str(project) if project else None)
        candidate = candidates.setdefault(key, {
            "id": key,
            "type": fact.get("fact_type"),
            "scope": scope,
            "project": project,
            "content": content,
            "tags": list(fact.get("tags", [])),
            "evidence": [],
        })
        for ref in fact.get("evidence_refs", []):
            candidate["evidence"].append({
                "event_id": ref,
                "source": fact.get("source"),
                "session_id": fact.get("session_id"),
                "content_hash": _content_hash(content),
            })
        candidate["tags"] = sorted(set(candidate.get("tags", []) + list(fact.get("tags", []))))
    return [score_candidate(candidate) for candidate in candidates.values()]


def detect_candidate_conflicts(candidates: list[dict[str, Any]], memory_cards: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    conflicts: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        content = str(candidate.get("content") or "")
        candidate_scope = candidate.get("scope")
        candidate_project = candidate.get("project")
        candidate_type = candidate.get("type")
        for card in memory_cards:
            if card.get("status", "active") != "active":
                continue
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
    return [build_review_queue_item(candidate=candidate, conflicts=conflict_map.get(str(candidate["id"]), [])) for candidate in candidates]


def render_memory_markdown(cards: list[dict[str, Any]]) -> str:
    lines = ["# MEMORY.md", "", "## Approved Memory", ""]
    for card in sorted(cards, key=lambda item: (str(item.get("scope")), str(item.get("project") or ""), str(item.get("summary")))):
        if card.get("status") != "active":
            continue
        prefix = str(card.get("scope"))
        if card.get("project"):
            prefix = f"{prefix}: {card['project']}"
        evidence = ", ".join(str(ref) for ref in card.get("evidence_refs", []))
        suffix = f" _(Evidence: {evidence})_" if evidence else ""
        lines.append(f"- **{prefix} / {card['memory_type']}**: {card['summary']}{suffix}")
    return "\n".join(lines) + "\n"


def _memory_update_from_web_review(review: dict[str, Any]) -> dict[str, Any] | None:
    action = str(review.get("action") or review.get("status") or "")
    if action not in {"approved", "edited_and_approved", "merged"}:
        return None
    candidate = review.get("candidate") if isinstance(review.get("candidate"), dict) else {}
    summary = str(review.get("edited_content") or candidate.get("content") or "").strip()
    if not summary:
        return None
    scope = str(candidate.get("scope") or "global")
    project = candidate.get("project")
    memory_type = str(candidate.get("type") or candidate.get("memory_type") or "memory")
    evidence_refs = _evidence_refs_from_candidate(candidate)
    memory_id = str(review.get("memory_id") or _candidate_id(normalize_memory_text(summary), scope, str(project) if project else None))
    approved_at = str(review.get("reviewed_at") or datetime.now(timezone.utc).isoformat())
    hints = list(candidate.get("tags", [])) if isinstance(candidate.get("tags"), list) else []
    return build_memory_card(
        memory_id=memory_id,
        scope=scope,
        project=str(project) if project else None,
        memory_type=memory_type,
        summary=summary,
        evidence_refs=evidence_refs,
        approved_by=str(review.get("reviewer") or "user"),
        approved_at=approved_at,
        retrieval_hints=[str(hint) for hint in hints],
    )


def normalize_review_decision(review: dict[str, Any]) -> dict[str, Any]:
    status = str(review.get("status") or review.get("action") or "pending")
    if "memory_updates" in review and "status" in review:
        return dict(review)
    update = _memory_update_from_web_review(review)
    updates = [update] if update else []
    return build_review_decision(
        candidate_id=str(review.get("candidate_id") or review.get("candidate", {}).get("id") or "candidate"),
        status=status,
        reviewer=str(review.get("reviewer") or "user"),
        notes=str(review.get("notes") or review.get("note") or ""),
        memory_updates=updates,
    )


def apply_reviewed_memory(
    reviewed: list[dict[str, Any]],
    existing_cards: list[dict[str, Any]],
    *,
    return_decisions: bool = False,
) -> tuple[list[dict[str, Any]], str] | tuple[list[dict[str, Any]], str, list[dict[str, Any]]]:
    cards_by_id = {str(card["id"]): dict(card) for card in existing_cards}
    decisions: list[dict[str, Any]] = []
    for raw_decision in reviewed:
        decision = normalize_review_decision(raw_decision)
        decisions.append(decision)
        if decision.get("status") not in {"approved", "edited_and_approved", "merged"}:
            continue
        for superseded in raw_decision.get("supersedes", []):
            if str(superseded) in cards_by_id:
                cards_by_id[str(superseded)]["status"] = "superseded"
                cards_by_id[str(superseded)]["superseded_at"] = str(decision.get("reviewed_at") or datetime.now(timezone.utc).isoformat())
        for update in decision.get("memory_updates", []):
            if not isinstance(update, dict) or "id" not in update:
                continue
            cards_by_id[str(update["id"])] = dict(update)
    cards = list(cards_by_id.values())
    markdown = render_memory_markdown(cards)
    if return_decisions:
        return cards, markdown, decisions
    return cards, markdown


def build_agent_context(memory_cards: list[dict[str, Any]], *, project: str | None, limit: int = 12) -> dict[str, Any]:
    def rank(card: dict[str, Any]) -> tuple[int, str]:
        if project and card.get("scope") == "project" and card.get("project") == project:
            return (0, str(card.get("summary")))
        if card.get("scope") == "user":
            return (1, str(card.get("summary")))
        if card.get("scope") == "global":
            return (2, str(card.get("summary")))
        if card.get("scope") == "session":
            return (3, str(card.get("summary")))
        return (4, str(card.get("summary")))

    filtered = []
    for card in memory_cards:
        if card.get("status") != "active":
            continue
        if card.get("scope") == "project" and project and card.get("project") != project:
            continue
        filtered.append(card)
    ranked = sorted(filtered, key=rank)[:limit]
    return {"project": project, "count": len(ranked), "items": ranked}


def render_context_markdown(context: dict[str, Any]) -> str:
    lines = ["## Relevant Memory", ""]
    for item in context.get("items", []):
        prefix = str(item.get("scope"))
        if item.get("project"):
            prefix = f"{prefix}: {item['project']}"
        lines.append(f"- **{prefix} / {item.get('memory_type')}**: {item.get('summary')}")
    return "\n".join(lines) + "\n"

def _render_dreams(events: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> str:
    promoted = [c for c in candidates if c["status"] == "promote"]
    review = [c for c in candidates if c["status"] == "review"]
    rejected = [c for c in candidates if c["status"] == "reject"]
    lines = [
        "# DREAMS.md",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Sweep Summary",
        "",
        f"- Events scanned: {len(events)}",
        f"- Candidates: {len(candidates)}",
        f"- Promote: {len(promoted)}",
        f"- Review: {len(review)}",
        f"- Reject: {len(rejected)}",
        "",
        "## Promoted Candidates",
        "",
    ]
    for c in promoted[:30]:
        lines.append(f"- ({c['type']}, {c['score']}) {c['content']}")
    lines.extend(["", "## Review Candidates", ""])
    for c in review[:30]:
        lines.append(f"- ({c['type']}, {c['score']}) {c['content']}")
    return "\n".join(lines) + "\n"


def _render_memory_preview(candidates: list[dict[str, Any]]) -> str:
    promoted = [c for c in candidates if c["status"] == "promote"]
    review = [c for c in candidates if c["status"] == "review"]
    lines = ["# MEMORY.preview.md", "", "## Proposed Long-Term Memory", ""]
    for c in promoted:
        prefix = "Global" if c.get("scope") == "global" else f"Project: {c.get('project')}"
        lines.append(f"- **{prefix} / {c['type']}**: {c['content']}")
    if review:
        lines.extend(["", "## Needs Review", ""])
        for c in review:
            prefix = "Global" if c.get("scope") == "global" else f"Project: {c.get('project')}"
            lines.append(f"- **{prefix} / {c['type']} / {c['score']}**: {c['content']}")
    return "\n".join(lines) + "\n"


def dream_from_events(
    events: list[dict[str, Any]],
    *,
    project: str | None,
    output_dir: Path | str,
    apply: bool = False,
    agent_candidates: list[dict[str, Any]] | None = None,
    agent_mode: bool = False,
) -> DreamResult:
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    facts = extract_atomic_facts(events, project=project)
    write_jsonl_records(facts, output / "facts.jsonl")

    if agent_candidates is not None:
        raw_candidates = []
        for candidate in agent_candidates:
            normalized = dict(candidate)
            normalized.setdefault("id", _candidate_id(str(normalized.get("content", "")), str(normalized.get("scope", "global")), normalized.get("project")))
            normalized.setdefault("tags", [])
            normalized.setdefault("evidence", [])
            if "score" not in normalized:
                confidence = float(normalized.get("confidence", 0.5) or 0.5)
                decision = normalized.get("decision", "review")
                normalized["score"] = round(confidence, 3)
                normalized["status"] = "promote" if decision == "promote" else "reject" if decision == "reject" else "review"
            raw_candidates.append(normalized)
        candidates = raw_candidates
    else:
        candidates = build_candidates_from_facts(facts)
    candidates.sort(key=lambda item: (-item["score"], item["type"], item["content"]))

    candidates_path = output / ("ai-candidates.jsonl" if agent_mode else "candidates.jsonl")
    write_jsonl_records(candidates, candidates_path)

    dreams_path = output / "DREAMS.md"
    dreams_path.write_text(_render_dreams(events, candidates), encoding="utf-8")

    preview_path = output / "MEMORY.preview.md"
    preview_text = _render_memory_preview(candidates)
    preview_path.write_text(preview_text, encoding="utf-8")

    memory_path: Path | None = None
    if apply:
        memory_path = output / "MEMORY.md"
        existing = memory_path.read_text(encoding="utf-8") if memory_path.exists() else "# MEMORY.md\n\n"
        memory_path.write_text(existing.rstrip() + "\n\n" + preview_text, encoding="utf-8")

    promoted = len([c for c in candidates if c["status"] == "promote"])
    review = len([c for c in candidates if c["status"] == "review"])
    rejected = len([c for c in candidates if c["status"] == "reject"])
    return DreamResult(
        event_count=len(events),
        candidate_count=len(candidates),
        promoted_count=promoted,
        review_count=review,
        rejected_count=rejected,
        output_dir=str(output),
        candidates_path=str(candidates_path),
        dreams_path=str(dreams_path),
        memory_preview_path=str(preview_path),
        memory_path=str(memory_path) if memory_path else None,
        applied=apply,
    )
