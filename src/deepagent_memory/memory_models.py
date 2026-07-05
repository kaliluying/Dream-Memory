from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import hashlib


def _stable_id(prefix: str, raw: str) -> str:
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_atomic_fact(
    *,
    fact_type: str,
    statement: str,
    scope: str,
    project: str | None,
    source_event: dict[str, Any],
    confidence: float,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    event_id = str(source_event.get("event_id") or "event")
    now = _now_iso()
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
        "created_at": now,
        "updated_at": now,
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


def build_review_decision(
    *,
    candidate_id: str,
    status: str,
    reviewer: str,
    notes: str,
    memory_updates: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "status": status,
        "reviewer": reviewer,
        "notes": notes,
        "memory_updates": memory_updates,
        "reviewed_at": _now_iso(),
    }


def build_memory_card(
    *,
    memory_id: str,
    scope: str,
    project: str | None,
    memory_type: str,
    summary: str,
    evidence_refs: list[str],
    approved_by: str,
    approved_at: str,
    retrieval_hints: list[str],
    status: str = "active",
) -> dict[str, Any]:
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
