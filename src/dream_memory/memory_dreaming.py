from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .memory_models import build_atomic_fact, build_memory_card, build_review_decision, build_review_queue_item

SENSITIVE_RE = re.compile(
    r"(sk-[a-zA-Z0-9]|api[_-]?key\s*[=:]|access[_-]?token\s*[=:]|refresh[_-]?token\s*[=:]|password\s*[=:]|secret\s*[=:]|cookie\s*[=:]|bearer\s+[a-zA-Z0-9]|-----BEGIN [A-Z ]*PRIVATE KEY-----|AKIA[0-9A-Z]{16}|eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+)",
    re.I,
)
BLOCKED_EVENT_TYPES = {"project_state", "tool_output", "build_log"}
RAW_TRANSCRIPT_RE = re.compile(r"(^|\n)\s*(user|assistant|system)\s*:", re.I)
TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]+|[\u4e00-\u9fff]{2,}")
ONE_OFF_MEMORY_RE = re.compile(
    r"(删除|修改|改为|改成|实现|新增|接入|修复|清理|迁移|跑|测试|生成).{0,24}(页面|组件|按钮|接口|脚本|水印|首页|配置|任务|数据)",
    re.I,
)
DEFAULT_DREAM_PROMOTION_POLICY: dict[str, Any] = {
    "promote_threshold": 0.7,
    "review_threshold": 0.45,
    "reject_one_off": True,
    "require_evidence": True,
    "duplicate_action": "reject",
    "conflict_promote_action": "merge",
}
ACTION_ORDER = ["create", "merge", "needs_more_evidence", "review", "reject"]


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


def normalize_project_path(project: str | None) -> str | None:
    if not project:
        return None
    raw = str(project).strip()
    if not raw:
        return None
    if raw.startswith("/") and not raw.startswith("//"):
        return PurePosixPath(raw).as_posix()
    return str(Path(raw).expanduser().absolute())


def normalize_memory_text(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value).strip().lower())
    text = re.sub(r"[。．.!！?？,，;；:：]+$", "", text)
    return text.strip()


def _tokens(value: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(str(value)) if len(token.strip()) >= 2}


def _text_similarity(left: str, right: str) -> float:
    left_tokens = _tokens(normalize_memory_text(left))
    right_tokens = _tokens(normalize_memory_text(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


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
        event_project = normalize_project_path(str(event.get("project") or project)) if (event.get("project") or project) else None

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
        project = normalize_project_path(str(fact.get("project"))) if fact.get("project") else None
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
            "retrieval_hints": [],
            "quality_reason": "",
        })
        evidence_items = fact.get("evidence") if isinstance(fact.get("evidence"), list) else []
        if evidence_items:
            for evidence in evidence_items:
                if not isinstance(evidence, dict):
                    continue
                candidate["evidence"].append({
                    "event_id": evidence.get("event_id") or evidence.get("id"),
                    "source": evidence.get("source") or fact.get("source"),
                    "session_id": evidence.get("session_id") or fact.get("session_id"),
                    "quote": evidence.get("quote"),
                    "content_hash": _content_hash(content),
                })
        else:
            for ref in fact.get("evidence_refs", []):
                candidate["evidence"].append({
                    "event_id": ref,
                    "source": fact.get("source"),
                    "session_id": fact.get("session_id"),
                    "content_hash": _content_hash(content),
                })
        candidate["tags"] = sorted(set(candidate.get("tags", []) + list(fact.get("tags", []))))
        candidate["retrieval_hints"] = sorted(set(candidate.get("retrieval_hints", []) + [str(item) for item in fact.get("reuse_scenarios", [])]))
        quality_reasons = [candidate.get("quality_reason", "")]
        if fact.get("long_term") is not None:
            quality_reasons.append(f"long_term={bool(fact.get('long_term'))}")
        if fact.get("long_term_reason"):
            quality_reasons.append(str(fact.get("long_term_reason")))
        candidate["quality_reason"] = "; ".join(reason for reason in quality_reasons if reason)
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


def _active_matching_cards(candidate: dict[str, Any], memory_cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidate_scope = candidate.get("scope")
    candidate_project = normalize_project_path(str(candidate.get("project"))) if candidate.get("project") else None
    candidate_type = candidate.get("type")
    matches: list[dict[str, Any]] = []
    for card in memory_cards:
        if card.get("status", "active") != "active":
            continue
        card_project = normalize_project_path(str(card.get("project"))) if card.get("project") else None
        if card.get("scope") == candidate_scope and card_project == candidate_project and card.get("memory_type") == candidate_type:
            matches.append(card)
    return matches


def explain_candidate_quality(candidate: dict[str, Any], memory_cards: list[dict[str, Any]]) -> dict[str, Any]:
    content = str(candidate.get("content") or "")
    normalized_content = normalize_memory_text(content)
    evidence_count = len(candidate.get("evidence", []))
    score = float(candidate.get("score", candidate.get("confidence", 0.0)) or 0.0)
    memory_type = str(candidate.get("type") or "")
    one_off = bool(ONE_OFF_MEMORY_RE.search(content)) or "task" in {str(tag).lower() for tag in candidate.get("tags", [])}
    matching_cards = _active_matching_cards(candidate, memory_cards)
    exact_match = next((card for card in matching_cards if normalize_memory_text(str(card.get("summary") or "")) == normalized_content), None)
    similar_cards = [
        (card, _text_similarity(content, str(card.get("summary") or "")))
        for card in matching_cards
        if normalize_memory_text(str(card.get("summary") or "")) != normalized_content
    ]
    similar_cards = [(card, similarity) for card, similarity in similar_cards if similarity >= 0.18]
    best_similar = max(similar_cards, key=lambda item: item[1], default=(None, 0.0))
    matched_card = exact_match or best_similar[0]
    durable_types = {"preference", "decision", "workflow", "pitfall", "product_direction", "rejected_option"}
    stability = 0.25
    if memory_type in durable_types:
        stability += 0.3
    if evidence_count >= 2:
        stability += 0.2
    if score >= 0.8:
        stability += 0.15
    if one_off:
        stability -= 0.35
    reuse_value = 0.25
    if memory_type in durable_types:
        reuse_value += 0.3
    if candidate.get("scope") in {"user", "global", "project"}:
        reuse_value += 0.15
    if candidate.get("tags"):
        reuse_value += 0.1
    if one_off:
        reuse_value -= 0.3
    return {
        "stability": max(0.0, min(1.0, round(stability, 3))),
        "reuse_value": max(0.0, min(1.0, round(reuse_value, 3))),
        "evidence_strength": max(0.0, min(1.0, round(min(evidence_count, 4) / 4, 3))),
        "one_off_task": one_off,
        "duplicate": exact_match is not None,
        "similarity": round(1.0 if exact_match else best_similar[1], 3),
        "matched_memory_id": matched_card.get("id") if isinstance(matched_card, dict) else None,
    }


def _policy_value(policy: dict[str, Any], key: str, fallback: Any) -> Any:
    value = policy.get(key, fallback)
    if isinstance(fallback, bool):
        if isinstance(value, str):
            return value.strip().lower() not in {"0", "false", "no", "off", ""}
        return bool(value)
    if isinstance(fallback, float):
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback
    if isinstance(fallback, str):
        return str(value or fallback)
    return value


def _quality_float(quality_signals: dict[str, Any], key: str) -> float:
    try:
        return max(0.0, min(1.0, float(quality_signals.get(key, 0.0) or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _candidate_score(candidate: dict[str, Any]) -> float:
    try:
        return max(0.0, min(1.0, float(candidate.get("score", candidate.get("confidence", 0.0)) or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def analyze_dream_candidate(
    candidate: dict[str, Any],
    *,
    quality_signals: dict[str, Any],
    conflicts: list[dict[str, Any]],
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active_policy = dict(DEFAULT_DREAM_PROMOTION_POLICY)
    if policy:
        active_policy.update(policy)

    promote_threshold = _policy_value(active_policy, "promote_threshold", 0.7)
    review_threshold = _policy_value(active_policy, "review_threshold", 0.45)
    reject_one_off = _policy_value(active_policy, "reject_one_off", True)
    require_evidence = _policy_value(active_policy, "require_evidence", True)
    duplicate_action = _policy_value(active_policy, "duplicate_action", "reject")
    conflict_action = _policy_value(active_policy, "conflict_promote_action", "merge")

    stability = _quality_float(quality_signals, "stability")
    reuse_value = _quality_float(quality_signals, "reuse_value")
    evidence_strength = _quality_float(quality_signals, "evidence_strength")
    base_score = _candidate_score(candidate)
    dream_score = round(
        stability * 0.4
        + reuse_value * 0.4
        + evidence_strength * 0.1
        + base_score * 0.1,
        3,
    )

    reasons: list[str] = []
    penalties: list[str] = []
    if stability >= 0.7:
        reasons.append("high stability")
    if reuse_value >= 0.7:
        reasons.append("high reuse value")
    if evidence_strength >= 0.5:
        reasons.append("strong evidence")
    if conflicts:
        reasons.append("conflicts with existing memory")
    similarity = _quality_float(quality_signals, "similarity")
    if quality_signals.get("matched_memory_id") and similarity > 0:
        reasons.append("similar existing memory")

    one_off = bool(quality_signals.get("one_off_task"))
    duplicate = bool(quality_signals.get("duplicate"))
    if duplicate:
        penalties.append("duplicate")
    if one_off:
        penalties.append("one-off task")
        dream_score = min(dream_score, max(0.0, review_threshold - 0.01))
    if require_evidence and evidence_strength <= 0:
        penalties.append("missing evidence")

    if duplicate:
        suggested_action = duplicate_action
    elif reject_one_off and one_off:
        suggested_action = "reject"
    elif require_evidence and evidence_strength <= 0:
        suggested_action = "needs_more_evidence"
    elif (conflicts or quality_signals.get("matched_memory_id")) and dream_score >= review_threshold:
        suggested_action = conflict_action
    elif dream_score >= promote_threshold:
        suggested_action = "create"
    elif dream_score >= review_threshold:
        suggested_action = "review"
    else:
        suggested_action = "reject"

    return {
        "dream_score": dream_score,
        "suggested_action": suggested_action,
        "reasons": reasons,
        "penalties": penalties,
        "matched_memory_id": quality_signals.get("matched_memory_id"),
        "policy": {
            "promote_threshold": promote_threshold,
            "review_threshold": review_threshold,
            "reject_one_off": reject_one_off,
            "require_evidence": require_evidence,
            "duplicate_action": duplicate_action,
            "conflict_promote_action": conflict_action,
        },
    }


def suggest_review_action(candidate: dict[str, Any], quality_signals: dict[str, Any], conflicts: list[dict[str, Any]]) -> str:
    if quality_signals.get("duplicate"):
        return "reject"
    if quality_signals.get("one_off_task"):
        return "reject"
    if quality_signals.get("evidence_strength", 0) <= 0:
        return "needs_more_evidence"
    if conflicts:
        return "replace" if candidate.get("status") == "promote" and quality_signals.get("evidence_strength", 0) >= 0.5 else "merge"
    if quality_signals.get("similarity", 0) >= 0.18 and quality_signals.get("matched_memory_id"):
        return "merge"
    if candidate.get("status") == "reject":
        return "reject"
    return "create"


def _signals_with_conflict_match(candidate: dict[str, Any], quality_signals: dict[str, Any], conflicts: list[dict[str, Any]]) -> dict[str, Any]:
    if not conflicts or quality_signals.get("duplicate"):
        return quality_signals
    signals = dict(quality_signals)
    candidate_content = str(candidate.get("content") or "")
    selected = max(conflicts, key=lambda item: _text_similarity(str(item.get("summary") or ""), candidate_content))
    signals["matched_memory_id"] = selected.get("memory_id")
    signals["similarity"] = round(_text_similarity(str(selected.get("summary") or ""), candidate_content), 3)
    return signals


def _status_from_dream_action(action: str) -> str:
    if action in {"create", "merge"}:
        return "promote"
    if action in {"review", "needs_more_evidence"}:
        return "review"
    return "reject"


def apply_dream_analysis_to_candidates(candidates: list[dict[str, Any]], memory_cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conflict_map = detect_candidate_conflicts(candidates, memory_cards)
    analyzed_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        conflicts = conflict_map.get(str(candidate["id"]), [])
        quality_signals = _signals_with_conflict_match(candidate, explain_candidate_quality(candidate, memory_cards), conflicts)
        dream_analysis = analyze_dream_candidate(
            candidate,
            quality_signals=quality_signals,
            conflicts=conflicts,
        )
        analyzed = dict(candidate)
        analyzed["quality_signals"] = quality_signals
        analyzed["dream_analysis"] = dream_analysis
        analyzed["status"] = _status_from_dream_action(str(dream_analysis["suggested_action"]))
        analyzed_candidates.append(analyzed)
    return analyzed_candidates


def build_review_queue(candidates: list[dict[str, Any]], memory_cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conflict_map = detect_candidate_conflicts(candidates, memory_cards)
    queue: list[dict[str, Any]] = []
    for candidate in candidates:
        conflicts = conflict_map.get(str(candidate["id"]), [])
        quality_signals = _signals_with_conflict_match(candidate, explain_candidate_quality(candidate, memory_cards), conflicts)
        dream_analysis = analyze_dream_candidate(
            candidate,
            quality_signals=quality_signals,
            conflicts=conflicts,
        )
        queue.append(build_review_queue_item(
            candidate=candidate,
            conflicts=conflicts,
            suggested_action=str(dream_analysis["suggested_action"]),
            quality_signals=quality_signals,
            dream_analysis=dream_analysis,
        ))
    return queue


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
    project = normalize_project_path(str(candidate.get("project"))) if candidate.get("project") else None
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


def build_agent_context(memory_cards: list[dict[str, Any]], *, project: str | None, limit: int = 12, task: str | None = None) -> dict[str, Any]:
    normalized_project = normalize_project_path(project)
    task_tokens = _tokens(task or "")

    def relevance(card: dict[str, Any]) -> float:
        if not task_tokens:
            return 0.0
        searchable = " ".join(
            str(value)
            for value in [
                card.get("summary"),
                card.get("memory_type"),
                " ".join(str(item) for item in card.get("retrieval_hints", []) if item),
                " ".join(str(item) for item in card.get("tags", []) if item),
            ]
        )
        card_tokens = _tokens(searchable)
        if not card_tokens:
            return 0.0
        return len(task_tokens & card_tokens) / len(task_tokens)

    def scope_rank(card: dict[str, Any]) -> int:
        card_project = normalize_project_path(str(card.get("project"))) if card.get("project") else None
        if normalized_project and card.get("scope") == "project" and card_project == normalized_project:
            return 0
        if card.get("scope") == "user":
            return 1
        if card.get("scope") == "global":
            return 2
        if card.get("scope") == "session":
            return 3
        return 4

    def rank(card: dict[str, Any]) -> tuple[float, int, int, str]:
        card_relevance = relevance(card)
        default_rank = 0 if task_tokens and card_relevance == 0 and card.get("scope") in {"user", "global"} else 1
        return (-card_relevance, default_rank, scope_rank(card), str(card.get("summary")))

    filtered = []
    for card in memory_cards:
        if card.get("status") != "active":
            continue
        if card.get("scope") == "project":
            card_project = normalize_project_path(str(card.get("project"))) if card.get("project") else None
            if not normalized_project or card_project != normalized_project:
                continue
            card = dict(card)
            card["project"] = card_project
        filtered.append(card)
    ranked = sorted(filtered, key=rank)[:limit]
    payload = {"project": normalized_project, "count": len(ranked), "items": ranked}
    if task:
        payload["task"] = task
    return payload


def render_context_markdown(context: dict[str, Any]) -> str:
    lines = ["## Relevant Memory", ""]
    for item in context.get("items", []):
        prefix = str(item.get("scope"))
        if item.get("project"):
            prefix = f"{prefix}: {item['project']}"
        lines.append(f"- **{prefix} / {item.get('memory_type')}**: {item.get('summary')}")
    return "\n".join(lines) + "\n"

def _analysis_for_report(candidate: dict[str, Any]) -> dict[str, Any]:
    if isinstance(candidate.get("dream_analysis"), dict):
        return dict(candidate["dream_analysis"])
    quality_signals = explain_candidate_quality(candidate, [])
    return analyze_dream_candidate(candidate, quality_signals=quality_signals, conflicts=[])


def _action_heading(action: str) -> str:
    return {
        "create": "Create",
        "merge": "Merge",
        "needs_more_evidence": "Needs More Evidence",
        "review": "Review",
        "reject": "Reject",
    }.get(action, action.replace("_", " ").title())


def _candidate_report_line(candidate: dict[str, Any], analysis: dict[str, Any]) -> str:
    reasons = ", ".join(str(item) for item in analysis.get("reasons", [])) or "none"
    penalties = ", ".join(str(item) for item in analysis.get("penalties", [])) or "none"
    return (
        f"- ({candidate.get('type')}, dream_score={analysis.get('dream_score')}, "
        f"action={analysis.get('suggested_action')}) {candidate.get('content')} "
        f"[reasons: {reasons}; penalties: {penalties}]"
    )


def _render_dreams(events: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> str:
    analyzed: list[tuple[dict[str, Any], dict[str, Any]]] = [
        (candidate, _analysis_for_report(candidate))
        for candidate in candidates
    ]
    counts = {action: 0 for action in ACTION_ORDER}
    for _, analysis in analyzed:
        action = str(analysis.get("suggested_action") or "review")
        counts[action] = counts.get(action, 0) + 1
    policy = DEFAULT_DREAM_PROMOTION_POLICY
    lines = [
        "# DREAMS.md",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Sweep Summary",
        "",
        f"- Events scanned: {len(events)}",
        f"- Candidates: {len(candidates)}",
        "",
        "## Promotion Policy",
        "",
        f"- Promote threshold: {policy['promote_threshold']}",
        f"- Review threshold: {policy['review_threshold']}",
        f"- Reject one-off tasks: {str(policy['reject_one_off']).lower()}",
        f"- Require evidence: {str(policy['require_evidence']).lower()}",
        "",
        "## Action Summary",
        "",
    ]
    for action in ACTION_ORDER:
        lines.append(f"- {_action_heading(action)}: {counts.get(action, 0)}")

    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {action: [] for action in ACTION_ORDER}
    for candidate, analysis in analyzed:
        action = str(analysis.get("suggested_action") or "review")
        grouped.setdefault(action, []).append((candidate, analysis))

    for action in ACTION_ORDER:
        lines.extend(["", f"## {_action_heading(action)}", ""])
        rows = sorted(
            grouped.get(action, []),
            key=lambda item: (
                -float(item[1].get("dream_score", 0.0) or 0.0),
                str(item[0].get("type") or ""),
                str(item[0].get("content") or ""),
            ),
        )
        if not rows:
            lines.append("- None")
            continue
        for candidate, analysis in rows[:30]:
            lines.append(_candidate_report_line(candidate, analysis))
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
    candidates = apply_dream_analysis_to_candidates(candidates, [])
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
