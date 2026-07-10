from __future__ import annotations

from pathlib import Path
from typing import Any

from .memory_agent import agent_extract_memory_candidates
from .memory_dreaming import (
    _text_similarity,
    apply_dream_analysis_to_candidates,
    build_candidates_from_facts,
    extract_atomic_facts,
    load_events_jsonl,
    normalize_memory_text,
)

_EXPECTED_OUTCOMES = {"reviewable", "deferred", "rejected", "none"}


def _row_events(row: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(row.get("events"), list):
        return [event for event in row["events"] if isinstance(event, dict)]
    if isinstance(row.get("event"), dict):
        return [dict(row["event"])]
    return [dict(row)]


def _row_expected(row: dict[str, Any]) -> list[dict[str, Any]]:
    expected = row.get("expected") or row.get("expected_memories") or []
    if isinstance(expected, dict):
        return [expected]
    if isinstance(expected, list):
        return [item for item in expected if isinstance(item, dict)]
    return []


def _row_expected_outcomes(row: dict[str, Any]) -> list[str] | None:
    if "expected_outcomes" not in row:
        return None
    raw = row.get("expected_outcomes")
    if not isinstance(raw, list) or not raw:
        raise ValueError("expected_outcomes must be a non-empty list")
    outcomes = sorted({str(item).strip() for item in raw if str(item).strip()})
    if not outcomes:
        raise ValueError("expected_outcomes must contain at least one outcome")
    invalid = sorted(set(outcomes) - _EXPECTED_OUTCOMES)
    if invalid:
        raise ValueError(f"unsupported expected outcome: {invalid[0]}")
    return outcomes


def _row_label(row: dict[str, Any], row_index: int) -> str:
    return str(row.get("id") or row.get("name") or f"row_{row_index}")


def _character_similarity(left: str, right: str) -> float:
    left_chars = {char for char in normalize_memory_text(left) if char.strip()}
    right_chars = {char for char in normalize_memory_text(right) if char.strip()}
    if not left_chars or not right_chars:
        return 0.0
    return len(left_chars & right_chars) / len(left_chars | right_chars)


def _matches_expected(candidate: dict[str, Any], expected: dict[str, Any]) -> bool:
    candidate_text = normalize_memory_text(str(candidate.get("content") or candidate.get("summary") or ""))
    expected_text = normalize_memory_text(str(expected.get("content") or expected.get("summary") or expected.get("statement") or ""))
    if not candidate_text or not expected_text:
        return False
    type_match = not expected.get("type") or expected.get("type") == candidate.get("type")
    scope_match = not expected.get("scope") or expected.get("scope") == candidate.get("scope")
    text_match = (
        expected_text in candidate_text
        or candidate_text in expected_text
        or _text_similarity(candidate_text, expected_text) >= 0.5
        or _character_similarity(candidate_text, expected_text) >= 0.72
    )
    return bool(type_match and scope_match and text_match)


def _candidate_outcome(candidate: dict[str, Any]) -> str:
    analysis = (
        candidate.get("dream_analysis")
        if isinstance(candidate.get("dream_analysis"), dict)
        else {}
    )
    action = str(analysis.get("suggested_action") or "")
    if action in {"create", "review", "merge"}:
        return "reviewable"
    if action == "needs_more_evidence":
        return "deferred"
    if action == "reject":
        return "rejected"
    raise ValueError(f"unsupported dream action: {action}")


def _scored_candidates(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, list[str]]:
    normalized = []
    for index, candidate in enumerate(candidates, start=1):
        item = dict(candidate)
        item.setdefault("id", f"eval_candidate_{index}")
        normalized.append(item)
    analyzed = apply_dream_analysis_to_candidates(normalized, [])
    outcomes = (
        sorted({_candidate_outcome(candidate) for candidate in analyzed})
        if analyzed
        else ["none"]
    )
    deferred = sum(
        1
        for candidate in analyzed
        if _candidate_outcome(candidate) == "deferred"
    )
    reviewable = [
        candidate
        for candidate in analyzed
        if _candidate_outcome(candidate) == "reviewable"
    ]
    return reviewable, deferred, outcomes


def _extract_candidates(
    events: list[dict[str, Any]],
    *,
    project: str | None,
    mode: str,
    model: Any = "anthropic:claude-sonnet-4-6",
    runtime_config: dict[str, Any] | None = None,
    invoke_model: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if mode == "rules":
        facts = extract_atomic_facts(events, project=project)
        return build_candidates_from_facts(facts), None
    if mode == "ai":
        extraction = agent_extract_memory_candidates(
            events,
            project=project,
            model=model,
            invoke_model=invoke_model,
            runtime_config=runtime_config,
        )
        meta = {
            "dry_run": bool(extraction.get("dry_run", False)),
            "candidate_count": len(extraction.get("candidates", [])),
            "atomic_fact_count": len(extraction.get("atomic_facts", [])),
            "model": extraction.get("model"),
            "model_runtime": extraction.get("model_runtime"),
        }
        for key in ("input_event_count", "prompt_event_count", "filtered_prompt_event_count"):
            if key in extraction:
                meta[key] = extraction[key]
        return list(extraction.get("candidates", [])), meta
    raise ValueError(f"unsupported eval mode: {mode}")


def evaluate_labeled_events(
    path: Path | str,
    *,
    project: str | None,
    mode: str = "rules",
    model: Any = "anthropic:claude-sonnet-4-6",
    runtime_config: dict[str, Any] | None = None,
    invoke_model: bool = True,
    continue_on_error: bool = False,
    max_rows: int | None = None,
    fallback_rules_on_error: bool = False,
    fallback_rules_on_empty: bool = False,
) -> dict[str, Any]:
    rows = load_events_jsonl(path)
    if max_rows is not None:
        rows = rows[:max(0, int(max_rows))]
    false_negatives: list[dict[str, Any]] = []
    false_positives: list[dict[str, Any]] = []
    extraction_summaries: list[dict[str, Any]] = []
    extraction_errors: list[dict[str, Any]] = []
    fallback_count = 0
    fallback_empty_count = 0
    extraction_success_count = 0
    true_positive = 0
    expected_total = 0
    predicted_total = 0
    raw_candidate_total = 0
    scored_candidate_total = 0
    fallback_candidate_total = 0
    deferred_candidate_total = 0
    outcome_checked_rows = 0
    outcome_correct_rows = 0
    outcome_mismatches: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows, start=1):
        row_label = _row_label(row, row_index)
        events = _row_events(row)
        expected = _row_expected(row)
        expected_outcomes = _row_expected_outcomes(row)
        actual_outcomes = ["none"]
        expected_total += len(expected)
        raw_candidate_count = 0
        fallback_candidate_count = 0
        try:
            candidates, extraction_meta = _extract_candidates(
                events,
                project=project,
                mode=mode,
                model=model,
                runtime_config=runtime_config,
                invoke_model=invoke_model,
            )
        except Exception as exc:
            if not continue_on_error and not fallback_rules_on_error:
                raise
            extraction_errors.append({
                "row": row_index,
                "row_id": row_label,
                "error_type": exc.__class__.__name__,
                "error": str(exc),
                "fallback": "rules" if fallback_rules_on_error else None,
            })
            if fallback_rules_on_error:
                candidates, _ = _extract_candidates(events, project=project, mode="rules")
                candidates, deferred_count, actual_outcomes = _scored_candidates(candidates)
                deferred_candidate_total += deferred_count
                fallback_candidate_count = len(candidates)
                fallback_candidate_total += fallback_candidate_count
                extraction_meta = {"fallback": "rules", "candidate_count": len(candidates)}
                fallback_count += 1
            else:
                candidates = []
                extraction_meta = None
        else:
            raw_candidate_count = len(candidates)
            raw_candidate_total += raw_candidate_count
            candidates, deferred_count, actual_outcomes = _scored_candidates(candidates)
            deferred_candidate_total += deferred_count
        if mode == "ai" and fallback_rules_on_empty and not candidates:
            fallback_candidates, _ = _extract_candidates(events, project=project, mode="rules")
            fallback_candidates, deferred_count, fallback_outcomes = _scored_candidates(
                fallback_candidates,
            )
            deferred_candidate_total += deferred_count
            if fallback_candidates:
                fallback_candidate_count = len(fallback_candidates)
                fallback_candidate_total += fallback_candidate_count
                candidates = fallback_candidates
                actual_outcomes = fallback_outcomes
                extraction_meta = {"fallback": "rules_empty_ai", "candidate_count": len(candidates)}
                fallback_count += 1
                fallback_empty_count += 1
        scored_candidate_total += len(candidates)
        predicted_total += len(candidates)
        if extraction_meta is not None:
            if not extraction_meta.get("fallback"):
                extraction_success_count += 1
            extraction_summaries.append({
                "row": row_index,
                "row_id": row_label,
                **extraction_meta,
                "raw_candidate_count": raw_candidate_count,
                "fallback_candidate_count": fallback_candidate_count,
                "scored_candidate_count": len(candidates),
            })

        matched_candidates: set[int] = set()
        for expected_item in expected:
            match_index = next((idx for idx, candidate in enumerate(candidates) if idx not in matched_candidates and _matches_expected(candidate, expected_item)), None)
            if match_index is None:
                false_negatives.append({"row": row_index, "row_id": row_label, "expected": expected_item, "events": events})
            else:
                true_positive += 1
                matched_candidates.add(match_index)
        for idx, candidate in enumerate(candidates):
            if idx not in matched_candidates:
                false_positives.append({"row": row_index, "row_id": row_label, "candidate": candidate})
        if expected_outcomes is not None:
            outcome_checked_rows += 1
            if actual_outcomes == expected_outcomes:
                outcome_correct_rows += 1
            else:
                outcome_mismatches.append({
                    "row": row_index,
                    "row_id": row_label,
                    "expected_outcomes": expected_outcomes,
                    "actual_outcomes": actual_outcomes,
                })

    precision = true_positive / predicted_total if predicted_total else 0.0
    recall = true_positive / expected_total if expected_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "mode": mode,
        "rows": len(rows),
        "expected_total": expected_total,
        "predicted_total": predicted_total,
        "raw_candidate_total": raw_candidate_total,
        "fallback_candidate_total": fallback_candidate_total,
        "scored_candidate_total": scored_candidate_total,
        "deferred_candidate_count": deferred_candidate_total,
        "outcome_checked_rows": outcome_checked_rows,
        "outcome_correct_rows": outcome_correct_rows,
        "outcome_accuracy": round(
            outcome_correct_rows / outcome_checked_rows,
            3,
        ) if outcome_checked_rows else 0.0,
        "outcome_mismatches": outcome_mismatches[:20],
        "true_positive": true_positive,
        "false_positive_count": len(false_positives),
        "false_negative_count": len(false_negatives),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "false_positives": false_positives[:20],
        "false_negatives": false_negatives[:20],
        "extractions": extraction_summaries[:20],
        "extraction_success_count": extraction_success_count,
        "extraction_error_count": len(extraction_errors),
        "fallback_count": fallback_count,
        "fallback_empty_count": fallback_empty_count,
        "extraction_errors": extraction_errors[:20],
    }
