from __future__ import annotations

from pathlib import Path
from typing import Any

from .memory_dreaming import build_candidates_from_facts, extract_atomic_facts, load_events_jsonl, normalize_memory_text


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


def _matches_expected(candidate: dict[str, Any], expected: dict[str, Any]) -> bool:
    candidate_text = normalize_memory_text(str(candidate.get("content") or candidate.get("summary") or ""))
    expected_text = normalize_memory_text(str(expected.get("content") or expected.get("summary") or expected.get("statement") or ""))
    if not candidate_text or not expected_text:
        return False
    type_match = not expected.get("type") or expected.get("type") == candidate.get("type")
    scope_match = not expected.get("scope") or expected.get("scope") == candidate.get("scope")
    text_match = expected_text in candidate_text or candidate_text in expected_text
    return bool(type_match and scope_match and text_match)


def evaluate_labeled_events(path: Path | str, *, project: str | None, mode: str = "rules") -> dict[str, Any]:
    rows = load_events_jsonl(path)
    false_negatives: list[dict[str, Any]] = []
    false_positives: list[dict[str, Any]] = []
    true_positive = 0
    expected_total = 0
    predicted_total = 0

    for row in rows:
        events = _row_events(row)
        expected = _row_expected(row)
        expected_total += len(expected)
        if mode == "rules":
            facts = extract_atomic_facts(events, project=project)
            candidates = build_candidates_from_facts(facts)
        else:
            candidates = []
        predicted_total += len(candidates)

        matched_candidates: set[int] = set()
        for expected_item in expected:
            match_index = next((idx for idx, candidate in enumerate(candidates) if idx not in matched_candidates and _matches_expected(candidate, expected_item)), None)
            if match_index is None:
                false_negatives.append({"expected": expected_item, "events": events})
            else:
                true_positive += 1
                matched_candidates.add(match_index)
        for idx, candidate in enumerate(candidates):
            if idx not in matched_candidates:
                false_positives.append(candidate)

    precision = true_positive / predicted_total if predicted_total else 0.0
    recall = true_positive / expected_total if expected_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "mode": mode,
        "rows": len(rows),
        "expected_total": expected_total,
        "predicted_total": predicted_total,
        "true_positive": true_positive,
        "false_positive_count": len(false_positives),
        "false_negative_count": len(false_negatives),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "false_positives": false_positives[:20],
        "false_negatives": false_negatives[:20],
    }
