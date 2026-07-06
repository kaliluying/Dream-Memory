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


def test_build_review_queue_item_includes_dream_analysis():
    candidate = {
        "id": "mem_1",
        "type": "workflow",
        "scope": "project",
        "project": "/tmp/project",
        "content": "Run focused tests before applying memory changes.",
    }
    analysis = {
        "dream_score": 0.81,
        "suggested_action": "create",
        "reasons": ["high reuse value", "strong evidence"],
        "penalties": [],
        "policy": {"promote_threshold": 0.7},
    }

    item = build_review_queue_item(
        candidate=candidate,
        conflicts=[],
        suggested_action="create",
        quality_signals={"reuse_value": 0.9},
        dream_analysis=analysis,
    )

    assert item["dream_analysis"] == analysis
    assert item["suggested_action"] == "create"
    assert item["quality_signals"] == {"reuse_value": 0.9}


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
