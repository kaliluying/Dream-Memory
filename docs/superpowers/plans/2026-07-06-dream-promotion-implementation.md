# Dream Promotion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explainable dream promotion layer that scores memory candidates, records reasons and penalties, and renders those explanations in review queues and `DREAMS.md`.

**Architecture:** Keep the feature centered in `memory_dreaming.py`, where candidate quality and review actions already live. Extend `memory_models.build_review_queue_item()` to carry optional `dream_analysis` metadata while preserving existing queue fields. Update the report renderer to compute and display the same analysis so CLI and run artifacts become auditable without changing the manual review gate.

**Tech Stack:** Python 3.11, dataclass-free dict models, JSONL artifacts, pytest/unittest, existing argparse CLI and FastAPI payloads.

---

## Scope Check

The spec covers one coherent subsystem: explainable candidate promotion. It does not include background scheduling, three-stage sleep orchestration, provider changes, vector recall, or automatic memory writes. This plan implements only scoring, queue metadata, and reporting.

## File Structure

- Modify: `src/dream_memory/memory_models.py`
  - Responsibility: JSON-compatible builders for queue items, review decisions, and memory cards.
  - Change: add optional `dream_analysis` to `build_review_queue_item()`.

- Modify: `src/dream_memory/memory_dreaming.py`
  - Responsibility: candidate extraction, quality signals, promotion actions, review queues, markdown reports, context rendering.
  - Change: add promotion policy helpers, attach `dream_analysis`, and render explainable `DREAMS.md`.

- Modify: `tests/test_memory_models.py`
  - Responsibility: builder-level schema tests.
  - Change: verify queue item can include `dream_analysis`.

- Modify: `tests/test_memory_dreaming.py`
  - Responsibility: candidate quality, review queue, and artifact behavior.
  - Change: verify scoring actions, queue metadata, and report contents.

No new runtime module is required. Splitting promotion policy into a new file would be premature because the current logic is tightly coupled to existing quality signals in `memory_dreaming.py`.

## Baseline Note

Before this plan starts, the full suite is known to have one unrelated existing failure:

```text
tests/test_memory_agent.py::MemoryAgentTests::test_agent_extract_aggregates_atomic_facts_into_candidates
KeyError: 'atomic_facts'
```

Use targeted tests for this feature while implementing. Run the full suite at the end and report whether this pre-existing failure remains.

---

### Task 1: Extend Review Queue Item Schema

**Files:**
- Modify: `src/dream_memory/memory_models.py`
- Test: `tests/test_memory_models.py`

- [ ] **Step 1: Write the failing schema test**

Add this test to `tests/test_memory_models.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
uv run --with pytest pytest tests/test_memory_models.py::test_build_review_queue_item_includes_dream_analysis -v
```

Expected: FAIL with a `TypeError` mentioning unexpected keyword argument `dream_analysis`.

- [ ] **Step 3: Update the builder signature**

Change `build_review_queue_item()` in `src/dream_memory/memory_models.py` to:

```python
def build_review_queue_item(
    *,
    candidate: dict[str, Any],
    conflicts: list[dict[str, Any]],
    suggested_action: str = "review",
    quality_signals: dict[str, Any] | None = None,
    dream_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    item = {
        "candidate_id": candidate["id"],
        "status": "pending",
        "suggested_action": suggested_action,
        "candidate": candidate,
        "conflicts": conflicts,
        "quality_signals": quality_signals or {},
        "created_at": _now_iso(),
    }
    if dream_analysis is not None:
        item["dream_analysis"] = dream_analysis
    return item
```

This preserves old queue shape when callers do not pass `dream_analysis`.

- [ ] **Step 4: Run schema tests**

Run:

```bash
uv run --with pytest pytest tests/test_memory_models.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dream_memory/memory_models.py tests/test_memory_models.py
git commit -m "feat: allow dream analysis in review queue items"
```

---

### Task 2: Add Dream Candidate Analysis

**Files:**
- Modify: `src/dream_memory/memory_dreaming.py`
- Test: `tests/test_memory_dreaming.py`

- [ ] **Step 1: Add failing tests for promotion decisions**

Add imports in `tests/test_memory_dreaming.py`:

```python
from dream_memory.memory_dreaming import (
    analyze_dream_candidate,
    build_agent_context,
    build_candidates_from_facts,
    build_review_queue,
    detect_candidate_conflicts,
    dream_from_events,
    extract_atomic_facts,
    load_events_jsonl,
    normalize_project_path,
)
```

If the file already imports these names in a grouped import, add only `analyze_dream_candidate`.

Add these tests to `MemoryDreamingTests`:

```python
def test_analyze_dream_candidate_rejects_one_off_task(self):
    candidate = {
        "id": "mem_task",
        "type": "requirement",
        "scope": "project",
        "project": "/tmp/project",
        "content": "删除首页水印按钮",
        "score": 0.9,
        "evidence": [{"event_id": "event_1"}],
    }
    analysis = analyze_dream_candidate(
        candidate,
        quality_signals={
            "stability": 0.2,
            "reuse_value": 0.1,
            "evidence_strength": 0.5,
            "one_off_task": True,
            "duplicate": False,
            "similarity": 0.0,
            "matched_memory_id": None,
        },
        conflicts=[],
    )

    self.assertEqual(analysis["suggested_action"], "reject")
    self.assertIn("one-off task", analysis["penalties"])
    self.assertLess(analysis["dream_score"], 0.45)

def test_analyze_dream_candidate_requires_evidence(self):
    candidate = {
        "id": "mem_no_evidence",
        "type": "workflow",
        "scope": "project",
        "project": "/tmp/project",
        "content": "Run targeted tests before changing memory logic.",
        "score": 0.9,
        "evidence": [],
    }
    analysis = analyze_dream_candidate(
        candidate,
        quality_signals={
            "stability": 0.9,
            "reuse_value": 0.8,
            "evidence_strength": 0.0,
            "one_off_task": False,
            "duplicate": False,
            "similarity": 0.0,
            "matched_memory_id": None,
        },
        conflicts=[],
    )

    self.assertEqual(analysis["suggested_action"], "needs_more_evidence")
    self.assertIn("missing evidence", analysis["penalties"])

def test_analyze_dream_candidate_creates_durable_memory(self):
    candidate = {
        "id": "mem_workflow",
        "type": "workflow",
        "scope": "project",
        "project": "/tmp/project",
        "content": "Run targeted tests before changing memory logic.",
        "score": 0.88,
        "evidence": [{"event_id": "event_1"}, {"event_id": "event_2"}],
    }
    analysis = analyze_dream_candidate(
        candidate,
        quality_signals={
            "stability": 0.9,
            "reuse_value": 0.85,
            "evidence_strength": 0.75,
            "one_off_task": False,
            "duplicate": False,
            "similarity": 0.0,
            "matched_memory_id": None,
        },
        conflicts=[],
    )

    self.assertEqual(analysis["suggested_action"], "create")
    self.assertGreaterEqual(analysis["dream_score"], 0.7)
    self.assertIn("high stability", analysis["reasons"])
    self.assertIn("high reuse value", analysis["reasons"])

def test_analyze_dream_candidate_merges_similar_memory(self):
    candidate = {
        "id": "mem_merge",
        "type": "workflow",
        "scope": "project",
        "project": "/tmp/project",
        "content": "Run targeted tests before memory changes.",
        "score": 0.8,
        "evidence": [{"event_id": "event_1"}],
    }
    analysis = analyze_dream_candidate(
        candidate,
        quality_signals={
            "stability": 0.8,
            "reuse_value": 0.8,
            "evidence_strength": 0.5,
            "one_off_task": False,
            "duplicate": False,
            "similarity": 0.42,
            "matched_memory_id": "mem_existing",
        },
        conflicts=[],
    )

    self.assertEqual(analysis["suggested_action"], "merge")
    self.assertEqual(analysis["matched_memory_id"], "mem_existing")
    self.assertIn("similar existing memory", analysis["reasons"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
uv run --with pytest pytest tests/test_memory_dreaming.py::MemoryDreamingTests::test_analyze_dream_candidate_rejects_one_off_task tests/test_memory_dreaming.py::MemoryDreamingTests::test_analyze_dream_candidate_requires_evidence tests/test_memory_dreaming.py::MemoryDreamingTests::test_analyze_dream_candidate_creates_durable_memory tests/test_memory_dreaming.py::MemoryDreamingTests::test_analyze_dream_candidate_merges_similar_memory -v
```

Expected: FAIL with `ImportError` or `AttributeError` because `analyze_dream_candidate` does not exist.

- [ ] **Step 3: Add policy and helper functions**

Add this near the existing regex constants in `src/dream_memory/memory_dreaming.py`:

```python
DEFAULT_DREAM_PROMOTION_POLICY: dict[str, Any] = {
    "promote_threshold": 0.7,
    "review_threshold": 0.45,
    "reject_one_off": True,
    "require_evidence": True,
    "duplicate_action": "reject",
    "conflict_promote_action": "merge",
}
```

Add these helper functions before `suggest_review_action()`:

```python
def _policy_value(policy: dict[str, Any], key: str, fallback: Any) -> Any:
    value = policy.get(key, fallback)
    if isinstance(fallback, bool):
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
```

Add the analyzer after those helpers:

```python
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
        stability * 0.35
        + reuse_value * 0.35
        + evidence_strength * 0.25
        + base_score * 0.05,
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
    if quality_signals.get("matched_memory_id") and float(quality_signals.get("similarity", 0.0) or 0.0) > 0:
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
```

- [ ] **Step 4: Run analyzer tests**

Run:

```bash
uv run --with pytest pytest tests/test_memory_dreaming.py::MemoryDreamingTests::test_analyze_dream_candidate_rejects_one_off_task tests/test_memory_dreaming.py::MemoryDreamingTests::test_analyze_dream_candidate_requires_evidence tests/test_memory_dreaming.py::MemoryDreamingTests::test_analyze_dream_candidate_creates_durable_memory tests/test_memory_dreaming.py::MemoryDreamingTests::test_analyze_dream_candidate_merges_similar_memory -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/dream_memory/memory_dreaming.py tests/test_memory_dreaming.py
git commit -m "feat: analyze dream promotion candidates"
```

---

### Task 3: Attach Dream Analysis To Review Queue

**Files:**
- Modify: `src/dream_memory/memory_dreaming.py`
- Test: `tests/test_memory_dreaming.py`

- [ ] **Step 1: Write failing review queue tests**

Add this test to `MemoryDreamingTests`:

```python
def test_build_review_queue_includes_dream_analysis(self):
    candidates = [
        {
            "id": "mem_workflow",
            "type": "workflow",
            "scope": "project",
            "project": "/tmp/project",
            "content": "Run targeted tests before memory changes.",
            "score": 0.9,
            "status": "promote",
            "tags": ["workflow"],
            "evidence": [{"event_id": "event_1"}, {"event_id": "event_2"}],
        }
    ]

    queue = build_review_queue(candidates, [])

    self.assertEqual(len(queue), 1)
    self.assertIn("dream_analysis", queue[0])
    self.assertEqual(queue[0]["suggested_action"], queue[0]["dream_analysis"]["suggested_action"])
    self.assertIn(queue[0]["suggested_action"], {"create", "review"})
    self.assertGreater(queue[0]["dream_analysis"]["dream_score"], 0)

def test_build_review_queue_uses_dream_analysis_for_one_off_reject(self):
    candidates = [
        {
            "id": "mem_task",
            "type": "requirement",
            "scope": "project",
            "project": "/tmp/project",
            "content": "删除首页水印按钮",
            "score": 0.95,
            "status": "promote",
            "tags": ["task"],
            "evidence": [{"event_id": "event_1"}],
        }
    ]

    queue = build_review_queue(candidates, [])

    self.assertEqual(queue[0]["suggested_action"], "reject")
    self.assertEqual(queue[0]["dream_analysis"]["suggested_action"], "reject")
    self.assertIn("one-off task", queue[0]["dream_analysis"]["penalties"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run --with pytest pytest tests/test_memory_dreaming.py::MemoryDreamingTests::test_build_review_queue_includes_dream_analysis tests/test_memory_dreaming.py::MemoryDreamingTests::test_build_review_queue_uses_dream_analysis_for_one_off_reject -v
```

Expected: FAIL because `dream_analysis` is missing or `suggested_action` still comes from the old helper.

- [ ] **Step 3: Update `build_review_queue()`**

Replace the body of `build_review_queue()` in `src/dream_memory/memory_dreaming.py` with:

```python
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
```

Do not remove `suggest_review_action()` yet. Keeping it avoids unnecessary churn for callers or tests that may still import it.

- [ ] **Step 4: Run queue tests**

Run:

```bash
uv run --with pytest pytest tests/test_memory_dreaming.py::MemoryDreamingTests::test_build_review_queue_includes_dream_analysis tests/test_memory_dreaming.py::MemoryDreamingTests::test_build_review_queue_uses_dream_analysis_for_one_off_reject -v
```

Expected: PASS.

- [ ] **Step 5: Run related review tests**

Run:

```bash
uv run --with pytest pytest tests/test_memory_dreaming.py tests/test_memory_review_web.py -v
```

Expected: PASS for these files. If Web tests assert exact queue shapes, update those assertions to allow the new `dream_analysis` field while preserving old fields.

- [ ] **Step 6: Commit**

```bash
git add src/dream_memory/memory_dreaming.py tests/test_memory_dreaming.py tests/test_memory_review_web.py
git commit -m "feat: attach dream analysis to review queue"
```

---

### Task 4: Render Explainable DREAMS.md

**Files:**
- Modify: `src/dream_memory/memory_dreaming.py`
- Test: `tests/test_memory_dreaming.py`

- [ ] **Step 1: Write failing report test**

Add this test to `MemoryDreamingTests`:

```python
def test_dream_from_events_writes_explainable_dream_report(self):
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / ".dream-memory"
        candidates = [
            {
                "id": "mem_workflow",
                "type": "workflow",
                "scope": "project",
                "project": normalize_project_path("/tmp/project"),
                "content": "Run targeted tests before memory changes.",
                "score": 0.9,
                "status": "promote",
                "tags": ["workflow"],
                "evidence": [{"event_id": "event_1"}, {"event_id": "event_2"}],
            },
            {
                "id": "mem_task",
                "type": "requirement",
                "scope": "project",
                "project": normalize_project_path("/tmp/project"),
                "content": "删除首页水印按钮",
                "score": 0.95,
                "status": "promote",
                "tags": ["task"],
                "evidence": [{"event_id": "event_3"}],
            },
        ]

        result = dream_from_events(
            [{"event_id": "event_1", "source": "codex", "role": "user", "content": "memory input"}],
            project="/tmp/project",
            output_dir=output_dir,
            agent_candidates=candidates,
            agent_mode=True,
        )

        report = Path(result.dreams_path).read_text(encoding="utf-8")
        self.assertIn("## Promotion Policy", report)
        self.assertIn("## Action Summary", report)
        self.assertIn("## Create", report)
        self.assertIn("## Reject", report)
        self.assertIn("dream_score=", report)
        self.assertIn("reasons:", report)
        self.assertIn("penalties:", report)
        self.assertIn("one-off task", report)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
uv run --with pytest pytest tests/test_memory_dreaming.py::MemoryDreamingTests::test_dream_from_events_writes_explainable_dream_report -v
```

Expected: FAIL because current `DREAMS.md` lacks promotion policy, action groups, and analysis text.

- [ ] **Step 3: Add report helpers**

Add these helpers near `_render_dreams()` in `src/dream_memory/memory_dreaming.py`:

```python
ACTION_ORDER = ["create", "merge", "needs_more_evidence", "review", "reject"]


def _analysis_for_report(candidate: dict[str, Any]) -> dict[str, Any]:
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
```

- [ ] **Step 4: Replace `_render_dreams()`**

Replace `_render_dreams()` with:

```python
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
```

- [ ] **Step 5: Run report test**

Run:

```bash
uv run --with pytest pytest tests/test_memory_dreaming.py::MemoryDreamingTests::test_dream_from_events_writes_explainable_dream_report -v
```

Expected: PASS.

- [ ] **Step 6: Run dreaming tests**

Run:

```bash
uv run --with pytest pytest tests/test_memory_dreaming.py -v
```

Expected: PASS. If older assertions expect exact section names like `## Promoted Candidates`, update them to assert the new stable sections: `## Action Summary`, `## Create`, and `## Review`.

- [ ] **Step 7: Commit**

```bash
git add src/dream_memory/memory_dreaming.py tests/test_memory_dreaming.py
git commit -m "feat: render explainable dream reports"
```

---

### Task 5: CLI And Web Compatibility

**Files:**
- Modify: `tests/test_memory_cli.py`
- Modify: `tests/test_memory_review_web.py`
- Modify only if tests expose a real compatibility issue: `src/dream_memory/memory_cli.py`, `src/dream_memory/web.py`

- [ ] **Step 1: Add CLI compatibility assertion**

Find the CLI test that runs `dream --mode rules` and reads `DREAMS.md`. Add these assertions after the report is read:

```python
self.assertIn("## Promotion Policy", dreams)
self.assertIn("## Action Summary", dreams)
self.assertIn("dream_score=", dreams)
```

If the test does not currently read `DREAMS.md`, add:

```python
dreams = (output_dir / "DREAMS.md").read_text(encoding="utf-8")
self.assertIn("## Promotion Policy", dreams)
self.assertIn("## Action Summary", dreams)
self.assertIn("dream_score=", dreams)
```

- [ ] **Step 2: Add Web queue payload assertion**

In `tests/test_memory_review_web.py`, find a test that calls `/api/memory/runs/{run_id}/review-queue`. Add:

```python
items = response.json()["items"]
self.assertIn("dream_analysis", items[0])
self.assertEqual(items[0]["suggested_action"], items[0]["dream_analysis"]["suggested_action"])
```

If the existing fixture builds queue items manually without `dream_analysis`, create the queue through `build_review_queue()` in the test fixture so the endpoint exercises the new payload.

- [ ] **Step 3: Run compatibility tests**

Run:

```bash
uv run --with pytest pytest tests/test_memory_cli.py tests/test_memory_review_web.py -v
```

Expected: PASS. If a Web route simply returns JSONL records, no production change should be needed.

- [ ] **Step 4: Fix only real compatibility issues**

If tests fail because a manually created fixture lacks `dream_analysis`, update the fixture instead of production code:

```python
queue = build_review_queue(candidates, [])
write_jsonl_records(queue, review_queue_path)
```

If tests fail because production code strips unknown fields, change that production code to preserve dict rows exactly as loaded.

- [ ] **Step 5: Run compatibility tests again**

Run:

```bash
uv run --with pytest pytest tests/test_memory_cli.py tests/test_memory_review_web.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/test_memory_cli.py tests/test_memory_review_web.py src/dream_memory/memory_cli.py src/dream_memory/web.py
git commit -m "test: cover dream analysis compatibility"
```

If `src/dream_memory/memory_cli.py` and `src/dream_memory/web.py` were not modified, omit them from `git add`.

---

### Task 6: Final Verification And Cleanup

**Files:**
- Inspect: `src/dream_memory/memory_dreaming.py`
- Inspect: `src/dream_memory/memory_models.py`
- Inspect: `tests/test_memory_dreaming.py`
- Inspect: `tests/test_memory_models.py`
- Inspect: `tests/test_memory_cli.py`
- Inspect: `tests/test_memory_review_web.py`

- [ ] **Step 1: Run targeted feature tests**

Run:

```bash
uv run --with pytest pytest tests/test_memory_models.py tests/test_memory_dreaming.py tests/test_memory_cli.py tests/test_memory_review_web.py -v
```

Expected: PASS.

- [ ] **Step 2: Run full test suite**

Run:

```bash
uv run --with pytest pytest -q
```

Expected: Either full PASS or the known pre-existing `atomic_facts` failure remains. Do not hide that failure in the final report.

- [ ] **Step 3: Inspect git diff**

Run:

```bash
git diff -- src/dream_memory/memory_models.py src/dream_memory/memory_dreaming.py tests/test_memory_models.py tests/test_memory_dreaming.py tests/test_memory_cli.py tests/test_memory_review_web.py
```

Expected: Diff only contains dream promotion schema, analyzer, report rendering, and tests. No unrelated formatting churn.

- [ ] **Step 4: Commit final cleanup if needed**

If Step 3 shows small cleanup edits after the task commits, commit them:

```bash
git add src/dream_memory/memory_models.py src/dream_memory/memory_dreaming.py tests/test_memory_models.py tests/test_memory_dreaming.py tests/test_memory_cli.py tests/test_memory_review_web.py
git commit -m "chore: finalize dream promotion implementation"
```

Skip this commit if there are no cleanup edits.

- [ ] **Step 5: Final implementation summary**

Report:

```text
Implemented explainable dream promotion.
Changed files:
- src/dream_memory/memory_models.py
- src/dream_memory/memory_dreaming.py
- tests/test_memory_models.py
- tests/test_memory_dreaming.py
- tests/test_memory_cli.py
- tests/test_memory_review_web.py

Verification:
- Targeted feature tests: <PASS/FAIL>
- Full suite: <PASS/FAIL and known pre-existing failure if present>
```

---

## Self-Review

Spec coverage:

- Explainable score: Task 2.
- Reasons and penalties: Task 2 and Task 4.
- Review queue metadata: Task 1 and Task 3.
- Better `DREAMS.md`: Task 4.
- Compatibility with CLI/Web/API: Task 5.
- Manual review preserved: Tasks do not change `apply_reviewed_memory()` or auto-apply behavior.
- No scheduler, no three-stage state machine, no vector recall: no task introduces these.

Placeholder scan:

- The plan contains no `TBD`, no open implementation blanks, and no instruction to "add appropriate handling" without code.

Type consistency:

- The new field is consistently named `dream_analysis`.
- The new function is consistently named `analyze_dream_candidate()`.
- The score field is consistently named `dream_score`.
- The action field remains `suggested_action` for backward compatibility.
