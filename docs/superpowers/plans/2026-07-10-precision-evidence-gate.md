# Precision-First Evidence Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent weak or duplicated evidence from reaching human memory review, while preserving a one-event exception for explicit long-term instructions and forbidding automatic creation or merging of formal memory.

**Architecture:** Reuse the existing candidate aggregation and dream-analysis pipeline. Count unique evidence event IDs in `memory_dreaming.py`, apply a fixed `2`-event gate before score-based promotion, filter deferred candidates from the review queue, apply the same downstream analysis in evaluation, and make automatic review manual-only for create/review/merge actions.

**Tech Stack:** Python 3.11+, standard library, `unittest`, JSONL artifacts, `uv`.

## Global Constraints

- Use `uv` for every Python command.
- PowerShell commands must set UTF-8 output explicitly.
- Add no dependency, database, background process, or new configuration option.
- Ordinary candidates require two unique non-empty `event_id` values.
- Explicit instructions require one unique non-empty `event_id`.
- Model confidence must never bypass the evidence gate.
- Candidate and dream artifacts retain deferred candidates.
- The review queue contains only `create`, `review`, or `merge` candidates.
- No automatic command path may approve creation or merging of formal memory.
- Preserve compatibility with old candidate and review records that lack new analysis fields.

---

### Task 1: Independent Evidence Gate

**Files:**
- Modify: `src/dream_memory/memory_dreaming.py:1276-1367`
- Test: `tests/test_memory_dreaming.py:512-555`

**Interfaces:**
- Produces: `_independent_evidence_event_ids(candidate: dict[str, Any]) -> set[str]`
- Extends: `analyze_dream_candidate(...) -> dict[str, Any]`
- Adds analysis fields: `independent_evidence_count: int`, `required_evidence_count: int`

- [ ] **Step 1: Write failing evidence-gate tests**

Add focused tests beside the existing `analyze_dream_candidate` tests:

```python
def test_analyze_dream_candidate_requires_two_independent_events(self):
    candidate = {
        "id": "mem_editor",
        "type": "preference",
        "scope": "user",
        "content": "用户偏好使用简洁的代码编辑器。",
        "score": 0.95,
        "tags": ["preference"],
        "evidence": [{"event_id": "event_1", "source": "codex"}],
    }

    signals = explain_candidate_quality(candidate, [])
    analysis = analyze_dream_candidate(candidate, quality_signals=signals, conflicts=[])

    self.assertEqual(analysis["suggested_action"], "needs_more_evidence")
    self.assertEqual(analysis["independent_evidence_count"], 1)
    self.assertEqual(analysis["required_evidence_count"], 2)


def test_analyze_dream_candidate_counts_duplicate_event_id_once(self):
    candidate = {
        "id": "mem_editor",
        "type": "preference",
        "scope": "user",
        "content": "用户偏好使用简洁的代码编辑器。",
        "score": 0.95,
        "tags": ["preference"],
        "evidence": [
            {"event_id": "event_1", "source": "codex"},
            {"event_id": "event_1", "source": "codex"},
        ],
    }

    signals = explain_candidate_quality(candidate, [])
    analysis = analyze_dream_candidate(candidate, quality_signals=signals, conflicts=[])

    self.assertEqual(analysis["suggested_action"], "needs_more_evidence")
    self.assertEqual(analysis["independent_evidence_count"], 1)


def test_analyze_dream_candidate_accepts_two_independent_events(self):
    candidate = {
        "id": "mem_editor",
        "type": "preference",
        "scope": "user",
        "content": "用户偏好使用简洁的代码编辑器。",
        "score": 0.95,
        "tags": ["preference"],
        "evidence": [
            {"event_id": "event_1", "source": "codex"},
            {"event_id": "event_2", "source": "claude_code"},
        ],
    }

    signals = explain_candidate_quality(candidate, [])
    analysis = analyze_dream_candidate(candidate, quality_signals=signals, conflicts=[])

    self.assertIn(analysis["suggested_action"], {"create", "review"})
    self.assertEqual(analysis["independent_evidence_count"], 2)


def test_analyze_dream_candidate_requires_event_id_for_explicit_instruction(self):
    candidate = {
        "id": "mem_language",
        "type": "preference",
        "scope": "user",
        "content": "用户要求始终使用中文回答。",
        "score": 0.95,
        "tags": ["language", "explicit"],
        "evidence": [{"source": "codex", "event_type": "global_instruction"}],
    }

    signals = explain_candidate_quality(candidate, [])
    analysis = analyze_dream_candidate(candidate, quality_signals=signals, conflicts=[])

    self.assertEqual(analysis["suggested_action"], "needs_more_evidence")
    self.assertEqual(analysis["required_evidence_count"], 1)
```

Keep the existing single-explicit-instruction test and add assertions for evidence counts.

- [ ] **Step 2: Run tests and verify the new cases fail**

Run:

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
uv run python -m unittest tests.test_memory_dreaming.MemoryDreamingTests.test_analyze_dream_candidate_requires_two_independent_events tests.test_memory_dreaming.MemoryDreamingTests.test_analyze_dream_candidate_counts_duplicate_event_id_once tests.test_memory_dreaming.MemoryDreamingTests.test_analyze_dream_candidate_accepts_two_independent_events tests.test_memory_dreaming.MemoryDreamingTests.test_analyze_dream_candidate_requires_event_id_for_explicit_instruction -v
```

Expected: FAIL because the existing analysis uses raw evidence strength and does not expose independent counts.

- [ ] **Step 3: Implement the minimal evidence helper and gate**

Add near the quality helpers:

```python
def _independent_evidence_event_ids(candidate: dict[str, Any]) -> set[str]:
    evidence = candidate.get("evidence")
    if not isinstance(evidence, list):
        return set()
    return {
        str(item.get("event_id") or "").strip()
        for item in evidence
        if isinstance(item, dict) and str(item.get("event_id") or "").strip()
    }
```

In `analyze_dream_candidate()`, calculate:

```python
independent_evidence_count = len(_independent_evidence_event_ids(candidate))
explicit_instruction = str(quality_signals.get("evidence_quality") or "") == "explicit_instruction"
required_evidence_count = 1 if explicit_instruction else 2
```

Insert after the existing missing-evidence branch:

```python
elif independent_evidence_count < required_evidence_count:
    suggested_action = "needs_more_evidence"
    decision_reason = (
        f"candidate has {independent_evidence_count} independent evidence events; "
        f"{required_evidence_count} required"
    )
```

Add both counts to the returned analysis object. Do not add a policy/config field.

- [ ] **Step 4: Run the focused analysis tests**

Run the command from Step 2.

Expected: PASS.

---

### Task 2: Evidence Deduplication and Review Eligibility

**Files:**
- Modify: `src/dream_memory/memory_dreaming.py:1041-1092`
- Modify: `src/dream_memory/memory_dreaming.py:1408-1426`
- Test: `tests/test_memory_dreaming.py`

**Interfaces:**
- Consumes: `_independent_evidence_event_ids(candidate)`
- Produces: `build_candidates_from_facts()` candidates with at most one stored evidence record per non-empty event ID
- Produces: `build_review_queue()` items only for actions in `{"create", "review", "merge"}`

- [ ] **Step 1: Write failing candidate and queue tests**

Add:

```python
def test_build_candidates_from_facts_deduplicates_same_event_id(self):
    facts = [
        {
            "fact_type": "preference",
            "statement": "用户偏好简洁回答。",
            "scope": "user",
            "project": None,
            "tags": ["preference"],
            "evidence": [{"event_id": "event_1", "source": "codex"}],
        },
        {
            "fact_type": "preference",
            "statement": "用户偏好简洁回答。",
            "scope": "user",
            "project": None,
            "tags": ["preference"],
            "evidence": [{"event_id": "event_1", "source": "codex"}],
        },
    ]

    candidates = build_candidates_from_facts(facts)

    self.assertEqual(len(candidates), 1)
    self.assertEqual([item["event_id"] for item in candidates[0]["evidence"]], ["event_1"])


def test_build_review_queue_excludes_needs_more_evidence(self):
    candidate = {
        "id": "mem_editor",
        "type": "preference",
        "scope": "user",
        "content": "用户偏好使用简洁的代码编辑器。",
        "score": 0.95,
        "tags": ["preference"],
        "evidence": [{"event_id": "event_1", "source": "codex"}],
    }

    self.assertEqual(build_review_queue([candidate], []), [])


def test_build_review_queue_includes_two_event_candidate(self):
    candidate = {
        "id": "mem_editor",
        "type": "preference",
        "scope": "user",
        "content": "用户偏好使用简洁的代码编辑器。",
        "score": 0.95,
        "tags": ["preference"],
        "evidence": [
            {"event_id": "event_1", "source": "codex"},
            {"event_id": "event_2", "source": "claude_code"},
        ],
    }

    queue = build_review_queue([candidate], [])

    self.assertEqual(len(queue), 1)
    self.assertIn(queue[0]["suggested_action"], {"create", "review"})
```

Update existing duplicate and one-off queue tests to expect an empty review queue. Their diagnostic action remains covered by `analyze_dream_candidate()` or `apply_dream_analysis_to_candidates()`.

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
uv run python -m unittest tests.test_memory_dreaming -v
```

Expected: new deduplication and queue-filter tests FAIL.

- [ ] **Step 3: Store one evidence record per event ID**

Before appending an evidence dictionary in `build_candidates_from_facts()`, skip it when the candidate already contains that non-empty event ID:

```python
event_id = str(evidence.get("event_id") or evidence.get("id") or "").strip()
if event_id and event_id in _independent_evidence_event_ids(candidate):
    continue
candidate["evidence"].append({
    "event_id": event_id or None,
    "source": evidence.get("source") or fact.get("source"),
    "session_id": evidence.get("session_id") or fact.get("session_id"),
    "quote": evidence.get("quote"),
    "content_hash": _content_hash(content),
})
```

Apply the same guard to the `evidence_refs` branch.

- [ ] **Step 4: Filter the review queue**

In `build_review_queue()`, after analysis:

```python
if dream_analysis["suggested_action"] not in {"create", "review", "merge"}:
    continue
```

Keep all candidates in `apply_dream_analysis_to_candidates()`, candidate JSONL, and `DREAMS.md`.

- [ ] **Step 5: Run the dreaming tests**

Run the command from Step 2.

Expected: PASS.

---

### Task 3: Review-Eligible Evaluation Metrics

**Files:**
- Modify: `src/dream_memory/memory_eval.py:18-224`
- Test: `tests/test_memory_eval.py`
- Modify: `examples/labeled-events.jsonl`

**Interfaces:**
- Consumes: `apply_dream_analysis_to_candidates(candidates, [])`
- Produces: evaluation candidates limited to dream actions in `{"create", "review", "merge"}`
- Adds report field: `deferred_candidate_count: int`

- [ ] **Step 1: Write failing evaluation tests**

Add:

```python
def test_eval_excludes_needs_more_evidence_from_predictions(self):
    candidate = {
        "id": "mem_editor",
        "type": "preference",
        "scope": "user",
        "content": "用户偏好使用简洁的代码编辑器。",
        "score": 0.95,
        "tags": ["preference"],
        "evidence": [{"event_id": "event_1", "source": "codex"}],
    }

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "labeled.jsonl"
        path.write_text(json.dumps({"id": "weak", "events": [], "expected": []}, ensure_ascii=False) + "\n", encoding="utf-8")
        with patch("dream_memory.memory_eval._extract_candidates", return_value=([candidate], None)):
            result = evaluate_labeled_events(path, project=None, mode="rules")

    self.assertEqual(result["predicted_total"], 0)
    self.assertEqual(result["deferred_candidate_count"], 1)
    self.assertEqual(result["false_positive_count"], 0)
```

Add a second test with two event IDs and one expected item; assert one prediction and one true positive.

- [ ] **Step 2: Run the focused evaluation tests and verify failure**

Run:

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
uv run python -m unittest tests.test_memory_eval.MemoryEvalTests.test_eval_excludes_needs_more_evidence_from_predictions -v
```

Expected: FAIL because the evaluator currently counts every non-rejected candidate.

- [ ] **Step 3: Analyze candidates before evaluation filtering**

Import `apply_dream_analysis_to_candidates`.

Change `_scored_candidates()` to:

```python
def _scored_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        candidate
        for candidate in candidates
        if str((candidate.get("dream_analysis") or {}).get("suggested_action") or "")
        in {"create", "review", "merge"}
    ]
```

In `evaluate_labeled_events()`, after extraction and before `_scored_candidates()`:

```python
candidates = apply_dream_analysis_to_candidates(candidates, [])
deferred_candidate_total += len([
    candidate
    for candidate in candidates
    if str((candidate.get("dream_analysis") or {}).get("suggested_action") or "")
    == "needs_more_evidence"
])
candidates = _scored_candidates(candidates)
```

Apply the same sequence to rule fallbacks. Initialize `deferred_candidate_total = 0` and return it as `deferred_candidate_count`.

- [ ] **Step 4: Run all evaluation tests**

Run:

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
uv run python -m unittest tests.test_memory_eval -v
```

Expected: PASS.

- [ ] **Step 5: Run the labeled rules evaluation**

Run:

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
uv run dream-memory eval --input examples/labeled-events.jsonl --project . --mode rules
```

Expected:

- precision remains `1.000`;
- true positives remain at least `7`;
- output includes `deferred_candidate_count`.

If a previously true-positive row becomes deferred, update extraction metadata only when the event is demonstrably an explicit long-term instruction. Do not weaken the two-event gate for ordinary candidates.

---

### Task 4: Manual-Only Create and Merge Decisions

**Files:**
- Modify: `src/dream_memory/memory_cli.py:685-789`
- Test: `tests/test_memory_cli.py:231-370`

**Interfaces:**
- Changes: `_auto_review_run()` never emits `approved` or `merged`
- Preserves: automatic `rejected` and `needs_more_evidence` decisions for compatible old queues
- Produces skip reason: `requires_manual_review`

- [ ] **Step 1: Write failing automatic-review safety tests**

Replace the existing auto-approval expectation with:

```python
def test_auto_review_cli_never_approves_create_candidate(self):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        memory_dir = root / "memory"
        config_path = root / "config.json"
        config_path.write_text(
            json.dumps({"output_dir": str(memory_dir)}, ensure_ascii=False),
            encoding="utf-8",
        )
        state = create_run_state(
            memory_dir=memory_dir,
            project=str(root),
            input_path="events.jsonl",
            mode="rules",
            model="rules",
            invoke_model=False,
        )
        run_dir = Path(state["run_dir"])
        queue_path = run_dir / "review_queue.jsonl"
        queue_path.write_text(json.dumps({
            "candidate_id": "create_1",
            "suggested_action": "create",
            "candidate": {
                "id": "create_1",
                "type": "preference",
                "scope": "user",
                "content": "用户偏好中文回答。",
                "evidence": [{"event_id": "event_1"}],
                "tags": ["language"],
            },
            "dream_analysis": {"dream_score": 0.82},
        }, ensure_ascii=False) + "\n", encoding="utf-8")
        update_run_state(
            state,
            status="waiting_review",
            phase="review",
            artifacts={"review_queue_path": str(queue_path)},
        )

        exit_code = main([
            "--config", str(config_path),
            "auto-review",
            "--run-id", str(state["run_id"]),
            "--min-score", "0.5",
        ])

        self.assertEqual(exit_code, 0)
        reviewed_path = run_dir / "reviewed.jsonl"
        rows = [
            json.loads(line)
            for line in reviewed_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(rows, [])
        updated = load_run_state(memory_dir, str(state["run_id"]))
        self.assertEqual(updated["counts"]["auto_review_count"], 0)
```

Add an equivalent merge test using `--include-merges`; it must still produce no approved decision.

Update the skip-reason test to expect `requires_manual_review` for `create`, `merge`, and `review`.

- [ ] **Step 2: Run the focused CLI tests and verify failure**

Run:

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
uv run python -m unittest tests.test_memory_cli.MemoryCliTests.test_auto_review_cli_never_approves_create_candidate -v
```

Expected: FAIL because current auto-review approves high-scoring create candidates.

- [ ] **Step 3: Remove automatic approval branches**

Replace the create, merge, and review approval branches in `_auto_review_run()` with:

```python
if suggested_action in {"create", "merge", "review"}:
    skip("requires_manual_review")
    if suggested_action == "merge":
        merge_skipped += 1
    continue
elif suggested_action == "reject":
    review_action = "rejected"
    rejected += 1
elif not args.keep_review and suggested_action == "needs_more_evidence":
    review_action = "needs_more_evidence"
    needs_more_evidence += 1
else:
    skip("unhandled_action")
```

Keep CLI arguments such as `--include-merges` and `--include-review` accepted for compatibility, but they must not bypass the manual gate.

- [ ] **Step 4: Run automatic-review and sync tests**

Run:

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
uv run python -m unittest tests.test_memory_cli -v
```

Expected: PASS. `sync --auto` stops at `waiting_review` when the queue contains review-eligible candidates.

---

### Task 5: Documentation and Full Verification

**Files:**
- Modify: `docs/cli.md`
- Modify: `README.md`
- Verify: all files changed by Tasks 1-4

**Interfaces:**
- Documents: two-event ordinary threshold, one-event explicit exception, cumulative sync evidence, and manual-only creation/merge behavior

- [ ] **Step 1: Update user documentation**

Add concise text:

```markdown
Ordinary memory candidates need evidence from two different event IDs before
they enter the review queue. An explicit long-term instruction needs one event.
Deferred candidates remain in candidate and DREAMS artifacts and may become
reviewable after a later sync imports additional evidence.

`auto-review` and `sync --auto` do not approve creation or merging of formal
memory. A human-reviewed decision is required before applying those changes.
```

Do not add configuration examples for the fixed thresholds.

- [ ] **Step 2: Run formatting and diff checks**

Run:

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
git diff --check
```

Expected: no output and exit code `0`.

- [ ] **Step 3: Run the full test suite**

Run:

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
uv run python -m unittest discover -s tests -q
```

Expected: all tests pass.

- [ ] **Step 4: Run the final rules evaluation**

Run:

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
uv run dream-memory eval --input examples/labeled-events.jsonl --project . --mode rules
```

Expected:

- precision `1.000`;
- true positives at least `7`;
- zero false positives;
- `deferred_candidate_count` present;
- focused gate tests prove the unsafe pre-change cases are now blocked.

- [ ] **Step 5: Review the final diff**

Run:

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
git status --short
git diff --stat
git diff
```

Confirm no unrelated files, generated reports, credentials, caches, or temporary evaluation output are included.

- [ ] **Step 6: Commit the complete implementation**

Run:

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
git add src/dream_memory/memory_dreaming.py src/dream_memory/memory_eval.py src/dream_memory/memory_cli.py tests/test_memory_dreaming.py tests/test_memory_eval.py tests/test_memory_cli.py README.md docs/cli.md
git commit -m "feat: require independent memory evidence"
```

Expected: one implementation commit containing only the evidence gate, evaluation, automatic-review safety, tests, and documentation.
