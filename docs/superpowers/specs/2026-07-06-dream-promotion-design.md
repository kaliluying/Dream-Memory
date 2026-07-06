# Dream Promotion And Explainable Dreams Design

## Summary

Dream Memory should borrow the strongest practical idea from OpenClaw's dreaming mode without turning into a background scheduler: candidate memories need an explicit promotion policy and a better explanation trail.

This design adds a lightweight dream promotion layer on top of the current `dream -> review -> apply` workflow. It computes a `dream_score`, records why a candidate should be created, merged, rejected, or sent back for more evidence, and renders those reasons in `review_queue.jsonl` and `DREAMS.md`.

## Goals

- Make memory promotion less opaque by turning existing quality signals into one explainable score.
- Reduce low-value memories by strongly penalizing one-off implementation tasks, duplicates, weak evidence, and conflicts.
- Keep the current manual review workflow intact.
- Improve `DREAMS.md` so it reads like an audit report, not only a candidate list.
- Preserve compatibility with existing CLI, Web API, run trace, and JSONL artifacts.

## Non-Goals

- No background scheduler or daemon.
- No shallow / REM / deep sleep state machine in this iteration.
- No vector database, embedding model, or semantic recall subsystem.
- No automatic write to long-term memory without human review.
- No changes to provider APIs beyond passing through the new analysis metadata.

## Current Context

The current project already has most of the raw material needed for a promotion policy:

- `memory_dreaming.explain_candidate_quality()` computes `stability`, `reuse_value`, `evidence_strength`, duplicate status, similarity, and matched memory.
- `memory_dreaming.suggest_review_action()` maps quality signals and conflicts to actions.
- `memory_dreaming.build_review_queue()` writes queue items with `quality_signals`.
- `memory_dreaming._render_dreams()` writes `DREAMS.md`, but the report is currently mostly a count summary plus promoted/review candidates.
- `memory_models.build_review_queue_item()` provides the queue schema surface.

The improvement should therefore stay close to `memory_dreaming.py` and `memory_models.py`, with focused CLI/Web compatibility tests.

## Proposed Approach

Add a new dream promotion helper in `memory_dreaming.py`:

```python
def analyze_dream_candidate(
    candidate: dict[str, Any],
    *,
    quality_signals: dict[str, Any],
    conflicts: list[dict[str, Any]],
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ...
```

The helper returns a `dream_analysis` object:

```json
{
  "dream_score": 0.74,
  "suggested_action": "create",
  "reasons": ["high reuse value", "strong evidence"],
  "penalties": [],
  "policy": {
    "promote_threshold": 0.7,
    "review_threshold": 0.45,
    "reject_one_off": true,
    "require_evidence": true
  }
}
```

`build_review_queue()` should call this helper after computing `quality_signals` and conflicts. The queue item keeps the existing `suggested_action` field for compatibility, but the value should come from `dream_analysis["suggested_action"]`. The queue item also gains a new `dream_analysis` field.

## Promotion Policy

Use a small default policy, defined in code first and optionally made configurable later:

```python
DEFAULT_DREAM_PROMOTION_POLICY = {
    "promote_threshold": 0.7,
    "review_threshold": 0.45,
    "reject_one_off": True,
    "require_evidence": True,
    "duplicate_action": "reject",
    "conflict_promote_action": "merge",
}
```

Initial scoring:

- `stability`: 35%
- `reuse_value`: 35%
- `evidence_strength`: 25%
- candidate confidence / score: 5%

Penalties:

- duplicate: force `reject`
- one-off task with `reject_one_off`: force `reject`
- no evidence with `require_evidence`: force `needs_more_evidence`
- conflicts: prefer `merge` unless the candidate is otherwise below review threshold
- similarity with an existing memory: prefer `merge`

Suggested action rules:

- Force `reject` for duplicates and one-off tasks.
- Force `needs_more_evidence` when evidence is absent.
- Use `merge` when conflicts or a matched memory exist and the score is at least `review_threshold`.
- Use `create` when score is at least `promote_threshold`.
- Use `review` when score is at least `review_threshold`.
- Otherwise use `reject`.

This keeps the project conservative: it recommends promotion, but review remains the gate before permanent memory.

## Data Model Changes

Review queue item shape becomes:

```json
{
  "candidate_id": "mem_...",
  "status": "pending",
  "suggested_action": "create",
  "candidate": {},
  "conflicts": [],
  "quality_signals": {},
  "dream_analysis": {},
  "created_at": "..."
}
```

Existing consumers that read `suggested_action`, `candidate`, or `quality_signals` continue to work. Web UI can display `dream_analysis` later, but this iteration only needs to make the API payload richer.

## DREAMS.md Report

Replace the current sparse report with an explainable report:

- `# DREAMS.md`
- `Generated`
- `Sweep Summary`
- `Promotion Policy`
- `Action Summary`
- `Create`
- `Merge`
- `Needs More Evidence`
- `Review`
- `Reject`

Each candidate line should include:

- memory type
- dream score
- suggested action
- content
- short reasons
- short penalties when present

The report should be deterministic enough for tests: sort candidates by action group, descending `dream_score`, type, and content.

## Error Handling

- If a candidate lacks quality signals, score it conservatively with zeros and route to `needs_more_evidence` or `reject`.
- If a score component cannot be parsed, treat it as `0.0`.
- If old review queue records lack `dream_analysis`, Web/API readers should continue to load them unchanged.
- If policy values are missing or invalid, use defaults.

## Testing Plan

Add focused tests around `memory_dreaming.py`:

- One-off implementation task receives `reject` and a penalty.
- Candidate with no evidence receives `needs_more_evidence`.
- Durable candidate with strong evidence receives `create`.
- Similar or conflicting candidate receives `merge`.
- `build_review_queue()` includes `dream_analysis` while preserving `suggested_action`.
- `dream_from_events()` writes `DREAMS.md` with `Promotion Policy`, action groups, dream scores, and reasons.

Add compatibility checks where useful:

- Existing CLI `dream --mode rules` still writes the same core artifacts.
- Existing Web review queue endpoint can return items with `dream_analysis`.

## Rollout

1. Add the policy constant and `analyze_dream_candidate()`.
2. Update `build_review_queue()` to attach dream analysis.
3. Update `build_review_queue_item()` to accept optional `dream_analysis`.
4. Update `dream_from_events()` / `_render_dreams()` so reports include analysis.
5. Add and run targeted tests, then run the full test suite.

## Open Questions Resolved

- The policy starts as code defaults, not user config, to keep this iteration narrow.
- Manual review remains required before writing formal memory cards.
- The feature is intentionally not a background OpenClaw clone; it is a conservative quality and audit layer for the current pipeline.
