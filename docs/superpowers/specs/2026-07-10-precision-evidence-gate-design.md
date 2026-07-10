# Precision-First Evidence Gate Design

## Summary

Dream Memory should prefer missing a weak memory over storing a false one.

This design adds a small evidence gate between candidate extraction and human review:

- ordinary candidates need support from two independent events;
- explicit long-term instructions need one independent event;
- weak candidates remain visible in candidate and dream artifacts but do not enter the review queue;
- automatic review cannot approve creation or merging of long-term memory.

The change reuses the existing cumulative event log, candidate normalization, quality analysis, review queue, and evaluation command. It adds no dependency, database, background process, or user-facing threshold configuration.

## Goals

- Reduce false memories caused by one-off tasks, plausible guesses, and duplicated evidence.
- Require two independent events before an ordinary candidate reaches human review.
- Let an explicit long-term instruction reach human review after one event.
- Prevent `sync --auto` and `auto-review` from automatically creating or merging formal memory.
- Preserve explainability in `candidates.jsonl`, `ai-candidates.jsonl`, and `DREAMS.md`.
- Verify the improvement with focused regression cases and the existing labeled evaluation set.

## Non-Goals

- No embedding model, vector database, or semantic clustering service.
- No new persistent staging database.
- No configurable evidence threshold in this iteration.
- No redesign of rule extraction or model prompts unless a focused regression proves it necessary.
- No attempt to fix the four known recall misses in the current rules evaluation.
- No automatic promotion path for any newly extracted memory.

## Decisions

- Precision is more important than recall.
- Ordinary candidates need two independent source events.
- Explicit long-term instructions may enter review after one independent source event.
- Candidates below the evidence threshold are deferred, not permanently discarded.
- New memories and merges require a human decision before application.
- The evidence threshold remains a code constant until measured usage justifies configuration.

## Current Context

The repository already contains nearly all required mechanics:

- `build_candidates_from_facts()` merges facts using normalized content, scope, and project.
- Candidate evidence currently appends every evidence item without deduplicating source events.
- `score_candidate()` and `explain_candidate_quality()` use the raw evidence list length, so repeated references to one event can inflate confidence.
- `analyze_dream_candidate()` already returns `needs_more_evidence`, but only when evidence is completely absent.
- `build_review_queue()` currently adds every candidate, including low-evidence and rejected candidates.
- `_handle_sync()` rebuilds from `imports/all-events.jsonl`, which already contains the accumulated imported event history needed for cross-run evidence.
- `_handle_sync()` can currently call automatic review and apply approved candidates without a human decision.

The pre-change rules evaluation on `examples/labeled-events.jsonl` is:

- rows: 13
- expected memories: 11
- true positives: 7
- false positives: 0
- false negatives: 4
- precision: 1.000
- recall: 0.636
- F1: 0.778

Because the existing set already has perfect measured precision, it cannot by itself prove that the new gate reduces false review candidates. Focused gate-specific cases are required.

## Proposed Approach

### 1. Count Independent Evidence

Add one small helper in `memory_dreaming.py` that returns unique evidence event IDs for a candidate.

Rules:

- A non-empty `event_id` is the identity of one independent event.
- Repeated evidence entries with the same `event_id` count once.
- Evidence without an `event_id` does not satisfy the promotion gate.
- The helper does not infer independence from quotes, list positions, or model confidence.

`build_candidates_from_facts()` should also avoid appending duplicate evidence records for the same event ID. This keeps stored artifacts honest while the helper provides the final defensive count for both rule and AI candidates.

### 2. Reuse the Existing Explicit-Instruction Signal

Do not introduce another model field or classifier.

The existing evidence-quality logic already identifies explicit instructions through event metadata and tags. A candidate is eligible for the one-event exception only when `_evidence_quality()` classifies it as `explicit_instruction`.

This exception changes only review eligibility. It does not bypass human review.

### 3. Apply the Gate Before Score-Based Promotion

`analyze_dream_candidate()` should apply decisions in this order:

1. Existing duplicate handling.
2. Existing one-off-task rejection.
3. Missing-evidence handling.
4. New independent-evidence gate.
5. Existing conflict and score-based action selection.

Gate behavior:

- explicit instruction with at least one independent event: continue to normal analysis;
- ordinary candidate with at least two independent events: continue to normal analysis;
- otherwise: return `needs_more_evidence`, regardless of model confidence or candidate score.

The analysis payload should include:

- `independent_evidence_count`;
- `required_evidence_count`;
- a concise decision reason when evidence is insufficient.

No new policy configuration is required. The ordinary threshold is the constant `2`; the explicit-instruction threshold is `1`.

### 4. Keep Deferred Candidates Out of Human Review

All candidates remain in the candidate JSONL and `DREAMS.md` for auditability.

`build_review_queue()` should only emit candidates whose analyzed action is eligible for a human memory decision:

- `create`;
- `review`;
- `merge`.

Candidates analyzed as `needs_more_evidence` or `reject` stay in the diagnostic artifacts and do not inflate pending-review counts.

This makes the existing candidate artifacts the temporary holding area. No separate staging store is needed:

- `sync` already imports the full available history into `all-events.jsonl`;
- a later sync rebuilds the candidate and naturally observes newly accumulated evidence;
- a manual `run` or `pipeline` only considers the input file supplied by the user.

### 5. Disable Automatic Memory Approval

Automatic review may classify or skip candidates, but it must not produce approved `create` or `merge` decisions.

Required behavior:

- `auto-review` never writes an approval that can create or merge formal memory;
- `sync --auto` stops at `waiting_review` when review-eligible candidates exist;
- `_resume_run()` applies memory only from an explicit human-reviewed decision file;
- duplicate and rejected candidates remain diagnostic outcomes rather than new memory.

The existing commands remain available for compatibility. Their safer behavior should be documented rather than replaced with a new command.

## Data Flow

1. Importers write the cumulative event set to `imports/all-events.jsonl`.
2. Rule or AI extraction produces atomic facts and candidates.
3. Candidate construction merges normalized facts and deduplicates evidence by event ID.
4. Quality analysis computes existing signals plus the independent evidence count.
5. The evidence gate marks weak candidates `needs_more_evidence`.
6. Candidate JSONL and `DREAMS.md` retain all analyzed candidates.
7. The review queue contains only `create`, `review`, or `merge` candidates.
8. A human review decision is required before `_resume_run()` writes formal memory.

## Error Handling

- Missing or malformed evidence lists behave as zero independent evidence.
- Blank event IDs do not count.
- Duplicate event IDs count once.
- An explicit-instruction tag without a valid event ID does not pass the gate.
- Old candidate and review artifacts without the new analysis fields remain readable.
- A custom input file that omits earlier events cannot inherit evidence from an unrelated run.
- Model or rule confidence never overrides insufficient independent evidence.

## Evaluation Design

### Existing Evaluation

Update evaluation filtering so precision and recall are calculated from candidates eligible for human review, not from every non-rejected extracted candidate. `needs_more_evidence` is not a predicted long-term memory.

After implementation:

- existing precision must remain `1.000`;
- the existing seven true positives must remain true positives;
- the four existing false negatives may remain unchanged in this iteration;
- evaluation output must report deferred candidate counts separately.

### New Focused Cases

Add labeled cases and focused tests for:

1. One ordinary durable-looking statement from one event is deferred.
2. The same event duplicated in evidence still counts as one and is deferred.
3. Two different event IDs supporting the same normalized fact enter review.
4. One explicit long-term instruction with a valid event ID enters review.
5. An explicit-looking candidate without an event ID is deferred.
6. `sync --auto` cannot create or merge formal memory.

The first, second, and sixth cases should demonstrate behavior that is unsafe before the change and safe after it.

### Verification Commands

Run:

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
uv run python -m unittest discover -s tests -q
uv run dream-memory eval --input examples/labeled-events.jsonl --project . --mode rules
```

If an AI provider is configured, also run the labeled evaluation in AI mode as a non-blocking diagnostic. Provider availability is not required for completion because the evidence gate is downstream of both extractors.

## Success Criteria

- All existing unit tests pass.
- All new evidence-gate and auto-approval safety tests pass.
- Existing rules precision remains `1.000`.
- Existing rules true positives remain at least `7`.
- Deferred candidates are excluded from `predicted_total` and reported separately.
- A repeated reference to one event cannot satisfy the two-event gate.
- Two independent events can satisfy the gate after normalization.
- No automatic command path writes a newly created or merged formal memory.

## Expected Files

Keep the implementation focused:

- `src/dream_memory/memory_dreaming.py`
- `src/dream_memory/memory_eval.py`
- `src/dream_memory/memory_cli.py`
- focused tests in the existing test modules
- `examples/labeled-events.jsonl` only if additional labeled rows are needed
- CLI documentation describing the safer automatic-review behavior

Avoid new modules unless the existing files make a small, tested change impossible.

## Rollout

1. Add independent evidence counting and evidence deduplication.
2. Add the evidence gate to dream analysis.
3. Filter deferred and rejected candidates out of the review queue.
4. Prevent automatic approval of create and merge decisions.
5. Update evaluation filtering and add focused labeled cases.
6. Run targeted tests, the full suite, and before/after evaluation.
7. Record measured results in the implementation handoff.
