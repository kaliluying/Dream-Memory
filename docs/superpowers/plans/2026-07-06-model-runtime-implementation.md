# Model Runtime Medium Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add named model profiles, retry/backoff, fallback chains, and run trace metadata while preserving the existing Dream Memory CLI behavior.

**Architecture:** Add focused runtime data structures and orchestration to the model provider layer, then thread runtime metadata through the LangGraph extraction path and CLI persistent run flow. Existing flat config remains valid by normalizing it into an implicit profile and policy.

**Tech Stack:** Python 3.11+, stdlib `urllib`, `dataclasses`, existing `argparse` CLI, LangGraph, pytest/unittest.

---

## File Structure

- Modify `src/dream_memory/memory_config.py`: extend defaults with `models/model_policy`, add config normalization helpers.
- Modify `src/dream_memory/model_providers.py`: add error classes, retry policy/profile dataclasses, runtime orchestration, and diagnostics for all profiles.
- Modify `src/dream_memory/memory_graph.py`: accept runtime config and expose runtime metadata in graph state.
- Modify `src/dream_memory/memory_agent.py`: pass runtime config into the graph and include runtime metadata in extraction output.
- Modify `src/dream_memory/memory_cli.py`: resolve runtime config from config/CLI overrides, append run trace events, extend `check-provider`.
- Modify `src/dream_memory/web.py`: keep run start compatible with the new CLI/runtime config path.
- Modify tests under `tests/`: add coverage for config normalization, runtime retry/fallback, graph metadata, CLI trace, and diagnostics.
- Modify `README.md` and `docs/config.md`: document the medium model runtime configuration.

## Task 1: Config Normalization

**Files:**
- Modify: `src/dream_memory/memory_config.py`
- Test: `tests/test_memory_config.py`

- [ ] **Step 1: Write failing config tests**

Add tests that assert:

```python
def test_load_memory_config_adds_implicit_model_profile_for_flat_config(self):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.json"
        path.write_text(json.dumps({
            "provider": "openai",
            "model": "gpt-4.1",
            "api_key_env": "OPENAI_API_KEY",
            "timeout_seconds": 45,
        }), encoding="utf-8")

        config = load_memory_config(path)

        self.assertEqual(config["models"]["default"]["provider"], "openai")
        self.assertEqual(config["models"]["default"]["model"], "gpt-4.1")
        self.assertEqual(config["model_policy"]["fallback_chain"], ["default"])
```

```python
def test_load_memory_config_preserves_named_profiles_and_policy(self):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.json"
        path.write_text(json.dumps({
            "models": {
                "primary": {"provider": "anthropic", "model": "claude-sonnet-4-6", "api_key_env": "ANTHROPIC_API_KEY"},
                "backup": {"provider": "openai", "model": "gpt-4.1", "api_key_env": "OPENAI_API_KEY"},
            },
            "model_policy": {
                "default_profile": "primary",
                "fallback_chain": ["primary", "backup"],
                "retry": {"max_attempts": 2},
            },
        }), encoding="utf-8")

        config = load_memory_config(path)

        self.assertEqual(config["model_policy"]["default_profile"], "primary")
        self.assertEqual(config["model_policy"]["fallback_chain"], ["primary", "backup"])
        self.assertEqual(config["model_policy"]["retry"]["max_attempts"], 2)
        self.assertEqual(config["model_policy"]["retry"]["retry_on_status"], [429, 500, 502, 503, 504])
```

- [ ] **Step 2: Run config tests and verify failure**

Run: `uv run --with pytest pytest tests/test_memory_config.py -q`

Expected: FAIL because `models` and `model_policy` are not populated.

- [ ] **Step 3: Implement config defaults and normalization**

Add default retry/policy keys, accept `models` and `model_policy`, and normalize loaded configs so old flat configs produce:

```python
config["models"] = {
    "default": {
        "provider": config["provider"],
        "model": config["model"],
        "api_key_env": config["api_key_env"],
        "base_url": config["base_url"],
        "timeout_seconds": config["timeout_seconds"],
    }
}
config["model_policy"] = {
    "default_profile": "default",
    "fallback_chain": ["default"],
    "retry": DEFAULT_RETRY_POLICY,
    "allow_rules_fallback": False,
}
```

For named profiles, merge each profile over flat defaults and merge retry over default retry.

- [ ] **Step 4: Run config tests and verify pass**

Run: `uv run --with pytest pytest tests/test_memory_config.py -q`

Expected: PASS.

- [ ] **Step 5: Commit config normalization**

Run:

```powershell
git add src/dream_memory/memory_config.py tests/test_memory_config.py
git commit -m "feat: normalize model runtime config"
```

## Task 2: Runtime Retry and Fallback

**Files:**
- Modify: `src/dream_memory/model_providers.py`
- Test: `tests/test_model_providers.py`

- [ ] **Step 1: Write failing runtime tests**

Add tests for:

```python
def test_model_runtime_retries_retryable_http_error_then_succeeds(self):
    attempts = []

    def provider_factory(config):
        class Provider:
            def invoke(self, prompt):
                attempts.append(config.model)
                if len(attempts) == 1:
                    raise ModelHTTPError(429, "rate limited")
                return '{"candidates":[]}'
        return Provider()

    runtime = ModelRuntime(provider_factory=provider_factory, sleeper=lambda _: None)
    result = runtime.invoke(
        "prompt",
        profiles={"primary": ModelProfile("primary", ProviderConfig("openai", "gpt-4.1", "OPENAI_API_KEY"))},
        policy=ModelPolicy(default_profile="primary", fallback_chain=["primary"], retry=RetryPolicy(max_attempts=2)),
    )

    self.assertEqual(result.text, '{"candidates":[]}')
    self.assertEqual(len(result.attempts), 2)
    self.assertEqual(result.selected_profile, "primary")
```

```python
def test_model_runtime_falls_back_to_backup_profile(self):
    def provider_factory(config):
        class Provider:
            def invoke(self, prompt):
                if config.model == "bad-model":
                    raise ModelHTTPError(503, "unavailable")
                return '{"candidates":[]}'
        return Provider()

    runtime = ModelRuntime(provider_factory=provider_factory, sleeper=lambda _: None)
    result = runtime.invoke(
        "prompt",
        profiles={
            "primary": ModelProfile("primary", ProviderConfig("anthropic", "bad-model", "A")),
            "backup": ModelProfile("backup", ProviderConfig("openai", "gpt-4.1", "B")),
        },
        policy=ModelPolicy(default_profile="primary", fallback_chain=["primary", "backup"], retry=RetryPolicy(max_attempts=1)),
    )

    self.assertEqual(result.selected_profile, "backup")
    self.assertTrue(any(attempt.profile == "primary" and not attempt.ok for attempt in result.attempts))
```

- [ ] **Step 2: Run runtime tests and verify failure**

Run: `uv run --with pytest pytest tests/test_model_providers.py -q`

Expected: FAIL because runtime classes do not exist.

- [ ] **Step 3: Implement runtime classes and classified errors**

Add:

- `ModelProviderError`
- `ModelHTTPError`
- `ModelTimeoutError`
- `ModelAuthError`
- `ModelRuntimeError`
- `RetryPolicy`
- `ModelProfile`
- `ModelPolicy`
- `ModelAttempt`
- `ModelRuntimeResult`
- `ModelRuntime`
- `build_runtime_from_config`
- `invoke_model_runtime`

Make `_api_key()` raise `ModelAuthError`. Make `_post_json()` raise `ModelHTTPError` for HTTP status and `ModelTimeoutError` for timeout-like failures.

- [ ] **Step 4: Run runtime tests and verify pass**

Run: `uv run --with pytest pytest tests/test_model_providers.py -q`

Expected: PASS.

- [ ] **Step 5: Commit runtime layer**

Run:

```powershell
git add src/dream_memory/model_providers.py tests/test_model_providers.py
git commit -m "feat: add model runtime retry fallback"
```

## Task 3: Thread Runtime Through Graph and Agent

**Files:**
- Modify: `src/dream_memory/memory_graph.py`
- Modify: `src/dream_memory/memory_agent.py`
- Test: `tests/test_memory_graph.py`
- Test: `tests/test_memory_agent.py`

- [ ] **Step 1: Write failing graph metadata test**

Add a graph test asserting runtime metadata is surfaced:

```python
with patch("dream_memory.memory_graph.invoke_model_runtime") as invoke:
    invoke.return_value = ModelRuntimeResult(
        text=raw,
        selected_profile="primary",
        attempts=[],
        elapsed_ms=1,
    )
    result = run_memory_extraction_graph(..., runtime_config={"models": ..., "model_policy": ...})

self.assertEqual(result["model_runtime"]["selected_profile"], "primary")
```

- [ ] **Step 2: Implement graph and agent runtime plumbing**

Allow `run_memory_extraction_graph()` and `agent_extract_memory_candidates()` to accept optional `runtime_config` and `trace_callback`. In the model node, call `invoke_model_runtime()` when runtime config is present, otherwise keep the compatibility `invoke_model_provider()` path.

- [ ] **Step 3: Run graph/agent tests**

Run: `uv run --with pytest pytest tests/test_memory_graph.py tests/test_memory_agent.py -q`

Expected: PASS.

- [ ] **Step 4: Commit graph plumbing**

Run:

```powershell
git add src/dream_memory/memory_graph.py src/dream_memory/memory_agent.py tests/test_memory_graph.py tests/test_memory_agent.py
git commit -m "feat: pass model runtime through extraction graph"
```

## Task 4: CLI Integration and Run Trace

**Files:**
- Modify: `src/dream_memory/memory_cli.py`
- Modify: `src/dream_memory/web.py`
- Test: `tests/test_memory_cli.py`
- Test: `tests/test_memory_review_web.py`

- [ ] **Step 1: Write failing CLI tests**

Add tests for:

- persistent `run` writes `model_attempt_started` and `model_attempt_succeeded` events when runtime invokes a fake provider.
- `check-provider --all` returns profile diagnostics.
- `--dry-run` does not invoke runtime and still writes prompt artifacts.

- [ ] **Step 2: Implement CLI runtime resolution**

Add helper:

```python
def _runtime_config_from_args(args, config):
    # command-line provider/model override creates one-off profile
    # otherwise use config["models"] and config["model_policy"]
```

Use it in `dream`, `pipeline`, and `run`. For persistent runs, pass a `trace_callback` that maps runtime event dicts to `append_trace(state, event_type, payload)`.

- [ ] **Step 3: Extend check-provider**

Add `--all` and `--profile`. For `--all`, return:

```json
{
  "profiles": {
    "primary": {"ok": true, "...": "..."},
    "backup": {"ok": false, "error": "..."}
  }
}
```

- [ ] **Step 4: Run CLI/Web tests**

Run: `uv run --with pytest pytest tests/test_memory_cli.py tests/test_memory_review_web.py -q`

Expected: PASS, except any pre-existing unrelated Windows SQLite cleanup failure should be noted separately.

- [ ] **Step 5: Commit CLI integration**

Run:

```powershell
git add src/dream_memory/memory_cli.py src/dream_memory/web.py tests/test_memory_cli.py tests/test_memory_review_web.py
git commit -m "feat: integrate model runtime with cli runs"
```

## Task 5: Docs and Full Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/config.md`

- [ ] **Step 1: Update docs**

Document:

- `models`
- `model_policy`
- retry defaults
- fallback chain
- `check-provider --all`
- old flat config compatibility

- [ ] **Step 2: Run full targeted test suite**

Run:

```powershell
uv run --with pytest pytest tests/test_memory_config.py tests/test_model_providers.py tests/test_memory_graph.py tests/test_memory_agent.py tests/test_memory_cli.py tests/test_memory_review_web.py -q
```

Expected: PASS for changed surfaces.

- [ ] **Step 3: Run full suite**

Run:

```powershell
uv run --with pytest pytest -q
```

Expected: PASS, or report any pre-existing failures separately.

- [ ] **Step 4: Commit docs**

Run:

```powershell
git add README.md docs/config.md
git commit -m "docs: describe model runtime profiles"
```

