# Model Runtime Medium Upgrade Design

## Context

Dream Memory currently has a single active model configuration made of `provider`, `model`, `api_key_env`, `base_url`, `timeout_seconds`, and `invoke_model`. The CLI merges command-line overrides over `.dream-memory/config.json`, converts the active provider and model into a `provider:model` string, and passes that into the direct extraction path. The provider layer performs one HTTP request through `urllib.request.urlopen` and raises on HTTP failures.

This is simple and easy to understand, but too fragile for a memory extraction workflow. A temporary provider outage, rate limit, or model-specific parse failure can stop an entire run. It also leaves little evidence about which model was tried, how long it took, why it failed, or whether a fallback was used.

## Goals

- Support named model profiles while preserving the existing flat config format.
- Add retry with bounded exponential backoff for transient failures.
- Add fallback chains across model profiles.
- Record model attempt trace data in persistent runs.
- Keep existing CLI commands and call sites mostly stable.
- Keep rules mode as an explicit extraction mode, not as an automatic first-line substitute for model failures.

## Non-Goals

- Do not introduce LangChain, LiteLLM, or another large routing dependency.
- Do not add cost accounting, token accounting, quotas, or circuit breakers in this upgrade.
- Do not change the candidate validation schema or review workflow.
- Do not require users to migrate existing `.dream-memory/config.json` files immediately.

## Recommended Configuration Shape

The new format adds `models` and `model_policy` while keeping the existing fields valid:

```json
{
  "models": {
    "primary": {
      "provider": "anthropic",
      "model": "claude-sonnet-4-6",
      "api_key_env": "ANTHROPIC_API_KEY",
      "base_url": null,
      "timeout_seconds": 60
    },
    "openai_backup": {
      "provider": "openai",
      "model": "gpt-4.1",
      "api_key_env": "OPENAI_API_KEY",
      "base_url": null,
      "timeout_seconds": 45
    }
  },
  "model_policy": {
    "default_profile": "primary",
    "fallback_chain": ["primary", "openai_backup"],
    "retry": {
      "max_attempts": 3,
      "initial_delay_seconds": 1.0,
      "backoff_factor": 2.0,
      "max_delay_seconds": 8.0,
      "retry_on_status": [429, 500, 502, 503, 504],
      "retry_on_timeout": true
    },
    "allow_rules_fallback": false
  }
}
```

Backward compatibility rules:

- If `models` is missing, build an implicit `default` profile from the current flat fields.
- If `model_policy` is missing, use a single-profile chain containing the implicit or configured default profile.
- Existing CLI overrides `--provider`, `--model`, `--api-key-env`, `--base-url`, and `--timeout-seconds` still create a one-off profile for that command.
- `--dry-run` and `--invoke-model` keep their current behavior.

## Runtime Design

Introduce a lightweight model runtime layer in `model_providers.py` or a new focused module such as `model_runtime.py`.

Core objects:

- `ProviderConfig`: existing per-provider configuration, kept mostly unchanged.
- `RetryPolicy`: max attempts, delay, backoff, retryable statuses, timeout retry flag.
- `ModelProfile`: named provider config.
- `ModelPolicy`: default profile, fallback chain, retry policy, optional rules fallback flag.
- `ModelAttempt`: trace-friendly result for one try.
- `ModelRuntimeResult`: final text plus selected profile, attempts, fallback metadata, and elapsed time.
- `ModelRuntime`: orchestrates retries and profile fallback while delegating actual HTTP calls to provider classes.

The existing `invoke_model(prompt, model=...) -> str` can remain as a compatibility wrapper. A new `invoke_model_runtime(prompt, runtime_config, trace_callback=None) -> ModelRuntimeResult` should be used by the CLI and direct extraction path once the config is available.

## Data Flow

1. CLI loads config with existing `load_memory_config()`.
2. CLI resolves command-line overrides into either:
   - a configured model policy from `models/model_policy`, or
   - a one-off profile when `--provider` or `--model` is supplied.
3. The direct model invocation step calls the runtime instead of a single provider request.
4. Runtime tries each profile in `fallback_chain`.
5. For each profile, runtime performs retry attempts according to `RetryPolicy`.
6. On success, runtime returns the raw model response and trace metadata.
7. Candidate validation continues unchanged.
8. Persistent runs append model trace events.

## Error Handling

Provider errors should be classified enough for retry and diagnostics:

- `ModelHTTPError`: carries HTTP status and response preview.
- `ModelTimeoutError`: wraps timeout-related failures.
- `ModelAuthError`: non-retryable auth/config failures such as missing API key or 401/403.
- `ModelProviderError`: general provider failure.
- `ModelRuntimeError`: all profiles failed; includes attempt summaries.

Retry policy:

- Retry `429`, `500`, `502`, `503`, and `504` by default.
- Retry socket timeout and URL timeout errors when `retry_on_timeout` is true.
- Do not retry missing API key, unsupported provider, invalid config, 400, 401, or 403.
- Sleep between attempts using bounded exponential backoff.
- Tests should inject a fake sleeper so retry tests run quickly.

Fallback policy:

- Move to the next profile only after the current profile exhausts retryable attempts or fails with a provider-level retryable error.
- For non-retryable config/auth errors, record the failure and continue to the next profile if one exists.
- If all profiles fail, raise `ModelRuntimeError` with concise summaries.
- `allow_rules_fallback` should not silently apply formal memory. If enabled later, it should produce rule candidates and clearly mark the run as degraded.

## Trace Events

Persistent runs should record these events in `trace.jsonl`:

- `model_attempt_started`: profile, provider, model, attempt index.
- `model_attempt_succeeded`: profile, provider, model, attempt index, elapsed milliseconds.
- `model_attempt_failed`: profile, provider, model, attempt index, retryable flag, error kind, status if available, elapsed milliseconds.
- `model_fallback_used`: from profile, to profile, reason.
- `model_runtime_failed`: total attempts and final error summary.

For non-persistent CLI commands, the JSON payload can include a compact `model_runtime` object with selected profile and attempt count.

## CLI Behavior

Existing commands remain valid:

```powershell
uv run dream-memory dream --input .dream-memory/imports/all-events.jsonl --project .
uv run dream-memory pipeline --input .dream-memory/imports/all-events.jsonl --provider openai --model gpt-4.1
```

New optional commands and flags:

- `check-provider --all`: validate all configured profiles.
- `check-provider --profile primary`: validate one profile.
- `dream/pipeline/run --model-profile primary`: select a configured profile.
- `dream/pipeline/run --fallback-chain primary,openai_backup`: override chain for one run.
- `dream/pipeline/run --retry-attempts 2`: override retry attempts for one run.

The first implementation slice can skip some flags if the config file supports the feature and tests cover the runtime.

## Testing Plan

- Config tests:
  - old flat config still loads and creates an implicit profile.
  - new `models/model_policy` config validates.
  - invalid fallback profile names fail clearly.
  - CLI overrides produce a one-off profile.
- Runtime tests:
  - success on first attempt.
  - retry on 429 then success.
  - timeout retry obeys max attempts.
  - non-retryable auth error does not retry.
  - primary profile failure falls back to backup profile.
  - all profiles failed raises `ModelRuntimeError` with attempt summaries.
- CLI/run tests:
  - persistent run writes model trace events.
  - dry run does not invoke runtime.
  - `check-provider --all` reports each profile.
- Regression tests:
  - existing AI mode, rules mode, pipeline, and run/resume behavior remain intact.

## Implementation Order

1. Extend config defaults and normalization helpers for old and new config shapes.
2. Add runtime dataclasses and provider error classes.
3. Refactor `_post_json` to classify HTTP and timeout errors.
4. Add `ModelRuntime` retry and fallback orchestration.
5. Thread runtime metadata through `memory_agent.py` and `memory_cli.py`.
6. Append model trace events for persistent runs.
7. Extend `check-provider`.
8. Update docs and examples.

## Open Decisions

- Keep the initial fallback chain config-only, then add CLI chain/profile flags if needed during implementation.
- Keep provider dependencies stdlib-only for now; add `tenacity` only if custom retry logic becomes noisy.
- Keep automatic rules fallback disabled by default.

