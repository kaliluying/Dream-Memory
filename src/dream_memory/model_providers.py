from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


class MemoryModelProvider(Protocol):
    def invoke(self, prompt: str) -> str:
        ...


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model: str
    api_key_env: str | None = None
    base_url: str | None = None
    timeout_seconds: int = 60


class ModelProviderError(RuntimeError):
    retryable = False
    status_code: int | None = None


class ModelHTTPError(ModelProviderError):
    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self.body = body
        self.retryable = status_code in {429, 500, 502, 503, 504}
        super().__init__(f"Model provider HTTP {status_code}: {body}")


class ModelTimeoutError(ModelProviderError):
    retryable = True


class ModelAuthError(ModelProviderError):
    retryable = False


class ModelRuntimeError(ModelProviderError):
    def __init__(self, attempts: list["ModelAttempt"]) -> None:
        self.attempts = attempts
        summary = "; ".join(
            f"{attempt.profile}/{attempt.provider}:{attempt.model} attempt {attempt.attempt}: {attempt.error_kind} {attempt.error or ''}".strip()
            for attempt in attempts
            if not attempt.ok
        )
        super().__init__(f"All model profiles failed: {summary}")


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    initial_delay_seconds: float = 1.0
    backoff_factor: float = 2.0
    max_delay_seconds: float = 8.0
    retry_on_status: list[int] = field(default_factory=lambda: [429, 500, 502, 503, 504])
    retry_on_timeout: bool = True


@dataclass(frozen=True)
class ModelProfile:
    name: str
    config: ProviderConfig


@dataclass(frozen=True)
class ModelPolicy:
    default_profile: str
    fallback_chain: list[str]
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    allow_rules_fallback: bool = False


@dataclass(frozen=True)
class ModelAttempt:
    profile: str
    provider: str
    model: str
    attempt: int
    ok: bool
    retryable: bool
    elapsed_ms: int
    error_kind: str | None = None
    error: str | None = None
    status_code: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "provider": self.provider,
            "model": self.model,
            "attempt": self.attempt,
            "ok": self.ok,
            "retryable": self.retryable,
            "elapsed_ms": self.elapsed_ms,
            "error_kind": self.error_kind,
            "error": self.error,
            "status_code": self.status_code,
        }


@dataclass(frozen=True)
class ModelRuntimeResult:
    text: str
    selected_profile: str
    attempts: list[ModelAttempt]
    elapsed_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_profile": self.selected_profile,
            "attempt_count": len(self.attempts),
            "elapsed_ms": self.elapsed_ms,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }


TraceCallback = Callable[[str, dict[str, Any]], None]


class StaticModelProvider:
    """Simple test provider returning a fixed response."""

    def __init__(self, response: str) -> None:
        self.response = response

    def invoke(self, prompt: str) -> str:
        return self.response


class AnthropicProvider:
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    def invoke(self, prompt: str) -> str:
        api_key = _api_key(self.config.api_key_env or "ANTHROPIC_API_KEY")
        payload = {
            "model": self.config.model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
        data = _post_json(
            self.config.base_url or "https://api.anthropic.com/v1/messages",
            payload,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout_seconds=self.config.timeout_seconds,
        )
        content = data.get("content")
        if isinstance(content, list):
            return "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        return str(data)


class OpenAICompatibleProvider:
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    def invoke(self, prompt: str) -> str:
        provider = self.config.provider
        default_env = "OPENROUTER_API_KEY" if provider == "openrouter" else "OPENAI_API_KEY"
        api_key = _api_key(self.config.api_key_env or default_env)
        default_url = "https://openrouter.ai/api/v1/chat/completions" if provider == "openrouter" else "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        data = _post_json(
            self.config.base_url or default_url,
            payload,
            headers={
                "authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            },
            timeout_seconds=self.config.timeout_seconds,
        )
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    return str(message.get("content") or "")
                return str(first.get("text") or "")
        return str(data)


def parse_model_ref(model_ref: str, *, provider: str | None = None, api_key_env: str | None = None, base_url: str | None = None, timeout_seconds: int = 60) -> ProviderConfig:
    if provider:
        model = model_ref.split(":", 1)[1] if ":" in model_ref and model_ref.split(":", 1)[0] == provider else model_ref
        return ProviderConfig(provider=provider, model=model, api_key_env=api_key_env, base_url=base_url, timeout_seconds=timeout_seconds)
    if ":" in model_ref:
        parsed_provider, parsed_model = model_ref.split(":", 1)
        return ProviderConfig(provider=parsed_provider, model=parsed_model, api_key_env=api_key_env, base_url=base_url, timeout_seconds=timeout_seconds)
    return ProviderConfig(provider="anthropic", model=model_ref, api_key_env=api_key_env, base_url=base_url, timeout_seconds=timeout_seconds)


def build_model_provider(config: ProviderConfig) -> MemoryModelProvider:
    if config.provider == "anthropic":
        return AnthropicProvider(config)
    if config.provider in {"openai", "openrouter"}:
        return OpenAICompatibleProvider(config)
    raise ValueError(f"Unsupported model provider: {config.provider}")


class ModelRuntime:
    def __init__(
        self,
        *,
        provider_factory: Callable[[ProviderConfig], MemoryModelProvider] = build_model_provider,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.provider_factory = provider_factory
        self.sleeper = sleeper

    def invoke(
        self,
        prompt: str,
        *,
        profiles: dict[str, ModelProfile],
        policy: ModelPolicy,
        trace_callback: TraceCallback | None = None,
    ) -> ModelRuntimeResult:
        attempts: list[ModelAttempt] = []
        started = time.monotonic()
        chain = policy.fallback_chain or [policy.default_profile]
        for index, profile_name in enumerate(chain):
            profile = profiles.get(profile_name)
            if profile is None:
                attempt = ModelAttempt(
                    profile=profile_name,
                    provider="",
                    model="",
                    attempt=1,
                    ok=False,
                    retryable=False,
                    elapsed_ms=0,
                    error_kind="ModelProviderError",
                    error=f"unknown model profile: {profile_name}",
                )
                attempts.append(attempt)
                self._emit(trace_callback, "model_attempt_failed", attempt.to_dict())
                continue
            result = self._invoke_profile(prompt, profile=profile, policy=policy, attempts=attempts, trace_callback=trace_callback)
            if result is not None:
                elapsed_ms = int((time.monotonic() - started) * 1000)
                return ModelRuntimeResult(text=result, selected_profile=profile.name, attempts=attempts, elapsed_ms=elapsed_ms)
            if index + 1 < len(chain):
                self._emit(
                    trace_callback,
                    "model_fallback_used",
                    {"from_profile": profile_name, "to_profile": chain[index + 1], "reason": "profile_failed"},
                )
        self._emit(trace_callback, "model_runtime_failed", {"attempt_count": len(attempts), "profiles": chain})
        raise ModelRuntimeError(attempts)

    def _invoke_profile(
        self,
        prompt: str,
        *,
        profile: ModelProfile,
        policy: ModelPolicy,
        attempts: list[ModelAttempt],
        trace_callback: TraceCallback | None,
    ) -> str | None:
        delay = policy.retry.initial_delay_seconds
        max_attempts = max(1, int(policy.retry.max_attempts))
        for attempt_index in range(1, max_attempts + 1):
            self._emit(
                trace_callback,
                "model_attempt_started",
                {
                    "profile": profile.name,
                    "provider": profile.config.provider,
                    "model": profile.config.model,
                    "attempt": attempt_index,
                },
            )
            started = time.monotonic()
            try:
                text = self.provider_factory(profile.config).invoke(prompt)
            except Exception as exc:
                elapsed_ms = int((time.monotonic() - started) * 1000)
                retryable = self._is_retryable(exc, policy.retry)
                attempt = ModelAttempt(
                    profile=profile.name,
                    provider=profile.config.provider,
                    model=profile.config.model,
                    attempt=attempt_index,
                    ok=False,
                    retryable=retryable,
                    elapsed_ms=elapsed_ms,
                    error_kind=type(exc).__name__,
                    error=str(exc),
                    status_code=getattr(exc, "status_code", None),
                )
                attempts.append(attempt)
                self._emit(trace_callback, "model_attempt_failed", attempt.to_dict())
                if not retryable or attempt_index == max_attempts:
                    return None
                self.sleeper(min(delay, policy.retry.max_delay_seconds))
                delay *= policy.retry.backoff_factor
            else:
                elapsed_ms = int((time.monotonic() - started) * 1000)
                attempt = ModelAttempt(
                    profile=profile.name,
                    provider=profile.config.provider,
                    model=profile.config.model,
                    attempt=attempt_index,
                    ok=True,
                    retryable=False,
                    elapsed_ms=elapsed_ms,
                )
                attempts.append(attempt)
                self._emit(trace_callback, "model_attempt_succeeded", attempt.to_dict())
                return text
        return None

    @staticmethod
    def _is_retryable(exc: Exception, retry: RetryPolicy) -> bool:
        if isinstance(exc, ModelHTTPError):
            return exc.status_code in retry.retry_on_status
        if isinstance(exc, ModelTimeoutError):
            return retry.retry_on_timeout
        if isinstance(exc, ModelProviderError):
            return bool(getattr(exc, "retryable", False))
        return False

    @staticmethod
    def _emit(trace_callback: TraceCallback | None, event_type: str, payload: dict[str, Any]) -> None:
        if trace_callback:
            trace_callback(event_type, payload)


def _retry_policy_from_config(value: dict[str, Any] | None) -> RetryPolicy:
    value = value or {}
    return RetryPolicy(
        max_attempts=int(value.get("max_attempts", 3)),
        initial_delay_seconds=float(value.get("initial_delay_seconds", 1.0)),
        backoff_factor=float(value.get("backoff_factor", 2.0)),
        max_delay_seconds=float(value.get("max_delay_seconds", 8.0)),
        retry_on_status=[int(status) for status in value.get("retry_on_status", [429, 500, 502, 503, 504])],
        retry_on_timeout=bool(value.get("retry_on_timeout", True)),
    )


def runtime_parts_from_config(config: dict[str, Any]) -> tuple[dict[str, ModelProfile], ModelPolicy]:
    raw_models = config.get("models")
    if not isinstance(raw_models, dict) or not raw_models:
        provider_config = parse_model_ref(
            str(config["model"]),
            provider=str(config.get("provider") or "anthropic"),
            api_key_env=str(config["api_key_env"]) if config.get("api_key_env") else None,
            base_url=str(config["base_url"]) if config.get("base_url") else None,
            timeout_seconds=int(config.get("timeout_seconds", 60)),
        )
        raw_models = {"default": {
            "provider": provider_config.provider,
            "model": provider_config.model,
            "api_key_env": provider_config.api_key_env,
            "base_url": provider_config.base_url,
            "timeout_seconds": provider_config.timeout_seconds,
        }}
    profiles: dict[str, ModelProfile] = {}
    for name, value in raw_models.items():
        if not isinstance(value, dict):
            continue
        profiles[str(name)] = ModelProfile(
            str(name),
            ProviderConfig(
                provider=str(value.get("provider") or "anthropic"),
                model=str(value.get("model") or ""),
                api_key_env=str(value["api_key_env"]) if value.get("api_key_env") else None,
                base_url=str(value["base_url"]) if value.get("base_url") else None,
                timeout_seconds=int(value.get("timeout_seconds", 60)),
            ),
        )
    if not profiles:
        raise ValueError("No model profiles configured")
    raw_policy = config.get("model_policy") if isinstance(config.get("model_policy"), dict) else {}
    default_profile = str(raw_policy.get("default_profile") or next(iter(profiles)))
    fallback_chain = raw_policy.get("fallback_chain")
    if not isinstance(fallback_chain, list) or not fallback_chain:
        fallback_chain = [default_profile]
    policy = ModelPolicy(
        default_profile=default_profile,
        fallback_chain=[str(name) for name in fallback_chain],
        retry=_retry_policy_from_config(raw_policy.get("retry") if isinstance(raw_policy.get("retry"), dict) else None),
        allow_rules_fallback=bool(raw_policy.get("allow_rules_fallback", False)),
    )
    return profiles, policy


def invoke_model_runtime(
    prompt: str,
    *,
    runtime_config: dict[str, Any],
    trace_callback: TraceCallback | None = None,
    runtime: ModelRuntime | None = None,
) -> ModelRuntimeResult:
    profiles, policy = runtime_parts_from_config(runtime_config)
    return (runtime or ModelRuntime()).invoke(prompt, profiles=profiles, policy=policy, trace_callback=trace_callback)


def invoke_model(prompt: str, *, model: str, provider: str | None = None, api_key_env: str | None = None, base_url: str | None = None, timeout_seconds: int = 60) -> str:
    env_api_key = os.environ.get("DEEPAGENT_MEMORY_API_KEY_ENV")
    env_base_url = os.environ.get("DEEPAGENT_MEMORY_BASE_URL")
    env_timeout = os.environ.get("DEEPAGENT_MEMORY_TIMEOUT_SECONDS")
    timeout = int(env_timeout) if env_timeout else timeout_seconds
    config = parse_model_ref(model, provider=provider, api_key_env=api_key_env or env_api_key, base_url=base_url or env_base_url, timeout_seconds=timeout)
    return build_model_provider(config).invoke(prompt)


def _api_key(env_var: str) -> str:
    value = os.environ.get(env_var)
    if not value:
        raise ModelAuthError(f"Missing API key environment variable: {env_var}")
    return value


def _post_json(url: str, payload: dict[str, Any], *, headers: dict[str, str], timeout_seconds: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise ModelHTTPError(exc.code, body) from exc
    except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
        raise ModelTimeoutError(f"Model provider timeout: {exc}") from exc
    return json.loads(raw)



def looks_like_inline_secret(value: str | None) -> bool:
    if not value:
        return False
    lowered = value.lower()
    return value.startswith("sk-") or value.startswith("sk_") or "api_key=" in lowered or "bearer " in lowered


def provider_diagnostics(*, provider: str, model: str, api_key_env: str | None, base_url: str | None, timeout_seconds: int, invoke: bool = False) -> dict[str, Any]:
    inline_secret = looks_like_inline_secret(api_key_env)
    api_key_present = bool(api_key_env and not inline_secret and os.environ.get(api_key_env))
    payload: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "api_key_env": api_key_env,
        "api_key_present": api_key_present,
        "api_key_env_looks_like_secret": inline_secret,
        "base_url": base_url,
        "timeout_seconds": timeout_seconds,
        "ok": bool(api_key_present and not inline_secret),
        "invoked": False,
    }
    if inline_secret:
        payload["error"] = "api_key_env must be an environment variable name, not a raw API key"
        return payload
    if not api_key_present:
        payload["error"] = f"missing environment variable: {api_key_env}"
        return payload
    if invoke:
        try:
            raw = invoke_model(
                'Return JSON only: {"candidates": []}',
                model=f"{provider}:{model}" if provider and ":" not in model else model,
                api_key_env=api_key_env,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
            )
            payload["invoked"] = True
            payload["response_preview"] = raw[:500]
        except Exception as exc:  # pragma: no cover - network/provider dependent
            payload["ok"] = False
            payload["error"] = str(exc)
    return payload
