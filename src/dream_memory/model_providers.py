from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model: str
    api_key_env: str | None = None
    api_key: str | None = None
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
    switch_model_on_retry: bool = False


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

SUPPORTED_MODEL_PROVIDERS = {"anthropic", "openai", "openrouter"}

MODEL_CATALOG: dict[str, dict[str, Any]] = {
    "anthropic": {
        "label": "Anthropic",
        "models": [
            "claude-sonnet-4-6",
            "claude-opus-4-1",
            "claude-haiku-4-5",
            "claude-3-5-sonnet-latest",
        ],
    },
    "openai": {
        "label": "OpenAI",
        "models": [
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4o",
            "gpt-4o-mini",
            "o3-mini",
        ],
    },
    "openrouter": {
        "label": "OpenRouter",
        "models": [
            "openai/gpt-4.1",
            "anthropic/claude-sonnet-4.5",
            "google/gemini-2.5-pro",
            "meta-llama/llama-3.3-70b-instruct",
            "nvidia/nemotron-3-ultra-550b-a55b:free",
        ],
    },
}
class AnthropicProvider:
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    def invoke(self, prompt: str) -> str:
        api_key = _api_key(self.config.api_key, self.config.api_key_env or "ANTHROPIC_API_KEY")
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
        _raise_provider_payload_error(data)
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
        api_key = _api_key(self.config.api_key, self.config.api_key_env or default_env)
        default_url = "https://openrouter.ai/api/v1/chat/completions" if provider == "openrouter" else "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        data = _post_json(
            _openai_chat_completions_url(self.config.base_url, default_url),
            payload,
            headers={
                "authorization": f"Bearer {api_key}",
                "content-type": "application/json",
            },
            timeout_seconds=self.config.timeout_seconds,
        )
        _raise_provider_payload_error(data)
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    return str(message.get("content") or "")
                return str(first.get("text") or "")
        return str(data)


def parse_model_ref(
    model_ref: str,
    *,
    provider: str | None = None,
    api_key_env: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout_seconds: int = 60,
) -> ProviderConfig:
    if provider:
        model = model_ref.split(":", 1)[1] if ":" in model_ref and model_ref.split(":", 1)[0] == provider else model_ref
        return ProviderConfig(provider=provider, model=model, api_key_env=api_key_env, api_key=api_key, base_url=base_url, timeout_seconds=timeout_seconds)
    if ":" in model_ref:
        parsed_provider, parsed_model = model_ref.split(":", 1)
        if parsed_provider in SUPPORTED_MODEL_PROVIDERS:
            return ProviderConfig(provider=parsed_provider, model=parsed_model, api_key_env=api_key_env, api_key=api_key, base_url=base_url, timeout_seconds=timeout_seconds)
    return ProviderConfig(provider="anthropic", model=model_ref, api_key_env=api_key_env, api_key=api_key, base_url=base_url, timeout_seconds=timeout_seconds)


def build_model_provider(config: ProviderConfig) -> Callable[[str], str]:
    if config.provider == "anthropic":
        return AnthropicProvider(config).invoke
    if config.provider in {"openai", "openrouter"}:
        return OpenAICompatibleProvider(config).invoke
    raise ValueError(f"Unsupported model provider: {config.provider}")


def _call_provider(provider: Callable[[str], str] | Any, prompt: str) -> str:
    return provider(prompt) if callable(provider) else provider.invoke(prompt)


class ModelRuntime:
    def __init__(
        self,
        *,
        provider_factory: Callable[[ProviderConfig], Callable[[str], str] | Any] = build_model_provider,
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
        if policy.retry.switch_model_on_retry:
            result = self._invoke_switching_profiles(prompt, profiles=profiles, policy=policy, chain=chain, attempts=attempts, trace_callback=trace_callback)
            if result is not None:
                selected_profile, text = result
                elapsed_ms = int((time.monotonic() - started) * 1000)
                return ModelRuntimeResult(text=text, selected_profile=selected_profile, attempts=attempts, elapsed_ms=elapsed_ms)
            self._emit(trace_callback, "model_runtime_failed", {"attempt_count": len(attempts), "profiles": chain})
            raise ModelRuntimeError(attempts)
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

    def _invoke_switching_profiles(
        self,
        prompt: str,
        *,
        profiles: dict[str, ModelProfile],
        policy: ModelPolicy,
        chain: list[str],
        attempts: list[ModelAttempt],
        trace_callback: TraceCallback | None,
    ) -> tuple[str, str] | None:
        delay = policy.retry.initial_delay_seconds
        max_attempts = max(1, int(policy.retry.max_attempts))
        visited_profiles: set[str] = set()
        for attempt_index in range(1, max_attempts + 1):
            profile_name = chain[(attempt_index - 1) % len(chain)]
            visited_profiles.add(profile_name)
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
            result = self._invoke_profile_once(prompt, profile=profile, attempt_index=1, policy=policy, attempts=attempts, trace_callback=trace_callback)
            if result is not None:
                return profile.name, result
            last_attempt = attempts[-1] if attempts else None
            if last_attempt is None or not last_attempt.retryable:
                return None
            if attempt_index == max_attempts:
                return None
            next_profile = chain[attempt_index % len(chain)]
            self._emit(
                trace_callback,
                "model_fallback_used",
                {"from_profile": profile_name, "to_profile": next_profile, "reason": "retry_switch_model"},
            )
            self.sleeper(min(delay, policy.retry.max_delay_seconds))
            delay *= policy.retry.backoff_factor

        for profile_name in chain:
            if profile_name in visited_profiles:
                continue
            profile = profiles.get(profile_name)
            if profile is None:
                continue
            result = self._invoke_profile(prompt, profile=profile, policy=policy, attempts=attempts, trace_callback=trace_callback)
            if result is not None:
                return profile.name, result
        return None

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
            result = self._invoke_profile_once(prompt, profile=profile, attempt_index=attempt_index, policy=policy, attempts=attempts, trace_callback=trace_callback)
            if result is not None:
                return result
            last_attempt = attempts[-1] if attempts else None
            if last_attempt is None or not last_attempt.retryable or attempt_index == max_attempts:
                return None
            self.sleeper(min(delay, policy.retry.max_delay_seconds))
            delay *= policy.retry.backoff_factor
        return None

    def _invoke_profile_once(
        self,
        prompt: str,
        *,
        profile: ModelProfile,
        attempt_index: int,
        policy: ModelPolicy,
        attempts: list[ModelAttempt],
        trace_callback: TraceCallback | None,
    ) -> str | None:
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
                provider = self.provider_factory(profile.config)
                text = _call_provider(provider, prompt)
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
                return None
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
        switch_model_on_retry=bool(value.get("switch_model_on_retry", False)),
    )


def runtime_parts_from_config(config: dict[str, Any]) -> tuple[dict[str, ModelProfile], ModelPolicy]:
    raw_models = config.get("models")
    if not isinstance(raw_models, dict) or not raw_models:
        raise ValueError("Memory config requires non-empty models")
    profiles: dict[str, ModelProfile] = {}
    for name, value in raw_models.items():
        if not isinstance(value, dict):
            raise ValueError(f"Model profile {name} must be a mapping")
        if not value.get("model"):
            raise ValueError(f"Model profile {name} requires model")
        profiles[str(name)] = ModelProfile(
            str(name),
            ProviderConfig(
                provider=str(value.get("provider") or "anthropic"),
                model=str(value.get("model") or ""),
                api_key_env=str(value["api_key_env"]) if value.get("api_key_env") else None,
                api_key=str(value["api_key"]) if value.get("api_key") else None,
                base_url=str(value["base_url"]) if value.get("base_url") else None,
                timeout_seconds=int(value.get("timeout_seconds", 60)),
            ),
        )
    if not profiles:
        raise ValueError("No model profiles configured")
    raw_policy = config.get("model_policy")
    if not isinstance(raw_policy, dict):
        raise ValueError("Memory config requires model_policy")
    default_profile = str(raw_policy.get("default_profile") or "")
    if not default_profile:
        raise ValueError("model_policy.default_profile is required")
    if default_profile not in profiles:
        raise ValueError(f"model_policy.default_profile references unknown profile: {default_profile}")
    fallback_chain = raw_policy.get("fallback_chain")
    if not isinstance(fallback_chain, list) or not fallback_chain:
        raise ValueError("model_policy.fallback_chain must be a non-empty list")
    normalized_chain = [str(name) for name in fallback_chain]
    unknown_profiles = [name for name in normalized_chain if name not in profiles]
    if unknown_profiles:
        raise ValueError(f"model_policy.fallback_chain references unknown profiles: {', '.join(unknown_profiles)}")
    policy = ModelPolicy(
        default_profile=default_profile,
        fallback_chain=normalized_chain,
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


def invoke_model(
    prompt: str,
    *,
    model: str,
    provider: str | None = None,
    api_key_env: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout_seconds: int = 60,
) -> str:
    env_api_key = os.environ.get("DEEPAGENT_MEMORY_API_KEY_ENV")
    env_base_url = os.environ.get("DEEPAGENT_MEMORY_BASE_URL")
    env_timeout = os.environ.get("DEEPAGENT_MEMORY_TIMEOUT_SECONDS")
    timeout = int(env_timeout) if env_timeout else timeout_seconds
    config = parse_model_ref(model, provider=provider, api_key_env=api_key_env or env_api_key, api_key=api_key, base_url=base_url or env_base_url, timeout_seconds=timeout)
    return _call_provider(build_model_provider(config), prompt)


def list_provider_models(config: ProviderConfig) -> list[str]:
    if config.provider == "anthropic":
        api_key = _api_key(config.api_key, config.api_key_env or "ANTHROPIC_API_KEY")
        data = _get_json(
            _anthropic_models_url(config.base_url),
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            timeout_seconds=config.timeout_seconds,
        )
        return _model_ids_from_payload(data)
    if config.provider in {"openai", "openrouter"}:
        default_env = "OPENROUTER_API_KEY" if config.provider == "openrouter" else "OPENAI_API_KEY"
        api_key = _api_key(config.api_key, config.api_key_env or default_env)
        default_url = "https://openrouter.ai/api/v1/models" if config.provider == "openrouter" else "https://api.openai.com/v1/models"
        data = _get_json(
            _openai_models_url(config.base_url, default_url),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout_seconds=config.timeout_seconds,
        )
        return _model_ids_from_payload(data)
    raise ValueError(f"Unsupported model provider: {config.provider}")


def _api_key(api_key: str | None, env_var: str) -> str:
    if api_key:
        return api_key
    value = os.environ.get(env_var)
    if not value:
        raise ModelAuthError(f"Missing API key environment variable: {env_var}")
    return value


def _raise_provider_payload_error(data: dict[str, Any]) -> None:
    error = data.get("error")
    if not error:
        return
    if isinstance(error, dict):
        code = error.get("code")
        message = str(error.get("message") or error)
        try:
            status_code = int(code)
        except (TypeError, ValueError):
            raise ModelProviderError(f"Model provider returned error: {message}")
        raise ModelHTTPError(status_code, message)
    raise ModelProviderError(f"Model provider returned error: {error}")


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
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        preview = raw[:200].replace("\n", "\\n")
        raise ModelProviderError(f"Model provider returned non-JSON response from {url}: {preview}") from exc


def _get_json(url: str, *, headers: dict[str, str], timeout_seconds: int) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise ModelHTTPError(exc.code, body) from exc
    except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
        raise ModelTimeoutError(f"Model provider timeout: {exc}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        preview = raw[:200].replace("\n", "\\n")
        raise ModelProviderError(f"Model provider returned non-JSON response from {url}: {preview}") from exc


def _openai_chat_completions_url(base_url: str | None, default_url: str) -> str:
    if not base_url:
        return default_url
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def _openai_models_url(base_url: str | None, default_url: str) -> str:
    if not base_url:
        return default_url
    normalized = base_url.rstrip("/")
    if normalized.endswith("/models"):
        return normalized
    if normalized.endswith("/chat/completions"):
        return f"{normalized.removesuffix('/chat/completions')}/models"
    if normalized.endswith("/v1"):
        return f"{normalized}/models"
    return f"{normalized}/v1/models"


def _anthropic_models_url(base_url: str | None) -> str:
    if not base_url:
        return "https://api.anthropic.com/v1/models?limit=100"
    normalized = base_url.rstrip("/")
    if normalized.endswith("/models"):
        return normalized
    if normalized.endswith("/messages"):
        return f"{normalized.removesuffix('/messages')}/models"
    if normalized.endswith("/v1"):
        return f"{normalized}/models"
    return f"{normalized}/v1/models"


def _model_ids_from_payload(data: dict[str, Any]) -> list[str]:
    _raise_provider_payload_error(data)
    raw_items = data.get("data")
    if raw_items is None:
        raw_items = data.get("models")
    if not isinstance(raw_items, list):
        raise ModelProviderError("Model provider response does not include a model list")
    models: set[str] = set()
    for item in raw_items:
        if isinstance(item, str):
            models.add(item)
        elif isinstance(item, dict):
            model_id = item.get("id") or item.get("name")
            if model_id:
                models.add(str(model_id))
    if not models:
        raise ModelProviderError("Model provider returned an empty model list")
    return sorted(models)



def looks_like_inline_secret(value: str | None) -> bool:
    if not value:
        return False
    lowered = value.lower()
    return value.startswith("sk-") or value.startswith("sk_") or "api_key=" in lowered or "bearer " in lowered


def provider_diagnostics(
    *,
    provider: str,
    model: str,
    api_key_env: str | None,
    api_key: str | None = None,
    base_url: str | None,
    timeout_seconds: int,
    invoke: bool = False,
) -> dict[str, Any]:
    inline_secret = looks_like_inline_secret(api_key_env)
    direct_key_present = bool(api_key)
    env_key_present = bool(api_key_env and not inline_secret and os.environ.get(api_key_env))
    api_key_present = direct_key_present or env_key_present
    payload: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "api_key_env": api_key_env,
        "api_key_configured": direct_key_present,
        "api_key_present": api_key_present,
        "api_key_env_looks_like_secret": inline_secret,
        "base_url": base_url,
        "timeout_seconds": timeout_seconds,
        "ok": bool(api_key_present and not inline_secret),
        "invoked": False,
    }
    if inline_secret:
        payload["error"] = "api_key_env must be an environment variable name; use api_key for direct keys"
        return payload
    if not api_key_present:
        payload["error"] = f"missing api_key or environment variable: {api_key_env}"
        return payload
    if invoke:
        try:
            raw = invoke_model(
                'Return JSON only: {"candidates": []}',
                provider=provider,
                model=model,
                api_key_env=api_key_env,
                api_key=api_key,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
            )
            payload["invoked"] = True
            payload["response_preview"] = raw[:500]
        except Exception as exc:  # pragma: no cover - network/provider dependent
            payload["ok"] = False
            payload["error"] = str(exc)
    return payload
