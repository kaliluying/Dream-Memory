from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


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
        raise RuntimeError(f"Missing API key environment variable: {env_var}")
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
        raise RuntimeError(f"Model provider HTTP {exc.code}: {body}") from exc
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
