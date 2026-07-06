from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(".dream-memory/config.json")

DEFAULT_RETRY_POLICY: dict[str, Any] = {
    "max_attempts": 3,
    "initial_delay_seconds": 1.0,
    "backoff_factor": 2.0,
    "max_delay_seconds": 8.0,
    "retry_on_status": [429, 500, 502, 503, 504],
    "retry_on_timeout": True,
}

DEFAULT_MEMORY_CONFIG: dict[str, Any] = {
    "provider": "anthropic",
    "model": "claude-sonnet-4-6",
    "api_key_env": "ANTHROPIC_API_KEY",
    "base_url": None,
    "timeout_seconds": 60,
    "invoke_model": True,
    "mode": "ai",
    "output_dir": ".dream-memory",
    "imports_output_dir": ".dream-memory/imports",
    "memory_cards": ".dream-memory/memory_cards.jsonl",
    "context_limit": 12,
    "context_format": "json",
    "auto_export": False,
    "export_target": "both",
    "export_scope": "project",
    "export_output_dir": None,
    "codex_home": "~/.codex",
    "claude_home": "~/.claude",
    "claude_state": "~/.claude.json",
    "models": {},
    "model_policy": {
        "default_profile": "default",
        "fallback_chain": ["default"],
        "retry": DEFAULT_RETRY_POLICY,
        "allow_rules_fallback": False,
    },
}


def _flat_model_profile(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "provider": config["provider"],
        "model": config["model"],
        "api_key_env": config.get("api_key_env"),
        "base_url": config.get("base_url"),
        "timeout_seconds": config.get("timeout_seconds", 60),
    }


def normalize_memory_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(config)
    flat_profile = _flat_model_profile(normalized)
    loaded_models = normalized.get("models")
    if isinstance(loaded_models, dict) and loaded_models:
        models: dict[str, dict[str, Any]] = {}
        for name, value in loaded_models.items():
            if not isinstance(value, dict):
                continue
            profile = dict(flat_profile)
            for key in ("provider", "model", "api_key_env", "base_url", "timeout_seconds"):
                if key in value:
                    profile[key] = value[key]
            models[str(name)] = profile
        if not models:
            models = {"default": flat_profile}
    else:
        models = {"default": flat_profile}
    normalized["models"] = models

    loaded_policy = normalized.get("model_policy")
    default_profile = "default" if "default" in models else next(iter(models))
    policy: dict[str, Any] = {
        "default_profile": default_profile,
        "fallback_chain": [default_profile],
        "retry": deepcopy(DEFAULT_RETRY_POLICY),
        "allow_rules_fallback": False,
    }
    if isinstance(loaded_policy, dict):
        if loaded_policy.get("default_profile") in models:
            policy["default_profile"] = loaded_policy["default_profile"]
        fallback_chain = loaded_policy.get("fallback_chain")
        if isinstance(fallback_chain, list):
            valid_chain = [str(name) for name in fallback_chain if str(name) in models]
            if valid_chain:
                policy["fallback_chain"] = valid_chain
        elif policy["default_profile"] in models:
            policy["fallback_chain"] = [policy["default_profile"]]
        retry = loaded_policy.get("retry")
        if isinstance(retry, dict):
            merged_retry = deepcopy(DEFAULT_RETRY_POLICY)
            for key, value in retry.items():
                if key in merged_retry:
                    merged_retry[key] = value
            policy["retry"] = merged_retry
        if "allow_rules_fallback" in loaded_policy:
            policy["allow_rules_fallback"] = bool(loaded_policy["allow_rules_fallback"])
    if policy["default_profile"] not in policy["fallback_chain"]:
        policy["fallback_chain"] = [policy["default_profile"], *policy["fallback_chain"]]
    normalized["model_policy"] = policy
    return normalized


def load_memory_config(path: Path | str | None = None) -> dict[str, Any]:
    config_path = Path(path).expanduser() if path else DEFAULT_CONFIG_PATH
    config = deepcopy(DEFAULT_MEMORY_CONFIG)
    if not config_path.exists():
        return normalize_memory_config(config)
    try:
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid memory config JSON: {config_path}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"Memory config must be a JSON object: {config_path}")
    for key, value in loaded.items():
        if key in config:
            config[key] = value
    return normalize_memory_config(config)


def write_default_memory_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Path:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(normalize_memory_config(DEFAULT_MEMORY_CONFIG), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output
