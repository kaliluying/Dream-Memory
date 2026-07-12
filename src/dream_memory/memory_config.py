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
    "switch_model_on_retry": False,
}

DEFAULT_MEMORY_CONFIG: dict[str, Any] = {
    "invoke_model": True,
    "mode": "ai",
    "default_input": ".dream-memory/imports/all-events.jsonl",
    "default_project": ".",
    "project_roots": [],
    "check_provider_invoke": False,
    "check_provider_all": False,
    "check_provider_profile": "primary",
    "init_path": ".",
    "init_force": False,
    "init_config_output": ".dream-memory/config.json",
    "scan_output": None,
    "import_source": "all",
    "import_dry_run": False,
    "dream_apply": False,
    "extract_input": ".dream-memory/imports/all-events.jsonl",
    "extract_project": ".",
    "extract_output_dir": ".dream-memory",
    "review_candidates": ".dream-memory/ai-candidates.jsonl",
    "apply_reviewed": ".dream-memory/reviewed.jsonl",
    "reviewer": "user",
    "status_run_id": None,
    "resume_run_id": None,
    "resume_reviewed": None,
    "trace_run_id": None,
    "trace_candidate_id": None,
    "summary_scope": "all-projects",
    "summary_output": None,
    "eval_input": None,
    "eval_project": ".",
    "eval_mode": "rules",
    "eval_output": None,
    "eval_max_rows": None,
    "eval_max_attempts": None,
    "eval_continue_on_error": False,
    "eval_fallback_rules_on_error": False,
    "eval_fallback_rules_on_empty": False,
    "output_dir": ".dream-memory",
    "imports_output_dir": ".dream-memory/imports",
    "memory_cards": ".dream-memory/memory_cards.jsonl",
    "context_limit": 12,
    "context_format": "json",
    "auto_export": False,
    "export_target": "both",
    "export_scope": "project",
    "export_limit": None,
    "export_output_dir": None,
    "codex_home": "~/.codex",
    "claude_home": "~/.claude",
    "claude_state": "~/.claude.json",
    "models": {
        "primary": {
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
            "api_key": "",
            "api_key_env": None,
            "base_url": None,
            "timeout_seconds": 60,
        }
    },
    "model_policy": {
        "default_profile": "primary",
        "fallback_chain": ["primary"],
        "retry": DEFAULT_RETRY_POLICY,
        "allow_rules_fallback": False,
    },
}


def normalize_memory_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(config)
    loaded_models = normalized.get("models")
    if not isinstance(loaded_models, dict) or not loaded_models:
        raise ValueError("Memory config requires non-empty models")
    models: dict[str, dict[str, Any]] = {}
    for name, value in loaded_models.items():
        if not isinstance(value, dict):
            raise ValueError(f"Model profile must be an object: {name}")
        try:
            timeout_seconds = int(value.get("timeout_seconds", 60))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Model profile timeout_seconds must be a positive integer: {name}") from exc
        if timeout_seconds < 1:
            raise ValueError(f"Model profile timeout_seconds must be a positive integer: {name}")
        profile = {
            "provider": value.get("provider"),
            "model": value.get("model"),
            "api_key": value.get("api_key"),
            "api_key_env": value.get("api_key_env"),
            "base_url": value.get("base_url"),
            "timeout_seconds": timeout_seconds,
        }
        if not profile["provider"] or not profile["model"]:
            raise ValueError(f"Model profile requires provider and model: {name}")
        models[str(name)] = profile
    normalized["models"] = models

    loaded_policy = normalized.get("model_policy")
    if not isinstance(loaded_policy, dict):
        raise ValueError("Memory config requires model_policy")
    default_profile = str(loaded_policy.get("default_profile") or "")
    if default_profile not in models:
        raise ValueError(f"model_policy.default_profile is not configured in models: {default_profile}")
    fallback_chain = loaded_policy.get("fallback_chain")
    if not isinstance(fallback_chain, list) or not fallback_chain:
        raise ValueError("model_policy.fallback_chain must include at least one profile")
    unknown_profiles = [str(name) for name in fallback_chain if str(name) not in models]
    if unknown_profiles:
        raise ValueError(f"model_policy.fallback_chain contains unknown profiles: {', '.join(unknown_profiles)}")
    policy: dict[str, Any] = {
        "default_profile": default_profile,
        "fallback_chain": [str(name) for name in fallback_chain],
        "retry": deepcopy(DEFAULT_RETRY_POLICY),
        "allow_rules_fallback": False,
    }
    retry = loaded_policy.get("retry")
    if isinstance(retry, dict):
        merged_retry = deepcopy(DEFAULT_RETRY_POLICY)
        for key, value in retry.items():
            if key in merged_retry:
                merged_retry[key] = value
        _validate_retry_policy(merged_retry)
        policy["retry"] = merged_retry
    if "allow_rules_fallback" in loaded_policy:
        policy["allow_rules_fallback"] = bool(loaded_policy["allow_rules_fallback"])
    if policy["default_profile"] not in policy["fallback_chain"]:
        policy["fallback_chain"] = [policy["default_profile"], *policy["fallback_chain"]]
    normalized["model_policy"] = policy
    return normalized


def _validate_retry_policy(retry: dict[str, Any]) -> None:
    try:
        max_attempts = int(retry.get("max_attempts", DEFAULT_RETRY_POLICY["max_attempts"]))
        initial_delay = float(retry.get("initial_delay_seconds", DEFAULT_RETRY_POLICY["initial_delay_seconds"]))
        backoff_factor = float(retry.get("backoff_factor", DEFAULT_RETRY_POLICY["backoff_factor"]))
        max_delay = float(retry.get("max_delay_seconds", DEFAULT_RETRY_POLICY["max_delay_seconds"]))
        retry_on_status = retry.get("retry_on_status", DEFAULT_RETRY_POLICY["retry_on_status"])
        statuses = [int(status) for status in retry_on_status]
    except (TypeError, ValueError) as exc:
        raise ValueError("model_policy.retry contains invalid numeric values") from exc
    if max_attempts < 1:
        raise ValueError("model_policy.retry.max_attempts must be at least 1")
    if initial_delay < 0:
        raise ValueError("model_policy.retry.initial_delay_seconds must be non-negative")
    if backoff_factor < 1:
        raise ValueError("model_policy.retry.backoff_factor must be at least 1")
    if max_delay < 0:
        raise ValueError("model_policy.retry.max_delay_seconds must be non-negative")
    if not statuses:
        raise ValueError("model_policy.retry.retry_on_status must not be empty")


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
    if any(key in loaded for key in ("provider", "model", "api_key", "api_key_env", "base_url", "timeout_seconds")) and "models" not in loaded:
        raise ValueError("Memory config no longer supports flat model fields; configure models and model_policy")
    for key, value in loaded.items():
        if key in config:
            config[key] = value
    return normalize_memory_config(config)


def write_default_memory_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Path:
    output = Path(path).expanduser()
    if output.exists() and not output.is_file():
        raise FileExistsError(f"config path is not writable: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.tmp")
    try:
        tmp.write_text(json.dumps(normalize_memory_config(DEFAULT_MEMORY_CONFIG), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(output)
    except OSError as exc:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise FileExistsError(f"config path is not writable: {output}") from exc
    return output
