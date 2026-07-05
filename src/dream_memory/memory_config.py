from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(".dream-memory/config.json")

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
}


def load_memory_config(path: Path | str | None = None) -> dict[str, Any]:
    config_path = Path(path).expanduser() if path else DEFAULT_CONFIG_PATH
    config = dict(DEFAULT_MEMORY_CONFIG)
    if not config_path.exists():
        return config
    try:
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid memory config JSON: {config_path}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"Memory config must be a JSON object: {config_path}")
    for key, value in loaded.items():
        if key in config:
            config[key] = value
    return config


def write_default_memory_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Path:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(DEFAULT_MEMORY_CONFIG, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output
