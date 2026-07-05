from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

SENSITIVE_KEY_RE = re.compile(r"(token|key|secret|password|cookie|auth|credential)", re.I)


@dataclass(frozen=True)
class NormalizedSessionEvent:
    source: str
    session_id: str
    project: str | None
    timestamp: str | None
    role: str
    content: str
    event_type: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def redact_sensitive(value: Any) -> Any:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if SENSITIVE_KEY_RE.search(str(key)):
                output[key] = "<redacted>"
            else:
                output[key] = redact_sensitive(item)
        return output
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    return value


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _message_content(obj: dict[str, Any]) -> str:
    for key in ["content", "message", "text", "body", "prompt", "response"]:
        value = obj.get(key)
        if isinstance(value, str):
            return value
    if isinstance(obj.get("content"), list):
        parts = []
        for item in obj["content"]:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return ""


def write_events_jsonl(events: Iterable[NormalizedSessionEvent], path: Path | str) -> Path:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
    return output


class CodexImporter:
    def __init__(self, codex_home: Path | str | None = None):
        self.codex_home = Path(codex_home or Path.home() / ".codex").expanduser()

    @property
    def history_path(self) -> Path:
        return self.codex_home / "history.jsonl"

    @property
    def session_index_path(self) -> Path:
        return self.codex_home / "session_index.jsonl"

    @property
    def state_db_path(self) -> Path:
        primary = self.codex_home / "sqlite" / "state_5.sqlite"
        return primary if primary.exists() else self.codex_home / "state_5.sqlite"

    def scan(self) -> dict[str, Any]:
        thread_count = 0
        rollout_count = 0
        if self.state_db_path.exists():
            try:
                with sqlite3.connect(self.state_db_path) as con:
                    thread_count = con.execute("select count(*) from threads").fetchone()[0]
                    try:
                        rollout_count = con.execute("select count(*) from threads where rollout_path is not null and rollout_path != ''").fetchone()[0]
                    except sqlite3.Error:
                        rollout_count = 0
            except sqlite3.Error:
                pass
        memories_dir = self.codex_home / "memories"
        return {
            "source": "codex",
            "home": str(self.codex_home),
            "history_found": self.history_path.exists(),
            "session_index_found": self.session_index_path.exists(),
            "state_db_found": self.state_db_path.exists(),
            "thread_count": thread_count,
            "rollout_path_count": rollout_count,
            "memories_found": memories_dir.exists(),
            "memory_files": [str(path) for path in [memories_dir / "MEMORY.md", memories_dir / "memory_summary.md", memories_dir / "raw_memories.md"] if path.exists()],
        }

    def _thread_rows(self) -> list[dict[str, Any]]:
        if not self.state_db_path.exists():
            return []
        try:
            con = sqlite3.connect(self.state_db_path)
            con.row_factory = sqlite3.Row
            rows = con.execute("select id, rollout_path, cwd, title, first_user_message, updated_at, model from threads").fetchall()
            con.close()
            return [dict(row) for row in rows]
        except sqlite3.Error:
            return []

    def import_events(self) -> list[NormalizedSessionEvent]:
        events: list[NormalizedSessionEvent] = []
        for row in _read_jsonl(self.history_path):
            content = _message_content(row)
            if not content:
                continue
            events.append(NormalizedSessionEvent(
                source="codex",
                session_id=str(row.get("session_id", "history")),
                project=None,
                timestamp=str(row.get("ts")) if row.get("ts") is not None else None,
                role="user",
                content=content,
                event_type="history_prompt",
                metadata=redact_sensitive({"path": str(self.history_path)}),
            ))

        for thread in self._thread_rows():
            rollout_path = Path(thread.get("rollout_path") or "")
            session_id = str(thread.get("id") or "")
            project = thread.get("cwd")
            if thread.get("first_user_message"):
                events.append(NormalizedSessionEvent(
                    source="codex",
                    session_id=session_id,
                    project=project,
                    timestamp=str(thread.get("updated_at")) if thread.get("updated_at") is not None else None,
                    role="user",
                    content=str(thread["first_user_message"]),
                    event_type="thread_first_user_message",
                    metadata=redact_sensitive({"title": thread.get("title"), "model": thread.get("model"), "rollout_path": str(rollout_path) if str(rollout_path) else None}),
                ))
            if rollout_path.exists():
                for row in _read_jsonl(rollout_path):
                    content = _message_content(row)
                    if not content:
                        continue
                    role = str(row.get("role") or row.get("type") or row.get("kind") or "event")
                    events.append(NormalizedSessionEvent(
                        source="codex",
                        session_id=session_id,
                        project=project,
                        timestamp=str(row.get("timestamp") or row.get("ts") or thread.get("updated_at")),
                        role=role,
                        content=content,
                        event_type="rollout_event",
                        metadata=redact_sensitive({"rollout_path": str(rollout_path), "thread_title": thread.get("title")}),
                    ))
        return events


class ClaudeCodeImporter:
    def __init__(
        self,
        claude_home: Path | str | None = None,
        global_state_path: Path | str | None = None,
        project_roots: list[Path | str] | None = None,
    ):
        self.claude_home = Path(claude_home or Path.home() / ".claude").expanduser()
        self.global_state_path = Path(global_state_path or Path.home() / ".claude.json").expanduser()
        self.project_roots = [Path(path).expanduser() for path in (project_roots or [])]

    def scan(self) -> dict[str, Any]:
        global_projects = 0
        if self.global_state_path.exists():
            try:
                obj = json.loads(self.global_state_path.read_text(encoding="utf-8", errors="ignore"))
                if isinstance(obj.get("projects"), dict):
                    global_projects = len(obj["projects"])
            except Exception:
                pass
        file_history = self.claude_home / "file-history"
        return {
            "source": "claude_code",
            "home": str(self.claude_home),
            "global_state_found": self.global_state_path.exists(),
            "claude_md_found": (self.claude_home / "CLAUDE.md").exists(),
            "global_project_count": global_projects,
            "file_history_found": file_history.exists(),
            "file_history_project_count": len([path for path in file_history.iterdir() if path.is_dir()]) if file_history.exists() else 0,
            "project_settings_found": [str(root / ".claude" / "settings.local.json") for root in self.project_roots if (root / ".claude" / "settings.local.json").exists()],
            "transcripts_found": False,
        }

    def import_events(self) -> list[NormalizedSessionEvent]:
        events: list[NormalizedSessionEvent] = []
        claude_md = self.claude_home / "CLAUDE.md"
        if claude_md.exists():
            events.append(NormalizedSessionEvent(
                source="claude_code",
                session_id="global-claude-md",
                project=None,
                timestamp=None,
                role="system",
                content=claude_md.read_text(encoding="utf-8", errors="ignore"),
                event_type="global_instruction",
                metadata={"path": str(claude_md)},
            ))
        if self.global_state_path.exists():
            try:
                obj = json.loads(self.global_state_path.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                obj = {}
            projects = obj.get("projects") if isinstance(obj, dict) else None
            if isinstance(projects, dict):
                for project, metadata in projects.items():
                    events.append(NormalizedSessionEvent(
                        source="claude_code",
                        session_id=f"project-state:{project}",
                        project=project,
                        timestamp=None,
                        role="system",
                        content=f"Claude Code project state for {project}",
                        event_type="project_state",
                        metadata=redact_sensitive(metadata if isinstance(metadata, dict) else {"value": metadata}),
                    ))
        for root in self.project_roots:
            settings = root / ".claude" / "settings.local.json"
            if settings.exists():
                try:
                    metadata = json.loads(settings.read_text(encoding="utf-8", errors="ignore"))
                except Exception:
                    metadata = {"path": str(settings)}
                events.append(NormalizedSessionEvent(
                    source="claude_code",
                    session_id=f"project-settings:{root}",
                    project=str(root),
                    timestamp=None,
                    role="system",
                    content=f"Claude Code local project settings for {root}",
                    event_type="project_settings",
                    metadata=redact_sensitive(metadata),
                ))
        return events
