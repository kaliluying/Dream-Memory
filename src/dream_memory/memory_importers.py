from __future__ import annotations

import json
import re
import sqlite3
import urllib.parse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .memory_dreaming import load_events_jsonl

SENSITIVE_KEY_RE = re.compile(r"(token|key|secret|password|cookie|auth|credential)", re.I)
SENSITIVE_VALUE_PATTERNS = [
    re.compile(r"(?i)\b[A-Z0-9_-]*(?:api[_-]?key|access[_-]?token|refresh[_-]?token|token|password|secret|cookie)\s*[=:]\s*[^,\s;]+"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"\bsk-[A-Za-z0-9._-]+"),
    re.compile(r"\bsk_[A-Za-z0-9._-]+"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
    re.compile(r"(?i)(密钥|key|token|api[_-]?key).{0,12}(在|文件|path|路径).{0,24}(\.txt|\.env|json|yaml|yml|配置)"),
]
URL_PATTERN = re.compile(r"https?://[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+")


def _redact_url_credentials(value: str) -> str:
    try:
        parsed = urllib.parse.urlsplit(value)
    except ValueError:
        return value
    if not parsed.scheme or not parsed.netloc:
        return value
    hostname = parsed.hostname or ""
    netloc = hostname
    try:
        port = parsed.port
    except ValueError:
        return value
    if port is not None:
        netloc = f"{netloc}:{port}"
    if parsed.username or parsed.password:
        netloc = f"<redacted>@{netloc}"
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    redacted_query = [
        (key, "<redacted>" if SENSITIVE_KEY_RE.search(key) else item)
        for key, item in query
    ]
    return urllib.parse.urlunsplit((
        parsed.scheme,
        netloc,
        parsed.path,
        urllib.parse.urlencode(redacted_query, doseq=True),
        parsed.fragment,
    ))


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
            redacted_key = redact_sensitive_text(str(key))
            if redacted_key != str(key) or SENSITIVE_KEY_RE.search(str(key)):
                output[redacted_key] = "<redacted>"
            else:
                output[redacted_key] = redact_sensitive(item)
        return output
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, str):
        return redact_sensitive_text(value)
    return value


def _redact_sensitive_plain_text(value: str) -> str:
    text = str(value or "")
    for pattern in SENSITIVE_VALUE_PATTERNS:
        text = pattern.sub("<redacted>", text)
    return text


def redact_sensitive_text(value: str) -> str:
    text = str(value or "")
    parts: list[str] = []
    offset = 0
    for match in URL_PATTERN.finditer(text):
        parts.append(_redact_sensitive_plain_text(text[offset:match.start()]))
        parts.append(_redact_url_credentials(match.group(0)))
        offset = match.end()
    parts.append(_redact_sensitive_plain_text(text[offset:]))
    return "".join(parts)


def _read_jsonl(path: Path, *, warnings: list[dict[str, Any]] | None = None) -> Iterable[dict[str, Any]]:
    if not path.exists() or not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                if warnings is not None:
                    warnings.append({"path": str(path), "line": line_number, "error": f"invalid JSON: {exc.msg}"})
                continue
            if isinstance(obj, dict):
                rows.append(obj)
            elif warnings is not None:
                warnings.append({"path": str(path), "line": line_number, "error": "expected JSON object"})
    return rows


def _read_json_object(path: Path, *, warnings: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except json.JSONDecodeError as exc:
        if warnings is not None:
            warnings.append({"path": str(path), "error": f"invalid JSON: {exc.msg}"})
        return {}
    if not isinstance(obj, dict):
        if warnings is not None:
            warnings.append({"path": str(path), "error": "expected JSON object"})
        return {}
    return obj


def _content_parts(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            parts.extend(_content_parts(item))
        return parts
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return [value["text"]]
        if isinstance(value.get("content"), (str, list, dict)):
            return _content_parts(value.get("content"))
        if isinstance(value.get("message"), (str, list, dict)):
            return _content_parts(value.get("message"))
    return []


def _message_content(obj: dict[str, Any]) -> str:
    for key in ["content", "message", "text", "body", "prompt", "response"]:
        parts = _content_parts(obj.get(key))
        if parts:
            return "\n".join(part for part in parts if part.strip())
    return ""

def _strip_generated_memory_block(text: str) -> str:
    start = "<!-- DREAM_MEMORY_START -->"
    end = "<!-- DREAM_MEMORY_END -->"
    while start in text and end in text.split(start, 1)[1]:
        before, rest = text.split(start, 1)
        _, after = rest.split(end, 1)
        text = before + after
    return text.strip()


def _clean_project_instruction_text(text: str) -> str:
    text = _strip_generated_memory_block(str(text or ""))
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if stripped in {"## Dream Memory Context", "--- project-doc ---"}:
            continue
        lines.append(line)
    cleaned = "\n".join(lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _extract_project_instructions(text: str) -> str:
    if not text.startswith("# AGENTS.md instructions"):
        return ""
    start_marker = "<INSTRUCTIONS>"
    end_marker = "</INSTRUCTIONS>"
    if start_marker not in text:
        return ""
    body = text.split(start_marker, 1)[1]
    if end_marker in body:
        body = body.split(end_marker, 1)[0]
    return _clean_project_instruction_text(body)


def _normalize_rollout_content(content: str, *, role: str) -> tuple[str, str]:
    text = str(content or "").strip()
    if not text:
        return "", "rollout_message"
    project_instructions = _extract_project_instructions(text)
    if project_instructions:
        return project_instructions, "project_instruction"
    request_marker = "## My request for Codex:"
    if request_marker in text:
        return text.split(request_marker, 1)[1].strip(), "rollout_message"
    if role == "user" and text.startswith("<environment_context>"):
        return "", "rollout_message"
    return text, "rollout_message"


def _rollout_message(row: dict[str, Any]) -> tuple[str, str, str] | None:
    payload = row.get("payload") if isinstance(row.get("payload"), dict) else None
    if not payload:
        return None
    if row.get("type") != "response_item" or payload.get("type") != "message":
        return None
    role = str(payload.get("role") or "").strip()
    if role not in {"user", "assistant"}:
        return None
    content, event_type = _normalize_rollout_content(_message_content(payload), role=role)
    if not content:
        return None
    return role, content, event_type




def import_project_instruction_events(project_roots: Iterable[Path | str]) -> list[NormalizedSessionEvent]:
    events: list[NormalizedSessionEvent] = []
    seen: set[str] = set()
    for raw_root in project_roots:
        root = Path(raw_root).expanduser()
        root_key = str(root.absolute())
        if root_key in seen:
            continue
        seen.add(root_key)
        for filename, target in [("AGENTS.md", "codex"), ("CLAUDE.md", "claude_code")]:
            path = root / filename
            if not path.exists() or not path.is_file():
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
            instructions = _extract_project_instructions(content) or _clean_project_instruction_text(content)
            if not instructions:
                continue
            events.append(NormalizedSessionEvent(
                source=target,
                session_id=f"project-instruction:{root}:{filename}",
                project=str(root),
                timestamp=None,
                role="system",
                content=instructions,
                event_type="project_instruction",
                metadata={"path": str(path)},
            ))
    return events


def _relative_marker_paths(root: Path, filename: str, *, max_depth: int = 3) -> list[str]:
    paths: list[str] = []
    if not root.exists() or not root.is_dir():
        return paths
    for path in sorted(root.rglob(filename)):
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        if len(rel.parts) > max_depth:
            continue
        if any(part in {".git", ".venv", "venv", "node_modules", ".dream-memory"} for part in rel.parts):
            continue
        paths.append(rel.as_posix())
    return paths


def import_project_marker_events(project_roots: Iterable[Path | str]) -> list[NormalizedSessionEvent]:
    events: list[NormalizedSessionEvent] = []
    seen: set[str] = set()
    for raw_root in project_roots:
        root = Path(raw_root).expanduser()
        root_key = str(root.absolute())
        if root_key in seen:
            continue
        seen.add(root_key)
        pyproject_paths = _relative_marker_paths(root, "pyproject.toml")
        uv_lock_paths = _relative_marker_paths(root, "uv.lock")
        package_json_paths = _relative_marker_paths(root, "package.json")
        pnpm_lock_paths = _relative_marker_paths(root, "pnpm-lock.yaml")
        npm_lock_paths = _relative_marker_paths(root, "package-lock.json")
        yarn_lock_paths = _relative_marker_paths(root, "yarn.lock")
        python_test_paths = [
            path
            for path in _relative_marker_paths(root, "test_*.py") + _relative_marker_paths(root, "*_test.py")
            if path.startswith("tests/") or "/tests/" in path
        ]
        pyproject_text = ""
        for rel in pyproject_paths[:3]:
            try:
                pyproject_text += "\n" + (root / rel).read_text(encoding="utf-8", errors="ignore")[:20000]
            except OSError:
                pass
        test_text = ""
        for rel in python_test_paths[:12]:
            try:
                test_text += "\n" + (root / rel).read_text(encoding="utf-8", errors="ignore")[:4000]
            except OSError:
                pass
        markers: list[str] = []
        if pyproject_paths and uv_lock_paths:
            markers.append("python_package_manager=uv")
        elif pyproject_paths:
            markers.append("python_project=pyproject")
        if pnpm_lock_paths:
            markers.append("frontend_package_manager=pnpm")
        elif npm_lock_paths:
            markers.append("frontend_package_manager=npm")
        elif yarn_lock_paths:
            markers.append("frontend_package_manager=yarn")
        elif package_json_paths:
            markers.append("frontend_project=package_json")
        lower_pyproject = pyproject_text.lower()
        lower_tests = test_text.lower()
        pyproject_configures_pytest = "pytest" in lower_pyproject
        tests_use_pytest = pyproject_configures_pytest or "import pytest" in lower_tests or "from pytest" in lower_tests
        tests_use_unittest = "import unittest" in lower_tests or "unittest.testcase" in lower_tests
        if tests_use_pytest:
            markers.append("python_test_runner=pytest")
        elif tests_use_unittest:
            markers.append("python_test_runner=unittest")
        if "fastapi" in lower_pyproject:
            markers.append("python_framework=fastapi")
        elif "django" in lower_pyproject:
            markers.append("python_framework=django")
        if not markers:
            continue
        frontend_paths = sorted({
            str(Path(path).parent.as_posix())
            for path in package_json_paths + pnpm_lock_paths + npm_lock_paths + yarn_lock_paths
        })
        frontend_paths = ["." if path == "." else path for path in frontend_paths]
        events.append(NormalizedSessionEvent(
            source="project",
            session_id=f"project-markers:{root}",
            project=str(root),
            timestamp=None,
            role="system",
            content="; ".join(markers),
            event_type="project_markers",
            metadata=redact_sensitive({
                "pyproject_paths": pyproject_paths,
                "uv_lock_paths": uv_lock_paths,
                "package_json_paths": package_json_paths,
                "pnpm_lock_paths": pnpm_lock_paths,
                "npm_lock_paths": npm_lock_paths,
                "yarn_lock_paths": yarn_lock_paths,
                "python_test_paths": python_test_paths,
                "frontend_paths": frontend_paths,
            }),
        ))
    return events

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
        self.import_warnings: list[dict[str, Any]] = []

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
            con = None
            try:
                con = sqlite3.connect(self.state_db_path)
                thread_count = con.execute("select count(*) from threads").fetchone()[0]
                try:
                    rollout_count = con.execute("select count(*) from threads where rollout_path is not null and rollout_path != ''").fetchone()[0]
                except sqlite3.Error:
                    rollout_count = 0
            except sqlite3.Error:
                pass
            finally:
                if con is not None:
                    con.close()
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
        con = None
        try:
            con = sqlite3.connect(self.state_db_path)
            con.row_factory = sqlite3.Row
            rows = con.execute("select id, rollout_path, cwd, title, first_user_message, updated_at, model from threads").fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error:
            return []
        finally:
            if con is not None:
                con.close()

    def import_events(self) -> list[NormalizedSessionEvent]:
        self.import_warnings = []
        events: list[NormalizedSessionEvent] = []
        for row in _read_jsonl(self.history_path, warnings=self.import_warnings):
            content = _message_content(row)
            if not content:
                continue
            events.append(NormalizedSessionEvent(
                source="codex",
                session_id=str(row.get("session_id", "history")),
                project=None,
                timestamp=str(row.get("ts")) if row.get("ts") is not None else None,
                role="user",
                content=redact_sensitive_text(content),
                event_type="history_prompt",
                metadata=redact_sensitive({"path": str(self.history_path)}),
            ))

        for thread in self._thread_rows():
            raw_rollout_path = str(thread.get("rollout_path") or "").strip()
            rollout_path = Path(raw_rollout_path) if raw_rollout_path else None
            session_id = str(thread.get("id") or "")
            project = thread.get("cwd")
            if thread.get("first_user_message"):
                events.append(NormalizedSessionEvent(
                    source="codex",
                    session_id=session_id,
                    project=project,
                    timestamp=str(thread.get("updated_at")) if thread.get("updated_at") is not None else None,
                    role="user",
                    content=redact_sensitive_text(str(thread["first_user_message"])),
                    event_type="thread_first_user_message",
                    metadata=redact_sensitive({"title": thread.get("title"), "model": thread.get("model"), "rollout_path": str(rollout_path) if rollout_path is not None else None}),
                ))
            if rollout_path is not None and rollout_path.exists():
                for row in _read_jsonl(rollout_path, warnings=self.import_warnings):
                    rollout_message = _rollout_message(row)
                    if rollout_message is not None:
                        role, content, event_type = rollout_message
                    else:
                        content = _message_content(row)
                        if not content:
                            continue
                        role = str(row.get("role") or row.get("type") or row.get("kind") or "event")
                        event_type = "rollout_event"
                    events.append(NormalizedSessionEvent(
                        source="codex",
                        session_id=session_id,
                        project=project,
                        timestamp=str(row.get("timestamp") or row.get("ts") or thread.get("updated_at")),
                        role=role,
                        content=redact_sensitive_text(content),
                        event_type=event_type,
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
        self.import_warnings: list[dict[str, Any]] = []

    @property
    def transcripts_dir(self) -> Path:
        return self.claude_home / "transcripts"

    @property
    def projects_dir(self) -> Path:
        return self.claude_home / "projects"

    def _transcript_paths(self) -> list[Path]:
        paths: list[Path] = []
        if self.transcripts_dir.exists():
            paths.extend(sorted(self.transcripts_dir.glob("*.jsonl")))
        if self.projects_dir.exists():
            paths.extend(sorted(self.projects_dir.glob("*/*.jsonl")))
        seen: set[str] = set()
        unique: list[Path] = []
        for path in paths:
            key = str(path)
            if key not in seen:
                seen.add(key)
                unique.append(path)
        return unique

    def scan(self) -> dict[str, Any]:
        self.import_warnings = []
        global_projects = 0
        if self.global_state_path.exists():
            obj = _read_json_object(self.global_state_path, warnings=self.import_warnings)
            if isinstance(obj.get("projects"), dict):
                global_projects = len(obj["projects"])
        file_history = self.claude_home / "file-history"
        transcript_paths = self._transcript_paths()
        return {
            "source": "claude_code",
            "home": str(self.claude_home),
            "global_state_found": self.global_state_path.exists(),
            "claude_md_found": (self.claude_home / "CLAUDE.md").exists(),
            "global_project_count": global_projects,
            "file_history_found": file_history.exists(),
            "file_history_project_count": len([path for path in file_history.iterdir() if path.is_dir()]) if file_history.exists() else 0,
            "project_settings_found": [str(root / ".claude" / "settings.local.json") for root in self.project_roots if (root / ".claude" / "settings.local.json").exists()],
            "transcripts_found": bool(transcript_paths),
            "transcript_count": len(transcript_paths),
            "warnings": list(self.import_warnings),
        }

    def _import_transcript_events(self) -> list[NormalizedSessionEvent]:
        events: list[NormalizedSessionEvent] = []
        for path in self._transcript_paths():
            for row in _read_jsonl(path, warnings=self.import_warnings):
                row_type = str(row.get("type") or "")
                if row_type not in {"user", "assistant"}:
                    continue
                if row.get("isSidechain") is True:
                    continue
                content = _message_content(row)
                if not content or "<command-name>/init</command-name>" in content or "<command-message>init</command-message>" in content:
                    continue
                message = row.get("message") if isinstance(row.get("message"), dict) else {}
                role = str(row.get("role") or message.get("role") or row_type)
                project = row.get("cwd")
                session_id = str(row.get("sessionId") or path.stem)
                events.append(NormalizedSessionEvent(
                    source="claude_code",
                    session_id=session_id,
                    project=str(project) if project else None,
                    timestamp=str(row.get("timestamp")) if row.get("timestamp") is not None else None,
                    role=role,
                    content=redact_sensitive_text(content),
                    event_type="transcript_message",
                    metadata=redact_sensitive({"path": str(path), "uuid": row.get("uuid"), "type": row_type}),
                ))
        return events

    def import_events(self) -> list[NormalizedSessionEvent]:
        self.import_warnings = []
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
            obj = _read_json_object(self.global_state_path, warnings=self.import_warnings)
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
                metadata = _read_json_object(settings, warnings=self.import_warnings)
                if not metadata:
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
        events.extend(self._import_transcript_events())
        return events
