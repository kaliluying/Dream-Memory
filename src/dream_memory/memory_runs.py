from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from .memory_dreaming import write_jsonl_records
from .memory_importers import redact_sensitive, redact_sensitive_text

SAFE_ARTIFACT_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ_") + uuid4().hex[:8]


def runs_root(memory_dir: Path | str) -> Path:
    return Path(memory_dir).expanduser() / "runs"


def validate_artifact_id(value: str, *, label: str) -> str:
    text = str(value)
    if not text or not SAFE_ARTIFACT_ID_RE.fullmatch(text):
        raise ValueError(f"invalid {label}: {value}")
    return text


def safe_candidate_id(candidate_id: str) -> str:
    return validate_artifact_id(candidate_id, label="candidate_id")


def safe_run_id(run_id: str) -> str:
    return validate_artifact_id(run_id, label="run_id")


def run_dir(memory_dir: Path | str, run_id: str) -> Path:
    return runs_root(memory_dir) / safe_run_id(run_id)


def canonical_state(state: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(state)
    run_id = safe_run_id(str(normalized["run_id"]))
    memory_dir = Path(str(normalized["memory_dir"])).expanduser()
    normalized["run_id"] = run_id
    normalized["memory_dir"] = str(memory_dir)
    normalized["run_dir"] = str(run_dir(memory_dir, run_id))
    return normalized


def state_path(memory_dir: Path | str, run_id: str) -> Path:
    return run_dir(memory_dir, run_id) / "state.json"


def trace_path(memory_dir: Path | str, run_id: str) -> Path:
    return run_dir(memory_dir, run_id) / "trace.jsonl"


def candidate_trace_path(memory_dir: Path | str, run_id: str, candidate_id: str) -> Path:
    return run_dir(memory_dir, run_id) / "candidates" / f"{safe_candidate_id(candidate_id)}.json"


def read_candidate_trace(memory_dir: Path | str, run_id: str, candidate_id: str) -> dict[str, Any] | None:
    safe_id = safe_candidate_id(candidate_id)
    path = candidate_trace_path(memory_dir, run_id, safe_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid candidate trace JSON at {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"candidate trace must be a JSON object: {path}")
    payload_id = str(payload.get("candidate_id") or "").strip()
    if payload_id != safe_id:
        raise ValueError(f"candidate trace id mismatch for {safe_id}: {path}")
    return payload


def run_artifact_path(state: dict[str, Any], artifact_key: str, *, strict: bool = True) -> Path | None:
    state = canonical_state(state)
    artifacts = state.get("artifacts", {}) if isinstance(state.get("artifacts"), dict) else {}
    raw_path = artifacts.get(artifact_key)
    if not raw_path:
        return None
    try:
        candidate = Path(str(raw_path)).expanduser().resolve()
        candidate.relative_to(Path(str(state["run_dir"])).expanduser().resolve())
    except (OSError, ValueError) as exc:
        if strict:
            raise ValueError(f"{artifact_key} is outside run directory: {raw_path}") from exc
        return None
    return candidate


def create_run_state(
    *,
    memory_dir: Path | str,
    project: str | None,
    input_path: str | None,
    mode: str,
    model: str,
    invoke_model: bool,
) -> dict[str, Any]:
    run_id = new_run_id()
    directory = run_dir(memory_dir, run_id)
    directory.mkdir(parents=True, exist_ok=True)
    state = {
        "run_id": run_id,
        "status": "created",
        "phase": "created",
        "project": project,
        "input_path": input_path,
        "mode": mode,
        "model": model,
        "invoke_model": invoke_model,
        "memory_dir": str(Path(memory_dir).expanduser()),
        "run_dir": str(directory),
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "artifacts": {},
        "counts": {},
        "next_actions": [],
    }
    save_run_state(state)
    append_trace(state, "run_created", {"project": project, "mode": mode, "model": model, "invoke_model": invoke_model})
    return state


def load_run_state(memory_dir: Path | str, run_id: str) -> dict[str, Any]:
    path = state_path(memory_dir, run_id)
    if not path.exists():
        raise FileNotFoundError(f"Run state not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Run state must be a JSON object: {path}")
    payload["run_id"] = run_id
    payload["memory_dir"] = str(Path(memory_dir).expanduser())
    return canonical_state(payload)


def save_run_state(state: dict[str, Any]) -> Path:
    state = canonical_state(state)
    path = Path(str(state["run_dir"])) / "state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(redact_sensitive(state))
    state["updated_at"] = utc_now()
    tmp = path.with_name(".state.json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def update_run_state(state: dict[str, Any], *, status: str | None = None, phase: str | None = None, artifacts: dict[str, str] | None = None, counts: dict[str, int] | None = None, next_actions: list[str] | None = None, error: str | None = None) -> dict[str, Any]:
    state = canonical_state(state)
    if status is not None:
        state["status"] = status
    if phase is not None:
        state["phase"] = phase
    if artifacts:
        merged = dict(state.get("artifacts", {}))
        merged.update(artifacts)
        state["artifacts"] = merged
    if counts:
        merged_counts = dict(state.get("counts", {}))
        merged_counts.update(counts)
        state["counts"] = merged_counts
    if next_actions is not None:
        state["next_actions"] = next_actions
    if error is not None:
        state["error"] = redact_sensitive_text(error)
    save_run_state(state)
    return load_run_state(state["memory_dir"], state["run_id"])


def append_trace(state: dict[str, Any], event_type: str, payload: dict[str, Any] | None = None) -> Path:
    state = canonical_state(state)
    path = Path(str(state["run_dir"])) / "trace.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "run_id": state["run_id"],
        "event_type": event_type,
        "timestamp": utc_now(),
        "payload": redact_sensitive(payload or {}),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_trace(
    memory_dir: Path | str,
    run_id: str,
    *,
    candidate_id: str | None = None,
    strict: bool = True,
) -> list[dict[str, Any]]:
    path = trace_path(memory_dir, run_id)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                if strict:
                    raise ValueError(f"invalid trace JSON at {path}:{line_number}") from exc
                continue
            if not isinstance(row, dict):
                if strict:
                    raise ValueError(f"invalid trace row at {path}:{line_number}: expected object")
                continue
            payload = row.get("payload", {})
            if not isinstance(payload, dict):
                if strict:
                    raise ValueError(f"invalid trace payload at {path}:{line_number}: expected object")
                continue
            if candidate_id and payload.get("candidate_id") != candidate_id:
                continue
            rows.append(row)
    return rows


def list_runs(memory_dir: Path | str) -> list[dict[str, Any]]:
    root = runs_root(memory_dir)
    if not root.exists():
        return []
    runs: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/state.json"), reverse=True):
        try:
            run_id = safe_run_id(path.parent.name)
        except ValueError:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            directory = run_dir(memory_dir, run_id)
            runs.append({
                "run_id": run_id,
                "status": "invalid",
                "phase": "invalid_state",
                "project": None,
                "input_path": None,
                "mode": None,
                "model": None,
                "invoke_model": False,
                "memory_dir": str(Path(memory_dir).expanduser()),
                "run_dir": str(directory),
                "created_at": None,
                "updated_at": None,
                "artifacts": {},
                "counts": {},
                "next_actions": ["inspect state.json", "repair or remove the run directory"],
                "error": redact_sensitive_text(f"invalid run state JSON at {path}: line {exc.lineno} column {exc.colno}"),
            })
            continue
        if isinstance(payload, dict):
            payload["run_id"] = run_id
            payload["memory_dir"] = str(Path(memory_dir).expanduser())
            try:
                runs.append(canonical_state(payload))
            except ValueError:
                continue
        else:
            directory = run_dir(memory_dir, run_id)
            runs.append({
                "run_id": run_id,
                "status": "invalid",
                "phase": "invalid_state",
                "project": None,
                "input_path": None,
                "mode": None,
                "model": None,
                "invoke_model": False,
                "memory_dir": str(Path(memory_dir).expanduser()),
                "run_dir": str(directory),
                "created_at": None,
                "updated_at": None,
                "artifacts": {},
                "counts": {},
                "next_actions": ["inspect state.json", "repair or remove the run directory"],
                "error": redact_sensitive_text(f"run state must be a JSON object: {path}"),
            })
    return runs


def copy_input_events(input_path: Path | str, state: dict[str, Any]) -> Path:
    state = canonical_state(state)
    source = Path(input_path).expanduser()
    target = Path(str(state["run_dir"])) / "events.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target


def write_candidate_traces(state: dict[str, Any], candidates: list[dict[str, Any]]) -> None:
    state = canonical_state(state)
    candidates_dir = Path(str(state["run_dir"])) / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        candidate_id = str(candidate.get("id") or "candidate")
        try:
            path = candidate_trace_path(Path(str(state["memory_dir"])), str(state["run_id"]), candidate_id)
        except ValueError as exc:
            append_trace(state, "candidate_trace_skipped", {"candidate_id": candidate_id, "error": str(exc)})
            continue
        payload = {
            "run_id": state["run_id"],
            "candidate_id": candidate_id,
            "candidate": redact_sensitive(candidate),
            "lineage": {
                "events_path": state.get("artifacts", {}).get("events_path"),
                "prompt_path": state.get("artifacts", {}).get("ai_prompt_path"),
                "raw_response_path": state.get("artifacts", {}).get("ai_raw_response_path"),
                "candidates_path": state.get("artifacts", {}).get("candidates_path"),
            },
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        append_trace(state, "candidate_ready", {"candidate_id": candidate_id, "status": candidate.get("status"), "type": candidate.get("type")})
