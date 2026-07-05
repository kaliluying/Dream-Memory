from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .memory_dreaming import write_jsonl_records


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ_") + uuid4().hex[:8]


def runs_root(memory_dir: Path | str) -> Path:
    return Path(memory_dir).expanduser() / "runs"


def run_dir(memory_dir: Path | str, run_id: str) -> Path:
    return runs_root(memory_dir) / run_id


def state_path(memory_dir: Path | str, run_id: str) -> Path:
    return run_dir(memory_dir, run_id) / "state.json"


def trace_path(memory_dir: Path | str, run_id: str) -> Path:
    return run_dir(memory_dir, run_id) / "trace.jsonl"


def candidate_trace_path(memory_dir: Path | str, run_id: str, candidate_id: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in candidate_id)
    return run_dir(memory_dir, run_id) / "candidates" / f"{safe}.json"


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
    return payload


def save_run_state(state: dict[str, Any]) -> Path:
    path = Path(str(state["run_dir"])) / "state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(state)
    state["updated_at"] = utc_now()
    tmp = path.with_name(".state.json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def update_run_state(state: dict[str, Any], *, status: str | None = None, phase: str | None = None, artifacts: dict[str, str] | None = None, counts: dict[str, int] | None = None, next_actions: list[str] | None = None, error: str | None = None) -> dict[str, Any]:
    state = dict(state)
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
        state["error"] = error
    save_run_state(state)
    return state


def append_trace(state: dict[str, Any], event_type: str, payload: dict[str, Any] | None = None) -> Path:
    path = Path(str(state["run_dir"])) / "trace.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "run_id": state["run_id"],
        "event_type": event_type,
        "timestamp": utc_now(),
        "payload": payload or {},
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_trace(memory_dir: Path | str, run_id: str, *, candidate_id: str | None = None) -> list[dict[str, Any]]:
    path = trace_path(memory_dir, run_id)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if candidate_id and row.get("payload", {}).get("candidate_id") != candidate_id:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def list_runs(memory_dir: Path | str) -> list[dict[str, Any]]:
    root = runs_root(memory_dir)
    if not root.exists():
        return []
    runs: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/state.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            runs.append(payload)
    return runs


def copy_input_events(input_path: Path | str, state: dict[str, Any]) -> Path:
    source = Path(input_path).expanduser()
    target = Path(str(state["run_dir"])) / "events.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return target


def write_candidate_traces(state: dict[str, Any], candidates: list[dict[str, Any]]) -> None:
    candidates_dir = Path(str(state["run_dir"])) / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        candidate_id = str(candidate.get("id") or "candidate")
        path = candidate_trace_path(Path(str(state["memory_dir"])), str(state["run_id"]), candidate_id)
        payload = {
            "run_id": state["run_id"],
            "candidate_id": candidate_id,
            "candidate": candidate,
            "lineage": {
                "events_path": state.get("artifacts", {}).get("events_path"),
                "prompt_path": state.get("artifacts", {}).get("ai_prompt_path"),
                "raw_response_path": state.get("artifacts", {}).get("ai_raw_response_path"),
                "candidates_path": state.get("artifacts", {}).get("candidates_path"),
            },
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        append_trace(state, "candidate_ready", {"candidate_id": candidate_id, "status": candidate.get("status"), "type": candidate.get("type")})


def write_json_artifact(path: Path | str, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output
