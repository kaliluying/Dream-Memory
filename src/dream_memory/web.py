from __future__ import annotations

import json
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

from .memory_cli import _auto_review_run, _resume_run, _run_dream_to_review
from .memory_config import load_memory_config, normalize_memory_config
from .memory_dreaming import load_events_jsonl, normalize_review_decision
from .memory_runs import append_trace, create_run_state, list_runs, load_run_state, read_trace, update_run_state
from .model_providers import ModelProviderError, ProviderConfig, list_provider_models, runtime_parts_from_config


class MemoryReviewRequest(BaseModel):
    candidate_id: str
    action: str
    edited_content: str | None = None
    reviewer: str = "user"
    note: str | None = None
    candidate: dict[str, Any] = Field(default_factory=dict)


class MemoryRunStartRequest(BaseModel):
    input: str
    project: str | None = None
    mode: str | None = None
    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None
    timeout_seconds: int | None = None
    invoke_model: bool | None = None
    memory_cards: str | None = None


class MemoryConfigUpdateRequest(BaseModel):
    config: dict[str, Any] = Field(default_factory=dict)


class MemoryModelsRequest(BaseModel):
    provider: str = "anthropic"
    model: str = ""
    api_key: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None
    timeout_seconds: int | None = None


class MemoryAutoReviewRequest(BaseModel):
    reviewer: str = "auto-review"
    min_score: float = 0.7
    keep_review: bool = False
    include_duplicates: bool = False
    include_merges: bool = False
    force: bool = False


TEMPLATE_DIR = Path(__file__).with_name("templates")


def _template(name: str) -> str:
    return (TEMPLATE_DIR / name).read_text(encoding="utf-8")


def _web_config(memory_dir: Path) -> dict[str, Any]:
    config = load_memory_config(memory_dir / "config.json")
    config["output_dir"] = str(memory_dir)
    config["memory_cards"] = str(memory_dir / "memory_cards.jsonl")
    config["imports_output_dir"] = str(memory_dir / "imports")
    return config


def _config_path(memory_dir: Path) -> Path:
    return memory_dir / "config.json"


def _config_payload(memory_dir: Path) -> dict[str, Any]:
    path = _config_path(memory_dir)
    config = load_memory_config(path)
    return {
        "ok": True,
        "config_path": str(path),
        "config": config,
        "cli_options": {
            "global": ["--config", "--codex-home", "--claude-home", "--claude-state", "--project"],
            "init": ["--path", "--output-dir", "--force"],
            "init-config": ["--output"],
            "check-provider": ["--provider", "--model", "--api-key", "--api-key-env", "--base-url", "--timeout-seconds", "--invoke", "--all", "--profile"],
            "scan": ["--output"],
            "import": ["source", "--output-dir", "--dry-run"],
            "dream": ["--input", "--project", "--output-dir", "--apply", "--mode", "--agent", "--provider", "--model", "--api-key", "--api-key-env", "--base-url", "--timeout-seconds", "--dry-run", "--invoke-model"],
            "extract-facts": ["--input", "--project", "--output-dir"],
            "review": ["--candidates", "--memory-cards", "--output-dir"],
            "apply": ["--reviewed", "--memory-cards", "--output-dir", "--reviewer"],
            "run": ["--input", "--project", "--output-dir", "--memory-cards", "--mode", "--provider", "--model", "--api-key", "--api-key-env", "--base-url", "--timeout-seconds", "--dry-run", "--invoke-model"],
            "status": ["--run-id", "--output-dir"],
            "resume": ["--run-id", "--output-dir", "--reviewed", "--memory-cards", "--reviewer"],
            "trace": ["--run-id", "--candidate-id", "--output-dir"],
            "context": ["--project", "--memory-cards", "--limit", "--format"],
            "summary": ["--scope", "--memory-cards", "--output"],
            "export": ["--target", "--scope", "--project", "--memory-cards", "--output-dir", "--limit"],
            "eval": ["--input", "--project", "--mode", "--output", "--provider", "--model", "--api-key", "--api-key-env", "--base-url", "--timeout-seconds", "--max-rows", "--max-attempts", "--continue-on-error", "--fallback-rules-on-error", "--fallback-rules-on-empty"],
        },
    }


def _model_list_payload(request: MemoryModelsRequest) -> dict[str, Any]:
    config = ProviderConfig(
        provider=request.provider,
        model=request.model or "",
        api_key=request.api_key or None,
        api_key_env=request.api_key_env,
        base_url=request.base_url,
        timeout_seconds=int(request.timeout_seconds or 60),
    )
    models = sorted({model for model in list_provider_models(config) if model})
    return {"ok": True, "provider": config.provider, "models": models}


def _write_config(memory_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_memory_config(config)
    path = _config_path(memory_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return normalized


def _model_label_from_request(request: MemoryRunStartRequest, config: dict[str, Any]) -> str:
    if request.model:
        return f"{request.provider}:{request.model}" if request.provider and ":" not in request.model else request.model
    profiles, policy = runtime_parts_from_config(config)
    profile = profiles[policy.default_profile]
    return f"{profile.config.provider}:{profile.config.model}"


def _run_namespace(request: MemoryRunStartRequest, memory_dir: Path) -> Namespace:
    return Namespace(
        input=request.input,
        project=request.project,
        output_dir=str(memory_dir),
        memory_cards=request.memory_cards,
        mode=request.mode,
        provider=request.provider,
        model=request.model,
        api_key=request.api_key,
        api_key_env=request.api_key_env,
        base_url=request.base_url,
        timeout_seconds=request.timeout_seconds,
        invoke_model=request.invoke_model,
    )


def _resume_namespace(run_id: str, reviewed: str | None, memory_cards: str | None, memory_dir: Path) -> Namespace:
    return Namespace(run_id=run_id, reviewed=reviewed, memory_cards=memory_cards, reviewer="user", output_dir=str(memory_dir))


def _auto_review_namespace(run_id: str, request: MemoryAutoReviewRequest, memory_dir: Path, *, dry_run: bool) -> Namespace:
    return Namespace(
        run_id=run_id,
        output_dir=str(memory_dir),
        reviewer=request.reviewer,
        min_score=float(request.min_score),
        review_queue=None,
        reviewed_output=None,
        keep_review=bool(request.keep_review),
        include_duplicates=bool(request.include_duplicates),
        include_merges=bool(request.include_merges),
        force=bool(request.force),
        dry_run=dry_run,
    )


def _auto_review_preview_from_queue(queue: list[dict[str, Any]], payload: dict[str, Any], request: MemoryAutoReviewRequest) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    include_duplicates = bool(request.include_duplicates)
    keep_review = bool(request.keep_review)
    for item in queue:
        candidate = item.get("candidate") if isinstance(item.get("candidate"), dict) else {}
        analysis = item.get("dream_analysis") if isinstance(item.get("dream_analysis"), dict) else {}
        quality = item.get("quality_signals") if isinstance(item.get("quality_signals"), dict) else {}
        try:
            score_value = float(analysis.get("dream_score") or 0.0)
        except (TypeError, ValueError):
            score_value = 0.0
        suggested = str(item.get("suggested_action") or analysis.get("suggested_action") or "review")
        decision = "skip"
        reason = "requires_manual_review"
        if not candidate:
            reason = "missing_candidate"
        elif quality.get("duplicate") and not include_duplicates:
            reason = "duplicate"
        elif suggested in {"create", "merge", "review"}:
            reason = "requires_manual_review"
        elif suggested == "reject":
            decision = "rejected"
            reason = "suggested_reject"
        elif suggested == "needs_more_evidence" and not keep_review:
            decision = "needs_more_evidence"
            reason = "needs_more_evidence"
        elif suggested == "needs_more_evidence":
            reason = "requires_manual_review"
        else:
            reason = "unhandled_action"
        rows.append({
            "candidate_id": item.get("candidate_id") or candidate.get("id"),
            "content": candidate.get("content"),
            "type": candidate.get("type"),
            "scope": candidate.get("scope"),
            "suggested_action": suggested,
            "decision": decision,
            "reason": reason,
            "dream_score": score_value,
            "duplicate": bool(quality.get("duplicate")),
            "evidence_quality": quality.get("evidence_quality"),
        })
    return rows


def _review_progress(state: dict[str, Any]) -> dict[str, Any]:
    artifacts = state.get("artifacts", {}) if isinstance(state.get("artifacts"), dict) else {}
    queue_path = Path(str(artifacts.get("review_queue_path") or ""))
    candidates_path = Path(str(artifacts.get("candidates_path") or ""))
    queue_items = load_events_jsonl(queue_path) if queue_path.is_file() else []
    candidates = load_events_jsonl(candidates_path) if candidates_path.is_file() else []
    source_items = queue_items or candidates
    source = "review_queue" if queue_items else "candidates"
    candidate_ids: list[str] = []
    suggested_actions: dict[str, int] = {}
    statuses: dict[str, int] = {}
    for item in source_items:
        candidate = item.get("candidate") if isinstance(item.get("candidate"), dict) else item
        candidate_id = str(item.get("candidate_id") or candidate.get("id") or "")
        if candidate_id:
            candidate_ids.append(candidate_id)
        analysis = item.get("dream_analysis") if isinstance(item.get("dream_analysis"), dict) else {}
        suggested = str(item.get("suggested_action") or analysis.get("suggested_action") or "unknown")
        suggested_actions[suggested] = suggested_actions.get(suggested, 0) + 1
        status = str(item.get("status") or candidate.get("status") or "unknown")
        statuses[status] = statuses.get(status, 0) + 1
    reviewed_path = Path(str(state["run_dir"])) / "reviewed.jsonl"
    reviewed = load_events_jsonl(reviewed_path) if reviewed_path.is_file() else []
    reviewed_ids = {str(row.get("candidate_id")) for row in reviewed if row.get("candidate_id")}
    actions: dict[str, int] = {}
    for row in reviewed:
        action = str(row.get("action") or row.get("status") or "unknown")
        actions[action] = actions.get(action, 0) + 1
    pending_ids = [candidate_id for candidate_id in candidate_ids if candidate_id not in reviewed_ids]
    return {
        "run_id": state["run_id"],
        "source": source,
        "total": len(candidate_ids),
        "reviewed": len(reviewed_ids & set(candidate_ids)),
        "pending": len(pending_ids),
        "pending_ids": pending_ids[:20],
        "actions": actions,
        "suggested_actions": dict(sorted(suggested_actions.items())),
        "statuses": dict(sorted(statuses.items())),
    }


def _review_queue_summary(items: list[dict[str, Any]]) -> dict[str, Any]:
    def bump(bucket: dict[str, int], key: object) -> None:
        name = str(key or "unknown")
        bucket[name] = bucket.get(name, 0) + 1

    by_status: dict[str, int] = {}
    by_suggested_action: dict[str, int] = {}
    by_type: dict[str, int] = {}
    by_scope: dict[str, int] = {}
    by_evidence_quality: dict[str, int] = {}
    by_value_class: dict[str, int] = {}
    duplicate_count = 0
    conflict_count = 0
    low_score_count = 0
    needs_manual_count = 0
    scores: list[float] = []
    for item in items:
        candidate = item.get("candidate") if isinstance(item.get("candidate"), dict) else {}
        analysis = item.get("dream_analysis") if isinstance(item.get("dream_analysis"), dict) else {}
        quality = item.get("quality_signals") if isinstance(item.get("quality_signals"), dict) else {}
        action = str(item.get("suggested_action") or analysis.get("suggested_action") or "unknown")
        bump(by_status, item.get("status") or candidate.get("status"))
        bump(by_suggested_action, action)
        bump(by_type, candidate.get("type"))
        bump(by_scope, candidate.get("scope"))
        value_class = str(quality.get("value_class") or ("existing_duplicate" if quality.get("duplicate") else "similar_existing" if quality.get("matched_memory_id") else "new_value"))
        bump(by_evidence_quality, quality.get("evidence_quality"))
        bump(by_value_class, value_class)
        duplicate_count += 1 if quality.get("duplicate") else 0
        conflict_count += len(item.get("conflicts") or []) if isinstance(item.get("conflicts"), list) else 0
        needs_manual_count += 1 if action in {"review", "needs_more_evidence"} else 0
        try:
            score = float(analysis.get("dream_score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        scores.append(score)
        if action in {"create", "merge"} and score < 0.7:
            low_score_count += 1
    return {
        "total": len(items),
        "by_status": dict(sorted(by_status.items())),
        "by_suggested_action": dict(sorted(by_suggested_action.items())),
        "by_type": dict(sorted(by_type.items())),
        "by_scope": dict(sorted(by_scope.items())),
        "by_evidence_quality": dict(sorted(by_evidence_quality.items())),
        "by_value_class": dict(sorted(by_value_class.items())),
        "new_value_count": by_value_class.get("new_value", 0),
        "existing_duplicate_count": by_value_class.get("existing_duplicate", 0),
        "duplicate_count": duplicate_count,
        "conflict_count": conflict_count,
        "low_score_count": low_score_count,
        "needs_manual_count": needs_manual_count,
        "score_min": round(min(scores), 4) if scores else None,
        "score_max": round(max(scores), 4) if scores else None,
        "score_avg": round(sum(scores) / len(scores), 4) if scores else None,
    }


def create_app(default_output_dir: Path | str = "outputs/runs", default_memory_dir: Path | str = ".dream-memory") -> FastAPI:
    memory_dir = Path(default_memory_dir).expanduser()
    app = FastAPI(title="Dream Memory", version="0.1.0")

    @app.get("/")
    def home() -> RedirectResponse:
        return RedirectResponse(url="/memory-review")

    @app.get("/memory-review", response_class=HTMLResponse)
    def memory_review() -> str:
        return _template("memory_review.html")

    @app.get("/memory-config", response_class=HTMLResponse)
    def memory_config_page() -> HTMLResponse:
        return HTMLResponse(
            _template("memory_config.html"),
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
            },
        )

    @app.get("/api/memory/config")
    def memory_config_read() -> dict[str, Any]:
        try:
            return _config_payload(memory_dir)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/memory/models")
    def memory_models(request: MemoryModelsRequest) -> dict[str, Any]:
        try:
            return _model_list_payload(request)
        except (ModelProviderError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.put("/api/memory/config")
    def memory_config_update(request: MemoryConfigUpdateRequest) -> dict[str, Any]:
        try:
            _write_config(memory_dir, request.config)
            return _config_payload(memory_dir)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/memory/config/reset")
    def memory_config_reset() -> dict[str, Any]:
        from .memory_config import DEFAULT_MEMORY_CONFIG

        _write_config(memory_dir, DEFAULT_MEMORY_CONFIG)
        return _config_payload(memory_dir)

    @app.post("/api/memory/runs/start")
    def memory_run_start(request: MemoryRunStartRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
        config = _web_config(memory_dir)
        args = _run_namespace(request, memory_dir)
        mode = str(request.mode or config["mode"])
        model = _model_label_from_request(request, config)
        invoke_model = bool(config["invoke_model"] if request.invoke_model is None else request.invoke_model)
        state = create_run_state(
            memory_dir=memory_dir,
            project=request.project,
            input_path=request.input,
            mode=mode,
            model=model,
            invoke_model=invoke_model,
        )
        state = update_run_state(
            state,
            status="queued",
            phase="queued",
            next_actions=["poll /api/memory/runs/{run_id}", "wait for waiting_review"],
        )
        append_trace(state, "run_queued", {"input_path": request.input, "project": request.project})

        def run_task() -> None:
            try:
                _run_dream_to_review(args=args, config=config, persistent=True, existing_state=state)
            except Exception as exc:
                try:
                    current_state = load_run_state(memory_dir, str(state["run_id"]))
                except Exception:
                    current_state = state
                failed_state = update_run_state(
                    current_state,
                    status="failed",
                    phase="failed",
                    error=str(exc),
                    next_actions=["检查 .dream-memory/config.json 后重新生成候选"],
                )
                append_trace(failed_state, "run_failed", {"error_type": exc.__class__.__name__, "error": str(exc)})

        background_tasks.add_task(run_task)
        return {
            "ok": True,
            "run_id": state["run_id"],
            "state_path": str(Path(str(state["run_dir"])) / "state.json"),
            "run_dir": state["run_dir"],
            "status": "queued",
        }

    @app.get("/api/memory/runs")
    def memory_runs() -> dict[str, Any]:
        return {"memory_dir": str(memory_dir), "runs": list_runs(memory_dir)}

    @app.get("/api/memory/runs/{run_id}")
    def memory_run_state(run_id: str) -> dict[str, Any]:
        try:
            return load_run_state(memory_dir, run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc

    @app.get("/api/memory/runs/{run_id}/trace")
    def memory_run_trace(run_id: str, candidate_id: str | None = None) -> dict[str, Any]:
        try:
            load_run_state(memory_dir, run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        return {"run_id": run_id, "candidate_id": candidate_id, "trace": read_trace(memory_dir, run_id, candidate_id=candidate_id)}

    @app.get("/api/memory/runs/{run_id}/review-queue")
    def memory_run_review_queue(run_id: str) -> dict[str, Any]:
        try:
            state = load_run_state(memory_dir, run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        queue_path = Path(str(state.get("artifacts", {}).get("review_queue_path") or ""))
        items = load_events_jsonl(queue_path) if queue_path.is_file() else []
        return {"run_id": run_id, "count": len(items), "items": items}

    @app.get("/api/memory/runs/{run_id}/review-summary")
    def memory_run_review_summary(run_id: str) -> dict[str, Any]:
        try:
            state = load_run_state(memory_dir, run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        queue_path = Path(str(state.get("artifacts", {}).get("review_queue_path") or ""))
        items = load_events_jsonl(queue_path) if queue_path.is_file() else []
        return {"run_id": run_id, "summary": _review_queue_summary(items)}

    @app.get("/api/memory/runs/{run_id}/review-progress")
    def memory_run_review_progress(run_id: str) -> dict[str, Any]:
        try:
            state = load_run_state(memory_dir, run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        return _review_progress(state)

    @app.get("/api/memory/runs/{run_id}/candidates")
    def memory_run_candidates(run_id: str) -> dict[str, Any]:
        try:
            state = load_run_state(memory_dir, run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        candidates_path = Path(str(state.get("artifacts", {}).get("candidates_path") or ""))
        candidates = load_events_jsonl(candidates_path) if candidates_path.is_file() else []
        return {"run_id": run_id, "count": len(candidates), "candidates": candidates}

    @app.post("/api/memory/runs/{run_id}/review")
    def memory_run_review_submit(run_id: str, request: MemoryReviewRequest) -> dict[str, Any]:
        allowed = {"approved", "rejected", "edited_and_approved", "merged", "needs_more_evidence"}
        if request.action not in allowed:
            raise HTTPException(status_code=400, detail="invalid review action")
        try:
            state = load_run_state(memory_dir, run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        reviewed_path = Path(str(state["run_dir"])) / "reviewed.jsonl"
        raw_payload = {
            "candidate_id": request.candidate_id,
            "action": request.action,
            "edited_content": request.edited_content,
            "reviewer": request.reviewer,
            "note": request.note,
            "candidate": request.candidate,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        }
        payload = normalize_review_decision(raw_payload)
        payload["action"] = request.action
        payload["edited_content"] = request.edited_content
        payload["candidate"] = request.candidate
        with reviewed_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        append_trace(state, "review_recorded", {"candidate_id": request.candidate_id, "action": request.action, "reviewed_path": str(reviewed_path)})
        return {"ok": True, "run_id": run_id, "reviewed_path": str(reviewed_path), "review": payload, "progress": _review_progress(state)}

    @app.post("/api/memory/runs/{run_id}/auto-review/preview")
    def memory_run_auto_review_preview(run_id: str, request: MemoryAutoReviewRequest) -> dict[str, Any]:
        config = _web_config(memory_dir)
        args = _auto_review_namespace(run_id, request, memory_dir, dry_run=True)
        try:
            state = load_run_state(memory_dir, run_id)
            payload = _auto_review_run(args=args, config=config)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        queue_path = Path(str(state.get("artifacts", {}).get("review_queue_path") or ""))
        queue = load_events_jsonl(queue_path) if queue_path.is_file() else []
        payload["preview"] = _auto_review_preview_from_queue(queue, payload, request)
        return payload

    @app.post("/api/memory/runs/{run_id}/auto-review")
    def memory_run_auto_review_apply(run_id: str, request: MemoryAutoReviewRequest) -> dict[str, Any]:
        config = _web_config(memory_dir)
        args = _auto_review_namespace(run_id, request, memory_dir, dry_run=False)
        try:
            state = load_run_state(memory_dir, run_id)
            payload = _auto_review_run(args=args, config=config)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        queue_path = Path(str(state.get("artifacts", {}).get("review_queue_path") or ""))
        queue = load_events_jsonl(queue_path) if queue_path.is_file() else []
        payload["preview"] = _auto_review_preview_from_queue(queue, payload, request)
        return payload

    @app.post("/api/memory/runs/{run_id}/resume")
    def memory_run_resume(run_id: str) -> dict[str, Any]:
        config = _web_config(memory_dir)
        args = _resume_namespace(run_id, None, None, memory_dir)
        try:
            return _resume_run(args=args, config=config)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc

    @app.get("/api/memory/candidates")
    def memory_candidates() -> dict[str, Any]:
        candidates_path = memory_dir / "ai-candidates.jsonl"
        if not candidates_path.exists():
            candidates_path = memory_dir / "candidates.jsonl"
        candidates = load_events_jsonl(candidates_path) if candidates_path.is_file() else []
        return {
            "memory_dir": str(memory_dir),
            "candidates_path": str(candidates_path),
            "count": len(candidates),
            "candidates": candidates,
        }

    @app.post("/api/memory/review")
    def memory_review_submit(request: MemoryReviewRequest) -> dict[str, Any]:
        allowed = {"approved", "rejected", "edited_and_approved", "merged", "needs_more_evidence"}
        if request.action not in allowed:
            raise HTTPException(status_code=400, detail="invalid review action")
        memory_dir.mkdir(parents=True, exist_ok=True)
        reviewed_path = memory_dir / "reviewed.jsonl"
        raw_payload = {
            "candidate_id": request.candidate_id,
            "action": request.action,
            "edited_content": request.edited_content,
            "reviewer": request.reviewer,
            "note": request.note,
            "candidate": request.candidate,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        }
        payload = normalize_review_decision(raw_payload)
        payload["action"] = request.action
        payload["edited_content"] = request.edited_content
        payload["candidate"] = request.candidate
        with reviewed_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return {"ok": True, "reviewed_path": str(reviewed_path), "review": payload}

    return app


app = create_app()
