from __future__ import annotations

import json
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

from .memory_cli import _auto_review_run, _resume_run, _run_dream_to_review, _safe_error_text
from .memory_config import load_memory_config, normalize_memory_config
from .memory_dreaming import _evidence_refs_from_candidate, contains_sensitive_memory_text, load_events_jsonl, normalize_review_decision
from .memory_importers import redact_sensitive, redact_sensitive_text
from .memory_runs import append_trace, create_run_state, list_runs, load_run_state, read_trace, run_artifact_path, safe_candidate_id, safe_run_id, update_run_state
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


def _redact_config_secrets(config: dict[str, Any]) -> dict[str, Any]:
    redacted = json.loads(json.dumps(config))
    models = redacted.get("models")
    if not isinstance(models, dict):
        return redacted
    for profile in models.values():
        if not isinstance(profile, dict):
            continue
        api_key = profile.get("api_key")
        profile["api_key_configured"] = bool(api_key)
        if api_key:
            profile["api_key"] = ""
    return redacted


def _redact_config_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {redact_sensitive_text(str(key)): _redact_config_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_config_values(item) for item in value]
    if isinstance(value, str):
        return redact_sensitive_text(value)
    return value


def _profile_name_aliases(profile_names: Any) -> dict[str, str]:
    if not isinstance(profile_names, dict):
        return {}
    aliases: dict[str, str] = {}
    used: set[str] = set()
    for name in profile_names:
        original = str(name)
        base = redact_sensitive_text(original)
        alias = base
        if alias in used:
            index = 2
            while f"{base}-{index}" in used:
                index += 1
            alias = f"{base}-{index}"
        aliases[original] = alias
        used.add(alias)
    return aliases


def _redact_config_profile_names(config: dict[str, Any]) -> dict[str, Any]:
    models = config.get("models")
    aliases = _profile_name_aliases(models)
    if not aliases:
        return config
    redacted = json.loads(json.dumps(config))
    redacted_models = redacted.get("models")
    if isinstance(redacted_models, dict):
        redacted["models"] = {
            aliases.get(str(name), str(name)): profile
            for name, profile in redacted_models.items()
        }
    policy = redacted.get("model_policy")
    if isinstance(policy, dict):
        default_profile = policy.get("default_profile")
        if default_profile is not None:
            policy["default_profile"] = aliases.get(str(default_profile), str(default_profile))
        fallback_chain = policy.get("fallback_chain")
        if isinstance(fallback_chain, list):
            policy["fallback_chain"] = [aliases.get(str(name), str(name)) for name in fallback_chain]
    return redacted


def _restore_redacted_config_profile_names(config: dict[str, Any], existing: dict[str, Any]) -> dict[str, Any]:
    incoming_models = config.get("models")
    existing_models = existing.get("models")
    if not isinstance(incoming_models, dict) or not isinstance(existing_models, dict):
        return config
    profile_name_map = {alias: original for original, alias in _profile_name_aliases(existing_models).items()}
    if not any(redacted != original and redacted in incoming_models for redacted, original in profile_name_map.items()):
        return config

    restored_models: dict[str, Any] = {}
    for name, profile in incoming_models.items():
        restored_models[profile_name_map.get(str(name), str(name))] = profile
    config["models"] = restored_models

    policy = config.get("model_policy")
    if isinstance(policy, dict):
        default_profile = policy.get("default_profile")
        if default_profile is not None:
            policy["default_profile"] = profile_name_map.get(str(default_profile), str(default_profile))
        fallback_chain = policy.get("fallback_chain")
        if isinstance(fallback_chain, list):
            policy["fallback_chain"] = [profile_name_map.get(str(name), str(name)) for name in fallback_chain]
    return config


def _restore_redacted_config_values(value: Any, existing: Any) -> Any:
    if isinstance(value, dict) and isinstance(existing, dict):
        return {
            key: _restore_redacted_config_values(item, existing.get(key))
            for key, item in value.items()
        }
    if isinstance(value, list) and isinstance(existing, list):
        return [
            _restore_redacted_config_values(item, existing[index]) if index < len(existing) else item
            for index, item in enumerate(value)
        ]
    if isinstance(value, str) and isinstance(existing, str):
        redacted_existing = redact_sensitive_text(existing)
        if redacted_existing != existing and value == redacted_existing:
            return existing
    return value


def _preserve_existing_config_secrets(memory_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    existing_path = _config_path(memory_dir)
    if not existing_path.exists():
        return config
    try:
        existing = load_memory_config(existing_path)
    except ValueError:
        return config
    config = _restore_redacted_config_profile_names(config, existing)
    config = _restore_redacted_config_values(config, existing)
    incoming_models = config.get("models")
    existing_models = existing.get("models")
    if not isinstance(incoming_models, dict) or not isinstance(existing_models, dict):
        return config
    for name, profile in incoming_models.items():
        if not isinstance(profile, dict):
            continue
        existing_profile = existing_models.get(str(name))
        if not isinstance(existing_profile, dict):
            continue
        if profile.get("api_key") in (None, "") and existing_profile.get("api_key"):
            profile["api_key"] = existing_profile.get("api_key")
        profile.pop("api_key_configured", None)
    return config


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
    safe_config = _redact_config_values(_redact_config_profile_names(_redact_config_secrets(config)))
    return {
        "ok": True,
        "config_path": redact_sensitive_text(str(path)),
        "config": safe_config,
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
    return _redact_api_payload({"ok": True, "provider": config.provider, "models": models})


def _write_config(memory_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    path = _config_path(memory_dir)
    if path.exists() and not path.is_file():
        raise FileExistsError(f"config path is not writable: {path}")
    config = _preserve_existing_config_secrets(memory_dir, config)
    normalized = normalize_memory_config(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        tmp.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise FileExistsError(f"config path is not writable: {path}") from exc
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
    min_score = float(payload.get("min_score") or 0.0)
    include_duplicates = bool(request.include_duplicates)
    include_merges = bool(request.include_merges)
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
        has_evidence = bool(_evidence_refs_from_candidate(candidate))
        decision = "skip"
        reason = "requires_manual_review"
        if not candidate:
            reason = "missing_candidate"
        elif contains_sensitive_memory_text(candidate):
            reason = "sensitive_candidate"
        elif contains_sensitive_memory_text(item):
            reason = "sensitive_queue_metadata"
        elif quality.get("duplicate") and not include_duplicates:
            reason = "duplicate"
        elif suggested in {"create", "merge"} and score_value < min_score:
            reason = "below_min_score"
        elif suggested in {"create", "merge", "review"} and score_value >= min_score and not has_evidence:
            decision = "skip" if keep_review else "needs_more_evidence"
            reason = "missing_evidence"
        elif suggested == "create":
            decision = "approved"
            reason = "meets_min_score"
        elif suggested == "merge":
            if include_merges:
                decision = "merged"
                reason = "include_merges"
            else:
                reason = "merge_requires_explicit_include"
        elif suggested == "reject":
            decision = "rejected"
            reason = "suggested_reject"
        elif suggested == "needs_more_evidence" and not keep_review:
            decision = "needs_more_evidence"
            reason = "needs_more_evidence"
        elif suggested in {"review", "needs_more_evidence"}:
            reason = "requires_manual_review"
        else:
            reason = "unhandled_action"
        rows.append(redact_sensitive({
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
        }))
    return rows


def _read_run_artifact_jsonl(state: dict[str, Any], artifact_key: str) -> list[dict[str, Any]]:
    try:
        path = run_artifact_path(state, artifact_key, strict=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=_safe_http_detail(exc)) from exc
    if not path or not path.is_file():
        return []
    labels = {
        "review_queue_path": "review queue",
        "candidates_path": "candidates",
        "reviewed_path": "reviewed decisions",
    }
    return _read_jsonl_dicts(path, strict=True, label=labels.get(artifact_key, artifact_key))


def _load_run_state_for_api(memory_dir: Path, run_id: str) -> dict[str, Any]:
    try:
        safe_run_id(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid run_id") from exc
    try:
        return load_run_state(memory_dir, run_id)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"run state invalid: {_safe_http_detail(exc)}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"run state invalid: {_safe_http_detail(exc)}") from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="run not found") from exc


def _redact_api_payload(payload: Any) -> Any:
    return redact_sensitive(payload)


def _stable_redacted_labels(names: list[str]) -> dict[str, str]:
    labels: dict[str, str] = {}
    used: set[str] = set()
    for name in names:
        base = redact_sensitive_text(str(name))
        label = base
        if label in used:
            index = 2
            while f"{base}-{index}" in used:
                index += 1
            label = f"{base}-{index}"
        labels[str(name)] = label
        used.add(label)
    return labels


def _redact_count_bucket(bucket: dict[str, int]) -> dict[str, int]:
    labels = _stable_redacted_labels(list(bucket))
    return {
        labels.get(str(name), str(name)): count
        for name, count in sorted(bucket.items(), key=lambda item: labels.get(str(item[0]), str(item[0])))
    }


def _safe_http_detail(value: object) -> str:
    return str(redact_sensitive(str(value)))


def _review_progress(state: dict[str, Any]) -> dict[str, Any]:
    queue_items = _read_run_artifact_jsonl(state, "review_queue_path")
    candidates = _read_run_artifact_jsonl(state, "candidates_path")
    source_items = queue_items or candidates
    source = "review_queue" if queue_items else "candidates"
    candidate_ids: list[str] = []
    seen_candidate_ids: set[str] = set()
    suggested_actions: dict[str, int] = {}
    statuses: dict[str, int] = {}
    for item in source_items:
        candidate = item.get("candidate") if isinstance(item.get("candidate"), dict) else item
        candidate_id = str(item.get("candidate_id") or candidate.get("id") or "")
        if not candidate_id or candidate_id in seen_candidate_ids:
            continue
        seen_candidate_ids.add(candidate_id)
        candidate_ids.append(candidate_id)
        analysis = item.get("dream_analysis") if isinstance(item.get("dream_analysis"), dict) else {}
        suggested = str(item.get("suggested_action") or analysis.get("suggested_action") or "unknown")
        suggested_actions[suggested] = suggested_actions.get(suggested, 0) + 1
        status = str(item.get("status") or candidate.get("status") or "unknown")
        statuses[status] = statuses.get(status, 0) + 1
    reviewed_path = Path(str(state["run_dir"])) / "reviewed.jsonl"
    candidate_id_set = set(candidate_ids)
    reviewed_by_id: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl_dicts(reviewed_path, strict=True, label="reviewed decisions"):
        candidate_id = str(row.get("candidate_id") or "")
        if candidate_id in candidate_id_set:
            reviewed_by_id[candidate_id] = row
    reviewed_ids = set(reviewed_by_id)
    actions: dict[str, int] = {}
    for row in reviewed_by_id.values():
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
        "actions": _redact_count_bucket(actions),
        "suggested_actions": _redact_count_bucket(suggested_actions),
        "statuses": _redact_count_bucket(statuses),
    }


def _run_review_candidate_ids(state: dict[str, Any]) -> set[str] | None:
    queue_items = _read_run_artifact_jsonl(state, "review_queue_path")
    candidates = _read_run_artifact_jsonl(state, "candidates_path")
    source_items = queue_items or candidates
    if not source_items:
        return None
    ids: set[str] = set()
    for item in source_items:
        candidate = item.get("candidate") if isinstance(item.get("candidate"), dict) else item
        candidate_id = str(item.get("candidate_id") or candidate.get("id") or "")
        if candidate_id:
            ids.add(candidate_id)
    return ids


def _validate_run_review_candidate(state: dict[str, Any], candidate_id: str) -> None:
    candidate_ids = _run_review_candidate_ids(state)
    if candidate_ids is not None and candidate_id not in candidate_ids:
        raise HTTPException(status_code=400, detail="candidate not found in run review queue")


def _validate_review_request(request: MemoryReviewRequest) -> None:
    if isinstance(request.candidate, dict):
        payload_candidate_id = str(request.candidate.get("id") or "").strip()
        if payload_candidate_id and payload_candidate_id != request.candidate_id:
            raise HTTPException(status_code=400, detail="candidate payload id does not match review candidate")
    if contains_sensitive_memory_text(request.edited_content, request.note, request.candidate):
        raise HTTPException(status_code=400, detail="sensitive review content rejected")
    if request.action not in {"approved", "edited_and_approved", "merged"}:
        return
    candidate_content = str(request.candidate.get("content") or "").strip() if isinstance(request.candidate, dict) else ""
    edited_content = str(request.edited_content or "").strip()
    if not edited_content and not candidate_content:
        raise HTTPException(status_code=400, detail="approved review requires memory content")
    evidence = request.candidate.get("evidence") if isinstance(request.candidate, dict) else None
    evidence_refs = request.candidate.get("evidence_refs") if isinstance(request.candidate, dict) else None
    has_evidence = (
        isinstance(evidence, list)
        and any(item for item in evidence)
    ) or (
        isinstance(evidence_refs, list)
        and any(item for item in evidence_refs)
    )
    if not has_evidence:
        raise HTTPException(status_code=400, detail="approved review requires evidence")


def _read_jsonl_dicts(path: Path, *, strict: bool = False, label: str = "JSONL") -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                if strict:
                    raise HTTPException(status_code=400, detail=_safe_http_detail(f"invalid {label} JSON at {path}:{line_number}")) from exc
                continue
            if isinstance(payload, dict):
                rows.append(payload)
            elif strict:
                raise HTTPException(status_code=400, detail=_safe_http_detail(f"invalid {label} row at {path}:{line_number}: expected object"))
    return rows


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
        "by_status": _redact_count_bucket(by_status),
        "by_suggested_action": _redact_count_bucket(by_suggested_action),
        "by_type": _redact_count_bucket(by_type),
        "by_scope": _redact_count_bucket(by_scope),
        "by_evidence_quality": _redact_count_bucket(by_evidence_quality),
        "by_value_class": _redact_count_bucket(by_value_class),
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
            raise HTTPException(status_code=400, detail=_safe_http_detail(exc)) from exc

    @app.post("/api/memory/models")
    def memory_models(request: MemoryModelsRequest) -> dict[str, Any]:
        try:
            return _model_list_payload(request)
        except (ModelProviderError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=_safe_http_detail(exc)) from exc

    @app.put("/api/memory/config")
    def memory_config_update(request: MemoryConfigUpdateRequest) -> dict[str, Any]:
        try:
            _write_config(memory_dir, request.config)
            return _config_payload(memory_dir)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=_safe_http_detail(exc)) from exc
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=_safe_http_detail(exc)) from exc

    @app.post("/api/memory/config/reset")
    def memory_config_reset() -> dict[str, Any]:
        from .memory_config import DEFAULT_MEMORY_CONFIG

        try:
            _write_config(memory_dir, DEFAULT_MEMORY_CONFIG)
            return _config_payload(memory_dir)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=_safe_http_detail(exc)) from exc
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=_safe_http_detail(exc)) from exc

    @app.post("/api/memory/runs/start")
    def memory_run_start(request: MemoryRunStartRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
        config = _web_config(memory_dir)
        args = _run_namespace(request, memory_dir)
        input_path = Path(request.input).expanduser()
        if not input_path.exists():
            raise HTTPException(status_code=400, detail=_safe_http_detail(f"run input not found: {input_path}"))
        try:
            events = load_events_jsonl(input_path, strict=True, label="run input")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=_safe_http_detail(exc)) from exc
        if not events:
            raise HTTPException(status_code=400, detail=_safe_http_detail(f"run input has no valid events: {input_path}"))
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
                safe_error = _safe_error_text(exc)
                failed_state = update_run_state(
                    current_state,
                    status="failed",
                    phase="failed",
                    error=safe_error,
                    next_actions=["检查 .dream-memory/config.json 后重新生成候选"],
                )
                append_trace(failed_state, "run_failed", {"error_type": exc.__class__.__name__, "error": safe_error})

        background_tasks.add_task(run_task)
        return _redact_api_payload({
            "ok": True,
            "run_id": state["run_id"],
            "state_path": str(Path(str(state["run_dir"])) / "state.json"),
            "run_dir": state["run_dir"],
            "status": "queued",
        })

    @app.get("/api/memory/runs")
    def memory_runs() -> dict[str, Any]:
        return _redact_api_payload({"memory_dir": str(memory_dir), "runs": list_runs(memory_dir)})

    @app.get("/api/memory/runs/{run_id}")
    def memory_run_state(run_id: str) -> dict[str, Any]:
        return _redact_api_payload(_load_run_state_for_api(memory_dir, run_id))

    @app.get("/api/memory/runs/{run_id}/trace")
    def memory_run_trace(run_id: str, candidate_id: str | None = None) -> dict[str, Any]:
        _load_run_state_for_api(memory_dir, run_id)
        if candidate_id is not None:
            try:
                safe_candidate_id(candidate_id)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="invalid candidate_id") from exc
        try:
            trace = read_trace(memory_dir, run_id, candidate_id=candidate_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=_safe_http_detail(exc)) from exc
        return {"run_id": run_id, "candidate_id": candidate_id, "trace": _redact_api_payload(trace)}

    @app.get("/api/memory/runs/{run_id}/review-queue")
    def memory_run_review_queue(run_id: str) -> dict[str, Any]:
        state = _load_run_state_for_api(memory_dir, run_id)
        items = _read_run_artifact_jsonl(state, "review_queue_path")
        return {"run_id": run_id, "count": len(items), "items": _redact_api_payload(items)}

    @app.get("/api/memory/runs/{run_id}/review-summary")
    def memory_run_review_summary(run_id: str) -> dict[str, Any]:
        state = _load_run_state_for_api(memory_dir, run_id)
        items = _read_run_artifact_jsonl(state, "review_queue_path")
        return {"run_id": run_id, "summary": _redact_api_payload(_review_queue_summary(items))}

    @app.get("/api/memory/runs/{run_id}/review-progress")
    def memory_run_review_progress(run_id: str) -> dict[str, Any]:
        state = _load_run_state_for_api(memory_dir, run_id)
        return _redact_api_payload(_review_progress(state))

    @app.get("/api/memory/runs/{run_id}/candidates")
    def memory_run_candidates(run_id: str) -> dict[str, Any]:
        state = _load_run_state_for_api(memory_dir, run_id)
        candidates = _read_run_artifact_jsonl(state, "candidates_path")
        return {"run_id": run_id, "count": len(candidates), "candidates": _redact_api_payload(candidates)}

    @app.post("/api/memory/runs/{run_id}/review")
    def memory_run_review_submit(run_id: str, request: MemoryReviewRequest) -> dict[str, Any]:
        allowed = {"approved", "rejected", "edited_and_approved", "merged", "needs_more_evidence"}
        if request.action not in allowed:
            raise HTTPException(status_code=400, detail="invalid review action")
        _validate_review_request(request)
        state = _load_run_state_for_api(memory_dir, run_id)
        _validate_run_review_candidate(state, request.candidate_id)
        reviewed_path = Path(str(state["run_dir"])) / "reviewed.jsonl"
        _read_jsonl_dicts(reviewed_path, strict=True, label="reviewed decisions")
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
        return _redact_api_payload({"ok": True, "run_id": run_id, "reviewed_path": str(reviewed_path), "review": payload, "progress": _review_progress(state)})

    @app.post("/api/memory/runs/{run_id}/auto-review/preview")
    def memory_run_auto_review_preview(run_id: str, request: MemoryAutoReviewRequest) -> dict[str, Any]:
        config = _web_config(memory_dir)
        args = _auto_review_namespace(run_id, request, memory_dir, dry_run=True)
        state = _load_run_state_for_api(memory_dir, run_id)
        try:
            payload = _auto_review_run(args=args, config=config)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=_safe_http_detail(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=_safe_http_detail(exc)) from exc
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=_safe_http_detail(exc)) from exc
        queue = _read_run_artifact_jsonl(state, "review_queue_path")
        payload["preview"] = _auto_review_preview_from_queue(queue, payload, request)
        return payload

    @app.post("/api/memory/runs/{run_id}/auto-review")
    def memory_run_auto_review_apply(run_id: str, request: MemoryAutoReviewRequest) -> dict[str, Any]:
        config = _web_config(memory_dir)
        args = _auto_review_namespace(run_id, request, memory_dir, dry_run=False)
        state = _load_run_state_for_api(memory_dir, run_id)
        try:
            payload = _auto_review_run(args=args, config=config)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=_safe_http_detail(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=_safe_http_detail(exc)) from exc
        except FileExistsError as exc:
            raise HTTPException(status_code=409, detail=_safe_http_detail(exc)) from exc
        queue = _read_run_artifact_jsonl(state, "review_queue_path")
        payload["preview"] = _auto_review_preview_from_queue(queue, payload, request)
        return payload

    @app.post("/api/memory/runs/{run_id}/resume")
    def memory_run_resume(run_id: str) -> dict[str, Any]:
        config = _web_config(memory_dir)
        args = _resume_namespace(run_id, None, None, memory_dir)
        _load_run_state_for_api(memory_dir, run_id)
        try:
            return _resume_run(args=args, config=config)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=_safe_http_detail(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=_safe_http_detail(exc)) from exc

    @app.get("/api/memory/candidates")
    def memory_candidates() -> dict[str, Any]:
        candidates_path = memory_dir / "ai-candidates.jsonl"
        if not candidates_path.exists():
            candidates_path = memory_dir / "candidates.jsonl"
        candidates = _read_jsonl_dicts(candidates_path, strict=True, label="candidates")
        return _redact_api_payload({
            "memory_dir": str(memory_dir),
            "candidates_path": str(candidates_path),
            "count": len(candidates),
            "candidates": candidates,
        })

    @app.post("/api/memory/review")
    def memory_review_submit(request: MemoryReviewRequest) -> dict[str, Any]:
        allowed = {"approved", "rejected", "edited_and_approved", "merged", "needs_more_evidence"}
        if request.action not in allowed:
            raise HTTPException(status_code=400, detail="invalid review action")
        _validate_review_request(request)
        memory_dir.mkdir(parents=True, exist_ok=True)
        reviewed_path = memory_dir / "reviewed.jsonl"
        _read_jsonl_dicts(reviewed_path, strict=True, label="reviewed decisions")
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
        return _redact_api_payload({"ok": True, "reviewed_path": str(reviewed_path), "review": payload})

    return app


app = create_app()
