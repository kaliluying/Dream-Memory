from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(?P<json>\{.*?\})\s*```", re.DOTALL)
SENSITIVE_RE = re.compile(
    r"(sk-[a-zA-Z0-9]|api[_-]?key\s*[=:]|access[_-]?token\s*[=:]|refresh[_-]?token\s*[=:]|password\s*[=:]|secret\s*[=:]|cookie\s*[=:]|bearer\s+[a-zA-Z0-9]|-----BEGIN [A-Z ]*PRIVATE KEY-----|AKIA[0-9A-Z]{16}|eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+)",
    re.I,
)
ALLOWED_MEMORY_TYPES = {"preference", "project_fact", "decision", "workflow", "pitfall", "requirement", "rejected_option", "product_direction", "global_fact"}
ALLOWED_SCOPES = {"user", "global", "project", "session"}
ALLOWED_DECISIONS = {"promote", "review", "reject"}


def _event_preview(events: list[dict[str, Any]], limit: int = 80) -> list[dict[str, Any]]:
    preview = []
    for index, event in enumerate(events[:limit], start=1):
        content = str(event.get("content") or "")
        preview.append({
            "event_id": f"event_{index}",
            "source": event.get("source"),
            "session_id": event.get("session_id"),
            "project": event.get("project"),
            "role": event.get("role"),
            "event_type": event.get("event_type"),
            "content": content[:1200],
        })
    return preview


def build_memory_extraction_prompt(events: list[dict[str, Any]], *, project: str | None) -> str:
    event_json = json.dumps(_event_preview(events), ensure_ascii=False, indent=2)
    return f"""You are a memory consolidation agent for a multi-agent coding assistant.

Project filter: {project or "global"}

Your job:
- Extract only stable, reusable memories from Claude Code / Codex session events.
- Prefer explicit user preferences, project direction, durable decisions, workflows, pitfalls, and rejected options.
- You must reject tool state, project index records, one-off logs, temporary command output, and low-value status metadata.
- API keys, auth tokens, cookie values, passwords, and raw credential strings must be omitted entirely, not summarized.
- Keep every candidate grounded in evidence.

Return JSON only with this schema:
{{
  "candidates": [
    {{
      "content": "short durable memory",
      "type": "preference|project_fact|decision|workflow|pitfall|requirement|rejected_option|product_direction",
      "scope": "user|global|project|session",
      "project": "project path or null",
      "confidence": 0.0,
      "decision": "promote|review|reject",
      "reason": "why this decision is correct",
      "evidence": [{{"event_id":"event_1","source":"codex","session_id":"..."}}],
      "tags": ["short-tag"]
    }}
  ]
}}

Important policy:
- If an event says only "Claude Code project state for /path", reject it or omit it.
- If an event contains explicit preferences like always answer in Chinese, promote it.
- If an event says the project should become Claude Code-like, promote it as product_direction.
- Do not include prose outside JSON.

Events:
{event_json}
"""


def extract_json_payload(text: str) -> dict[str, Any]:
    match = JSON_FENCE_RE.search(text)
    raw = match.group("json") if match else text.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {"candidates": []}
    try:
        payload = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return {"candidates": []}
    if not isinstance(payload, dict):
        return {"candidates": []}
    if not isinstance(payload.get("candidates"), list):
        payload["candidates"] = []
    return payload


def _stable_candidate_id(content: str, scope: str, project: str | None) -> str:
    raw = f"{scope}|{project or ''}|{content}".encode("utf-8")
    return "mem_" + hashlib.sha1(raw).hexdigest()[:12]


def _normalize_text(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value).strip().lower())
    text = re.sub(r"[。．.!！?？,，;；:：]+$", "", text)
    return text.strip()


def _coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = 0.5
    return max(0.0, min(1.0, round(confidence, 3)))


def _valid_evidence(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        return []
    evidence: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict) and any(item.get(key) for key in ("event_id", "source", "session_id")):
            evidence.append(dict(item))
    return evidence


def validate_agent_candidates(candidates: list[dict[str, Any]], *, project: str | None) -> list[dict[str, Any]]:
    """Validate and normalize model-proposed memory candidates.

    AI can decide memory semantics, but code owns safety, schema, evidence,
    dedupe, score/status normalization, and project scoping.
    """
    valid: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str | None, str]] = set()
    for raw in candidates:
        if not isinstance(raw, dict):
            continue
        content = str(raw.get("content") or "").strip()
        if not content or SENSITIVE_RE.search(content) or "```" in content or len(content) > 1200:
            continue
        memory_type = str(raw.get("type") or "").strip()
        if memory_type not in ALLOWED_MEMORY_TYPES:
            continue
        scope = str(raw.get("scope") or "").strip()
        if scope not in ALLOWED_SCOPES:
            continue
        decision = str(raw.get("decision") or "review").strip()
        if decision not in ALLOWED_DECISIONS:
            continue
        evidence = _valid_evidence(raw.get("evidence"))
        if not evidence:
            continue
        candidate_project = raw.get("project")
        if scope == "project":
            candidate_project = str(candidate_project or project or "") or None
            if not candidate_project:
                continue
        elif scope in {"user", "global"}:
            candidate_project = None
        confidence = _coerce_confidence(raw.get("confidence", 0.5))
        key = (scope, str(candidate_project), memory_type, _normalize_text(content))
        if key in seen:
            continue
        seen.add(key)
        normalized = dict(raw)
        normalized.update({
            "id": str(raw.get("id") or _stable_candidate_id(_normalize_text(content), scope, str(candidate_project) if candidate_project else None)),
            "content": content,
            "type": memory_type,
            "scope": scope,
            "project": candidate_project,
            "confidence": confidence,
            "score": confidence,
            "decision": decision,
            "status": "promote" if decision == "promote" else "reject" if decision == "reject" else "review",
            "evidence": evidence,
            "tags": [str(tag) for tag in raw.get("tags", [])] if isinstance(raw.get("tags"), list) else [],
        })
        valid.append(normalized)
    return valid


def _message_text(agent_result: Any) -> str:
    if isinstance(agent_result, dict) and "messages" in agent_result:
        messages = agent_result["messages"]
        if messages:
            last = messages[-1]
            content = getattr(last, "content", None)
            if content is None and isinstance(last, dict):
                content = last.get("content")
            return str(content or "")
    return str(agent_result)


def agent_extract_memory_candidates(
    events: list[dict[str, Any]],
    *,
    project: str | None,
    model: Any = "anthropic:claude-sonnet-4-6",
    invoke_model: bool = False,
) -> dict[str, Any]:
    prompt = build_memory_extraction_prompt(events, project=project)
    if not invoke_model:
        return {
            "runtime": "deepagents-memory-agent",
            "dry_run": True,
            "model": str(model),
            "prompt": prompt,
            "candidates": [],
        }
    if isinstance(model, str):
        try:
            from deepagents import create_deep_agent
        except ImportError as exc:
            raise RuntimeError("deepagents is required for --invoke-model with a string model") from exc
        agent = create_deep_agent(model=model, tools=[], prompt=prompt)
        result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        text = _message_text(result)
    else:
        result = model.invoke(prompt)
        text = _message_text({"messages": [result]})
    payload = extract_json_payload(text)
    candidates = validate_agent_candidates(list(payload.get("candidates", [])), project=project)
    return {
        "runtime": "deepagents-memory-agent",
        "dry_run": False,
        "model": str(model),
        "prompt": prompt,
        "raw_response": text,
        "candidates": candidates,
    }
