from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .memory_dreaming import (
    SENSITIVE_RE,
    _candidate_id as _stable_candidate_id,
    build_candidates_from_facts,
    normalize_memory_text as _normalize_text,
    normalize_project_path,
    _is_code_or_listing_dump_content,
    _is_low_value_event_content,
    _is_long_generic_memory_content,
    _is_structural_or_one_off_artifact,
)
from .model_providers import invoke_model as invoke_model_provider, invoke_model_runtime

JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(?P<json>\{.*?\})\s*```", re.DOTALL)
ALLOWED_MEMORY_TYPES = {"preference", "project_fact", "decision", "workflow", "pitfall", "requirement", "rejected_option", "product_direction", "global_fact"}
ALLOWED_SCOPES = {"user", "global", "project", "session"}
ALLOWED_DECISIONS = {"promote", "review", "reject"}
ONE_OFF_TASK_RE = re.compile(
    r"(删除|修改|改为|实现|接入|测试|跑|生成|调用|修复|新增|更新|清理|迁移|检查).{0,18}(组件|页面|功能|接口|脚本|数据|水印|首页|配置中心|测试|任务)",
    re.I,
)
SHALLOW_PROJECT_TASK_RE = re.compile(
    r"(首页|页面|组件|配置中心|侧边栏|前端|后端|接口|菜单|水印|测试).{0,18}(需要|需|要|使用|改为|改成|全部|重点|关注|删除|修复|更新|真实数据|中文)",
    re.I,
)
CREDENTIAL_LOCATION_RE = re.compile(r"(密钥|key|token|api[_-]?key).{0,12}(在|文件|path|路径).{0,24}(\.txt|\.env|json|yaml|yml|配置)", re.I)


def _is_prompt_noise_event(event: dict[str, Any]) -> bool:
    event_type = str(event.get("event_type") or "")
    content = str(event.get("content") or "")
    if not content.strip():
        return True
    if event_type in {"project_state", "project_settings", "tool_output", "build_log"}:
        return True
    if event_type == "project_markers":
        return False
    if _is_low_value_event_content(content):
        return True
    if _is_code_or_listing_dump_content(content):
        return True
    if _is_long_generic_memory_content(content, event_type):
        return True
    if _is_structural_or_one_off_artifact(content, event_type):
        return True
    return False


def _event_preview(events: list[dict[str, Any]], limit: int = 80) -> list[dict[str, Any]]:
    preview = []
    for index, event in enumerate(events, start=1):
        if len(preview) >= limit:
            break
        content = str(event.get("content") or "")
        if _is_prompt_noise_event(event):
            continue
        preview.append({
            "event_id": str(event.get("event_id") or event.get("id") or f"event_{index}"),
            "source": event.get("source"),
            "session_id": event.get("session_id"),
            "project": event.get("project"),
            "role": event.get("role"),
            "event_type": event.get("event_type"),
            "content": content[:1200],
        })
    return preview


def _prompt_event_counts(events: list[dict[str, Any]], limit: int = 80) -> dict[str, int]:
    input_event_count = len(events)
    prompt_event_count = min(limit, sum(1 for event in events if not _is_prompt_noise_event(dict(event))))
    return {
        "input_event_count": input_event_count,
        "prompt_event_count": prompt_event_count,
        "filtered_prompt_event_count": input_event_count - prompt_event_count,
    }


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
- Write all human-readable output fields in Simplified Chinese, especially content, reason, and tags.
- Keep product names, model names, file paths, CLI commands, code identifiers, and API names unchanged.
- Project filter is strict for project-scoped facts: if a project filter is provided, only emit project-scoped candidates for that exact project. User-scope preferences and workflows may still be emitted even when their source event came from another project. If the filter is global, omit project-scoped tasks from other projects.
- Do not promote one-off implementation tasks, old TODOs, bug reports, transient commands, endpoint failures, "delete/modify this UI", or single-run scripts unless they reveal a durable reusable rule.
- A valuable memory must change future assistant behavior: user preference, durable architecture decision, reusable workflow/pitfall, rejected option, or long-lived product direction.
- Never include credential locations such as "key is in key.txt"; treat them as sensitive operational details. If the supporting evidence quote contains a credential location, omit the candidate entirely instead of turning it into a safety recommendation.

Return JSON only with this schema. Prefer `atomic_facts`; the application will aggregate those facts into review candidates:
{{
  "atomic_facts": [
    {{
      "statement": "one atomic durable fact",
      "fact_type": "preference|project_fact|decision|workflow|pitfall|requirement|rejected_option|product_direction",
      "scope": "user|global|project|session",
      "project": "project path or null",
      "confidence": 0.0,
      "evidence": [{{"event_id":"event_1","source":"codex","session_id":"...","quote":"short supporting quote"}}],
      "long_term": true,
      "long_term_reason": "why this should matter beyond the current task",
      "reuse_scenarios": ["when this fact should be retrieved"],
      "tags": ["short-tag"]
    }}
  ],
  "candidates": [
    {{
      "content": "legacy aggregated memory candidate, only if atomic_facts cannot express it",
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
- If an event is only project-state metadata for a path, reject it or omit it.
- If an event contains explicit preferences like always answer in Chinese, promote it.
- If an event is `project_instruction` and states package managers such as "Python uses uv" / "frontend uses pnpm", extract it as a project workflow.
- If an event is `project_markers` and contains markers like `python_package_manager=uv`, `python_test_runner=pytest`, or `python_framework=fastapi`, extract durable project workflow/fact memories.
- Canonicalize common durable memories when possible: use "用户偏好中文回答。" for Chinese-language preference; use "Python 后端使用 uv 进行包管理，前端使用 pnpm 进行管理。" for uv/pnpm project instructions.
- If an event says the project should become Claude Code-like, promote it as product_direction.
- If an event is merely "delete watermark", "change page text", "use real data", "run two tests", or similar implementation work, reject it or omit it.
- If the original event is in another language, summarize the durable memory in Simplified Chinese.
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
    if not isinstance(payload.get("atomic_facts"), list):
        payload["atomic_facts"] = []
    if not isinstance(payload.get("candidates"), list):
        payload["candidates"] = []
    return payload


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


def _evidence_has_sensitive_operational_detail(evidence: list[dict[str, Any]]) -> bool:
    for item in evidence:
        text = " ".join(str(item.get(key) or "") for key in ("quote", "content", "text", "summary"))
        if text and (SENSITIVE_RE.search(text) or CREDENTIAL_LOCATION_RE.search(text)):
            return True
    return False


def _evidence_refs(evidence: list[dict[str, Any]]) -> list[str]:
    refs: list[str] = []
    for index, item in enumerate(evidence, start=1):
        ref = item.get("event_id") or item.get("id")
        if ref:
            refs.append(str(ref))
            continue
        source = item.get("source") or "evidence"
        session_id = item.get("session_id") or index
        refs.append(f"{source}:{session_id}")
    return refs


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _is_low_value_candidate(content: str, memory_type: str) -> bool:
    normalized = _normalize_text(content)
    lowered = content.lower()
    durable_tooling_workflow = memory_type == "workflow" and (
        "uv" in lowered
        or "pnpm" in lowered
        or "python_package_manager" in lowered
        or "frontend_package_manager" in lowered
        or "包管理" in content
    )
    if CREDENTIAL_LOCATION_RE.search(content):
        return True
    if memory_type == "requirement" and ONE_OFF_TASK_RE.search(content):
        return True
    if memory_type in {"requirement", "workflow"} and SHALLOW_PROJECT_TASK_RE.search(content) and not durable_tooling_workflow:
        return True
    if "全流程测试重点关注" in content:
        return True
    if "脚本需求" in content and any(token in content for token in ("三并发", "两轮", "测试", "密钥")):
        return True
    if len(normalized) < 12 and memory_type not in {"preference", "rejected_option"}:
        return True
    return False


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
        if _is_low_value_candidate(content, memory_type):
            continue
        scope = str(raw.get("scope") or "").strip()
        if scope not in ALLOWED_SCOPES:
            continue
        decision = str(raw.get("decision") or "review").strip()
        if decision not in ALLOWED_DECISIONS:
            continue
        evidence = _valid_evidence(raw.get("evidence"))
        if not evidence or _evidence_has_sensitive_operational_detail(evidence):
            continue
        candidate_project = raw.get("project")
        if scope == "project":
            if not project:
                continue
            candidate_project = normalize_project_path(str(candidate_project or project or ""))
            if not candidate_project:
                continue
            expected_project = normalize_project_path(project)
            if expected_project and candidate_project != expected_project:
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


def validate_agent_atomic_facts(facts: list[dict[str, Any]], *, project: str | None) -> list[dict[str, Any]]:
    """Validate model-proposed atomic facts before candidate aggregation."""
    valid: list[dict[str, Any]] = []
    for raw in facts:
        if not isinstance(raw, dict):
            continue
        statement = str(raw.get("statement") or raw.get("content") or "").strip()
        if not statement or SENSITIVE_RE.search(statement) or "```" in statement or len(statement) > 1200:
            continue
        fact_type = str(raw.get("fact_type") or raw.get("type") or "").strip()
        if fact_type not in ALLOWED_MEMORY_TYPES:
            continue
        if _is_low_value_candidate(statement, fact_type):
            continue
        scope = str(raw.get("scope") or "").strip()
        if scope not in ALLOWED_SCOPES:
            continue
        evidence = _valid_evidence(raw.get("evidence"))
        if not evidence or _evidence_has_sensitive_operational_detail(evidence):
            continue
        fact_project = raw.get("project")
        if scope == "project":
            if not project:
                continue
            fact_project = normalize_project_path(str(fact_project or project or ""))
            expected_project = normalize_project_path(project)
            if not fact_project or (expected_project and fact_project != expected_project):
                continue
        elif scope in {"user", "global"}:
            fact_project = None

        long_term = bool(raw.get("long_term", raw.get("is_long_term", False)))
        reuse_scenarios = _string_list(raw.get("reuse_scenarios") or raw.get("retrieval_hints"))
        long_term_reason = str(raw.get("long_term_reason") or raw.get("reason") or "").strip()
        confidence = _coerce_confidence(raw.get("confidence", 0.5))
        tags = _string_list(raw.get("tags"))
        canonical_project = fact_project
        if project and (
            ("正式记忆" in statement and "人工审核" in statement)
            or ("未经审核" in statement and "自动写入" in statement)
        ):
            canonical_project = normalize_project_path(project)

        canonical_facts = _canonical_agent_facts_from_statement(
            statement=statement,
            fact_type=fact_type,
            scope=scope,
            project=canonical_project,
            evidence=evidence,
            confidence=confidence,
            long_term=long_term,
            long_term_reason=long_term_reason,
            reuse_scenarios=reuse_scenarios,
            tags=tags,
        )
        if canonical_facts:
            valid.extend(canonical_facts)
            continue
        valid.append({
            "id": str(raw.get("id") or _stable_candidate_id(_normalize_text(statement), scope, str(fact_project) if fact_project else None)),
            "fact_type": fact_type,
            "statement": statement,
            "scope": scope,
            "project": fact_project,
            "source": evidence[0].get("source"),
            "session_id": evidence[0].get("session_id"),
            "evidence": evidence,
            "evidence_refs": _evidence_refs(evidence),
            "confidence": confidence,
            "long_term": long_term,
            "long_term_reason": long_term_reason,
            "reuse_scenarios": reuse_scenarios,
            "tags": tags,
            "status": "active",
        })
    return valid



def _canonical_agent_facts_from_statement(
    *,
    statement: str,
    fact_type: str,
    scope: str,
    project: str | None,
    evidence: list[dict[str, Any]],
    confidence: float,
    long_term: bool,
    long_term_reason: str,
    reuse_scenarios: list[str],
    tags: list[str],
) -> list[dict[str, Any]]:
    lowered = statement.lower()
    canonical: list[dict[str, Any]] = []

    def build(statement_value: str, fact_type_value: str, extra_tags: list[str]) -> dict[str, Any]:
        return {
            "id": _stable_candidate_id(_normalize_text(statement_value), scope, str(project) if project else None),
            "fact_type": fact_type_value,
            "statement": statement_value,
            "scope": scope,
            "project": project,
            "source": evidence[0].get("source"),
            "session_id": evidence[0].get("session_id"),
            "evidence": evidence,
            "evidence_refs": _evidence_refs(evidence),
            "confidence": confidence,
            "long_term": long_term,
            "long_term_reason": long_term_reason,
            "reuse_scenarios": reuse_scenarios,
            "tags": sorted(set(tags + extra_tags)),
            "status": "active",
        }

    if fact_type == "workflow" and ("uv" in lowered or "pnpm" in lowered or "包管理" in statement):
        if "uv" in lowered and ("pnpm" in lowered or "前端" in statement):
            canonical.append(build("Python 后端使用 uv 进行包管理，前端使用 pnpm 进行管理。", "workflow", ["package-manager", "uv", "pnpm", "python", "frontend"]))
        elif "uv" in lowered:
            canonical.append(build("Python 项目使用 uv 进行包管理和命令执行。", "workflow", ["package-manager", "uv", "python"]))
    if "pytest" in lowered:
        canonical.append(build("Python 测试使用 pytest，验证时应优先运行对应测试命令。", "workflow", ["testing", "python", "pytest"]))
    if "fastapi" in lowered:
        canonical.append(build("项目使用 FastAPI 作为 Python Web 框架。", "project_fact", ["framework", "python", "fastapi"]))
    if "正式记忆" in statement and "人工审核" in statement:
        review_scope = "project" if project else scope
        canonical.append({**build("正式记忆必须经过人工审核，只有审核通过的候选才允许写入长期记忆。", "workflow", ["review-gate", "memory-safety"]), "scope": review_scope, "project": project if review_scope == "project" else None})
    if "api" in lowered and ("真实" in statement or "ui" in lowered or "流程" in statement) and ("验证" in statement or "跑" in statement):
        canonical.append({**build("不要只看 API 返回成功就判断问题已修复，涉及登录跳转、退出状态等可见产品问题必须真实跑 UI 流程验证。", "pitfall", ["pitfall", "ui-validation", "real-flow"]), "scope": "user", "project": None})
    if "未经审核" in statement and "自动写入" in statement:
        rejected_scope = "project" if project else scope
        canonical.append({**build("不要把未经审核的候选自动写入长期记忆或 MEMORY.md。", "rejected_option", ["rejected-option", "memory-safety", "review-gate"]), "scope": rejected_scope, "project": project if rejected_scope == "project" else None})
    return canonical

def build_agent_candidates_from_payload(payload: dict[str, Any], *, project: str | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    atomic_facts = validate_agent_atomic_facts(list(payload.get("atomic_facts", [])), project=project)
    if atomic_facts:
        return atomic_facts, build_candidates_from_facts(atomic_facts)
    return [], validate_agent_candidates(list(payload.get("candidates", [])), project=project)


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
    runtime_config: dict[str, Any] | None = None,
    trace_callback: Any = None,
) -> dict[str, Any]:
    prompt = build_memory_extraction_prompt(events, project=project)
    counts = _prompt_event_counts(events)
    runtime = "direct-memory-extraction"
    if not invoke_model:
        return {
            "runtime": runtime,
            "dry_run": True,
            "model": str(model),
            "prompt": prompt,
            "atomic_facts": [],
            "candidates": [],
            **counts,
        }

    model_runtime: dict[str, Any] | None = None
    if isinstance(runtime_config, dict):
        result = invoke_model_runtime(
            prompt,
            runtime_config=runtime_config,
            trace_callback=trace_callback,
        )
        text = result.text
        model_runtime = result.to_dict()
    elif isinstance(model, str):
        text = invoke_model_provider(prompt, model=model)
    else:
        result = model.invoke(prompt)
        text = _message_text({"messages": [result]})

    payload = extract_json_payload(text)
    atomic_facts, candidates = build_agent_candidates_from_payload(payload, project=project)
    response: dict[str, Any] = {
        "runtime": runtime,
        "dry_run": False,
        "model": str(model),
        "prompt": prompt,
        "raw_response": text,
        "atomic_facts": atomic_facts,
        "candidates": candidates,
        **counts,
    }
    if model_runtime is not None:
        response["model_runtime"] = model_runtime
    return response
