from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from .memory_models import build_atomic_fact, build_memory_card, build_review_decision, build_review_queue_item

SENSITIVE_RE = re.compile(
    r"(sk-[a-zA-Z0-9]|api[_-]?key\s*[=:]|access[_-]?token\s*[=:]|refresh[_-]?token\s*[=:]|password\s*[=:]|secret\s*[=:]|cookie\s*[=:]|bearer\s+[a-zA-Z0-9]|-----BEGIN [A-Z ]*PRIVATE KEY-----|AKIA[0-9A-Z]{16}|eyJ[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+)",
    re.I,
)
BLOCKED_EVENT_TYPES = {"project_state", "tool_output", "build_log"}
RAW_TRANSCRIPT_RE = re.compile(r"(^|\n)\s*(user|assistant|system)\s*:", re.I)
TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]+|[\u4e00-\u9fff]{2,}")
ONE_OFF_MEMORY_RE = re.compile(
    r"(删除|修改|改为|改成|实现|新增|接入|修复|清理|迁移|跑|测试|生成).{0,24}(页面|组件|按钮|接口|脚本|水印|首页|配置|任务|数据)",
    re.I,
)
LOW_VALUE_EVENT_RE = re.compile(
    r"^(Exit code|Traceback|File created successfully|File does not exist"
    r"|The file .+ has been (updated|created|deleted) successfully"
    r"|Note: your current working directory|WARNING:|Task #\d+ created successfully"
    r"|Using [a-z0-9:_-]+ to |[-dlrwxs]{10}\s+\d+\s+"
    r"|我需要先|我需要查看|我需要继续|让我|好的[，,]|现在开始|已开始并行生成"
    r"|generated_bills_|你好！有什么需要我帮忙的吗|我理解您想"
    r"|Fichier créé avec succès|Le fichier .+ a été mis à jour"
    r"|remotes/origin/|  remotes/|origin/HEAD ->"
    r"|<system.?reminder|<system_reminder|<image name=|<tool_use_error"
    r"|\d+\tINFO:\s+|\d+\tWARNING:\s+|\d+\tERROR:\s+"
    r"|Async agent launched successfully\. agentId:"
    r"|Commande s.ex.cutant en arri.re-plan avec"
    r"|Already on '|Switched to (a new )?branch"
    r"|Handles\s+NPM\(K\)\s+PM\(K\)"
    r"|\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+ \| (INFO|WARNING|ERROR|DEBUG)\s+\|"
    r"|HTTP/1\.[01]\" [1-5]\d{2}"
    r"|Initialized empty Git repository"
    r"|^Python \d+\.\d+\.\d+"
    r"|^(uv|pip|npm|node|cargo|go|rustc|java|mvn) \d+\.\d+"
    r"|^fatal: (not a git|couldn't|unable to|ambiguous)"
    r"|^(ls|cat|sh|bash): (cannot|can't|No such)"
    r"|^Ŀ¼:|^目录:|^LastWriteTime\s+Length"
    r"|^\[dream-memory\]"
    r")",
    re.I,
)
TASK_BRIEF_RE = re.compile(r"^\s*\d+\.\s*TASK:", re.I)
SHORT_ONE_OFF_RE = re.compile(r"(不需要|不要|改成|改为|名字|箭头|图片|模板|脚本|字体|生成|重新生成|多久|侧边栏|样式|快麦|最近爬去|六角|星号|摄像头|openclaw)", re.I)
TRANSIENT_QUESTION_RE = re.compile(r"(怎么|如何|哪里|多久|需不需要|要不要|该怎么办|是什么|吗[？?]?$|为什么|能不能|可不可以|是否)")
CODE_DUMP_RE = re.compile(r"(^|\n)\s*\d+\s*\t\s*(from|import|class|def|if|for|while|return|#|//|const|let|function|<template>)\b", re.I)
DIRECTORY_DUMP_RE = re.compile(r"(^|\n)={3,}[^\n]{0,80}={3,}($|\n)")
MEMORY_PRODUCT_DIRECTION_RE = re.compile(r"(整理|导入|提取).{0,24}(claude code|codex|会话|聊天).{0,40}(记忆|memory|agent)", re.I)
MEMORY_REVIEW_GATE_RE = re.compile(r"(人工审核|审核).{0,18}(写入|落|正式).{0,12}(记忆|memory)|只有通过.{0,12}(写入|落).{0,12}(记忆|memory)", re.I)
AUTONOMY_PREFERENCE_RE = re.compile(r"(不需要|不用).{0,10}(再)?问我|按照你的建议|你来决定|你决定|直接做|你推荐", re.I)
DEFAULT_DREAM_PROMOTION_POLICY: dict[str, Any] = {
    "promote_threshold": 0.7,
    "review_threshold": 0.45,
    "reject_one_off": True,
    "require_evidence": True,
    "duplicate_action": "reject",
    "conflict_promote_action": "merge",
}
ACTION_ORDER = ["create", "merge", "needs_more_evidence", "review", "reject"]
TASK_INTENT_ALIASES: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
    (("提交", "推送", "拉取", "分支", "版本号", "commit", "push", "branch"), ("git", "repo-maintenance", "commit", "push", "branch", "仓库维护", "提交", "推送", "分支")),
    (("梳理", "审查", "分析", "可行性", "看看", "review"), ("repo-inspection", "code-grounded", "analysis", "真实仓库", "梳理", "审查", "分析")),
    (("继续", "下一步", "开始", "做"), ("continuation", "execution-style", "继续", "下一步", "已确认计划")),
    (("agent", "协作", "几个人", "并行"), ("agent-team", "task-planning", "协作", "并行")),
    (("中文", "语言"), ("language", "chinese", "中文")),
    (("不需要再问", "不用问", "按照你的建议", "直接推进", "直接做", "你决定", "你推荐"), ("autonomy", "direct-execution", "直接推进", "不要反复询问", "判断直接推进")),
    (("人工审核", "审核", "正式记忆", "写入记忆", "长期记忆"), ("review-gate", "memory-safety", "人工审核", "正式记忆", "长期记忆")),
    (("测试", "跑测试", "验证", "单测", "pytest", "unittest", "test"), ("testing", "pytest", "unittest", "测试", "验证", "python 测试")),
    (("uv", "依赖", "包管理", "安装", "运行命令", "sync"), ("package-manager", "uv", "python", "包管理", "命令执行")),
    (("fastapi", "后端", "接口", "api", "web 框架", "服务"), ("framework", "fastapi", "python web", "后端", "接口")),
]
DURABLE_MEMORY_MARKERS = [
    "记住",
    "始终",
    "偏好",
    "必须",
    "人工审核",
    "长期记忆",
    "正式记忆",
    "按照你的建议",
    "问我",
    "直接做",
    "你决定",
    "你推荐",
]


@dataclass(frozen=True)
class DreamResult:
    event_count: int
    candidate_count: int
    promoted_count: int
    review_count: int
    rejected_count: int
    output_dir: str
    candidates_path: str
    dreams_path: str
    memory_preview_path: str
    memory_path: str | None
    applied: bool

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def load_events_jsonl(path: Path | str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with Path(path).expanduser().open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                events.append(obj)
    return events



def write_jsonl_records(records: list[dict[str, Any]], path: Path | str) -> Path:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(output)
    return output


def normalize_project_path(project: str | None) -> str | None:
    if not project:
        return None
    raw = str(project).strip()
    if not raw:
        return None
    if raw.startswith("/") and not raw.startswith("//"):
        return PurePosixPath(raw).as_posix()
    return str(Path(raw).expanduser().absolute())


def normalize_memory_text(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value).strip().lower())
    text = re.sub(r"[。．.!！?？,，;；:：]+$", "", text)
    return text.strip()


def _tokens(value: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(str(value)) if len(token.strip()) >= 2}


def _text_similarity(left: str, right: str) -> float:
    left_tokens = _tokens(normalize_memory_text(left))
    right_tokens = _tokens(normalize_memory_text(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _content_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def _is_raw_transcript_like(content: str) -> bool:
    if len(content) > 1200:
        return True
    if content.count("\n") >= 6:
        return True
    if "```" in content:
        return True
    return bool(RAW_TRANSCRIPT_RE.search(content))


def _evidence_refs_from_candidate(candidate: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for index, evidence in enumerate(candidate.get("evidence", []), start=1):
        if isinstance(evidence, dict):
            ref = evidence.get("event_id") or evidence.get("id") or evidence.get("source_event_id")
            if ref:
                refs.append(str(ref))
            else:
                source = evidence.get("source") or "evidence"
                session = evidence.get("session_id") or index
                refs.append(f"{source}:{session}")
        elif evidence:
            refs.append(str(evidence))
    return refs or [str(candidate.get("id") or "candidate")]


def _event_text(event: dict[str, Any]) -> str:
    return str(event.get("content") or "").strip()


def _event_ref(event: dict[str, Any], fallback_index: int) -> dict[str, Any]:
    content = _event_text(event)
    return {
        "event_id": event.get("event_id") or f"event_{fallback_index}",
        "source": event.get("source"),
        "session_id": event.get("session_id"),
        "event_type": event.get("event_type"),
        "quote": content[:180],
    }


def _build_derived_fact(
    *,
    fact_type: str,
    statement: str,
    scope: str,
    evidence_events: list[tuple[int, dict[str, Any]]],
    confidence: float,
    tags: list[str],
    project: str | None = None,
    reuse_scenarios: list[str] | None = None,
    reason: str = "derived from repeated user behavior",
) -> dict[str, Any]:
    return {
        "id": _candidate_id(normalize_memory_text(statement), scope, project),
        "fact_type": fact_type,
        "statement": statement,
        "scope": scope,
        "project": project,
        "confidence": confidence,
        "tags": tags,
        "evidence": [_event_ref(event, index) for index, event in evidence_events],
        "reuse_scenarios": reuse_scenarios or [],
        "long_term": True,
        "long_term_reason": reason,
    }


def _memory_source_role(event: dict[str, Any]) -> str:
    return str(event.get("role") or event.get("type") or "").strip().lower()


def _is_user_memory_source(event: dict[str, Any]) -> bool:
    role = _memory_source_role(event)
    event_type = str(event.get("event_type") or "")
    return role == "user" or event_type in {"global_instruction", "project_instruction", "project_markers"}


def derive_behavioral_facts(events: list[dict[str, Any]], *, project: str | None) -> list[dict[str, Any]]:
    """Infer durable user-level memories from repeated short commands.

    Direct rules catch explicit sentences, but real agent transcripts often expose
    reusable preferences as command patterns: "继续", "做", "提交",
    "推送", "拉取代码", or "梳理这个项目". These are not good
    standalone memories, but repeated usage is valuable future context.
    """
    indexed = [
        (index, dict(event))
        for index, event in enumerate(events, start=1)
        if _event_matches_project_filter(dict(event), project=project)
    ]

    def matches(predicate) -> list[tuple[int, dict[str, Any]]]:
        return [(index, event) for index, event in indexed if _is_user_memory_source(event) and predicate(_event_text(event), event)]

    derived: list[dict[str, Any]] = []

    continuation = matches(lambda text, event: text in {"继续", "做", "下一步", "开始"})
    if len(continuation) >= 2:
        derived.append(_build_derived_fact(
            fact_type="workflow",
            statement="用户常用“继续/做/下一步”推进已确认路线；当上下文已有明确计划时，应继续执行下一个合理步骤，不要反复要求确认。",
            scope="user",
            evidence_events=continuation[:5],
            confidence=0.86,
            tags=["continuation", "execution-style"],
            reuse_scenarios=["用户说继续、做、下一步或开始时", "已存在明确计划或路线图时"],
        ))

    git_ops = matches(lambda text, event: text in {"提交", "推送", "拉取代码", "查看我当前的分支"} or any(token in text for token in ["新建一个分支", "切换到主分支", "调整主分支的版本号"]))
    if len(git_ops) >= 3:
        derived.append(_build_derived_fact(
            fact_type="workflow",
            statement="用户经常用简短指令要求仓库维护（拉取代码、查看分支、提交、推送、切分支、调版本）；应直接检查真实 git 状态并执行，完成后说明 commit/push/branch 状态。",
            scope="user",
            evidence_events=git_ops[:6],
            confidence=0.88,
            tags=["git", "repo-maintenance", "direct-execution"],
            reuse_scenarios=["用户要求提交、推送、拉取代码、查看分支或切分支时"],
        ))

    cleanup_branch = matches(lambda text, event: any(token in text for token in ["精简", "删除模块", "主分支保留", "新建一个分支"]))
    if len(cleanup_branch) >= 2:
        derived.append(_build_derived_fact(
            fact_type="workflow",
            statement="做大规模功能精简或删除模块时，用户偏好先新建独立分支、主分支保留完整功能，并按计划逐步删除和验证。",
            scope="user",
            evidence_events=cleanup_branch[:5],
            confidence=0.82,
            tags=["branching", "cleanup", "risk-control"],
            reuse_scenarios=["大规模删除功能、精简项目或拆分分支时"],
        ))

    project_review = matches(lambda text, event: text == "梳理这个项目" or "审查" in text or "完整收尾" in text or "先分析" in text or "看看" in text)
    if len(project_review) >= 2:
        derived.append(_build_derived_fact(
            fact_type="workflow",
            statement="用户要求梳理、审查或分析项目时，期望先读取真实仓库结构、关键文件和测试现状，再给出代码依据充分的结论或修改方案。",
            scope="user",
            evidence_events=project_review[:5],
            confidence=0.8,
            tags=["code-grounded", "repo-inspection", "analysis"],
            reuse_scenarios=["用户要求梳理项目、审查链路、分析可行性时"],
        ))

    # Only split language / agent-team directives out of explicit global instructions.
    # Ordinary user preference events such as “用户偏好中文回答” are already captured
    # directly and should not produce a second normalized duplicate candidate.
    language_team = matches(lambda text, event: str(event.get("event_type") or "") == "global_instruction" and ("中文" in text or "agent team" in text or "几个人工作" in text))
    if language_team:
        text = _event_text(language_team[0][1])
        if "中文" in text:
            derived.append(_build_derived_fact(
                fact_type="preference",
                statement="用户偏好始终使用中文回答。",
                scope="user",
                evidence_events=language_team[:2],
                confidence=0.96,
                tags=["language", "chinese"],
                reuse_scenarios=["所有对话回复"],
                reason="explicit global instruction",
            ))
        if "agent team" in text or "几个人工作" in text:
            derived.append(_build_derived_fact(
                fact_type="workflow",
                statement="用户希望布置任务时先判断需要几个人/几个 agent 协作，并在任务适合并行时使用 agent team。",
                scope="user",
                evidence_events=language_team[:2],
                confidence=0.94,
                tags=["agent-team", "task-planning"],
                reuse_scenarios=["复杂任务、多模块任务或可并行任务开始前"],
                reason="explicit global instruction",
            ))

    unique: dict[str, dict[str, Any]] = {}
    for fact in derived:
        key = _candidate_id(normalize_memory_text(str(fact.get("statement") or "")), str(fact.get("scope") or "user"), str(fact.get("project")) if fact.get("project") else None)
        unique[key] = fact
    return list(unique.values())


def _is_product_direction_content(content: str) -> bool:
    return bool(MEMORY_PRODUCT_DIRECTION_RE.search(content))


def _is_memory_review_gate_content(content: str) -> bool:
    return bool(MEMORY_REVIEW_GATE_RE.search(content))


def _is_autonomy_preference_content(content: str) -> bool:
    return bool(AUTONOMY_PREFERENCE_RE.search(content))


def _is_short_one_off_requirement(content: str) -> bool:
    normalized = normalize_memory_text(content)
    if len(normalized) > 90:
        return False
    if any(token in content for token in DURABLE_MEMORY_MARKERS):
        return False
    return bool(SHORT_ONE_OFF_RE.search(content))



def _is_code_or_listing_dump_content(content: str) -> bool:
    stripped = str(content or "").strip()
    if not stripped:
        return False
    lines = [line for line in stripped.splitlines() if line.strip()]
    if len(lines) < 5:
        return False
    lowered = stripped.lower()

    large_dump_markers = [
        "base directory for this skill:",
        "the user just ran /insights",
        "here is the full insights data:",
        "# update config skill",
        "## skills",
        "a skill is a set of local instructions",
        "<ultrawork-mode>",
        "</ultrawork-mode>",
        "[code red]",
        "<codex_internal_context",
        "</codex_internal_context>",
        "<purpose>",
        "<required_reading>",
        "<process>",
        "<<<<<<< head",
        "failed to load resource",
        "[routeguard]",
        "# 知识图谱全面改进",
        "项目目标达成",
        "最终总结",
        "this file provides guidance to codex",
        "# agents.md",
        'run the "deep-research" workflow',
        "deep research harness",
        "imagefont.truetype",
        "draw.text(",
        "===docs tree===",
        "===backend root===",
        "=== 磁盘空间",
        "=== 内存使用评估",
        "错误的推断过程",
        "当前磁盘空间",
        "test session starts",
        "collected ",
        " passed [",
        "please analyze this codebase and create a claude.md",
        "what to add:",
        "修改账单图片中的文字信息",
        "edit_bill_image",
        "imagefont.truetype",
    ]
    if any(marker in lowered for marker in large_dump_markers):
        return True

    numbered = sum(1 for line in lines if re.match(r"^\s*\d+\s*\t", line))
    codeish = sum(
        1
        for line in lines
        if re.search(r"\b(from|import|class|def|return|const|let|function|dependencies)\b|<template>|</template>|\{|\}|\[project\]", line)
    )
    if numbered >= 3 and (codeish >= 2 or CODE_DUMP_RE.search(stripped)):
        return True

    unnumbered_codeish = sum(
        1
        for line in lines
        if re.search(r'^\s*(from|import|class|def)\b|^\s*"""|^\s*async\s+def\b|imagefont\.|image\.', line.lower())
    )
    if len(stripped) > 500 and unnumbered_codeish >= 3:
        return True

    ls_listing_lines = sum(1 for line in lines if re.match(r"^[d-][rwx-]{9}\s+\d+\s+", line.strip()) or line.strip().startswith("total "))
    if ls_listing_lines >= 3:
        return True

    absolute_path_lines = sum(1 for line in lines if re.match(r"^/(Users|tmp|var|private|home)/.*\.(py|ts|tsx|vue|json|toml|md|lock|pyc|ttf|otf|woff|woff2)$", line.strip()))
    if absolute_path_lines >= 5:
        return True

    if DIRECTORY_DUMP_RE.search(stripped):
        fileish = sum(1 for line in lines if re.search(r"(\.py|\.ts|\.vue|\.json|\.toml|\.lock|\.md|__pycache__|node_modules|tests?$)", line))
        if fileish >= 3:
            return True

    if stripped.startswith("On branch ") and ("Changes to be committed:" in stripped or "Unmerged paths:" in stripped):
        return True
    return False

def _is_low_value_event_content(content: str) -> bool:
    durable_hint = any(token in content for token in [
        "用户偏好",
        "偏好",
        "始终",
        "必须",
        "人工审核",
        "长期记忆",
        "正式记忆",
        "不需要再问",
        "不用问",
        "按照你的建议",
        "你决定",
        "直接做",
        "直接推进",
    ])
    if LOW_VALUE_EVENT_RE.search(content):
        return True
    if TASK_BRIEF_RE.search(content):
        return True
    if "file state is current in your context" in content:
        return True
    if "No need to Read it back" in content:
        return True
    if "pip install --upgrade pip" in content:
        return True
    if content.startswith("# In app browser:"):
        return True
    if content.startswith("# Files mentioned by the user:"):
        return True
    if "## My request for Codex:" in content and content.startswith("#"):
        return True
    # 工具成功/失败输出
    if re.search(r"has been (updated|created|deleted|written) successfully", content, re.I):
        return True
    if re.search(r"^(File|Directory) (does not exist|not found|already exists)", content, re.I):
        return True
    if re.search(r"^\s*\(file state is current", content, re.I):
        return True
    if re.search(r"^Bash tool|^PowerShell tool|^Read tool|^Edit tool|^Write tool|^Glob tool|^Grep tool", content, re.I):
        return True
    # runtime system messages that appear anywhere in content
    if "Async agent launched successfully" in content:
        return True
    # 后台命令消息（英文和法语版本）
    # Background command messages (English and French)
    if "Output is being written to:" in content:
        if "running in background with ID:" in content or "en arrière-plan avec" in content:
            return True
    if "Web search results for query:" in content:
        return True
    # ANSI 转义码（日志彩色输出）
    # ANSI escape codes in log output
    if "\x1b[" in content:
        return True
    # numbered-line prefixed content (Read tool output)
    if re.match(r"^\d+\t", content) and content.count("\t") >= 3:
        return True
    # pure JSON structures (config dumps, API responses)
    stripped_c = content.strip()
    if len(stripped_c) > 50 and (
        (stripped_c.startswith("{") and stripped_c.endswith("}"))
        or (stripped_c.startswith("[") and stripped_c.endswith("]"))
    ):
        try:
            json.loads(stripped_c)
            return True
        except Exception:
            pass
    # Windows directory listing header
    if re.match(r"^(Mode\s+LastWriteTime\s+Length|Directory of )", content, re.I):
        return True
    # garbled encoding: many U+FFFD replacement characters
    if content.count("\ufffd") > 3:
        return True
    # git branch listing (output of git branch -a)
    if re.match(r"^[*+ ] ", content) and "remotes/" in content:
        return True
    # single bare path line
    stripped_c2 = content.strip()
    if (
        re.match(r"^([a-zA-Z]:[/\\]|/[a-z]|/[cd]/)[^\n]{5,80}$", stripped_c2)
        and "\n" not in stripped_c2
    ):
        return True
    # Python sys.path list or ModuleSpec
    if re.match(r"^\['?'?,? ?'?[a-zA-Z]:", content) or re.match(r"^ModuleSpec\(name=", content):
        return True
    # git status modified-file lines
    if re.match(r"^[MADRCU?!]{1,2}\s+(backend|frontend|src|tests|app)/", content):
        return True
    # multi-line path list (60%+ are path lines)
    lines_c = [l.strip() for l in content.splitlines() if l.strip()]
    if lines_c and len(lines_c) >= 2:
        path_lines = sum(
            1 for l in lines_c
            if re.match(r"^([a-zA-Z]:[/\\]|/[a-z]|/[cd]/|\.worktrees|\.venv)", l)
        )
        if path_lines / len(lines_c) > 0.6:
            return True
    if TRANSIENT_QUESTION_RE.search(content) and not durable_hint:
        return True
    # <system_reminder> blocks (localized tags)
    if re.match(r"^<system[_-]?reminder", content, re.I):
        return True
    # dream-memory CLI progress messages captured as events
    if re.match(r"^\[dream-memory\]", content):
        return True
    # <image name= blocks (codex clipboard / screenshot paths inside content)
    if re.search(r"<image name=\[Image #\d+\]", content):
        return True
    # database migration / alembic output
    if re.search(r"(alembic_version|Table .+ already exists|alembic upgrade)", content, re.I):
        return True
    # debug session output (key=value lines from our own debug scripts)
    if re.match(r"^has [a-z _]+: (True|False)\r?\n", content):
        return True
    # relative path lists (./foo/bar.py style, 2+ lines, 60%+ are path lines)
    if lines_c and len(lines_c) >= 2:
        rel_path_lines = sum(1 for l in lines_c if re.match(r"^\./[\w/._-]+\.(py|ts|js|go|rs|md|json|yaml|yml)$", l))
        if rel_path_lines / len(lines_c) > 0.6:
            return True
    # deep venv / worktrees site-packages paths
    if re.match(r"^\.worktrees[\\/]|.*site-packages[\\/]", stripped_c2, re.I):
        return True
    # pip install command history (grep output)
    if re.search(r"^\d+:pip install\b", content, re.M) and content.count("pip install") >= 2:
        return True
    # ls -la style output block (multiple rwx lines)
    rwx_lines = sum(1 for l in (lines_c or []) if re.match(r"^-?[dlrwxs-]{9}", l))
    if rwx_lines >= 2:
        return True
    # grep output: path:line_num:content pattern (2+ lines)
    grep_lines = sum(1 for l in (lines_c or []) if re.match(r"^[^\s:]+\.(py|ts|js|vue|go|rs|java):\d+:", l))
    if grep_lines >= 2:
        return True
    # git stash message
    if re.match(r"^Saved working directory and index state", content):
        return True
    # PowerShell Get-ChildItem header
    if re.match(r"^Path\s*\r?\n-+", content) or re.match(r"^\s*Path\s+LastWriteTime", content, re.I):
        return True
    # branch name lists (multiple feature/worktree/codex lines)
    branch_lines = sum(1 for l in (lines_c or []) if re.match(r"^(feature/|worktree-|codex/|hotfix/|bugfix/|release/)", l))
    if lines_c and len(lines_c) >= 2 and branch_lines / len(lines_c) > 0.6:
        return True
    # bare URL lines with no surrounding context
    if re.match(r"^https?://[^\s]+$", stripped_c2) and "\n" not in stripped_c2:
        return True
    if durable_hint:
        return False
    # GBK/GB2312 mojibake: dense non-CJK-range bytes mixed with latin
    mojibake_count = len(re.findall(r"[\x80-\xbf]", content.encode("latin-1", errors="replace").decode("latin-1")))
    if mojibake_count > 8:
        return True
    # storage/snapshot path lists (app internal storage structure)
    if re.match(r"^storage[\\/](snapshots|repair_requests|runs)[\\/]", stripped_c2):
        return True
    # single-line grep output (file:line:content, not multi-line); search (not match)
    # so Windows drive-letter paths (D:\...) don't break the anchor
    if re.search(r"[^\s:]+\.(py|ts|js|vue|go|rs|java):\d+:\s+\S", stripped_c2) and "\n" not in stripped_c2:
        return True
    # content too short to be a memory (< 12 chars after strip)
    if len(stripped_c2) < 12 and not durable_hint:
        return True
    # IDE auto-notification (file opened in editor)
    if "<ide_opened_file>" in content or "<ide_selection>" in content:
        return True
    # python interpreter crash / cannot open file
    if re.search(r"python\.exe: can't open file", content, re.I):
        return True
    # loguru-style log line anywhere in content (not just at start)
    if re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+ \| (INFO|WARNING|ERROR|DEBUG)\s+\|", content):
        return True
    # ad-hoc test/verification script header + interpreter version dump
    if re.search(r"={3,}\s*(验证|测试).{0,10}={3,}", content) and re.search(r"Python\s*(版本|version)?\s*[:：]?\s*3\.\d+", content, re.I):
        return True
    # script progress output: "正在移动/复制/处理 N 张/个/条 ... 到/文件夹"
    if re.search(r"正在(移动|复制|处理)\s*\d+\s*(张|个|条).{0,20}(文件夹|到)", content):
        return True
    # shell cwd reset notice (appears standalone or appended to other tool output)
    if "Shell cwd was reset to" in content:
        return True
    # find/grep "no such file or directory" tool errors
    if re.search(r"^(find|grep):\s.+No such file or directory", content, re.I):
        return True
    # bare file path with no explanatory sentence (single or few lines, no CJK/sentence content)
    if lines_c and 1 <= len(lines_c) <= 3:
        path_only_lines = sum(
            1 for l in lines_c
            if re.match(r"^([a-zA-Z]:[\\/]|/[a-z]|\./|\.\./)[^\s]*$", l)
        )
        if path_only_lines == len(lines_c):
            return True
    # simple confirmation messages ("uv.lock 已找到" style, no explanation)
    if re.match(r"^[\w.\-]+\s*[\r\n]+\s*(✅|❌)\s*", content):
        return True
    # tsc/typescript compiler error output
    if re.search(r"error TS\d+:", content):
        return True
    return False







def _is_structural_or_one_off_artifact(content: str, event_type: str) -> bool:
    if event_type in {"global_instruction", "project_instruction", "project_markers"}:
        return False
    stripped = str(content or "").strip()
    lowered = stripped.lower()
    durable_markers = ["用户偏好", "始终使用中文", "不需要再问", "不用问", "按照你的建议", "你决定", "直接做", "直接推进", "人工审核", "长期记忆", "正式记忆"]
    if any(marker in stripped for marker in durable_markers):
        return False
    if len(stripped) > 500 and any(marker in lowered for marker in ["请修复", "修改文件", "目标", "localstorage", "jwt token", "httpOnly".lower()]):
        return True
    if (
        "this file provides guidance to codex" in lowered
        or "please analyze this codebase and create a claude.md" in lowered
        or "what to add:" in lowered and "commands that will be commonly used" in lowered
        or stripped.startswith("# AGENTS.md")
        or "箭头图片" in stripped
        or "箭头字体" in stripped
        or "font_arrow" in lowered
        or "arrow_img" in lowered
        or "typescript类型系统" in lowered
        or "eslint配置" in lowered
        or "⚠️ 待完善" in stripped
        or "磁盘空间推断" in stripped
        or "内存使用评估" in stripped
    ):
        return True
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if len(lines) >= 8:
        short_pathish = sum(1 for line in lines if re.match(r"^[A-Za-z0-9_.@/-]+$", line) or line.startswith("---"))
        fileish = sum(1 for line in lines if re.search(r"(\.py|\.ts|\.vue|\.json|\.toml|\.lock|__tests__|assets|components|stores|views|router|utils|api$|app\.vue|main\.ts)", line, re.I))
        if short_pathish >= 7 and fileish >= 4:
            return True
    return False

def _is_long_generic_memory_content(content: str, event_type: str) -> bool:
    if event_type in {"global_instruction", "project_instruction", "project_markers"}:
        return False
    stripped = str(content or "").strip()
    if len(stripped) <= 1500:
        return False
    durable_markers = [
        "人工审核",
        "长期记忆",
        "按照你的建议",
        "不需要再问",
        "用户偏好",
        "始终使用中文",
    ]
    if any(marker in stripped for marker in durable_markers):
        return False
    return True


def _merge_fact_evidence(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    merged_refs = list(merged.get("evidence_refs") or [])
    for ref in incoming.get("evidence_refs") or []:
        if ref not in merged_refs:
            merged_refs.append(ref)
    if merged_refs:
        merged["evidence_refs"] = merged_refs

    merged_evidence = list(merged.get("evidence") or [])
    seen_evidence = {json.dumps(item, sort_keys=True, ensure_ascii=False) for item in merged_evidence if isinstance(item, dict)}
    for item in incoming.get("evidence") or []:
        if not isinstance(item, dict):
            continue
        key = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if key not in seen_evidence:
            merged_evidence.append(item)
            seen_evidence.add(key)
    if merged_evidence:
        merged["evidence"] = merged_evidence

    merged["tags"] = sorted(set(merged.get("tags") or []) | set(incoming.get("tags") or []))
    try:
        merged["confidence"] = round(max(float(merged.get("confidence", 0) or 0), float(incoming.get("confidence", 0) or 0)), 3)
    except (TypeError, ValueError):
        pass
    if incoming.get("long_term") is True:
        merged["long_term"] = True
    if incoming.get("long_term_reason") and incoming.get("long_term_reason") != merged.get("long_term_reason"):
        reasons = [reason for reason in [merged.get("long_term_reason"), incoming.get("long_term_reason")] if reason]
        merged["long_term_reason"] = "; ".join(dict.fromkeys(str(reason) for reason in reasons))
    return merged


def _dedupe_atomic_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[tuple[str, str, str | None, str], dict[str, Any]] = {}
    order: list[tuple[str, str, str | None, str]] = []
    for fact in facts:
        statement = str(fact.get("statement") or "").strip()
        key = (
            str(fact.get("fact_type") or ""),
            str(fact.get("scope") or "global"),
            normalize_project_path(str(fact.get("project"))) if fact.get("project") else None,
            normalize_memory_text(statement),
        )
        if key not in deduped:
            deduped[key] = fact
            order.append(key)
        else:
            deduped[key] = _merge_fact_evidence(deduped[key], fact)
    return [deduped[key] for key in order]

def _event_matches_project_filter(event: dict[str, Any], *, project: str | None) -> bool:
    """Return whether an event belongs to the requested project context.

    Project-scoped system metadata from another project must not leak into the
    current run. User-authored events can still yield user-scope memories such
    as language preferences or repeated workflow preferences across projects.
    """
    if not project or not event.get("project"):
        return True
    expected_project = normalize_project_path(project)
    event_project = normalize_project_path(str(event.get("project")))
    if not expected_project or not event_project or event_project == expected_project:
        return True
    role = _memory_source_role(event)
    event_type = str(event.get("event_type") or "")
    return role == "user" or event_type == "global_instruction"


def extract_atomic_facts(events: list[dict[str, Any]], *, project: str | None) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for index, raw_event in enumerate(events, start=1):
        event = dict(raw_event)
        event.setdefault("event_id", f"event_{index}")
        event_type = str(event.get("event_type") or "")
        if event_type in BLOCKED_EVENT_TYPES:
            continue
        if not _event_matches_project_filter(event, project=project):
            continue
        content = str(event.get("content") or "").strip()
        if not content or SENSITIVE_RE.search(content):
            continue
        if not _is_user_memory_source(event):
            continue
        if _is_low_value_event_content(content):
            continue
        if event_type != "project_markers" and _is_code_or_listing_dump_content(content):
            continue
        if _is_long_generic_memory_content(content, event_type):
            continue
        if _is_structural_or_one_off_artifact(content, event_type):
            continue
        lowered = content.lower()
        event_project = normalize_project_path(str(event.get("project") or project)) if (event.get("project") or project) else None

        if "不要" in content and "未经审核" in content and "自动写入" in content and ("风险" in content or "方案" in content):
            facts.append(build_atomic_fact(
                fact_type="rejected_option",
                statement="不要把未经审核的候选自动写入长期记忆或 MEMORY.md。",
                scope="project" if event_project else "global",
                project=str(event_project) if event_project else None,
                source_event=event,
                confidence=0.86,
                tags=["rejected-option", "memory-safety", "review-gate"],
            ))
            continue

        if _is_short_one_off_requirement(content):
            continue

        if event_type == "project_instruction" and "uv" in lowered and "pnpm" in lowered:
            facts.append(build_atomic_fact(
                fact_type="workflow",
                statement="Python 后端使用 uv 进行包管理，前端使用 pnpm 进行管理。",
                scope="project" if event_project else "global",
                project=str(event_project) if event_project else None,
                source_event=event,
                confidence=0.92,
                tags=["package-manager", "uv", "pnpm", "python", "frontend"],
            ))
            continue

        if event_type == "project_markers":
            has_uv = "python_package_manager=uv" in lowered
            has_pnpm = "frontend_package_manager=pnpm" in lowered
            has_unittest = "python_test_runner=unittest" in lowered
            has_pytest = "python_test_runner=pytest" in lowered
            has_fastapi = "python_framework=fastapi" in lowered
            has_django = "python_framework=django" in lowered
            if has_uv or has_pnpm:
                statement_parts = []
                tags = ["package-manager"]
                if has_uv:
                    statement_parts.append("Python 项目使用 uv 进行包管理和命令执行")
                    tags.extend(["uv", "python"])
                if has_pnpm:
                    statement_parts.append("前端项目使用 pnpm 进行包管理和脚本执行")
                    tags.extend(["pnpm", "frontend"])
                facts.append(build_atomic_fact(
                    fact_type="workflow",
                    statement="，".join(statement_parts) + "。",
                    scope="project" if event_project else "global",
                    project=str(event_project) if event_project else None,
                    source_event=event,
                    confidence=0.86,
                    tags=tags,
                ))
            if has_unittest or has_pytest:
                runner = "pytest" if has_pytest else "unittest"
                facts.append(build_atomic_fact(
                    fact_type="workflow",
                    statement=f"Python 测试使用 {runner}，验证时应优先运行对应测试命令。",
                    scope="project" if event_project else "global",
                    project=str(event_project) if event_project else None,
                    source_event=event,
                    confidence=0.82,
                    tags=["testing", "python", runner],
                ))
            if has_fastapi or has_django:
                framework = "FastAPI" if has_fastapi else "Django"
                facts.append(build_atomic_fact(
                    fact_type="project_fact",
                    statement=f"项目使用 {framework} 作为 Python Web 框架。",
                    scope="project" if event_project else "global",
                    project=str(event_project) if event_project else None,
                    source_event=event,
                    confidence=0.78,
                    tags=["framework", "python", framework.lower()],
                ))
            continue

        if _is_product_direction_content(content):
            facts.append(build_atomic_fact(
                fact_type="product_direction",
                statement="Dream Memory 的产品方向是整理 Claude Code 和 Codex 会话，从中提取关键可复用信息并形成可被后续 agent 使用的共享记忆。",
                scope="project" if event_project else "global",
                project=str(event_project) if event_project else None,
                source_event=event,
                confidence=0.9,
                tags=["product-direction", "memory", "claude-code", "codex"],
            ))
            continue

        if _is_memory_review_gate_content(content):
            facts.append(build_atomic_fact(
                fact_type="workflow",
                statement="正式记忆必须经过人工审核，只有审核通过的候选才允许写入长期记忆。",
                scope="project" if event_project else "global",
                project=str(event_project) if event_project else None,
                source_event=event,
                confidence=0.9,
                tags=["review-gate", "memory-safety"],
            ))
            continue

        if _is_autonomy_preference_content(content):
            facts.append(build_atomic_fact(
                fact_type="preference",
                statement="用户偏好在上下文清楚时由助手按判断直接推进，不要反复询问确认。",
                scope="user",
                project=None,
                source_event=event,
                confidence=0.88,
                tags=["autonomy", "direct-execution"],
            ))
            continue

        if event_type != "global_instruction" and "中文" in content and ("回答" in content or "回复" in content) and ("偏好" in content or "始终" in content or "请" in content):
            facts.append(build_atomic_fact(
                fact_type="preference",
                statement="用户偏好中文回答。",
                scope="user",
                project=None,
                source_event=event,
                confidence=0.95,
                tags=["preference", "language", "chinese"],
            ))
            continue

        if "不要只看" in content and ("真实跑" in content or "真实" in content and "验证" in content):
            facts.append(build_atomic_fact(
                fact_type="pitfall",
                statement="不要只看 API 返回成功就判断问题已修复，涉及登录跳转、退出状态等可见产品问题必须真实跑 UI 流程验证。",
                scope="user",
                project=None,
                source_event=event,
                confidence=0.88,
                tags=["pitfall", "ui-validation", "real-flow"],
            ))
            continue

        # Global instructions are split into smaller behavioral memories by
        # derive_behavioral_facts() so one long instruction does not duplicate
        # more specific preferences/workflows such as language and agent-team use.
        if event_type != "global_instruction" and ("始终" in content or "偏好" in content or "prefer" in lowered):
            facts.append(build_atomic_fact(
                fact_type="preference",
                statement=content,
                scope="user",
                project=None,
                source_event=event,
                confidence=0.95,
                tags=["preference"],
            ))

        # Global instructions often contain words like “需要”, but they describe a
        # user-level preference/workflow, not a project-scoped requirement.
        if event_type != "global_instruction" and any(word in content for word in ["希望", "想", "需要", "不要", "必须", "人工审核"]):
            facts.append(build_atomic_fact(
                fact_type="requirement",
                statement=content,
                scope="project" if event_project else "global",
                project=str(event_project) if event_project else None,
                source_event=event,
                confidence=0.82,
                tags=["requirement"],
            ))

        if any(token in lowered for token in ["uv", "python", "claude code", "codex", "dream", "runtime", "patch"]) and not _is_low_value_event_content(content):
            tags = [
                tag for tag, needle in [
                    ("uv", "uv"),
                    ("python", "python"),
                    ("dreams", "dream"),
                    ("claude-code", "claude code"),
                    ("codex", "codex"),
                    ("patch", "patch"),
                    ("runtime", "runtime"),
                ] if needle in lowered
            ]
            facts.append(build_atomic_fact(
                fact_type="project_fact" if event_project else "global_fact",
                statement=content,
                scope="project" if event_project else "global",
                project=str(event_project) if event_project else None,
                source_event=event,
                confidence=0.76,
                tags=tags,
            ))
    facts.extend(derive_behavioral_facts(events, project=project))
    return _dedupe_atomic_facts(facts)

def _candidate_id(content: str, scope: str, project: str | None) -> str:
    raw = f"{scope}|{project or ''}|{content}".encode("utf-8")
    return "mem_" + hashlib.sha1(raw).hexdigest()[:12]


def score_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    content = str(candidate.get("content") or "")
    evidence_count = len(candidate.get("evidence", []))
    tags = set(candidate.get("tags", []))
    score = 0.2
    score += min(evidence_count, 4) * 0.12
    if candidate.get("scope") == "project":
        score += 0.18
    if candidate.get("scope") in {"user", "global"}:
        score += 0.12
    if candidate.get("type") in {"preference", "requirement", "project_fact", "workflow", "decision", "pitfall", "product_direction", "rejected_option"}:
        score += 0.18
    if tags & {"uv", "python", "dreams", "claude-code", "codex", "patch", "git", "repo-maintenance", "continuation", "agent-team", "code-grounded"}:
        score += 0.12
    if len(content) >= 12:
        score += 0.08
    if SENSITIVE_RE.search(content):
        score -= 0.6
    score = max(0.0, min(1.0, round(score, 3)))
    candidate = dict(candidate)
    candidate["score"] = score
    candidate["status"] = "promote" if score >= 0.72 else "review" if score >= 0.5 else "reject"
    return candidate



def build_candidates_from_facts(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for fact in facts:
        if fact.get("fact_type") == "system_state" or "project_state" in fact.get("tags", []):
            continue
        content = str(fact.get("statement") or "").strip()
        if not content or SENSITIVE_RE.search(content) or _is_raw_transcript_like(content) or _is_low_value_event_content(content):
            continue
        scope = str(fact.get("scope") or "global")
        project = normalize_project_path(str(fact.get("project"))) if fact.get("project") else None
        key_content = normalize_memory_text(content)
        key = _candidate_id(key_content, scope, str(project) if project else None)
        candidate = candidates.setdefault(key, {
            "id": key,
            "type": fact.get("fact_type"),
            "scope": scope,
            "project": project,
            "content": content,
            "tags": list(fact.get("tags", [])),
            "evidence": [],
            "retrieval_hints": [],
            "quality_reason": "",
        })
        evidence_items = fact.get("evidence") if isinstance(fact.get("evidence"), list) else []
        if evidence_items:
            for evidence in evidence_items:
                if not isinstance(evidence, dict):
                    continue
                candidate["evidence"].append({
                    "event_id": evidence.get("event_id") or evidence.get("id"),
                    "source": evidence.get("source") or fact.get("source"),
                    "session_id": evidence.get("session_id") or fact.get("session_id"),
                    "quote": evidence.get("quote"),
                    "content_hash": _content_hash(content),
                })
        else:
            for ref in fact.get("evidence_refs", []):
                candidate["evidence"].append({
                    "event_id": ref,
                    "source": fact.get("source"),
                    "session_id": fact.get("session_id"),
                    "content_hash": _content_hash(content),
                })
        candidate["tags"] = sorted(set(candidate.get("tags", []) + list(fact.get("tags", []))))
        candidate["retrieval_hints"] = sorted(set(candidate.get("retrieval_hints", []) + [str(item) for item in fact.get("reuse_scenarios", [])]))
        quality_reasons = [candidate.get("quality_reason", "")]
        if fact.get("long_term") is not None:
            quality_reasons.append(f"long_term={bool(fact.get('long_term'))}")
        if fact.get("long_term_reason"):
            quality_reasons.append(str(fact.get("long_term_reason")))
        candidate["quality_reason"] = "; ".join(reason for reason in quality_reasons if reason)
    return [score_candidate(candidate) for candidate in candidates.values()]


def detect_candidate_conflicts(candidates: list[dict[str, Any]], memory_cards: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    conflicts: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        content = str(candidate.get("content") or "")
        candidate_scope = candidate.get("scope")
        candidate_project = candidate.get("project")
        candidate_type = candidate.get("type")
        for card in memory_cards:
            if card.get("status", "active") != "active":
                continue
            if card.get("scope") != candidate_scope:
                continue
            if card.get("project") != candidate_project:
                continue
            if card.get("memory_type") != candidate_type:
                continue
            summary = str(card.get("summary") or "")
            if summary == content:
                continue
            similarity = _text_similarity(content, summary)
            if similarity < 0.18:
                continue
            conflicts.setdefault(str(candidate["id"]), []).append({
                "memory_id": card.get("id"),
                "reason": "similar-same-scope-type",
                "summary": card.get("summary"),
                "similarity": round(similarity, 3),
            })
    return conflicts


def _active_matching_cards(candidate: dict[str, Any], memory_cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidate_scope = candidate.get("scope")
    candidate_project = normalize_project_path(str(candidate.get("project"))) if candidate.get("project") else None
    candidate_type = candidate.get("type")
    matches: list[dict[str, Any]] = []
    for card in memory_cards:
        if card.get("status", "active") != "active":
            continue
        card_project = normalize_project_path(str(card.get("project"))) if card.get("project") else None
        if card.get("scope") == candidate_scope and card_project == candidate_project and card.get("memory_type") == candidate_type:
            matches.append(card)
    return matches


def _evidence_quality(candidate: dict[str, Any]) -> tuple[str, float]:
    evidence = candidate.get("evidence", []) if isinstance(candidate.get("evidence"), list) else []
    tags = {str(tag).lower() for tag in candidate.get("tags", []) if str(tag).strip()}
    event_types = {str(item.get("event_type") or "") for item in evidence if isinstance(item, dict)}
    if "global_instruction" in event_types or "explicit" in tags or "language" in tags:
        return "explicit_instruction", 0.75
    if len(evidence) >= 3:
        return "repeated_behavior", 0.7
    if len(evidence) >= 2:
        return "multi_event", 0.5
    if len(evidence) == 1:
        return "single_event", 0.25
    return "missing", 0.0


def _intent_relevance_boost(task: str | None, searchable: str) -> float:
    if not task:
        return 0.0
    normalized_task = str(task).lower()
    normalized_searchable = searchable.lower()
    boost = 0.0
    matched_trigger_count = 0
    for triggers, targets in TASK_INTENT_ALIASES:
        if any(trigger.lower() in normalized_task for trigger in triggers):
            matched_trigger_count += 1
            if any(target.lower() in normalized_searchable for target in targets):
                boost = max(boost, 0.9)

    # When a short task names one specific execution intent, avoid broad terms
    # like "python" making adjacent project-marker memories tie the precise one.
    if matched_trigger_count == 1 and boost > 0:
        if any(trigger.lower() in normalized_task for trigger in ("uv", "依赖", "包管理", "安装", "运行命令", "sync")):
            if "package-manager" in normalized_searchable or "uv" in normalized_searchable or "包管理" in normalized_searchable or "命令执行" in normalized_searchable:
                return 1.05
        if any(trigger.lower() in normalized_task for trigger in ("测试", "跑测试", "验证", "单测", "pytest", "unittest", "test")):
            if "testing" in normalized_searchable or "pytest" in normalized_searchable or "unittest" in normalized_searchable or "python 测试" in normalized_searchable:
                return 1.05
        if any(trigger.lower() in normalized_task for trigger in ("fastapi", "后端", "接口", "api", "web 框架", "服务")):
            if "framework" in normalized_searchable or "fastapi" in normalized_searchable or "python web" in normalized_searchable:
                return 1.05
    return boost


def explain_candidate_quality(candidate: dict[str, Any], memory_cards: list[dict[str, Any]]) -> dict[str, Any]:
    content = str(candidate.get("content") or "")
    normalized_content = normalize_memory_text(content)
    evidence_count = len(candidate.get("evidence", []))
    evidence_quality, evidence_strength = _evidence_quality(candidate)
    score = float(candidate.get("score", candidate.get("confidence", 0.0)) or 0.0)
    memory_type = str(candidate.get("type") or "")
    tags = {str(tag).lower() for tag in candidate.get("tags", []) if str(tag).strip()}
    one_off = bool(ONE_OFF_MEMORY_RE.search(content)) or "task" in tags
    matching_cards = _active_matching_cards(candidate, memory_cards)
    exact_match = next((card for card in matching_cards if normalize_memory_text(str(card.get("summary") or "")) == normalized_content), None)
    similar_cards = [
        (card, _text_similarity(content, str(card.get("summary") or "")))
        for card in matching_cards
        if normalize_memory_text(str(card.get("summary") or "")) != normalized_content
    ]
    similar_cards = [(card, similarity) for card, similarity in similar_cards if similarity >= 0.18]
    best_similar = max(similar_cards, key=lambda item: item[1], default=(None, 0.0))
    matched_card = exact_match or best_similar[0]
    matched_summary = str(matched_card.get("summary") or "") if isinstance(matched_card, dict) else None
    if exact_match is not None:
        value_class = "existing_duplicate"
    elif matched_card is not None:
        value_class = "similar_existing"
    else:
        value_class = "new_value"
    durable_types = {"preference", "decision", "workflow", "pitfall", "product_direction", "rejected_option"}
    durable_project_fact_tags = {"framework", "package-manager", "testing"}
    is_durable_project_fact = memory_type == "project_fact" and bool(tags & durable_project_fact_tags)
    stability = 0.25
    if memory_type in durable_types or is_durable_project_fact:
        stability += 0.3
    if evidence_quality == "explicit_instruction":
        stability += 0.2
    elif evidence_count >= 2:
        stability += 0.2
    if score >= 0.8:
        stability += 0.15
    if one_off:
        stability -= 0.35
    reuse_value = 0.25
    if memory_type in durable_types or is_durable_project_fact:
        reuse_value += 0.3
    if candidate.get("scope") in {"user", "global", "project"}:
        reuse_value += 0.15
    if candidate.get("tags"):
        reuse_value += 0.1
    if one_off:
        reuse_value -= 0.3
    return {
        "stability": max(0.0, min(1.0, round(stability, 3))),
        "reuse_value": max(0.0, min(1.0, round(reuse_value, 3))),
        "evidence_strength": max(0.0, min(1.0, round(evidence_strength, 3))),
        "evidence_quality": evidence_quality,
        "one_off_task": one_off,
        "duplicate": exact_match is not None,
        "value_class": value_class,
        "similarity": round(1.0 if exact_match else best_similar[1], 3),
        "matched_memory_id": matched_card.get("id") if isinstance(matched_card, dict) else None,
        "matched_memory_summary": matched_summary,
    }


def _policy_value(policy: dict[str, Any], key: str, fallback: Any) -> Any:
    value = policy.get(key, fallback)
    if isinstance(fallback, bool):
        if isinstance(value, str):
            return value.strip().lower() not in {"0", "false", "no", "off", ""}
        return bool(value)
    if isinstance(fallback, float):
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback
    if isinstance(fallback, str):
        return str(value or fallback)
    return value


def _quality_float(quality_signals: dict[str, Any], key: str) -> float:
    try:
        return max(0.0, min(1.0, float(quality_signals.get(key, 0.0) or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _candidate_score(candidate: dict[str, Any]) -> float:
    try:
        return max(0.0, min(1.0, float(candidate.get("score", candidate.get("confidence", 0.0)) or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def analyze_dream_candidate(
    candidate: dict[str, Any],
    *,
    quality_signals: dict[str, Any],
    conflicts: list[dict[str, Any]],
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active_policy = dict(DEFAULT_DREAM_PROMOTION_POLICY)
    if policy:
        active_policy.update(policy)

    promote_threshold = _policy_value(active_policy, "promote_threshold", 0.7)
    review_threshold = _policy_value(active_policy, "review_threshold", 0.45)
    reject_one_off = _policy_value(active_policy, "reject_one_off", True)
    require_evidence = _policy_value(active_policy, "require_evidence", True)
    duplicate_action = _policy_value(active_policy, "duplicate_action", "reject")
    conflict_action = _policy_value(active_policy, "conflict_promote_action", "merge")

    stability = _quality_float(quality_signals, "stability")
    reuse_value = _quality_float(quality_signals, "reuse_value")
    evidence_strength = _quality_float(quality_signals, "evidence_strength")
    base_score = _candidate_score(candidate)
    dream_score = round(
        stability * 0.4
        + reuse_value * 0.4
        + evidence_strength * 0.1
        + base_score * 0.1,
        3,
    )

    reasons: list[str] = []
    penalties: list[str] = []
    if stability >= 0.7:
        reasons.append("high stability")
    if reuse_value >= 0.7:
        reasons.append("high reuse value")
    if evidence_strength >= 0.5:
        reasons.append("strong evidence")
    if conflicts:
        reasons.append("conflicts with existing memory")
    similarity = _quality_float(quality_signals, "similarity")
    if quality_signals.get("matched_memory_id") and similarity > 0:
        reasons.append("similar existing memory")

    one_off = bool(quality_signals.get("one_off_task"))
    duplicate = bool(quality_signals.get("duplicate"))
    if duplicate:
        penalties.append("duplicate")
    if one_off:
        penalties.append("one-off task")
        dream_score = min(dream_score, max(0.0, review_threshold - 0.01))
    if require_evidence and evidence_strength <= 0:
        penalties.append("missing evidence")

    if duplicate:
        suggested_action = duplicate_action
        decision_reason = f"candidate already exists in memory {quality_signals.get('matched_memory_id') or 'unknown'}"
    elif reject_one_off and one_off:
        suggested_action = "reject"
        decision_reason = "one-off task is not durable long-term memory"
    elif require_evidence and evidence_strength <= 0:
        suggested_action = "needs_more_evidence"
        decision_reason = "candidate needs evidence before promotion"
    elif (conflicts or quality_signals.get("matched_memory_id")) and dream_score >= review_threshold:
        suggested_action = conflict_action
        decision_reason = f"candidate overlaps existing memory {quality_signals.get('matched_memory_id') or 'unknown'}"
    elif dream_score >= promote_threshold:
        suggested_action = "create"
        decision_reason = "new reusable memory with enough stability and reuse value"
    elif dream_score >= review_threshold:
        suggested_action = "review"
        decision_reason = "potentially useful memory that needs human judgment"
    else:
        suggested_action = "reject"
        decision_reason = "low dream score for long-term reuse"

    return {
        "dream_score": dream_score,
        "suggested_action": suggested_action,
        "reasons": reasons,
        "penalties": penalties,
        "matched_memory_id": quality_signals.get("matched_memory_id"),
        "decision_reason": decision_reason,
        "policy": {
            "promote_threshold": promote_threshold,
            "review_threshold": review_threshold,
            "reject_one_off": reject_one_off,
            "require_evidence": require_evidence,
            "duplicate_action": duplicate_action,
            "conflict_promote_action": conflict_action,
        },
    }


def _signals_with_conflict_match(candidate: dict[str, Any], quality_signals: dict[str, Any], conflicts: list[dict[str, Any]]) -> dict[str, Any]:
    if not conflicts or quality_signals.get("duplicate"):
        return quality_signals
    signals = dict(quality_signals)
    candidate_content = str(candidate.get("content") or "")
    selected = max(conflicts, key=lambda item: _text_similarity(str(item.get("summary") or ""), candidate_content))
    signals["matched_memory_id"] = selected.get("memory_id")
    signals["similarity"] = round(_text_similarity(str(selected.get("summary") or ""), candidate_content), 3)
    return signals


def _status_from_dream_action(action: str) -> str:
    if action in {"create", "merge"}:
        return "promote"
    if action in {"review", "needs_more_evidence"}:
        return "review"
    return "reject"


def apply_dream_analysis_to_candidates(candidates: list[dict[str, Any]], memory_cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conflict_map = detect_candidate_conflicts(candidates, memory_cards)
    analyzed_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        conflicts = conflict_map.get(str(candidate["id"]), [])
        quality_signals = _signals_with_conflict_match(candidate, explain_candidate_quality(candidate, memory_cards), conflicts)
        dream_analysis = analyze_dream_candidate(
            candidate,
            quality_signals=quality_signals,
            conflicts=conflicts,
        )
        analyzed = dict(candidate)
        analyzed["quality_signals"] = quality_signals
        analyzed["dream_analysis"] = dream_analysis
        analyzed["status"] = _status_from_dream_action(str(dream_analysis["suggested_action"]))
        analyzed_candidates.append(analyzed)
    return analyzed_candidates


def build_review_queue(candidates: list[dict[str, Any]], memory_cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conflict_map = detect_candidate_conflicts(candidates, memory_cards)
    queue: list[dict[str, Any]] = []
    for candidate in candidates:
        conflicts = conflict_map.get(str(candidate["id"]), [])
        quality_signals = _signals_with_conflict_match(candidate, explain_candidate_quality(candidate, memory_cards), conflicts)
        dream_analysis = analyze_dream_candidate(
            candidate,
            quality_signals=quality_signals,
            conflicts=conflicts,
        )
        queue.append(build_review_queue_item(
            candidate=candidate,
            conflicts=conflicts,
            suggested_action=str(dream_analysis["suggested_action"]),
            quality_signals=quality_signals,
            dream_analysis=dream_analysis,
        ))
    return queue


def render_memory_markdown(cards: list[dict[str, Any]]) -> str:
    lines = ["# MEMORY.md", "", "## Approved Memory", ""]
    for card in sorted(cards, key=lambda item: (str(item.get("scope")), str(item.get("project") or ""), str(item.get("summary")))):
        if card.get("status") != "active":
            continue
        prefix = str(card.get("scope"))
        if card.get("project"):
            prefix = f"{prefix}: {card['project']}"
        evidence = ", ".join(str(ref) for ref in card.get("evidence_refs", []))
        suffix = f" _(Evidence: {evidence})_" if evidence else ""
        lines.append(f"- **{prefix} / {card['memory_type']}**: {card['summary']}{suffix}")
    return "\n".join(lines) + "\n"


def _memory_update_from_web_review(review: dict[str, Any]) -> dict[str, Any] | None:
    action = str(review.get("action") or review.get("status") or "")
    if action not in {"approved", "edited_and_approved", "merged"}:
        return None
    candidate = review.get("candidate") if isinstance(review.get("candidate"), dict) else {}
    summary = str(review.get("edited_content") or candidate.get("content") or "").strip()
    if not summary:
        return None
    scope = str(candidate.get("scope") or "global")
    project = normalize_project_path(str(candidate.get("project"))) if candidate.get("project") else None
    memory_type = str(candidate.get("type") or candidate.get("memory_type") or "memory")
    evidence_refs = _evidence_refs_from_candidate(candidate)
    memory_id = str(review.get("memory_id") or _candidate_id(normalize_memory_text(summary), scope, str(project) if project else None))
    approved_at = str(review.get("reviewed_at") or datetime.now(timezone.utc).isoformat())
    hints = list(candidate.get("tags", [])) if isinstance(candidate.get("tags"), list) else []
    return build_memory_card(
        memory_id=memory_id,
        scope=scope,
        project=str(project) if project else None,
        memory_type=memory_type,
        summary=summary,
        evidence_refs=evidence_refs,
        approved_by=str(review.get("reviewer") or "user"),
        approved_at=approved_at,
        retrieval_hints=[str(hint) for hint in hints],
    )


def normalize_review_decision(review: dict[str, Any]) -> dict[str, Any]:
    status = str(review.get("status") or review.get("action") or "pending")
    if "memory_updates" in review and "status" in review:
        return dict(review)
    update = _memory_update_from_web_review(review)
    updates = [update] if update else []
    return build_review_decision(
        candidate_id=str(review.get("candidate_id") or review.get("candidate", {}).get("id") or "candidate"),
        status=status,
        reviewer=str(review.get("reviewer") or "user"),
        notes=str(review.get("notes") or review.get("note") or ""),
        memory_updates=updates,
    )


def apply_reviewed_memory(
    reviewed: list[dict[str, Any]],
    existing_cards: list[dict[str, Any]],
    *,
    return_decisions: bool = False,
) -> tuple[list[dict[str, Any]], str] | tuple[list[dict[str, Any]], str, list[dict[str, Any]]]:
    cards_by_id = {str(card["id"]): dict(card) for card in existing_cards}
    decisions: list[dict[str, Any]] = []
    for raw_decision in reviewed:
        decision = normalize_review_decision(raw_decision)
        decisions.append(decision)
        if decision.get("status") not in {"approved", "edited_and_approved", "merged"}:
            continue
        for superseded in raw_decision.get("supersedes", []):
            if str(superseded) in cards_by_id:
                cards_by_id[str(superseded)]["status"] = "superseded"
                cards_by_id[str(superseded)]["superseded_at"] = str(decision.get("reviewed_at") or datetime.now(timezone.utc).isoformat())
        for update in decision.get("memory_updates", []):
            if not isinstance(update, dict) or "id" not in update:
                continue
            cards_by_id[str(update["id"])] = dict(update)
    cards = list(cards_by_id.values())
    markdown = render_memory_markdown(cards)
    if return_decisions:
        return cards, markdown, decisions
    return cards, markdown


def build_agent_context(memory_cards: list[dict[str, Any]], *, project: str | None, limit: int = 12, task: str | None = None) -> dict[str, Any]:
    normalized_project = normalize_project_path(project)
    task_tokens = _tokens(task or "")

    def searchable_text(card: dict[str, Any]) -> str:
        return " ".join(
            str(value)
            for value in [
                card.get("summary"),
                card.get("memory_type"),
                " ".join(str(item) for item in card.get("retrieval_hints", []) if item),
                " ".join(str(item) for item in card.get("tags", []) if item),
            ]
        )

    def relevance_parts(card: dict[str, Any]) -> dict[str, Any]:
        searchable = searchable_text(card)
        card_tokens = _tokens(searchable)
        matched_tokens = sorted(task_tokens & card_tokens)
        token_score = len(matched_tokens) / len(task_tokens) if task_tokens and card_tokens else 0.0
        intent_score = _intent_relevance_boost(task, searchable) if task_tokens else 0.0
        score = max(token_score, intent_score)
        if not task_tokens:
            reason = "default_scope_order"
        elif intent_score > token_score:
            reason = "intent_alias_match"
        elif token_score > 0:
            reason = "token_overlap"
        else:
            reason = "scope_fallback"
        return {
            "relevance": round(score, 4),
            "token_score": round(token_score, 4),
            "intent_score": round(intent_score, 4),
            "matched_tokens": matched_tokens,
            "reason": reason,
        }

    def relevance(card: dict[str, Any]) -> float:
        if not task_tokens:
            return 0.0
        return float(relevance_parts(card)["relevance"])

    def scope_rank(card: dict[str, Any]) -> int:
        card_project = normalize_project_path(str(card.get("project"))) if card.get("project") else None
        if normalized_project and card.get("scope") == "project" and card_project == normalized_project:
            return 0
        if card.get("scope") == "user":
            return 1
        if card.get("scope") == "global":
            return 2
        if card.get("scope") == "session":
            return 3
        return 4

    def rank(card: dict[str, Any]) -> tuple[float, int, int, str]:
        card_relevance = relevance(card)
        if not task_tokens:
            default_rank = 0
        elif card_relevance > 0:
            default_rank = 0
        elif card.get("scope") in {"user", "global"}:
            default_rank = 1
        else:
            default_rank = 2
        return (-card_relevance, default_rank, scope_rank(card), str(card.get("summary")))

    filtered = []
    for card in memory_cards:
        if card.get("status") != "active":
            continue
        if card.get("scope") == "project":
            card_project = normalize_project_path(str(card.get("project"))) if card.get("project") else None
            if not normalized_project or card_project != normalized_project:
                continue
            card = dict(card)
            card["project"] = card_project
        filtered.append(card)
    ranked = sorted(filtered, key=rank)[:limit]
    diagnostics = []
    for index, card in enumerate(ranked, start=1):
        diagnostics.append({
            "rank": index,
            "id": card.get("id"),
            "scope": card.get("scope"),
            "memory_type": card.get("memory_type"),
            "scope_rank": scope_rank(card),
            **relevance_parts(card),
        })
    payload = {"project": normalized_project, "count": len(ranked), "items": ranked, "diagnostics": diagnostics}
    if task:
        payload["task"] = task
    return payload


def render_context_markdown(context: dict[str, Any]) -> str:
    lines = ["## Relevant Memory", ""]
    diagnostics_by_id = {str(item.get("id")): item for item in context.get("diagnostics", []) if item.get("id")}
    for item in context.get("items", []):
        prefix = str(item.get("scope"))
        if item.get("project"):
            prefix = f"{prefix}: {item['project']}"
        summary = str(item.get("summary") or item.get("content") or "")
        diag = diagnostics_by_id.get(str(item.get("id")))
        suffix = ""
        if diag:
            suffix = f" _(rank_reason={diag.get('reason')}, relevance={diag.get('relevance')})_"
        lines.append(f"- **{prefix} / {item.get('memory_type')}**: {summary}{suffix}")
    return "\n".join(lines) + "\n"


def _analysis_for_report(candidate: dict[str, Any]) -> dict[str, Any]:
    if isinstance(candidate.get("dream_analysis"), dict):
        return dict(candidate["dream_analysis"])
    quality_signals = explain_candidate_quality(candidate, [])
    return analyze_dream_candidate(candidate, quality_signals=quality_signals, conflicts=[])


def _action_heading(action: str) -> str:
    return {
        "create": "Create",
        "merge": "Merge",
        "needs_more_evidence": "Needs More Evidence",
        "review": "Review",
        "reject": "Reject",
    }.get(action, action.replace("_", " ").title())


def _candidate_report_line(candidate: dict[str, Any], analysis: dict[str, Any]) -> str:
    reasons = ", ".join(str(item) for item in analysis.get("reasons", [])) or "none"
    penalties = ", ".join(str(item) for item in analysis.get("penalties", [])) or "none"
    return (
        f"- ({candidate.get('type')}, dream_score={analysis.get('dream_score')}, "
        f"action={analysis.get('suggested_action')}) {candidate.get('content')} "
        f"[reasons: {reasons}; penalties: {penalties}]"
    )


def _count_by_key(items: list[dict[str, Any]], key: str, *, fallback: str = "unknown") -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = str(item.get(key) or fallback)
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _format_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{key}={value}" for key, value in counts.items())


def _render_dreams(events: list[dict[str, Any]], candidates: list[dict[str, Any]], *, facts: list[dict[str, Any]] | None = None) -> str:
    analyzed: list[tuple[dict[str, Any], dict[str, Any]]] = [
        (candidate, _analysis_for_report(candidate))
        for candidate in candidates
    ]
    counts = {action: 0 for action in ACTION_ORDER}
    for _, analysis in analyzed:
        action = str(analysis.get("suggested_action") or "review")
        counts[action] = counts.get(action, 0) + 1
    policy = DEFAULT_DREAM_PROMOTION_POLICY
    lines = [
        "# DREAMS.md",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Sweep Summary",
        "",
        f"- Events scanned: {len(events)}",
        f"- Candidates: {len(candidates)}",
        "",
        "## Fact Diagnostics",
        "",
        f"- Facts extracted: {len(facts or [])}",
        f"- Facts by type: {_format_counts(_count_by_key(facts or [], 'fact_type'))}",
        f"- Candidates by type: {_format_counts(_count_by_key(candidates, 'type'))}",
        "",
        "## Evidence Quality",
        "",
        f"- Quality tiers: {_format_counts(_count_by_key([candidate.get('quality_signals', {}) if isinstance(candidate.get('quality_signals'), dict) else {} for candidate in candidates], 'evidence_quality'))}",
        "",
        "## Promotion Policy",
        "",
        f"- Promote threshold: {policy['promote_threshold']}",
        f"- Review threshold: {policy['review_threshold']}",
        f"- Reject one-off tasks: {str(policy['reject_one_off']).lower()}",
        f"- Require evidence: {str(policy['require_evidence']).lower()}",
        "",
        "## Action Summary",
        "",
    ]
    for action in ACTION_ORDER:
        lines.append(f"- {_action_heading(action)}: {counts.get(action, 0)}")

    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {action: [] for action in ACTION_ORDER}
    for candidate, analysis in analyzed:
        action = str(analysis.get("suggested_action") or "review")
        grouped.setdefault(action, []).append((candidate, analysis))

    for action in ACTION_ORDER:
        lines.extend(["", f"## {_action_heading(action)}", ""])
        rows = sorted(
            grouped.get(action, []),
            key=lambda item: (
                -float(item[1].get("dream_score", 0.0) or 0.0),
                str(item[0].get("type") or ""),
                str(item[0].get("content") or ""),
            ),
        )
        if not rows:
            lines.append("- None")
            continue
        for candidate, analysis in rows[:30]:
            lines.append(_candidate_report_line(candidate, analysis))
    return "\n".join(lines) + "\n"


def _render_memory_preview(candidates: list[dict[str, Any]]) -> str:
    promoted = [c for c in candidates if c["status"] == "promote"]
    review = [c for c in candidates if c["status"] == "review"]
    lines = ["# MEMORY.preview.md", "", "## Proposed Long-Term Memory", ""]
    for c in promoted:
        prefix = "Global" if c.get("scope") == "global" else f"Project: {c.get('project')}"
        lines.append(f"- **{prefix} / {c['type']}**: {c['content']}")
    if review:
        lines.extend(["", "## Needs Review", ""])
        for c in review:
            prefix = "Global" if c.get("scope") == "global" else f"Project: {c.get('project')}"
            lines.append(f"- **{prefix} / {c['type']} / {c['score']}**: {c['content']}")
    return "\n".join(lines) + "\n"


def render_review_queue_memory_preview(queue: list[dict[str, Any]]) -> str:
    candidates: list[dict[str, Any]] = []
    for item in queue:
        candidate = item.get("candidate") if isinstance(item.get("candidate"), dict) else {}
        if not candidate:
            continue
        normalized = dict(candidate)
        action = str(item.get("suggested_action") or "")
        normalized["status"] = _status_from_dream_action(action)
        if isinstance(item.get("dream_analysis"), dict):
            normalized["dream_analysis"] = dict(item["dream_analysis"])
        if isinstance(item.get("quality_signals"), dict):
            normalized["quality_signals"] = dict(item["quality_signals"])
        candidates.append(normalized)
    return _render_memory_preview(candidates)


def dream_from_events(
    events: list[dict[str, Any]],
    *,
    project: str | None,
    output_dir: Path | str,
    apply: bool = False,
    agent_candidates: list[dict[str, Any]] | None = None,
    agent_mode: bool = False,
) -> DreamResult:
    output = Path(output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    facts = extract_atomic_facts(events, project=project)
    write_jsonl_records(facts, output / "facts.jsonl")

    if agent_candidates is not None:
        raw_candidates = []
        for candidate in agent_candidates:
            normalized = dict(candidate)
            normalized.setdefault("id", _candidate_id(str(normalized.get("content", "")), str(normalized.get("scope", "global")), normalized.get("project")))
            normalized.setdefault("tags", [])
            normalized.setdefault("evidence", [])
            if "score" not in normalized:
                confidence = float(normalized.get("confidence", 0.5) or 0.5)
                decision = normalized.get("decision", "review")
                normalized["score"] = round(confidence, 3)
                normalized["status"] = "promote" if decision == "promote" else "reject" if decision == "reject" else "review"
            raw_candidates.append(normalized)
        candidates = raw_candidates
    else:
        candidates = build_candidates_from_facts(facts)
    candidates = apply_dream_analysis_to_candidates(candidates, [])
    candidates.sort(key=lambda item: (-item["score"], item["type"], item["content"]))

    candidates_path = output / ("ai-candidates.jsonl" if agent_mode else "candidates.jsonl")
    write_jsonl_records(candidates, candidates_path)

    dreams_path = output / "DREAMS.md"
    dreams_path.write_text(_render_dreams(events, candidates, facts=facts), encoding="utf-8")

    preview_path = output / "MEMORY.preview.md"
    preview_text = _render_memory_preview(candidates)
    preview_path.write_text(preview_text, encoding="utf-8")

    memory_path: Path | None = None
    if apply:
        memory_path = output / "MEMORY.md"
        existing = memory_path.read_text(encoding="utf-8") if memory_path.exists() else "# MEMORY.md\n\n"
        memory_path.write_text(existing.rstrip() + "\n\n" + preview_text, encoding="utf-8")

    promoted = len([c for c in candidates if c["status"] == "promote"])
    review = len([c for c in candidates if c["status"] == "review"])
    rejected = len([c for c in candidates if c["status"] == "reject"])
    return DreamResult(
        event_count=len(events),
        candidate_count=len(candidates),
        promoted_count=promoted,
        review_count=review,
        rejected_count=rejected,
        output_dir=str(output),
        candidates_path=str(candidates_path),
        dreams_path=str(dreams_path),
        memory_preview_path=str(preview_path),
        memory_path=str(memory_path) if memory_path else None,
        applied=apply,
    )
