from __future__ import annotations

import json
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .memory_cli import _resume_run, _run_dream_to_review
from .memory_config import DEFAULT_MEMORY_CONFIG
from .memory_dreaming import normalize_review_decision
from .memory_runs import append_trace, create_run_state, list_runs, load_run_state, read_trace, update_run_state


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
    api_key_env: str | None = None
    base_url: str | None = None
    timeout_seconds: int | None = None
    invoke_model: bool | None = None
    memory_cards: str | None = None


HOME_HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Dream Memory</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f6f7fb; color: #172033; }
    header { background: #172033; color: white; padding: 24px 32px; }
    main { max-width: 960px; margin: 24px auto; padding: 0 20px; display: grid; gap: 20px; }
    section { background: white; border: 1px solid #e4e7ef; border-radius: 14px; padding: 20px; box-shadow: 0 8px 24px rgba(20, 30, 55, .06); }
    a { color: #2563eb; font-weight: 700; }
    code { background: #eef2ff; padding: 2px 6px; border-radius: 6px; }
  </style>
</head>
<body>
  <header>
    <h1>Dream Memory</h1>
    <p>本地 agent 记忆控制台：导入会话、审核候选记忆、生成任务上下文。</p>
  </header>
  <main>
    <section>
      <h2>Dream Memory Workflow</h2>
      <p>使用 <code>dream-memory</code> CLI 运行 scan/import/dream/run/status/resume/trace。</p>
      <p><a href="/memory-review">打开候选记忆审核界面</a></p>
    </section>
  </main>
</body>
</html>
"""

MEMORY_REVIEW_HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Dream Memory Review</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f6f7fb; color: #172033; }
    header { background: #172033; color: white; padding: 22px 28px; }
    main { max-width: 1180px; margin: 22px auto; padding: 0 18px; display: grid; grid-template-columns: 320px 1fr; gap: 18px; }
    section, aside { background: white; border: 1px solid #e4e7ef; border-radius: 14px; padding: 16px; box-shadow: 0 8px 24px rgba(20, 30, 55, .06); }
    input, select, textarea { width: 100%; box-sizing: border-box; padding: 9px 10px; border: 1px solid #cfd6e6; border-radius: 9px; font: inherit; margin: 6px 0 10px; }
    textarea { min-height: 110px; }
    button { margin: 5px 6px 5px 0; padding: 9px 12px; border: 0; border-radius: 9px; background: #2563eb; color: white; font-weight: 700; cursor: pointer; }
    button.reject { background: #b42318; }
    button.more { background: #936316; }
    .item { padding: 10px; border: 1px solid #e4e7ef; border-radius: 10px; margin-bottom: 8px; cursor: pointer; }
    .item.active { outline: 2px solid #2563eb; }
    .muted { color: #667085; }
    .pill { display: inline-block; padding: 2px 7px; border-radius: 999px; background: #eef2ff; color: #3730a3; font-size: 12px; margin-right: 4px; }
    .group-title { margin: 12px 0 6px; font-size: 13px; font-weight: 800; color: #344054; text-transform: uppercase; }
    .progress { display: grid; gap: 6px; font-size: 13px; }
    .progress strong { color: #172033; }
    pre { white-space: pre-wrap; background: #0f172a; color: #e5e7eb; padding: 12px; border-radius: 10px; }
  </style>
</head>
<body>
<header><h1>Dream Memory Review</h1><p>候选记忆人工审核：批准、拒绝、合并或要求更多证据。</p></header>
<main>
  <aside>
    <h2>运行状态</h2>
    <div id="runs" class="muted">加载运行状态...</div>
    <h2>审核进度</h2>
    <div id="reviewProgress" class="muted">选择 run 后显示进度</div>
    <h2>候选分组</h2>
    <input id="search" placeholder="搜索内容/标签" />
    <select id="statusFilter"><option value="">全部状态</option><option value="promote">promote</option><option value="review">review</option><option value="reject">reject</option></select>
    <select id="scopeFilter"><option value="">全部范围</option><option value="user">user</option><option value="global">global</option><option value="project">project</option></select>
    <div id="list" class="muted">加载中...</div>
  </aside>
  <section>
    <h2 id="title">选择一个候选项</h2>
    <p class="muted" id="meta"></p>
    <textarea id="content" placeholder="候选内容，可编辑后批准"></textarea>
    <textarea id="note" placeholder="审核备注"></textarea>
    <div>
      <button onclick="submitReview('approved')">批准</button>
      <button onclick="submitReview('edited_and_approved')">编辑后批准</button>
      <button onclick="submitReview('merged')">合并</button>
      <button class="more" onclick="submitReview('needs_more_evidence')">需要更多证据</button>
      <button class="reject" onclick="submitReview('rejected')">拒绝</button>
      <button onclick="resumeSelectedRun()">恢复并应用 Run</button>
    </div>
    <h3>Evidence</h3>
    <pre id="evidence"></pre>
    <h3>Run Trace</h3>
    <pre id="trace"></pre>
    <p id="status" class="muted"></p>
  </section>
</main>
<script>
let candidates = [];
let selected = null;
let selectedRunId = null;
async function loadRuns() {
  const res = await fetch('/api/memory/runs');
  const data = await res.json();
  const runs = data.runs || [];
  const box = document.getElementById('runs');
  box.innerHTML = runs.slice(0, 8).map(r => `<div class="item" onclick="selectRun('${r.run_id}')"><b>${escapeHtml(r.run_id)}</b><br><span class="pill">${escapeHtml(r.status || '')}</span><span class="pill">${escapeHtml(r.phase || '')}</span><br>${escapeHtml(r.updated_at || '')}</div>`).join('') || '暂无 run';
}
async function selectRun(runId) {
  selectedRunId = runId;
  const res = await fetch(`/api/memory/runs/${runId}/candidates`);
  const data = await res.json();
  candidates = data.candidates || [];
  document.getElementById('status').textContent = `当前 run: ${runId}`;
  await loadReviewProgress();
  await loadTrace();
  renderList();
}
async function loadReviewProgress() {
  if (!selectedRunId) return;
  const res = await fetch(`/api/memory/runs/${selectedRunId}/review-progress`);
  const data = await res.json();
  const actions = data.actions || {};
  document.getElementById('reviewProgress').innerHTML = `<div class="progress"><div><strong>${data.reviewed || 0}</strong> / ${data.total || 0} reviewed</div><div>Pending: <strong>${data.pending || 0}</strong></div><div>Approved: ${actions.approved || 0} · Rejected: ${actions.rejected || 0} · Needs evidence: ${actions.needs_more_evidence || 0}</div></div>`;
}
async function loadTrace() {
  if (!selectedRunId) return;
  const res = await fetch(`/api/memory/runs/${selectedRunId}/trace`);
  const data = await res.json();
  document.getElementById('trace').textContent = JSON.stringify(data.trace || [], null, 2);
}
async function loadCandidates() {
  const res = await fetch('/api/memory/candidates');
  const data = await res.json();
  candidates = data.candidates || [];
  renderList();
}
function groupCandidates(items) {
  return items.reduce((groups, candidate) => {
    const key = candidate.status || 'unknown';
    if (!groups[key]) groups[key] = [];
    groups[key].push(candidate);
    return groups;
  }, {});
}
function candidateHtml(c) {
  const conflictCount = (c.conflicts || []).length;
  const conflict = conflictCount ? `<span class="pill">conflicts ${conflictCount}</span>` : '';
  return `<div class="item ${selected && selected.id === c.id ? 'active' : ''}" onclick="selectCandidate('${c.id}')"><b>${escapeHtml(c.type || '')}</b><br>${escapeHtml((c.content || '').slice(0, 120))}<br><span class="pill">${escapeHtml(c.scope || '')}</span><span class="pill">${escapeHtml(String(c.score || ''))}</span>${conflict}</div>`;
}
function renderList() {
  const q = document.getElementById('search').value.toLowerCase();
  const sf = document.getElementById('statusFilter').value;
  const scf = document.getElementById('scopeFilter').value;
  const list = document.getElementById('list');
  const filtered = candidates.filter(c => (!sf || c.status === sf) && (!scf || c.scope === scf) && (!q || JSON.stringify(c).toLowerCase().includes(q)));
  const groups = groupCandidates(filtered);
  const order = ['promote', 'review', 'reject', 'unknown'];
  const html = order.filter(key => groups[key] && groups[key].length).map(key => `<div class="group-title">${escapeHtml(key)} (${groups[key].length})</div>` + groups[key].map(candidateHtml).join('')).join('');
  list.innerHTML = html || '没有候选项';
}
function selectCandidate(id) {
  selected = candidates.find(c => c.id === id);
  document.getElementById('title').textContent = selected.type + ' / ' + selected.id;
  document.getElementById('meta').textContent = `${selected.scope || ''} ${selected.project || ''} ${selected.status || ''}`;
  document.getElementById('content').value = selected.content || '';
  document.getElementById('evidence').textContent = JSON.stringify(selected.evidence || [], null, 2);
  renderList();
}
async function submitReview(action) {
  if (!selected) return;
  const payload = { candidate_id: selected.id, action, edited_content: document.getElementById('content').value, note: document.getElementById('note').value, reviewer: 'user', candidate: selected };
  const url = selectedRunId ? `/api/memory/runs/${selectedRunId}/review` : '/api/memory/review';
  const res = await fetch(url, {method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(payload)});
  const data = await res.json();
  document.getElementById('status').textContent = res.ok ? '已保存审核结果' : JSON.stringify(data);
  await loadTrace();
}
async function resumeSelectedRun() {
  if (!selectedRunId) return;
  const res = await fetch(`/api/memory/runs/${selectedRunId}/resume`, {method: 'POST'});
  const data = await res.json();
  document.getElementById('status').textContent = res.ok ? `已恢复并应用: ${data.status}` : JSON.stringify(data);
  await loadRuns();
  await loadTrace();
}
function escapeHtml(s) { return String(s).replace(/[&<>"]/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[ch])); }
['search','statusFilter','scopeFilter'].forEach(id => document.addEventListener('input', e => { if (e.target && e.target.id === id) renderList(); }));
loadRuns();
loadCandidates();
setInterval(loadRuns, 3000);
</script>
</body>
</html>
"""


def _web_config(memory_dir: Path) -> dict[str, Any]:
    config = dict(DEFAULT_MEMORY_CONFIG)
    config["output_dir"] = str(memory_dir)
    config["memory_cards"] = str(memory_dir / "memory_cards.jsonl")
    config["imports_output_dir"] = str(memory_dir / "imports")
    return config


def _run_namespace(request: MemoryRunStartRequest, memory_dir: Path) -> Namespace:
    return Namespace(
        input=request.input,
        project=request.project,
        output_dir=str(memory_dir),
        memory_cards=request.memory_cards,
        mode=request.mode,
        provider=request.provider,
        model=request.model,
        api_key_env=request.api_key_env,
        base_url=request.base_url,
        timeout_seconds=request.timeout_seconds,
        invoke_model=request.invoke_model,
    )


def _resume_namespace(run_id: str, reviewed: str | None, memory_cards: str | None, memory_dir: Path) -> Namespace:
    return Namespace(run_id=run_id, reviewed=reviewed, memory_cards=memory_cards, reviewer="user", output_dir=str(memory_dir))


def _review_progress(state: dict[str, Any]) -> dict[str, Any]:
    candidates_path = Path(str(state.get("artifacts", {}).get("candidates_path") or ""))
    candidates = _read_jsonl_dicts(candidates_path) if candidates_path.is_file() else []
    reviewed_path = Path(str(state["run_dir"])) / "reviewed.jsonl"
    reviewed = _read_jsonl_dicts(reviewed_path)
    reviewed_ids = {str(row.get("candidate_id")) for row in reviewed if row.get("candidate_id")}
    actions: dict[str, int] = {}
    for row in reviewed:
        action = str(row.get("action") or row.get("status") or "unknown")
        actions[action] = actions.get(action, 0) + 1
    return {
        "run_id": state["run_id"],
        "total": len(candidates),
        "reviewed": len(reviewed_ids),
        "pending": max(0, len(candidates) - len(reviewed_ids)),
        "actions": actions,
    }


def _read_jsonl_dicts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def create_app(default_output_dir: Path | str = "outputs/runs", default_memory_dir: Path | str = ".dream-memory") -> FastAPI:
    memory_dir = Path(default_memory_dir).expanduser()
    app = FastAPI(title="Dream Memory", version="0.1.0")

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return HOME_HTML

    @app.get("/memory-review", response_class=HTMLResponse)
    def memory_review() -> str:
        return MEMORY_REVIEW_HTML

    @app.post("/api/memory/runs/start")
    def memory_run_start(request: MemoryRunStartRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
        config = _web_config(memory_dir)
        args = _run_namespace(request, memory_dir)
        mode = str(request.mode or config["mode"])
        provider = request.provider or config.get("provider")
        configured_model = str(request.model or config["model"])
        model = f"{provider}:{configured_model}" if provider and ":" not in configured_model else configured_model
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
            _run_dream_to_review(args=args, config=config, persistent=True, existing_state=state)

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
        candidates = _read_jsonl_dicts(candidates_path) if candidates_path.is_file() else []
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
        candidates = _read_jsonl_dicts(candidates_path)
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
