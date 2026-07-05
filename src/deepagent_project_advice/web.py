from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from .analyzer import build_project_analysis
from .cli import build_patch_proposal, create_run_artifacts, parse_verify_commands
from .tools import create_patch, run_verification, summarize_changes
from .storage import RunStore
from .memory_dreaming import normalize_review_decision


class AnalyzeRequest(BaseModel):
    project: str
    task: str


class RunRequest(BaseModel):
    project: str
    task: str
    patch: bool = False
    verify: bool = False
    verify_commands: list[str] = Field(default_factory=list)


class MemoryReviewRequest(BaseModel):
    candidate_id: str
    action: str
    edited_content: str | None = None
    reviewer: str = "user"
    note: str | None = None
    candidate: dict[str, Any] = Field(default_factory=dict)


HOME_HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>DeepAgent Memory</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f6f7fb; color: #172033; }
    header { background: #172033; color: white; padding: 24px 32px; }
    main { max-width: 1120px; margin: 24px auto; padding: 0 20px; display: grid; gap: 20px; }
    section { background: white; border: 1px solid #e4e7ef; border-radius: 14px; padding: 20px; box-shadow: 0 8px 24px rgba(20, 30, 55, .06); }
    label { display: block; font-weight: 700; margin-top: 12px; }
    input, textarea { width: 100%; box-sizing: border-box; padding: 10px 12px; border: 1px solid #cfd6e6; border-radius: 10px; font: inherit; }
    textarea { min-height: 90px; }
    button { margin-top: 14px; padding: 10px 16px; border: 0; border-radius: 10px; background: #2563eb; color: white; font-weight: 700; cursor: pointer; }
    pre { white-space: pre-wrap; background: #0f172a; color: #e5e7eb; padding: 16px; border-radius: 12px; overflow: auto; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; }
    .muted { color: #667085; }
  </style>
</head>
<body>
  <header>
    <h1>DeepAgent Memory</h1>
    <p>本地 agent 记忆控制台：导入会话、审核候选记忆、生成任务上下文。</p>
  </header>
  <main>
    <section>
      <h2>任务输入区</h2>
      <label>项目路径</label>
      <input id="project" value="." />
      <label>需求</label>
      <textarea id="task">增加登录页面</textarea>
      <label><input id="patch" type="checkbox" /> 生成文件改动 Diff</label>
      <label><input id="verify" type="checkbox" /> 运行测试结果验证</label>
      <label>验证命令（每行一条，可选）</label>
      <textarea id="commands" placeholder="python -m unittest discover -s tests"></textarea>
      <button onclick="runTask()">运行任务</button>
      <p class="muted" id="status"></p>
    </section>
    <div class="grid">
      <section><h2>执行计划展示</h2><pre id="plan">等待运行...</pre></section>
      <section><h2>文件改动 Diff</h2><pre id="diff">启用 Patch 后显示。</pre></section>
      <section><h2>测试结果</h2><pre id="verification">启用验证后显示。</pre></section>
      <section><h2>最终报告</h2><pre id="report">运行产物路径会显示在这里。</pre></section>
    </div>
  </main>
<script>
async function runTask() {
  const status = document.getElementById('status');
  status.textContent = '运行中...';
  const commands = document.getElementById('commands').value.split('\n').map(s => s.trim()).filter(Boolean);
  const body = {
    project: document.getElementById('project').value,
    task: document.getElementById('task').value,
    patch: document.getElementById('patch').checked,
    verify: document.getElementById('verify').checked,
    verify_commands: commands,
  };
  const res = await fetch('/api/run', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body) });
  const payload = await res.json();
  if (!res.ok) { status.textContent = payload.detail || '运行失败'; return; }
  document.getElementById('plan').textContent = payload.report || '';
  document.getElementById('diff').textContent = payload.patch || '未生成 patch';
  document.getElementById('verification').textContent = payload.verification ? JSON.stringify(payload.verification, null, 2) : '未运行验证';
  document.getElementById('report').textContent = '运行目录：' + payload.run_dir + '\n\nmetadata.json 已保存结构化结果。';
  status.textContent = '完成';
}
</script>
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
    :root {
      --bg: #0b1020;
      --panel: rgba(255,255,255,.08);
      --panel-2: rgba(255,255,255,.12);
      --text: #eef2ff;
      --muted: #aab4d4;
      --line: rgba(255,255,255,.14);
      --blue: #7c9cff;
      --green: #35d399;
      --red: #ff6b7a;
      --amber: #ffd166;
      --violet: #b794ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at 10% 10%, rgba(124,156,255,.28), transparent 28%),
        radial-gradient(circle at 90% 20%, rgba(183,148,255,.20), transparent 30%),
        radial-gradient(circle at 50% 100%, rgba(53,211,153,.14), transparent 26%),
        var(--bg);
    }
    header { padding: 28px 34px 18px; border-bottom: 1px solid var(--line); backdrop-filter: blur(20px); position: sticky; top: 0; z-index: 3; background: rgba(11,16,32,.76); }
    .top { display: flex; gap: 18px; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; }
    h1 { margin: 0; font-size: 30px; letter-spacing: -.04em; }
    .subtitle { color: var(--muted); margin-top: 8px; max-width: 780px; line-height: 1.55; }
    .pill { display: inline-flex; gap: 8px; align-items: center; padding: 8px 11px; border: 1px solid var(--line); border-radius: 999px; background: rgba(255,255,255,.06); color: var(--muted); font-size: 13px; }
    main { padding: 24px 34px 40px; display: grid; grid-template-columns: 360px minmax(0, 1fr); gap: 22px; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 22px; box-shadow: 0 20px 70px rgba(0,0,0,.32); backdrop-filter: blur(20px); overflow: hidden; }
    .toolbar { padding: 16px; display: grid; gap: 12px; border-bottom: 1px solid var(--line); }
    input, select, textarea { width: 100%; border: 1px solid var(--line); color: var(--text); background: rgba(255,255,255,.08); border-radius: 14px; padding: 11px 12px; font: inherit; outline: none; }
    textarea { min-height: 160px; resize: vertical; line-height: 1.55; }
    input::placeholder, textarea::placeholder { color: rgba(238,242,255,.45); }
    .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
    .stat { padding: 12px; border-radius: 16px; background: rgba(255,255,255,.07); border: 1px solid var(--line); }
    .stat strong { display: block; font-size: 20px; }
    .stat span { color: var(--muted); font-size: 12px; }
    .list { max-height: calc(100vh - 260px); overflow: auto; padding: 10px; }
    .card { border: 1px solid transparent; background: rgba(255,255,255,.06); border-radius: 18px; padding: 14px; margin-bottom: 10px; cursor: pointer; transition: .16s ease; }
    .card:hover, .card.active { border-color: rgba(124,156,255,.7); background: rgba(124,156,255,.13); transform: translateY(-1px); }
    .card-title { font-weight: 750; line-height: 1.4; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
    .meta { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 12px; }
    .tag { font-size: 12px; color: var(--muted); padding: 5px 8px; border: 1px solid var(--line); border-radius: 999px; background: rgba(255,255,255,.06); }
    .tag.promote { color: #c8ffeb; border-color: rgba(53,211,153,.5); }
    .tag.review { color: #fff0b8; border-color: rgba(255,209,102,.5); }
    .tag.reject { color: #ffc6cc; border-color: rgba(255,107,122,.5); }
    .detail { padding: 22px; display: grid; gap: 18px; }
    .empty { color: var(--muted); padding: 50px; text-align: center; }
    .detail-head { display: flex; justify-content: space-between; gap: 14px; align-items: flex-start; flex-wrap: wrap; }
    .detail-title { margin: 0; font-size: 24px; letter-spacing: -.03em; line-height: 1.25; }
    .score { min-width: 86px; text-align: center; padding: 12px 14px; border-radius: 18px; border: 1px solid rgba(124,156,255,.45); background: rgba(124,156,255,.13); }
    .score strong { font-size: 24px; display: block; }
    .section-title { color: var(--muted); text-transform: uppercase; font-size: 12px; letter-spacing: .12em; margin-bottom: 8px; }
    .box { border: 1px solid var(--line); background: rgba(255,255,255,.055); border-radius: 18px; padding: 14px; }
    pre { white-space: pre-wrap; margin: 0; color: #dbe5ff; font-size: 13px; line-height: 1.55; }
    .actions { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
    button { border: 0; color: white; border-radius: 14px; padding: 12px 14px; font-weight: 760; cursor: pointer; transition: .15s ease; }
    button:hover { transform: translateY(-1px); filter: brightness(1.06); }
    .approve { background: linear-gradient(135deg, #16a34a, #35d399); }
    .reject { background: linear-gradient(135deg, #dc2626, #ff6b7a); }
    .needs { background: linear-gradient(135deg, #b7791f, #ffd166); color: #211400; }
    .ghost { background: rgba(255,255,255,.10); color: var(--text); border: 1px solid var(--line); }
    .toast { position: fixed; right: 22px; bottom: 22px; padding: 14px 16px; background: rgba(10, 20, 35, .94); border: 1px solid var(--line); border-radius: 16px; color: var(--text); box-shadow: 0 18px 45px rgba(0,0,0,.35); display: none; }
    @media (max-width: 920px) { main { grid-template-columns: 1fr; padding: 18px; } .list { max-height: 460px; } header { padding: 22px 18px 14px; } }
  </style>
</head>
<body>
  <header>
    <div class="top">
      <div>
        <h1>Dream Memory Review</h1>
        <div class="subtitle">人工审核 Dream Memory 候选：只把稳定、可复用、有证据的内容提升为正式共享记忆。默认写入 reviewed.jsonl，不直接污染 MEMORY.md。</div>
      </div>
      <div class="pill" id="memoryPath">Loading...</div>
    </div>
  </header>
  <main>
    <aside class="panel">
      <div class="toolbar">
        <input id="search" placeholder="搜索候选内容 / 项目 / 标签" />
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
          <select id="statusFilter"><option value="all">全部状态</option><option value="promote">Promote</option><option value="review">Review</option><option value="reject">Reject</option></select>
          <select id="scopeFilter"><option value="all">全部作用域</option><option value="global">Global</option><option value="project">Project</option></select>
        </div>
        <div class="stats">
          <div class="stat"><strong id="totalCount">0</strong><span>候选</span></div>
          <div class="stat"><strong id="promoteCount">0</strong><span>Promote</span></div>
          <div class="stat"><strong id="reviewCount">0</strong><span>Review</span></div>
        </div>
        <button class="ghost" onclick="loadCandidates()">刷新候选</button>
      </div>
      <div class="list" id="candidateList"></div>
    </aside>
    <section class="panel detail" id="detail"><div class="empty">请选择左侧候选记忆。</div></section>
  </main>
  <div class="toast" id="toast"></div>
<script>
let allCandidates = [];
let selected = null;
function toast(msg) { const el = document.getElementById('toast'); el.textContent = msg; el.style.display = 'block'; setTimeout(() => el.style.display = 'none', 2400); }
function clsStatus(s) { return s === 'promote' ? 'promote' : s === 'reject' ? 'reject' : 'review'; }
async function loadCandidates() {
  const res = await fetch('/api/memory/candidates');
  const payload = await res.json();
  if (!res.ok) { toast(payload.detail || '加载失败'); return; }
  allCandidates = payload.candidates || [];
  document.getElementById('memoryPath').textContent = payload.memory_dir || '';
  renderList();
  if (allCandidates.length && !selected) selectCandidate(allCandidates[0].id);
}
function filteredCandidates() {
  const q = document.getElementById('search').value.toLowerCase();
  const status = document.getElementById('statusFilter').value;
  const scope = document.getElementById('scopeFilter').value;
  return allCandidates.filter(c => {
    const hay = [c.content, c.project, c.type, c.status, ...(c.tags || [])].join(' ').toLowerCase();
    return (!q || hay.includes(q)) && (status === 'all' || c.status === status) && (scope === 'all' || c.scope === scope);
  });
}
function renderList() {
  const list = document.getElementById('candidateList');
  const items = filteredCandidates();
  document.getElementById('totalCount').textContent = allCandidates.length;
  document.getElementById('promoteCount').textContent = allCandidates.filter(c => c.status === 'promote').length;
  document.getElementById('reviewCount').textContent = allCandidates.filter(c => c.status === 'review').length;
  list.innerHTML = items.map(c => `
    <div class="card ${selected && selected.id === c.id ? 'active' : ''}" onclick="selectCandidate('${c.id}')">
      <div class="card-title">${escapeHtml(c.content || '')}</div>
      <div class="meta">
        <span class="tag ${clsStatus(c.status)}">${c.status || 'unknown'}</span>
        <span class="tag">${c.type || 'type?'}</span>
        <span class="tag">${c.scope || 'scope?'}</span>
        <span class="tag">${Number(c.score || c.confidence || 0).toFixed(2)}</span>
      </div>
    </div>`).join('') || '<div class="empty">没有匹配候选。</div>';
}
function selectCandidate(id) {
  selected = allCandidates.find(c => c.id === id) || null;
  renderList(); renderDetail();
}
function renderDetail() {
  const detail = document.getElementById('detail');
  if (!selected) { detail.innerHTML = '<div class="empty">请选择左侧候选记忆。</div>'; return; }
  const evidence = JSON.stringify(selected.evidence || [], null, 2);
  detail.innerHTML = `
    <div class="detail-head">
      <div><div class="section-title">候选记忆</div><h2 class="detail-title">${escapeHtml(selected.content || '')}</h2></div>
      <div class="score"><strong>${Number(selected.score || selected.confidence || 0).toFixed(2)}</strong><span>score</span></div>
    </div>
    <div class="meta">
      <span class="tag ${clsStatus(selected.status)}">${selected.status || 'unknown'}</span>
      <span class="tag">${selected.type || 'type?'}</span>
      <span class="tag">${selected.scope || 'scope?'}</span>
      ${(selected.tags || []).map(t => `<span class="tag">#${escapeHtml(t)}</span>`).join('')}
    </div>
    <div><div class="section-title">项目</div><div class="box">${escapeHtml(selected.project || 'Global')}</div></div>
    <div><div class="section-title">编辑后内容</div><textarea id="editedContent">${escapeHtml(selected.content || '')}</textarea></div>
    <div><div class="section-title">审核备注</div><textarea id="reviewNote" placeholder="为什么批准/拒绝/需要更多证据？"></textarea></div>
    <div><div class="section-title">证据</div><div class="box"><pre>${escapeHtml(evidence)}</pre></div></div>
    <div class="actions">
      <button class="approve" onclick="submitReview('approved')">批准</button>
      <button class="needs" onclick="submitReview('needs_more_evidence')">需要更多证据</button>
      <button class="reject" onclick="submitReview('rejected')">拒绝</button>
    </div>`;
}
async function submitReview(action) {
  if (!selected) return;
  const body = { candidate_id: selected.id, action, edited_content: document.getElementById('editedContent').value, note: document.getElementById('reviewNote').value, reviewer: 'user', candidate: selected };
  const res = await fetch('/api/memory/review', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
  const payload = await res.json();
  if (!res.ok) { toast(payload.detail || '保存失败'); return; }
  selected.review_action = action;
  toast(`已保存：${action}`);
}
function escapeHtml(s) { return String(s).replace(/[&<>"]/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[ch])); }
['search','statusFilter','scopeFilter'].forEach(id => document.addEventListener('input', e => { if (e.target && e.target.id === id) renderList(); }));
loadCandidates();
</script>
</body>
</html>
"""


def create_app(default_output_dir: Path | str = "outputs/runs", default_memory_dir: Path | str = ".deepagent/memory") -> FastAPI:
    output_dir = Path(default_output_dir).expanduser()
    memory_dir = Path(default_memory_dir).expanduser()
    app = FastAPI(title="DeepAgent Memory", version="0.1.0")

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return HOME_HTML

    @app.post("/api/analyze")
    def analyze(request: AnalyzeRequest) -> dict[str, Any]:
        project = Path(request.project).expanduser()
        if not project.exists() or not project.is_dir():
            raise HTTPException(status_code=400, detail="项目路径不存在或不是目录")
        return build_project_analysis(project, request.task).to_json_payload()

    @app.post("/api/run")
    def run(request: RunRequest) -> dict[str, Any]:
        project = Path(request.project).expanduser()
        if not project.exists() or not project.is_dir():
            raise HTTPException(status_code=400, detail="项目路径不存在或不是目录")
        analysis = build_project_analysis(project, request.task)
        patch_text = None
        patch_summary = None
        verification = None
        if request.patch:
            patch_text = create_patch(project, build_patch_proposal(request.task, analysis.report))
            patch_summary = summarize_changes(patch_text)
        if request.verify:
            commands = parse_verify_commands(request.verify_commands, analysis.to_metadata()["suggested_commands"])
            verification = run_verification(project, commands)
        run_dir = create_run_artifacts(
            output_dir,
            analysis.to_metadata(),
            analysis.report,
            patch=patch_text,
            patch_summary=patch_summary,
            verification=verification,
        )
        return {
            **analysis.to_json_payload(),
            "run_dir": str(run_dir),
            "patch": patch_text,
            "patch_summary": patch_summary,
            "verification": verification,
        }

    @app.get("/api/runs")
    def runs() -> list[dict[str, Any]]:
        db_path = output_dir / "runs.db"
        if db_path.exists():
            return RunStore(db_path).list_runs()
        if not output_dir.exists():
            return []
        items: list[dict[str, Any]] = []
        for metadata_file in sorted(output_dir.glob("*/metadata.json"), reverse=True):
            try:
                metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            metadata["run_dir"] = str(metadata_file.parent)
            items.append(metadata)
        return items


    @app.get("/memory-review", response_class=HTMLResponse)
    def memory_review() -> str:
        return MEMORY_REVIEW_HTML

    def _read_candidates_file(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        candidates: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    candidate = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, dict):
                    candidates.append(candidate)
        return candidates

    @app.get("/api/memory/candidates")
    def memory_candidates() -> dict[str, Any]:
        candidates_path = memory_dir / "agent-candidates.jsonl"
        if not candidates_path.exists():
            candidates_path = memory_dir / "candidates.jsonl"
        candidates = _read_candidates_file(candidates_path)
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
