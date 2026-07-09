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
from .memory_dreaming import normalize_review_decision
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


MEMORY_CONFIG_HTML = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Dream Memory 配置</title>
  <style>
    :root { --ink:#18211d; --paper:#f7f3ea; --panel:#fffdf8; --line:#ddd2c1; --muted:#736b5f; --green:#2f6b4f; --green-soft:#e6f1ea; --amber:#8a6118; --amber-soft:#fff3cf; --red:#a9473f; --red-soft:#f8ded9; --shadow:0 18px 50px rgba(47,39,27,.10); }
    * { box-sizing:border-box; }
    body { margin:0; min-height:100vh; background:var(--paper); color:var(--ink); font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
    button,input,select,textarea { font:inherit; }
    input,select,textarea { width:100%; border:1px solid var(--line); border-radius:8px; background:#fffaf1; color:var(--ink); padding:10px 11px; }
    textarea { min-height:132px; line-height:1.5; resize:vertical; font-family:"Cascadia Mono",Consolas,monospace; }
    button { min-height:40px; border:1px solid transparent; border-radius:8px; padding:9px 13px; font-weight:800; cursor:pointer; }
    button:focus-visible,input:focus-visible,select:focus-visible,textarea:focus-visible { outline:3px solid rgba(47,107,79,.24); outline-offset:2px; }
    .shell { min-height:100vh; display:grid; grid-template-columns:280px minmax(0,1fr); }
    .sidebar { padding:24px 18px; color:#eef1e8; background:linear-gradient(180deg,#17201c 0%,#223027 100%); }
    .brand { display:grid; gap:8px; padding-bottom:18px; border-bottom:1px solid rgba(255,255,255,.12); }
    .brand h1 { margin:0; font-size:28px; line-height:1; }
    .brand p { margin:0; color:#b7c2b4; font-size:13px; line-height:1.55; }
    .nav { display:grid; gap:8px; margin-top:20px; }
    .nav a { color:#edf3e9; text-decoration:none; border:1px solid rgba(255,255,255,.12); border-radius:9px; padding:10px 11px; background:rgba(255,255,255,.055); font-weight:800; }
    .content { min-width:0; padding:28px 32px 36px; }
    .hero { display:flex; justify-content:space-between; gap:18px; align-items:flex-start; margin-bottom:18px; }
    .hero h2 { margin:0; font-size:36px; letter-spacing:0; }
    .hero p { margin:8px 0 0; color:var(--muted); }
    .actions { display:flex; gap:10px; flex-wrap:wrap; }
    button:disabled { opacity:.68; cursor:progress; }
    .primary { background:var(--green); color:#fff; }
    .secondary { background:#fffaf2; color:var(--ink); border-color:var(--line); }
    .danger { background:var(--red); color:#fff; }
    .grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }
    .card { border:1px solid var(--line); border-radius:10px; background:var(--panel); box-shadow:var(--shadow); padding:18px; }
    .card h3 { margin:0 0 5px; font-size:17px; }
    .card p { margin:0 0 14px; color:var(--muted); font-size:13px; line-height:1.5; }
    .fields { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }
    .profile-list { display:flex; gap:8px; flex-wrap:wrap; margin:0 0 14px; }
    .profile-chip { min-height:34px; padding:7px 10px; border-color:var(--line); background:#fffaf2; color:var(--ink); }
    .profile-chip.active { background:var(--green); color:#fff; border-color:var(--green); }
    label { display:grid; gap:6px; color:#4e463d; font-size:12px; font-weight:850; letter-spacing:0; }
    .wide { grid-column:1 / -1; }
    .check { display:flex; align-items:center; gap:8px; min-height:40px; text-transform:none; letter-spacing:0; font-size:13px; }
    .check input[type="checkbox"] { flex:0 0 16px; width:16px; height:16px; min-height:0; margin:0; padding:0; }
    .status { min-height:24px; margin:12px 0 0; color:var(--muted); font-size:13px; }
    .status-banner { position:sticky; top:12px; z-index:3; margin:-4px 0 18px; padding:12px 14px; border:1px solid var(--line); border-radius:10px; background:#fffaf2; box-shadow:0 8px 26px rgba(47,39,27,.08); color:var(--ink); font-weight:750; }
    .status.ok { color:var(--green); border-color:#b8d8c2; background:var(--green-soft); }
    .status.error { color:var(--red); border-color:#e7aaa2; background:var(--red-soft); }
    .status.loading { color:var(--amber); border-color:#ecd08a; background:var(--amber-soft); }
    code { background:#f1eadf; padding:2px 5px; border-radius:6px; }
    @media (max-width:980px) { .shell { display:block; } .grid,.fields { grid-template-columns:1fr; } .hero { display:grid; } .content { padding:22px 16px; } }
  </style>
</head>
<body>
<main class="shell">
  <aside class="sidebar">
    <div class="brand"><h1>Dream Memory 配置</h1><p>把命令行参数集中到一个可保存的配置页，默认写入 <code>.dream-memory/config.json</code>。</p></div>
    <nav class="nav"><a href="/memory-review">候选审核</a><a href="/memory-config">配置页面</a></nav>
  </aside>
  <section class="content">
    <header class="hero">
      <div><h2>运行与模型配置</h2><p>覆盖 init / scan / import / dream / review / apply / run / status / resume / trace / context / summary / export / eval / check-provider 的参数。</p></div>
      <div class="actions"><button class="primary" onclick="saveConfig(event)">保存配置</button><button class="secondary" onclick="loadConfig(event)">重新加载</button><button class="danger" onclick="resetConfig(event)">恢复默认</button></div>
    </header>
    <div id="status" class="status status-banner">正在加载配置...</div>
    <div class="grid">
      <section class="card"><h3>模型配置</h3><p>配置多个模型配置档，重试失败时会按备用配置链依次切换。</p><div id="profileList" class="profile-list"></div><div class="fields">
        <label>当前配置档 <select id="activeProfile" onchange="switchActiveProfile(this.value)"></select></label>
        <div class="actions"><button type="button" class="secondary" onclick="addProfile()">新增配置档</button><button type="button" class="danger" onclick="deleteProfile()">删除配置档</button></div>
        <label>模型服务商 <select id="provider" onchange="refreshModelList()"><option value="anthropic">Anthropic</option><option value="openai">OpenAI</option><option value="openrouter">OpenRouter</option></select></label>
        <label>模型名称 <select id="model"><option value="">点击获取模型列表后选择</option></select></label>
        <div class="wide actions"><button type="button" class="secondary" onclick="loadModelCatalog(event)">获取模型列表</button></div>
        <label class="wide">接口密钥 <input id="apiKey" type="password" placeholder="直接写入 config.json" /></label>
        <label>密钥环境变量 <input id="apiKeyEnv" placeholder="OPENAI_API_KEY" /></label>
        <label>接口地址 <input id="baseUrl" placeholder="http://localhost:3000" /></label>
        <label>超时秒数 <input id="timeoutSeconds" type="number" min="1" /></label>
        <label>检查配置档 <select id="checkProviderProfile"></select></label>
        <label class="check"><input id="checkProviderInvoke" type="checkbox" /> 检查时调用模型</label>
        <label class="check"><input id="checkProviderAll" type="checkbox" /> 检查全部配置档</label>
      </div></section>
      <section class="card"><h3>重试与切换</h3><p>对应 <code>model_policy.default_profile</code>、<code>fallback_chain</code> 和 retry。</p><div class="fields">
        <label>默认配置档 <select id="defaultProfile"></select></label>
        <label>备用配置链 <input id="fallbackChain" placeholder="primary,backup" /></label>
        <label>最大重试次数 <input id="retryMaxAttempts" type="number" min="1" /></label>
        <label>初始等待秒数 <input id="retryInitialDelay" type="number" min="0" step="0.1" /></label>
        <label>退避倍数 <input id="retryBackoff" type="number" min="1" step="0.1" /></label>
        <label>最长等待秒数 <input id="retryMaxDelay" type="number" min="0" step="0.1" /></label>
        <label class="wide">重试状态码 <input id="retryStatus" placeholder="429,500,502,503,504" /></label>
        <label class="check"><input id="retryTimeout" type="checkbox" /> 超时后重试</label>
        <label class="check"><input id="retrySwitchModel" type="checkbox" /> 允许切换模型进行重试</label>
        <label class="check"><input id="allowRulesFallback" type="checkbox" /> 允许规则兜底</label>
      </div></section>
      <section class="card"><h3>导入与扫描</h3><p>对应 <code>--codex-home --claude-home --claude-state --project --output-dir</code>。</p><div class="fields">
        <label>Codex 目录 <input id="codexHome" /></label>
        <label>Claude 目录 <input id="claudeHome" /></label>
        <label>Claude 状态文件 <input id="claudeState" /></label>
        <label>导入输出目录 <input id="importsOutputDir" /></label>
        <label class="wide">项目根目录 <input id="projectRoots" placeholder="多个路径用逗号分隔" /></label>
      </div></section>
      <section class="card"><h3>运行参数</h3><p>对应 <code>run/dream/pipeline --input --project --mode --dry-run --invoke-model --memory-cards</code>。</p><div class="fields">
        <label>默认事件文件 <input id="defaultInput" /></label>
        <label>默认项目 <input id="defaultProject" placeholder=". 或留空" /></label>
        <label>运行模式 <select id="mode"><option value="ai">AI</option><option value="rules">规则</option></select></label>
        <label>记忆卡片文件 <input id="memoryCards" /></label>
        <label>输出目录 <input id="outputDir" /></label>
        <label class="check"><input id="invokeModel" type="checkbox" /> 调用模型</label>
      </div></section>
      <section class="card"><h3>上下文与导出</h3><p>对应 <code>context --limit --format</code> 与 <code>export --target --scope --output-dir --limit</code>。</p><div class="fields">
        <label>上下文条数 <input id="contextLimit" type="number" min="1" /></label>
        <label>上下文格式 <select id="contextFormat"><option value="json">JSON</option><option value="markdown">Markdown</option></select></label>
        <label>导出目标 <select id="exportTarget"><option value="both">Codex 和 Claude</option><option value="codex">Codex</option><option value="claude">Claude</option></select></label>
        <label>导出范围 <select id="exportScope"><option value="project">当前项目</option><option value="global">全局</option></select></label>
        <label>导出条数 <input id="exportLimit" type="number" min="1" /></label>
        <label>导出目录 <input id="exportOutputDir" /></label>
        <label class="check"><input id="autoExport" type="checkbox" /> 自动导出</label>
      </div></section>
      <section class="card"><h3>审核与应用</h3><p>对应 <code>review --candidates</code>、<code>apply --reviewed</code>、<code>resume --reviewed --reviewer</code>。</p><div class="fields">
        <label>候选文件 <input id="reviewCandidates" /></label>
        <label>审核结果文件 <input id="applyReviewed" /></label>
        <label>恢复用审核文件 <input id="resumeReviewed" /></label>
        <label>审核人 <input id="reviewer" /></label>
      </div></section>
      <section class="card"><h3>状态与追踪</h3><p>对应 <code>status --run-id</code>、<code>resume --run-id</code>、<code>trace --run-id --candidate-id</code>。</p><div class="fields">
        <label>状态 Run ID <input id="statusRunId" /></label>
        <label>恢复 Run ID <input id="resumeRunId" /></label>
        <label>Trace Run ID <input id="traceRunId" /></label>
        <label>Trace 候选 ID <input id="traceCandidateId" /></label>
      </div></section>
      <section class="card"><h3>初始化与评估</h3><p>对应 <code>init</code>、<code>init-config</code>、<code>extract-facts</code>、<code>summary</code>、<code>eval</code>。</p><div class="fields">
        <label>初始化路径 <input id="initPath" /></label>
        <label>初始配置输出 <input id="initConfigOutput" /></label>
        <label>扫描输出 <input id="scanOutput" /></label>
        <label>导入来源 <select id="importSource"><option value="all">全部</option><option value="codex">Codex</option><option value="claude">Claude</option></select></label>
        <label>事实抽取输入 <input id="extractInput" /></label>
        <label>事实抽取项目 <input id="extractProject" /></label>
        <label>事实抽取输出目录 <input id="extractOutputDir" /></label>
        <label>汇总范围 <select id="summaryScope"><option value="all-projects">所有项目</option></select></label>
        <label>汇总输出 <input id="summaryOutput" /></label>
        <label>评估输入 <input id="evalInput" /></label>
        <label>评估项目 <input id="evalProject" /></label>
        <label>评估模式 <select id="evalMode"><option value="rules">规则</option><option value="ai">AI</option></select></label>
        <label>评估输出 <input id="evalOutput" /></label>
        <label>评估最大行数 <input id="evalMaxRows" type="number" min="1" /></label>
        <label>评估最大尝试 <input id="evalMaxAttempts" type="number" min="1" /></label>
        <label class="check"><input id="evalContinueOnError" type="checkbox" /> 评估出错后继续</label>
        <label class="check"><input id="evalFallbackRulesOnError" type="checkbox" /> AI 失败时规则兜底</label>
        <label class="check"><input id="evalFallbackRulesOnEmpty" type="checkbox" /> AI 空结果时规则兜底</label>
        <label class="check"><input id="initForce" type="checkbox" /> 强制初始化</label>
        <label class="check"><input id="importDryRun" type="checkbox" /> 导入试运行</label>
        <label class="check"><input id="dreamApply" type="checkbox" /> 抽取后应用</label>
      </div></section>
      <section class="card"><h3>原始 JSON</h3><p>高级用户可以直接编辑完整配置。保存时会做结构校验。</p><textarea id="rawConfig"></textarea></section>
    </div>
  </section>
</main>
<script>
let currentConfig=null;let modelCatalog={};let activeProfile='primary';
async function loadConfig(event){await withBusyStatus(event&&event.currentTarget,'正在加载配置...',async()=>{const res=await fetch('/api/memory/config');const data=await res.json();if(!res.ok)throw new Error(data.detail||JSON.stringify(data));currentConfig=data.config;activeProfile=profileNames()[0]||'primary';fillForm(currentConfig);setStatus(`已加载 ${data.config_path}`,'ok');},'加载失败');}
async function saveConfig(event){await withBusyStatus(event&&event.currentTarget,'正在保存配置...',async()=>{const config=collectForm();const res=await fetch('/api/memory/config',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({config})});const data=await res.json();if(!res.ok)throw new Error(data.detail||JSON.stringify(data));currentConfig=data.config;fillForm(currentConfig);setStatus('配置已保存','ok');},'保存失败');}
async function resetConfig(event){await withBusyStatus(event&&event.currentTarget,'正在恢复默认配置...',async()=>{const res=await fetch('/api/memory/config/reset',{method:'POST'});const data=await res.json();if(!res.ok)throw new Error(data.detail||JSON.stringify(data));currentConfig=data.config;activeProfile=profileNames()[0]||'primary';fillForm(currentConfig);setStatus('已恢复默认配置','ok');},'重置失败');}
async function loadModelCatalog(event){syncActiveProfile();await withBusyStatus(event&&event.currentTarget,'正在获取模型列表...',async()=>{const request={provider:value('provider')||'anthropic',model:value('model')||modelValueFromConfig(),api_key:value('apiKey'),api_key_env:nullIfEmpty(value('apiKeyEnv')),base_url:nullIfEmpty(value('baseUrl')),timeout_seconds:numberValue('timeoutSeconds',60)};const res=await fetch('/api/memory/models',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(request)});const data=await res.json();if(!res.ok)throw new Error(data.detail||JSON.stringify(data));modelCatalog[data.provider]=data.models||[];refreshModelList();setStatus(`模型列表已获取：${(data.models||[]).length} 个模型`,'ok');},'模型列表加载失败');}
function addProfile(){syncActiveProfile();const raw=prompt('请输入新配置档名称，例如 backup');const name=safeProfileName(raw);if(!name){setStatus('新增失败：配置档名称只能使用字母、数字、下划线和短横线','error');return;}currentConfig=currentConfig||{};currentConfig.models=currentConfig.models||{};if(currentConfig.models[name]){setStatus(`新增失败：配置档 ${name} 已存在`,'error');return;}const source=currentProfile();currentConfig.models[name]={provider:source.provider||'openai',model:source.model||'',api_key:'',api_key_env:null,base_url:source.base_url||null,timeout_seconds:source.timeout_seconds||60};activeProfile=name;refreshProfileSelectors();fillActiveProfile();updateRawConfig();setStatus(`已新增配置档 ${name}`,'ok');}
function deleteProfile(){if(!currentConfig||!currentConfig.models)return;const names=profileNames();if(names.length<=1){setStatus('至少需要保留一个模型配置档','error');return;}const name=activeProfile;if(!confirm(`删除配置档 ${name}？`))return;delete currentConfig.models[name];const policy=currentConfig.model_policy||{};if(policy.default_profile===name)policy.default_profile=profileNames()[0];policy.fallback_chain=(policy.fallback_chain||[]).filter(item=>item!==name);if(policy.fallback_chain.length===0)policy.fallback_chain=[policy.default_profile];if(currentConfig.check_provider_profile===name)currentConfig.check_provider_profile=policy.default_profile;activeProfile=policy.default_profile;refreshProfileSelectors();fillActiveProfile();updateRawConfig();setStatus(`已删除配置档 ${name}`,'ok');}
function switchActiveProfile(name){syncActiveProfile();activeProfile=name;fillActiveProfile();refreshProfileSelectors();setStatus(`正在编辑配置档 ${name}`,'loading');}
function syncActiveProfile(){if(!currentConfig)return;currentConfig.models=currentConfig.models||{};currentConfig.models[activeProfile]={provider:value('provider')||'anthropic',model:value('model'),api_key:value('apiKey'),api_key_env:nullIfEmpty(value('apiKeyEnv')),base_url:nullIfEmpty(value('baseUrl')),timeout_seconds:numberValue('timeoutSeconds',60)};}
function refreshModelList(){const provider=value('provider')||'anthropic';const select=document.getElementById('model');const selected=select.value||modelValueFromConfig();const models=modelCatalog[provider]||[];const options=[];if(selected&&!models.includes(selected)){options.push(`<option value="${escapeAttr(selected)}">${escapeHtml(selected)}（当前配置）</option>`);}if(models.length===0){options.push('<option value="">点击获取模型列表后选择</option>');}else{options.push('<option value="">请选择模型</option>');options.push(...models.map(model=>`<option value="${escapeAttr(model)}">${escapeHtml(model)}</option>`));}select.innerHTML=options.join('');select.value=selected||'';}
function refreshProfileSelectors(){const names=profileNames();const optionHtml=names.map(name=>`<option value="${escapeAttr(name)}">${escapeHtml(name)}</option>`).join('');setSelectOptions('activeProfile',optionHtml,activeProfile);setSelectOptions('defaultProfile',optionHtml,(currentConfig.model_policy||{}).default_profile||activeProfile);setSelectOptions('checkProviderProfile',optionHtml,currentConfig.check_provider_profile||activeProfile);const list=document.getElementById('profileList');list.innerHTML=names.map(name=>`<button type="button" class="profile-chip ${name===activeProfile?'active':''}" onclick="switchActiveProfile('${escapeJs(name)}')">${escapeHtml(name)}</button>`).join('');}
function modelValueFromConfig(){return (currentProfile()||{}).model||'';}
function currentProfile(){return ((currentConfig||{}).models||{})[activeProfile]||{};}
function profileNames(){return Object.keys(((currentConfig||{}).models)||{});}
function fillActiveProfile(){const profile=currentProfile();setValue('provider',profile.provider||'anthropic');setValue('apiKey',profile.api_key);setValue('apiKeyEnv',profile.api_key_env);setValue('baseUrl',profile.base_url);setValue('timeoutSeconds',profile.timeout_seconds);refreshModelList();setValue('model',profile.model);}
function fillForm(config){const policy=config.model_policy||{};const retry=policy.retry||{};activeProfile=activeProfile&&config.models&&config.models[activeProfile]?activeProfile:(policy.default_profile||Object.keys(config.models||{})[0]||'primary');refreshProfileSelectors();fillActiveProfile();setChecked('checkProviderInvoke',config.check_provider_invoke);setChecked('checkProviderAll',config.check_provider_all);setValue('fallbackChain',(policy.fallback_chain||[]).join(','));setValue('retryMaxAttempts',retry.max_attempts);setValue('retryInitialDelay',retry.initial_delay_seconds);setValue('retryBackoff',retry.backoff_factor);setValue('retryMaxDelay',retry.max_delay_seconds);setValue('retryStatus',(retry.retry_on_status||[]).join(','));setChecked('retryTimeout',retry.retry_on_timeout);setChecked('retrySwitchModel',retry.switch_model_on_retry);setChecked('allowRulesFallback',policy.allow_rules_fallback);setValue('codexHome',config.codex_home);setValue('claudeHome',config.claude_home);setValue('claudeState',config.claude_state);setValue('importsOutputDir',config.imports_output_dir);setValue('projectRoots',(config.project_roots||[]).join(','));setValue('defaultInput',config.default_input||'.dream-memory/imports/all-events.jsonl');setValue('defaultProject',config.default_project||'.');setValue('mode',config.mode);setValue('memoryCards',config.memory_cards);setValue('outputDir',config.output_dir);setChecked('invokeModel',config.invoke_model);setValue('contextLimit',config.context_limit);setValue('contextFormat',config.context_format);setValue('exportTarget',config.export_target);setValue('exportScope',config.export_scope);setValue('exportLimit',config.export_limit);setValue('exportOutputDir',config.export_output_dir);setChecked('autoExport',config.auto_export);setValue('reviewCandidates',config.review_candidates);setValue('applyReviewed',config.apply_reviewed);setValue('resumeReviewed',config.resume_reviewed);setValue('reviewer',config.reviewer);setValue('statusRunId',config.status_run_id);setValue('resumeRunId',config.resume_run_id);setValue('traceRunId',config.trace_run_id);setValue('traceCandidateId',config.trace_candidate_id);setValue('initPath',config.init_path);setValue('initConfigOutput',config.init_config_output);setValue('scanOutput',config.scan_output);setValue('importSource',config.import_source);setValue('extractInput',config.extract_input);setValue('extractProject',config.extract_project);setValue('extractOutputDir',config.extract_output_dir);setValue('summaryScope',config.summary_scope);setValue('summaryOutput',config.summary_output);setValue('evalInput',config.eval_input);setValue('evalProject',config.eval_project);setValue('evalMode',config.eval_mode);setValue('evalOutput',config.eval_output);setValue('evalMaxRows',config.eval_max_rows);setValue('evalMaxAttempts',config.eval_max_attempts);setChecked('evalContinueOnError',config.eval_continue_on_error);setChecked('evalFallbackRulesOnError',config.eval_fallback_rules_on_error);setChecked('evalFallbackRulesOnEmpty',config.eval_fallback_rules_on_empty);setChecked('initForce',config.init_force);setChecked('importDryRun',config.import_dry_run);setChecked('dreamApply',config.dream_apply);updateRawConfig();}
function collectForm(){let config;const raw=document.getElementById('rawConfig').value.trim();try{config=raw?JSON.parse(raw):structuredClone(currentConfig||{});}catch(error){throw new Error('原始 JSON 不是合法 JSON');}syncActiveProfile();config.models=structuredClone(currentConfig.models||{});config.model_policy=config.model_policy||{};config.model_policy.retry=config.model_policy.retry||{};config.check_provider_profile=value('checkProviderProfile')||activeProfile;config.check_provider_invoke=checked('checkProviderInvoke');config.check_provider_all=checked('checkProviderAll');config.model_policy.default_profile=value('defaultProfile')||activeProfile;config.model_policy.fallback_chain=splitList(value('fallbackChain'));if(!config.model_policy.fallback_chain.includes(config.model_policy.default_profile)){config.model_policy.fallback_chain=[config.model_policy.default_profile,...config.model_policy.fallback_chain];}config.model_policy.retry.max_attempts=numberValue('retryMaxAttempts',3);config.model_policy.retry.initial_delay_seconds=numberValue('retryInitialDelay',1);config.model_policy.retry.backoff_factor=numberValue('retryBackoff',2);config.model_policy.retry.max_delay_seconds=numberValue('retryMaxDelay',8);config.model_policy.retry.retry_on_status=splitList(value('retryStatus')).map(Number).filter(Number.isFinite);config.model_policy.retry.retry_on_timeout=checked('retryTimeout');config.model_policy.retry.switch_model_on_retry=checked('retrySwitchModel');config.model_policy.allow_rules_fallback=checked('allowRulesFallback');config.codex_home=value('codexHome');config.claude_home=value('claudeHome');config.claude_state=value('claudeState');config.imports_output_dir=value('importsOutputDir');config.project_roots=splitList(value('projectRoots'));config.default_input=value('defaultInput');config.default_project=nullIfEmpty(value('defaultProject'));config.mode=value('mode');config.memory_cards=value('memoryCards');config.output_dir=value('outputDir');config.invoke_model=checked('invokeModel');config.context_limit=numberValue('contextLimit',12);config.context_format=value('contextFormat');config.export_target=value('exportTarget');config.export_scope=value('exportScope');config.export_limit=nullableNumberValue('exportLimit');config.export_output_dir=nullIfEmpty(value('exportOutputDir'));config.auto_export=checked('autoExport');config.review_candidates=value('reviewCandidates');config.apply_reviewed=value('applyReviewed');config.resume_reviewed=nullIfEmpty(value('resumeReviewed'));config.reviewer=value('reviewer')||'user';config.status_run_id=nullIfEmpty(value('statusRunId'));config.resume_run_id=nullIfEmpty(value('resumeRunId'));config.trace_run_id=nullIfEmpty(value('traceRunId'));config.trace_candidate_id=nullIfEmpty(value('traceCandidateId'));config.init_path=value('initPath')||'.';config.init_config_output=value('initConfigOutput')||'.dream-memory/config.json';config.scan_output=nullIfEmpty(value('scanOutput'));config.import_source=value('importSource')||'all';config.extract_input=value('extractInput');config.extract_project=nullIfEmpty(value('extractProject'));config.extract_output_dir=value('extractOutputDir');config.summary_scope=value('summaryScope')||'all-projects';config.summary_output=nullIfEmpty(value('summaryOutput'));config.eval_input=nullIfEmpty(value('evalInput'));config.eval_project=nullIfEmpty(value('evalProject'));config.eval_mode=value('evalMode')||'rules';config.eval_output=nullIfEmpty(value('evalOutput'));config.eval_max_rows=nullableNumberValue('evalMaxRows');config.eval_max_attempts=nullableNumberValue('evalMaxAttempts');config.eval_continue_on_error=checked('evalContinueOnError');config.eval_fallback_rules_on_error=checked('evalFallbackRulesOnError');config.eval_fallback_rules_on_empty=checked('evalFallbackRulesOnEmpty');config.init_force=checked('initForce');config.import_dry_run=checked('importDryRun');config.dream_apply=checked('dreamApply');currentConfig=config;return config;}
function value(id){return document.getElementById(id).value.trim();}
function setValue(id,value){document.getElementById(id).value=value==null?'':String(value);}
function setSelectOptions(id,html,value){const el=document.getElementById(id);el.innerHTML=html;el.value=value||'';}
function checked(id){return document.getElementById(id).checked;}
function setChecked(id,value){document.getElementById(id).checked=Boolean(value);}
function splitList(text){return String(text||'').split(',').map(item=>item.trim()).filter(Boolean);}
function nullIfEmpty(text){const value=String(text||'').trim();return value?value:null;}
function numberValue(id,fallback){const parsed=Number(value(id));return Number.isFinite(parsed)?parsed:fallback;}
function nullableNumberValue(id){const raw=value(id);if(!raw)return null;const parsed=Number(raw);return Number.isFinite(parsed)?parsed:null;}
function escapeHtml(text){return String(text).replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));}
function escapeAttr(text){return String(text).replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));}
function escapeJs(text){return String(text).replaceAll(String.fromCharCode(92),String.fromCharCode(92,92)).replaceAll("'",String.fromCharCode(92)+"'");}
function safeProfileName(text){const value=String(text||'').trim();return /^[A-Za-z0-9_-]+$/.test(value)?value:'';}
function updateRawConfig(){setValue('rawConfig',JSON.stringify(currentConfig,null,2));}
function setStatus(text,tone){const el=document.getElementById('status');el.textContent=text;el.className=`status status-banner ${tone||''}`;}
async function withBusyStatus(button,message,operation,errorPrefix){const oldText=button&&button.textContent;try{setStatus(message,'loading');if(button){button.disabled=true;button.textContent='处理中...';}await operation();}catch(error){setStatus(`${errorPrefix}：${error.message}`,'error');}finally{if(button){button.disabled=false;button.textContent=oldText;}}}
loadConfig();
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
    :root { --ink:#18211d; --ink-2:#24312b; --paper:#f7f3ea; --panel-2:#f1eadf; --line:#ddd2c1; --muted:#736b5f; --green:#2f6b4f; --green-soft:#dcebe1; --amber:#9b6823; --amber-soft:#f3e5c8; --red:#a9473f; --red-soft:#f0d7d2; --blue-soft:#dbe8f2; --shadow:0 18px 50px rgba(47,39,27,.10); color-scheme:light; }
    * { box-sizing:border-box; }
    body { margin:0; min-height:100vh; background:var(--paper); color:var(--ink); font-family:Inter,ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
    button,input,select,textarea { font:inherit; }
    button { min-height:40px; border:1px solid transparent; border-radius:8px; padding:9px 12px; font-weight:750; cursor:pointer; transition:transform .14s ease,box-shadow .14s ease,background .14s ease,border-color .14s ease; }
    button:hover { transform:translateY(-1px); box-shadow:0 8px 18px rgba(24,33,29,.12); }
    button:focus-visible,input:focus-visible,select:focus-visible,textarea:focus-visible { outline:3px solid rgba(47,107,79,.24); outline-offset:2px; }
    input,select,textarea { width:100%; border:1px solid var(--line); border-radius:8px; background:#fffaf1; color:var(--ink); padding:10px 11px; }
    textarea { min-height:92px; line-height:1.45; resize:vertical; }
    .app-shell { height:100vh; display:grid; grid-template-columns:286px minmax(0,1fr); overflow:hidden; background:linear-gradient(90deg,var(--ink) 0 286px,transparent 286px),radial-gradient(circle at 58% -10%,rgba(159,117,58,.10),transparent 38%); }
    .sidebar { height:100vh; min-height:0; padding:16px 14px; color:#eef1e8; background:linear-gradient(180deg,#17201c 0%,#223027 100%); overflow:auto; scrollbar-gutter:stable; border-right:1px solid rgba(255,255,255,.08); }
    .brand { display:grid; gap:5px; margin-bottom:12px; padding:2px 3px 12px; border-bottom:1px solid rgba(255,255,255,.12); }
    .brand a { color:#edf3e9; text-decoration:none; font-size:12px; font-weight:850; }
    .eyebrow { color:#c9d7c8; font-size:11px; font-weight:800; letter-spacing:.08em; text-transform:uppercase; }
    .brand h1 { margin:0; font-family:"Aptos Display","Segoe UI",sans-serif; font-size:24px; line-height:1; letter-spacing:0; }
    .brand p { margin:0; max-width:250px; color:#b7c2b4; font-size:12px; line-height:1.35; }
    .side-section { margin-top:12px; }
    .side-title { display:flex; align-items:center; justify-content:space-between; gap:10px; margin:0 0 7px; color:#edf3e9; font-size:12px; font-weight:850; }
    .run-list { display:grid; gap:6px; }
    .run-card { width:100%; min-height:0; padding:8px 10px; text-align:left; color:#eef1e8; border-color:rgba(255,255,255,.10); background:rgba(255,255,255,.055); box-shadow:none; }
    .run-card:hover { background:rgba(255,255,255,.09); }
    .run-card.active { background:#edf3e9; color:var(--ink); border-color:#edf3e9; }
    .run-id { display:block; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:12px; font-weight:850; }
    .run-meta { display:flex; flex-wrap:wrap; gap:5px; margin-top:5px; }
    .start-form { display:grid; gap:7px; border:1px solid rgba(255,255,255,.12); border-radius:10px; padding:9px; background:rgba(255,255,255,.055); }
    .start-form label { display:grid; gap:5px; color:#dbe6d8; font-size:11px; font-weight:800; text-transform:uppercase; }
    .start-form label.check { display:flex; align-items:center; min-height:28px; text-transform:none; font-size:12px; }
    .start-form label.check input[type="checkbox"] { flex:0 0 16px; width:16px; height:16px; min-height:0; margin:0; padding:0; }
    .start-form input { min-height:34px; padding:8px 9px; border-color:rgba(255,255,255,.14); background:rgba(255,255,255,.08); color:#f5f8f1; }
    .start-form input::placeholder { color:rgba(245,248,241,.54); }
    .start-hint { margin:-2px 0 0; color:#aebdad; font-size:12px; line-height:1.45; }
    .quick-row { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:7px; }
    .quick-row button { min-height:32px; padding:6px 8px; color:#eef1e8; border-color:rgba(255,255,255,.12); background:rgba(255,255,255,.07); box-shadow:none; font-size:12px; }
    .quick-row button:hover { background:rgba(255,255,255,.11); }
    .start-options { display:flex; align-items:center; justify-content:space-between; gap:10px; color:#cbd8c8; font-size:12px; }
    .start-options label { display:flex; grid-template-columns:none; align-items:center; gap:7px; text-transform:none; font-size:12px; font-weight:700; }
    .start-options input { width:auto; min-height:0; }
    .start-button { width:100%; background:#edf3e9; color:var(--ink); }
    .progress-card { display:grid; gap:7px; border:1px solid rgba(255,255,255,.12); border-radius:10px; padding:9px; background:rgba(255,255,255,.055); color:#dbe6d8; font-size:12px; }
    .progress-card strong { color:#f6faf1; }
    .progress-title { display:flex; align-items:center; justify-content:space-between; gap:8px; }
    .progress-bar { height:7px; border-radius:999px; background:rgba(255,255,255,.14); overflow:hidden; }
    .progress-bar span { display:block; height:100%; width:0%; background:#cfe2cf; transition:width .2s ease; }
    .progress-steps { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:5px 8px; }
    .progress-step { display:grid; grid-template-columns:14px minmax(0,1fr); gap:8px; align-items:start; color:#aebdad; }
    .progress-dot { width:8px; height:8px; margin-top:4px; border-radius:999px; background:rgba(255,255,255,.25); }
    .progress-step.done { color:#edf3e9; }
    .progress-step.done .progress-dot { background:#cfe2cf; }
    .progress-step.active .progress-dot { background:#f3d38b; box-shadow:0 0 0 4px rgba(243,211,139,.15); }
    .progress-log { display:grid; gap:4px; padding-top:2px; border-top:1px solid rgba(255,255,255,.10); color:#bdcab9; }
    .progress-error { color:#f0d7d2; }
    .layout { display:grid; grid-template-rows:auto minmax(0,1fr); min-width:0; height:100vh; min-height:0; overflow:hidden; }
    .topbar { display:grid; grid-template-columns:minmax(0,1fr) minmax(420px,620px); gap:16px; align-items:end; padding:15px 22px 12px; }
    .topbar h2 { margin:0; font-family:"Aptos Display","Segoe UI",sans-serif; font-size:clamp(22px,2.4vw,32px); line-height:1.04; letter-spacing:0; }
    .topbar p { margin:4px 0 0; color:var(--muted); line-height:1.35; font-size:13px; }
    .topbar-tools { display:grid; gap:10px; justify-items:end; min-width:min(100%,520px); }
    .config-button { display:inline-flex; align-items:center; justify-content:center; min-height:40px; border:1px solid var(--green); border-radius:8px; padding:9px 13px; background:var(--green); color:#fff; text-decoration:none; font-weight:850; line-height:1.2; }
    .config-button:hover { background:#25593f; border-color:#25593f; }
    .config-button:focus-visible { outline:3px solid rgba(47,107,79,.24); outline-offset:2px; }
    .toolbar { display:grid; grid-template-columns:minmax(190px,1fr) 132px 118px 118px; gap:8px; min-width:min(100%,620px); }
    .workspace { display:grid; grid-template-columns:minmax(340px,.88fr) minmax(430px,1.12fr); gap:14px; padding:0 22px 18px; min-height:0; overflow:hidden; }
    .panel { min-width:0; border:1px solid var(--line); border-radius:10px; background:rgba(255,253,248,.92); box-shadow:var(--shadow); }
    .queue-panel,.detail-panel { display:grid; grid-template-rows:auto minmax(0,1fr); min-height:0; height:100%; }
    .panel-head { padding:12px 14px 10px; border-bottom:1px solid var(--line); background:linear-gradient(180deg,rgba(255,255,255,.78),rgba(246,240,229,.74)); border-radius:10px 10px 0 0; }
    .panel-head h3 { margin:0; font-size:16px; letter-spacing:0; }
    .panel-head p { margin:4px 0 0; color:var(--muted); font-size:12px; }
    .stats-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:7px; margin-top:10px; }
    .stat { border:1px solid rgba(221,210,193,.9); border-radius:8px; background:#fff9ef; padding:8px; min-width:0; }
    .stat b { display:block; font-size:19px; line-height:1; color:var(--ink); }
    .stat span { display:block; margin-top:4px; color:var(--muted); font-size:10px; font-weight:750; text-transform:uppercase; }
    .queue-body { min-height:0; overflow:auto; padding:9px 10px 12px; }
    .group-title { display:flex; align-items:center; gap:10px; margin:10px 4px 6px; color:#5f574e; font-size:11px; font-weight:850; letter-spacing:.04em; text-transform:uppercase; }
    .group-title::after { content:""; height:1px; flex:1; background:var(--line); }
    .candidate-card { width:100%; min-height:76px; display:grid; gap:6px; margin-bottom:7px; padding:10px; text-align:left; color:var(--ink); border:1px solid var(--line); background:#fffaf2; box-shadow:none; }
    .candidate-card:hover { border-color:#bdac92; background:#fffdf8; }
    .candidate-card.active { border-color:var(--green); background:#f6fbf4; box-shadow:0 0 0 3px rgba(47,107,79,.12); }
    .candidate-card header { display:flex; align-items:center; justify-content:space-between; gap:12px; min-width:0; }
    .candidate-card strong { overflow:hidden; color:var(--ink); text-overflow:ellipsis; white-space:nowrap; font-size:14px; }
    .candidate-content { display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; margin:0; color:#3d352d; line-height:1.35; font-size:12px; }
    .candidate-footer { display:flex; flex-wrap:wrap; gap:5px; align-items:center; }
    .pill { display:inline-flex; align-items:center; min-height:20px; max-width:100%; border-radius:999px; padding:2px 7px; background:var(--panel-2); color:#4b453d; font-size:10px; font-weight:800; line-height:1.2; }
    .pill.green { background:var(--green-soft); color:#24563f; }
    .pill.amber { background:var(--amber-soft); color:#70490f; }
    .pill.red { background:var(--red-soft); color:#82352f; }
    .pill.blue { background:var(--blue-soft); color:#254f79; }
    .pill.value-new { background:#dcebe1; color:#24563f; }
    .pill.value-dup { background:#f0d7d2; color:#82352f; }
    .pill.value-similar { background:#f3e5c8; color:#70490f; }
    .match-note { margin:0; color:#6b5f52; font-size:11px; line-height:1.3; }
    .detail-body { min-height:0; overflow:auto; padding:13px 14px 16px; }
    .empty-state { min-height:220px; display:grid; place-items:center; border:1px dashed #cbbca5; border-radius:10px; background:#fbf6ed; color:var(--muted); text-align:center; padding:22px; }
    .detail-title { display:grid; gap:7px; margin-bottom:10px; }
    .detail-title h3 { margin:0; font-size:18px; line-height:1.18; overflow-wrap:anywhere; }
    .meta-line { display:flex; flex-wrap:wrap; gap:6px; align-items:center; }
    .field-label { display:block; margin:10px 0 6px; color:#4e463d; font-size:11px; font-weight:850; text-transform:uppercase; letter-spacing:.04em; }
    .actions { position:sticky; bottom:-16px; display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:8px; margin:12px -14px -16px; padding:10px 14px 14px; border-top:1px solid var(--line); background:rgba(255,253,248,.96); backdrop-filter:blur(8px); }
    .primary { background:var(--green); color:#fff; }
    .secondary { background:#fffaf2; color:var(--ink); border-color:var(--line); }
    .more { background:var(--amber); color:#fff; }
    .reject { background:var(--red); color:#fff; }
    .resume { width:100%; margin-top:9px; background:var(--ink-2); color:#fff; }
    .info-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; margin-top:12px; }
    .info-box { min-width:0; border:1px solid var(--line); border-radius:8px; background:#fffaf2; padding:9px; }
    .info-box b { display:block; color:var(--muted); font-size:11px; text-transform:uppercase; }
    .info-box span { display:block; margin-top:5px; overflow-wrap:anywhere; font-weight:750; }
    .code-block { max-height:158px; overflow:auto; white-space:pre-wrap; word-break:break-word; border:1px solid #2b352f; border-radius:8px; background:#151c18; color:#e9eee6; padding:10px; font:11px/1.45 "Cascadia Mono",Consolas,monospace; }
    .status-line { min-height:22px; margin:12px 0 0; color:var(--muted); font-size:13px; }
    .muted { color:var(--muted); }
    .loading { padding:12px; color:rgba(238,241,232,.72); font-size:13px; }
    .queue-body .loading { color:var(--muted); }
    @media (max-width:1180px) { body { overflow:auto; } .app-shell { height:auto; min-height:100vh; grid-template-columns:260px minmax(0,1fr); background:var(--paper); overflow:visible; } .sidebar { height:100vh; } .topbar { grid-template-columns:1fr; } .topbar-tools { justify-items:stretch; } .workspace { grid-template-columns:1fr; overflow:visible; } .queue-panel,.detail-panel { min-height:520px; } .toolbar { grid-template-columns:1fr 1fr; } .actions { grid-template-columns:repeat(2,minmax(0,1fr)); } }
    @media (max-width:760px) { body { overflow:auto; } .app-shell { display:block; height:auto; } .sidebar { position:relative; height:auto; min-height:auto; max-height:none; padding:18px 14px; } .topbar { grid-template-columns:1fr; padding:20px 14px 12px; } .topbar-tools { justify-items:stretch; } .workspace { padding:0 14px 18px; } .stats-grid,.info-grid,.actions,.toolbar { grid-template-columns:1fr; } .queue-panel,.detail-panel { min-height:auto; } button { width:100%; } }
    @media (prefers-reduced-motion:reduce) { * { scroll-behavior:auto !important; transition:none !important; } button:hover { transform:none; } }
  </style>
</head>
<body>
<main class="app-shell">
  <aside class="sidebar" aria-label="运行状态">
    <div class="brand"><span class="eyebrow">Dream Memory Review</span><h1>记忆审核工作台</h1><p>把模型提取的候选记忆逐条校准，再写入项目长期上下文。</p><a href="/memory-config">打开配置页面</a></div>
    <section class="side-section"><h2 class="side-title">新建 AI 提取</h2><form class="start-form" onsubmit="startAiRun(event)"><label>事件文件 <input id="startInput" value=".dream-memory/imports/all-events.jsonl" placeholder=".dream-memory/imports/all-events.jsonl" /></label><label>记忆归属 <input id="startProject" value="." placeholder="留空表示全局记忆" /></label><p class="start-hint">用于给候选记忆标记归属，不会扫描项目目录。当前项目通常保持 <code>.</code> 即可。</p><div class="quick-row"><button type="button" onclick="setProjectScope('.')">当前项目</button><button type="button" onclick="setProjectScope('')">全局记忆</button></div><div class="start-options"><label><input id="startInvoke" type="checkbox" checked /> 调用模型</label><label><span>mode</span><select id="startMode" aria-label="提取模式"><option value="ai">ai</option><option value="rules">rules</option></select></label></div><button class="start-button" type="submit">生成新候选</button></form></section>
    <section class="side-section"><h2 class="side-title">运行状态 <span id="runCount" class="pill">0</span></h2><div id="runs" class="run-list loading">正在加载运行记录...</div></section>
    <section class="side-section"><h2 class="side-title">运行进度</h2><div id="runProgress" class="loading">选择 run 后显示模型进度</div></section>
    <section class="side-section"><h2 class="side-title">审核进度</h2><div id="reviewProgress" class="loading">选择 run 后显示进度</div></section>
    <section class="side-section"><h2 class="side-title">候选汇总</h2><div id="reviewSummary" class="loading">选择 run 后显示候选分布</div></section>
    <section class="side-section"><h2 class="side-title">自动审核预览</h2><div class="start-form"><label>最低分 <input id="autoReviewMinScore" type="number" min="0" max="1" step="0.05" value="0.7" /></label><label class="check"><input id="autoReviewIncludeDuplicates" type="checkbox" /> 包含重复项</label><label class="check"><input id="autoReviewIncludeMerges" type="checkbox" /> 包含合并项</label><label class="check"><input id="autoReviewForce" type="checkbox" /> 覆盖现有 reviewed</label><div class="quick-row"><button type="button" onclick="previewAutoReview()">预览</button><button type="button" onclick="applyAutoReview()">写入 reviewed</button></div><div id="autoReviewPreview" class="loading">选择 run 后可预览自动审核影响</div></div></section>
  </aside>
  <section class="layout">
    <header class="topbar"><div><span class="eyebrow">Human-in-the-loop memory curation</span><h2>候选记忆审核</h2><p>快速筛掉噪声，补足证据，把真正有复用价值的信息沉淀下来。</p></div><div class="topbar-tools"><a class="config-button" href="/memory-config" aria-label="打开运行配置">运行配置</a><div class="toolbar" aria-label="候选过滤器"><input id="search" type="search" placeholder="搜索内容、标签、证据" /><select id="valueFilter" aria-label="价值筛选"><option value="">按价值分组</option><option value="new_value">新增价值</option><option value="existing_duplicate">已有记忆重复</option><option value="similar_existing">相似可合并</option></select><select id="statusFilter" aria-label="状态筛选"><option value="">全部状态</option><option value="promote">promote</option><option value="review">review</option><option value="reject">reject</option></select><select id="scopeFilter" aria-label="范围筛选"><option value="">全部范围</option><option value="user">user</option><option value="global">global</option><option value="project">project</option></select></div></div></header>
    <div class="workspace">
      <section class="panel queue-panel" aria-label="候选分组"><div class="panel-head"><h3>候选分组</h3><p id="queueSummary">正在加载候选项...</p><div class="stats-grid"><div class="stat"><b id="totalCount">0</b><span>Total</span></div><div class="stat"><b id="reviewCount">0</b><span>Review</span></div><div class="stat"><b id="promoteCount">0</b><span>Promote</span></div><div class="stat"><b id="rejectCount">0</b><span>Reject</span></div></div></div><div id="list" class="queue-body loading">正在加载...</div></section>
      <section class="panel detail-panel" aria-label="候选详情"><div class="panel-head"><h3>审核详情</h3><p id="status" class="status-line">选择一个候选项开始审核</p></div><div class="detail-body"><div id="emptyDetail" class="empty-state"><div><strong>还没有选择候选记忆</strong><p>从左侧队列点选一条记录，查看证据、冲突和运行 trace。</p></div></div><div id="detailContent" hidden><div class="detail-title"><h3 id="title">选择一个候选项</h3><div class="meta-line" id="meta"></div></div><label class="field-label" for="content">候选内容</label><textarea id="content" placeholder="候选内容，可编辑后批准"></textarea><label class="field-label" for="note">审核备注</label><textarea id="note" placeholder="记录保留、拒绝或合并的原因"></textarea><div class="actions"><button class="primary" onclick="submitReview('approved')">批准</button><button class="secondary" onclick="submitReview('edited_and_approved')">编辑后批准</button><button class="secondary" onclick="submitReview('merged')">合并</button><button class="more" onclick="submitReview('needs_more_evidence')">需要更多证据</button><button class="reject" onclick="submitReview('rejected')">拒绝</button></div><button class="resume" onclick="resumeSelectedRun()">恢复并应用 Run</button><div class="info-grid"><div class="info-box"><b>Candidate ID</b><span id="candidateId">-</span></div><div class="info-box"><b>Suggested Action</b><span id="suggestedAction">-</span></div></div><label class="field-label">Dream Analysis</label><pre id="dreamAnalysis" class="code-block"></pre><label class="field-label">Quality Signals</label><pre id="qualitySignals" class="code-block"></pre><label class="field-label">Evidence</label><pre id="evidence" class="code-block"></pre><label class="field-label">Run Trace</label><pre id="trace" class="code-block"></pre></div></div></section>
    </div>
  </section>
</main>
<script>
let candidates=[];let selected=null;let selectedRunId=null;let latestProgress=null;let startDefaultsLoaded=false;
function setProjectScope(value){document.getElementById('startProject').value=value;}
async function loadStartDefaults(){try{const res=await fetch('/api/memory/config');const data=await res.json();if(!res.ok)throw new Error(JSON.stringify(data));const config=data.config||{};document.getElementById('startInput').value=config.default_input||'.dream-memory/imports/all-events.jsonl';document.getElementById('startProject').value=config.default_project==null?'.':config.default_project;document.getElementById('startInvoke').checked=Boolean(config.invoke_model);document.getElementById('startMode').value=config.mode||'ai';startDefaultsLoaded=true;}catch(error){document.getElementById('reviewProgress').innerHTML=`<div class="loading">配置加载失败，已使用默认参数：${escapeHtml(error.message)}</div>`;}}
async function startAiRun(event){event.preventDefault();if(!startDefaultsLoaded){await loadStartDefaults();}const input=document.getElementById('startInput').value.trim();const project=document.getElementById('startProject').value.trim()||null;const invokeModel=document.getElementById('startInvoke').checked;const mode=document.getElementById('startMode').value||'ai';if(!input){document.getElementById('reviewProgress').innerHTML='<div class="loading">请先填写事件文件路径</div>';return;}const button=event.submitter;button.disabled=true;button.textContent='生成中...';try{const res=await fetch('/api/memory/runs/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({input,project,mode,invoke_model:invokeModel})});const data=await res.json();if(!res.ok){throw new Error(JSON.stringify(data));}document.getElementById('reviewProgress').innerHTML=`<div class="loading">已创建 run：${escapeHtml(data.run_id)}</div>`;await loadRuns();await selectRun(data.run_id);}catch(error){document.getElementById('reviewProgress').innerHTML=`<div class="loading">创建失败：${escapeHtml(error.message)}</div>`;}finally{button.disabled=false;button.textContent='生成新候选';}}
async function loadRuns(){try{const res=await fetch('/api/memory/runs');const data=await res.json();const runs=data.runs||[];document.getElementById('runCount').textContent=runs.length;const box=document.getElementById('runs');box.innerHTML=runs.slice(0,8).map(r=>runHtml(r)).join('')||'<div class="loading">暂无 run</div>';if(selectedRunId){await refreshSelectedRun();}}catch(error){document.getElementById('runs').innerHTML=`<div class="loading">运行记录加载失败：${escapeHtml(error.message)}</div>`;}}
function runHtml(r){const active=selectedRunId===r.run_id?' active':'';return `<button class="run-card${active}" onclick="selectRun(${jsArg(r.run_id)})"><span class="run-id">${escapeHtml(r.run_id)}</span><span class="run-meta"><span class="pill green">${escapeHtml(r.status||'unknown')}</span><span class="pill">${escapeHtml(r.phase||'phase')}</span></span><span class="run-id muted">${escapeHtml(formatDate(r.updated_at))}</span></button>`;}
async function selectRun(runId){selectedRunId=runId;selected=null;syncDetail();document.getElementById('status').textContent=`当前 run: ${runId}`;await refreshSelectedRun();await loadRuns();renderList();}
async function refreshSelectedRun(){if(!selectedRunId)return;const state=await loadRunState(selectedRunId);const trace=await loadTraceData(selectedRunId);renderRunProgress(state,trace);if(state.status==='waiting_review'||state.status==='completed'){await loadReviewQueue(selectedRunId);await loadReviewProgress();await loadReviewSummary();renderList();}else{candidates=[];renderList();document.getElementById('reviewProgress').innerHTML='<div class="loading">候选生成完成后显示审核进度</div>';}if(selected){await loadTrace();}}
async function loadRunState(runId){const res=await fetch(`/api/memory/runs/${runId}`);if(!res.ok)throw new Error(`run 状态加载失败：${res.status}`);return await res.json();}
async function loadTraceData(runId){const res=await fetch(`/api/memory/runs/${runId}/trace`);if(!res.ok)return[];const data=await res.json();return data.trace||[];}
function renderRunProgress(state,trace){const status=state.status||'unknown';const phase=state.phase||'unknown';const counts=state.counts||{};const attemptEvents=trace.filter(row=>String(row.event_type||'').startsWith('model_attempt_'));const latestAttempt=attemptEvents[attemptEvents.length-1];const pct=runPercent(status,phase,trace);const steps=[['queued','已创建 run',trace.some(e=>e.event_type==='run_queued')||status!=='created'],['events','已复制事件',trace.some(e=>e.event_type==='events_copied')],['model','模型提取候选',trace.some(e=>e.event_type==='ai_extraction_complete')],['review','等待审核',status==='waiting_review'||status==='completed'],['done','已应用',status==='completed']];const stepHtml=steps.map(([key,label,done])=>`<div class="progress-step ${done?'done':currentStepClass(key,status,phase,trace)}"><span class="progress-dot"></span><span>${escapeHtml(label)}</span></div>`).join('');const attemptHtml=attemptEvents.slice(-4).map(row=>attemptLine(row)).join('')||'<span>模型调用尚未开始</span>';const error=state.error?`<div class="progress-error">${escapeHtml(state.error)}</div>`:'';document.getElementById('runProgress').innerHTML=`<div class="progress-card"><div class="progress-title"><strong>${escapeHtml(statusLabel(status,phase))}</strong><span>${pct}%</span></div><div class="progress-bar" aria-label="运行进度"><span style="width:${pct}%"></span></div><div>${escapeHtml(state.model||'')}</div><div class="progress-steps">${stepHtml}</div><div class="progress-log">${attemptHtml}${error}</div><div>候选：<strong>${counts.candidate_count||0}</strong> · 待审核：<strong>${counts.review_count||0}</strong></div><div>送模：<strong>${counts.prompt_event_count??'-'}</strong> / 原始：<strong>${counts.input_event_count??'-'}</strong> · 过滤：<strong>${counts.filtered_prompt_event_count??0}</strong></div></div>`;}
function currentStepClass(key,status,phase,trace){if(status==='failed')return'';if(key==='model'&&phase==='extracting')return'active';if(key==='review'&&phase==='review')return'active';if(key==='queued'&&status==='queued')return'active';if(key==='events'&&trace.some(e=>e.event_type==='events_copied')&&!trace.some(e=>e.event_type==='model_attempt_started'))return'active';return'';}
function runPercent(status,phase,trace){if(status==='completed')return 100;if(status==='waiting_review')return 82;if(status==='failed')return 100;if(trace.some(e=>e.event_type==='ai_extraction_complete'))return 72;if(trace.some(e=>e.event_type==='model_attempt_started'))return 45;if(trace.some(e=>e.event_type==='events_copied'))return 28;if(status==='queued')return 12;return 5;}
function statusLabel(status,phase){if(status==='waiting_review')return'等待审核';if(status==='completed')return'已应用';if(status==='failed')return'运行失败';if(phase==='extracting')return'模型提取中';if(status==='queued')return'已排队';return `${status} / ${phase}`;}
function attemptLine(row){const payload=row.payload||{};const event=row.event_type||'';const attempt=payload.attempt||'-';const elapsed=payload.elapsed_ms?`${Math.round(payload.elapsed_ms/1000)}s`:'';if(event==='model_attempt_started')return`<span>第 ${escapeHtml(attempt)} 次调用开始 · ${escapeHtml(payload.model||'')}</span>`;if(event==='model_attempt_succeeded')return`<span>第 ${escapeHtml(attempt)} 次调用成功 ${escapeHtml(elapsed)}</span>`;if(event==='model_attempt_failed')return`<span class="progress-error">第 ${escapeHtml(attempt)} 次失败 ${escapeHtml(elapsed)} · ${escapeHtml(payload.error_kind||payload.error||'')}</span>`;return`<span>${escapeHtml(event)}</span>`;}
async function loadReviewQueue(runId){const res=await fetch(`/api/memory/runs/${runId}/review-queue`);const data=await res.json();const items=data.items||[];if(items.length){candidates=items.map(item=>Object.assign({},item.candidate||{},{conflicts:item.conflicts||[],quality_signals:item.quality_signals||{},dream_analysis:item.dream_analysis||{},review_queue_status:item.status,suggested_action:item.suggested_action,candidate_id:item.candidate_id}));return;}const fallback=await fetch(`/api/memory/runs/${runId}/candidates`);const fallbackData=await fallback.json();candidates=fallbackData.candidates||[];}
async function loadReviewProgress(){if(!selectedRunId)return;const res=await fetch(`/api/memory/runs/${selectedRunId}/review-progress`);const data=await res.json();latestProgress=data;const actions=data.actions||{};const total=data.total||0;const reviewed=data.reviewed||0;const pct=total?Math.round((reviewed/total)*100):0;document.getElementById('reviewProgress').innerHTML=`<div class="progress"><div><strong>${reviewed}</strong> / ${total} reviewed</div><div>Pending: <strong>${data.pending||0}</strong> · Source: ${escapeHtml(data.source||'-')}</div><div>Approved: ${actions.approved||0} &middot; Rejected: ${actions.rejected||0} &middot; Needs evidence: ${actions.needs_more_evidence||0}</div><div>Suggested: ${escapeHtml(JSON.stringify(data.suggested_actions||{}))}</div><div aria-label="progress" style="height:7px;border-radius:999px;background:rgba(255,255,255,.14);overflow:hidden;"><div style="height:100%;width:${pct}%;background:#cfe2cf;"></div></div></div>`;}
async function loadReviewSummary(){if(!selectedRunId)return;const res=await fetch(`/api/memory/runs/${selectedRunId}/review-summary`);const data=await res.json();if(!res.ok){document.getElementById('reviewSummary').innerHTML=`<div class="loading">汇总加载失败：${escapeHtml(JSON.stringify(data))}</div>`;return;}const s=data.summary||{};document.getElementById('reviewSummary').innerHTML=`<div class="progress"><div>Total: <strong>${s.total||0}</strong> · Manual: <strong>${s.needs_manual_count||0}</strong></div><div>新增价值: <strong>${s.new_value_count||0}</strong> · 已有记忆重复: <strong>${s.existing_duplicate_count||s.duplicate_count||0}</strong></div><div>Duplicates: ${s.duplicate_count||0} · Conflicts: ${s.conflict_count||0} · Low score: ${s.low_score_count||0}</div><div>Value: ${escapeHtml(JSON.stringify(s.by_value_class||{}))}</div><div>Actions: ${escapeHtml(JSON.stringify(s.by_suggested_action||{}))}</div><div>Evidence: ${escapeHtml(JSON.stringify(s.by_evidence_quality||{}))}</div><div>Score: ${escapeHtml(String(s.score_min??'-'))}–${escapeHtml(String(s.score_max??'-'))} avg ${escapeHtml(String(s.score_avg??'-'))}</div></div>`;}
async function loadTrace(){if(!selectedRunId)return;const candidateQuery=selected?`?candidate_id=${encodeURIComponent(selected.id)}`:'';const res=await fetch(`/api/memory/runs/${selectedRunId}/trace${candidateQuery}`);const data=await res.json();document.getElementById('trace').textContent=JSON.stringify(data.trace||[],null,2);}
async function loadCandidates(){const res=await fetch('/api/memory/candidates');const data=await res.json();candidates=data.candidates||[];renderList();}
function valueClass(c){const q=c.quality_signals||{};if(q.value_class)return q.value_class;if(q.duplicate)return'existing_duplicate';if(q.matched_memory_id)return'similar_existing';return'new_value';}
function valueLabel(value){return {new_value:'新增价值',existing_duplicate:'已有记忆重复',similar_existing:'相似可合并',unknown:'未分类'}[value]||value;}
function valueTone(value){if(value==='new_value')return'value-new';if(value==='existing_duplicate')return'value-dup';if(value==='similar_existing')return'value-similar';return'';}
function groupCandidates(items){return items.reduce((groups,candidate)=>{const key=valueClass(candidate)||'unknown';if(!groups[key])groups[key]=[];groups[key].push(candidate);return groups;},{});}
function candidateHtml(c){const q=c.quality_signals||{};const vc=valueClass(c);const conflictCount=(c.conflicts||[]).length;const conflict=conflictCount?`<span class="pill red">conflicts ${conflictCount}</span>`:'';const suggestion=c.suggested_action?`<span class="pill amber">${escapeHtml(c.suggested_action)}</span>`:'';const matched=q.matched_memory_summary?`<p class="match-note">已有：${escapeHtml(truncate(q.matched_memory_summary,88))}</p>`:'';const score=c.score==null?'-':c.score;return `<button class="candidate-card ${selected&&selected.id===c.id?'active':''}" onclick="selectCandidate(${jsArg(c.id)})"><header><strong>${escapeHtml(c.type||'memory')}</strong><span class="pill ${statusTone(c.status)}">${escapeHtml(c.status||'unknown')}</span></header><p class="candidate-content">${escapeHtml(truncate(c.content||'',150))}</p>${matched}<div class="candidate-footer"><span class="pill ${valueTone(vc)}">${escapeHtml(valueLabel(vc))}</span><span class="pill blue">${escapeHtml(c.scope||'scope')}</span><span class="pill">score ${escapeHtml(String(score))}</span>${suggestion}${conflict}</div></button>`;}
function renderList(){const q=document.getElementById('search').value.toLowerCase();const vf=document.getElementById('valueFilter').value;const sf=document.getElementById('statusFilter').value;const scf=document.getElementById('scopeFilter').value;const list=document.getElementById('list');const filtered=candidates.filter(c=>(!vf||valueClass(c)===vf)&&(!sf||c.status===sf)&&(!scf||c.scope===scf)&&(!q||JSON.stringify(c).toLowerCase().includes(q)));updateStats(filtered);const groups=groupCandidates(filtered);const order=['new_value','similar_existing','existing_duplicate','unknown'];const html=order.filter(key=>groups[key]&&groups[key].length).map(key=>`<div class="group-title">${escapeHtml(valueLabel(key))} (${groups[key].length})</div>`+groups[key].map(candidateHtml).join('')).join('');list.innerHTML=html||'<div class="empty-state">没有匹配的候选项</div>';}
async function selectCandidate(id){selected=candidates.find(c=>c.id===id);syncDetail();await loadTrace();renderList();}
function syncDetail(){const hasSelection=Boolean(selected);document.getElementById('emptyDetail').hidden=hasSelection;document.getElementById('detailContent').hidden=!hasSelection;if(!selected){document.getElementById('title').textContent='选择一个候选项';document.getElementById('meta').innerHTML='';document.getElementById('content').value='';document.getElementById('note').value='';document.getElementById('candidateId').textContent='-';document.getElementById('suggestedAction').textContent='-';document.getElementById('dreamAnalysis').textContent='';document.getElementById('qualitySignals').textContent='';document.getElementById('evidence').textContent='';return;}const vc=valueClass(selected);document.getElementById('title').textContent=`${selected.type||'memory'} / ${selected.id}`;document.getElementById('meta').innerHTML=[`价值:${valueLabel(vc)}`,selected.scope,selected.project,selected.status,selected.review_queue_status].filter(Boolean).map(value=>`<span class="pill ${String(value).startsWith('价值:')?valueTone(vc):''}">${escapeHtml(value)}</span>`).join('');document.getElementById('content').value=selected.content||'';document.getElementById('candidateId').textContent=selected.id||'-';document.getElementById('suggestedAction').textContent=selected.suggested_action||'-';document.getElementById('dreamAnalysis').textContent=JSON.stringify(selected.dream_analysis||{},null,2);document.getElementById('qualitySignals').textContent=JSON.stringify(selected.quality_signals||{},null,2);document.getElementById('evidence').textContent=JSON.stringify({evidence:selected.evidence||[],conflicts:selected.conflicts||[]},null,2);}
async function submitReview(action){if(!selected)return;const payload={candidate_id:selected.id,action,edited_content:document.getElementById('content').value,note:document.getElementById('note').value,reviewer:'user',candidate:selected};const url=selectedRunId?`/api/memory/runs/${selectedRunId}/review`:'/api/memory/review';const res=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const data=await res.json();document.getElementById('status').textContent=res.ok?`已保存审核结果：${action}`:JSON.stringify(data);if(selectedRunId)await loadReviewProgress();await loadTrace();}
function autoReviewRequest(){return{min_score:Number(document.getElementById('autoReviewMinScore').value||0.7),include_duplicates:document.getElementById('autoReviewIncludeDuplicates').checked,include_merges:document.getElementById('autoReviewIncludeMerges').checked,force:document.getElementById('autoReviewForce').checked};}
function renderAutoReviewPreview(data){const reasons=data.skip_reasons||{};const reasonHtml=Object.keys(reasons).sort().map(key=>`<div>${escapeHtml(key)}: <strong>${escapeHtml(String(reasons[key]))}</strong></div>`).join('')||'<div>无跳过项</div>';const rows=(data.preview||[]).slice(0,6).map(row=>`<div class="run-card"><span class="run-id">${escapeHtml(row.candidate_id||'-')}</span><span>${escapeHtml(row.decision||'-')} · ${escapeHtml(row.reason||'-')} · score ${escapeHtml(String(row.dream_score??'-'))}</span></div>`).join('');document.getElementById('autoReviewPreview').innerHTML=`<div class="progress"><div>Decisions: <strong>${data.decision_count||0}</strong> · Skipped: <strong>${data.skipped||0}</strong></div>${reasonHtml}${rows}</div>`;}
async function previewAutoReview(){if(!selectedRunId){document.getElementById('autoReviewPreview').innerHTML='<div class="loading">请先选择 run</div>';return;}const res=await fetch(`/api/memory/runs/${selectedRunId}/auto-review/preview`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(autoReviewRequest())});const data=await res.json();if(!res.ok){document.getElementById('autoReviewPreview').innerHTML=`<div class="loading">预览失败：${escapeHtml(JSON.stringify(data))}</div>`;return;}renderAutoReviewPreview(data);}
async function applyAutoReview(){if(!selectedRunId){document.getElementById('autoReviewPreview').innerHTML='<div class="loading">请先选择 run</div>';return;}const res=await fetch(`/api/memory/runs/${selectedRunId}/auto-review`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(autoReviewRequest())});const data=await res.json();if(!res.ok){document.getElementById('autoReviewPreview').innerHTML=`<div class="loading">写入失败：${escapeHtml(JSON.stringify(data))}</div>`;return;}renderAutoReviewPreview(data);await loadReviewProgress();await loadTrace();}
async function resumeSelectedRun(){if(!selectedRunId)return;const res=await fetch(`/api/memory/runs/${selectedRunId}/resume`,{method:'POST'});const data=await res.json();document.getElementById('status').textContent=res.ok?`已恢复并应用：${data.status}`:JSON.stringify(data);await loadRuns();await loadTrace();}
function updateStats(items){const counts=items.reduce((acc,item)=>{acc[item.status||'unknown']=(acc[item.status||'unknown']||0)+1;return acc;},{});document.getElementById('totalCount').textContent=items.length;document.getElementById('reviewCount').textContent=counts.review||0;document.getElementById('promoteCount').textContent=counts.promote||0;document.getElementById('rejectCount').textContent=counts.reject||0;const runLabel=selectedRunId?` &middot; ${selectedRunId}`:'';document.getElementById('queueSummary').innerHTML=`${items.length} 条候选项${runLabel}`;}
function statusTone(status){if(status==='promote')return'green';if(status==='review')return'amber';if(status==='reject')return'red';return'';}
function truncate(text,limit){const value=String(text||'');return value.length>limit?`${value.slice(0,limit)}...`:value;}
function formatDate(value){if(!value)return'';const date=new Date(value);if(Number.isNaN(date.getTime()))return value;return date.toLocaleString('zh-CN',{hour12:false});}
function escapeHtml(s){return String(s).replace(/[&<>"]/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[ch]));}
function jsArg(value){return escapeHtml(JSON.stringify(String(value||'')));}
['search','valueFilter','statusFilter','scopeFilter'].forEach(id=>document.addEventListener('input',e=>{if(e.target&&e.target.id===id)renderList();}));
loadStartDefaults();loadRuns();loadCandidates();setInterval(loadRuns,3000);
</script>
</body>
</html>
"""


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
        decision = "skip"
        reason = "requires_manual_review"
        if not candidate:
            reason = "missing_candidate"
        elif quality.get("duplicate") and not include_duplicates:
            reason = "duplicate"
        elif suggested in {"create", "merge"} and score_value < min_score:
            reason = "below_min_score"
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
    queue_items = _read_jsonl_dicts(queue_path) if queue_path.is_file() else []
    candidates = _read_jsonl_dicts(candidates_path) if candidates_path.is_file() else []
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
    reviewed = _read_jsonl_dicts(reviewed_path)
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
        return MEMORY_REVIEW_HTML

    @app.get("/memory-config", response_class=HTMLResponse)
    def memory_config_page() -> HTMLResponse:
        return HTMLResponse(
            MEMORY_CONFIG_HTML,
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
        items = _read_jsonl_dicts(queue_path) if queue_path.is_file() else []
        return {"run_id": run_id, "count": len(items), "items": items}

    @app.get("/api/memory/runs/{run_id}/review-summary")
    def memory_run_review_summary(run_id: str) -> dict[str, Any]:
        try:
            state = load_run_state(memory_dir, run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        queue_path = Path(str(state.get("artifacts", {}).get("review_queue_path") or ""))
        items = _read_jsonl_dicts(queue_path) if queue_path.is_file() else []
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
        queue = _read_jsonl_dicts(queue_path) if queue_path.is_file() else []
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
        queue = _read_jsonl_dicts(queue_path) if queue_path.is_file() else []
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
