# Dream Memory

Dream Memory 是一个本地 Dream Memory / 梦境记忆系统，用来从 Claude Code / Codex 会话材料中导入事件、生成候选记忆、人工审核，并把正式记忆注入后续 AI 上下文。

## 功能

AI 抽取路径由 LangGraph 状态图编排：`build_prompt -> invoke_model -> validate_candidates`。


- 导入 Claude Code / Codex 会话事件
- 默认使用 AI 生成候选记忆
- 保留规则模式作为 fallback/debug：`--mode rules`
- 对候选记忆做安全过滤、schema 校验、证据校验和去重
- 通过 review queue 进行人工审核
- 将审核后的正式记忆写入 `memory_cards.jsonl`
- 派生生成可读的 `MEMORY.md`
- 按项目生成 AI 可用上下文
- 提供 FastAPI 审核页面 `/memory-review`

## 安装与运行

```bash
uv sync
uv run dream-memory init
uv run dream-memory --help
```


## 轻量 Provider 配置

项目使用 LangGraph 编排 AI 候选记忆抽取流程，并通过轻量 provider 层调用模型。当前 provider 支持：

- `anthropic`
- `openai`
- `openrouter`

## 配置文件

默认配置路径是 `.dream-memory/config.json`。先生成可编辑配置：

```bash
uv run dream-memory init-config
```

也可以指定位置：

```bash
uv run dream-memory --config ./memory-config.json init-config --output ./memory-config.json
```

配置示例：

```json
{
  "provider": "anthropic",
  "model": "claude-sonnet-4-6",
  "api_key_env": "ANTHROPIC_API_KEY",
  "base_url": null,
  "timeout_seconds": 60,
  "invoke_model": true,
  "mode": "ai",
  "output_dir": ".dream-memory",
  "imports_output_dir": ".dream-memory/imports",
  "memory_cards": ".dream-memory/memory_cards.jsonl",
  "context_limit": 12,
  "context_format": "json",
  "auto_export": false,
  "export_target": "both",
  "export_scope": "project",
  "export_output_dir": null,
  "codex_home": "~/.codex",
  "claude_home": "~/.claude",
  "claude_state": "~/.claude.json"
}
```

命令行参数优先级高于配置文件。例如临时切模型：

```bash
uv run dream-memory --config .dream-memory/config.json pipeline \
  --input .dream-memory/imports/all-events.jsonl \
  --provider openai \
  --model gpt-4.1
```

检查 provider 配置：

```bash
uv run dream-memory check-provider
```

API Key 仍通过环境变量配置，例如：

```bash
export ANTHROPIC_API_KEY="你的 key"
# 或
export OPENAI_API_KEY="你的 key"
```

## 常用流程

```bash
# 1. 扫描可用来源
uv run dream-memory scan --output .dream-memory/scan.json

# 2. 导入事件
uv run dream-memory import all --output-dir .dream-memory/imports --dry-run

# 3. 默认 AI 梦境抽取：调用模型生成候选记忆
uv run dream-memory dream   --input .dream-memory/imports/all-events.jsonl   --project .   --output-dir .dream-memory

# 4. 调用模型生成候选记忆
uv run dream-memory dream   --input .dream-memory/imports/all-events.jsonl   --project .   --output-dir .dream-memory   --invoke-model

# 5. 如需规则 fallback/debug
uv run dream-memory dream   --input .dream-memory/imports/all-events.jsonl   --project .   --output-dir .dream-memory   --mode rules

# 6. 生成审核队列
uv run dream-memory review   --candidates .dream-memory/ai-candidates.jsonl   --memory-cards .dream-memory/memory_cards.jsonl   --output-dir .dream-memory

# 7. 应用人工审核结果
uv run dream-memory apply   --reviewed .dream-memory/reviewed.jsonl   --memory-cards .dream-memory/memory_cards.jsonl   --output-dir .dream-memory   --reviewer user

# 8. 生成 AI 上下文
uv run dream-memory context   --project .   --memory-cards .dream-memory/memory_cards.jsonl   --limit 12   --format markdown
```

## 一键流程

```bash
uv run dream-memory pipeline   --input .dream-memory/imports/all-events.jsonl   --project .   --output-dir .dream-memory
```

`pipeline` 默认会调用 AI 生成候选记忆；如果只想生成 `ai-prompt.md` 而不调用模型，添加 `--dry-run`。规则 fallback 可使用 `--mode rules`。


## 可恢复 Run 工作流

除了单次 `pipeline`，项目现在支持持久化 run：每次 run 会生成 `run_id`，状态写入 `.dream-memory/runs/{run_id}/state.json`，trace 写入 `trace.jsonl`，候选详情写入 `candidates/{candidate_id}.json`。

```bash
# 创建可恢复 run，执行到 waiting_review 后暂停
uv run dream-memory run \
  --input .dream-memory/imports/all-events.jsonl \
  --project .

# 查看全部 run
uv run dream-memory status

# 查看单个 run
uv run dream-memory status --run-id <run_id>

# 人工审核后恢复并 apply
uv run dream-memory resume --run-id <run_id>

# 查看 run trace
uv run dream-memory trace --run-id <run_id>

# 查看候选记忆 lineage
uv run dream-memory trace --run-id <run_id> --candidate-id <candidate_id>
```

Web API 也支持 run 状态查询：

- `GET /api/memory/runs`
- `GET /api/memory/runs/{run_id}`
- `GET /api/memory/runs/{run_id}/trace`
- `GET /api/memory/runs/{run_id}/candidates`
- `POST /api/memory/runs/{run_id}/review`


Web 审核页 `/memory-review` 会轮询 run 状态并展示最近 run，选择 run 后可查看该 run 的候选记忆、审核进度和 trace。候选会按状态分组，审核结果会写入 run 专属的 `reviewed.jsonl`。Web API 也提供 `POST /api/memory/runs/start` 用于启动 run，`POST /api/memory/runs/{run_id}/resume` 用于审核后恢复并应用正式记忆。


## 导出给 Codex / Claude Code

项目级上下文默认导出到当前项目下的 `AGENTS.md` / `CLAUDE.md`，只包含当前项目记忆、用户级记忆和全局记忆，不会注入其他项目的 project memory。

```bash
# 导出当前项目上下文给 Codex 和 Claude Code
uv run dream-memory export --target both --project .

# 只导出给 Codex
uv run dream-memory export --target codex --project .

# 只导出给 Claude Code
uv run dream-memory export --target claude --project .
```

导出会替换 `<!-- DREAM_MEMORY_START -->` 和 `<!-- DREAM_MEMORY_END -->` 之间的内容，保留文件其他部分。

## 所有项目总览

如果想查看所有项目的正式记忆，使用 summary，而不是把所有项目注入单个项目上下文：

```bash
uv run dream-memory summary --scope all-projects --output .dream-memory/PROJECTS.md
```

## 输出文件

运行产物默认在 `.dream-memory/` 下生成：

- `facts.jsonl`：规则模式下的 atomic facts，或安全/审计辅助文件
- `ai-prompt.md`：AI 候选记忆抽取 prompt
- `ai-candidates.jsonl`：AI 候选记忆
- `candidates.jsonl`：规则 fallback 候选记忆
- `review_queue.jsonl`：待人工审核队列
- `reviewed.jsonl`：Web/人工审核提交
- `review_decisions.jsonl`：审核决策流水账
- `memory_cards.jsonl`：正式记忆卡片状态
- `MEMORY.md`：正式记忆的人类可读投影

`.dream-memory/` 下的运行产物已被 `.gitignore` 忽略，只保留 `.gitkeep`。

## Web 审核界面

```bash
uv run uvicorn dream_memory.web:app --reload
```

打开：

```text
http://127.0.0.1:8000/memory-review
```

## 更多文档

- `docs/architecture.md`
- `docs/cli.md`
- `docs/config.md`
- `docs/run-workflow.md`

## 当前源码结构

```text
src/dream_memory/
├── __init__.py
├── memory_agent.py       # AI prompt、模型输出解析、候选校验
├── memory_cli.py         # dream-memory CLI
├── memory_dreaming.py    # 安全过滤、候选/审核/应用/context 核心逻辑
├── memory_export.py      # AGENTS.md / CLAUDE.md 导出和全项目 summary
├── memory_importers.py   # Claude/Codex 会话导入
├── memory_models.py      # 记忆数据结构 builders
└── web.py                # Web 审核 UI/API
```

## 评估抽取质量

```bash
uv run dream-memory eval --input examples/labeled-events.jsonl --output .dream-memory/eval.json
```

## 测试

```bash
uv run --with pytest pytest -v
```
