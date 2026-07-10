# Dream Memory

Dream Memory 是一个本地 Dream Memory / 梦境记忆系统，用来从 Claude Code / Codex 会话材料中导入事件、生成候选记忆、人工审核，并把正式记忆注入后续 AI 上下文。

## 功能

AI 抽取路径现在由 `memory_agent.py` 直接执行：`build prompt -> invoke model/runtime -> validate candidates`。


- 导入 Claude Code / Codex 会话事件
- 默认使用 AI 生成候选记忆
- 保留规则模式作为 fallback/debug：`--mode rules`
- 对候选记忆做安全过滤、schema 校验、证据校验和去重
- 通过 review queue 进行人工审核
- 支持 auto-review 预览和安全决策草稿，并输出 skip reasons 解释为什么跳过
- 将审核后的正式记忆写入 `memory_cards.jsonl`
- 派生生成可读的 `MEMORY.md`
- 按项目生成 AI 可用上下文，并返回 relevance diagnostics 解释召回原因
- 提供 FastAPI 审核页面 `/memory-review`

普通候选需要来自两个不同 `event_id` 的证据才会进入 review queue；带 `explicit` 标签的明确长期指令只需要一个有效 `event_id`。未达门槛的候选仍保留在 candidate 和 `DREAMS.md` 产物中，后续 `sync` 导入新的独立证据后可重新评估。

`auto-review` 和 `sync --auto` 不会批准 `create`、`review` 或 `merge`。正式记忆的创建和合并始终需要人工审核；自动流程只兼容处理旧队列中的拒绝和证据不足决策。

## 安装与运行

```bash
uv sync
uv run dream-memory init
uv run dream-memory --help
```

如需直接初始化一个自包含 memory 目录：

```bash
uv run dream-memory init --output-dir /tmp/dream-memory-workspace
uv run dream-memory eval \
  --input /tmp/dream-memory-workspace/examples/labeled-events.jsonl \
  --project /tmp/project \
  --output /tmp/dream-memory-workspace/eval.json
```

`--output-dir` 生成的配置会指向该目录内的 `imports/`、`memory_cards.jsonl` 和评估样例；设置好 `default_input` 后，`dream` / `run` / `pipeline` 可省略 `--input`，设置好 `extract_input` 后 `extract-facts` 可省略 `--input`。


## 轻量 Provider 配置

项目通过轻量 provider/runtime 层直接调用模型并校验 AI 候选记忆。当前 provider 支持：

- `anthropic`
- `openai`
- `openrouter`

模型层使用命名 profiles、重试和 fallback chain。配置文件必须声明 `models` 和 `model_policy`；旧版顶层 `provider/model/api_key/api_key_env/base_url/timeout_seconds` 写法不再兼容。

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
  "claude_state": "~/.claude.json",
  "models": {
    "primary": {
      "provider": "anthropic",
      "model": "claude-sonnet-4-6",
      "api_key": "your-anthropic-api-key",
      "api_key_env": null,
      "base_url": null,
      "timeout_seconds": 60
    },
    "openai_backup": {
      "provider": "openai",
      "model": "gpt-4.1",
      "api_key": "your-openai-api-key",
      "api_key_env": null,
      "base_url": null,
      "timeout_seconds": 45
    }
  },
  "model_policy": {
    "default_profile": "primary",
    "fallback_chain": ["primary", "openai_backup"],
    "retry": {
      "max_attempts": 3,
      "initial_delay_seconds": 1.0,
      "backoff_factor": 2.0,
      "max_delay_seconds": 8.0,
      "retry_on_status": [429, 500, 502, 503, 504],
      "retry_on_timeout": true
    },
    "allow_rules_fallback": false
  }
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
uv run dream-memory check-provider --all
uv run dream-memory check-provider --profile primary
```

`check-provider --all` 会检查所有 configured profiles。`dream` / `pipeline` / `run` 会按 `model_policy.fallback_chain` 尝试模型；单个 profile 内会按 retry policy 对 `429/500/502/503/504` 和超时做有界指数退避重试。持久化 run 会把模型尝试、失败、成功和 fallback 事件写入 `trace.jsonl`。

API Key 主用法是直接写在 `.dream-memory/config.json` 的 profile 里：

```json
"api_key": "你的 key"
```

`check-provider` 只输出 `api_key_configured` / `api_key_present`，不会打印明文 key。`api_key_env` 仍可作为可选兜底。

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

# 8. 生成 AI 上下文；--task 会按任务意图重排并输出 diagnostics
uv run dream-memory context   --project .   --memory-cards .dream-memory/memory_cards.jsonl   --limit 12   --format markdown   --task "跑测试并验证"
```

## 一键流程

```bash
uv run dream-memory pipeline   --input .dream-memory/imports/all-events.jsonl   --project .   --output-dir .dream-memory
```

`pipeline` 默认会调用 AI 生成候选记忆；如果只想生成 `ai-prompt.md` 而不调用模型，添加 `--dry-run`。规则 fallback 可使用 `--mode rules`。


## 可恢复 Run 工作流

除了单次 `pipeline`，项目现在支持持久化 run：每次 run 会生成 `run_id`，状态写入 `.dream-memory/runs/{run_id}/state.json`，trace 写入 `trace.jsonl`，候选详情写入 `candidates/{candidate_id}.json`。

AI run 会在输出、`state.json` 和 `trace.jsonl` 中记录 prompt 输入过滤统计：`input_event_count` 是原始事件数，`prompt_event_count` 是实际进入模型 prompt 的事件数，`filtered_prompt_event_count` 是被噪声过滤或超出 preview limit 的事件数，用于审计真实模型输入质量；eval 报告也会在每行 `extractions` 中透传这些字段。

```bash
# 创建可恢复 run，执行到 waiting_review 后暂停
uv run dream-memory run \
  --input .dream-memory/imports/all-events.jsonl \
  --project .

# 查看全部 run
uv run dream-memory status

# 查看单个 run
uv run dream-memory status --run-id <run_id>

# 可选：先看候选分布，判断这批队列是否值得逐条审核
uv run dream-memory review-summary --run-id <run_id>

# 可选：先预览自动审核影响，不写 reviewed.jsonl、不改 run state
uv run dream-memory auto-review --run-id <run_id> --min-score 0.7 --dry-run
# 生成安全 reviewed 草稿；create/review/merge 只报告 requires_manual_review
uv run dream-memory auto-review --run-id <run_id> --min-score 0.7
# 如果 reviewed.jsonl 已存在，auto-review 默认拒绝覆盖；确认要重写时添加 --force
# 默认跳过重复候选；如需写入重复候选的 rejected 决策，添加 --include-duplicates
# --include-merges / --include-review 为兼容参数，不会绕过人工审核门禁
# skip_reasons 会解释跳过原因，例如 duplicate / requires_manual_review

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
- `GET /api/memory/runs/{run_id}/review-queue`
- `GET /api/memory/runs/{run_id}/review-progress`
- `GET /api/memory/runs/{run_id}/review-summary`
- `POST /api/memory/runs/{run_id}/review`
- `POST /api/memory/runs/{run_id}/auto-review/preview`
- `POST /api/memory/runs/{run_id}/auto-review`


Web 审核页 `/memory-review` 会轮询 run 状态并展示最近 run，选择 run 后会优先读取该 run 的 `review_queue.jsonl`，展示候选记忆、冲突信息、审核建议、审核进度、候选汇总、Dream Analysis、Quality Signals 和 trace。候选会按状态分组；候选汇总会按 suggested action、类型、scope、证据质量、重复数、冲突数、低分数和分数区间给出总览，便于先判断这批候选是否值得逐条审核。审核结果会写入 run 专属的 `reviewed.jsonl`。页面内置自动审核预览：可以先查看自动拒绝、证据不足和必须人工审核的候选、skip reasons、分数和原因；只有点击写入 reviewed 才会落盘，且默认不覆盖已有 `reviewed.jsonl`，需要勾选覆盖才会重写。Web API 也提供 `POST /api/memory/runs/start` 用于启动 run，`POST /api/memory/runs/{run_id}/resume` 用于审核后恢复并应用正式记忆。


## 上下文召回诊断

`context` 会在 JSON 输出中附带 `diagnostics`，用于解释每条记忆为什么被排到前面：

- `relevance`：最终相关性分数
- `token_score`：任务文本和记忆文本的直接 token overlap
- `intent_score`：短指令/意图别名命中的 boost，例如“跑测试”命中 pytest 流程
- `matched_tokens`：直接命中的 token
- `reason`：排序原因，常见值包括 `intent_alias_match`、`token_overlap`、`scope_fallback`、`default_scope_order`

Markdown 输出会在每条记忆后追加 `rank_reason` 和 `relevance`，方便后续 agent 判断当前上下文是否召回正确。

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
- `reviewed.jsonl`：Web/人工审核提交；`auto-review` 只会写入安全的拒绝或证据不足决策，默认不会覆盖已有文件
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

审核页左侧包含候选汇总和自动审核预览区：

- 候选汇总调用 `/review-summary`，展示动作分布、类型分布、证据质量、重复/冲突/低分统计和分数区间
- “预览”只调用 `/auto-review/preview`，不会写入文件
- “写入 reviewed”调用 `/auto-review`，默认不覆盖已有审核文件
- 如需覆盖已有草稿或人工审核文件，先勾选“覆盖现有 reviewed”
- “包含重复项”可允许自动写入重复候选的拒绝决策；“包含合并项”不会绕过人工合并门禁

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

规则抽取 baseline：

```bash
uv run dream-memory eval \
  --input examples/labeled-events.jsonl \
  --project /tmp/project \
  --mode rules \
  --output .dream-memory/eval.rules.json
```

真实模型抽取对比：

```bash
uv run dream-memory eval \
  --input examples/labeled-events.jsonl \
  --project /tmp/project \
  --mode ai \
  --timeout-seconds 20 \
  --max-attempts 1 \
  --continue-on-error \
  --output .dream-memory/eval.ai.json
```

当前维护的 `examples/labeled-events.jsonl` 覆盖 16 行评估样本，包含用户偏好、项目事实、产品方向、人工审核门禁、失败教训、一次性任务噪声、内部上下文噪声、重复记忆、rejected option、凭据位置噪声、跨项目隔离，以及单事件、重复事件和双事件普通偏好的证据门禁。自包含初始化会把示例评估项目设为 `/tmp/project`，因此 `dream-memory --config <dir>/config.json eval` 可直接复现基准结果。

评估 JSONL 可选使用 `expected_outcomes` 验证候选状态，允许值为 `reviewable`、`deferred`、`rejected` 和 `none`。报告中的 `outcome_checked_rows`、`outcome_correct_rows`、`outcome_accuracy` 和 `outcome_mismatches` 用于检查状态标注，不替代原有内容匹配指标。

`project_instruction` 和结构化 `project_markers` 属于明确项目指令，单个有效事件即可进入人工审核；普通偏好仍需要两个不同的有效 `event_id`。

如果模型服务不稳定，可显式记录 AI 失败并回退到规则抽取，报告会同时输出 `extraction_success_count`、`extraction_error_count`、`fallback_count` 和 `fallback_empty_count`，避免把规则兜底误读成纯 AI 效果。AI 评估报告中的 `raw_candidate_count` 是单行模型原始候选数，`scored_candidate_count` 是经过 Dream Analysis 后真正计入 precision 的候选数；顶层 `raw_candidate_total` / `fallback_candidate_total` / `scored_candidate_total` 汇总全量原始候选、规则兜底候选和计分候选，`deferred_candidate_count` 记录因独立证据不足而延迟的候选：

```bash
uv run dream-memory eval \
  --input examples/labeled-events.jsonl \
  --project /tmp/project \
  --mode ai \
  --timeout-seconds 8 \
  --max-attempts 1 \
  --continue-on-error \
  --fallback-rules-on-error \
  --fallback-rules-on-empty \
  --output .dream-memory/eval.ai-with-rules-fallback.json
```

## 测试

```bash
uv run pytest -v
```
