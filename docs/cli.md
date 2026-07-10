# Dream Memory CLI

主命令：`dream-memory`。

## 推荐流程

```bash
uv run dream-memory init
uv run dream-memory import all --output-dir .dream-memory/imports --dry-run
uv run dream-memory import all --output-dir .dream-memory/imports
uv run dream-memory run --input .dream-memory/imports/all-events.jsonl --project . --mode rules
uv run dream-memory status
uv run dream-memory review-summary --run-id <run_id>
uv run dream-memory resume --run-id <run_id>
uv run dream-memory export --target both --project .
```

## 证据门禁与人工审核

普通记忆候选需要两个不同的有效 `event_id` 才会进入 review queue。带 `explicit` 标签的明确长期指令只需要一个有效 `event_id`。证据不足的候选仍保留在 candidate 和 `DREAMS.md` 产物中；后续 `sync` 导入新的独立证据后会重新评估。

`auto-review` 和 `sync --auto` 不会批准 `create`、`review` 或 `merge`，分数和 `--include-merges` / `--include-review` 都不会绕过人工门禁。它们只兼容处理旧队列中的 `reject` 和 `needs_more_evidence` 决策。

AI 模式的 `dream` / `run` 会同时输出 prompt 输入过滤统计：`input_event_count`、`prompt_event_count`、`filtered_prompt_event_count`，并在持久化 run 的 `state.json` / `trace.jsonl` 中保留，便于确认真实送模内容是否被噪声污染。eval 报告的每行 `extractions` 也会透传这些字段。

如果配置里已经设置了 `default_input` 和 `default_project`，`dream` / `run` / `pipeline` 可以省略 `--input` / `--project`；如果配置里设置了 `extract_input` 和 `extract_project`，`extract-facts` 也可以省略对应参数：

```bash
uv run dream-memory dream --mode rules
uv run dream-memory run --mode rules
uv run dream-memory pipeline --mode rules
uv run dream-memory extract-facts
```

如果要直接初始化某个 memory 目录，而不是 `PATH/.dream-memory`，使用 `--output-dir`。该目录会自包含 `config.json`、`memory_cards.jsonl`、`imports/`、`runs/` 和 `examples/`：

```bash
uv run dream-memory init --output-dir /tmp/dream-memory-workspace
uv run dream-memory eval \
  --input /tmp/dream-memory-workspace/examples/labeled-events.jsonl \
  --output /tmp/dream-memory-workspace/eval.json
```

## 上下文与缺省记忆文件

`context` 会把缺失的 `memory_cards.jsonl` 当作空记忆集合处理，便于新项目或临时目录先跑通命令：

```bash
uv run dream-memory context \
  --project . \
  --memory-cards .dream-memory/memory_cards.jsonl \
  --limit 12 \
  --format markdown \
  --task "跑测试并验证"
```

## 评估抽取质量

规则模式 baseline：

```bash
uv run dream-memory eval \
  --input examples/labeled-events.jsonl \
  --project . \
  --mode rules \
  --output .dream-memory/eval.rules.json
```

真实模型模式：

```bash
uv run dream-memory eval \
  --input examples/labeled-events.jsonl \
  --project . \
  --mode ai \
  --timeout-seconds 20 \
  --max-attempts 1 \
  --continue-on-error \
  --output .dream-memory/eval.ai.json
```

`examples/labeled-events.jsonl` 是维护中的 16 行基准集，覆盖正例、噪声、重复、敏感凭据位置、跨项目隔离、跨项目用户偏好，以及单事件、重复事件和双事件普通偏好的证据门禁。模型服务不稳定时，可以保留 AI 错误并用规则抽取兜底。报告中的 `extraction_success_count`、`extraction_error_count`、`fallback_count` 和 `fallback_empty_count` 用来区分纯 AI 成功与规则兜底成功；`raw_candidate_count` 表示单行模型原始候选数，`scored_candidate_count` 表示经过 Dream Analysis 后真正计入 precision 的候选数；顶层 `raw_candidate_total` / `fallback_candidate_total` / `scored_candidate_total` 汇总全量原始候选、规则兜底候选和计分候选，`deferred_candidate_count` 记录因独立证据不足而延迟的候选：

```bash
uv run dream-memory eval \
  --input examples/labeled-events.jsonl \
  --project . \
  --mode ai \
  --timeout-seconds 8 \
  --max-attempts 1 \
  --continue-on-error \
  --fallback-rules-on-error \
  --fallback-rules-on-empty \
  --output .dream-memory/eval.ai-with-rules-fallback.json
```
