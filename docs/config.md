# Configuration

默认配置文件：`.dream-memory/config.json`。

主用法是在每个模型 profile 中配置 `api_key_env`，并把真实 key 放在本地 shell 环境变量中。`api_key` 保留为空字符串；只有临时本机调试时才建议直写 key。

## Run Defaults

`default_input` 和 `default_project` 是 `dream` / `run` / `pipeline` 的默认事件输入和项目范围；`extract_input` 和 `extract_project` 是 `extract-facts` 的默认输入和项目范围。配置后可以直接运行：

```bash
uv run dream-memory dream --mode rules
uv run dream-memory run --mode rules
uv run dream-memory pipeline --mode rules
uv run dream-memory extract-facts
```

`init --output-dir <dir>` 会把常用路径写成该目录内的自包含绝对路径，包括 `default_input`、`extract_input`、`imports_output_dir`、`memory_cards` 和 `eval_input`。自包含示例的 `eval_project` 固定为 `/tmp/project`，用于匹配内置 labeled eval 样本。

## Model Profiles

模型配置必须使用命名 profiles，并显式声明 fallback chain。旧版顶层 `provider/model/api_key/api_key_env/base_url/timeout_seconds` 写法不再兼容：

```json
{
  "models": {
    "primary": {
      "provider": "anthropic",
      "model": "claude-sonnet-4-6",
      "api_key": "",
      "api_key_env": "ANTHROPIC_API_KEY",
      "base_url": null,
      "timeout_seconds": 60
    },
    "openai_backup": {
      "provider": "openai",
      "model": "gpt-4.1",
      "api_key": "",
      "api_key_env": "OPENAI_API_KEY",
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

使用前在本地 shell 设置对应环境变量，例如：

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
```

## Retry And Fallback

- `max_attempts` 控制单个 profile 的最大尝试次数。
- `initial_delay_seconds`、`backoff_factor` 和 `max_delay_seconds` 控制有界指数退避。
- `retry_on_status` 默认覆盖 `429/500/502/503/504`。
- `retry_on_timeout` 为 `true` 时会重试超时类错误。
- 当前 profile 尝试失败后，runtime 会按 `fallback_chain` 切到下一个 profile。
- `allow_rules_fallback` 默认关闭；规则模式仍推荐显式使用 `--mode rules`。

## Evaluation Defaults

`eval` 可以从配置文件读取默认输入、项目、模式、输出和容错参数，适合在 Web 配置页里保存一套固定评估口径：

```json
{
  "eval_input": "examples/labeled-events.jsonl",
  "eval_project": ".",
  "eval_mode": "rules",
  "eval_output": ".dream-memory/eval.rules.json",
  "eval_max_rows": null,
  "eval_max_attempts": null,
  "eval_continue_on_error": false,
  "eval_fallback_rules_on_error": false,
  "eval_fallback_rules_on_empty": false
}
```

- `eval_max_attempts` 只覆盖评估命令的模型重试次数，不修改全局 `model_policy.retry.max_attempts`。
- `eval_continue_on_error` 会把模型错误写入评估报告并继续后续行。
- `eval_fallback_rules_on_error` 会在 AI 调用失败时用规则抽取兜底，并增加 `fallback_count`。
- `eval_fallback_rules_on_empty` 会在 AI 成功但没有候选时用规则抽取兜底，并增加 `fallback_empty_count`。
- AI run 的 prompt 输入过滤统计包括 `input_event_count`、`prompt_event_count` 和 `filtered_prompt_event_count`，可用于确认真实模型输入中有多少事件被噪声过滤或 preview limit 截断。AI eval 每行也会透传 prompt 输入过滤统计，便于对比每条样本真实送模内容。
- 判断纯 AI 效果时应同时查看 `extraction_success_count`、`extraction_error_count`、`fallback_count` 和 `fallback_empty_count`，不要只看最终 precision / recall / f1。
- AI 评估每行还会记录 `raw_candidate_count` 和 `scored_candidate_count`：前者是单行模型返回的原始候选数，后者是过滤 reject / 敏感证据后真正计入 precision 的候选数；顶层 `raw_candidate_total` / `fallback_candidate_total` / `scored_candidate_total` 汇总全量原始候选、规则兜底候选和计分候选。
- 仓库自带的 `examples/labeled-events.jsonl` 当前覆盖 13 行样本，包括偏好、项目事实、产品方向、审核门禁、失败教训、一次性噪声、内部上下文噪声、重复记忆、rejected option、凭据位置噪声、跨项目项目事实隔离和跨项目用户偏好保留。

## Diagnostics

```bash
uv run dream-memory check-provider
uv run dream-memory check-provider --all
uv run dream-memory check-provider --profile primary
```

`check-provider` 只输出 `api_key_configured` / `api_key_present`，不会打印明文 key。

持久化 run 会把模型调用事件写入 `.dream-memory/runs/{run_id}/trace.jsonl`，包括 `model_attempt_started`、`model_attempt_failed`、`model_attempt_succeeded`、`model_fallback_used` 和 `model_runtime_failed`。
