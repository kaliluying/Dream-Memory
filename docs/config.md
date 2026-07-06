# Configuration

默认配置文件：`.dream-memory/config.json`。

`api_key_env` 必须是环境变量名，例如 `OPENAI_API_KEY`，不要把真实 API key 写进配置文件。

```bash
export OPENAI_API_KEY="..."
uv run dream-memory check-provider
```

## Model Profiles

模型配置支持旧版 flat 字段，也支持新版命名 profiles。旧字段仍可继续使用：

```json
{
  "provider": "anthropic",
  "model": "claude-sonnet-4-6",
  "api_key_env": "ANTHROPIC_API_KEY",
  "base_url": null,
  "timeout_seconds": 60
}
```

如果没有配置 `models`，系统会把上面的字段自动归一化成 `default` profile。

新版配置可以显式声明多个 profile 和 fallback chain：

```json
{
  "models": {
    "primary": {
      "provider": "anthropic",
      "model": "claude-sonnet-4-6",
      "api_key_env": "ANTHROPIC_API_KEY",
      "base_url": null,
      "timeout_seconds": 60
    },
    "openai_backup": {
      "provider": "openai",
      "model": "gpt-4.1",
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

## Retry And Fallback

- `max_attempts` 控制单个 profile 的最大尝试次数。
- `initial_delay_seconds`、`backoff_factor` 和 `max_delay_seconds` 控制有界指数退避。
- `retry_on_status` 默认覆盖 `429/500/502/503/504`。
- `retry_on_timeout` 为 `true` 时会重试超时类错误。
- 当前 profile 尝试失败后，runtime 会按 `fallback_chain` 切到下一个 profile。
- `allow_rules_fallback` 默认关闭；规则模式仍推荐显式使用 `--mode rules`。

## Diagnostics

```bash
uv run dream-memory check-provider
uv run dream-memory check-provider --all
uv run dream-memory check-provider --profile primary
```

持久化 run 会把模型调用事件写入 `.dream-memory/runs/{run_id}/trace.jsonl`，包括 `model_attempt_started`、`model_attempt_failed`、`model_attempt_succeeded`、`model_fallback_used` 和 `model_runtime_failed`。
