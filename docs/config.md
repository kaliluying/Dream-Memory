# Configuration

默认配置文件：`.dream-memory/config.json`。

主用法是在每个模型 profile 中直接配置 `api_key`。`api_key_env` 仍可作为可选兜底，但默认示例和 `init-config` 都使用 `api_key`。

## Model Profiles

模型配置必须使用命名 profiles，并显式声明 fallback chain。旧版顶层 `provider/model/api_key/api_key_env/base_url/timeout_seconds` 写法不再兼容：

```json
{
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

`check-provider` 只输出 `api_key_configured` / `api_key_present`，不会打印明文 key。

持久化 run 会把模型调用事件写入 `.dream-memory/runs/{run_id}/trace.jsonl`，包括 `model_attempt_started`、`model_attempt_failed`、`model_attempt_succeeded`、`model_fallback_used` 和 `model_runtime_failed`。
