# Configuration

默认配置文件：`.dream-memory/config.json`。

`api_key_env` 必须是环境变量名，例如 `OPENAI_API_KEY`，不要把真实 API key 写进配置文件。

```bash
export OPENAI_API_KEY="..."
uv run dream-memory check-provider
```
