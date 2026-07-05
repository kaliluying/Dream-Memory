# Dream Memory CLI

主命令：`dream-memory`。

推荐流程：

```bash
uv run dream-memory init
uv run dream-memory import all --output-dir .dream-memory/imports --dry-run
uv run dream-memory run --input .dream-memory/imports/all-events.jsonl --project .
uv run dream-memory status
uv run dream-memory resume --run-id <run_id>
uv run dream-memory export --target both --project .
```
