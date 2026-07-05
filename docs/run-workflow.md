# Resumable Run Workflow

`run` 会创建 `.dream-memory/runs/{run_id}/state.json` 和 `trace.jsonl`，并在 `waiting_review` 阶段暂停。

审核完成后执行：

```bash
uv run dream-memory resume --run-id <run_id>
```
