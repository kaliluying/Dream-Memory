# Architecture

Dream Memory 是 CLI-first 的本地长期记忆系统：

- `memory_agent.py`：AI prompt、模型调用结果解析和候选校验
- `model_providers.py`：模型供应商适配层
- `memory_runs.py`：可恢复 run 状态与 trace
- `memory_dreaming.py`：候选、审核、应用、上下文生成
- `memory_export.py`：导出到 AGENTS.md / CLAUDE.md 和所有项目总览
- `web.py`：轻量审核 UI/API
