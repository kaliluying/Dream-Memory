# DeepAgents 项目建议

## 推荐方向

我最推荐先做一个 **AI 项目研发助手 / 代码任务代理**，而不是一开始做通用聊天机器人或复杂多智能体平台。

这个方向非常适合 DeepAgents，因为它天然需要：

- 长任务规划
- 代码库上下文读取
- 文件系统中间产物管理
- 多步骤执行与验证
- 子 Agent 分工
- 最终交付总结

一句话定位：

> 输入一个需求或 GitHub Issue，Agent 自动读项目、拆任务、生成计划、修改代码、运行测试，并输出交付总结。

---

## 为什么推荐这个项目

相比其他方向，它更适合作为 DeepAgents 的第一个项目：

1. **结果容易验证**  
   代码是否改对、测试是否通过、构建是否成功，都可以客观判断。

2. **能体现 DeepAgents 的优势**  
   它不是一次性问答，而是一个需要规划、执行、检查和迭代的长任务。

3. **MVP 范围可控**  
   第一版可以只做“读代码 + 生成计划”，不用一开始就自动改代码。

4. **后续商业化空间大**  
   可以逐步扩展成类似研发 Copilot、Issue 修复助手、PR 助手、代码审查助手的产品。

5. **适合你后续反复和我协作迭代**  
   你可以进入这个文件夹后，让我逐步帮你搭项目结构、写代码、接入工具和完善 Agent 流程。

---

## MVP 版本设计

### V1：本地 CLI 计划生成器

目标：输入项目路径和自然语言需求，输出实现计划。

功能：

- 扫描项目目录
- 识别技术栈
- 阅读关键文件
- 生成任务拆解
- 输出修改建议、风险点和验收清单

暂时不自动改代码，降低风险。

示例输入：

```bash
python main.py --project ./my-app --task "增加用户权限管理模块"
```

示例输出：

- 项目技术栈判断
- 相关文件列表
- 推荐修改路径
- 实现步骤
- 测试建议
- 风险提示

---

### V2：半自动 Patch 生成器

目标：Agent 不直接改主项目，而是生成 patch 或建议修改文件。

功能：

- 基于 V1 的计划生成代码变更
- 输出 diff
- 用户确认后再应用
- 保存 agent 的中间分析文件

这个阶段重点是安全性和可控性。

---

### V3：自动验证与自我修复

目标：让 Agent 能运行测试并根据错误继续修复。

功能：

- 自动运行 lint
- 自动运行 test
- 自动运行 build
- 读取错误日志
- 二次修改代码
- 输出最终验证报告

---

### V4：Web 控制台

目标：把 CLI 能力产品化。

页面模块：

- 任务输入区
- 执行计划展示
- 实时日志
- 文件改动 Diff
- 测试结果
- 最终报告

---

### V5：GitHub 工作流集成

目标：对接真实研发流程。

功能：

- 读取 GitHub Issue
- 创建修复分支
- 自动提交代码
- 创建 Draft PR
- 生成 PR 描述
- 根据 Review 继续修改

---

## 推荐技术栈

### 后端

优先推荐：

- Python
- FastAPI
- DeepAgents
- LangChain / LangGraph

原因：DeepAgents 和 LangChain 生态结合更自然，Python 也更适合快速做 Agent 原型。

### 前端

可选：

- Next.js
- Vue 3
- React + Vite

如果你只是先做 MVP，可以暂时不做前端，直接从 CLI 开始。

### 存储

MVP 阶段：

- 本地文件系统
- SQLite

产品化阶段：

- Postgres
- Redis
- 对象存储，例如 S3 / MinIO

### 队列

长任务建议后续加入：

- Celery
- RQ
- Dramatiq
- 或者基于 LangGraph 的持久执行机制

### 观测与调试

建议接入：

- LangSmith
- OpenTelemetry
- 自定义任务日志

Agent 项目非常依赖可观测性，否则很难排查为什么某一步做错。

---

## Agent 设计建议

### 主 Agent

负责整体调度：

- 理解用户需求
- 制定执行计划
- 决定调用哪些工具
- 汇总结果
- 判断是否需要继续修复

### 工具层

建议先实现这些工具：

1. `list_files`：列出项目文件
2. `read_file`：读取文件内容
3. `search_code`：搜索代码
4. `write_file`：写入文件
5. `run_command`：执行测试或构建命令
6. `create_patch`：生成代码 diff
7. `summarize_changes`：总结修改内容

工具权限一定要分级，尤其是写文件和执行命令。

### 子 Agent

不要一开始就上很多子 Agent。建议从三个角色开始：

1. **Code Reader**  
   负责理解代码结构和现有实现模式。

2. **Implementation Agent**  
   负责根据计划修改代码。

3. **Reviewer Agent**  
   负责检查变更风险、测试覆盖和潜在 bug。

---

## 安全边界

这是这个项目最重要的部分之一。

建议规则：

- 默认只读项目文件
- 写入前生成 diff
- 执行命令需要白名单
- 禁止默认执行危险命令，例如 `rm -rf`、强制 reset、上传密钥等
- 所有工具调用都要记录日志
- 每次任务生成最终审计报告

如果之后要接入真实代码仓库，必须加入权限控制和人工确认流程。

---

## 项目目录建议

可以先这样组织：

```text
deepagent-memory/
  README.md
  docs/
    product-spec.md
    architecture.md
    agent-design.md
    roadmap.md
  src/
    agents/
    tools/
    workflows/
    storage/
  examples/
  tests/
```

后续如果你决定正式开始，我建议下一步创建：

```text
deepagent-dev-assistant/
  README.md
  pyproject.toml
  src/
    main.py
    agents/
    tools/
    prompts/
    workflows/
  workspace/
  outputs/
  tests/
```

---

## 第一阶段任务清单

建议你接下来按这个顺序做：

1. 确定产品名称
2. 创建 Python 项目骨架
3. 实现项目目录扫描工具
4. 实现文件读取工具
5. 实现代码搜索工具
6. 接入 DeepAgents 主流程
7. 让 Agent 输出实现计划
8. 保存每次任务的中间文件
9. 加入简单 CLI
10. 用一个真实项目测试效果

---

## 第一个 Demo 场景

建议第一个 Demo 做这个：

> 输入一个本地项目路径和一句需求，Agent 自动分析项目，并输出一份实现计划。

例如：

```text
需求：给这个 Vue 项目增加一个用户登录页面。
```

Agent 输出：

```text
1. 项目使用 Vue 3 + Vite
2. 路由定义在 src/router/index.ts
3. 页面组件位于 src/views
4. 建议新增 src/views/LoginView.vue
5. 建议新增登录 API 封装 src/api/auth.ts
6. 建议修改路由守卫
7. 验收方式：访问 /login，输入账号密码，成功后跳转首页
```

这个 Demo 小而完整，很适合验证 DeepAgents 的基础能力。

---

## 我建议的最终目标

最终可以把它做成：

> 一个面向开发者的 DeepAgents 研发助手：从需求到代码、测试、PR，总结全流程自动化。

它可以逐步演进成：

- 本地代码助手
- GitHub Issue 修复助手
- PR 生成助手
- 自动测试修复助手
- 团队研发自动化平台

---

## 下一步建议

如果你接下来进入这个文件夹继续和我对话，我建议直接让我做这几件事之一：

1. “帮我初始化这个项目的 Python 代码结构”
2. “帮我写 DeepAgents 项目的技术方案”
3. “帮我实现第一个 CLI MVP”
4. “帮我设计 Agent tools”
5. “帮我把这个项目做成 Web 产品”

我最建议从第 3 个开始：

> 帮我实现第一个 CLI MVP：输入项目路径和需求，输出代码修改计划。
---

## 当前 CLI MVP 用法

本仓库已经实现 V1 本地 CLI 计划生成器。

后端使用 `uv` 管理 Python 环境、lockfile 和 CLI 命令。

初始化/同步环境：

```bash
uv sync
```

运行测试：

```bash
uv run python -m unittest discover -s tests
```

直接运行：

```bash
uv run python main.py --project ./my-app --task "增加用户权限管理模块"
```

或使用安装后的 CLI 命令：

```bash
uv run deepagent-advice --project ./my-app --task "增加用户权限管理模块"
```

可选保存单个 Markdown 输出：

```bash
uv run deepagent-advice --project ./my-app --task "增加登录页面" --output outputs/login-plan.md
```

推荐保存一次任务运行产物，便于后续审计和接入 Agent 工作流：

```bash
uv run deepagent-advice --project ./my-app --task "增加登录页面" --output-dir outputs/runs
```

生成半自动 Patch 建议（只生成 diff，不修改目标项目）：

```bash
uv run deepagent-advice --project ./my-app --task "更新 README 说明" --output-dir outputs/runs --patch
```

使用模型生成 Patch（显式调用模型，需配置 API Key）：

```bash
uv run deepagent-advice --project ./my-app --task "更新 README 说明" --output-dir outputs/runs --runtime deepagents --model anthropic:claude-sonnet-4-6 --invoke-model --model-patch
```

运行验证并保存验证报告：

```bash
uv run deepagent-advice --project ./my-app --task "验证项目" --output-dir outputs/runs --verify --verify-command "python -m unittest discover -s tests"
```

启动 Web 控制台：

```bash
uv run uvicorn deepagent_memory.web:app --reload
```

记忆人工审核界面：

```text
http://127.0.0.1:8000/memory-review
```

生成 GitHub Draft PR 工作流计划（dry-run，不直接创建分支或 PR）：

```bash
uv run deepagent-advice --project ./my-app --task "修复登录错误" --output-dir outputs/runs --github-issue owner/repo#123
```

显式执行 GitHub 工作流（需本机已登录 `gh`，且只有传入 `--github-apply` 才执行）：

```bash
uv run deepagent-advice --project ./my-app --task "修复登录错误" --output-dir outputs/runs --github-issue owner/repo#123 --github-apply
```

也可以输出 JSON，便于脚本、Web 控制台或 Agent 工作流直接消费：

```bash
uv run deepagent-advice --project ./my-app --task "增加登录页面" --format json
uv run deepagent-advice --project ./my-app --task "增加登录页面" --format json --output outputs/login-analysis.json
```

输出 Agent 编排结果：

```bash
uv run deepagent-advice --project ./my-app --task "增加登录页面" --format json --workflow
```

使用官方 DeepAgents runtime（默认 dry-run，只构建官方 graph，不调用模型）：

```bash
uv run deepagent-advice --project ./my-app --task "增加登录页面" --format json --runtime deepagents
```

实际调用模型需要配置对应 provider API Key，并显式传入：

```bash
uv run deepagent-advice --project ./my-app --task "增加登录页面" --format json --runtime deepagents --model anthropic:claude-sonnet-4-6 --invoke-model
```

Claude Code 风格交互模式：

```bash
uv run deepagent-code --project ./my-app
```

交互命令：

```text
/help
/project ./other-app
/run 增加登录页面
/patch 更新 README
/verify 验证项目 :: python -m unittest discover -s tests
/runs
/exit
```

该命令会创建带时间戳的运行目录，并写入：

- `plan.md`：实现计划
- `metadata.json`：项目路径、需求、创建时间、技术栈、相关文件、推荐命令、风险数量等结构化元数据
- `runs.db`：SQLite 运行索引，记录 run_dir、task、status 和关键产物路径


共享记忆导入（Claude Code / Codex 会话扫描与标准化）：

```bash
uv run deepagent-memory scan --project .
uv run deepagent-memory import codex --dry-run --output-dir .deepagent/memory/imports
uv run deepagent-memory import claude --dry-run --project . --output-dir .deepagent/memory/imports
uv run deepagent-memory import all --dry-run --project . --output-dir .deepagent/memory/imports
```

导入器会统一输出 normalized session events，并对 token/key/secret/password/cookie/auth 等敏感字段脱敏。

当前能力：

- 使用 `uv` 管理后端 Python 环境、lockfile 和 CLI 运行命令
- 零第三方运行时依赖，直接使用 Python 标准库运行
- 扫描项目目录并过滤常见依赖/构建产物目录
- 识别 Python、Node.js、Vue、React、Next.js、Vite、TypeScript 等技术栈线索
- 技术栈识别优先基于配置文件、依赖声明和实际源码文件，避免 README 泛化描述导致误判
- 根据需求关键词选择相关文件
- 输出实现步骤、推荐修改路径、测试建议和风险提示
- 支持按运行目录保存 `plan.md` 和 `metadata.json`，为后续 Agent 审计日志做准备
- `metadata.json` 现在包含 `stack`、`relevant_files`、`suggested_commands`、`risk_count` 等结构化字段
- 支持 `--format json` 输出完整结构化分析结果，包含 Markdown report 和可机器读取字段
- 初步实现工具层：`list_files`、`read_file`、`search_code`、`write_file`、`run_command`、`create_patch`、`summarize_changes`
- 支持 `--patch` 在运行产物目录中生成 `suggested.patch` 和 `patch-summary.json`，默认不修改目标项目
- 支持 `--model-patch` 通过官方 DeepAgents runtime 调用模型生成可审阅 unified diff
- 提供 `deepagent-memory`，可扫描/导入 Codex 与 Claude Code 会话为共享记忆事件
- 支持 `--verify` 运行白名单验证命令并保存 `verification.json`，记录 stdout/stderr/退出码和汇总状态
- 提供 FastAPI Web 控制台，包含任务输入、计划展示、Diff、测试结果和最终报告区域
- 支持 GitHub Issue 到 Draft PR 的 dry-run 工作流计划，生成 `github-workflow.json`
- 支持显式 `--github-apply` 执行 GitHub 工作流命令，并保存 `github-execution.json`
- 运行产物目录写入 `audit.log`，记录分析、patch、验证和 GitHub 工作流生成事件
- 本地 SQLite `runs.db` 自动索引每次运行产物，Web/API 可查询历史任务
- 实现主 Agent 工作流编排，包含 Code Reader、Implementation Agent、Reviewer Agent 三个角色
- 提供 `deepagent-code` 交互式命令，支持连续输入任务、切换项目、生成 patch、运行验证和查看历史记录
- 接入官方 `deepagents.create_deep_agent` runtime，包含工具和 Code Reader / Implementation / Reviewer subagents

## Dream Memory Workflow

The Dream Memory pipeline keeps imported Claude Code / Codex session events as evidence, extracts atomic facts, proposes candidates, and requires explicit human review before formal memories are projected into `MEMORY.md`. Use `uv` for the Python CLI commands:

```bash
uv run deepagent-memory scan --output .deepagent/memory/scan.json
uv run deepagent-memory import all --output-dir .deepagent/memory/imports --dry-run
uv run deepagent-memory extract-facts --input .deepagent/memory/imports/all-events.jsonl --project . --output-dir .deepagent/memory
uv run deepagent-memory dream --input .deepagent/memory/imports/all-events.jsonl --project . --output-dir .deepagent/memory
uv run deepagent-memory review --candidates .deepagent/memory/candidates.jsonl --memory-cards .deepagent/memory/memory_cards.jsonl --output-dir .deepagent/memory
uv run deepagent-memory apply --reviewed .deepagent/memory/reviewed.jsonl --memory-cards .deepagent/memory/memory_cards.jsonl --output-dir .deepagent/memory --reviewer user
uv run deepagent-memory context --project . --memory-cards .deepagent/memory/memory_cards.jsonl --limit 12
uv run deepagent-memory context --project . --memory-cards .deepagent/memory/memory_cards.jsonl --limit 12 --format markdown

# Shortcut for extract-facts -> dream -> review:
uv run deepagent-memory pipeline --input .deepagent/memory/imports/all-events.jsonl --project . --output-dir .deepagent/memory
```

`facts.jsonl`, `candidates.jsonl`, `review_queue.jsonl`, `review_decisions.jsonl`, `memory_cards.jsonl`, and the derived `MEMORY.md` are written under `.deepagent/memory/`. Sensitive content, raw tool output, build logs, and project-state records are allowed only as source evidence and are not promoted into formal memory cards automatically.


Review submissions from the web UI are apply-compatible: approved or edited candidates are normalized into review decisions with `memory_updates`, while `review_decisions.jsonl` remains an append-only ledger. `context` excludes other projects' project-scoped cards and can render Markdown for direct agent prompt injection.
