# 评估结果状态与项目指令证据设计

## 目标

在不降低普通记忆两事件门禁的前提下，让明确项目指令和结构化项目标记使用单事件例外，并让评估集能够区分候选是可审核、待补证据、已拒绝还是完全未抽取。

## 范围

- `project_instruction` 和 `project_markers` 中由现有规则识别出的项目级长期规则视为明确指令。
- 普通历史消息、普通偏好和行为推断继续要求两个不同的有效 `event_id`。
- 现有 precision、recall、F1 和 deferred 指标保持兼容。
- 不增加配置项、依赖、数据库或新的运行模式。
- 本轮只增强当前 16 条基准集的表达能力，不扩展为大型语料库。

## 方案

### 明确项目指令

在 `extract_atomic_facts()` 的现有项目规则分支中增加 `explicit` 标签：

- 明确指定 Python / 前端包管理器的项目指令；
- `project_markers` 解析出的包管理器、测试框架和 Web 框架事实。

只修改这些已存在的规则分支，不根据任意项目文本自动推断 `explicit`。`analyze_dream_candidate()` 继续复用现有逻辑：`explicit_instruction` 需要一个有效事件，普通候选需要两个。

### 结果状态标注

JSONL 行可选增加：

```json
{"expected_outcomes":["reviewable"]}
```

允许的值固定为：

- `reviewable`：分析结果为 `create`、`review` 或 `merge`；
- `deferred`：分析结果为 `needs_more_evidence`；
- `rejected`：分析结果为 `reject`；
- `none`：没有产生候选。

`expected_outcomes` 是去重后的集合语义，不要求候选顺序，也不取代现有 `expected` 记忆内容标注。未提供该字段的旧数据不参与状态准确率计算。

### 评估数据流

每行候选经过现有 Dream Analysis 后，同时产生两类结果：

1. `reviewable` 候选继续进入现有 precision / recall / F1 计算；
2. 所有分析结果归一化为状态集合，与可选的 `expected_outcomes` 比较。

当没有候选时，实际状态为 `["none"]`。非法的 `expected_outcomes` 值直接抛出 `ValueError`，避免静默接受错误标注。

评估报告新增：

- `outcome_checked_rows`；
- `outcome_correct_rows`；
- `outcome_accuracy`；
- `outcome_mismatches`，最多保留前 20 条，包含行号、行 ID、期望状态和实际状态。

### 基准集更新

优先为以下高价值行增加状态标注：

- 单事件普通偏好：`deferred`；
- 同一事件重复偏好：`deferred`；
- 两个独立事件普通偏好：`reviewable`；
- 明确项目包管理器指令：`reviewable`；
- 结构化项目标记：`reviewable`；
- 一次性任务、内部上下文和凭据位置噪声：根据实际安全路径标注为 `rejected` 或 `none`。

仓库示例和包内示例必须保持逐字一致。

## 兼容性

- 没有 `expected_outcomes` 的现有 JSONL 继续按原逻辑评估。
- CLI 输出只新增字段，不删除或重命名现有字段。
- `auto-review`、review queue 和正式记忆写入逻辑不变。
- 固定证据门禁不新增用户配置。

## 测试

- 项目明确指令单事件可进入 review queue。
- `project_markers` 单事件可进入 review queue。
- 普通历史偏好单事件仍为 `needs_more_evidence`。
- 状态归一化覆盖 `reviewable`、`deferred`、`rejected` 和 `none`。
- 非法状态标注抛出明确错误。
- 缺少状态标注的旧评估数据保持兼容。
- 仓库示例与包内示例一致。
- 全量测试、规则评估和 `git diff --check` 通过。

## 验收标准

- 当前 16 条规则评估 precision 保持 `1.000`，false positives 保持 `0`。
- 项目指令相关的 4 个 false negatives 消失，预计 true positives 从 `8` 提升到 `12`。
- 单事件普通偏好和重复同一事件偏好仍为 `deferred`。
- 所有带 `expected_outcomes` 的行状态匹配，`outcome_accuracy` 为 `1.000`。
- 不新增依赖、配置或通用抽象。
