# Evaluation Outcome Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让明确项目指令使用单事件证据例外，并让评估报告验证每行候选的 reviewable、deferred、rejected 或 none 状态。

**Architecture:** 复用现有 `explicit` 标签和 Dream Analysis，不增加新的证据策略层。评估器在现有候选分析后归一化状态集合，同时保留原有 precision、recall、F1 和 deferred 统计。

**Tech Stack:** Python 3.11+、标准库、`unittest`、JSONL、`uv`。

## Global Constraints

- Windows PowerShell 命令必须显式设置 UTF-8。
- 所有 Python 命令必须通过 `uv run` 执行。
- 不增加依赖、配置项、数据库、后台进程或新的运行模式。
- 普通候选仍要求两个不同的有效 `event_id`。
- 仅现有 `project_instruction` / `project_markers` 长期规则分支增加 `explicit`。
- `expected_outcomes` 允许值固定为 `reviewable`、`deferred`、`rejected`、`none`。
- 没有 `expected_outcomes` 的旧评估数据必须保持兼容。
- 仓库示例与 `src/dream_memory/examples/` 包内示例必须逐字一致。
- 每个完整任务测试通过后使用中文提交信息提交。

---

### Task 1: 明确项目指令单事件证据

**Files:**
- Modify: `src/dream_memory/memory_dreaming.py:838-896`
- Test: `tests/test_memory_dreaming.py`

**Interfaces:**
- Consumes: `extract_atomic_facts(events, project=...)`
- Consumes: `build_candidates_from_facts(facts)`
- Consumes: `build_review_queue(candidates, memory_cards)`
- Produces: 项目指令和项目标记事实的 `tags` 包含 `explicit`

- [ ] **Step 1: 写项目指令红灯测试**

在 `tests/test_memory_dreaming.py` 增加：

```python
def test_project_instruction_single_event_is_reviewable(self):
    events = [{
        "event_id": "event_project_instruction",
        "source": "project",
        "role": "system",
        "event_type": "project_instruction",
        "project": "/tmp/project",
        "content": "如果后端使用 Python，则使用 uv 进行包管理；前端使用 pnpm。",
    }]

    facts = extract_atomic_facts(events, project="/tmp/project")
    candidates = build_candidates_from_facts(facts)
    queue = build_review_queue(candidates, [])

    self.assertEqual(len(candidates), 1)
    self.assertIn("explicit", candidates[0]["tags"])
    self.assertEqual(len(queue), 1)
    self.assertEqual(queue[0]["dream_analysis"]["required_evidence_count"], 1)
```

- [ ] **Step 2: 写项目标记红灯测试**

在同一测试类增加：

```python
def test_project_markers_single_event_are_reviewable(self):
    events = [{
        "event_id": "event_project_markers",
        "source": "project",
        "role": "system",
        "event_type": "project_markers",
        "project": "/tmp/project",
        "content": "python_package_manager=uv; python_test_runner=pytest; python_framework=fastapi",
    }]

    facts = extract_atomic_facts(events, project="/tmp/project")
    candidates = build_candidates_from_facts(facts)
    queue = build_review_queue(candidates, [])

    self.assertEqual(len(facts), 3)
    self.assertTrue(all("explicit" in fact["tags"] for fact in facts))
    self.assertEqual(len(queue), 3)
    self.assertTrue(all(
        item["dream_analysis"]["required_evidence_count"] == 1
        for item in queue
    ))
```

- [ ] **Step 3: 运行红灯测试**

Run:

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
$env:PYTHONUTF8 = "1"
uv run python -m unittest tests.test_memory_dreaming.MemoryDreamingTests.test_project_instruction_single_event_is_reviewable tests.test_memory_dreaming.MemoryDreamingTests.test_project_markers_single_event_are_reviewable -v
```

Expected: FAIL，因为现有项目指令和项目标记事实没有 `explicit`，review queue 为空。

- [ ] **Step 4: 增加最小 `explicit` 标签**

在 `src/dream_memory/memory_dreaming.py` 修改现有标签：

```python
tags=["package-manager", "uv", "pnpm", "python", "frontend", "explicit"],
```

项目标记包管理器标签初始化改为：

```python
tags = ["package-manager", "explicit"]
```

测试运行器标签改为：

```python
tags=["testing", "python", runner, "explicit"],
```

框架标签改为：

```python
tags=["framework", "python", framework.lower(), "explicit"],
```

不要修改其他普通历史消息规则。

- [ ] **Step 5: 运行聚焦测试和 dreaming 套件**

Run:

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
$env:PYTHONUTF8 = "1"
uv run python -m unittest tests.test_memory_dreaming -q
```

Expected: PASS。

- [ ] **Step 6: 提交 Task 1**

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
git add src/dream_memory/memory_dreaming.py tests/test_memory_dreaming.py
git commit -m "标记明确项目指令为单事件证据"
```

---

### Task 2: 评估候选结果状态

**Files:**
- Modify: `src/dream_memory/memory_eval.py:25-253`
- Test: `tests/test_memory_eval.py`

**Interfaces:**
- Produces: `_row_expected_outcomes(row: dict[str, Any]) -> list[str] | None`
- Produces: `_candidate_outcome(candidate: dict[str, Any]) -> str`
- Changes: `_scored_candidates(candidates) -> tuple[list[dict[str, Any]], int, list[str]]`
- Adds report fields: `outcome_checked_rows`, `outcome_correct_rows`, `outcome_accuracy`, `outcome_mismatches`

- [ ] **Step 1: 写状态归一化红灯测试**

更新 `tests/test_memory_eval.py` 的 import，加入 `_candidate_outcome`，并增加：

```python
def test_candidate_outcome_normalizes_dream_actions(self):
    self.assertEqual(
        _candidate_outcome({"dream_analysis": {"suggested_action": "create"}}),
        "reviewable",
    )
    self.assertEqual(
        _candidate_outcome({"dream_analysis": {"suggested_action": "merge"}}),
        "reviewable",
    )
    self.assertEqual(
        _candidate_outcome({"dream_analysis": {"suggested_action": "needs_more_evidence"}}),
        "deferred",
    )
    self.assertEqual(
        _candidate_outcome({"dream_analysis": {"suggested_action": "reject"}}),
        "rejected",
    )
```

- [ ] **Step 2: 写行级状态准确率红灯测试**

增加：

```python
def test_eval_reports_expected_outcome_accuracy(self):
    deferred = {
        "id": "mem_deferred",
        "content": "User prefers Vim.",
        "type": "preference",
        "scope": "user",
        "score": 0.95,
        "tags": ["preference"],
        "evidence": [{"event_id": "event_1"}],
    }
    reviewable = {
        "id": "mem_reviewable",
        "content": "User prefers concise answers.",
        "type": "preference",
        "scope": "user",
        "score": 0.95,
        "tags": ["preference"],
        "evidence": [{"event_id": "event_1"}, {"event_id": "event_2"}],
    }
    rows = [
        {"id": "deferred", "events": [], "expected": [], "expected_outcomes": ["deferred"]},
        {
            "id": "reviewable",
            "events": [],
            "expected": [{
                "content": "User prefers concise answers.",
                "type": "preference",
                "scope": "user",
            }],
            "expected_outcomes": ["reviewable"],
        },
        {"id": "none", "events": [], "expected": [], "expected_outcomes": ["none"]},
    ]

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "labeled.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        with patch("dream_memory.memory_eval._extract_candidates") as extract:
            extract.side_effect = [
                ([deferred], None),
                ([reviewable], None),
                ([], None),
            ]
            result = evaluate_labeled_events(path, project=None, mode="rules")

    self.assertEqual(result["outcome_checked_rows"], 3)
    self.assertEqual(result["outcome_correct_rows"], 3)
    self.assertEqual(result["outcome_accuracy"], 1.0)
    self.assertEqual(result["outcome_mismatches"], [])
```

- [ ] **Step 3: 写非法状态和旧数据兼容红灯测试**

增加：

```python
def test_eval_rejects_invalid_expected_outcome(self):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "labeled.jsonl"
        path.write_text(json.dumps({
            "id": "invalid",
            "events": [],
            "expected": [],
            "expected_outcomes": ["later"],
        }) + "\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "unsupported expected outcome"):
            evaluate_labeled_events(path, project=None, mode="rules")


def test_eval_keeps_legacy_rows_without_outcome_labels(self):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "labeled.jsonl"
        path.write_text(json.dumps({
            "id": "legacy",
            "events": [],
            "expected": [],
        }) + "\n", encoding="utf-8")

        with patch("dream_memory.memory_eval._extract_candidates", return_value=([], None)):
            result = evaluate_labeled_events(path, project=None, mode="rules")

    self.assertEqual(result["outcome_checked_rows"], 0)
    self.assertEqual(result["outcome_correct_rows"], 0)
    self.assertEqual(result["outcome_accuracy"], 0.0)
```

- [ ] **Step 4: 运行红灯测试**

Run:

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
$env:PYTHONUTF8 = "1"
uv run python -m unittest tests.test_memory_eval.MemoryEvalTests.test_candidate_outcome_normalizes_dream_actions tests.test_memory_eval.MemoryEvalTests.test_eval_reports_expected_outcome_accuracy tests.test_memory_eval.MemoryEvalTests.test_eval_rejects_invalid_expected_outcome tests.test_memory_eval.MemoryEvalTests.test_eval_keeps_legacy_rows_without_outcome_labels -v
```

Expected: FAIL，因为状态 helper 和报告字段尚不存在。

- [ ] **Step 5: 实现期望状态解析**

在 `src/dream_memory/memory_eval.py` 的行解析 helper 附近增加：

```python
_EXPECTED_OUTCOMES = {"reviewable", "deferred", "rejected", "none"}


def _row_expected_outcomes(row: dict[str, Any]) -> list[str] | None:
    if "expected_outcomes" not in row:
        return None
    raw = row.get("expected_outcomes")
    if not isinstance(raw, list) or not raw:
        raise ValueError("expected_outcomes must be a non-empty list")
    outcomes = sorted({str(item).strip() for item in raw if str(item).strip()})
    if not outcomes:
        raise ValueError("expected_outcomes must contain at least one outcome")
    invalid = sorted(set(outcomes) - _EXPECTED_OUTCOMES)
    if invalid:
        raise ValueError(f"unsupported expected outcome: {invalid[0]}")
    return outcomes
```

- [ ] **Step 6: 实现状态归一化和 `_scored_candidates` 新返回值**

增加：

```python
def _candidate_outcome(candidate: dict[str, Any]) -> str:
    analysis = candidate.get("dream_analysis") if isinstance(candidate.get("dream_analysis"), dict) else {}
    action = str(analysis.get("suggested_action") or "")
    if action in {"create", "review", "merge"}:
        return "reviewable"
    if action == "needs_more_evidence":
        return "deferred"
    if action == "reject":
        return "rejected"
    raise ValueError(f"unsupported dream action: {action}")
```

将 `_scored_candidates()` 改为：

```python
def _scored_candidates(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, list[str]]:
    normalized = []
    for index, candidate in enumerate(candidates, start=1):
        item = dict(candidate)
        item.setdefault("id", f"eval_candidate_{index}")
        normalized.append(item)
    analyzed = apply_dream_analysis_to_candidates(normalized, [])
    outcomes = sorted({_candidate_outcome(candidate) for candidate in analyzed}) if analyzed else ["none"]
    deferred = sum(
        1
        for candidate in analyzed
        if _candidate_outcome(candidate) == "deferred"
    )
    reviewable = [
        candidate
        for candidate in analyzed
        if _candidate_outcome(candidate) == "reviewable"
    ]
    return reviewable, deferred, outcomes
```

- [ ] **Step 7: 接入行级结果比较**

在 `evaluate_labeled_events()` 初始化：

```python
outcome_checked_rows = 0
outcome_correct_rows = 0
outcome_mismatches: list[dict[str, Any]] = []
```

每行读取：

```python
expected_outcomes = _row_expected_outcomes(row)
actual_outcomes = ["none"]
```

所有 `_scored_candidates()` 调用改为接收第三个返回值：

```python
candidates, deferred_count, actual_outcomes = _scored_candidates(candidates)
```

规则 fallback 被实际选用时，使用 fallback 的状态：

```python
fallback_candidates, deferred_count, fallback_outcomes = _scored_candidates(fallback_candidates)
if fallback_candidates:
    candidates = fallback_candidates
    actual_outcomes = fallback_outcomes
```

在每行 false positive 处理后增加：

```python
if expected_outcomes is not None:
    outcome_checked_rows += 1
    if actual_outcomes == expected_outcomes:
        outcome_correct_rows += 1
    else:
        outcome_mismatches.append({
            "row": row_index,
            "row_id": row_label,
            "expected_outcomes": expected_outcomes,
            "actual_outcomes": actual_outcomes,
        })
```

返回报告增加：

```python
"outcome_checked_rows": outcome_checked_rows,
"outcome_correct_rows": outcome_correct_rows,
"outcome_accuracy": round(
    outcome_correct_rows / outcome_checked_rows,
    3,
) if outcome_checked_rows else 0.0,
"outcome_mismatches": outcome_mismatches[:20],
```

- [ ] **Step 8: 运行评估测试**

Run:

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
$env:PYTHONUTF8 = "1"
uv run python -m unittest tests.test_memory_eval -q
```

Expected: PASS。

- [ ] **Step 9: 提交 Task 2**

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
git add src/dream_memory/memory_eval.py tests/test_memory_eval.py
git commit -m "增加评估结果状态指标"
```

---

### Task 3: 基准集状态标注和回归指标

**Files:**
- Modify: `examples/labeled-events.jsonl`
- Modify: `src/dream_memory/examples/labeled-events.jsonl`
- Modify: `tests/test_memory_cli.py`
- Modify: `tests/test_memory_eval.py`

**Interfaces:**
- Consumes: `expected_outcomes`
- Verifies: 当前 16 行基准的内容指标和状态指标

- [ ] **Step 1: 写基准状态红灯测试**

在 `tests/test_memory_eval.py` 增加：

```python
def test_repository_labeled_rules_eval_has_perfect_outcome_accuracy(self):
    path = Path(__file__).resolve().parents[1] / "examples" / "labeled-events.jsonl"

    result = evaluate_labeled_events(path, project="/tmp/project", mode="rules")

    self.assertEqual(result["rows"], 16)
    self.assertEqual(result["precision"], 1.0)
    self.assertEqual(result["true_positive"], 12)
    self.assertEqual(result["false_positive_count"], 0)
    self.assertEqual(result["false_negative_count"], 0)
    self.assertEqual(result["deferred_candidate_count"], 2)
    self.assertEqual(result["outcome_checked_rows"], 16)
    self.assertEqual(result["outcome_accuracy"], 1.0)
    self.assertEqual(result["outcome_mismatches"], [])
```

Expected: 在添加标注前 FAIL，因为 `outcome_checked_rows` 为 `0`。

- [ ] **Step 2: 为 16 行添加精确状态标注**

在两个 JSONL 文件中使用以下映射，保持其他字段不变：

| Row ID | `expected_outcomes` |
|---|---|
| `preference_language` | `["reviewable"]` |
| `project_instruction_package_managers` | `["reviewable"]` |
| `project_marker_pytest_fastapi` | `["reviewable"]` |
| `product_direction` | `["reviewable"]` |
| `human_review_gate` | `["reviewable"]` |
| `pitfall_real_flow` | `["reviewable"]` |
| `one_off_task_noise` | `["none"]` |
| `internal_context_noise` | `["none"]` |
| `duplicate_memory_pair` | `["reviewable"]` |
| `rejected_option` | `["reviewable"]` |
| `credential_location_noise` | `["none"]` |
| `cross_project_noise` | `["none"]` |
| `cross_project_user_preference` | `["reviewable"]` |
| `single_event_ordinary_preference` | `["deferred"]` |
| `duplicate_same_event_preference` | `["deferred"]` |
| `two_event_ordinary_preference` | `["reviewable"]` |

每行格式示例：

```json
{"id":"single_event_ordinary_preference","event":{"event_id":"eval_single_editor_1","source":"codex","role":"user","event_type":"history_prompt","content":"用户偏好使用 Vim 编辑器。"},"expected":[],"expected_outcomes":["deferred"]}
```

- [ ] **Step 3: 更新 CLI 初始化样本指标断言**

在 `tests/test_memory_cli.py` 的两个初始化评估测试中，将旧指标改为：

```python
self.assertEqual(payload["rows"], 16)
self.assertEqual(payload["precision"], 1.0)
self.assertEqual(payload["recall"], 1.0)
self.assertEqual(payload["f1"], 1.0)
self.assertEqual(payload["true_positive"], 12)
self.assertEqual(payload["false_positive_count"], 0)
self.assertEqual(payload["deferred_candidate_count"], 2)
self.assertEqual(payload["outcome_checked_rows"], 16)
self.assertEqual(payload["outcome_accuracy"], 1.0)
```

- [ ] **Step 4: 运行基准和 CLI 测试**

Run:

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
$env:PYTHONUTF8 = "1"
uv run python -m unittest tests.test_memory_eval tests.test_memory_cli -q
```

Expected: PASS。

- [ ] **Step 5: 运行规则评估确认指标**

Run:

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
$env:PYTHONUTF8 = "1"
uv run dream-memory eval --input examples/labeled-events.jsonl --project /tmp/project --mode rules
```

Expected:

- precision `1.000`
- recall `1.000`
- F1 `1.000`
- true positive `12`
- false positive `0`
- false negative `0`
- deferred candidate `2`
- outcome accuracy `1.000`

- [ ] **Step 6: 提交 Task 3**

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
git add examples/labeled-events.jsonl src/dream_memory/examples/labeled-events.jsonl tests/test_memory_cli.py tests/test_memory_eval.py
git commit -m "完善基准样本结果状态标注"
```

---

### Task 4: 文档与完整验收

**Files:**
- Modify: `README.md`
- Modify: `docs/cli.md`
- Verify: Tasks 1-3 的所有文件

**Interfaces:**
- Documents: `expected_outcomes` 和新增报告字段
- Verifies: 完整测试、评估、差异和工作树

- [ ] **Step 1: 更新文档**

在 README 和 CLI 文档的评估章节增加以下信息：

```markdown
评估 JSONL 可选使用 `expected_outcomes` 验证候选状态，允许值为
`reviewable`、`deferred`、`rejected` 和 `none`。报告中的
`outcome_checked_rows`、`outcome_correct_rows`、`outcome_accuracy`
和 `outcome_mismatches` 用于检查状态标注，不替代原有内容匹配指标。
```

同时说明 `project_instruction` 和结构化 `project_markers` 属于明确项目指令，单个有效事件即可进入人工审核；普通偏好仍需要两个独立事件。

- [ ] **Step 2: 运行完整测试**

Run:

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
$env:PYTHONUTF8 = "1"
uv run python -m unittest discover -s tests -q
```

Expected: PASS，零失败。

- [ ] **Step 3: 运行最终评估**

Run:

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
$env:PYTHONUTF8 = "1"
uv run dream-memory eval --input examples/labeled-events.jsonl --project /tmp/project --mode rules
```

Expected: Task 3 Step 5 的全部指标成立。

- [ ] **Step 4: 检查差异和样本同步**

Run:

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
git diff --check
if ((Get-Content examples/labeled-events.jsonl -Raw -Encoding UTF8) -ne (Get-Content src/dream_memory/examples/labeled-events.jsonl -Raw -Encoding UTF8)) {
    throw "packaged labeled events differ from repository sample"
}
git status --short
git diff --stat
```

Expected: `git diff --check` 成功，两个 JSONL 完全一致，没有临时文件或无关修改。

- [ ] **Step 5: 提交 Task 4**

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
git add README.md docs/cli.md
git commit -m "更新评估结果状态文档"
```

- [ ] **Step 6: 最终提交后状态检查**

Run:

```powershell
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [Console]::OutputEncoding
git status --short
git log -5 --oneline
```

Expected: 工作树干净，最近四个实现提交均为中文提交信息。
