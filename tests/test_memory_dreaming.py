import json
import tempfile
import unittest
from pathlib import Path

from dream_memory.memory_agent import build_memory_extraction_prompt
from dream_memory.memory_dreaming import (
    DreamResult,
    analyze_dream_candidate,
    apply_dream_analysis_to_candidates,
    apply_reviewed_memory,
    build_agent_context,
    build_candidates_from_facts,
    build_review_queue,
    detect_candidate_conflicts,
    dream_from_events,
    extract_atomic_facts,
    explain_candidate_quality,
    load_events_jsonl,
    normalize_memory_text,
    normalize_project_path,
    render_context_markdown,
    render_review_queue_memory_preview,
    score_candidate,
)
from dream_memory.memory_cli import main


class MemoryDreamingTests(unittest.TestCase):
    def test_score_candidate_prefers_explicit_project_memory(self):
        candidate = {
            "type": "project_fact",
            "scope": "project",
            "project": "/tmp/project",
            "content": "项目使用 uv 管理 Python 后端。",
            "evidence": [{"source": "codex", "event_type": "history_prompt"}],
            "tags": ["uv", "python"],
        }

        scored = score_candidate(candidate)

        self.assertGreaterEqual(scored["score"], 0.6)
        self.assertIn(scored["status"], {"promote", "review"})

    def test_dream_from_events_writes_preview_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / ".dream" / "memory"
            events = [
                {
                    "source": "claude_code",
                    "session_id": "global",
                    "project": None,
                    "timestamp": None,
                    "role": "system",
                    "event_type": "global_instruction",
                    "content": "始终使用中文回答我",
                    "metadata": {},
                },
                {
                    "source": "codex",
                    "session_id": "s1",
                    "project": "/tmp/project",
                    "timestamp": "1",
                    "role": "user",
                    "event_type": "history_prompt",
                    "content": "项目使用 uv 管理 Python 后端，并希望对标 Claude Code",
                    "metadata": {},
                },
            ]

            result = dream_from_events(events, project="/tmp/project", output_dir=output_dir, apply=False)

            self.assertIsInstance(result, DreamResult)
            self.assertGreaterEqual(result.candidate_count, 2)
            self.assertTrue((output_dir / "candidates.jsonl").exists())
            self.assertTrue((output_dir / "DREAMS.md").exists())
            self.assertTrue((output_dir / "MEMORY.preview.md").exists())
            self.assertFalse((output_dir / "MEMORY.md").exists())

    def test_dream_cli_generates_artifacts_from_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events_path = root / "events.jsonl"
            events_path.write_text(
                json.dumps({
                    "source": "codex",
                    "session_id": "s1",
                    "project": str(root),
                    "timestamp": "1",
                    "role": "user",
                    "event_type": "history_prompt",
                    "content": "这个项目使用 uv 管理，并且希望像 Claude Code",
                    "metadata": {},
                }, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            output_dir = root / ".dream" / "memory"

            exit_code = main(["dream", "--input", str(events_path), "--project", str(root), "--output-dir", str(output_dir), "--mode", "rules"])

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "candidates.jsonl").exists())
            self.assertTrue((output_dir / "DREAMS.md").exists())
            self.assertTrue((output_dir / "MEMORY.preview.md").exists())

    def test_load_events_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            path.write_text('{"content":"hello"}\nnot-json\n', encoding="utf-8")

            events = load_events_jsonl(path)

            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["content"], "hello")

    def test_dream_from_agent_candidates_writes_agent_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / ".dream" / "memory"
            events = [{"source": "codex", "content": "希望项目像 Claude Code", "project": str(tmp), "role": "user"}]
            agent_candidates = [
                {
                    "content": "用户希望项目做成 Claude Code 风格的本地研发助手。",
                    "type": "product_direction",
                    "scope": "project",
                    "project": str(tmp),
                    "confidence": 0.95,
                    "decision": "promote",
                    "reason": "explicit user request",
                    "evidence": [{"event_id": "event_1", "source": "codex"}],
                    "tags": ["claude-code", "explicit"],
                }
            ]

            result = dream_from_events(events, project=str(tmp), output_dir=output_dir, apply=False, agent_candidates=agent_candidates, agent_mode=True)

            self.assertEqual(result.candidate_count, 1)
            self.assertEqual(result.promoted_count, 1)
            self.assertTrue((output_dir / "ai-candidates.jsonl").exists())
            preview = (output_dir / "MEMORY.preview.md").read_text(encoding="utf-8")
            self.assertIn("Claude Code 风格", preview)

    def test_extract_atomic_facts_derives_reusable_user_workflows_from_command_patterns(self):
        events = [
            {"source": "codex", "role": "user", "event_type": "history_prompt", "content": "继续"},
            {"source": "codex", "role": "user", "event_type": "history_prompt", "content": "做"},
            {"source": "codex", "role": "user", "event_type": "history_prompt", "content": "提交"},
            {"source": "codex", "role": "user", "event_type": "history_prompt", "content": "推送"},
            {"source": "codex", "role": "user", "event_type": "history_prompt", "content": "查看我当前的分支"},
            {"source": "codex", "role": "user", "event_type": "history_prompt", "content": "新建一个分支,开始精简"},
            {"source": "codex", "role": "user", "event_type": "history_prompt", "content": "制定一个计划,逐步的删除模块"},
            {"source": "codex", "role": "user", "event_type": "thread_first_user_message", "project": "/tmp/p", "content": "梳理这个项目"},
            {"source": "codex", "role": "user", "event_type": "thread_first_user_message", "project": "/tmp/p", "content": "先分析一下可行性"},
            {"source": "claude_code", "role": "system", "event_type": "global_instruction", "content": "记住在我布置任务时要优先判断这个任务需要几个人工作，根据情况使用agent team，始终使用中文回答我"},
        ]

        facts = extract_atomic_facts(events, project="/tmp/p")
        candidates = build_candidates_from_facts(facts)
        contents = "\n".join(candidate["content"] for candidate in candidates)

        self.assertIn("继续/做/下一步", contents)
        self.assertIn("仓库维护", contents)
        self.assertIn("独立分支", contents)
        self.assertIn("真实仓库结构", contents)
        self.assertIn("始终使用中文回答", contents)
        self.assertIn("几个 agent 协作", contents)
        self.assertFalse(any(candidate["type"] == "requirement" and "agent team" in candidate["content"] for candidate in candidates))
        self.assertFalse(any(candidate["content"] == "记住在我布置任务时要优先判断这个任务需要几个人工作，根据情况使用agent team，始终使用中文回答我" for candidate in candidates))

    def test_extract_atomic_facts_does_not_duplicate_plain_language_preference(self):
        events = [{"source": "codex", "role": "user", "event_type": "history_prompt", "content": "用户偏好中文回答。"}]

        facts = extract_atomic_facts(events, project=None)
        candidates = build_candidates_from_facts(facts)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["content"], "用户偏好中文回答。")

    def test_extract_atomic_facts_deduplicates_direct_facts_and_merges_evidence(self):
        events = [
            {"event_id": "event_1", "source": "codex", "session_id": "s1", "project": "/tmp/project", "role": "user", "event_type": "history_prompt", "content": "这个项目需要人工审核后才能写正式记忆"},
            {"event_id": "event_2", "source": "claude_code", "session_id": "s2", "project": "/tmp/project", "role": "user", "event_type": "transcript_message", "content": "这个项目需要人工审核后才能写正式记忆"},
            {"event_id": "event_3", "source": "codex", "session_id": "s3", "project": "/tmp/project", "role": "user", "event_type": "history_prompt", "content": "不需要再问我，按照你的建议来"},
            {"event_id": "event_4", "source": "claude_code", "session_id": "s4", "project": "/tmp/project", "role": "user", "event_type": "transcript_message", "content": "不用问我，你决定直接做"},
        ]

        facts = extract_atomic_facts(events, project="/tmp/project")
        review_gate = [fact for fact in facts if fact.get("statement") == "正式记忆必须经过人工审核，只有审核通过的候选才允许写入长期记忆。"]
        autonomy = [fact for fact in facts if fact.get("statement") == "用户偏好在上下文清楚时由助手按判断直接推进，不要反复询问确认。"]

        self.assertEqual(len(review_gate), 1)
        self.assertEqual(len(autonomy), 1)
        self.assertEqual(review_gate[0]["evidence_refs"], ["event_1", "event_2"])
        self.assertEqual(autonomy[0]["evidence_refs"], ["event_3", "event_4"])

    def test_build_candidates_from_facts_deduplicates_same_event_id(self):
        facts = [
            {
                "fact_type": "preference",
                "statement": "User prefers concise answers.",
                "scope": "user",
                "project": None,
                "tags": ["preference"],
                "evidence": [{"event_id": "event_1", "source": "codex"}],
            },
            {
                "fact_type": "preference",
                "statement": "User prefers concise answers.",
                "scope": "user",
                "project": None,
                "tags": ["preference"],
                "evidence": [{"event_id": "event_1", "source": "codex"}],
            },
        ]

        candidates = build_candidates_from_facts(facts)

        self.assertEqual(len(candidates), 1)
        self.assertEqual([item["event_id"] for item in candidates[0]["evidence"]], ["event_1"])

    def test_extract_atomic_facts_creates_fact_records(self):
        events = [
            {
                "event_id": "event_1",
                "source": "claude_code",
                "session_id": "global",
                "project": None,
                "role": "system",
                "event_type": "global_instruction",
                "content": "始终使用中文回答我",
            },
            {
                "event_id": "event_2",
                "source": "codex",
                "session_id": "s1",
                "project": "/tmp/project",
                "role": "user",
                "event_type": "history_prompt",
                "content": "这个项目需要人工审核后才能写正式记忆",
            },
        ]

        facts = extract_atomic_facts(events, project="/tmp/project")

        self.assertGreaterEqual(len(facts), 2)
        self.assertTrue(any(fact["fact_type"] == "preference" for fact in facts))
        self.assertTrue(any("人工审核" in fact["statement"] for fact in facts))

    def test_dream_from_events_writes_facts_before_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "memory"
            events = [{
                "event_id": "event_1",
                "source": "codex",
                "session_id": "s1",
                "project": "/tmp/project",
                "role": "user",
                "event_type": "history_prompt",
                "content": "这个项目需要人工审核后才能写正式记忆",
            }]

            dream_from_events(events, project="/tmp/project", output_dir=output_dir, apply=False)

            self.assertTrue((output_dir / "facts.jsonl").exists())
            self.assertTrue((output_dir / "candidates.jsonl").exists())

    def test_build_candidates_from_facts_keeps_project_state_as_evidence_only(self):
        facts = [
            {
                "id": "fact_1",
                "fact_type": "system_state",
                "statement": "Claude Code project state for /tmp/project",
                "scope": "project",
                "project": "/tmp/project",
                "evidence_refs": ["event_1"],
                "confidence": 0.6,
                "tags": ["project_state"],
            }
        ]

        candidates = build_candidates_from_facts(facts)

        self.assertEqual(candidates, [])

    def test_detect_candidate_conflicts_ignores_unrelated_same_type_preferences(self):
        candidates = [{
            "id": "autonomy",
            "scope": "user",
            "project": None,
            "type": "preference",
            "content": "用户偏好在上下文清楚时由助手按判断直接推进，不要反复询问确认。",
            "score": 0.8,
            "evidence": [{"event_id": "event_1"}, {"event_id": "event_2"}],
        }]
        cards = [{
            "id": "language",
            "scope": "user",
            "project": None,
            "memory_type": "preference",
            "summary": "用户偏好始终使用中文回答。",
            "status": "active",
        }]

        conflicts = detect_candidate_conflicts(candidates, cards)
        queue = build_review_queue(candidates, cards)

        self.assertEqual(conflicts, {})
        self.assertNotEqual(queue[0]["suggested_action"], "merge")

    def test_detect_candidate_conflicts_flags_same_scope_competing_summary(self):
        candidates = [{
            "id": "cand_1",
            "scope": "project",
            "project": "/tmp/project",
            "type": "decision",
            "content": "项目目标是 Claude Code 风格助手。",
            "score": 0.9,
            "status": "promote",
        }]
        memory_cards = [{
            "id": "mem_1",
            "scope": "project",
            "project": "/tmp/project",
            "memory_type": "decision",
            "summary": "项目目标是 Claude Code 风格本地研发助手。",
            "status": "active",
        }]

        conflicts = detect_candidate_conflicts(candidates, memory_cards)

        self.assertEqual(conflicts["cand_1"][0]["memory_id"], "mem_1")

    def test_build_review_queue_adds_quality_signals_and_action_suggestions(self):
        candidates = [
            {
                "id": "cand_duplicate",
                "scope": "project",
                "project": "/tmp/project",
                "type": "decision",
                "content": "项目目标是 Claude Code 风格助手。",
                "score": 0.92,
                "status": "promote",
                "evidence": [{"event_id": "event_1"}],
                "tags": ["claude-code"],
            },
            {
                "id": "cand_replace",
                "scope": "project",
                "project": "/tmp/project",
                "type": "decision",
                "content": "项目目标是 Claude Code 风格的本地研发助手。",
                "score": 0.9,
                "status": "promote",
                "evidence": [{"event_id": "event_2"}, {"event_id": "event_3"}],
                "tags": ["claude-code", "product-direction"],
            },
            {
                "id": "cand_more_evidence",
                "scope": "project",
                "project": "/tmp/project",
                "type": "workflow",
                "content": "构建前需要先跑 smoke 测试。",
                "score": 0.48,
                "status": "review",
                "evidence": [],
                "tags": ["workflow"],
            },
        ]
        memory_cards = [
            {
                "id": "mem_same",
                "scope": "project",
                "project": "/tmp/project",
                "memory_type": "decision",
                "summary": "项目目标是 Claude Code 风格助手。",
                "status": "active",
                "retrieval_hints": ["claude-code"],
            },
            {
                "id": "mem_old",
                "scope": "project",
                "project": "/tmp/project",
                "memory_type": "decision",
                "summary": "项目目标是通用聊天机器人。",
                "status": "active",
                "retrieval_hints": ["chatbot"],
            },
        ]

        queue = build_review_queue(candidates, memory_cards)
        by_id = {item["candidate_id"]: item for item in queue}
        analyzed = {
            item["id"]: item
            for item in apply_dream_analysis_to_candidates(candidates, memory_cards)
        }

        self.assertEqual(set(by_id), {"cand_replace"})
        self.assertEqual(analyzed["cand_duplicate"]["dream_analysis"]["suggested_action"], "reject")
        self.assertTrue(analyzed["cand_duplicate"]["quality_signals"]["duplicate"])
        self.assertEqual(analyzed["cand_duplicate"]["quality_signals"]["matched_memory_id"], "mem_same")
        self.assertEqual(by_id["cand_replace"]["suggested_action"], "merge")
        self.assertEqual(by_id["cand_replace"]["quality_signals"]["matched_memory_id"], "mem_same")
        self.assertGreater(by_id["cand_replace"]["quality_signals"]["evidence_strength"], 0)
        self.assertEqual(analyzed["cand_more_evidence"]["dream_analysis"]["suggested_action"], "needs_more_evidence")
        self.assertEqual(analyzed["cand_more_evidence"]["quality_signals"]["evidence_strength"], 0)

    def test_build_review_queue_excludes_needs_more_evidence(self):
        candidates = [{
            "id": "cand_single",
            "type": "preference",
            "scope": "user",
            "project": None,
            "content": "User prefers concise answers.",
            "score": 0.95,
            "tags": ["preference"],
            "evidence": [{"event_id": "event_1", "source": "codex"}],
        }]

        self.assertEqual(build_review_queue(candidates, []), [])

    def test_build_review_queue_includes_two_event_candidate(self):
        candidates = [{
            "id": "cand_repeated",
            "type": "preference",
            "scope": "user",
            "project": None,
            "content": "User prefers concise answers.",
            "score": 0.95,
            "tags": ["preference"],
            "evidence": [
                {"event_id": "event_1", "source": "codex"},
                {"event_id": "event_2", "source": "claude_code"},
            ],
        }]

        queue = build_review_queue(candidates, [])

        self.assertEqual(len(queue), 1)
        self.assertIn(queue[0]["suggested_action"], {"create", "review"})

    def test_build_review_queue_explains_duplicate_existing_memory_match(self):
        candidates = [{
            "id": "cand_same",
            "type": "workflow",
            "scope": "user",
            "project": None,
            "content": "用户偏好直接推进。",
            "score": 0.9,
            "status": "promote",
            "evidence": [{"event_id": "event_1"}],
        }]
        memory_cards = [{
            "id": "mem_same",
            "scope": "user",
            "project": None,
            "memory_type": "workflow",
            "summary": "用户偏好直接推进。",
            "status": "active",
        }]

        queue = build_review_queue(candidates, memory_cards)
        signals = explain_candidate_quality(candidates[0], memory_cards)
        analysis = analyze_dream_candidate(candidates[0], quality_signals=signals, conflicts=[])

        self.assertEqual(queue, [])
        self.assertTrue(signals["duplicate"])
        self.assertEqual(signals["value_class"], "existing_duplicate")
        self.assertEqual(signals["matched_memory_id"], "mem_same")
        self.assertEqual(signals["matched_memory_summary"], "用户偏好直接推进。")
        self.assertIn("already exists", analysis["decision_reason"])

    def test_review_queue_memory_preview_excludes_duplicate_rejects(self):
        candidates = [
            {
                "id": "cand_duplicate",
                "type": "preference",
                "scope": "user",
                "project": None,
                "content": "用户偏好中文回答。",
                "score": 0.9,
                "status": "promote",
                "evidence": [{"event_id": "event_1"}],
                "tags": ["language"],
            },
            {
                "id": "cand_new",
                "type": "workflow",
                "scope": "project",
                "project": "/tmp/project",
                "content": "Python 项目使用 uv 进行包管理和命令执行。",
                "score": 0.9,
                "status": "promote",
                "evidence": [{"event_id": "event_2"}, {"event_id": "event_3"}],
                "tags": ["package-manager", "uv"],
            },
        ]
        memory_cards = [{
            "id": "mem_existing",
            "scope": "user",
            "project": None,
            "memory_type": "preference",
            "summary": "用户偏好中文回答。",
            "status": "active",
        }]

        queue = build_review_queue(candidates, memory_cards)
        preview = render_review_queue_memory_preview(queue)

        self.assertNotIn("用户偏好中文回答", preview)
        self.assertIn("Python 项目使用 uv", preview)

    def test_build_review_queue_marks_new_memory_as_new_value(self):
        candidates = [{
            "id": "cand_new",
            "type": "workflow",
            "scope": "user",
            "project": None,
            "content": "用户希望 review queue 默认突出新增价值。",
            "score": 0.9,
            "status": "promote",
            "evidence": [{"event_id": "event_1"}, {"event_id": "event_2"}],
        }]

        queue = build_review_queue(candidates, [])
        signals = queue[0]["quality_signals"]
        analysis = queue[0]["dream_analysis"]

        self.assertFalse(signals["duplicate"])
        self.assertEqual(signals["value_class"], "new_value")
        self.assertIn("new reusable memory", analysis["decision_reason"])

    def test_build_review_queue_does_not_match_candidate_to_itself_when_memory_cards_reuse_candidates(self):
        candidates = [{
            "id": "cand_self",
            "type": "workflow",
            "scope": "project",
            "project": "/tmp/project",
            "content": "正式记忆必须经过人工审核，只有审核通过的候选才允许写入长期记忆。",
            "score": 0.76,
            "status": "review",
            "evidence": [{"event_id": "event_1"}, {"event_id": "event_2"}],
        }]

        queue = build_review_queue(candidates, candidates)
        signals = queue[0]["quality_signals"]
        analysis = queue[0]["dream_analysis"]

        self.assertFalse(signals["duplicate"])
        self.assertEqual(signals["value_class"], "new_value")
        self.assertIsNone(signals["matched_memory_id"])
        self.assertNotIn("duplicate", analysis["penalties"])

    def test_analyze_dream_candidate_rejects_one_off_task(self):
        candidate = {
            "id": "mem_task",
            "type": "requirement",
            "scope": "project",
            "project": "/tmp/project",
            "content": "删除首页水印按钮",
            "score": 0.9,
            "evidence": [{"event_id": "event_1"}],
        }
        analysis = analyze_dream_candidate(
            candidate,
            quality_signals={
                "stability": 0.2,
                "reuse_value": 0.1,
                "evidence_strength": 0.5,
                "one_off_task": True,
                "duplicate": False,
                "similarity": 0.0,
                "matched_memory_id": None,
            },
            conflicts=[],
        )

        self.assertEqual(analysis["suggested_action"], "reject")
        self.assertIn("one-off task", analysis["penalties"])
        self.assertLess(analysis["dream_score"], 0.45)

    def test_analyze_dream_candidate_requires_evidence(self):
        candidate = {
            "id": "mem_no_evidence",
            "type": "workflow",
            "scope": "project",
            "project": "/tmp/project",
            "content": "Run targeted tests before changing memory logic.",
            "score": 0.9,
            "evidence": [],
        }
        analysis = analyze_dream_candidate(
            candidate,
            quality_signals={
                "stability": 0.9,
                "reuse_value": 0.8,
                "evidence_strength": 0.0,
                "one_off_task": False,
                "duplicate": False,
                "similarity": 0.0,
                "matched_memory_id": None,
            },
            conflicts=[],
        )

        self.assertEqual(analysis["suggested_action"], "needs_more_evidence")
        self.assertIn("missing evidence", analysis["penalties"])

    def test_analyze_dream_candidate_requires_two_independent_events(self):
        candidate = {
            "id": "mem_editor",
            "type": "preference",
            "scope": "user",
            "content": "User prefers a concise code editor.",
            "score": 0.95,
            "tags": ["preference"],
            "evidence": [{"event_id": "event_1", "source": "codex"}],
        }

        signals = explain_candidate_quality(candidate, [])
        analysis = analyze_dream_candidate(candidate, quality_signals=signals, conflicts=[])

        self.assertEqual(analysis["suggested_action"], "needs_more_evidence")
        self.assertEqual(analysis["independent_evidence_count"], 1)
        self.assertEqual(analysis["required_evidence_count"], 2)

    def test_analyze_dream_candidate_counts_duplicate_event_id_once(self):
        candidate = {
            "id": "mem_editor",
            "type": "preference",
            "scope": "user",
            "content": "User prefers a concise code editor.",
            "score": 0.95,
            "tags": ["preference"],
            "evidence": [
                {"event_id": "event_1", "source": "codex"},
                {"event_id": "event_1", "source": "codex"},
            ],
        }

        signals = explain_candidate_quality(candidate, [])
        analysis = analyze_dream_candidate(candidate, quality_signals=signals, conflicts=[])

        self.assertEqual(analysis["suggested_action"], "needs_more_evidence")
        self.assertEqual(analysis["independent_evidence_count"], 1)

    def test_analyze_dream_candidate_accepts_two_independent_events(self):
        candidate = {
            "id": "mem_editor",
            "type": "preference",
            "scope": "user",
            "content": "User prefers a concise code editor.",
            "score": 0.95,
            "tags": ["preference"],
            "evidence": [
                {"event_id": "event_1", "source": "codex"},
                {"event_id": "event_2", "source": "claude_code"},
            ],
        }

        signals = explain_candidate_quality(candidate, [])
        analysis = analyze_dream_candidate(candidate, quality_signals=signals, conflicts=[])

        self.assertIn(analysis["suggested_action"], {"create", "review"})
        self.assertEqual(analysis["independent_evidence_count"], 2)

    def test_analyze_dream_candidate_requires_event_id_for_explicit_instruction(self):
        candidate = {
            "id": "mem_language",
            "type": "preference",
            "scope": "user",
            "content": "Always answer the user in Chinese.",
            "score": 0.95,
            "tags": ["language", "explicit"],
            "evidence": [{"source": "codex", "event_type": "global_instruction"}],
        }

        signals = explain_candidate_quality(candidate, [])
        analysis = analyze_dream_candidate(candidate, quality_signals=signals, conflicts=[])

        self.assertEqual(signals["evidence_quality"], "explicit_instruction")
        self.assertEqual(analysis["suggested_action"], "needs_more_evidence")
        self.assertEqual(analysis["required_evidence_count"], 1)

    def test_analyze_dream_candidate_promotes_explicit_user_preference_with_single_strong_evidence(self):
        candidate = {
            "id": "mem_language",
            "type": "preference",
            "scope": "user",
            "content": "用户偏好始终使用中文回答。",
            "score": 0.92,
            "tags": ["language", "explicit"],
            "evidence": [{"event_id": "event_1", "source": "claude_code", "event_type": "global_instruction"}],
        }

        signals = explain_candidate_quality(candidate, [])
        analysis = analyze_dream_candidate(candidate, quality_signals=signals, conflicts=[])

        self.assertEqual(signals["evidence_quality"], "explicit_instruction")
        self.assertGreaterEqual(signals["evidence_strength"], 0.5)
        self.assertEqual(analysis["suggested_action"], "create")
        self.assertEqual(analysis["independent_evidence_count"], 1)
        self.assertEqual(analysis["required_evidence_count"], 1)

    def test_analyze_dream_candidate_creates_durable_memory(self):
        candidate = {
            "id": "mem_workflow",
            "type": "workflow",
            "scope": "project",
            "project": "/tmp/project",
            "content": "Run targeted tests before changing memory logic.",
            "score": 0.88,
            "evidence": [{"event_id": "event_1"}, {"event_id": "event_2"}],
        }
        analysis = analyze_dream_candidate(
            candidate,
            quality_signals={
                "stability": 0.9,
                "reuse_value": 0.85,
                "evidence_strength": 0.75,
                "one_off_task": False,
                "duplicate": False,
                "similarity": 0.0,
                "matched_memory_id": None,
            },
            conflicts=[],
        )

        self.assertEqual(analysis["suggested_action"], "create")
        self.assertGreaterEqual(analysis["dream_score"], 0.7)
        self.assertIn("high stability", analysis["reasons"])
        self.assertIn("high reuse value", analysis["reasons"])

    def test_analyze_dream_candidate_merges_similar_memory(self):
        candidate = {
            "id": "mem_merge",
            "type": "workflow",
            "scope": "project",
            "project": "/tmp/project",
            "content": "Run targeted tests before memory changes.",
            "score": 0.8,
            "evidence": [{"event_id": "event_1"}, {"event_id": "event_2"}],
        }
        analysis = analyze_dream_candidate(
            candidate,
            quality_signals={
                "stability": 0.8,
                "reuse_value": 0.8,
                "evidence_strength": 0.5,
                "one_off_task": False,
                "duplicate": False,
                "similarity": 0.42,
                "matched_memory_id": "mem_existing",
            },
            conflicts=[],
        )

        self.assertEqual(analysis["suggested_action"], "merge")
        self.assertEqual(analysis["matched_memory_id"], "mem_existing")
        self.assertIn("similar existing memory", analysis["reasons"])

    def test_build_review_queue_promotes_framework_project_fact_as_reusable_context(self):
        candidates = [{
            "id": "framework",
            "type": "project_fact",
            "scope": "project",
            "project": "/tmp/project",
            "content": "项目使用 FastAPI 作为 Python Web 框架。",
            "score": 0.88,
            "status": "review",
            "tags": ["framework", "python", "fastapi"],
            "evidence": [
                {"event_id": "event_1", "event_type": "project_markers"},
                {"event_id": "event_2", "event_type": "project_markers"},
            ],
        }]

        queue = build_review_queue(candidates, [])

        self.assertEqual(queue[0]["suggested_action"], "create")
        self.assertGreaterEqual(queue[0]["quality_signals"]["stability"], 0.7)
        self.assertGreaterEqual(queue[0]["quality_signals"]["reuse_value"], 0.7)

    def test_build_review_queue_includes_dream_analysis(self):
        candidates = [
            {
                "id": "mem_workflow",
                "type": "workflow",
                "scope": "project",
                "project": "/tmp/project",
                "content": "Run targeted tests before memory changes.",
                "score": 0.9,
                "status": "promote",
                "tags": ["workflow"],
                "evidence": [{"event_id": "event_1"}, {"event_id": "event_2"}],
            }
        ]

        queue = build_review_queue(candidates, [])

        self.assertEqual(len(queue), 1)
        self.assertIn("dream_analysis", queue[0])
        self.assertEqual(queue[0]["suggested_action"], queue[0]["dream_analysis"]["suggested_action"])
        self.assertIn(queue[0]["suggested_action"], {"create", "review"})
        self.assertGreater(queue[0]["dream_analysis"]["dream_score"], 0)

    def test_build_review_queue_uses_dream_analysis_for_one_off_reject(self):
        candidates = [
            {
                "id": "mem_task",
                "type": "requirement",
                "scope": "project",
                "project": "/tmp/project",
                "content": "删除首页水印按钮",
                "score": 0.95,
                "status": "promote",
                "tags": ["task"],
                "evidence": [{"event_id": "event_1"}],
            }
        ]

        queue = build_review_queue(candidates, [])
        signals = explain_candidate_quality(candidates[0], [])
        analysis = analyze_dream_candidate(candidates[0], quality_signals=signals, conflicts=[])

        self.assertEqual(queue, [])
        self.assertEqual(analysis["suggested_action"], "reject")
        self.assertIn("one-off task", analysis["penalties"])

    def test_build_review_queue_matches_most_similar_conflict(self):
        candidates = [
            {
                "id": "mem_workflow",
                "type": "workflow",
                "scope": "project",
                "project": "/tmp/project",
                "content": "Run targeted tests before memory changes.",
                "score": 0.9,
                "status": "promote",
                "tags": ["workflow"],
                "evidence": [{"event_id": "event_1"}, {"event_id": "event_2"}],
            }
        ]
        memory_cards = [
            {
                "id": "mem_less_similar",
                "scope": "project",
                "project": "/tmp/project",
                "memory_type": "workflow",
                "summary": "Document release notes after deployment.",
                "status": "active",
            },
            {
                "id": "mem_more_similar",
                "scope": "project",
                "project": "/tmp/project",
                "memory_type": "workflow",
                "summary": "Run targeted tests before changing memory logic.",
                "status": "active",
            },
        ]

        queue = build_review_queue(candidates, memory_cards)

        self.assertEqual(queue[0]["quality_signals"]["matched_memory_id"], "mem_more_similar")
        self.assertEqual(queue[0]["dream_analysis"]["matched_memory_id"], "mem_more_similar")

    def test_dream_from_events_keeps_live_memory_baseline_clean_and_general(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "memory"
            project = "/tmp/project"
            code_dump = "def helper():\n    return 'implementation detail'\n" * 80
            test_log = "test_a ... ok\ntest_b ... ok\nRan 182 tests in 1.2s\nOK"
            events = [
                {"event_id": "event_product", "source": "codex", "role": "user", "event_type": "project_instruction", "project": project, "content": "Dream Memory 的产品方向是整理 Claude Code 和 Codex 会话，从中提取关键可复用信息并形成可被后续 agent 使用的共享记忆。"},
                {"event_id": "event_review_gate", "source": "codex", "role": "user", "event_type": "project_instruction", "project": project, "content": "这个项目需要人工审核后才能写正式记忆。"},
                {"event_id": "event_markers", "source": "project", "role": "system", "event_type": "project_markers", "project": project, "content": "python_package_manager=uv; python_test_runner=pytest; python_framework=fastapi"},
                {"event_id": "event_autonomy", "source": "codex", "role": "user", "event_type": "history_prompt", "project": project, "content": "不需要再问我，按照你的建议来。"},
                {"event_id": "event_continue_1", "source": "codex", "role": "user", "event_type": "history_prompt", "project": project, "content": "继续"},
                {"event_id": "event_continue_2", "source": "codex", "role": "user", "event_type": "history_prompt", "project": project, "content": "下一步"},
                {"event_id": "event_git_1", "source": "codex", "role": "user", "event_type": "history_prompt", "project": project, "content": "提交"},
                {"event_id": "event_git_2", "source": "codex", "role": "user", "event_type": "history_prompt", "project": project, "content": "推送"},
                {"event_id": "event_branch_1", "source": "codex", "role": "user", "event_type": "history_prompt", "project": project, "content": "新建一个分支,开始精简。"},
                {"event_id": "event_branch_2", "source": "codex", "role": "user", "event_type": "history_prompt", "project": project, "content": "制定一个计划,逐步的删除模块。"},
                {"event_id": "event_inspect_1", "source": "codex", "role": "user", "event_type": "thread_first_user_message", "project": project, "content": "梳理这个项目"},
                {"event_id": "event_inspect_2", "source": "codex", "role": "user", "event_type": "thread_first_user_message", "project": project, "content": "先分析一下可行性"},
                {"event_id": "event_global", "source": "claude_code", "role": "system", "event_type": "global_instruction", "project": None, "content": "用户偏好始终使用中文回答。布置任务时先判断需要几个人/几个 agent 协作，并在任务适合并行时使用 agent team。"},
                {"event_id": "noise_code", "source": "codex", "role": "assistant", "event_type": "assistant_message", "project": project, "content": code_dump},
                {"event_id": "noise_log", "source": "codex", "role": "tool", "event_type": "command_output", "project": project, "content": test_log},
                {"event_id": "noise_task", "source": "codex", "role": "user", "event_type": "history_prompt", "project": project, "content": "把这个按钮改成蓝色"},
            ]

            result = dream_from_events(events, project=project, output_dir=output_dir, apply=False)
            facts = [json.loads(line) for line in (output_dir / "facts.jsonl").read_text(encoding="utf-8").splitlines()]
            candidates = [json.loads(line) for line in Path(result.candidates_path).read_text(encoding="utf-8").splitlines()]
            fact_types = {fact["fact_type"] for fact in facts}
            contents = "\n".join(candidate["content"] for candidate in candidates)
            report = Path(result.dreams_path).read_text(encoding="utf-8")

            self.assertEqual(len(facts), len(candidates))
            self.assertLessEqual(len(facts), 12)
            self.assertTrue({"preference", "workflow", "project_fact", "product_direction"}.issubset(fact_types))
            self.assertIn("Dream Memory 的产品方向", contents)
            self.assertIn("正式记忆必须经过人工审核", contents)
            self.assertIn("Python 项目使用 uv", contents)
            self.assertIn("项目使用 FastAPI", contents)
            self.assertIn("用户偏好始终使用中文回答", contents)
            self.assertIn("几个 agent 协作", contents)
            self.assertIn("继续/做/下一步", contents)
            self.assertIn("仓库维护", contents)
            self.assertIn("独立分支", contents)
            self.assertIn("真实仓库结构", contents)
            self.assertNotIn("implementation detail", contents)
            self.assertNotIn("Ran 182 tests", contents)
            self.assertNotIn("按钮改成蓝色", contents)
            self.assertIn(f"- Facts extracted: {len(facts)}", report)
            self.assertIn("- Facts by type:", report)
            self.assertIn("- Quality tiers:", report)


    def test_dream_from_events_report_includes_fact_and_quality_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "memory"
            events = [
                {"event_id": "event_1", "source": "project", "role": "system", "event_type": "project_markers", "project": "/tmp/project", "content": "python_package_manager=uv; python_test_runner=pytest; python_framework=fastapi"},
                {"event_id": "event_2", "source": "codex", "role": "user", "event_type": "history_prompt", "project": "/tmp/project", "content": "这个项目需要人工审核后才能写正式记忆"},
            ]

            result = dream_from_events(events, project="/tmp/project", output_dir=output_dir, apply=False)
            report = Path(result.dreams_path).read_text(encoding="utf-8")

            self.assertIn("## Fact Diagnostics", report)
            self.assertIn("Facts extracted:", report)
            self.assertIn("## Evidence Quality", report)
            self.assertIn("project_fact", report)
            self.assertIn("workflow", report)

    def test_dream_from_events_writes_explainable_dream_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / ".dream" / "memory"
            candidates = [
                {
                    "id": "mem_workflow",
                    "type": "workflow",
                    "scope": "project",
                    "project": normalize_project_path("/tmp/project"),
                    "content": "Run targeted tests before memory changes.",
                    "score": 0.9,
                    "status": "promote",
                    "tags": ["workflow"],
                    "evidence": [{"event_id": "event_1"}, {"event_id": "event_2"}],
                },
                {
                    "id": "mem_task",
                    "type": "requirement",
                    "scope": "project",
                    "project": normalize_project_path("/tmp/project"),
                    "content": "删除首页水印按钮",
                    "score": 0.95,
                    "status": "promote",
                    "tags": ["task"],
                    "evidence": [{"event_id": "event_3"}],
                },
            ]

            result = dream_from_events(
                [{"event_id": "event_1", "source": "codex", "role": "user", "content": "memory input"}],
                project="/tmp/project",
                output_dir=output_dir,
                agent_candidates=candidates,
                agent_mode=True,
            )

            report = Path(result.dreams_path).read_text(encoding="utf-8")
            self.assertIn("## Promotion Policy", report)
            self.assertIn("## Action Summary", report)
            self.assertIn("## Create", report)
            self.assertIn("## Reject", report)
            self.assertIn("dream_score=", report)
            self.assertIn("reasons:", report)
            self.assertIn("penalties:", report)
            self.assertIn("one-off task", report)
            preview = Path(result.memory_preview_path).read_text(encoding="utf-8")
            written_candidates = [
                json.loads(line)
                for line in Path(result.candidates_path).read_text(encoding="utf-8").splitlines()
            ]
            by_id = {candidate["id"]: candidate for candidate in written_candidates}
            self.assertEqual(by_id["mem_task"]["status"], "reject")
            self.assertEqual(by_id["mem_task"]["dream_analysis"]["suggested_action"], "reject")
            self.assertNotIn("删除首页水印按钮", preview)
            self.assertEqual(result.rejected_count, 1)

    def test_apply_reviewed_memory_writes_memory_cards_and_markdown_projection(self):
        reviewed = [{
            "candidate_id": "cand_1",
            "status": "approved",
            "reviewer": "user",
            "notes": "looks good",
            "memory_updates": [{
                "id": "mem_1",
                "scope": "project",
                "project": "/tmp/project",
                "memory_type": "decision",
                "summary": "项目目标是 Claude Code 风格的本地研发助手。",
                "evidence_refs": ["event_1"],
                "approved_by": "user",
                "approved_at": "2026-07-05T00:00:00Z",
                "status": "active",
                "retrieval_hints": ["claude code"],
            }],
        }]

        cards, markdown = apply_reviewed_memory(reviewed, existing_cards=[])

        self.assertEqual(cards[0]["summary"], "项目目标是 Claude Code 风格的本地研发助手。")
        self.assertIn("Claude Code 风格", markdown)

    def test_build_agent_context_prioritizes_project_then_user_then_global(self):
        cards = [
            {"id": "mem_1", "scope": "global", "project": None, "memory_type": "workflow", "summary": "正式记忆必须人工审核。", "retrieval_hints": [], "status": "active"},
            {"id": "mem_2", "scope": "user", "project": None, "memory_type": "preference", "summary": "用户偏好中文回答。", "retrieval_hints": [], "status": "active"},
            {"id": "mem_3", "scope": "project", "project": "/tmp/project", "memory_type": "decision", "summary": "项目目标是 Claude Code 风格助手。", "retrieval_hints": [], "status": "active"},
        ]

        context = build_agent_context(cards, project="/tmp/project", limit=3)

        self.assertEqual(context["items"][0]["id"], "mem_3")
        self.assertEqual(context["items"][1]["id"], "mem_2")
        self.assertEqual(context["items"][2]["id"], "mem_1")

    def test_build_agent_context_boosts_autonomy_and_review_gate_intents(self):
        cards = [
            {"id": "branch", "scope": "user", "memory_type": "workflow", "summary": "做大规模功能精简或删除模块时，用户偏好先新建独立分支。", "retrieval_hints": ["branching"], "status": "active"},
            {"id": "autonomy", "scope": "user", "memory_type": "preference", "summary": "用户偏好在上下文清楚时由助手按判断直接推进，不要反复询问确认。", "retrieval_hints": ["autonomy", "direct-execution"], "status": "active"},
            {"id": "review_gate", "scope": "project", "project": "/tmp/project", "memory_type": "workflow", "summary": "正式记忆必须经过人工审核，只有审核通过的候选才允许写入长期记忆。", "retrieval_hints": ["review-gate", "memory-safety"], "status": "active"},
        ]

        autonomy_context = build_agent_context(cards, project="/tmp/project", task="不需要再问我，按照你的建议来", limit=1)
        review_context = build_agent_context(cards, project="/tmp/project", task="实现 Dream Memory 记忆抽取和人工审核", limit=1)

        self.assertEqual(autonomy_context["items"][0]["id"], "autonomy")
        self.assertEqual(review_context["items"][0]["id"], "review_gate")

    def test_build_agent_context_boosts_project_marker_memories_for_execution_tasks(self):
        cards = [
            {"id": "uv", "scope": "project", "project": "/tmp/project", "memory_type": "workflow", "summary": "Python 项目使用 uv 进行包管理和命令执行。", "retrieval_hints": ["package-manager", "uv", "python"], "status": "active"},
            {"id": "pytest", "scope": "project", "project": "/tmp/project", "memory_type": "workflow", "summary": "Python 测试使用 pytest，验证时应优先运行对应测试命令。", "retrieval_hints": ["testing", "python", "pytest"], "status": "active"},
            {"id": "fastapi", "scope": "project", "project": "/tmp/project", "memory_type": "project_fact", "summary": "项目使用 FastAPI 作为 Python Web 框架。", "retrieval_hints": ["framework", "python", "fastapi"], "status": "active"},
            {"id": "generic", "scope": "user", "memory_type": "workflow", "summary": "用户要求梳理项目时先读取真实仓库结构。", "retrieval_hints": ["repo-inspection"], "status": "active"},
        ]

        test_context = build_agent_context(cards, project="/tmp/project", task="跑测试并验证", limit=1)
        deps_context = build_agent_context(cards, project="/tmp/project", task="安装依赖并运行命令", limit=1)
        backend_context = build_agent_context(cards, project="/tmp/project", task="修复 FastAPI 后端接口", limit=1)

        self.assertEqual(test_context["items"][0]["id"], "pytest")
        self.assertEqual(deps_context["items"][0]["id"], "uv")
        self.assertEqual(backend_context["items"][0]["id"], "fastapi")

    def test_build_agent_context_explains_relevance_diagnostics(self):
        cards = [
            {"id": "pytest", "scope": "project", "project": "/tmp/project", "memory_type": "workflow", "summary": "Python 测试使用 pytest，验证时应优先运行对应测试命令。", "retrieval_hints": ["testing", "pytest"], "status": "active"},
            {"id": "language", "scope": "user", "memory_type": "preference", "summary": "用户偏好中文回答。", "retrieval_hints": ["language"], "status": "active"},
        ]

        context = build_agent_context(cards, project="/tmp/project", task="跑测试并验证", limit=2)
        markdown = render_context_markdown(context)

        self.assertEqual(context["diagnostics"][0]["id"], "pytest")
        self.assertEqual(context["diagnostics"][0]["reason"], "intent_alias_match")
        self.assertGreater(context["diagnostics"][0]["relevance"], 0)
        self.assertIn("rank_reason=intent_alias_match", markdown)

    def test_build_agent_context_boosts_keyword_intent_matches_for_short_tasks(self):
        cards = [
            {"id": "git", "scope": "user", "memory_type": "workflow", "summary": "用户经常用简短指令要求仓库维护（拉取代码、查看分支、提交、推送、切分支、调版本）；应直接检查真实 git 状态并执行。", "retrieval_hints": ["git", "repo-maintenance"], "status": "active"},
            {"id": "review", "scope": "user", "memory_type": "workflow", "summary": "用户要求梳理、审查或分析项目时，期望先读取真实仓库结构。", "retrieval_hints": ["repo-inspection"], "status": "active"},
            {"id": "continue", "scope": "user", "memory_type": "workflow", "summary": "用户常用继续推进已确认路线。", "retrieval_hints": ["continuation"], "status": "active"},
        ]

        submit_context = build_agent_context(cards, project="/tmp/project", task="提交", limit=1)
        inspect_context = build_agent_context(cards, project="/tmp/project", task="梳理这个项目", limit=1)
        continue_context = build_agent_context(cards, project="/tmp/project", task="继续", limit=1)

        self.assertEqual(submit_context["items"][0]["id"], "git")
        self.assertEqual(inspect_context["items"][0]["id"], "review")
        self.assertEqual(continue_context["items"][0]["id"], "continue")

    def test_build_agent_context_prioritizes_task_relevant_memory(self):
        cards = [
            {"id": "workflow", "scope": "project", "project": "/tmp/project", "memory_type": "workflow", "summary": "部署前必须先跑 smoke 测试。", "retrieval_hints": ["deploy", "smoke"], "tags": ["testing"], "status": "active"},
            {"id": "decision", "scope": "project", "project": "/tmp/project", "memory_type": "decision", "summary": "项目目标是 Claude Code 风格助手。", "retrieval_hints": ["claude-code"], "tags": ["product"], "status": "active"},
            {"id": "user", "scope": "user", "project": None, "memory_type": "preference", "summary": "用户偏好中文回答。", "retrieval_hints": ["language"], "status": "active"},
        ]

        context = build_agent_context(cards, project="/tmp/project", task="准备部署并执行 smoke 测试", limit=2)

        self.assertEqual([item["id"] for item in context["items"]], ["workflow", "user"])
        self.assertEqual(context["task"], "准备部署并执行 smoke 测试")

    def test_extract_atomic_facts_promotes_autonomy_preference(self):
        events = [
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/project", "content": "不需要再问我，按照你的建议来"},
        ]

        facts = extract_atomic_facts(events, project="/tmp/project")
        candidates = build_candidates_from_facts(facts)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["type"], "preference")
        self.assertEqual(candidates[0]["scope"], "user")
        self.assertIn("直接推进", candidates[0]["content"])



    def test_extract_atomic_facts_promotes_project_marker_package_managers(self):
        events = [{
            "event_id": "event_markers",
            "source": "project",
            "session_id": "project-markers:/tmp/project",
            "project": "/tmp/project",
            "role": "system",
            "event_type": "project_markers",
            "content": "python_package_manager=uv; frontend_package_manager=pnpm",
            "metadata": {},
        }]

        facts = extract_atomic_facts(events, project="/tmp/project")
        candidates = build_candidates_from_facts(facts)
        contents = "\n".join(candidate["content"] for candidate in candidates)

        self.assertIn("Python 项目使用 uv", contents)
        self.assertIn("前端项目使用 pnpm", contents)

    def test_extract_atomic_facts_promotes_project_marker_test_runner_and_framework(self):
        events = [{
            "event_id": "event_markers",
            "source": "project",
            "session_id": "project-markers:/tmp/project",
            "project": "/tmp/project",
            "role": "system",
            "event_type": "project_markers",
            "content": "python_package_manager=uv; python_test_runner=unittest; python_framework=fastapi",
            "metadata": {},
        }]

        facts = extract_atomic_facts(events, project="/tmp/project")
        candidates = build_candidates_from_facts(facts)
        contents = "\n".join(candidate["content"] for candidate in candidates)

        self.assertIn("Python 测试使用 unittest", contents)
        self.assertIn("项目使用 FastAPI", contents)

    def test_extract_atomic_facts_prefers_pytest_when_marker_contains_both_test_runners(self):
        events = [{
            "event_id": "event_markers",
            "source": "project",
            "session_id": "project-markers:/tmp/project",
            "project": "/tmp/project",
            "role": "system",
            "event_type": "project_markers",
            "content": "python_package_manager=uv; python_test_runner=unittest; python_test_runner=pytest; python_framework=fastapi",
            "metadata": {},
        }]

        facts = extract_atomic_facts(events, project="/tmp/project")
        candidates = build_candidates_from_facts(facts)
        contents = "\n".join(candidate["content"] for candidate in candidates)

        self.assertIn("Python 测试使用 pytest", contents)
        self.assertNotIn("Python 测试使用 unittest", contents)

    def test_extract_atomic_facts_canonicalizes_language_preference_variants(self):
        events = [
            {"source": "codex", "role": "user", "event_type": "history_prompt", "content": "用户偏好中文回答。"},
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "content": "请始终用中文回答我。"},
        ]

        facts = extract_atomic_facts(events, project=None)
        candidates = build_candidates_from_facts(facts)

        self.assertEqual(len([candidate for candidate in candidates if "中文回答" in candidate["content"]]), 1)
        self.assertEqual(candidates[0]["content"], "用户偏好中文回答。")

    def test_extract_atomic_facts_promotes_real_flow_warning_as_pitfall(self):
        events = [{
            "source": "codex",
            "role": "user",
            "event_type": "history_prompt",
            "project": "/tmp/project",
            "content": "以后不要只看 API 返回成功就说修好了，登录跳转和退出状态这类问题必须真实跑 UI 流程验证。",
        }]

        facts = extract_atomic_facts(events, project="/tmp/project")
        candidates = build_candidates_from_facts(facts)

        self.assertEqual(candidates[0]["type"], "pitfall")
        self.assertEqual(candidates[0]["scope"], "user")
        self.assertIn("不要只看 API 返回成功", candidates[0]["content"])
        self.assertIn("explicit", candidates[0]["tags"])

    def test_extract_atomic_facts_promotes_rejected_auto_apply_memory_option(self):
        events = [{
            "source": "codex",
            "role": "user",
            "event_type": "history_prompt",
            "project": "/tmp/project",
            "content": "之前讨论过不要把未经审核的候选自动写入 MEMORY.md，这个方案风险太高。",
        }]

        facts = extract_atomic_facts(events, project="/tmp/project")
        candidates = build_candidates_from_facts(facts)

        self.assertEqual(candidates[0]["type"], "rejected_option")
        self.assertEqual(candidates[0]["scope"], "project")
        self.assertIn("不要把未经审核的候选自动写入", candidates[0]["content"])
        self.assertIn("explicit", candidates[0]["tags"])

    def test_project_instruction_single_event_is_reviewable(self):
        events = [{
            "event_id": "event_agents",
            "source": "codex",
            "session_id": "s1",
            "project": "/tmp/project",
            "role": "system",
            "event_type": "project_instruction",
            "content": "如果后端使用的是 Python，则使用 uv 进行包管理\n前端使用 pnpm 进行管理",
        }]

        facts = extract_atomic_facts(events, project="/tmp/project")
        candidates = build_candidates_from_facts(facts)
        queue = build_review_queue(candidates, [])
        contents = "\n".join(candidate["content"] for candidate in candidates)

        self.assertIn("Python 后端使用 uv", contents)
        self.assertIn("前端使用 pnpm", contents)
        self.assertTrue(any(candidate["type"] == "workflow" for candidate in candidates))
        self.assertIn("explicit", candidates[0]["tags"])
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["dream_analysis"]["required_evidence_count"], 1)

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

    def test_extract_atomic_facts_promotes_memory_product_direction_and_review_gate(self):
        events = [
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/dream", "content": "我希望他能整理 claude code 和 codex 的会话聊天，从中提取出关键的信息，放到记忆中，让这些 agent 都能看到，先不用写代码，现在是探讨时间"},
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/dream", "content": "认同,但是最后写入记忆时要人工审核,只有通过的才会写入到记忆中"},
        ]

        facts = extract_atomic_facts(events, project="/tmp/dream")
        candidates = build_candidates_from_facts(facts)
        by_type = {candidate["type"]: candidate for candidate in candidates}

        self.assertIn("product_direction", by_type)
        self.assertIn("共享记忆", by_type["product_direction"]["content"])
        self.assertIn("explicit", by_type["product_direction"]["tags"])
        self.assertIn("workflow", by_type)
        self.assertIn("人工审核", by_type["workflow"]["content"])
        self.assertIn("explicit", by_type["workflow"]["tags"])
        self.assertFalse(any(candidate["type"] == "requirement" for candidate in candidates))

    def test_extract_atomic_facts_ignores_short_one_off_requirements(self):
        events = [
            {"source": "codex", "role": "user", "event_type": "history_prompt", "project": "/tmp/project", "content": "快麦上新同步这个不需要"},
            {"source": "codex", "role": "user", "event_type": "history_prompt", "project": "/tmp/project", "content": "2.jpg 模板的脚本中名字不需要箭头了"},
            {"source": "codex", "role": "user", "event_type": "history_prompt", "project": "/tmp/project", "content": "不要用图片"},
            {"source": "codex", "role": "user", "event_type": "history_prompt", "project": "/tmp/project", "content": "用户偏好中文回答。"},
        ]

        facts = extract_atomic_facts(events, project="/tmp/project")
        candidates = build_candidates_from_facts(facts)
        contents = "\n".join(candidate["content"] for candidate in candidates)

        self.assertIn("用户偏好中文回答", contents)
        self.assertNotIn("快麦上新同步", contents)
        self.assertNotIn("名字不需要箭头", contents)
        self.assertNotIn("不要用图片", contents)

    def test_extract_atomic_facts_ignores_transient_questions(self):
        events = [
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/project", "content": "这个项目哪里有需要 ai 的地方吗"},
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/project", "content": "磁盘空间充足（需要50.3GB，可用53.1GB）这个是怎么推断出来的"},
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/project", "content": "用户偏好中文回答。"},
        ]

        facts = extract_atomic_facts(events, project="/tmp/project")
        candidates = build_candidates_from_facts(facts)
        contents = "\n".join(candidate["content"] for candidate in candidates)

        self.assertIn("用户偏好中文回答", contents)
        self.assertNotIn("哪里有需要 ai", contents)
        self.assertNotIn("怎么推断", contents)

    def test_extract_atomic_facts_ignores_codex_ui_context_wrappers(self):
        events = [
            {"source": "codex", "role": "user", "event_type": "rollout_message", "project": "/tmp/project", "content": "# In app browser:\n- Current URL: file:///tmp/demo.html\n\n## My request for Codex:\n继续"},
            {"source": "codex", "role": "user", "event_type": "rollout_message", "project": "/tmp/project", "content": "# Files mentioned by the user:\n\n## 珠宝客户界面.pdf: /tmp/file.pdf\n\n## My request for Codex:\n侧边栏的样式改成这种风格"},
            {"source": "codex", "role": "user", "event_type": "rollout_message", "project": "/tmp/project", "content": "用户偏好中文回答。"},
        ]

        facts = extract_atomic_facts(events, project="/tmp/project")
        candidates = build_candidates_from_facts(facts)
        contents = "\n".join(candidate["content"] for candidate in candidates)

        self.assertIn("用户偏好中文回答", contents)
        self.assertNotIn("In app browser", contents)
        self.assertNotIn("Files mentioned", contents)
        self.assertNotIn("珠宝客户界面", contents)

    def test_extract_atomic_facts_ignores_low_value_transcript_wrappers(self):
        events = [
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/project", "content": "Exit code 1\nTraceback (most recent call last):\nFileNotFoundError: missing"},
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/project", "content": "File created successfully at: /tmp/plan.md (file state is current in your context — no need to Read it back)"},
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/project", "content": "1. TASK: Inspect the current project structure\n2. EXPECTED OUTCOME: Return a concise map"},
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/project", "content": "用户偏好中文回答。"},
        ]

        facts = extract_atomic_facts(events, project="/tmp/project")
        candidates = build_candidates_from_facts(facts)
        contents = "\n".join(candidate["content"] for candidate in candidates)

        self.assertIn("用户偏好中文回答", contents)
        self.assertNotIn("Traceback", contents)
        self.assertNotIn("File created successfully", contents)
        self.assertNotIn("TASK:", contents)

    def test_extract_atomic_facts_ignores_agents_docs_font_snippets_and_tree_outputs(self):
        agents_doc = "# AGENTS.md\n\nThis file provides guidance to Codex (Codex.ai/code) when working with code in this repository.\n\n## 项目概述\n" + ("### 常用命令\ncd FrontEnd\nnpm run dev\n" * 60)
        font_snippet = "font_small = ImageFont.truetype(msyh_path, 32)\nfont_top_time = ImageFont.truetype(msyh_path, 42)\n" + ("draw.text((1, 1), 'x', font=font_small)\n" * 60)
        tree_dump = "===docs tree===\n" + "\n".join([f"/Users/gemaolin/code/FastapiAdmin/docs/superpowers/specs/file_{i}.md" for i in range(40)])
        research_workflow = "Run the \"deep-research\" workflow.\n\nDeep research harness — fan-out web searches, fetch sources, adversarially verify claims.\n" + ("research step\n" * 80)
        events = [
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/project", "content": agents_doc},
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/project", "content": font_snippet},
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/project", "content": tree_dump},
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/project", "content": research_workflow},
            {"source": "project", "role": "system", "event_type": "project_markers", "project": "/tmp/project", "content": "python_package_manager=uv; python_test_runner=pytest; python_framework=fastapi"},
        ]

        facts = extract_atomic_facts(events, project="/tmp/project")
        statements = "\n".join(str(fact.get("statement") or "") for fact in facts)

        self.assertNotIn("This file provides guidance to Codex", statements)
        self.assertNotIn("ImageFont.truetype", statements)
        self.assertNotIn("docs tree", statements)
        self.assertNotIn("deep-research", statements)
        self.assertIn("Python 项目使用 uv", statements)
        self.assertIn("Python 测试使用 pytest", statements)
        self.assertIn("项目使用 FastAPI", statements)

    def test_extract_atomic_facts_ignores_arrow_metrics_todo_and_compact_trees(self):
        arrow_snippet = "# 1.1 添加姓名右侧箭头图片\nif arrow_img:\n    arrow_height = 40\n    arrow_resized = arrow_img.resize((10, 40))\nfont_arrow = ImageFont.truetype(msyh_path, 35)"
        todo_report = "⚠️ 待完善\n\n  - ⚠️ TypeScript类型系统 (~60个类型错误)\n  - ⚠️ ESLint配置 (需要迁移到v9)"
        compact_tree = "api\nApp.vue\nassets\ncomponents\ncomposables\nconfig\nmain.ts\nrouter\nstores\nstyles\ntest-setup.ts\ntypes\nutils\nviews\n---API---\n__tests__\nai.ts\ndesign.ts\nfeedback.ts\nobstacle.ts"
        disk_analysis = "=== 磁盘空间推断详解 ===\n当前磁盘空间: 53GB\n推断逻辑和错误分析。"
        events = [
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/project", "content": arrow_snippet},
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/project", "content": todo_report},
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/project", "content": compact_tree},
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/project", "content": disk_analysis},
            {"source": "project", "role": "system", "event_type": "project_markers", "project": "/tmp/project", "content": "python_package_manager=uv; python_test_runner=pytest; python_framework=fastapi"},
        ]

        facts = extract_atomic_facts(events, project="/tmp/project")
        statements = "\n".join(str(fact.get("statement") or "") for fact in facts)

        self.assertNotIn("箭头图片", statements)
        self.assertNotIn("TypeScript类型系统", statements)
        self.assertNotIn("App.vue", statements)
        self.assertNotIn("磁盘空间", statements)
        self.assertIn("Python 项目使用 uv", statements)
        self.assertIn("Python 测试使用 pytest", statements)
        self.assertIn("项目使用 FastAPI", statements)

    def test_extract_atomic_facts_ignores_short_script_ls_metrics_and_test_logs(self):
        short_image_script = "\n".join([
            "\"\"\"",
            "修改账单图片中的文字信息",
            "\"\"\"",
            "from PIL import Image, ImageDraw, ImageFont",
            "import os",
            "def edit_bill_image(input_image_path, output_image_path):",
            "    draw = ImageDraw.Draw(Image.open(input_image_path))",
            "    return True",
        ])
        ls_output = "\n".join([
            "total 672",
            "drwxr-xr-x  12 user staff 384 Jul 5 .",
            "drwxr-xr-x  30 user staff 960 Jul 5 ..",
            "-rw-r--r-- 1 user staff 102 .gitignore",
            "-rw-r--r-- 1 user staff 744 pyproject.toml",
            "drwxr-xr-x 3 user staff 96 src",
            "drwxr-xr-x 18 user staff 576 tests",
            "-rw-r--r-- 1 user staff 308329 uv.lock",
        ])
        metrics_dump = "=== 内存使用评估 ===\n总内存: 16GB\n当前 Python 进程: 15MB\n=== 磁盘空间推断错误分析 ===\n错误的推断过程和正确推断。"
        test_log = "============================= test session starts ==============================\ncollected 14 items\napps/test_demo.py::test_a PASSED [  7%]\napps/test_demo.py::test_b PASSED [ 14%]"
        claude_init = "Please analyze this codebase and create a CLAUDE.md file, which will be given to future instances of Claude Code to operate in this repository.\nWhat to add:\n1. Commands that will be commonly used."
        events = [
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/project", "content": short_image_script},
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/project", "content": ls_output},
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/project", "content": metrics_dump},
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/project", "content": test_log},
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/project", "content": claude_init},
            {"source": "project", "role": "system", "event_type": "project_markers", "project": "/tmp/project", "content": "python_package_manager=uv; python_test_runner=pytest; python_framework=fastapi"},
        ]

        facts = extract_atomic_facts(events, project="/tmp/project")
        statements = "\n".join(str(fact.get("statement") or "") for fact in facts)

        self.assertNotIn("edit_bill_image", statements)
        self.assertNotIn("pyproject.toml", statements)
        self.assertNotIn("内存使用评估", statements)
        self.assertNotIn("test session starts", statements)
        self.assertNotIn("create a CLAUDE.md", statements)
        self.assertIn("Python 项目使用 uv", statements)
        self.assertIn("Python 测试使用 pytest", statements)
        self.assertIn("项目使用 FastAPI", statements)

    def test_extract_atomic_facts_drops_long_one_off_requirements_and_project_docs(self):
        one_off_security_task = "请修复 blog-ai 项目中的 JWT Token 存储安全问题：\n\n**目标**：将 JWT token 从 localStorage 移至 httpOnly cookie。\n" + ("修改文件和实现细节。\n" * 40)
        project_doc = "# AGENTS.md\n\nThis file provides guidance to Codex when working with code in this repository.\n" + ("## 常用命令\nuv run python -m unittest\n" * 40)
        tree_dump = "api\nApp.vue\nassets\ncomponents\ncomposables\nconfig\nmain.ts\nrouter\nstores\nstyles\ntest-setup.ts\ntypes\nutils\nviews\n---API---\n__tests__\nai.ts\ndesign.ts\nfeedback.ts\nobstacle.ts\norder.ts\ntemplate.ts\nuser.ts"
        durable_pref = "用户偏好在上下文清楚时由助手按判断直接推进，不要反复询问确认。"
        events = [
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/project", "content": one_off_security_task},
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/project", "content": project_doc},
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/project", "content": tree_dump},
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/project", "content": durable_pref},
            {"source": "project", "role": "system", "event_type": "project_markers", "project": "/tmp/project", "content": "python_package_manager=uv; python_test_runner=pytest; python_framework=fastapi"},
        ]

        facts = extract_atomic_facts(events, project="/tmp/project")
        statements = "\n".join(str(fact.get("statement") or "") for fact in facts)

        self.assertNotIn("JWT Token", statements)
        self.assertNotIn("This file provides guidance", statements)
        self.assertNotIn("App.vue", statements)
        self.assertIn("判断直接推进", statements)
        self.assertIn("Python 项目使用 uv", statements)
        self.assertIn("Python 测试使用 pytest", statements)
        self.assertIn("项目使用 FastAPI", statements)

    def test_extract_atomic_facts_skips_long_generic_docs_but_keeps_project_markers(self):
        long_agents_doc = "# AGENTS.md\n\nThis file provides guidance to Codex when working with code in this repository.\n" + ("Use Python, Codex, and patch carefully.\n" * 120)
        long_readme = "# 马术障碍赛路线设计器\n\n## 技术栈\n- Python 3.12+\n- Django\n" + ("项目说明和功能列表。\n" * 160)
        events = [
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/project", "content": long_agents_doc},
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/project", "content": long_readme},
            {"source": "project", "role": "system", "event_type": "project_markers", "project": "/tmp/project", "content": "python_package_manager=uv; python_test_runner=pytest; python_framework=fastapi"},
        ]

        facts = extract_atomic_facts(events, project="/tmp/project")
        statements = "\n".join(str(fact.get("statement") or "") for fact in facts)

        self.assertNotIn("This file provides guidance to Codex", statements)
        self.assertNotIn("马术障碍赛路线设计器", statements)
        self.assertIn("Python 项目使用 uv", statements)
        self.assertIn("Python 测试使用 pytest", statements)
        self.assertIn("项目使用 FastAPI", statements)

    def test_extract_atomic_facts_ignores_mode_skill_font_and_report_artifacts(self):
        mode_dump = "<ultrawork-mode>\n**MANDATORY**: You MUST say something first.\n" + ("[CODE RED] Maximum precision required.\n" * 80)
        skill_list = "## Skills\nA skill is a set of local instructions to follow that is stored in a `SKILL.md` file.\n" + ("- skill-name: description and file path\n" * 80)
        font_listing = "\n".join([f"/Users/gemaolin/code/temp/font_{i}.ttf" for i in range(40)] + [f"/Users/gemaolin/code/temp/.venv/lib/python3.10/site-packages/pkg_{i}.py" for i in range(40)])
        final_report = "# 知识图谱全面改进 - 最终总结\n\n## 🎯 项目目标达成\n" + ("| 任务 | 状态 | 交付物 |\n|---|---|---|\n" * 80)
        events = [
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/project", "content": mode_dump},
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/project", "content": skill_list},
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/project", "content": font_listing},
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/project", "content": final_report},
            {"source": "project", "role": "system", "event_type": "project_markers", "project": "/tmp/project", "content": "python_package_manager=uv; python_test_runner=pytest; python_framework=fastapi"},
        ]

        facts = extract_atomic_facts(events, project="/tmp/project")
        statements = "\n".join(str(fact.get("statement") or "") for fact in facts)

        self.assertNotIn("ultrawork-mode", statements)
        self.assertNotIn("## Skills", statements)
        self.assertNotIn("font_10.ttf", statements)
        self.assertNotIn("知识图谱全面改进", statements)
        self.assertIn("Python 项目使用 uv", statements)
        self.assertIn("Python 测试使用 pytest", statements)
        self.assertIn("项目使用 FastAPI", statements)

    def test_extract_atomic_facts_ignores_internal_context_and_script_artifacts(self):
        internal_context = "<codex_internal_context source=\"goal\">\nContinue working toward the active thread goal.\n<objective>继续</objective>\n</codex_internal_context>" + ("\nmore context" * 80)
        image_script = "\"\"\"\n修改账单图片中的文字信息\n\"\"\"\nfrom PIL import Image, ImageDraw, ImageFont\nimport os\n\ndef edit_bill_image(input_image_path, output_image_path):\n    return True\n" + ("    draw.text((1, 1), 'x')\n" * 80)
        pyproject_dump = "1\t[project]\n2\tname = \"backend\"\n3\tdependencies = [\n" + ("4\t    \"fastapi==0.1\",\n" * 80)
        absolute_file_listing = "\n".join([f"/Users/gemaolin/code/deepagent-project-advice/src/package/module_{i}.py" for i in range(80)])
        events = [
            {"source": "codex", "role": "user", "event_type": "rollout_message", "project": "/tmp/project", "content": internal_context},
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/project", "content": image_script},
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/project", "content": pyproject_dump},
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/project", "content": absolute_file_listing},
            {"source": "project", "role": "system", "event_type": "project_markers", "project": "/tmp/project", "content": "python_package_manager=uv; python_test_runner=pytest; python_framework=fastapi"},
        ]

        facts = extract_atomic_facts(events, project="/tmp/project")
        statements = "\n".join(str(fact.get("statement") or "") for fact in facts)

        self.assertNotIn("codex_internal_context", statements)
        self.assertNotIn("edit_bill_image", statements)
        self.assertNotIn("dependencies =", statements)
        self.assertNotIn("module_10.py", statements)
        self.assertIn("Python 项目使用 uv", statements)
        self.assertIn("Python 测试使用 pytest", statements)
        self.assertIn("项目使用 FastAPI", statements)

    def test_extract_atomic_facts_ignores_large_skill_insights_and_log_dumps(self):
        skill_dump = "Base directory for this skill: /tmp/skills/brainstorming\n\n# Brainstorming Ideas Into Designs\n" + ("Do NOT implement yet.\n" * 120)
        insights_dump = "The user just ran /insights to generate a usage report analyzing their Claude Code sessions.\nHere is the full insights data:\n" + ("{\"session_count\": 2}\n" * 120)
        browser_log = "欢迎使用 Fastapi Admin！\n[RouteGuard] 需要初始化动态路由\napi/v1/system/user/current/info Failed to load resource: the server responded with a status of 401\n" * 40
        xml_skill = "<purpose>\nCheck for GSD updates via npm.\n</purpose>\n<process>\n" + ("<step name=\"x\">do work</step>\n" * 80)
        events = [
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/project", "content": skill_dump},
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/project", "content": insights_dump},
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/project", "content": browser_log},
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/project", "content": xml_skill},
            {"source": "project", "role": "system", "event_type": "project_markers", "project": "/tmp/project", "content": "python_package_manager=uv; python_test_runner=pytest; python_framework=fastapi"},
        ]

        facts = extract_atomic_facts(events, project="/tmp/project")
        statements = "\n".join(str(fact.get("statement") or "") for fact in facts)

        self.assertNotIn("Brainstorming Ideas", statements)
        self.assertNotIn("full insights data", statements)
        self.assertNotIn("RouteGuard", statements)
        self.assertNotIn("<purpose>", statements)
        self.assertIn("Python 项目使用 uv", statements)
        self.assertIn("Python 测试使用 pytest", statements)
        self.assertIn("项目使用 FastAPI", statements)

    def test_extract_atomic_facts_ignores_code_and_directory_dump_events(self):
        code_dump = "\n".join([
            "1\tfrom __future__ import annotations",
            "2\timport json",
            "3\tclass Demo:",
            "4\t    def run(self):",
            "5\t        return True",
            "6\t# more source",
        ])
        listing_dump = "\n".join([
            "===backend root===",
            "__pycache__",
            "alembic",
            "main.py",
            "pyproject.toml",
            "uv.lock",
            "tests",
        ])
        events = [
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/project", "content": code_dump},
            {"source": "claude_code", "role": "user", "event_type": "transcript_message", "project": "/tmp/project", "content": listing_dump},
            {"source": "project", "role": "system", "event_type": "project_markers", "project": "/tmp/project", "content": "python_package_manager=uv; python_test_runner=pytest"},
        ]

        facts = extract_atomic_facts(events, project="/tmp/project")
        statements = "\n".join(str(fact.get("statement") or "") for fact in facts)

        self.assertNotIn("from __future__", statements)
        self.assertNotIn("backend root", statements)
        self.assertIn("Python 项目使用 uv", statements)
        self.assertIn("Python 测试使用 pytest", statements)

    def test_extract_atomic_facts_ignores_assistant_and_tool_process_output(self):
        events = [
            {"source": "claude_code", "role": "assistant", "event_type": "transcript_message", "project": "/tmp/project", "content": "我需要查看前端代码才能评估侧边栏设计。"},
            {"source": "claude_code", "role": "tool_result", "event_type": "transcript_message", "project": "/tmp/project", "content": "-rw-r--r-- 1 user staff 811 pyproject.toml\n-rw-r--r-- 1 user staff 503 requirements.txt\n-rw-r--r-- 1 user staff 95083 uv.lock"},
            {"source": "codex", "role": "user", "event_type": "history_prompt", "project": "/tmp/project", "content": "这个项目需要人工审核后才能写正式记忆"},
        ]

        facts = extract_atomic_facts(events, project="/tmp/project")
        candidates = build_candidates_from_facts(facts)
        contents = "\n".join(candidate["content"] for candidate in candidates)

        self.assertIn("正式记忆必须经过人工审核", contents)
        self.assertNotIn("我需要查看前端代码", contents)
        self.assertNotIn("pyproject.toml", contents)

    def test_extract_atomic_facts_drops_secret_like_content(self):
        events = [{
            "event_id": "event_1",
            "source": "codex",
            "session_id": "s1",
            "project": "/tmp/project",
            "role": "user",
            "event_type": "history_prompt",
            "content": "OPENAI_API_KEY=sk-secret-value",
        }]

        facts = extract_atomic_facts(events, project="/tmp/project")

        self.assertEqual(facts, [])

    def test_agent_prompt_rejects_project_state_as_memory_content(self):
        prompt = build_memory_extraction_prompt([
            {
                "source": "claude_code",
                "session_id": "s1",
                "project": "/tmp/project",
                "role": "system",
                "event_type": "project_state",
                "content": "Claude Code project state for /tmp/project",
            }
        ], project="/tmp/project")

        self.assertIn("reject it or omit it", prompt)


    def test_extract_atomic_facts_drops_credential_location_hints(self):
        events = [
            {
                "event_id": "secret_location",
                "source": "codex",
                "role": "user",
                "event_type": "history_prompt",
                "project": "/tmp/project",
                "content": "以后调用真实模型时，API key 放在 key.txt，记得读取这个文件。",
            }
        ]

        facts = extract_atomic_facts(events, project="/tmp/project")

        self.assertEqual(facts, [])

    def test_extract_atomic_facts_drops_other_project_markers_when_project_filter_is_set(self):
        events = [
            {
                "event_id": "other_project",
                "source": "project",
                "role": "system",
                "event_type": "project_markers",
                "project": "/tmp/other-project",
                "content": "python_package_manager=uv; python_test_runner=pytest; python_framework=django",
            }
        ]

        facts = extract_atomic_facts(events, project="/tmp/project")

        self.assertEqual(facts, [])


    def test_extract_atomic_facts_keeps_user_preferences_from_other_project_when_project_filter_is_set(self):
        events = [
            {
                "event_id": "other_project_pref",
                "source": "codex",
                "role": "user",
                "event_type": "history_prompt",
                "project": "/tmp/other-project",
                "content": "请始终用中文回答我。",
            }
        ]

        facts = extract_atomic_facts(events, project="/tmp/project")

        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["fact_type"], "preference")
        self.assertEqual(facts[0]["scope"], "user")
        self.assertIn("中文回答", facts[0]["statement"])

    def test_extract_atomic_facts_uses_project_argument_when_event_project_missing(self):
        events = [{
            "event_id": "event_1",
            "source": "codex",
            "session_id": "s1",
            "role": "user",
            "event_type": "history_prompt",
            "content": "这个项目需要人工审核后才能写正式记忆",
        }]

        facts = extract_atomic_facts(events, project="/tmp/project")

        self.assertTrue(any(fact["scope"] == "project" and fact["project"] == "/tmp/project" for fact in facts))

    def test_build_agent_context_excludes_other_project_cards_and_renders_markdown(self):
        cards = [
            {"id": "mem_other", "scope": "project", "project": "/tmp/other", "memory_type": "decision", "summary": "其他项目记忆。", "retrieval_hints": [], "status": "active"},
            {"id": "mem_project", "scope": "project", "project": "/tmp/project", "memory_type": "decision", "summary": "当前项目记忆。", "retrieval_hints": [], "status": "active"},
            {"id": "mem_user", "scope": "user", "project": None, "memory_type": "preference", "summary": "用户偏好中文回答。", "retrieval_hints": [], "status": "active"},
        ]

        context = build_agent_context(cards, project="/tmp/project", limit=5)
        markdown = render_context_markdown(context)

        self.assertEqual([item["id"] for item in context["items"]], ["mem_project", "mem_user"])
        self.assertIn("Relevant Memory", markdown)
        self.assertIn("当前项目记忆", markdown)
        self.assertNotIn("其他项目记忆", markdown)

    def test_build_candidates_from_facts_normalizes_duplicate_wording(self):
        facts = [
            {"id": "fact_1", "fact_type": "requirement", "statement": "项目需要人工审核后才能写正式记忆。", "scope": "project", "project": "/tmp/project", "evidence_refs": ["event_1"], "confidence": 0.8, "tags": ["requirement"]},
            {"id": "fact_2", "fact_type": "requirement", "statement": "项目需要人工审核后才能写正式记忆", "scope": "project", "project": "/tmp/project", "evidence_refs": ["event_2"], "confidence": 0.8, "tags": ["requirement"]},
        ]

        candidates = build_candidates_from_facts(facts)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(len(candidates[0]["evidence"]), 2)
        self.assertEqual(normalize_memory_text("项目需要人工审核后才能写正式记忆。"), normalize_memory_text("项目需要人工审核后才能写正式记忆"))

    def test_build_candidates_from_facts_rejects_raw_transcript_like_content(self):
        facts = [{
            "id": "fact_1",
            "fact_type": "requirement",
            "statement": "User: hello\nAssistant: hi\n```bash\npytest\n```\n" + "日志" * 300,
            "scope": "project",
            "project": "/tmp/project",
            "evidence_refs": ["event_1"],
            "confidence": 0.8,
            "tags": ["requirement"],
        }]

        self.assertEqual(build_candidates_from_facts(facts), [])

    def test_apply_reviewed_memory_accepts_web_review_payload_and_writes_ledger(self):
        reviewed = [{
            "candidate_id": "cand_1",
            "action": "approved",
            "edited_content": "用户偏好中文回答。",
            "reviewer": "user",
            "note": "明确偏好",
            "candidate": {
                "id": "cand_1",
                "type": "preference",
                "scope": "user",
                "project": None,
                "content": "用户偏好中文回答。",
                "evidence": [{"event_id": "event_1"}],
                "tags": ["language"],
            },
        }]

        cards, markdown, decisions = apply_reviewed_memory(reviewed, existing_cards=[], return_decisions=True)

        self.assertEqual(cards[0]["summary"], "用户偏好中文回答。")
        self.assertEqual(decisions[0]["status"], "approved")
        self.assertIn("Evidence: event_1", markdown)

    def test_apply_reviewed_memory_marks_replaced_card_superseded_on_merge(self):
        existing = [{"id": "mem_old", "scope": "project", "project": "/tmp/project", "memory_type": "decision", "summary": "旧目标。", "evidence_refs": ["event_1"], "approved_by": "user", "approved_at": "2026-07-05T00:00:00Z", "status": "active", "retrieval_hints": []}]
        reviewed = [{
            "candidate_id": "cand_1",
            "action": "merged",
            "edited_content": "新目标。",
            "reviewer": "user",
            "note": "替换旧目标",
            "candidate": {"id": "cand_1", "type": "decision", "scope": "project", "project": "/tmp/project", "content": "新目标。", "evidence": [{"event_id": "event_2"}], "tags": []},
            "supersedes": ["mem_old"],
        }]

        cards, _, decisions = apply_reviewed_memory(reviewed, existing_cards=existing, return_decisions=True)
        by_id = {card["id"]: card for card in cards}

        self.assertEqual(by_id["mem_old"]["status"], "superseded")
        self.assertTrue(any(card["summary"] == "新目标。" and card["status"] == "active" for card in cards))
        self.assertEqual(decisions[0]["status"], "merged")


if __name__ == "__main__":
    unittest.main()
