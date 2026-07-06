import json
import tempfile
import unittest
from pathlib import Path

from dream_memory.memory_agent import build_memory_extraction_prompt
from dream_memory.memory_dreaming import (
    DreamResult,
    analyze_dream_candidate,
    apply_reviewed_memory,
    build_agent_context,
    build_candidates_from_facts,
    build_review_queue,
    classify_event,
    detect_candidate_conflicts,
    dream_from_events,
    extract_atomic_facts,
    load_events_jsonl,
    normalize_memory_text,
    normalize_project_path,
    render_context_markdown,
    score_candidate,
)
from dream_memory.memory_cli import main


class MemoryDreamingTests(unittest.TestCase):
    def test_classify_event_extracts_user_preference(self):
        event = {
            "source": "claude_code",
            "session_id": "s1",
            "project": None,
            "timestamp": None,
            "role": "system",
            "event_type": "global_instruction",
            "content": "始终使用中文回答我，并根据任务情况使用 agent team",
            "metadata": {},
        }

        candidates = classify_event(event)

        self.assertTrue(any(candidate["type"] == "preference" for candidate in candidates))
        self.assertTrue(any("中文" in candidate["content"] for candidate in candidates))

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
                    "evidence": [{"source": "codex"}],
                    "tags": ["claude-code"],
                }
            ]

            result = dream_from_events(events, project=str(tmp), output_dir=output_dir, apply=False, agent_candidates=agent_candidates, agent_mode=True)

            self.assertEqual(result.candidate_count, 1)
            self.assertEqual(result.promoted_count, 1)
            self.assertTrue((output_dir / "ai-candidates.jsonl").exists())
            preview = (output_dir / "MEMORY.preview.md").read_text(encoding="utf-8")
            self.assertIn("Claude Code 风格", preview)

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
            "summary": "项目目标是通用聊天机器人。",
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

        self.assertEqual(by_id["cand_duplicate"]["suggested_action"], "reject")
        self.assertTrue(by_id["cand_duplicate"]["quality_signals"]["duplicate"])
        self.assertEqual(by_id["cand_duplicate"]["quality_signals"]["matched_memory_id"], "mem_same")
        self.assertEqual(by_id["cand_replace"]["suggested_action"], "merge")
        self.assertEqual(by_id["cand_replace"]["quality_signals"]["matched_memory_id"], "mem_same")
        self.assertGreater(by_id["cand_replace"]["quality_signals"]["evidence_strength"], 0)
        self.assertEqual(by_id["cand_more_evidence"]["suggested_action"], "needs_more_evidence")
        self.assertEqual(by_id["cand_more_evidence"]["quality_signals"]["evidence_strength"], 0)

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
            "evidence": [{"event_id": "event_1"}],
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

        self.assertEqual(queue[0]["suggested_action"], "reject")
        self.assertEqual(queue[0]["dream_analysis"]["suggested_action"], "reject")
        self.assertIn("one-off task", queue[0]["dream_analysis"]["penalties"])

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

    def test_build_agent_context_prioritizes_task_relevant_memory(self):
        cards = [
            {"id": "workflow", "scope": "project", "project": "/tmp/project", "memory_type": "workflow", "summary": "部署前必须先跑 smoke 测试。", "retrieval_hints": ["deploy", "smoke"], "tags": ["testing"], "status": "active"},
            {"id": "decision", "scope": "project", "project": "/tmp/project", "memory_type": "decision", "summary": "项目目标是 Claude Code 风格助手。", "retrieval_hints": ["claude-code"], "tags": ["product"], "status": "active"},
            {"id": "user", "scope": "user", "project": None, "memory_type": "preference", "summary": "用户偏好中文回答。", "retrieval_hints": ["language"], "status": "active"},
        ]

        context = build_agent_context(cards, project="/tmp/project", task="准备部署并执行 smoke 测试", limit=2)

        self.assertEqual([item["id"] for item in context["items"]], ["workflow", "user"])
        self.assertEqual(context["task"], "准备部署并执行 smoke 测试")

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
