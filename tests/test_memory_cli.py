import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from deepagent_memory.memory_cli import main


class MemoryCliTests(unittest.TestCase):
    def test_scan_outputs_codex_and_claude_summary_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            codex = root / ".codex"; codex.mkdir()
            (codex / "history.jsonl").write_text('{"session_id":"s1","ts":1,"text":"hello"}\n', encoding="utf-8")
            claude = root / ".claude"; claude.mkdir()
            (claude / "CLAUDE.md").write_text("中文", encoding="utf-8")
            out = root / "scan.json"

            exit_code = main(["scan", "--codex-home", str(codex), "--claude-home", str(claude), "--output", str(out)])

            self.assertEqual(exit_code, 0)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertTrue(payload["codex"]["history_found"])
            self.assertTrue(payload["claude"]["claude_md_found"])

    def test_import_codex_dry_run_writes_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            codex = root / ".codex"; codex.mkdir()
            (codex / "history.jsonl").write_text('{"session_id":"s1","ts":1,"text":"hello"}\n', encoding="utf-8")
            out_dir = root / "imports"

            exit_code = main(["import", "codex", "--codex-home", str(codex), "--output-dir", str(out_dir), "--dry-run"])

            self.assertEqual(exit_code, 0)
            events_file = out_dir / "codex-events.jsonl"
            report_file = out_dir / "import-report.json"
            self.assertTrue(events_file.exists())
            self.assertTrue(report_file.exists())
            self.assertIn("hello", events_file.read_text(encoding="utf-8"))
            report = json.loads(report_file.read_text(encoding="utf-8"))
            self.assertTrue(report["dry_run"])

    def test_import_all_combines_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            codex = root / ".codex"; codex.mkdir()
            (codex / "history.jsonl").write_text('{"session_id":"s1","ts":1,"text":"hello"}\n', encoding="utf-8")
            claude = root / ".claude"; claude.mkdir()
            (claude / "CLAUDE.md").write_text("中文", encoding="utf-8")
            out_dir = root / "imports"

            exit_code = main(["import", "all", "--codex-home", str(codex), "--claude-home", str(claude), "--output-dir", str(out_dir), "--dry-run"])

            self.assertEqual(exit_code, 0)
            self.assertTrue((out_dir / "all-events.jsonl").exists())
            text = (out_dir / "all-events.jsonl").read_text(encoding="utf-8")
            self.assertIn("codex", text)
            self.assertIn("claude_code", text)

    def test_dream_agent_without_invoke_writes_prompt_only_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = root / "events.jsonl"
            events.write_text('{"source":"codex","role":"user","content":"希望项目像 Claude Code","project":"/tmp/p"}\n', encoding="utf-8")
            output_dir = root / "memory"

            exit_code = main(["dream", "--input", str(events), "--project", "/tmp/p", "--output-dir", str(output_dir), "--agent"])

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "agent-prompt.md").exists())
            self.assertTrue((output_dir / "DREAMS.md").exists())

    def test_extract_facts_cli_writes_facts_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events_path = root / "events.jsonl"
            events_path.write_text(json.dumps({
                "event_id": "event_1",
                "source": "codex",
                "session_id": "s1",
                "project": str(root),
                "role": "user",
                "event_type": "history_prompt",
                "content": "这个项目需要人工审核后才能写正式记忆",
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            output_dir = root / "memory"

            exit_code = main([
                "extract-facts",
                "--input", str(events_path),
                "--project", str(root),
                "--output-dir", str(output_dir),
            ])

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "facts.jsonl").exists())

    def test_review_cli_writes_review_queue_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates_path = root / "candidates.jsonl"
            memory_cards_path = root / "memory_cards.jsonl"
            candidates_path.write_text(json.dumps({
                "id": "cand_1",
                "scope": "project",
                "project": str(root),
                "type": "decision",
                "content": "项目目标是 Claude Code 风格助手。",
                "score": 0.9,
                "status": "promote",
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            memory_cards_path.write_text(json.dumps({
                "id": "mem_1",
                "scope": "project",
                "project": str(root),
                "memory_type": "decision",
                "summary": "项目目标是通用聊天机器人。",
                "status": "active",
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            output_dir = root / "memory"

            exit_code = main([
                "review",
                "--candidates", str(candidates_path),
                "--memory-cards", str(memory_cards_path),
                "--output-dir", str(output_dir),
            ])

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "review_queue.jsonl").exists())

    def test_apply_cli_writes_memory_cards_and_memory_md(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviewed_path = root / "reviewed.jsonl"
            reviewed_path.write_text(json.dumps({
                "candidate_id": "cand_1",
                "status": "approved",
                "reviewer": "user",
                "notes": "looks good",
                "memory_updates": [{
                    "id": "mem_1",
                    "scope": "project",
                    "project": str(root),
                    "memory_type": "decision",
                    "summary": "项目目标是 Claude Code 风格的本地研发助手。",
                    "evidence_refs": ["event_1"],
                    "approved_by": "user",
                    "approved_at": "2026-07-05T00:00:00Z",
                    "status": "active",
                    "retrieval_hints": ["claude code"],
                }],
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            output_dir = root / "memory"

            exit_code = main([
                "apply",
                "--reviewed", str(reviewed_path),
                "--output-dir", str(output_dir),
                "--reviewer", "user",
            ])

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "memory_cards.jsonl").exists())
            self.assertTrue((output_dir / "MEMORY.md").exists())

    def test_context_cli_prints_ranked_context_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cards_path = root / "memory_cards.jsonl"
            cards_path.write_text("\n".join([
                json.dumps({"id": "mem_1", "scope": "global", "project": None, "memory_type": "workflow", "summary": "正式记忆必须人工审核。", "retrieval_hints": [], "status": "active"}, ensure_ascii=False),
                json.dumps({"id": "mem_2", "scope": "project", "project": "/tmp/project", "memory_type": "decision", "summary": "项目目标是 Claude Code 风格助手。", "retrieval_hints": [], "status": "active"}, ensure_ascii=False),
            ]) + "\n", encoding="utf-8")

            exit_code = main([
                "context",
                "--project", "/tmp/project",
                "--memory-cards", str(cards_path),
                "--limit", "2",
            ])

            self.assertEqual(exit_code, 0)

    def test_review_cli_treats_missing_memory_cards_as_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates_path = root / "candidates.jsonl"
            missing_cards_path = root / "missing-memory-cards.jsonl"
            output_dir = root / "memory"
            candidates_path.write_text(json.dumps({"id": "cand_1", "scope": "user", "project": None, "type": "preference", "content": "用户偏好中文回答。"}, ensure_ascii=False) + "\n", encoding="utf-8")

            exit_code = main(["review", "--candidates", str(candidates_path), "--memory-cards", str(missing_cards_path), "--output-dir", str(output_dir)])

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "review_queue.jsonl").exists())

    def test_apply_cli_writes_review_decisions_ledger_from_web_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviewed_path = root / "reviewed.jsonl"
            output_dir = root / "memory"
            reviewed_path.write_text(json.dumps({
                "candidate_id": "cand_1",
                "action": "approved",
                "edited_content": "用户偏好中文回答。",
                "reviewer": "user",
                "note": "明确偏好",
                "candidate": {"id": "cand_1", "type": "preference", "scope": "user", "project": None, "content": "用户偏好中文回答。", "evidence": [{"event_id": "event_1"}], "tags": ["language"]},
            }, ensure_ascii=False) + "\n", encoding="utf-8")

            exit_code = main(["apply", "--reviewed", str(reviewed_path), "--output-dir", str(output_dir), "--reviewer", "user"])

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "memory_cards.jsonl").exists())
            self.assertTrue((output_dir / "review_decisions.jsonl").exists())
            self.assertIn("用户偏好中文回答", (output_dir / "MEMORY.md").read_text(encoding="utf-8"))

    def test_context_cli_supports_markdown_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cards_path = root / "memory_cards.jsonl"
            cards_path.write_text(json.dumps({"id": "mem_1", "scope": "user", "project": None, "memory_type": "preference", "summary": "用户偏好中文回答。", "retrieval_hints": [], "status": "active"}, ensure_ascii=False) + "\n", encoding="utf-8")

            exit_code = main(["context", "--project", "/tmp/project", "--memory-cards", str(cards_path), "--format", "markdown"])

            self.assertEqual(exit_code, 0)

    def test_pipeline_cli_runs_extract_dream_and_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events_path = root / "events.jsonl"
            output_dir = root / "memory"
            events_path.write_text(json.dumps({"event_id": "event_1", "source": "codex", "session_id": "s1", "role": "user", "event_type": "history_prompt", "content": "这个项目需要人工审核后才能写正式记忆"}, ensure_ascii=False) + "\n", encoding="utf-8")

            exit_code = main(["pipeline", "--input", str(events_path), "--project", str(root), "--output-dir", str(output_dir)])

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "facts.jsonl").exists())
            self.assertTrue((output_dir / "agent-candidates.jsonl").exists())
            self.assertTrue((output_dir / "review_queue.jsonl").exists())

    def test_dream_defaults_to_ai_dry_run_prompt_without_rule_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = root / "events.jsonl"
            events.write_text(json.dumps({"source": "codex", "role": "user", "content": "希望项目像 Claude Code", "project": "/tmp/p"}, ensure_ascii=False) + "\n", encoding="utf-8")
            output_dir = root / "memory"

            exit_code = main(["dream", "--input", str(events), "--project", "/tmp/p", "--output-dir", str(output_dir)])

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "agent-prompt.md").exists())
            self.assertTrue((output_dir / "agent-candidates.jsonl").exists())
            self.assertFalse((output_dir / "candidates.jsonl").exists())

    def test_dream_rules_mode_preserves_rule_candidate_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = root / "events.jsonl"
            events.write_text(json.dumps({"source": "codex", "role": "user", "event_type": "history_prompt", "content": "这个项目需要人工审核后才能写正式记忆", "project": str(root)}, ensure_ascii=False) + "\n", encoding="utf-8")
            output_dir = root / "memory"

            exit_code = main(["dream", "--input", str(events), "--project", str(root), "--output-dir", str(output_dir), "--mode", "rules"])

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "candidates.jsonl").exists())
            self.assertFalse((output_dir / "agent-candidates.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
