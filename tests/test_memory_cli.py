import io
import json
import sqlite3
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from dream_memory.memory_cli import build_parser, main
from dream_memory.model_providers import ModelRuntimeResult
from dream_memory.memory_runs import create_run_state, update_run_state


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

    def test_import_cli_uses_config_imports_output_dir_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            codex = root / "codex-home"
            claude = root / "claude-home"
            imports_dir = root / "configured-imports"
            codex_sessions = codex / "sessions"
            codex_sessions.mkdir(parents=True)
            claude.mkdir()
            (codex_sessions / "rollout.jsonl").write_text(json.dumps({
                "timestamp": "2026-07-08T10:00:00Z",
                "type": "response_item",
                "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "用户偏好中文回答。"}]},
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            config_path = root / "config.json"
            config_path.write_text(json.dumps({
                "codex_home": str(codex),
                "claude_home": str(claude),
                "claude_state": str(root / "claude.json"),
                "imports_output_dir": str(imports_dir),
            }, ensure_ascii=False), encoding="utf-8")

            exit_code = main(["--config", str(config_path), "import", "all"])

            self.assertEqual(exit_code, 0)
            self.assertTrue((imports_dir / "all-events.jsonl").exists())

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


    def test_import_all_includes_local_project_instruction_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            codex = root / ".codex"; codex.mkdir()
            claude = root / ".claude"; claude.mkdir()
            project = root / "project"; project.mkdir()
            (project / "AGENTS.md").write_text("如果后端使用的是 Python，则使用 uv 进行包管理\n前端使用 pnpm 进行管理", encoding="utf-8")
            out_dir = root / "imports"

            exit_code = main(["import", "all", "--codex-home", str(codex), "--claude-home", str(claude), "--project", str(project), "--output-dir", str(out_dir), "--dry-run"])

            self.assertEqual(exit_code, 0)
            text = (out_dir / "all-events.jsonl").read_text(encoding="utf-8")
            self.assertIn("project_instruction", text)
            self.assertIn("uv 进行包管理", text)
            self.assertTrue((out_dir / "project-instructions-events.jsonl").exists())


    def test_import_all_includes_project_marker_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            codex = root / ".codex"; codex.mkdir()
            claude = root / ".claude"; claude.mkdir()
            project = root / "project"; project.mkdir()
            (project / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
            (project / "uv.lock").write_text("version = 1\n", encoding="utf-8")
            out_dir = root / "imports"

            exit_code = main(["import", "all", "--codex-home", str(codex), "--claude-home", str(claude), "--project", str(project), "--output-dir", str(out_dir), "--dry-run"])

            self.assertEqual(exit_code, 0)
            text = (out_dir / "all-events.jsonl").read_text(encoding="utf-8")
            self.assertIn("project_markers", text)
            self.assertIn("python_package_manager=uv", text)
            self.assertTrue((out_dir / "project-marker-events.jsonl").exists())

    def test_dream_agent_without_invoke_writes_prompt_only_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = root / "events.jsonl"
            events.write_text('{"source":"codex","role":"user","content":"希望项目像 Claude Code","project":"/tmp/p"}\n', encoding="utf-8")
            output_dir = root / "memory"

            exit_code = main(["dream", "--input", str(events), "--project", "/tmp/p", "--output-dir", str(output_dir), "--agent", "--dry-run"])

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "ai-prompt.md").exists())
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

    def test_review_summary_cli_groups_run_review_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"output_dir": str(memory_dir)}, ensure_ascii=False), encoding="utf-8")
            state = create_run_state(memory_dir=memory_dir, project=str(root), input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            queue_path = Path(state["run_dir"]) / "review_queue.jsonl"
            queue_path.write_text("\n".join([
                json.dumps({"candidate_id": "create_1", "suggested_action": "create", "status": "promote", "candidate": {"id": "create_1", "type": "workflow", "scope": "user", "content": "高分", "evidence": [{"event_id": "event_1"}]}, "quality_signals": {"evidence_quality": "multi_event"}, "dream_analysis": {"dream_score": 0.82}}, ensure_ascii=False),
                json.dumps({"candidate_id": "review_1", "suggested_action": "review", "status": "review", "candidate": {"id": "review_1", "type": "preference", "scope": "user", "content": "人工", "evidence": [{"event_id": "event_2"}]}, "quality_signals": {"duplicate": True, "evidence_quality": "single_event"}, "dream_analysis": {"dream_score": 0.55}, "conflicts": [{"memory_id": "mem_1"}]}, ensure_ascii=False),
            ]) + "\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(queue_path)})

            with patch("sys.stdout", new_callable=lambda: __import__('io').StringIO()) as stdout:
                exit_code = main(["--config", str(config_path), "review-summary", "--run-id", state["run_id"]])

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["run_id"], state["run_id"])
            self.assertEqual(payload["total"], 2)
            self.assertEqual(payload["by_suggested_action"], {"create": 1, "review": 1})
            self.assertEqual(payload["by_type"], {"preference": 1, "workflow": 1})
            self.assertEqual(payload["duplicate_count"], 1)
            self.assertEqual(payload["conflict_count"], 1)
            self.assertEqual(payload["needs_manual_count"], 1)

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

    def test_auto_review_cli_never_approves_create_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"output_dir": str(memory_dir)}, ensure_ascii=False), encoding="utf-8")
            state = create_run_state(memory_dir=memory_dir, project=str(root), input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            run_dir = Path(state["run_dir"])
            queue_path = run_dir / "review_queue.jsonl"
            queue_path.write_text("\n".join([
                json.dumps({"candidate_id": "create_1", "suggested_action": "create", "candidate": {"id": "create_1", "type": "preference", "scope": "user", "content": "用户偏好中文回答。", "evidence": [{"event_id": "event_1"}], "tags": ["language"]}, "dream_analysis": {"dream_score": 0.82}}, ensure_ascii=False),
                json.dumps({"candidate_id": "reject_1", "suggested_action": "reject", "candidate": {"id": "reject_1", "type": "requirement", "scope": "project", "project": str(root), "content": "删除这个按钮", "evidence": [{"event_id": "event_2"}], "tags": ["task"]}, "dream_analysis": {"dream_score": 0.32}}, ensure_ascii=False),
                json.dumps({"candidate_id": "review_1", "suggested_action": "review", "candidate": {"id": "review_1", "type": "workflow", "scope": "user", "content": "需要人工判断。", "evidence": [{"event_id": "event_3"}]}, "dream_analysis": {"dream_score": 0.62}}, ensure_ascii=False),
            ]) + "\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(queue_path)})

            exit_code = main(["--config", str(config_path), "auto-review", "--run-id", state["run_id"], "--min-score", "0.7"])

            self.assertEqual(exit_code, 0)
            reviewed_path = run_dir / "reviewed.jsonl"
            rows = [json.loads(line) for line in reviewed_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["action"] for row in rows], ["rejected"])
            updated = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(updated["counts"]["auto_review_count"], 1)
            self.assertIn("auto_reviewed", (run_dir / "trace.jsonl").read_text(encoding="utf-8"))

    def test_auto_review_cli_never_approves_merge_even_when_included(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"output_dir": str(memory_dir)}, ensure_ascii=False), encoding="utf-8")
            state = create_run_state(memory_dir=memory_dir, project=str(root), input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            run_dir = Path(state["run_dir"])
            queue_path = run_dir / "review_queue.jsonl"
            queue_path.write_text(json.dumps({
                "candidate_id": "merge_1",
                "suggested_action": "merge",
                "candidate": {"id": "merge_1", "type": "preference", "scope": "user", "content": "用户偏好直接推进。", "evidence": [{"event_id": "event_1"}], "tags": ["autonomy"]},
                "quality_signals": {"matched_memory_id": "mem_existing", "similarity": 0.5},
                "dream_analysis": {"dream_score": 0.82},
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(queue_path)})

            exit_code = main(["--config", str(config_path), "auto-review", "--run-id", state["run_id"]])

            self.assertEqual(exit_code, 0)
            reviewed_path = run_dir / "reviewed.jsonl"
            self.assertEqual(reviewed_path.read_text(encoding="utf-8"), "")

            exit_code = main(["--config", str(config_path), "auto-review", "--run-id", state["run_id"], "--include-merges", "--force"])

            self.assertEqual(exit_code, 0)
            self.assertEqual(reviewed_path.read_text(encoding="utf-8"), "")

    def test_auto_review_cli_skips_duplicates_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"output_dir": str(memory_dir)}, ensure_ascii=False), encoding="utf-8")
            state = create_run_state(memory_dir=memory_dir, project=str(root), input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            run_dir = Path(state["run_dir"])
            queue_path = run_dir / "review_queue.jsonl"
            queue_path.write_text(json.dumps({
                "candidate_id": "dup_1",
                "suggested_action": "reject",
                "candidate": {"id": "dup_1", "type": "preference", "scope": "user", "content": "用户偏好中文回答。", "evidence": [{"event_id": "event_1"}], "tags": ["language"]},
                "quality_signals": {"duplicate": True, "matched_memory_id": "mem_existing"},
                "dream_analysis": {"dream_score": 0.8},
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(queue_path)})

            exit_code = main(["--config", str(config_path), "auto-review", "--run-id", state["run_id"]])

            self.assertEqual(exit_code, 0)
            reviewed_path = run_dir / "reviewed.jsonl"
            self.assertEqual(reviewed_path.read_text(encoding="utf-8"), "")
            updated = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(updated["counts"]["auto_review_count"], 0)
            trace = (run_dir / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn('"duplicate_skipped": 1', trace)

            exit_code = main(["--config", str(config_path), "auto-review", "--run-id", state["run_id"], "--include-duplicates", "--force"])

            self.assertEqual(exit_code, 0)
            rows = [json.loads(line) for line in reviewed_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["action"], "rejected")

    def test_auto_review_cli_dry_run_ignores_existing_reviewed_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"output_dir": str(memory_dir)}, ensure_ascii=False), encoding="utf-8")
            state = create_run_state(memory_dir=memory_dir, project=str(root), input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            run_dir = Path(state["run_dir"])
            queue_path = run_dir / "review_queue.jsonl"
            queue_path.write_text(json.dumps({"candidate_id": "cand_1", "suggested_action": "create", "candidate": {"id": "cand_1", "type": "workflow", "scope": "user", "content": "用户偏好直接推进。", "evidence": [{"event_id": "event_1"}]}, "dream_analysis": {"dream_score": 0.82}}, ensure_ascii=False) + "\n", encoding="utf-8")
            reviewed_path = run_dir / "reviewed.jsonl"
            reviewed_path.write_text(json.dumps({"candidate_id": "manual", "action": "approved"}, ensure_ascii=False) + "\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(queue_path)})

            with patch("sys.stdout", new_callable=lambda: __import__('io').StringIO()) as stdout:
                exit_code = main(["--config", str(config_path), "auto-review", "--run-id", state["run_id"], "--dry-run"])

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertTrue(payload["dry_run"])
            self.assertEqual(payload["decision_count"], 0)
            self.assertEqual(payload["skip_reasons"], {"requires_manual_review": 1})
            rows = [json.loads(line) for line in reviewed_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["candidate_id"], "manual")

    def test_auto_review_cli_dry_run_reports_without_writing_or_mutating_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"output_dir": str(memory_dir)}, ensure_ascii=False), encoding="utf-8")
            state = create_run_state(memory_dir=memory_dir, project=str(root), input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            run_dir = Path(state["run_dir"])
            queue_path = run_dir / "review_queue.jsonl"
            queue_path.write_text(json.dumps({"candidate_id": "cand_1", "suggested_action": "create", "candidate": {"id": "cand_1", "type": "workflow", "scope": "user", "content": "用户偏好直接推进。", "evidence": [{"event_id": "event_1"}]}, "dream_analysis": {"dream_score": 0.82}}, ensure_ascii=False) + "\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(queue_path)})
            before_state = (run_dir / "state.json").read_text(encoding="utf-8")

            with patch("sys.stdout", new_callable=lambda: __import__('io').StringIO()) as stdout:
                exit_code = main(["--config", str(config_path), "auto-review", "--run-id", state["run_id"], "--dry-run"])

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertTrue(payload["dry_run"])
            self.assertEqual(payload["decision_count"], 0)
            self.assertEqual(payload["skip_reasons"], {"requires_manual_review": 1})
            self.assertFalse((run_dir / "reviewed.jsonl").exists())
            self.assertEqual((run_dir / "state.json").read_text(encoding="utf-8"), before_state)

    def test_auto_review_cli_reports_manual_skip_reason_regardless_of_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.json"
            memory_dir = root / "memory"
            config_path.write_text(json.dumps({"models": {"primary": {"provider": "anthropic", "model": "test", "api_key": "key"}}, "model_policy": {"default_profile": "primary", "fallback_chain": ["primary"]}, "output_dir": str(memory_dir)}, ensure_ascii=False), encoding="utf-8")
            state = create_run_state(memory_dir=memory_dir, project=str(root), input_path=str(root / "events.jsonl"), mode="rules", model="test", invoke_model=False)
            run_dir = Path(state["run_dir"])
            queue_path = run_dir / "review_queue.jsonl"
            queue_path.write_text("\n".join([
                json.dumps({"candidate_id": "low", "suggested_action": "create", "candidate": {"id": "low", "type": "workflow", "scope": "user", "content": "低分候选", "evidence": [{"event_id": "event_1"}]}, "dream_analysis": {"dream_score": 0.4}}, ensure_ascii=False),
                json.dumps({"candidate_id": "review", "suggested_action": "review", "candidate": {"id": "review", "type": "workflow", "scope": "user", "content": "需要人工判断", "evidence": [{"event_id": "event_2"}]}, "dream_analysis": {"dream_score": 0.65}}, ensure_ascii=False),
            ]) + "\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(queue_path)})

            with patch("sys.stdout", new_callable=lambda: __import__('io').StringIO()) as stdout:
                exit_code = main(["--config", str(config_path), "auto-review", "--run-id", state["run_id"], "--min-score", "0.7"])

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["skipped"], 2)
            self.assertEqual(payload["skip_reasons"], {"requires_manual_review": 2})

    def test_auto_review_cli_refuses_to_overwrite_existing_reviewed_without_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"output_dir": str(memory_dir)}, ensure_ascii=False), encoding="utf-8")
            state = create_run_state(memory_dir=memory_dir, project=str(root), input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            run_dir = Path(state["run_dir"])
            queue_path = run_dir / "review_queue.jsonl"
            queue_path.write_text(json.dumps({"candidate_id": "cand_1", "suggested_action": "create", "candidate": {"id": "cand_1", "type": "preference", "scope": "user", "content": "用户偏好中文回答。", "evidence": [{"event_id": "event_1"}], "tags": ["language"]}, "dream_analysis": {"dream_score": 0.82}}, ensure_ascii=False) + "\n", encoding="utf-8")
            reviewed_path = run_dir / "reviewed.jsonl"
            reviewed_path.write_text(json.dumps({"candidate_id": "manual", "action": "approved"}, ensure_ascii=False) + "\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(queue_path)})

            with self.assertRaises(FileExistsError):
                main(["--config", str(config_path), "auto-review", "--run-id", state["run_id"]])

            rows = [json.loads(line) for line in reviewed_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["candidate_id"], "manual")

            exit_code = main(["--config", str(config_path), "auto-review", "--run-id", state["run_id"], "--force"])

            self.assertEqual(exit_code, 0)
            self.assertEqual(reviewed_path.read_text(encoding="utf-8"), "")

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

    def test_summary_and_export_treat_missing_memory_cards_as_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing_cards = root / "missing-memory_cards.jsonl"
            export_dir = root / "exported"

            self.assertEqual(main(["summary", "--memory-cards", str(missing_cards)]), 0)
            self.assertEqual(main(["export", "--target", "codex", "--project", "/tmp/project", "--memory-cards", str(missing_cards), "--output-dir", str(export_dir)]), 0)
            self.assertTrue((export_dir / "AGENTS.md").exists())

    def test_context_cli_treats_missing_memory_cards_as_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing_cards_path = root / "missing-memory-cards.jsonl"

            exit_code = main([
                "context",
                "--project", "/tmp/project",
                "--memory-cards", str(missing_cards_path),
                "--limit", "2",
            ])

            self.assertEqual(exit_code, 0)

    def test_init_cli_accepts_output_dir_alias_for_workspace_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "custom-memory"

            exit_code = main(["init", "--output-dir", str(workspace)])

            self.assertEqual(exit_code, 0)
            self.assertTrue((workspace / "config.json").exists())
            self.assertTrue((workspace / "memory_cards.jsonl").exists())
            self.assertTrue((workspace / "examples" / "labeled-events.jsonl").exists())
            self.assertFalse((root / "examples" / "labeled-events.jsonl").exists())
            payload = json.loads((workspace / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["default_input"], str(workspace / "imports" / "all-events.jsonl"))
            self.assertEqual(payload["init_config_output"], str(workspace / "config.json"))
            self.assertEqual(payload["output_dir"], str(workspace))
            self.assertEqual(payload["imports_output_dir"], str(workspace / "imports"))
            self.assertEqual(payload["extract_input"], str(workspace / "imports" / "all-events.jsonl"))
            self.assertEqual(payload["extract_output_dir"], str(workspace))
            self.assertEqual(payload["review_candidates"], str(workspace / "ai-candidates.jsonl"))
            self.assertEqual(payload["apply_reviewed"], str(workspace / "reviewed.jsonl"))
            self.assertEqual(payload["eval_input"], str(workspace / "examples" / "labeled-events.jsonl"))
            self.assertEqual(payload["eval_output"], str(workspace / "eval.json"))
            self.assertEqual(payload["memory_cards"], str(workspace / "memory_cards.jsonl"))

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

            exit_code = main(["pipeline", "--input", str(events_path), "--project", str(root), "--output-dir", str(output_dir), "--dry-run"])

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "facts.jsonl").exists())
            self.assertTrue((output_dir / "ai-candidates.jsonl").exists())
            self.assertTrue((output_dir / "review_queue.jsonl").exists())

    def test_dream_dry_run_writes_prompt_without_rule_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = root / "events.jsonl"
            events.write_text(json.dumps({"source": "codex", "role": "user", "content": "希望项目像 Claude Code", "project": "/tmp/p"}, ensure_ascii=False) + "\n", encoding="utf-8")
            output_dir = root / "memory"

            exit_code = main(["dream", "--input", str(events), "--project", "/tmp/p", "--output-dir", str(output_dir), "--dry-run"])

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "ai-prompt.md").exists())
            self.assertTrue((output_dir / "ai-candidates.jsonl").exists())
            self.assertFalse((output_dir / "candidates.jsonl").exists())

    def test_dream_rules_mode_preserves_rule_candidate_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = root / "events.jsonl"
            events.write_text(json.dumps({"source": "codex", "role": "user", "event_type": "history_prompt", "content": "这个项目需要人工审核后才能写正式记忆", "project": str(root)}, ensure_ascii=False) + "\n", encoding="utf-8")
            output_dir = root / "memory"

            exit_code = main(["dream", "--input", str(events), "--project", str(root), "--output-dir", str(output_dir), "--mode", "rules"])
            dreams = (output_dir / "DREAMS.md").read_text(encoding="utf-8")

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "candidates.jsonl").exists())
            self.assertFalse((output_dir / "ai-candidates.jsonl").exists())
            self.assertIn("## Promotion Policy", dreams)
            self.assertIn("## Action Summary", dreams)
            self.assertIn("dream_score=", dreams)

    def test_ai_mode_uses_config_default_and_dry_run_disables_it(self):
        parser = build_parser()

        dream_args = parser.parse_args(["dream", "--input", "events.jsonl"])
        dry_run_args = parser.parse_args(["dream", "--input", "events.jsonl", "--dry-run"])
        pipeline_args = parser.parse_args(["pipeline", "--input", "events.jsonl"])

        self.assertIsNone(dream_args.invoke_model)
        self.assertFalse(dry_run_args.invoke_model)
        self.assertIsNone(pipeline_args.invoke_model)

    def test_source_distribution_manifest_includes_maintained_docs_and_examples(self):
        manifest = Path("MANIFEST.in").read_text(encoding="utf-8")

        self.assertIn("include docs/*.md", manifest)
        self.assertIn("include examples/*.json", manifest)
        self.assertIn("include examples/*.jsonl", manifest)
        self.assertNotIn("recursive-include docs", manifest)
        self.assertNotIn("recursive-exclude .venv", manifest)
        self.assertNotIn("prune docs/superpowers", manifest)

    def test_eval_run_and_pipeline_missing_files_return_clean_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "missing.jsonl")

            self.assertEqual(main(["eval", "--input", missing]), 2)
            self.assertEqual(main(["run", "--input", missing]), 2)
            self.assertEqual(main(["pipeline", "--input", missing]), 2)

    def test_eval_run_and_pipeline_missing_files_have_no_tracebacks_in_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "missing.jsonl")
            commands = [
                (["uv", "run", "dream-memory", "eval", "--input", missing], "eval input not found"),
                (["uv", "run", "dream-memory", "run", "--input", missing], "run input not found"),
                (["uv", "run", "dream-memory", "pipeline", "--input", missing], "pipeline input not found"),
            ]

            for command, expected in commands:
                result = subprocess.run(command, cwd=Path.cwd(), text=True, capture_output=True, check=False)
                self.assertEqual(result.returncode, 2)
                self.assertIn(expected, result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_eval_cli_missing_input_returns_clean_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text("{}", encoding="utf-8")

            exit_code = main(["--config", str(config_path), "eval"])

            self.assertEqual(exit_code, 2)

    def test_eval_cli_uses_config_eval_mode_when_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labeled = root / "labeled.jsonl"
            output = root / "configured-ai-eval.json"
            config_path = root / "config.json"
            labeled.write_text(json.dumps({
                "event": {"event_id": "event_1", "source": "codex", "role": "user", "event_type": "history_prompt", "content": "用户偏好中文回答。"},
                "expected": [{"content": "用户偏好中文回答", "type": "preference", "scope": "user"}],
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            config_path.write_text(json.dumps({
                "eval_input": str(labeled),
                "eval_project": "/tmp/project",
                "eval_mode": "ai",
                "eval_output": str(output),
                "models": {"primary": {"provider": "openai", "model": "gpt-test", "api_key": "key"}},
                "model_policy": {"default_profile": "primary", "fallback_chain": ["primary"]},
            }, ensure_ascii=False), encoding="utf-8")

            with patch("dream_memory.memory_cli.evaluate_labeled_events", return_value={
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "extraction_success_count": 0,
                "extraction_error_count": 0,
                "fallback_count": 0,
            }) as evaluate:
                exit_code = main(["--config", str(config_path), "eval"])

            self.assertEqual(exit_code, 0)
            self.assertEqual(evaluate.call_args.kwargs["mode"], "ai")

    def test_eval_cli_uses_config_defaults_when_args_omitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labeled = root / "labeled.jsonl"
            output = root / "configured-eval.json"
            config_path = root / "config.json"
            labeled.write_text(json.dumps({
                "event": {"event_id": "event_1", "source": "codex", "role": "user", "event_type": "history_prompt", "content": "用户偏好中文回答。"},
                "expected": [{"content": "用户偏好中文回答", "type": "preference", "scope": "user"}],
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            config_path.write_text(json.dumps({
                "eval_input": str(labeled),
                "eval_project": "/tmp/project",
                "eval_mode": "rules",
                "eval_output": str(output),
                "eval_max_rows": 1,
                "eval_continue_on_error": True,
                "eval_fallback_rules_on_error": True,
                "eval_fallback_rules_on_empty": True,
            }, ensure_ascii=False), encoding="utf-8")

            exit_code = main(["--config", str(config_path), "eval"])

            self.assertEqual(exit_code, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["rows"], 1)
            self.assertEqual(payload["precision"], 1.0)

    def test_eval_cli_config_default_empty_fallback_is_forwarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labeled = root / "labeled.jsonl"
            config_path = root / "config.json"
            labeled.write_text(json.dumps({
                "event": {"event_id": "event_1", "source": "codex", "role": "user", "event_type": "history_prompt", "content": "用户偏好中文回答。"},
                "expected": [{"content": "用户偏好中文回答", "type": "preference", "scope": "user"}],
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            config_path.write_text(json.dumps({
                "eval_input": str(labeled),
                "eval_mode": "ai",
                "eval_fallback_rules_on_empty": True,
                "models": {"primary": {"provider": "openai", "model": "gpt-test", "api_key": "key"}},
                "model_policy": {"default_profile": "primary", "fallback_chain": ["primary"]},
            }, ensure_ascii=False), encoding="utf-8")

            with patch("dream_memory.memory_cli.evaluate_labeled_events", return_value={
                "precision": 1.0, "recall": 1.0, "f1": 1.0,
                "extraction_success_count": 0, "extraction_error_count": 0,
                "fallback_count": 1, "fallback_empty_count": 1,
            }) as evaluate:
                exit_code = main(["--config", str(config_path), "eval"])

            self.assertEqual(exit_code, 0)
            self.assertTrue(evaluate.call_args.kwargs["fallback_rules_on_empty"])

    def test_eval_cli_passes_model_runtime_and_resilience_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labeled = root / "labeled.jsonl"
            output = root / "eval.json"
            config_path = root / "config.json"
            labeled.write_text(json.dumps({
                "event": {"event_id": "event_1", "source": "codex", "role": "user", "event_type": "history_prompt", "content": "用户偏好中文回答。"},
                "expected": [{"content": "用户偏好中文回答", "type": "preference", "scope": "user"}],
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            config_path.write_text(json.dumps({
                "models": {
                    "primary": {"provider": "openai", "model": "gpt-test", "api_key": "key", "timeout_seconds": 60}
                },
                "model_policy": {"default_profile": "primary", "fallback_chain": ["primary"], "retry": {"max_attempts": 3}},
            }, ensure_ascii=False), encoding="utf-8")

            with patch("dream_memory.memory_cli.evaluate_labeled_events", return_value={
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "extraction_success_count": 0,
                "extraction_error_count": 1,
                "fallback_count": 1,
            }) as evaluate:
                exit_code = main([
                    "--config", str(config_path),
                    "eval",
                    "--input", str(labeled),
                    "--project", "/tmp/project",
                    "--mode", "ai",
                    "--provider", "openai",
                    "--model", "gpt-override",
                    "--timeout-seconds", "7",
                    "--max-attempts", "1",
                    "--max-rows", "2",
                    "--continue-on-error",
                    "--fallback-rules-on-error",
                    "--fallback-rules-on-empty",
                    "--output", str(output),
                ])

            self.assertEqual(exit_code, 0)
            kwargs = evaluate.call_args.kwargs
            self.assertEqual(kwargs["project"], "/tmp/project")
            self.assertEqual(kwargs["mode"], "ai")
            self.assertEqual(kwargs["model"], "openai:gpt-override")
            self.assertEqual(kwargs["runtime_config"]["models"]["override"]["timeout_seconds"], 7)
            self.assertEqual(kwargs["runtime_config"]["model_policy"]["retry"]["max_attempts"], 1)
            self.assertEqual(kwargs["max_rows"], 2)
            self.assertTrue(kwargs["continue_on_error"])
            self.assertTrue(kwargs["fallback_rules_on_error"])
            self.assertTrue(kwargs["fallback_rules_on_empty"])
            self.assertTrue(output.exists())


    def test_eval_cli_output_summary_includes_candidate_totals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labeled = root / "labeled.jsonl"
            output = root / "eval.json"
            labeled.write_text(json.dumps({
                "event": {"event_id": "event_1", "source": "codex", "role": "user", "event_type": "history_prompt", "content": "用户偏好中文回答。"},
                "expected": [{"content": "用户偏好中文回答", "type": "preference", "scope": "user"}],
            }, ensure_ascii=False) + "\n", encoding="utf-8")

            with patch("dream_memory.memory_cli.evaluate_labeled_events", return_value={
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "raw_candidate_total": 2,
                "fallback_candidate_total": 1,
                "scored_candidate_total": 1,
                "extraction_success_count": 1,
                "extraction_error_count": 0,
                "fallback_count": 1,
                "fallback_empty_count": 1,
            }):
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    exit_code = main(["eval", "--input", str(labeled), "--output", str(output)])

        self.assertEqual(exit_code, 0)
        summary = json.loads(stdout.getvalue())
        self.assertEqual(summary["raw_candidate_total"], 2)
        self.assertEqual(summary["fallback_candidate_total"], 1)
        self.assertEqual(summary["scored_candidate_total"], 1)

    def test_init_config_cli_writes_default_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"

            exit_code = main(["init-config", "--output", str(config_path)])

            self.assertEqual(exit_code, 0)
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["models"]["primary"]["provider"], "anthropic")
            self.assertEqual(payload["models"]["primary"]["model"], "claude-sonnet-4-6")
            self.assertEqual(payload["model_policy"]["fallback_chain"], ["primary"])

    def test_run_and_pipeline_missing_input_return_clean_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(json.dumps({"default_input": None}, ensure_ascii=False), encoding="utf-8")

            self.assertEqual(main(["--config", str(config_path), "run"]), 2)
            self.assertEqual(main(["--config", str(config_path), "pipeline"]), 2)

    def test_run_cli_uses_config_default_input_and_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = root / "events.jsonl"
            memory_dir = root / "memory"
            config_path = root / "config.json"
            events.write_text(json.dumps({
                "event_id": "event_1",
                "source": "codex",
                "role": "user",
                "event_type": "history_prompt",
                "project": "/tmp/project",
                "content": "正式记忆必须经过人工审核，只有审核通过的候选才允许写入长期记忆。",
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            config_path.write_text(json.dumps({
                "default_input": str(events),
                "default_project": "/tmp/project",
                "output_dir": str(memory_dir),
                "invoke_model": False,
                "mode": "rules",
            }, ensure_ascii=False), encoding="utf-8")

            exit_code = main(["--config", str(config_path), "run", "--mode", "rules"])

            self.assertEqual(exit_code, 0)
            state = json.loads(next((memory_dir / "runs").glob("*/state.json")).read_text(encoding="utf-8"))
            self.assertEqual(state["input_path"], str(events))
            self.assertEqual(state["project"], "/tmp/project")
            self.assertTrue(Path(state["artifacts"]["review_queue_path"]).exists())

    def test_pipeline_cli_uses_config_default_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = root / "events.jsonl"
            memory_dir = root / "memory"
            config_path = root / "config.json"
            events.write_text(json.dumps({
                "event_id": "event_1",
                "source": "codex",
                "role": "user",
                "event_type": "history_prompt",
                "project": "/tmp/project",
                "content": "正式记忆必须经过人工审核，只有审核通过的候选才允许写入长期记忆。",
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            config_path.write_text(json.dumps({
                "default_input": str(events),
                "default_project": "/tmp/project",
                "output_dir": str(memory_dir),
                "invoke_model": False,
                "mode": "rules",
            }, ensure_ascii=False), encoding="utf-8")

            exit_code = main(["--config", str(config_path), "pipeline", "--mode", "rules"])

            self.assertEqual(exit_code, 0)
            self.assertTrue((memory_dir / "review_queue.jsonl").exists())

    def test_dream_and_extract_facts_missing_files_have_no_tracebacks_in_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = str(Path(tmp) / "missing.jsonl")
            commands = [
                (["uv", "run", "dream-memory", "dream", "--input", missing], "dream input not found"),
                (["uv", "run", "dream-memory", "extract-facts", "--input", missing], "extract-facts input not found"),
            ]

            for command, expected in commands:
                result = subprocess.run(command, cwd=Path.cwd(), text=True, capture_output=True, check=False)
                self.assertEqual(result.returncode, 2)
                self.assertIn(expected, result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_dream_and_extract_facts_use_config_default_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = root / "events.jsonl"
            dream_dir = root / "dream-memory"
            extract_dir = root / "facts-memory"
            config_path = root / "config.json"
            events.write_text(json.dumps({
                "event_id": "event_1",
                "source": "codex",
                "role": "user",
                "event_type": "history_prompt",
                "project": "/tmp/project",
                "content": "正式记忆必须经过人工审核，只有审核通过的候选才允许写入长期记忆。",
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            config_path.write_text(json.dumps({
                "default_input": str(events),
                "default_project": "/tmp/project",
                "extract_input": str(events),
                "extract_project": "/tmp/project",
                "extract_output_dir": str(extract_dir),
                "output_dir": str(dream_dir),
                "invoke_model": False,
                "mode": "rules",
            }, ensure_ascii=False), encoding="utf-8")

            self.assertEqual(main(["--config", str(config_path), "dream", "--mode", "rules"]), 0)
            self.assertEqual(main(["--config", str(config_path), "extract-facts"]), 0)

            self.assertTrue((dream_dir / "candidates.jsonl").exists())
            self.assertTrue((extract_dir / "facts.jsonl").exists())

    def test_dream_and_extract_facts_missing_inputs_return_clean_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(json.dumps({"default_input": None, "extract_input": None}, ensure_ascii=False), encoding="utf-8")

            self.assertEqual(main(["--config", str(config_path), "dream"]), 2)
            self.assertEqual(main(["--config", str(config_path), "extract-facts"]), 2)

    def test_dream_uses_config_for_output_model_and_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = root / "events.jsonl"
            output_dir = root / "configured-memory"
            config_path = root / "config.json"
            events.write_text(json.dumps({"source": "codex", "role": "user", "content": "希望项目像 Claude Code", "project": str(root)}, ensure_ascii=False) + "\n", encoding="utf-8")
            config_path.write_text(json.dumps({
                "models": {
                    "primary": {
                        "provider": "anthropic",
                        "model": "test-model",
                        "api_key": "anthropic-key",
                    }
                },
                "model_policy": {"default_profile": "primary", "fallback_chain": ["primary"]},
                "invoke_model": False,
                "output_dir": str(output_dir),
            }, ensure_ascii=False), encoding="utf-8")

            exit_code = main(["--config", str(config_path), "dream", "--input", str(events), "--project", str(root)])

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "ai-prompt.md").exists())
            self.assertTrue((output_dir / "ai-candidates.jsonl").exists())

    def test_run_cli_creates_persistent_run_waiting_for_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events_path = root / "events.jsonl"
            memory_dir = root / "memory"
            config_path = root / "config.json"
            events_path.write_text(json.dumps({"event_id": "event_1", "source": "codex", "session_id": "s1", "role": "user", "event_type": "history_prompt", "project": str(root), "content": "这个项目需要人工审核后才能写正式记忆"}, ensure_ascii=False) + "\n", encoding="utf-8")
            config_path.write_text(json.dumps({"invoke_model": False, "output_dir": str(memory_dir)}, ensure_ascii=False), encoding="utf-8")

            exit_code = main(["--config", str(config_path), "run", "--input", str(events_path), "--project", str(root)])

            self.assertEqual(exit_code, 0)
            states = list((memory_dir / "runs").glob("*/state.json"))
            self.assertEqual(len(states), 1)
            state = json.loads(states[0].read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "waiting_review")
            self.assertTrue(Path(state["artifacts"]["events_path"]).exists())
            self.assertTrue(Path(state["artifacts"]["review_queue_path"]).exists())
            self.assertTrue((Path(state["run_dir"]) / "trace.jsonl").exists())

    def test_run_cli_marks_persistent_run_failed_when_extraction_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events_path = root / "events.jsonl"
            memory_dir = root / "memory"
            config_path = root / "config.json"
            events_path.write_text(json.dumps({"event_id": "event_1", "source": "codex", "role": "user", "event_type": "history_prompt", "project": str(root), "content": "记住这个偏好"}, ensure_ascii=False) + "\n", encoding="utf-8")
            config_path.write_text(json.dumps({"invoke_model": True, "output_dir": str(memory_dir)}, ensure_ascii=False), encoding="utf-8")

            with patch("dream_memory.memory_cli.agent_extract_memory_candidates", side_effect=RuntimeError("provider 403")):
                with self.assertRaises(RuntimeError):
                    main(["--config", str(config_path), "run", "--input", str(events_path), "--project", str(root)])

            state_path = next((memory_dir / "runs").glob("*/state.json"))
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["phase"], "failed")
            self.assertIn("provider 403", state["error"])
            trace = (Path(state["run_dir"]) / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("run_failed", trace)
            self.assertIn("RuntimeError", trace)

    def test_status_cli_reads_run_state_and_lists_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events_path = root / "events.jsonl"
            memory_dir = root / "memory"
            config_path = root / "config.json"
            events_path.write_text(json.dumps({"event_id": "event_1", "source": "codex", "role": "user", "event_type": "history_prompt", "project": str(root), "content": "这个项目需要人工审核后才能写正式记忆"}, ensure_ascii=False) + "\n", encoding="utf-8")
            config_path.write_text(json.dumps({"invoke_model": False, "output_dir": str(memory_dir)}, ensure_ascii=False), encoding="utf-8")
            main(["--config", str(config_path), "run", "--input", str(events_path), "--project", str(root)])
            run_id = json.loads(next((memory_dir / "runs").glob("*/state.json")).read_text(encoding="utf-8"))["run_id"]

            self.assertEqual(main(["--config", str(config_path), "status", "--run-id", run_id]), 0)
            self.assertEqual(main(["--config", str(config_path), "status"]), 0)

    def test_resume_cli_applies_reviewed_decisions_and_completes_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events_path = root / "events.jsonl"
            memory_dir = root / "memory"
            config_path = root / "config.json"
            events_path.write_text(json.dumps({"event_id": "event_1", "source": "codex", "role": "user", "event_type": "history_prompt", "project": str(root), "content": "这个项目需要人工审核后才能写正式记忆"}, ensure_ascii=False) + "\n", encoding="utf-8")
            config_path.write_text(json.dumps({"invoke_model": False, "output_dir": str(memory_dir)}, ensure_ascii=False), encoding="utf-8")
            main(["--config", str(config_path), "run", "--input", str(events_path), "--project", str(root), "--mode", "rules"])
            state_path = next((memory_dir / "runs").glob("*/state.json"))
            state = json.loads(state_path.read_text(encoding="utf-8"))
            candidate = json.loads(Path(state["artifacts"]["candidates_path"]).read_text(encoding="utf-8").splitlines()[0])
            reviewed_path = Path(state["run_dir"]) / "reviewed.jsonl"
            reviewed_path.write_text(json.dumps({"candidate_id": candidate["id"], "action": "approved", "edited_content": candidate["content"], "reviewer": "user", "candidate": candidate}, ensure_ascii=False) + "\n", encoding="utf-8")

            exit_code = main(["--config", str(config_path), "resume", "--run-id", state["run_id"]])

            self.assertEqual(exit_code, 0)
            completed = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(completed["status"], "completed")
            self.assertTrue((memory_dir / "memory_cards.jsonl").exists())
            self.assertTrue((memory_dir / "MEMORY.md").exists())

    def test_trace_cli_prints_run_trace_and_candidate_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events_path = root / "events.jsonl"
            memory_dir = root / "memory"
            config_path = root / "config.json"
            events_path.write_text(json.dumps({"event_id": "event_1", "source": "codex", "role": "user", "event_type": "history_prompt", "project": str(root), "content": "这个项目需要人工审核后才能写正式记忆"}, ensure_ascii=False) + "\n", encoding="utf-8")
            config_path.write_text(json.dumps({"invoke_model": False, "output_dir": str(memory_dir)}, ensure_ascii=False), encoding="utf-8")
            main(["--config", str(config_path), "run", "--input", str(events_path), "--project", str(root), "--mode", "rules"])
            state = json.loads(next((memory_dir / "runs").glob("*/state.json")).read_text(encoding="utf-8"))
            candidate = json.loads(Path(state["artifacts"]["candidates_path"]).read_text(encoding="utf-8").splitlines()[0])

            self.assertEqual(main(["--config", str(config_path), "trace", "--run-id", state["run_id"]]), 0)
            self.assertEqual(main(["--config", str(config_path), "trace", "--run-id", state["run_id"], "--candidate-id", candidate["id"]]), 0)


    def test_init_cli_creates_workspace_and_examples(self):
        with tempfile.TemporaryDirectory() as tmp:
            exit_code = main(["init", "--path", tmp])

            self.assertEqual(exit_code, 0)
            root = Path(tmp)
            self.assertTrue((root / ".dream-memory" / "config.json").exists())
            self.assertTrue((root / ".dream-memory" / "imports").is_dir())
            self.assertTrue((root / ".dream-memory" / "runs").is_dir())
            self.assertTrue((root / "examples" / "sample-events.jsonl").exists())
            self.assertTrue((root / "examples" / "labeled-events.jsonl").exists())




    def test_packaged_workspace_examples_match_repository_samples(self):
        from importlib import resources

        root = Path(__file__).resolve().parents[1] / "examples"
        for name in ["sample-events.jsonl", "reviewed.example.jsonl", "config.openai.json", "config.anthropic.json"]:
            packaged = resources.files("dream_memory") / "examples" / name
            self.assertEqual(packaged.read_text(encoding="utf-8"), (root / name).read_text(encoding="utf-8"))

    def test_packaged_labeled_eval_resource_matches_repository_sample(self):
        from importlib import resources

        repository_sample = Path(__file__).resolve().parents[1] / "examples" / "labeled-events.jsonl"
        packaged_sample = resources.files("dream_memory") / "examples" / "labeled-events.jsonl"

        self.assertEqual(packaged_sample.read_text(encoding="utf-8"), repository_sample.read_text(encoding="utf-8"))

    def test_init_cli_labeled_eval_example_matches_repository_sample_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "workspace"

            exit_code = main(["init", "--output-dir", str(root)])

            self.assertEqual(exit_code, 0)
            generated = root / "examples" / "labeled-events.jsonl"
            repository_sample = Path(__file__).resolve().parents[1] / "examples" / "labeled-events.jsonl"
            generated_rows = [line for line in generated.read_text(encoding="utf-8").splitlines() if line.strip()]
            repository_rows = [line for line in repository_sample.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(generated_rows), len(repository_rows))
            self.assertGreaterEqual(len(generated_rows), 13)
            self.assertIn("credential_location_noise", generated.read_text(encoding="utf-8"))
            self.assertIn("cross_project_noise", generated.read_text(encoding="utf-8"))
            self.assertIn("cross_project_user_preference", generated.read_text(encoding="utf-8"))


    def test_init_output_dir_configured_eval_example_runs_without_extra_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "workspace"

            exit_code = main(["init", "--output-dir", str(root)])
            self.assertEqual(exit_code, 0)

            exit_code = main(["--config", str(root / "config.json"), "eval"])

            self.assertEqual(exit_code, 0)
            payload = json.loads((root / "eval.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["rows"], 16)
            self.assertEqual(payload["precision"], 1.0)
            self.assertEqual(payload["recall"], 1.0)
            self.assertEqual(payload["f1"], 1.0)
            self.assertEqual(payload["true_positive"], 12)
            self.assertEqual(payload["false_positive_count"], 0)
            self.assertEqual(payload["deferred_candidate_count"], 2)
            self.assertEqual(payload["outcome_checked_rows"], 16)
            self.assertEqual(payload["outcome_accuracy"], 1.0)

    def test_init_cli_labeled_eval_example_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            main(["init", "--path", str(root)])
            output = root / "eval.json"

            exit_code = main(["eval", "--input", str(root / "examples" / "labeled-events.jsonl"), "--project", "/tmp/project", "--output", str(output)])

            self.assertEqual(exit_code, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["rows"], 16)
            self.assertEqual(payload["precision"], 1.0)
            self.assertEqual(payload["recall"], 1.0)
            self.assertEqual(payload["f1"], 1.0)
            self.assertEqual(payload["true_positive"], 12)
            self.assertEqual(payload["false_positive_count"], 0)
            self.assertEqual(payload["deferred_candidate_count"], 2)
            self.assertEqual(payload["outcome_checked_rows"], 16)
            self.assertEqual(payload["outcome_accuracy"], 1.0)

    def test_check_provider_detects_inline_secret_misconfiguration(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(
                json.dumps({
                    "models": {
                        "primary": {
                            "provider": "openai",
                            "model": "gpt-test",
                            "api_key_env": "sk-not-an-env-var",
                        }
                    },
                    "model_policy": {"default_profile": "primary", "fallback_chain": ["primary"]},
                }),
                encoding="utf-8",
            )

            exit_code = main(["--config", str(config_path), "check-provider"])

            self.assertEqual(exit_code, 1)

    def test_check_provider_all_reports_each_configured_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(json.dumps({
                "models": {
                    "primary": {"provider": "anthropic", "model": "claude-sonnet-4-6", "api_key": "anthropic-key"},
                    "backup": {"provider": "openai", "model": "gpt-4.1", "api_key": "openai-key"},
                },
                "model_policy": {"default_profile": "primary", "fallback_chain": ["primary", "backup"]},
            }), encoding="utf-8")

            exit_code = main(["--config", str(config_path), "check-provider", "--all"])

            self.assertEqual(exit_code, 0)

    def test_context_cli_accepts_task_query_for_ranking(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cards_path = root / "memory_cards.jsonl"
            cards_path.write_text("\n".join([
                json.dumps({"id": "deploy", "scope": "project", "project": "/tmp/project", "memory_type": "workflow", "summary": "部署前必须先跑 smoke 测试。", "retrieval_hints": ["deploy", "smoke"], "status": "active"}, ensure_ascii=False),
                json.dumps({"id": "product", "scope": "project", "project": "/tmp/project", "memory_type": "decision", "summary": "项目目标是 Claude Code 风格助手。", "retrieval_hints": ["claude-code"], "status": "active"}, ensure_ascii=False),
            ]) + "\n", encoding="utf-8")

            exit_code = main([
                "context",
                "--project", "/tmp/project",
                "--memory-cards", str(cards_path),
                "--task", "准备部署并执行 smoke 测试",
                "--limit", "1",
            ])

            self.assertEqual(exit_code, 0)

    def test_dream_dry_run_does_not_call_model_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = root / "events.jsonl"
            output_dir = root / "memory"
            config_path = root / "config.json"
            events.write_text(json.dumps({"source": "codex", "role": "user", "content": "希望项目像 Claude Code", "project": str(root)}, ensure_ascii=False) + "\n", encoding="utf-8")
            config_path.write_text(json.dumps({
                "output_dir": str(output_dir),
                "models": {
                    "primary": {"provider": "openai", "model": "gpt-4.1", "api_key": "openai-key"}
                },
                "model_policy": {"default_profile": "primary", "fallback_chain": ["primary"]},
            }, ensure_ascii=False), encoding="utf-8")

            with patch("dream_memory.memory_agent.invoke_model_runtime") as invoke:
                exit_code = main(["--config", str(config_path), "dream", "--input", str(events), "--project", str(root), "--dry-run"])

            self.assertEqual(exit_code, 0)
            invoke.assert_not_called()
            self.assertTrue((output_dir / "ai-prompt.md").exists())


    def test_run_cli_records_prompt_filter_counts_in_payload_state_and_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = root / "events.jsonl"
            memory_dir = root / "memory"
            events.write_text("".join([
                json.dumps({"event_id": "state", "source": "claude_code", "role": "system", "event_type": "project_state", "project": str(root), "content": "Claude Code project state for /tmp/project"}, ensure_ascii=False) + "\n",
                json.dumps({"event_id": "durable", "source": "codex", "role": "user", "event_type": "history_prompt", "project": str(root), "content": "正式记忆必须经过人工审核，只有审核通过的候选才允许写入长期记忆。"}, ensure_ascii=False) + "\n",
            ]), encoding="utf-8")

            with patch("sys.stdout", new_callable=lambda: __import__('io').StringIO()) as stdout:
                exit_code = main(["run", "--input", str(events), "--project", str(root), "--output-dir", str(memory_dir), "--mode", "ai", "--dry-run"])

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["input_event_count"], 2)
            self.assertEqual(payload["prompt_event_count"], 1)
            self.assertEqual(payload["filtered_prompt_event_count"], 1)
            state = json.loads(Path(payload["state_path"]).read_text(encoding="utf-8"))
            self.assertEqual(state["counts"]["input_event_count"], 2)
            self.assertEqual(state["counts"]["prompt_event_count"], 1)
            self.assertEqual(state["counts"]["filtered_prompt_event_count"], 1)
            trace = [json.loads(line) for line in (Path(payload["run_dir"]) / "trace.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            extraction_events = [event for event in trace if event.get("event_type") == "ai_extraction_complete"]
            self.assertEqual(extraction_events[-1]["payload"]["prompt_event_count"], 1)
            self.assertEqual(extraction_events[-1]["payload"]["filtered_prompt_event_count"], 1)

    def test_run_cli_records_model_runtime_trace_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events_path = root / "events.jsonl"
            memory_dir = root / "memory"
            config_path = root / "config.json"
            events_path.write_text(json.dumps({
                "event_id": "event_1",
                "source": "codex",
                "session_id": "s1",
                "role": "user",
                "event_type": "history_prompt",
                "project": str(root),
                "content": "用户偏好中文回答",
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            config_path.write_text(json.dumps({
                "output_dir": str(memory_dir),
                "models": {
                    "primary": {"provider": "openai", "model": "gpt-4.1", "api_key": "openai-key"}
                },
                "model_policy": {"default_profile": "primary", "fallback_chain": ["primary"], "retry": {"max_attempts": 1}},
            }, ensure_ascii=False), encoding="utf-8")
            raw = json.dumps({
                "candidates": [{
                    "content": "用户偏好中文回答。",
                    "type": "preference",
                    "scope": "user",
                    "confidence": 0.95,
                    "decision": "promote",
                    "reason": "explicit",
                    "evidence": [{"event_id": "event_1", "source": "codex"}],
                    "tags": ["language"],
                }]
            }, ensure_ascii=False)

            with patch("dream_memory.memory_agent.invoke_model_runtime") as invoke:
                def fake_runtime(prompt, *, runtime_config, trace_callback=None):
                    if trace_callback:
                        trace_callback("model_attempt_started", {"profile": "primary", "provider": "openai", "model": "gpt-4.1", "attempt": 1})
                        trace_callback("model_attempt_succeeded", {"profile": "primary", "provider": "openai", "model": "gpt-4.1", "attempt": 1, "elapsed_ms": 1})
                    return ModelRuntimeResult(text=raw, selected_profile="primary", attempts=[], elapsed_ms=1)

                invoke.side_effect = fake_runtime
                exit_code = main(["--config", str(config_path), "run", "--input", str(events_path), "--project", str(root), "--invoke-model"])

            self.assertEqual(exit_code, 0)
            state = json.loads(next((memory_dir / "runs").glob("*/state.json")).read_text(encoding="utf-8"))
            trace_text = (Path(state["run_dir"]) / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("model_attempt_started", trace_text)
            self.assertIn("model_attempt_succeeded", trace_text)

    def test_resume_auto_exports_when_configured(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events_path = root / "events.jsonl"
            memory_dir = root / "memory"
            export_dir = root / "export"
            config_path = root / "config.json"
            events_path.write_text(
                json.dumps({
                    "event_id": "event_1",
                    "source": "codex",
                    "role": "user",
                    "event_type": "history_prompt",
                    "project": str(export_dir),
                    "content": "这个项目需要人工审核后才能写正式记忆",
                }, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            config_path.write_text(
                json.dumps({
                    "invoke_model": False,
                    "output_dir": str(memory_dir),
                    "mode": "rules",
                    "auto_export": True,
                    "export_target": "both",
                    "export_scope": "project",
                    "export_output_dir": str(export_dir),
                }, ensure_ascii=False),
                encoding="utf-8",
            )
            main(["--config", str(config_path), "run", "--input", str(events_path), "--project", str(export_dir), "--mode", "rules"])
            state_path = next((memory_dir / "runs").glob("*/state.json"))
            state = json.loads(state_path.read_text(encoding="utf-8"))
            candidate = json.loads(Path(state["artifacts"]["candidates_path"]).read_text(encoding="utf-8").splitlines()[0])
            reviewed_path = Path(state["run_dir"]) / "reviewed.jsonl"
            reviewed_path.write_text(
                json.dumps({
                    "candidate_id": candidate["id"],
                    "action": "approved",
                    "edited_content": candidate["content"],
                    "reviewer": "user",
                    "candidate": candidate,
                }, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            exit_code = main(["--config", str(config_path), "resume", "--run-id", state["run_id"]])

            self.assertEqual(exit_code, 0)
            self.assertTrue((export_dir / "AGENTS.md").exists())
            self.assertTrue((export_dir / "CLAUDE.md").exists())


if __name__ == "__main__":
    unittest.main()
