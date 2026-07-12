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
from dream_memory.memory_dreaming import DreamResult
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

    def test_scan_cli_redacts_sensitive_paths_from_output_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            codex = root / "codex-sk-test-secret"
            codex.mkdir()
            (codex / "history.jsonl").write_text('{"session_id":"s1","ts":1,"text":"hello"}\n', encoding="utf-8")
            claude = root / ".claude"
            claude.mkdir()
            out = root / "scan.json"

            exit_code = main(["scan", "--codex-home", str(codex), "--claude-home", str(claude), "--output", str(out)])

            output_text = out.read_text(encoding="utf-8")
            self.assertEqual(exit_code, 0)
            self.assertNotIn("sk-test-secret", output_text)
            self.assertIn("<redacted>", output_text)

    def test_scan_cli_rejects_unwritable_output_without_traceback_or_tmp(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            codex = root / ".codex"
            codex.mkdir()
            (codex / "history.jsonl").write_text('{"session_id":"s1","ts":1,"text":"hello"}\n', encoding="utf-8")
            claude = root / ".claude"
            claude.mkdir()
            output = root / "scan.json"
            output.mkdir()

            with patch("sys.stderr", new_callable=lambda: io.StringIO()) as stderr:
                exit_code = main(["scan", "--codex-home", str(codex), "--claude-home", str(claude), "--output", str(output)])

            self.assertEqual(exit_code, 2)
            self.assertIn("scan output path is not writable", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertTrue(output.is_dir())
            self.assertFalse((root / ".scan.json.tmp").exists())

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

    def test_import_cli_reports_malformed_source_jsonl_warnings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            codex = root / ".codex"
            codex.mkdir()
            (codex / "history.jsonl").write_text(
                json.dumps({"session_id": "s1", "ts": 1, "text": "hello"}, ensure_ascii=False)
                + "\nnot-json\n[]\n",
                encoding="utf-8",
            )
            out_dir = root / "imports"

            with patch("sys.stdout", new_callable=lambda: io.StringIO()) as stdout:
                exit_code = main(["import", "codex", "--codex-home", str(codex), "--output-dir", str(out_dir), "--dry-run"])

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            report = json.loads((out_dir / "import-report.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(payload["event_count"], 1)
            self.assertEqual(payload["warning_count"], 2)
            self.assertIn("hello", (out_dir / "codex-events.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(report["warning_count"], 2)
            self.assertEqual(report["warnings"][0]["line"], 2)
            self.assertIn("invalid JSON", report["warnings"][0]["error"])
            self.assertEqual(report["warnings"][1]["line"], 3)
            self.assertIn("expected JSON object", report["warnings"][1]["error"])

    def test_import_cli_redacts_sensitive_paths_from_report_warnings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            codex = root / "codex-sk-test-secret"
            codex.mkdir()
            (codex / "history.jsonl").write_text("{not json\n", encoding="utf-8")
            out_dir = root / "imports"

            exit_code = main(["import", "codex", "--codex-home", str(codex), "--output-dir", str(out_dir), "--dry-run"])

            report_text = (out_dir / "import-report.json").read_text(encoding="utf-8")
            self.assertEqual(exit_code, 0)
            self.assertNotIn("sk-test-secret", report_text)
            self.assertIn("<redacted>", report_text)

    def test_import_cli_redacts_sensitive_output_dir_from_stdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            codex = root / ".codex"
            codex.mkdir()
            (codex / "history.jsonl").write_text('{"session_id":"s1","ts":1,"text":"hello"}\n', encoding="utf-8")
            out_dir = root / "imports-sk-test-secret"

            with patch("sys.stdout", new_callable=lambda: io.StringIO()) as stdout:
                exit_code = main(["import", "codex", "--codex-home", str(codex), "--output-dir", str(out_dir), "--dry-run"])

            output_text = stdout.getvalue()
            self.assertEqual(exit_code, 0)
            self.assertTrue((out_dir / "codex-events.jsonl").exists())
            self.assertNotIn("sk-test-secret", output_text)
            self.assertIn("<redacted>", output_text)

    def test_init_cli_redacts_sensitive_paths_from_stdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace-sk-test-secret"

            with patch("sys.stdout", new_callable=lambda: io.StringIO()) as stdout:
                exit_code = main(["init", "--path", str(workspace)])

            output_text = stdout.getvalue()
            self.assertEqual(exit_code, 0)
            self.assertTrue((workspace / ".dream-memory" / "config.json").exists())
            self.assertNotIn("sk-test-secret", output_text)
            self.assertIn("<redacted>", output_text)

    def test_init_config_cli_redacts_sensitive_output_path_from_stdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config-sk-test-secret.json"

            with patch("sys.stdout", new_callable=lambda: io.StringIO()) as stdout:
                exit_code = main(["init-config", "--output", str(config_path)])

            output_text = stdout.getvalue()
            self.assertEqual(exit_code, 0)
            self.assertTrue(config_path.exists())
            self.assertNotIn("sk-test-secret", output_text)
            self.assertIn("<redacted>", output_text)

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

    def test_extract_facts_cli_redacts_sensitive_facts_path_from_stdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events_path = root / "events.jsonl"
            output_dir = root / "facts-sk-test-secret"
            events_path.write_text(json.dumps({
                "event_id": "event_1",
                "source": "codex",
                "role": "user",
                "event_type": "history_prompt",
                "project": str(root),
                "content": "这个项目需要人工审核后才能写正式记忆",
            }, ensure_ascii=False) + "\n", encoding="utf-8")

            with patch("sys.stdout", new_callable=lambda: io.StringIO()) as stdout:
                exit_code = main(["extract-facts", "--input", str(events_path), "--output-dir", str(output_dir)])

            output_text = stdout.getvalue()
            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "facts.jsonl").exists())
            self.assertNotIn("sk-test-secret", output_text)
            self.assertIn("<redacted>", output_text)

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

    def test_review_summary_cli_redacts_sensitive_legacy_bucket_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"output_dir": str(memory_dir)}, ensure_ascii=False), encoding="utf-8")
            state = create_run_state(memory_dir=memory_dir, project=str(root), input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            queue_path = Path(state["run_dir"]) / "review_queue.jsonl"
            queue_path.write_text(json.dumps({
                "candidate_id": "cand_1",
                "suggested_action": "api_key=sk-test-secret",
                "status": "token 在 config.yaml 配置中",
                "candidate": {
                    "id": "cand_1",
                    "type": "项目的 API key 在 key.txt 文件中。",
                    "scope": "project",
                    "content": "安全配置说明需要人工审核。",
                    "evidence": [{"event_id": "event_1"}],
                },
                "quality_signals": {"evidence_quality": "api_key=sk-test-secret"},
                "dream_analysis": {"dream_score": 0.82},
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(queue_path)})

            with patch("sys.stdout", new_callable=lambda: __import__('io').StringIO()) as stdout:
                exit_code = main(["--config", str(config_path), "review-summary", "--run-id", state["run_id"]])

            self.assertEqual(exit_code, 0)
            output = stdout.getvalue()
            self.assertNotIn("sk-test-secret", output)
            self.assertNotIn("API key", output)
            self.assertNotIn("key.txt", output)
            self.assertNotIn("config.yaml", output)
            self.assertIn("<redacted>", output)

    def test_review_summary_cli_preserves_colliding_redacted_bucket_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"output_dir": str(memory_dir)}, ensure_ascii=False), encoding="utf-8")
            state = create_run_state(memory_dir=memory_dir, project=str(root), input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            queue_path = Path(state["run_dir"]) / "review_queue.jsonl"
            queue_path.write_text("\n".join([
                json.dumps({"candidate_id": "cand_1", "suggested_action": "api_key=sk-test-secret-a", "status": "token=sk-test-secret-a", "candidate": {"id": "cand_1", "type": "workflow", "scope": "project"}, "quality_signals": {"evidence_quality": "api_key=sk-test-secret-a"}, "dream_analysis": {"dream_score": 0.82}}, ensure_ascii=False),
                json.dumps({"candidate_id": "cand_2", "suggested_action": "api_key=sk-test-secret-b", "status": "token=sk-test-secret-b", "candidate": {"id": "cand_2", "type": "workflow", "scope": "project"}, "quality_signals": {"evidence_quality": "api_key=sk-test-secret-b"}, "dream_analysis": {"dream_score": 0.91}}, ensure_ascii=False),
            ]) + "\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(queue_path)})

            with patch("sys.stdout", new_callable=lambda: __import__('io').StringIO()) as stdout:
                exit_code = main(["--config", str(config_path), "review-summary", "--run-id", state["run_id"]])

            output = stdout.getvalue()
            payload = json.loads(output)
            self.assertEqual(exit_code, 0)
            self.assertNotIn("sk-test-secret", output)
            self.assertEqual(payload["by_suggested_action"], {"<redacted>": 1, "<redacted>-2": 1})
            self.assertEqual(payload["by_status"], {"<redacted>": 1, "<redacted>-2": 1})
            self.assertEqual(payload["by_evidence_quality"], {"<redacted>": 1, "<redacted>-2": 1})

    def test_review_summary_cli_redacts_sensitive_path_from_malformed_queue_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            review_queue = root / "review-queue-sk-test-secret.jsonl"
            review_queue.write_text("{not json\n", encoding="utf-8")

            with patch("sys.stderr", new_callable=lambda: __import__('io').StringIO()) as stderr:
                exit_code = main(["review-summary", "--review-queue", str(review_queue)])

            error_text = stderr.getvalue()
            self.assertEqual(exit_code, 2)
            self.assertIn("invalid review queue JSON", error_text)
            self.assertNotIn("sk-test-secret", error_text)
            self.assertIn("<redacted>", error_text)

    def test_run_scoped_cli_commands_reject_review_queue_outside_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"output_dir": str(memory_dir)}, ensure_ascii=False), encoding="utf-8")
            state = create_run_state(memory_dir=memory_dir, project=str(root), input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            outside_queue = root / "outside-review-queue.jsonl"
            outside_queue.write_text(json.dumps({
                "candidate_id": "leaked",
                "suggested_action": "create",
                "candidate": {"id": "leaked", "type": "workflow", "scope": "user", "content": "不应读取外部队列。"},
                "dream_analysis": {"dream_score": 0.9},
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(outside_queue)})

            with patch("sys.stderr", new_callable=lambda: __import__('io').StringIO()) as summary_stderr:
                summary_exit = main(["--config", str(config_path), "review-summary", "--run-id", state["run_id"]])
            with patch("sys.stderr", new_callable=lambda: __import__('io').StringIO()) as auto_stderr:
                auto_exit = main(["--config", str(config_path), "auto-review", "--run-id", state["run_id"]])

            self.assertEqual(summary_exit, 2)
            self.assertEqual(auto_exit, 2)
            self.assertIn("outside run directory", summary_stderr.getvalue())
            self.assertIn("outside run directory", auto_stderr.getvalue())
            self.assertFalse((Path(state["run_dir"]) / "reviewed.jsonl").exists())

    def test_run_scoped_auto_review_requires_review_queue_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"output_dir": str(memory_dir)}, ensure_ascii=False), encoding="utf-8")
            state = create_run_state(memory_dir=memory_dir, project=str(root), input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            update_run_state(state, status="waiting_review", phase="review")

            with patch("sys.stderr", new_callable=lambda: __import__('io').StringIO()) as stderr:
                exit_code = main(["--config", str(config_path), "auto-review", "--run-id", state["run_id"]])

            self.assertEqual(exit_code, 2)
            self.assertIn("review queue not found", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())

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

    def test_auto_review_cli_writes_reviewed_decisions_for_run_queue(self):
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
            self.assertEqual([row["action"] for row in rows], ["approved", "rejected"])
            updated = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(updated["counts"]["auto_review_count"], 2)
            self.assertIn("auto_reviewed", (run_dir / "trace.jsonl").read_text(encoding="utf-8"))

    def test_auto_review_cli_does_not_approve_candidate_without_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"output_dir": str(memory_dir)}, ensure_ascii=False), encoding="utf-8")
            state = create_run_state(memory_dir=memory_dir, project=str(root), input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            run_dir = Path(state["run_dir"])
            queue_path = run_dir / "review_queue.jsonl"
            queue_path.write_text(json.dumps({
                "candidate_id": "create_1",
                "suggested_action": "create",
                "candidate": {"id": "create_1", "type": "preference", "scope": "user", "content": "用户偏好中文回答。"},
                "dream_analysis": {"dream_score": 0.82},
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(queue_path)})

            exit_code = main(["--config", str(config_path), "auto-review", "--run-id", state["run_id"], "--min-score", "0.7"])

            self.assertEqual(exit_code, 0)
            rows = [json.loads(line) for line in (run_dir / "reviewed.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["action"], "needs_more_evidence")
            self.assertEqual(rows[0]["candidate_id"], "create_1")

    def test_auto_review_cli_skips_sensitive_candidate_without_writing_reviewed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"output_dir": str(memory_dir)}, ensure_ascii=False), encoding="utf-8")
            state = create_run_state(memory_dir=memory_dir, project=str(root), input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            run_dir = Path(state["run_dir"])
            queue_path = run_dir / "review_queue.jsonl"
            queue_path.write_text(json.dumps({
                "candidate_id": "create_1",
                "suggested_action": "create",
                "candidate": {
                    "id": "create_1",
                    "type": "workflow",
                    "scope": "project",
                    "project": str(root),
                    "content": "项目的 API key 在 key.txt 文件中。",
                    "evidence": [{"event_id": "event_1", "quote": "项目的 API key 在 key.txt 文件中。"}],
                },
                "dream_analysis": {"dream_score": 0.92},
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(queue_path)})

            exit_code = main(["--config", str(config_path), "auto-review", "--run-id", state["run_id"], "--min-score", "0.7"])

            self.assertEqual(exit_code, 0)
            self.assertFalse((run_dir / "reviewed.jsonl").exists())
            updated = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
            self.assertNotIn("auto_review_count", updated.get("counts", {}))

    def test_auto_review_cli_skips_sensitive_queue_metadata_without_writing_reviewed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"output_dir": str(memory_dir)}, ensure_ascii=False), encoding="utf-8")
            state = create_run_state(memory_dir=memory_dir, project=str(root), input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            run_dir = Path(state["run_dir"])
            queue_path = run_dir / "review_queue.jsonl"
            queue_path.write_text(json.dumps({
                "candidate_id": "create_1",
                "suggested_action": "create",
                "candidate": {
                    "id": "create_1",
                    "type": "workflow",
                    "scope": "project",
                    "project": str(root),
                    "content": "安全配置说明需要通过人工审核后再进入正式记忆。",
                    "evidence": [{"event_id": "event_1"}],
                },
                "quality_signals": {
                    "value_class": "similar_existing",
                    "matched_memory_id": "secret",
                    "matched_memory_summary": "项目的 API key 在 key.txt 文件中。",
                },
                "conflicts": [{
                    "memory_id": "secret",
                    "summary": "项目的 API key 在 key.txt 文件中。",
                    "similarity": 0.5,
                }],
                "dream_analysis": {"dream_score": 0.92},
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(queue_path)})

            exit_code = main(["--config", str(config_path), "auto-review", "--run-id", state["run_id"], "--min-score", "0.7"])

            self.assertEqual(exit_code, 0)
            self.assertFalse((run_dir / "reviewed.jsonl").exists())
            updated = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
            self.assertNotIn("auto_review_count", updated.get("counts", {}))

    def test_auto_review_cli_rejects_malformed_review_queue_without_partial_reviewed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"output_dir": str(memory_dir)}, ensure_ascii=False), encoding="utf-8")
            state = create_run_state(memory_dir=memory_dir, project=str(root), input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            run_dir = Path(state["run_dir"])
            queue_path = run_dir / "review_queue.jsonl"
            queue_path.write_text(json.dumps({
                "candidate_id": "create_1",
                "suggested_action": "create",
                "candidate": {
                    "id": "create_1",
                    "type": "preference",
                    "scope": "user",
                    "content": "用户偏好中文回答。",
                    "evidence": [{"event_id": "event_1", "quote": "用户偏好中文回答。"}],
                },
                "dream_analysis": {"dream_score": 0.9},
            }, ensure_ascii=False) + "\n{not json\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(queue_path)})

            result = subprocess.run(
                ["uv", "run", "dream-memory", "--config", str(config_path), "auto-review", "--run-id", state["run_id"], "--min-score", "0.7"],
                cwd=Path.cwd(),
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("invalid review queue JSON", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse((run_dir / "reviewed.jsonl").exists())

    def test_auto_review_cli_keep_review_skips_candidate_without_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"output_dir": str(memory_dir)}, ensure_ascii=False), encoding="utf-8")
            state = create_run_state(memory_dir=memory_dir, project=str(root), input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            run_dir = Path(state["run_dir"])
            queue_path = run_dir / "review_queue.jsonl"
            queue_path.write_text(json.dumps({
                "candidate_id": "create_1",
                "suggested_action": "create",
                "candidate": {"id": "create_1", "type": "preference", "scope": "user", "content": "用户偏好中文回答。"},
                "dream_analysis": {"dream_score": 0.82},
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(queue_path)})

            with patch("sys.stdout", new_callable=lambda: __import__('io').StringIO()) as stdout:
                exit_code = main(["--config", str(config_path), "auto-review", "--run-id", state["run_id"], "--min-score", "0.7", "--keep-review"])

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["decision_count"], 0)
            self.assertEqual(payload["skip_reasons"]["missing_evidence"], 1)
            self.assertFalse((run_dir / "reviewed.jsonl").exists())

    def test_auto_review_cli_skips_merges_by_default(self):
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
            before_state = (run_dir / "state.json").read_text(encoding="utf-8")

            exit_code = main(["--config", str(config_path), "auto-review", "--run-id", state["run_id"]])

            self.assertEqual(exit_code, 0)
            reviewed_path = run_dir / "reviewed.jsonl"
            self.assertFalse(reviewed_path.exists())
            self.assertEqual((run_dir / "state.json").read_text(encoding="utf-8"), before_state)

            exit_code = main(["--config", str(config_path), "auto-review", "--run-id", state["run_id"], "--include-merges", "--force"])

            self.assertEqual(exit_code, 0)
            rows = [json.loads(line) for line in reviewed_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["action"], "merged")
            self.assertEqual(rows[0]["candidate"]["quality_signals"]["matched_memory_id"], "mem_existing")

    def test_auto_review_cli_include_merges_resumes_and_supersedes_existing_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            cards_path = memory_dir / "memory_cards.jsonl"
            config_path = root / "config.json"
            memory_dir.mkdir()
            cards_path.write_text(json.dumps({
                "id": "mem_existing",
                "scope": "user",
                "memory_type": "preference",
                "summary": "用户偏好先确认再推进。",
                "evidence_refs": ["event_old"],
                "approved_by": "user",
                "approved_at": "2026-07-01T00:00:00Z",
                "status": "active",
                "retrieval_hints": ["autonomy"],
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            config_path.write_text(json.dumps({
                "output_dir": str(memory_dir),
                "memory_cards": str(cards_path),
            }, ensure_ascii=False), encoding="utf-8")
            state = create_run_state(memory_dir=memory_dir, project=str(root), input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            run_dir = Path(state["run_dir"])
            queue_path = run_dir / "review_queue.jsonl"
            queue_path.write_text(json.dumps({
                "candidate_id": "merge_1",
                "suggested_action": "merge",
                "candidate": {
                    "id": "merge_1",
                    "type": "preference",
                    "scope": "user",
                    "content": "用户偏好上下文清楚时直接推进。",
                    "evidence": [{"event_id": "event_new"}],
                    "tags": ["autonomy"],
                },
                "quality_signals": {"matched_memory_id": "mem_existing", "similarity": 0.72},
                "dream_analysis": {"dream_score": 0.82},
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(queue_path)})

            auto_review_exit = main(["--config", str(config_path), "auto-review", "--run-id", state["run_id"], "--include-merges"])
            resume_exit = main(["--config", str(config_path), "resume", "--run-id", state["run_id"]])

            self.assertEqual(auto_review_exit, 0)
            self.assertEqual(resume_exit, 0)
            completed = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(completed["status"], "completed")
            cards = [
                json.loads(line)
                for line in (memory_dir / "memory_cards.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            by_id = {card["id"]: card for card in cards}
            self.assertEqual(by_id["mem_existing"]["status"], "superseded")
            self.assertTrue(any(card["summary"] == "用户偏好上下文清楚时直接推进。" and card["status"] == "active" for card in cards))

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
            before_state = (run_dir / "state.json").read_text(encoding="utf-8")

            exit_code = main(["--config", str(config_path), "auto-review", "--run-id", state["run_id"]])

            self.assertEqual(exit_code, 0)
            reviewed_path = run_dir / "reviewed.jsonl"
            self.assertFalse(reviewed_path.exists())
            self.assertEqual((run_dir / "state.json").read_text(encoding="utf-8"), before_state)

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
            self.assertEqual(payload["decision_count"], 1)
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
            self.assertEqual(payload["decision_count"], 1)
            self.assertFalse((run_dir / "reviewed.jsonl").exists())
            self.assertEqual((run_dir / "state.json").read_text(encoding="utf-8"), before_state)

    def test_auto_review_cli_reports_skip_reasons_for_review_and_low_score(self):
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
            self.assertEqual(payload["skip_reasons"]["below_min_score"], 1)
            self.assertEqual(payload["skip_reasons"]["requires_manual_review"], 1)
            self.assertEqual(payload["decision_count"], 0)
            self.assertFalse((run_dir / "reviewed.jsonl").exists())

    def test_auto_review_cli_does_not_write_empty_reviewed_for_empty_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"output_dir": str(memory_dir)}, ensure_ascii=False), encoding="utf-8")
            state = create_run_state(memory_dir=memory_dir, project=str(root), input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            run_dir = Path(state["run_dir"])
            queue_path = run_dir / "review_queue.jsonl"
            queue_path.write_text("", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(queue_path)})
            before_state = (run_dir / "state.json").read_text(encoding="utf-8")

            with patch("sys.stdout", new_callable=lambda: __import__('io').StringIO()) as stdout:
                exit_code = main(["--config", str(config_path), "auto-review", "--run-id", state["run_id"]])

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["decision_count"], 0)
            self.assertEqual(payload["skipped"], 0)
            self.assertFalse((run_dir / "reviewed.jsonl").exists())
            self.assertEqual((run_dir / "state.json").read_text(encoding="utf-8"), before_state)

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

            with patch("sys.stderr", new_callable=lambda: __import__('io').StringIO()) as stderr:
                exit_code = main(["--config", str(config_path), "auto-review", "--run-id", state["run_id"]])

            self.assertEqual(exit_code, 2)
            self.assertIn("reviewed output already exists", stderr.getvalue())
            rows = [json.loads(line) for line in reviewed_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["candidate_id"], "manual")

            exit_code = main(["--config", str(config_path), "auto-review", "--run-id", state["run_id"], "--force"])

            self.assertEqual(exit_code, 0)
            overwritten = [json.loads(line) for line in reviewed_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(overwritten[0]["candidate_id"], "cand_1")

    def test_auto_review_cli_redacts_sensitive_existing_reviewed_output_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"output_dir": str(memory_dir)}, ensure_ascii=False), encoding="utf-8")
            state = create_run_state(memory_dir=memory_dir, project=str(root), input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            queue_path = Path(state["run_dir"]) / "review_queue.jsonl"
            queue_path.write_text(json.dumps({
                "candidate_id": "cand_1",
                "suggested_action": "create",
                "candidate": {
                    "id": "cand_1",
                    "type": "workflow",
                    "scope": "user",
                    "content": "需要保留人工审核链路。",
                    "evidence": [{"event_id": "event_1"}],
                },
                "dream_analysis": {"dream_score": 0.9},
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            reviewed_output = root / "reviewed-output-sk-test-secret.jsonl"
            reviewed_output.write_text("already exists\n", encoding="utf-8")

            with patch("sys.stderr", new_callable=lambda: io.StringIO()) as stderr:
                exit_code = main([
                    "--config", str(config_path),
                    "auto-review",
                    "--run-id", state["run_id"],
                    "--review-queue", str(queue_path),
                    "--reviewed-output", str(reviewed_output),
                    "--min-score", "0.7",
                ])

            error_text = stderr.getvalue()
            self.assertEqual(exit_code, 2)
            self.assertIn("reviewed output already exists", error_text)
            self.assertNotIn("sk-test-secret", error_text)
            self.assertIn("<redacted>", error_text)

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

    def test_context_cli_rejects_malformed_memory_cards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cards_path = root / "memory_cards.jsonl"
            cards_path.write_text(
                json.dumps({"id": "mem_1", "scope": "user", "memory_type": "preference", "summary": "用户偏好中文回答。", "status": "active"}, ensure_ascii=False)
                + "\n{not json\n",
                encoding="utf-8",
            )

            with patch("sys.stderr", new_callable=lambda: io.StringIO()) as stderr:
                exit_code = main([
                    "context",
                    "--project", "/tmp/project",
                    "--memory-cards", str(cards_path),
                    "--limit", "2",
                ])

            self.assertEqual(exit_code, 2)
            self.assertIn("invalid memory cards JSON", stderr.getvalue())

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

    def test_review_cli_redacts_sensitive_review_queue_path_from_stdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates_path = root / "candidates.jsonl"
            missing_cards_path = root / "missing-memory-cards.jsonl"
            output_dir = root / "review-sk-test-secret"
            candidates_path.write_text(json.dumps({
                "id": "cand_1",
                "scope": "user",
                "type": "preference",
                "content": "用户偏好中文回答。",
                "evidence": [{"event_id": "event_1"}],
            }, ensure_ascii=False) + "\n", encoding="utf-8")

            with patch("sys.stdout", new_callable=lambda: io.StringIO()) as stdout:
                exit_code = main(["review", "--candidates", str(candidates_path), "--memory-cards", str(missing_cards_path), "--output-dir", str(output_dir)])

            output_text = stdout.getvalue()
            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "review_queue.jsonl").exists())
            self.assertNotIn("sk-test-secret", output_text)
            self.assertIn("<redacted>", output_text)

    def test_review_cli_requires_existing_candidates_file_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing_candidates = root / "missing-candidates.jsonl"
            output_dir = root / "memory"

            result = subprocess.run(
                ["uv", "run", "dream-memory", "review", "--candidates", str(missing_candidates), "--output-dir", str(output_dir)],
                cwd=Path.cwd(),
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("candidates not found", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse((output_dir / "review_queue.jsonl").exists())

    def test_review_cli_rejects_empty_candidates_without_writing_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates_path = root / "candidates.jsonl"
            candidates_path.write_text("\n\n", encoding="utf-8")
            output_dir = root / "memory"

            result = subprocess.run(
                ["uv", "run", "dream-memory", "review", "--candidates", str(candidates_path), "--output-dir", str(output_dir)],
                cwd=Path.cwd(),
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("candidates is empty", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse((output_dir / "review_queue.jsonl").exists())

    def test_review_cli_rejects_malformed_candidates_line_without_partial_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates_path = root / "candidates.jsonl"
            candidates_path.write_text(json.dumps({
                "id": "cand_1",
                "type": "preference",
                "scope": "user",
                "content": "用户偏好中文回答。",
                "evidence": [{"event_id": "event_1", "quote": "用户偏好中文回答。"}],
            }, ensure_ascii=False) + "\n{not json\n", encoding="utf-8")
            output_dir = root / "memory"

            result = subprocess.run(
                ["uv", "run", "dream-memory", "review", "--candidates", str(candidates_path), "--output-dir", str(output_dir)],
                cwd=Path.cwd(),
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("invalid candidates JSON", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse((output_dir / "review_queue.jsonl").exists())

    def test_review_cli_rejects_malformed_memory_cards_without_partial_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates_path = root / "candidates.jsonl"
            memory_cards = root / "memory_cards.jsonl"
            output_dir = root / "memory"
            candidates_path.write_text(json.dumps({
                "id": "cand_1",
                "type": "preference",
                "scope": "user",
                "content": "用户偏好中文回答。",
                "evidence": [{"event_id": "event_1", "quote": "用户偏好中文回答。"}],
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            memory_cards.write_text(
                json.dumps({"id": "mem_1", "scope": "user", "memory_type": "preference", "summary": "已有记忆。", "status": "active"}, ensure_ascii=False)
                + "\n{not json\n",
                encoding="utf-8",
            )

            with patch("sys.stderr", new_callable=lambda: io.StringIO()) as stderr:
                exit_code = main([
                    "review",
                    "--candidates", str(candidates_path),
                    "--memory-cards", str(memory_cards),
                    "--output-dir", str(output_dir),
                ])

            self.assertEqual(exit_code, 2)
            self.assertIn("invalid memory cards JSON", stderr.getvalue())
            self.assertFalse((output_dir / "review_queue.jsonl").exists())

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

    def test_apply_cli_redacts_sensitive_output_paths_from_stdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviewed_path = root / "reviewed.jsonl"
            output_dir = root / "apply-sk-test-secret"
            reviewed_path.write_text(json.dumps({
                "candidate_id": "cand_1",
                "action": "approved",
                "edited_content": "用户偏好中文回答。",
                "reviewer": "user",
                "candidate": {"id": "cand_1", "type": "preference", "scope": "user", "content": "用户偏好中文回答。", "evidence": [{"event_id": "event_1"}]},
            }, ensure_ascii=False) + "\n", encoding="utf-8")

            with patch("sys.stdout", new_callable=lambda: io.StringIO()) as stdout:
                exit_code = main(["apply", "--reviewed", str(reviewed_path), "--output-dir", str(output_dir), "--reviewer", "user"])

            output_text = stdout.getvalue()
            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "memory_cards.jsonl").exists())
            self.assertTrue((output_dir / "review_decisions.jsonl").exists())
            self.assertNotIn("sk-test-secret", output_text)
            self.assertIn("<redacted>", output_text)

    def test_apply_cli_redacts_sensitive_reviewed_path_from_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            malformed_reviewed = root / "bad-reviewed-sk-test-secret.jsonl"
            malformed_reviewed.write_text("{not json\n", encoding="utf-8")
            cases = [
                ("missing", root / "reviewed-sk-test-secret.jsonl", "reviewed decisions not found"),
                ("malformed", malformed_reviewed, "invalid reviewed decisions JSON"),
            ]

            for label, reviewed_path, expected in cases:
                with self.subTest(label):
                    with patch("sys.stderr", new_callable=lambda: io.StringIO()) as stderr:
                        exit_code = main(["apply", "--reviewed", str(reviewed_path), "--reviewer", "tester"])

                    error_text = stderr.getvalue()
                    self.assertEqual(exit_code, 2)
                    self.assertIn(expected, error_text)
                    self.assertNotIn("sk-test-secret", error_text)
                    self.assertIn("<redacted>", error_text)

    def test_apply_cli_rejects_malformed_existing_memory_cards_without_rewriting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviewed_path = root / "reviewed.jsonl"
            memory_cards = root / "memory_cards.jsonl"
            output_dir = root / "memory"
            reviewed_path.write_text(json.dumps({
                "candidate_id": "cand_1",
                "action": "approved",
                "edited_content": "用户偏好中文回答。",
                "reviewer": "user",
                "candidate": {
                    "id": "cand_1",
                    "type": "preference",
                    "scope": "user",
                    "content": "用户偏好中文回答。",
                    "evidence": [{"event_id": "event_1"}],
                },
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            memory_cards.write_text(
                json.dumps({"id": "mem_1", "scope": "user", "memory_type": "preference", "summary": "已有记忆。", "status": "active"}, ensure_ascii=False)
                + "\n{not json\n",
                encoding="utf-8",
            )

            with patch("sys.stderr", new_callable=lambda: io.StringIO()) as stderr:
                exit_code = main([
                    "apply",
                    "--reviewed", str(reviewed_path),
                    "--memory-cards", str(memory_cards),
                    "--output-dir", str(output_dir),
                    "--reviewer", "user",
                ])

            self.assertEqual(exit_code, 2)
            self.assertIn("invalid memory cards JSON", stderr.getvalue())
            self.assertFalse((output_dir / "memory_cards.jsonl").exists())
            self.assertIn("{not json", memory_cards.read_text(encoding="utf-8"))

    def test_apply_cli_rejects_unwritable_memory_markdown_without_partial_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviewed_path = root / "reviewed.jsonl"
            output_dir = root / "memory"
            output_dir.mkdir()
            (output_dir / "MEMORY.md").mkdir()
            reviewed_path.write_text(json.dumps({
                "candidate_id": "cand_1",
                "action": "approved",
                "edited_content": "用户偏好中文回答。",
                "reviewer": "user",
                "candidate": {
                    "id": "cand_1",
                    "type": "preference",
                    "scope": "user",
                    "content": "用户偏好中文回答。",
                    "evidence": [{"event_id": "event_1"}],
                },
            }, ensure_ascii=False) + "\n", encoding="utf-8")

            result = subprocess.run(
                ["uv", "run", "dream-memory", "apply", "--reviewed", str(reviewed_path), "--output-dir", str(output_dir), "--reviewer", "user"],
                cwd=Path.cwd(),
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("memory output path is not writable", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse((output_dir / "memory_cards.jsonl").exists())
            self.assertFalse((output_dir / "review_decisions.jsonl").exists())

    def test_summary_cli_redacts_sensitive_output_path_from_stdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cards_path = root / "memory_cards.jsonl"
            output_path = root / "summary-sk-test-secret.md"
            cards_path.write_text(json.dumps({
                "id": "mem_1",
                "scope": "user",
                "memory_type": "preference",
                "summary": "用户偏好中文回答。",
                "status": "active",
            }, ensure_ascii=False) + "\n", encoding="utf-8")

            with patch("sys.stdout", new_callable=lambda: io.StringIO()) as stdout:
                exit_code = main(["summary", "--memory-cards", str(cards_path), "--output", str(output_path)])

            output_text = stdout.getvalue()
            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.exists())
            self.assertNotIn("sk-test-secret", output_text)
            self.assertIn("<redacted>", output_text)

    def test_summary_cli_rejects_malformed_memory_cards_without_writing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cards_path = root / "memory_cards.jsonl"
            output_path = root / "summary.md"
            cards_path.write_text(
                json.dumps({"id": "mem_1", "scope": "user", "memory_type": "preference", "summary": "用户偏好中文回答。", "status": "active"}, ensure_ascii=False)
                + "\n{not json\n",
                encoding="utf-8",
            )

            with patch("sys.stderr", new_callable=lambda: io.StringIO()) as stderr:
                exit_code = main(["summary", "--memory-cards", str(cards_path), "--output", str(output_path)])

            self.assertEqual(exit_code, 2)
            self.assertIn("invalid memory cards JSON", stderr.getvalue())
            self.assertFalse(output_path.exists())

    def test_apply_cli_uses_latest_review_decision_per_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviewed_path = root / "reviewed.jsonl"
            output_dir = root / "memory"
            reviewed_path.write_text("\n".join([
                json.dumps({
                    "candidate_id": "cand_1",
                    "action": "approved",
                    "edited_content": "用户偏好中文回答。",
                    "reviewer": "user",
                    "candidate": {
                        "id": "cand_1",
                        "type": "preference",
                        "scope": "user",
                        "content": "用户偏好中文回答。",
                        "evidence": [{"event_id": "event_1"}],
                    },
                }, ensure_ascii=False),
                json.dumps({"candidate_id": "cand_1", "action": "rejected", "reviewer": "user", "candidate": {"id": "cand_1"}}, ensure_ascii=False),
            ]) + "\n", encoding="utf-8")

            exit_code = main(["apply", "--reviewed", str(reviewed_path), "--output-dir", str(output_dir), "--reviewer", "user"])

            self.assertEqual(exit_code, 0)
            self.assertNotIn("用户偏好中文回答", (output_dir / "MEMORY.md").read_text(encoding="utf-8"))
            decisions = [json.loads(line) for line in (output_dir / "review_decisions.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([decision["status"] for decision in decisions], ["rejected"])

    def test_apply_cli_requires_existing_reviewed_file_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing_reviewed = root / "missing-reviewed.jsonl"
            output_dir = root / "memory"

            result = subprocess.run(
                ["uv", "run", "dream-memory", "apply", "--reviewed", str(missing_reviewed), "--output-dir", str(output_dir), "--reviewer", "user"],
                cwd=Path.cwd(),
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("reviewed decisions not found", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse((output_dir / "MEMORY.md").exists())

    def test_apply_cli_rejects_empty_reviewed_decisions_without_writing_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviewed_path = root / "reviewed.jsonl"
            reviewed_path.write_text("\n\n", encoding="utf-8")
            output_dir = root / "memory"

            with patch("sys.stderr", new_callable=lambda: __import__('io').StringIO()) as stderr:
                exit_code = main(["apply", "--reviewed", str(reviewed_path), "--output-dir", str(output_dir), "--reviewer", "user"])

            self.assertEqual(exit_code, 2)
            self.assertIn("reviewed decisions is empty", stderr.getvalue())
            self.assertFalse((output_dir / "MEMORY.md").exists())

    def test_apply_cli_rejects_malformed_reviewed_line_without_partial_apply(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviewed_path = root / "reviewed.jsonl"
            reviewed_path.write_text(
                "\n".join([
                    json.dumps({
                        "candidate_id": "cand_1",
                        "action": "approved",
                        "edited_content": "用户偏好中文回答。",
                        "reviewer": "user",
                        "candidate": {
                            "id": "cand_1",
                            "type": "preference",
                            "scope": "user",
                            "content": "用户偏好中文回答。",
                            "evidence": [{"event_id": "event_1"}],
                        },
                    }, ensure_ascii=False),
                    "{not json",
                ]) + "\n",
                encoding="utf-8",
            )
            output_dir = root / "memory"

            with patch("sys.stderr", new_callable=lambda: __import__('io').StringIO()) as stderr:
                exit_code = main(["apply", "--reviewed", str(reviewed_path), "--output-dir", str(output_dir), "--reviewer", "user"])

            self.assertEqual(exit_code, 2)
            self.assertIn("invalid reviewed decisions JSON", stderr.getvalue())
            self.assertFalse((output_dir / "MEMORY.md").exists())

    def test_apply_cli_rejects_approved_decision_without_memory_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviewed_path = root / "reviewed.jsonl"
            reviewed_path.write_text(
                json.dumps({"candidate_id": "cand_1", "status": "approved", "memory_updates": []}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            output_dir = root / "memory"

            result = subprocess.run(
                ["uv", "run", "dream-memory", "apply", "--reviewed", str(reviewed_path), "--output-dir", str(output_dir), "--reviewer", "user"],
                cwd=Path.cwd(),
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("approved decisions have no memory updates", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse((output_dir / "MEMORY.md").exists())

    def test_apply_cli_rejects_approved_decision_without_applicable_memory_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviewed_path = root / "reviewed.jsonl"
            reviewed_path.write_text(
                json.dumps(
                    {"candidate_id": "cand_1", "status": "approved", "memory_updates": [{"summary": "缺少 id 的更新"}]},
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            output_dir = root / "memory"

            result = subprocess.run(
                ["uv", "run", "dream-memory", "apply", "--reviewed", str(reviewed_path), "--output-dir", str(output_dir), "--reviewer", "user"],
                cwd=Path.cwd(),
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("incomplete memory update for cand_1", result.stderr)
            self.assertIn("missing", result.stderr)
            self.assertIn("id", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse((output_dir / "MEMORY.md").exists())

    def test_apply_cli_rejects_incomplete_approved_memory_update_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviewed_path = root / "reviewed.jsonl"
            reviewed_path.write_text(
                json.dumps(
                    {"candidate_id": "cand_1", "status": "approved", "memory_updates": [{"id": "mem_1"}]},
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            output_dir = root / "memory"

            result = subprocess.run(
                ["uv", "run", "dream-memory", "apply", "--reviewed", str(reviewed_path), "--output-dir", str(output_dir), "--reviewer", "user"],
                cwd=Path.cwd(),
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("incomplete memory update for cand_1", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse((output_dir / "MEMORY.md").exists())

    def test_apply_cli_rejects_sensitive_approved_memory_update_without_writing_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reviewed_path = root / "reviewed.jsonl"
            reviewed_path.write_text(json.dumps({
                "candidate_id": "cand_1",
                "status": "approved",
                "memory_updates": [{
                    "id": "mem_1",
                    "scope": "user",
                    "memory_type": "preference",
                    "summary": "OPENAI_API_KEY=sk-test-secret",
                    "evidence_refs": ["event_1"],
                    "approved_by": "user",
                    "approved_at": "2026-07-05T00:00:00Z",
                    "status": "active",
                    "retrieval_hints": [],
                }],
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            output_dir = root / "memory"

            result = subprocess.run(
                ["uv", "run", "dream-memory", "apply", "--reviewed", str(reviewed_path), "--output-dir", str(output_dir), "--reviewer", "user"],
                cwd=Path.cwd(),
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("sensitive memory update", result.stderr)
            self.assertNotIn("sk-test-secret", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse((output_dir / "MEMORY.md").exists())

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

    def test_pipeline_cli_rejects_malformed_memory_cards_without_partial_review_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events_path = root / "events.jsonl"
            memory_cards = root / "memory_cards.jsonl"
            output_dir = root / "memory"
            events_path.write_text(json.dumps({
                "event_id": "event_1",
                "source": "codex",
                "session_id": "s1",
                "role": "user",
                "event_type": "history_prompt",
                "content": "这个项目需要人工审核后才能写正式记忆",
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            memory_cards.write_text(
                json.dumps({"id": "mem_1", "scope": "user", "memory_type": "preference", "summary": "已有记忆。", "status": "active"}, ensure_ascii=False)
                + "\n{not json\n",
                encoding="utf-8",
            )

            with patch("sys.stderr", new_callable=lambda: io.StringIO()) as stderr:
                exit_code = main([
                    "pipeline",
                    "--input", str(events_path),
                    "--project", str(root),
                    "--output-dir", str(output_dir),
                    "--memory-cards", str(memory_cards),
                    "--mode", "rules",
                ])

            self.assertEqual(exit_code, 2)
            self.assertIn("invalid memory cards JSON", stderr.getvalue())
            self.assertFalse((output_dir / "review_queue.jsonl").exists())

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

    def test_run_id_commands_reject_invalid_ids_without_tracebacks_in_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = str(Path(tmp) / "memory")
            config_path = str(Path(tmp) / "config.json")
            commands = [
                ["uv", "run", "dream-memory", "--config", config_path, "status", "--run-id", "bad.run", "--output-dir", memory_dir],
                ["uv", "run", "dream-memory", "--config", config_path, "trace", "--run-id", "bad.run", "--output-dir", memory_dir],
                ["uv", "run", "dream-memory", "--config", config_path, "review-summary", "--run-id", "bad.run", "--output-dir", memory_dir],
                ["uv", "run", "dream-memory", "--config", config_path, "auto-review", "--run-id", "bad.run", "--output-dir", memory_dir],
                ["uv", "run", "dream-memory", "--config", config_path, "resume", "--run-id", "bad.run", "--output-dir", memory_dir],
            ]

            for command in commands:
                result = subprocess.run(command, cwd=Path.cwd(), text=True, capture_output=True, check=False)
                self.assertEqual(result.returncode, 2, command)
                self.assertIn("invalid run_id", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_invalid_config_json_returns_clean_error_in_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "bad-config.json"
            config_path.write_text("{not json", encoding="utf-8")

            result = subprocess.run(["uv", "run", "dream-memory", "--config", str(config_path), "status"], cwd=Path.cwd(), text=True, capture_output=True, check=False)

            self.assertEqual(result.returncode, 2)
            self.assertIn("Invalid memory config JSON", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_invalid_config_json_redacts_sensitive_config_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config-sk-test-secret.json"
            config_path.write_text("{not json", encoding="utf-8")

            with patch("sys.stderr", new_callable=lambda: io.StringIO()) as stderr:
                exit_code = main(["--config", str(config_path), "status"])

            error_text = stderr.getvalue()
            self.assertEqual(exit_code, 2)
            self.assertIn("Invalid memory config JSON", error_text)
            self.assertNotIn("sk-test-secret", error_text)
            self.assertIn("<redacted>", error_text)

    def test_trace_cli_missing_run_returns_clean_error_in_subprocess(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            run_id = "run_20260711T000000Z_abcd1234"

            result = subprocess.run(
                ["uv", "run", "dream-memory", "trace", "--run-id", run_id, "--output-dir", str(memory_dir)],
                cwd=Path.cwd(),
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("Run state not found", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertEqual(result.stdout, "")

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

    def test_eval_cli_redacts_sensitive_output_path_from_stdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labeled = root / "labeled.jsonl"
            output_path = root / "eval-output-sk-test-secret.json"
            labeled.write_text(json.dumps({"id": "row_1", "events": [], "expected": []}, ensure_ascii=False) + "\n", encoding="utf-8")

            with patch("sys.stdout", new_callable=lambda: io.StringIO()) as stdout:
                exit_code = main(["eval", "--input", str(labeled), "--mode", "rules", "--output", str(output_path)])

            output_text = stdout.getvalue()
            self.assertEqual(exit_code, 0)
            self.assertTrue(output_path.exists())
            self.assertNotIn("sk-test-secret", output_text)
            self.assertIn("<redacted>", output_text)


    def test_eval_cli_rejects_empty_labeled_input_without_success_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labeled = root / "empty-labeled.jsonl"
            labeled.write_text("\n\n", encoding="utf-8")

            result = subprocess.run(
                ["uv", "run", "dream-memory", "eval", "--input", str(labeled)],
                cwd=Path.cwd(),
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("eval input has no valid rows", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_eval_cli_rejects_malformed_labeled_input_without_success_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labeled = root / "bad-labeled.jsonl"
            labeled.write_text(json.dumps({"id": "row_1", "events": [], "expected": []}, ensure_ascii=False) + "\nnot-json\n", encoding="utf-8")

            result = subprocess.run(
                ["uv", "run", "dream-memory", "eval", "--input", str(labeled)],
                cwd=Path.cwd(),
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("invalid eval input JSON", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_eval_cli_rejects_invalid_config_mode_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labeled = root / "labeled.jsonl"
            config_path = root / "config.json"
            labeled.write_text(json.dumps({"id": "row_1", "events": [], "expected": []}, ensure_ascii=False) + "\n", encoding="utf-8")
            config_path.write_text(json.dumps({"eval_input": str(labeled), "eval_mode": "bad-mode"}, ensure_ascii=False), encoding="utf-8")

            result = subprocess.run(
                ["uv", "run", "dream-memory", "--config", str(config_path), "eval"],
                cwd=Path.cwd(),
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("unsupported eval mode", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

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

    def test_sync_auto_does_not_complete_when_auto_review_only_needs_more_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            codex = root / "codex-home"
            claude = root / "claude-home"
            project = root / "project"
            memory_dir = root / "memory"
            imports_dir = root / "imports"
            config_path = root / "config.json"
            codex.mkdir()
            claude.mkdir()
            project.mkdir()
            (project / "AGENTS.md").write_text("正式记忆必须经过人工审核后才能写入。", encoding="utf-8")
            config_path.write_text(json.dumps({
                "codex_home": str(codex),
                "claude_home": str(claude),
                "claude_state": str(root / "claude.json"),
                "output_dir": str(memory_dir),
                "imports_output_dir": str(imports_dir),
            }, ensure_ascii=False), encoding="utf-8")
            run_state = {"run_id": "run_20260711T000000Z_abcdef12", "run_dir": str(memory_dir / "runs" / "run_20260711T000000Z_abcdef12")}

            with (
                patch("dream_memory.memory_cli._run_dream_to_review", return_value=({"candidate_count": 1}, run_state)),
                patch("dream_memory.memory_cli._auto_review_run", return_value={"approved": 0, "needs_more_evidence": 1, "decision_count": 1, "skipped": 0}),
                patch("dream_memory.memory_cli._resume_run") as resume,
            ):
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    exit_code = main(["--config", str(config_path), "sync", "--project", str(project), "--auto"])

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["status"], "waiting_review")
            self.assertIn("manual review required", payload["message"])
            resume.assert_not_called()

    def test_sync_reports_import_warnings_in_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            codex = root / "codex-home"
            claude = root / "claude-home"
            project = root / "project"
            memory_dir = root / "memory"
            imports_dir = root / "imports"
            config_path = root / "config.json"
            codex.mkdir()
            claude.mkdir()
            project.mkdir()
            (codex / "history.jsonl").write_text(
                json.dumps({"session_id": "s1", "ts": 1, "text": "用户偏好中文回答。"}, ensure_ascii=False)
                + "\nnot-json\n[]\n",
                encoding="utf-8",
            )
            config_path.write_text(json.dumps({
                "codex_home": str(codex),
                "claude_home": str(claude),
                "claude_state": str(root / "claude.json"),
                "output_dir": str(memory_dir),
                "imports_output_dir": str(imports_dir),
                "mode": "rules",
            }, ensure_ascii=False), encoding="utf-8")

            with patch("sys.stdout", new_callable=lambda: io.StringIO()) as stdout:
                exit_code = main(["--config", str(config_path), "sync", "--project", str(project), "--dry-run"])

            payload = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertGreaterEqual(payload["event_count"], 1)
            self.assertEqual(payload["warning_count"], 2)
            self.assertEqual(payload["warnings"][0]["line"], 2)
            self.assertIn("invalid JSON", payload["warnings"][0]["error"])
            self.assertEqual(payload["warnings"][1]["line"], 3)
            self.assertIn("expected JSON object", payload["warnings"][1]["error"])

    def test_sync_rejects_no_events_without_success_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            codex = root / "codex-home"
            claude = root / "claude-home"
            project = root / "project"
            memory_dir = root / "memory"
            imports_dir = root / "imports"
            config_path = root / "config.json"
            codex.mkdir()
            claude.mkdir()
            project.mkdir()
            config_path.write_text(json.dumps({
                "codex_home": str(codex),
                "claude_home": str(claude),
                "claude_state": str(root / "claude.json"),
                "output_dir": str(memory_dir),
                "imports_output_dir": str(imports_dir),
            }, ensure_ascii=False), encoding="utf-8")

            result = subprocess.run(
                ["uv", "run", "dream-memory", "--config", str(config_path), "sync", "--project", str(project)],
                cwd=Path.cwd(),
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("no events found; nothing to sync", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def test_init_config_cli_writes_default_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"

            exit_code = main(["init-config", "--output", str(config_path)])

            self.assertEqual(exit_code, 0)
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["models"]["primary"]["provider"], "anthropic")
            self.assertEqual(payload["models"]["primary"]["model"], "claude-sonnet-4-6")
            self.assertEqual(payload["model_policy"]["fallback_chain"], ["primary"])

    def test_init_config_cli_rejects_unwritable_output_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.mkdir()

            with patch("sys.stderr", new_callable=lambda: io.StringIO()) as stderr:
                exit_code = main(["init-config", "--output", str(config_path)])

            self.assertEqual(exit_code, 2)
            self.assertIn("config path is not writable", stderr.getvalue())
            self.assertNotIn("Traceback", stderr.getvalue())
            self.assertTrue(config_path.is_dir())

    def test_run_and_pipeline_missing_input_return_clean_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(json.dumps({"default_input": None}, ensure_ascii=False), encoding="utf-8")

            self.assertEqual(main(["--config", str(config_path), "run"]), 2)
            self.assertEqual(main(["--config", str(config_path), "pipeline"]), 2)

    def test_cli_missing_input_errors_redact_sensitive_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = [
                ("eval", ["eval", "--input", str(root / "eval-sk-test-secret.jsonl")], "eval input not found"),
                ("extract-facts", ["extract-facts", "--input", str(root / "extract-sk-test-secret.jsonl")], "extract-facts input not found"),
                ("dream", ["dream", "--input", str(root / "dream-sk-test-secret.jsonl")], "dream input not found"),
                ("run", ["run", "--input", str(root / "run-sk-test-secret.jsonl")], "run input not found"),
                ("pipeline", ["pipeline", "--input", str(root / "pipeline-sk-test-secret.jsonl")], "pipeline input not found"),
                ("review", ["review", "--candidates", str(root / "candidates-sk-test-secret.jsonl")], "candidates not found"),
            ]

            for label, argv, expected in cases:
                with self.subTest(label=label):
                    with patch("sys.stderr", new_callable=lambda: io.StringIO()) as stderr:
                        exit_code = main(argv)

                    error_text = stderr.getvalue()
                    self.assertEqual(exit_code, 2)
                    self.assertIn(expected, error_text)
                    self.assertNotIn("sk-test-secret", error_text)
                    self.assertIn("<redacted>", error_text)

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

    def test_dream_run_and_pipeline_reject_empty_event_inputs_without_tracebacks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty_events = root / "empty.jsonl"
            empty_events.write_text("", encoding="utf-8")
            output_dir = root / "memory"
            commands = [
                (["uv", "run", "dream-memory", "dream", "--input", str(empty_events), "--output-dir", str(output_dir / "dream")], "dream input has no valid events"),
                (["uv", "run", "dream-memory", "run", "--input", str(empty_events), "--output-dir", str(output_dir / "run"), "--mode", "rules"], "run input has no valid events"),
                (["uv", "run", "dream-memory", "pipeline", "--input", str(empty_events), "--output-dir", str(output_dir / "pipeline"), "--mode", "rules"], "pipeline input has no valid events"),
            ]

            for command, expected in commands:
                result = subprocess.run(command, cwd=Path.cwd(), text=True, capture_output=True, check=False)
                self.assertEqual(result.returncode, 2, command)
                self.assertIn(expected, result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertFalse(Path(command[command.index("--output-dir") + 1]).exists())

    def test_extract_facts_rejects_empty_event_input_without_writing_facts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            empty_events = root / "empty.jsonl"
            empty_events.write_text("\n\n", encoding="utf-8")
            output_dir = root / "facts-output"

            result = subprocess.run(
                ["uv", "run", "dream-memory", "extract-facts", "--input", str(empty_events), "--output-dir", str(output_dir)],
                cwd=Path.cwd(),
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("extract-facts input has no valid events", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse((output_dir / "facts.jsonl").exists())

    def test_dream_run_pipeline_and_extract_facts_reject_malformed_event_inputs_without_partial_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad_events = root / "bad.jsonl"
            bad_events.write_text(json.dumps({"event_id": "event_1", "source": "codex", "role": "user", "content": "用户偏好中文回答。"}, ensure_ascii=False) + "\nnot-json\n", encoding="utf-8")
            output_dir = root / "memory"
            commands = [
                (["uv", "run", "dream-memory", "dream", "--input", str(bad_events), "--output-dir", str(output_dir / "dream")], "invalid dream input JSON"),
                (["uv", "run", "dream-memory", "run", "--input", str(bad_events), "--output-dir", str(output_dir / "run"), "--mode", "rules"], "invalid run input JSON"),
                (["uv", "run", "dream-memory", "pipeline", "--input", str(bad_events), "--output-dir", str(output_dir / "pipeline"), "--mode", "rules"], "invalid pipeline input JSON"),
                (["uv", "run", "dream-memory", "extract-facts", "--input", str(bad_events), "--output-dir", str(output_dir / "facts")], "invalid extract-facts input JSON"),
            ]

            for command, expected in commands:
                result = subprocess.run(command, cwd=Path.cwd(), text=True, capture_output=True, check=False)
                self.assertEqual(result.returncode, 2, command)
                self.assertIn(expected, result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertFalse(Path(command[command.index("--output-dir") + 1]).exists())

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

    def test_run_cli_rejects_malformed_generated_candidates_without_review_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events_path = root / "events.jsonl"
            memory_dir = root / "memory"
            config_path = root / "config.json"
            events_path.write_text(json.dumps({"event_id": "event_1", "source": "codex", "role": "user", "event_type": "history_prompt", "project": str(root), "content": "这个项目需要人工审核后才能写正式记忆"}, ensure_ascii=False) + "\n", encoding="utf-8")
            config_path.write_text(json.dumps({"invoke_model": False, "output_dir": str(memory_dir)}, ensure_ascii=False), encoding="utf-8")

            def fake_dream_from_events(events, *, project, output_dir, apply, **kwargs):
                output = Path(output_dir)
                output.mkdir(parents=True, exist_ok=True)
                candidates_path = output / "candidates.jsonl"
                candidates_path.write_text(
                    json.dumps({"id": "cand_1", "scope": "project", "project": project, "type": "workflow", "content": "正式记忆必须经过人工审核。"}, ensure_ascii=False) + "\n"
                    + "{bad json\n",
                    encoding="utf-8",
                )
                preview_path = output / "memory_preview.md"
                preview_path.write_text("", encoding="utf-8")
                return DreamResult(
                    event_count=len(events),
                    candidate_count=1,
                    promoted_count=0,
                    review_count=1,
                    rejected_count=0,
                    output_dir=str(output),
                    candidates_path=str(candidates_path),
                    dreams_path=str(output / "dreams.jsonl"),
                    memory_preview_path=str(preview_path),
                    memory_path=None,
                    applied=False,
                )

            with patch("dream_memory.memory_cli.dream_from_events", side_effect=fake_dream_from_events):
                with patch("sys.stderr", new_callable=lambda: io.StringIO()) as stderr:
                    exit_code = main(["--config", str(config_path), "run", "--input", str(events_path), "--project", str(root), "--mode", "rules"])

            self.assertEqual(exit_code, 2)
            self.assertIn("invalid generated candidates JSON", stderr.getvalue())
            state_path = next((memory_dir / "runs").glob("*/state.json"))
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "failed")
            self.assertIn("invalid generated candidates JSON", state["error"])
            self.assertFalse((Path(state["run_dir"]) / "review_queue.jsonl").exists())

    def test_run_cli_marks_persistent_run_failed_when_extraction_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events_path = root / "events.jsonl"
            memory_dir = root / "memory"
            config_path = root / "config.json"
            events_path.write_text(json.dumps({"event_id": "event_1", "source": "codex", "role": "user", "event_type": "history_prompt", "project": str(root), "content": "记住这个偏好"}, ensure_ascii=False) + "\n", encoding="utf-8")
            config_path.write_text(json.dumps({"invoke_model": True, "output_dir": str(memory_dir)}, ensure_ascii=False), encoding="utf-8")

            with patch("dream_memory.memory_cli.agent_extract_memory_candidates", side_effect=RuntimeError("provider 403 api_key=sk-test-secret Bearer abc123")):
                with self.assertRaises(RuntimeError):
                    main(["--config", str(config_path), "run", "--input", str(events_path), "--project", str(root)])

            state_path = next((memory_dir / "runs").glob("*/state.json"))
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["phase"], "failed")
            self.assertIn("provider 403", state["error"])
            self.assertNotIn("sk-test-secret", state["error"])
            self.assertNotIn("Bearer abc123", state["error"])
            self.assertIn("<redacted>", state["error"])
            trace = (Path(state["run_dir"]) / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("run_failed", trace)
            self.assertIn("RuntimeError", trace)
            self.assertNotIn("sk-test-secret", trace)
            self.assertNotIn("Bearer abc123", trace)
            self.assertIn("<redacted>", trace)

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

    def test_status_cli_redacts_sensitive_legacy_state_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"output_dir": str(memory_dir)}, ensure_ascii=False), encoding="utf-8")
            state = create_run_state(memory_dir=memory_dir, project=str(root), input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            state_path = Path(state["run_dir"]) / "state.json"
            state["error"] = "provider failed api_key=sk-test-secret"
            state["next_actions"] = ["token 在 config.yaml 配置中"]
            state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")

            with patch("sys.stdout", new_callable=lambda: __import__('io').StringIO()) as stdout:
                exit_code = main(["--config", str(config_path), "status", "--run-id", state["run_id"]])

            self.assertEqual(exit_code, 0)
            output = stdout.getvalue()
            self.assertNotIn("sk-test-secret", output)
            self.assertNotIn("config.yaml", output)
            self.assertIn("<redacted>", output)

    def test_status_cli_redacts_sensitive_path_from_missing_run_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory-sk-test-secret"
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"output_dir": str(memory_dir)}, ensure_ascii=False), encoding="utf-8")

            with patch("sys.stderr", new_callable=lambda: __import__('io').StringIO()) as stderr:
                exit_code = main(["--config", str(config_path), "status", "--run-id", "run_20260101T000000Z_missing"])

            error_text = stderr.getvalue()
            self.assertEqual(exit_code, 2)
            self.assertIn("Run state not found", error_text)
            self.assertNotIn("sk-test-secret", error_text)
            self.assertIn("<redacted>", error_text)

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

    def test_resume_cli_redacts_sensitive_legacy_state_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"output_dir": str(memory_dir)}, ensure_ascii=False), encoding="utf-8")
            state = create_run_state(memory_dir=memory_dir, project=str(root), input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            reviewed_path = Path(state["run_dir"]) / "reviewed.jsonl"
            reviewed_path.write_text(
                json.dumps({"candidate_id": "cand_1", "status": "rejected"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            state_path = Path(state["run_dir"]) / "state.json"
            state["diagnostics"] = {
                "provider_error": "failed with OPENAI_API_KEY=sk-test-secret",
                "next_hint": "token 在 config.yaml 配置中",
            }
            state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")

            with patch("sys.stdout", new_callable=lambda: __import__('io').StringIO()) as stdout:
                exit_code = main(["--config", str(config_path), "resume", "--run-id", state["run_id"]])

            output = stdout.getvalue()
            self.assertEqual(exit_code, 0)
            self.assertNotIn("sk-test-secret", output)
            self.assertNotIn("config.yaml", output)
            self.assertIn("<redacted>", output)

    def test_resume_cli_requires_reviewed_decisions_before_completing_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"output_dir": str(memory_dir)}, ensure_ascii=False), encoding="utf-8")
            state = create_run_state(memory_dir=memory_dir, project=str(root), input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            update_run_state(state, status="waiting_review", phase="review")

            with patch("sys.stderr", new_callable=lambda: __import__('io').StringIO()) as stderr:
                exit_code = main(["--config", str(config_path), "resume", "--run-id", state["run_id"]])

            self.assertEqual(exit_code, 2)
            self.assertIn("reviewed decisions not found", stderr.getvalue())
            updated = json.loads((Path(state["run_dir"]) / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(updated["status"], "waiting_review")
            self.assertFalse((memory_dir / "MEMORY.md").exists())

    def test_resume_cli_rejects_approved_decision_without_memory_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"output_dir": str(memory_dir)}, ensure_ascii=False), encoding="utf-8")
            state = create_run_state(memory_dir=memory_dir, project=str(root), input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            update_run_state(state, status="waiting_review", phase="review")
            reviewed_path = Path(state["run_dir"]) / "reviewed.jsonl"
            reviewed_path.write_text(
                json.dumps({"candidate_id": "cand_1", "status": "approved", "memory_updates": []}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            with patch("sys.stderr", new_callable=lambda: __import__('io').StringIO()) as stderr:
                exit_code = main(["--config", str(config_path), "resume", "--run-id", state["run_id"]])

            self.assertEqual(exit_code, 2)
            self.assertIn("approved decisions have no memory updates", stderr.getvalue())
            updated = json.loads((Path(state["run_dir"]) / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(updated["status"], "waiting_review")
            self.assertFalse((memory_dir / "MEMORY.md").exists())

    def test_resume_cli_rejects_incomplete_approved_memory_update_without_completing_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"output_dir": str(memory_dir)}, ensure_ascii=False), encoding="utf-8")
            state = create_run_state(memory_dir=memory_dir, project=str(root), input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            update_run_state(state, status="waiting_review", phase="review")
            reviewed_path = Path(state["run_dir"]) / "reviewed.jsonl"
            reviewed_path.write_text(
                json.dumps({"candidate_id": "cand_1", "status": "approved", "memory_updates": [{"id": "mem_1"}]}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            with patch("sys.stderr", new_callable=lambda: __import__('io').StringIO()) as stderr:
                exit_code = main(["--config", str(config_path), "resume", "--run-id", state["run_id"]])

            self.assertEqual(exit_code, 2)
            self.assertIn("incomplete memory update for cand_1", stderr.getvalue())
            updated = json.loads((Path(state["run_dir"]) / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(updated["status"], "waiting_review")
            self.assertFalse((memory_dir / "MEMORY.md").exists())

    def test_resume_cli_rejects_unwritable_memory_markdown_without_partial_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"output_dir": str(memory_dir)}, ensure_ascii=False), encoding="utf-8")
            state = create_run_state(memory_dir=memory_dir, project=str(root), input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            update_run_state(state, status="waiting_review", phase="review")
            reviewed_path = Path(state["run_dir"]) / "reviewed.jsonl"
            reviewed_path.write_text(json.dumps({
                "candidate_id": "cand_1",
                "status": "approved",
                "memory_updates": [{
                    "id": "mem_1",
                    "scope": "user",
                    "memory_type": "preference",
                    "summary": "用户偏好中文回答。",
                    "evidence_refs": ["event_1"],
                    "approved_by": "user",
                    "approved_at": "2026-07-05T00:00:00Z",
                    "status": "active",
                    "retrieval_hints": [],
                }],
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            (memory_dir / "MEMORY.md").mkdir()

            result = subprocess.run(
                ["uv", "run", "dream-memory", "--config", str(config_path), "resume", "--run-id", state["run_id"]],
                cwd=Path.cwd(),
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("memory output path is not writable", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            updated = json.loads((Path(state["run_dir"]) / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(updated["status"], "waiting_review")
            self.assertFalse((memory_dir / "memory_cards.jsonl").exists())
            self.assertFalse((memory_dir / "review_decisions.jsonl").exists())

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

    def test_trace_cli_redacts_sensitive_legacy_trace_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"output_dir": str(memory_dir)}, ensure_ascii=False), encoding="utf-8")
            state = create_run_state(memory_dir=memory_dir, project=str(root), input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            trace_path = Path(state["run_dir"]) / "trace.jsonl"
            trace_path.write_text(json.dumps({
                "run_id": state["run_id"],
                "event_type": "legacy",
                "payload": {
                    "error": "provider failed api_key=sk-test-secret",
                    "note": "token 在 config.yaml 配置中",
                },
            }, ensure_ascii=False) + "\n", encoding="utf-8")

            with patch("sys.stdout", new_callable=lambda: __import__('io').StringIO()) as stdout:
                exit_code = main(["--config", str(config_path), "trace", "--run-id", state["run_id"]])

            self.assertEqual(exit_code, 0)
            output = stdout.getvalue()
            self.assertNotIn("sk-test-secret", output)
            self.assertNotIn("config.yaml", output)
            self.assertIn("<redacted>", output)

    def test_trace_cli_rejects_invalid_candidate_id_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"output_dir": str(memory_dir)}, ensure_ascii=False), encoding="utf-8")
            state = create_run_state(memory_dir=memory_dir, project=str(root), input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)

            with patch("sys.stderr", new_callable=lambda: __import__('io').StringIO()) as stderr:
                exit_code = main(["--config", str(config_path), "trace", "--run-id", state["run_id"], "--candidate-id", "../state"])

            self.assertEqual(exit_code, 2)
            self.assertIn("invalid candidate_id", stderr.getvalue())

    def test_trace_cli_rejects_malformed_trace_without_silent_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"output_dir": str(memory_dir)}, ensure_ascii=False), encoding="utf-8")
            state = create_run_state(memory_dir=memory_dir, project=str(root), input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            trace_path = Path(state["run_dir"]) / "trace.jsonl"
            trace_path.write_text(
                json.dumps({"run_id": state["run_id"], "event_type": "run_created", "payload": {}}, ensure_ascii=False)
                + "\nnot-json\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                ["uv", "run", "dream-memory", "--config", str(config_path), "trace", "--run-id", state["run_id"]],
                cwd=Path.cwd(),
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("invalid trace JSON", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertEqual(result.stdout, "")

    def test_trace_cli_redacts_sensitive_path_from_malformed_trace_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory-sk-test-secret"
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"output_dir": str(memory_dir)}, ensure_ascii=False), encoding="utf-8")
            state = create_run_state(memory_dir=memory_dir, project=str(root), input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            trace_path = Path(state["run_dir"]) / "trace.jsonl"
            trace_path.write_text("{not json\n", encoding="utf-8")

            with patch("sys.stderr", new_callable=lambda: __import__('io').StringIO()) as stderr:
                exit_code = main(["--config", str(config_path), "trace", "--run-id", state["run_id"]])

            error_text = stderr.getvalue()
            self.assertEqual(exit_code, 2)
            self.assertIn("invalid trace JSON", error_text)
            self.assertNotIn("sk-test-secret", error_text)
            self.assertIn("<redacted>", error_text)

    def test_trace_cli_rejects_non_object_trace_payload_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"output_dir": str(memory_dir)}, ensure_ascii=False), encoding="utf-8")
            state = create_run_state(memory_dir=memory_dir, project=str(root), input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            trace_path = Path(state["run_dir"]) / "trace.jsonl"
            trace_path.write_text(
                json.dumps({"run_id": state["run_id"], "event_type": "candidate_ready", "payload": "bad-payload"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                ["uv", "run", "dream-memory", "--config", str(config_path), "trace", "--run-id", state["run_id"], "--candidate-id", "cand_1"],
                cwd=Path.cwd(),
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("invalid trace payload", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertEqual(result.stdout, "")

    def test_trace_cli_rejects_invalid_candidate_trace_shape_without_silent_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"output_dir": str(memory_dir)}, ensure_ascii=False), encoding="utf-8")
            state = create_run_state(memory_dir=memory_dir, project=str(root), input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            candidate_path = memory_dir / "runs" / state["run_id"] / "candidates" / "cand_1.json"
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            candidate_path.write_text("[]\n", encoding="utf-8")

            result = subprocess.run(
                ["uv", "run", "dream-memory", "--config", str(config_path), "trace", "--run-id", state["run_id"], "--candidate-id", "cand_1"],
                cwd=Path.cwd(),
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("candidate trace must be a JSON object", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertEqual(result.stdout, "")

    def test_trace_cli_rejects_mismatched_candidate_trace_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            config_path = root / "config.json"
            config_path.write_text(json.dumps({"output_dir": str(memory_dir)}, ensure_ascii=False), encoding="utf-8")
            state = create_run_state(memory_dir=memory_dir, project=str(root), input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            candidate_path = memory_dir / "runs" / state["run_id"] / "candidates" / "cand_1.json"
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            candidate_path.write_text(json.dumps({"candidate_id": "other"}, ensure_ascii=False) + "\n", encoding="utf-8")

            result = subprocess.run(
                ["uv", "run", "dream-memory", "--config", str(config_path), "trace", "--run-id", state["run_id"], "--candidate-id", "cand_1"],
                cwd=Path.cwd(),
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("candidate trace id mismatch", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertEqual(result.stdout, "")


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
            self.assertEqual(payload["rows"], 13)
            self.assertEqual(payload["precision"], 1.0)
            self.assertEqual(payload["recall"], 1.0)
            self.assertEqual(payload["f1"], 1.0)

    def test_init_cli_labeled_eval_example_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            main(["init", "--path", str(root)])
            output = root / "eval.json"

            exit_code = main(["eval", "--input", str(root / "examples" / "labeled-events.jsonl"), "--project", "/tmp/project", "--output", str(output)])

            self.assertEqual(exit_code, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["precision"], 1.0)
            self.assertEqual(payload["recall"], 1.0)

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

    def test_check_provider_all_preserves_profiles_with_colliding_redacted_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            config_path.write_text(json.dumps({
                "models": {
                    "profile-sk-test-secret-a": {"provider": "openai", "model": "gpt-a", "api_key": "key-a"},
                    "profile-sk-test-secret-b": {"provider": "anthropic", "model": "claude-b", "api_key": "key-b"},
                },
                "model_policy": {
                    "default_profile": "profile-sk-test-secret-a",
                    "fallback_chain": ["profile-sk-test-secret-a", "profile-sk-test-secret-b"],
                },
            }), encoding="utf-8")

            with patch("sys.stdout", new_callable=lambda: io.StringIO()) as stdout:
                exit_code = main(["--config", str(config_path), "check-provider", "--all"])

            output_text = stdout.getvalue()
            payload = json.loads(output_text)
            self.assertEqual(exit_code, 0)
            self.assertNotIn("sk-test-secret", output_text)
            self.assertIn("<redacted>", output_text)
            self.assertEqual(len(payload["profiles"]), 2)
            self.assertIn("profile-<redacted>", payload["profiles"])
            self.assertIn("profile-<redacted>-2", payload["profiles"])
            self.assertTrue(payload["profiles"]["profile-<redacted>"]["ok"])
            self.assertTrue(payload["profiles"]["profile-<redacted>-2"]["ok"])

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

    def test_dream_cli_provider_overrides_do_not_mutate_process_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = root / "events.jsonl"
            output_dir = root / "memory"
            events.write_text(json.dumps({"source": "codex", "role": "user", "content": "用户偏好中文回答", "project": str(root)}, ensure_ascii=False) + "\n", encoding="utf-8")

            with patch.dict("os.environ", {}, clear=True):
                exit_code = main([
                    "dream",
                    "--input", str(events),
                    "--project", str(root),
                    "--output-dir", str(output_dir),
                    "--mode", "ai",
                    "--dry-run",
                    "--api-key-env", "DREAM_MEMORY_TEST_KEY",
                    "--base-url", "http://localhost:3000",
                    "--timeout-seconds", "7",
                ])

                self.assertEqual(exit_code, 0)
                self.assertNotIn("DEEPAGENT_MEMORY_API_KEY_ENV", __import__("os").environ)
                self.assertNotIn("DEEPAGENT_MEMORY_BASE_URL", __import__("os").environ)
                self.assertNotIn("DEEPAGENT_MEMORY_TIMEOUT_SECONDS", __import__("os").environ)


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

    def test_dream_cli_uses_config_default_project_and_reports_prompt_filter_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = root / "events.jsonl"
            memory_dir = root / "memory"
            config_path = root / "config.json"
            events.write_text("".join([
                json.dumps({"event_id": "state", "source": "claude_code", "role": "system", "event_type": "project_state", "project": str(root), "content": "Claude Code project state for /tmp/project"}, ensure_ascii=False) + "\n",
                json.dumps({"event_id": "durable", "source": "codex", "role": "user", "event_type": "history_prompt", "project": str(root), "content": "正式记忆必须经过人工审核，只有审核通过的候选才允许写入长期记忆。"}, ensure_ascii=False) + "\n",
            ]), encoding="utf-8")
            config_path.write_text(json.dumps({
                "default_input": str(events),
                "default_project": str(root),
                "output_dir": str(memory_dir),
                "mode": "ai",
                "invoke_model": False,
            }, ensure_ascii=False), encoding="utf-8")

            with patch("sys.stdout", new_callable=lambda: io.StringIO()) as stdout:
                exit_code = main(["--config", str(config_path), "dream", "--dry-run"])

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["input_event_count"], 2)
            self.assertEqual(payload["prompt_event_count"], 1)
            self.assertEqual(payload["filtered_prompt_event_count"], 1)
            prompt = (memory_dir / "ai-prompt.md").read_text(encoding="utf-8")
            self.assertIn(f"Project filter: {root}", prompt)
            self.assertNotIn("Project filter: global", prompt)

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

    def test_run_cli_redacts_sensitive_values_from_ai_raw_response_artifact(self):
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
                "note": "provider echoed OPENAI_API_KEY=sk-test-secret",
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
                invoke.return_value = ModelRuntimeResult(text=raw, selected_profile="primary", attempts=[], elapsed_ms=1)
                exit_code = main(["--config", str(config_path), "run", "--input", str(events_path), "--project", str(root), "--invoke-model"])

            self.assertEqual(exit_code, 0)
            state = json.loads(next((memory_dir / "runs").glob("*/state.json")).read_text(encoding="utf-8"))
            raw_response = Path(state["artifacts"]["ai_raw_response_path"]).read_text(encoding="utf-8")
            candidates = Path(state["artifacts"]["candidates_path"]).read_text(encoding="utf-8")
            self.assertNotIn("sk-test-secret", raw_response)
            self.assertIn("<redacted>", raw_response)
            self.assertIn("用户偏好中文回答", candidates)

    def test_run_cli_redacts_sensitive_paths_from_output_streams(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events_path = root / "events.jsonl"
            memory_dir = root / "memory-sk-test-secret"
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

            with (
                patch("dream_memory.memory_agent.invoke_model_runtime") as invoke,
                patch("sys.stdout", new_callable=lambda: io.StringIO()) as stdout,
                patch("sys.stderr", new_callable=lambda: io.StringIO()) as stderr,
            ):
                invoke.return_value = ModelRuntimeResult(text=raw, selected_profile="primary", attempts=[], elapsed_ms=1)
                exit_code = main(["--config", str(config_path), "run", "--input", str(events_path), "--project", str(root), "--invoke-model"])

            output_text = stdout.getvalue()
            error_text = stderr.getvalue()
            self.assertEqual(exit_code, 0)
            self.assertNotIn("sk-test-secret", output_text)
            self.assertNotIn("sk-test-secret", error_text)
            self.assertIn("<redacted>", output_text)
            self.assertIn("<redacted>", error_text)

    def test_dream_cli_redacts_sensitive_values_from_ai_prompt_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events_path = root / "events.jsonl"
            output_dir = root / "memory"
            events_path.write_text("\n".join([
                json.dumps({
                    "event_id": "secret",
                    "source": "codex",
                    "role": "user",
                    "event_type": "history_prompt",
                    "project": str(root),
                    "content": "OPENAI_API_KEY=sk-test-secret",
                }, ensure_ascii=False),
                json.dumps({
                    "event_id": "durable",
                    "source": "codex",
                    "role": "user",
                    "event_type": "history_prompt",
                    "project": str(root),
                    "content": "用户偏好中文回答。",
                }, ensure_ascii=False),
            ]) + "\n", encoding="utf-8")

            exit_code = main([
                "dream",
                "--input", str(events_path),
                "--project", str(root),
                "--output-dir", str(output_dir),
                "--mode", "ai",
                "--dry-run",
            ])

            prompt = (output_dir / "ai-prompt.md").read_text(encoding="utf-8")
            self.assertEqual(exit_code, 0)
            self.assertNotIn("sk-test-secret", prompt)
            self.assertIn("用户偏好中文回答", prompt)

    def test_dream_cli_redacts_sensitive_paths_from_stdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events_path = root / "events.jsonl"
            output_dir = root / "memory-sk-test-secret"
            events_path.write_text(json.dumps({
                "event_id": "event_1",
                "source": "codex",
                "session_id": "s1",
                "role": "user",
                "event_type": "history_prompt",
                "project": str(root),
                "content": "用户偏好中文回答",
            }, ensure_ascii=False) + "\n", encoding="utf-8")
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

            with patch("dream_memory.memory_agent.invoke_model_runtime") as invoke, patch("sys.stdout", new_callable=lambda: io.StringIO()) as stdout:
                invoke.return_value = ModelRuntimeResult(text=raw, selected_profile="primary", attempts=[], elapsed_ms=1)
                exit_code = main([
                    "dream",
                    "--input", str(events_path),
                    "--project", str(root),
                    "--output-dir", str(output_dir),
                    "--mode", "ai",
                    "--invoke-model",
                ])

            output_text = stdout.getvalue()
            self.assertEqual(exit_code, 0)
            self.assertNotIn("sk-test-secret", output_text)
            self.assertIn("<redacted>", output_text)

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
