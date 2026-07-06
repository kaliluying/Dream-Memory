import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dream_memory.memory_cli import build_parser, main
from dream_memory.model_providers import ModelRuntimeResult


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

    def test_init_config_cli_writes_default_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"

            exit_code = main(["init-config", "--output", str(config_path)])

            self.assertEqual(exit_code, 0)
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["models"]["primary"]["provider"], "anthropic")
            self.assertEqual(payload["models"]["primary"]["model"], "claude-sonnet-4-6")
            self.assertEqual(payload["model_policy"]["fallback_chain"], ["primary"])

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

            with patch("dream_memory.memory_graph.invoke_model_runtime") as invoke:
                exit_code = main(["--config", str(config_path), "dream", "--input", str(events), "--project", str(root), "--dry-run"])

            self.assertEqual(exit_code, 0)
            invoke.assert_not_called()
            self.assertTrue((output_dir / "ai-prompt.md").exists())

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

            with patch("dream_memory.memory_graph.invoke_model_runtime") as invoke:
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
