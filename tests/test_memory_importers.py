import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from dream_memory.memory_importers import (
    ClaudeCodeImporter,
    CodexImporter,
    NormalizedSessionEvent,
    redact_sensitive,
    import_project_instruction_events,
    import_project_marker_events,
    write_events_jsonl,
)


class MemoryImporterTests(unittest.TestCase):
    def test_redact_sensitive_nested_values(self):
        data = {
            "api_key": "secret",
            "nested": {"token": "abc", "safe": "ok"},
            "items": [{"password": "pw", "text": "hello"}],
        }

        redacted = redact_sensitive(data)

        self.assertEqual(redacted["api_key"], "<redacted>")
        self.assertEqual(redacted["nested"]["token"], "<redacted>")
        self.assertEqual(redacted["nested"]["safe"], "ok")
        self.assertEqual(redacted["items"][0]["password"], "<redacted>")

    def test_codex_scan_detects_core_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            codex = home / ".codex"
            codex.mkdir()
            (codex / "history.jsonl").write_text('{"session_id":"s1","ts":1,"text":"你好"}\n', encoding="utf-8")
            (codex / "session_index.jsonl").write_text('{"id":"s1","thread_name":"测试","updated_at":"2026"}\n', encoding="utf-8")
            sqlite_dir = codex / "sqlite"
            sqlite_dir.mkdir()
            con = sqlite3.connect(sqlite_dir / "state_5.sqlite")
            con.execute("create table threads (id text, rollout_path text, cwd text, title text, first_user_message text, updated_at integer, model text)")
            con.execute("insert into threads values ('s1', '', '/tmp/project', '测试', '你好', 1, 'gpt')")
            con.commit(); con.close()

            scan = CodexImporter(codex_home=codex).scan()

            self.assertTrue(scan["history_found"])
            self.assertTrue(scan["session_index_found"])
            self.assertTrue(scan["state_db_found"])
            self.assertEqual(scan["thread_count"], 1)


    def test_codex_import_preserves_project_agents_instructions_without_environment_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            codex_home = Path(tmp) / ".codex"
            codex_home.mkdir()
            db_dir = codex_home / "sqlite"
            db_dir.mkdir()
            rollout = Path(tmp) / "rollout.jsonl"
            rollout.write_text(json.dumps({
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "# AGENTS.md instructions for /tmp/project\n\n<INSTRUCTIONS>\n如果后端使用的是 Python，则使用 uv 进行包管理\n前端使用 pnpm 进行管理\n</INSTRUCTIONS><environment_context>\n  <cwd>/tmp/project</cwd>\n</environment_context>"}],
                },
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            con = sqlite3.connect(db_dir / "state_5.sqlite")
            con.execute("create table threads (id text, rollout_path text, cwd text, title text, first_user_message text, updated_at text, model text)")
            con.execute("insert into threads values (?, ?, ?, ?, ?, ?, ?)", ("thread-1", str(rollout), "/tmp/project", "test", None, "1", "gpt"))
            con.commit()
            con.close()

            events = CodexImporter(codex_home=codex_home).import_events()

            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].event_type, "project_instruction")
            self.assertIn("uv 进行包管理", events[0].content)
            self.assertIn("pnpm 进行管理", events[0].content)
            self.assertNotIn("environment_context", events[0].content)


    def test_import_project_instruction_events_reads_local_agents_without_generated_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "AGENTS.md").write_text("""# AGENTS.md instructions for /tmp/project

<INSTRUCTIONS>
如果后端使用的是 Python，则使用 uv 进行包管理
前端使用 pnpm 进行管理
<!-- DREAM_MEMORY_START -->
这段是生成记忆，不应回灌。
<!-- DREAM_MEMORY_END -->
</INSTRUCTIONS>
""", encoding="utf-8")

            events = import_project_instruction_events([root])

            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].event_type, "project_instruction")
            self.assertEqual(events[0].project, str(root))
            self.assertIn("uv 进行包管理", events[0].content)
            self.assertIn("pnpm 进行管理", events[0].content)
            self.assertNotIn("生成记忆", events[0].content)


    def test_import_project_instruction_events_skips_generated_memory_only_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "AGENTS.md").write_text("""## Dream Memory Context

<!-- DREAM_MEMORY_START -->
## Relevant Memory

- generated memory only
<!-- DREAM_MEMORY_END -->
""", encoding="utf-8")

            events = import_project_instruction_events([root])

            self.assertEqual(events, [])

    def test_codex_project_instructions_strip_embedded_generated_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            codex_home = Path(tmp) / ".codex"
            codex_home.mkdir()
            db_dir = codex_home / "sqlite"
            db_dir.mkdir()
            rollout = Path(tmp) / "rollout.jsonl"
            rollout.write_text(json.dumps({
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "# AGENTS.md instructions for /tmp/project\n\n<INSTRUCTIONS>\n如果后端使用的是 Python，则使用 uv 进行包管理\n前端使用 pnpm 进行管理\n\n--- project-doc ---\n## Dream Memory Context\n<!-- DREAM_MEMORY_START -->\n- generated memory only\n<!-- DREAM_MEMORY_END -->\n</INSTRUCTIONS>"}],
                },
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            con = sqlite3.connect(db_dir / "state_5.sqlite")
            con.execute("create table threads (id text, rollout_path text, cwd text, title text, first_user_message text, updated_at text, model text)")
            con.execute("insert into threads values (?, ?, ?, ?, ?, ?, ?)", ("thread-1", str(rollout), "/tmp/project", "test", None, "1", "gpt"))
            con.commit()
            con.close()

            events = CodexImporter(codex_home=codex_home).import_events()

            self.assertEqual(len(events), 1)
            self.assertIn("uv 进行包管理", events[0].content)
            self.assertIn("pnpm 进行管理", events[0].content)
            self.assertNotIn("generated memory", events[0].content)
            self.assertNotIn("Dream Memory Context", events[0].content)


    def test_import_project_marker_events_detects_python_uv_and_frontend_pnpm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
            (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
            frontend = root / "frontend"
            frontend.mkdir()
            (frontend / "package.json").write_text('{"scripts":{"build":"vite build"}}', encoding="utf-8")
            (frontend / "pnpm-lock.yaml").write_text("lockfileVersion: '9.0'\n", encoding="utf-8")

            events = import_project_marker_events([root])

            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].event_type, "project_markers")
            self.assertIn("python_package_manager=uv", events[0].content)
            self.assertIn("frontend_package_manager=pnpm", events[0].content)
            self.assertIn("frontend", events[0].metadata["frontend_paths"])

    def test_import_project_marker_events_detects_unittest_and_python_app_markers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pyproject.toml").write_text("[project]\nname='demo'\ndependencies=['fastapi']\n", encoding="utf-8")
            (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
            tests = root / "tests"
            tests.mkdir()
            (tests / "test_demo.py").write_text("import unittest\n\nclass DemoTests(unittest.TestCase):\n    pass\n", encoding="utf-8")

            events = import_project_marker_events([root])

            self.assertEqual(len(events), 1)
            self.assertIn("python_test_runner=unittest", events[0].content)
            self.assertIn("python_framework=fastapi", events[0].content)
            self.assertIn("tests/test_demo.py", events[0].metadata["python_test_paths"])

    def test_import_project_marker_events_prefers_pytest_config_over_unittest_style_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pyproject.toml").write_text("[project]\nname='demo'\n\n[tool.pytest.ini_options]\ntestpaths=['tests']\n", encoding="utf-8")
            (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
            tests = root / "tests"
            tests.mkdir()
            (tests / "test_demo.py").write_text("import unittest\n\nclass DemoTests(unittest.TestCase):\n    pass\n", encoding="utf-8")

            events = import_project_marker_events([root])

            self.assertEqual(len(events), 1)
            self.assertIn("python_test_runner=pytest", events[0].content)
            self.assertNotIn("python_test_runner=unittest", events[0].content)

    def test_codex_import_reads_nested_rollout_response_items(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            codex = home / ".codex"
            codex.mkdir()
            rollout = codex / "sessions" / "rollout.jsonl"
            rollout.parent.mkdir()
            rollout.write_text(
                json.dumps({"timestamp": "2026", "type": "response_item", "payload": {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "ignore developer"}]}}, ensure_ascii=False) + "\n" +
                json.dumps({"timestamp": "2026", "type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "<environment_context>noise</environment_context>"}]}}, ensure_ascii=False) + "\n" +
                json.dumps({"timestamp": "2026", "type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "# Files mentioned by the user:\n\n## spec.md: /tmp/spec.md\n\n## My request for Codex:\n梳理这个项目"}]}}, ensure_ascii=False) + "\n" +
                json.dumps({"timestamp": "2026", "type": "response_item", "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "我会先读取真实仓库结构。"}]}}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            sqlite_dir = codex / "sqlite"
            sqlite_dir.mkdir()
            con = sqlite3.connect(sqlite_dir / "state_5.sqlite")
            con.execute("create table threads (id text, rollout_path text, cwd text, title text, first_user_message text, updated_at integer, model text)")
            con.execute("insert into threads values (?, ?, ?, ?, ?, ?, ?)", ("s1", str(rollout), "/tmp/project", "梳理", "梳理这个项目", 1, "gpt"))
            con.commit(); con.close()

            events = CodexImporter(codex_home=codex).import_events()

            rollout_messages = [event for event in events if event.event_type == "rollout_message"]
            self.assertTrue(any(event.role == "user" and event.content == "梳理这个项目" for event in rollout_messages))
            self.assertTrue(any(event.role == "assistant" and "真实仓库结构" in event.content for event in rollout_messages))
            self.assertFalse(any("ignore developer" in event.content for event in events))
            self.assertFalse(any("environment_context" in event.content for event in events))
            self.assertFalse(any("Files mentioned" in event.content for event in events))

    def test_codex_import_reads_history_and_rollout_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            codex = home / ".codex"
            codex.mkdir()
            rollout = codex / "sessions" / "rollout.jsonl"
            rollout.parent.mkdir()
            rollout.write_text(
                '{"role":"user","content":"实现登录"}\n'
                '{"role":"assistant","content":"已完成"}\n',
                encoding="utf-8",
            )
            (codex / "history.jsonl").write_text('{"session_id":"s1","ts":1,"text":"你好"}\n', encoding="utf-8")
            sqlite_dir = codex / "sqlite"
            sqlite_dir.mkdir()
            con = sqlite3.connect(sqlite_dir / "state_5.sqlite")
            con.execute("create table threads (id text, rollout_path text, cwd text, title text, first_user_message text, updated_at integer, model text)")
            con.execute("insert into threads values (?, ?, ?, ?, ?, ?, ?)", ("s1", str(rollout), "/tmp/project", "登录", "你好", 1, "gpt"))
            con.commit(); con.close()

            events = CodexImporter(codex_home=codex).import_events()

            self.assertGreaterEqual(len(events), 3)
            self.assertTrue(any(event.source == "codex" and event.role == "assistant" for event in events))
            self.assertTrue(any(event.metadata.get("rollout_path") == str(rollout) for event in events))

    def test_claude_scan_and_import_reads_transcripts(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            claude_home = home / ".claude"
            transcripts = claude_home / "transcripts"
            transcripts.mkdir(parents=True)
            transcript = transcripts / "ses_1.jsonl"
            transcript.write_text(
                json.dumps({"type": "user", "timestamp": "2026", "content": "始终中文回答"}, ensure_ascii=False) + "\n" +
                json.dumps({"type": "tool_result", "timestamp": "2026", "tool_output": "noise"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            projects = claude_home / "projects" / "-tmp-project"
            projects.mkdir(parents=True)
            project_transcript = projects / "session.jsonl"
            project_transcript.write_text(
                json.dumps({"type": "user", "sessionId": "s2", "timestamp": "2026", "cwd": "/tmp/project", "message": {"role": "user", "content": [{"type": "text", "text": "梳理这个项目"}]}}, ensure_ascii=False) + "\n" +
                json.dumps({"type": "user", "sessionId": "s2", "timestamp": "2026", "cwd": "/tmp/project", "message": {"role": "user", "content": "<command-message>init</command-message>\n<command-name>/init</command-name>"}}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            importer = ClaudeCodeImporter(claude_home=claude_home, global_state_path=home / ".claude.json")
            scan = importer.scan()
            events = importer.import_events()

            self.assertTrue(scan["transcripts_found"])
            self.assertEqual(scan["transcript_count"], 2)
            self.assertTrue(any(event.event_type == "transcript_message" and event.content == "始终中文回答" for event in events))
            self.assertTrue(any(event.event_type == "transcript_message" and event.project == "/tmp/project" and event.content == "梳理这个项目" for event in events))
            self.assertFalse(any("command-name" in event.content for event in events))

    def test_claude_import_reads_global_and_project_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            claude_home = home / ".claude"
            claude_home.mkdir()
            (claude_home / "CLAUDE.md").write_text("始终中文回答", encoding="utf-8")
            global_state = home / ".claude.json"
            global_state.write_text(json.dumps({"projects": {"/tmp/project": {"allowedTools": ["Read"]}}, "customApiKeyResponses": {"approved": ["x"]}}), encoding="utf-8")
            project = home / "project"
            project_settings = project / ".claude" / "settings.local.json"
            project_settings.parent.mkdir(parents=True)
            project_settings.write_text(json.dumps({"permissions": {"allow": ["Read"]}}), encoding="utf-8")

            events = ClaudeCodeImporter(claude_home=claude_home, global_state_path=global_state, project_roots=[project]).import_events()

            self.assertTrue(any(event.role == "system" and "始终中文回答" in event.content for event in events))
            self.assertTrue(any(event.event_type == "project_settings" for event in events))
            self.assertFalse(any("customApiKeyResponses" in json.dumps(event.metadata) for event in events))

    def test_write_events_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "events.jsonl"
            event = NormalizedSessionEvent(
                source="codex",
                session_id="s1",
                project="/tmp/project",
                timestamp="1",
                role="user",
                content="hello",
                event_type="message",
                metadata={"safe": True},
            )

            write_events_jsonl([event], out)

            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["source"], "codex")
            self.assertEqual(payload["content"], "hello")
