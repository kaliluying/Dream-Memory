import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from deepagent_memory.memory_importers import (
    ClaudeCodeImporter,
    CodexImporter,
    NormalizedSessionEvent,
    redact_sensitive,
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
