import json
import tempfile
import unittest
from pathlib import Path

from dream_memory.memory_cli import main
from dream_memory.memory_dreaming import build_agent_context, normalize_project_path
from dream_memory.memory_export import replace_marked_block


class MemoryExportTests(unittest.TestCase):
    def test_normalize_project_path_matches_relative_and_absolute_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            cards = [
                {"id": "project", "scope": "project", "project": str(root), "memory_type": "decision", "summary": "项目记忆。", "status": "active", "retrieval_hints": []},
                {"id": "other", "scope": "project", "project": str(root.parent), "memory_type": "decision", "summary": "其他项目。", "status": "active", "retrieval_hints": []},
                {"id": "user", "scope": "user", "project": None, "memory_type": "preference", "summary": "用户偏好中文。", "status": "active", "retrieval_hints": []},
            ]

            context = build_agent_context(cards, project=str(root / "."), limit=10)

            self.assertEqual(context["project"], normalize_project_path(str(root)))
            self.assertEqual([item["id"] for item in context["items"]], ["project", "user"])

    def test_summary_cli_writes_all_projects_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cards = root / "memory_cards.jsonl"
            output = root / "PROJECTS.md"
            cards.write_text("\n".join([
                json.dumps({"id": "user", "scope": "user", "memory_type": "preference", "summary": "用户偏好中文。", "status": "active"}, ensure_ascii=False),
                json.dumps({"id": "project", "scope": "project", "project": "/tmp/project-a", "memory_type": "decision", "summary": "项目 A 使用 uv。", "status": "active"}, ensure_ascii=False),
            ]) + "\n", encoding="utf-8")

            exit_code = main(["summary", "--memory-cards", str(cards), "--output", str(output)])

            self.assertEqual(exit_code, 0)
            text = output.read_text(encoding="utf-8")
            self.assertIn("Project Summary", text)
            self.assertIn("/tmp/project-a", text)
            self.assertIn("项目 A 使用 uv", text)

    def test_export_cli_writes_project_agents_and_claude_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            cards = root / "memory_cards.jsonl"
            cards.write_text("\n".join([
                json.dumps({"id": "project", "scope": "project", "project": str(root), "memory_type": "decision", "summary": "当前项目记忆。", "status": "active", "retrieval_hints": []}, ensure_ascii=False),
                json.dumps({"id": "other", "scope": "project", "project": "/tmp/other", "memory_type": "decision", "summary": "其他项目记忆。", "status": "active", "retrieval_hints": []}, ensure_ascii=False),
                json.dumps({"id": "user", "scope": "user", "project": None, "memory_type": "preference", "summary": "用户偏好中文。", "status": "active", "retrieval_hints": []}, ensure_ascii=False),
            ]) + "\n", encoding="utf-8")

            exit_code = main(["export", "--target", "both", "--project", str(root), "--memory-cards", str(cards), "--output-dir", str(root)])

            self.assertEqual(exit_code, 0)
            agents = (root / "AGENTS.md").read_text(encoding="utf-8")
            claude = (root / "CLAUDE.md").read_text(encoding="utf-8")
            self.assertIn("DREAM_MEMORY_START", agents)
            self.assertIn("当前项目记忆", agents)
            self.assertIn("用户偏好中文", claude)
            self.assertNotIn("其他项目记忆", agents)

    def test_replace_marked_block_preserves_surrounding_content(self):
        existing = "# Notes\n\n<!-- DREAM_MEMORY_START -->\nold\n<!-- DREAM_MEMORY_END -->\n\nFooter\n"
        updated = replace_marked_block(existing, "new")

        self.assertIn("# Notes", updated)
        self.assertIn("new", updated)
        self.assertNotIn("old", updated)
        self.assertIn("Footer", updated)

    def test_export_cli_creates_missing_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            cards = root / "memory_cards.jsonl"
            output_dir = root / "missing" / "nested"
            cards.write_text(json.dumps({"id": "project", "scope": "project", "project": str(root), "memory_type": "decision", "summary": "当前项目记忆。", "status": "active", "retrieval_hints": []}, ensure_ascii=False) + "\n", encoding="utf-8")

            exit_code = main(["export", "--target", "codex", "--project", str(root), "--memory-cards", str(cards), "--output-dir", str(output_dir)])

            self.assertEqual(exit_code, 0)
            self.assertTrue((output_dir / "AGENTS.md").exists())


if __name__ == "__main__":
    unittest.main()
