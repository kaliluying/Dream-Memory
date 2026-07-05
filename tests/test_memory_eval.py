import json
import tempfile
import unittest
from pathlib import Path

from dream_memory.memory_cli import main
from dream_memory.memory_eval import evaluate_labeled_events


class MemoryEvalTests(unittest.TestCase):
    def test_evaluate_labeled_events_reports_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labeled.jsonl"
            path.write_text(json.dumps({
                "event": {"event_id": "event_1", "source": "codex", "role": "user", "event_type": "history_prompt", "content": "用户偏好中文回答。"},
                "expected": [{"content": "用户偏好中文回答", "type": "preference", "scope": "user"}],
            }, ensure_ascii=False) + "\n", encoding="utf-8")

            result = evaluate_labeled_events(path, project=None, mode="rules")

            self.assertEqual(result["expected_total"], 1)
            self.assertGreaterEqual(result["recall"], 0.0)
            self.assertIn("precision", result)

    def test_eval_cli_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labeled.jsonl"
            output = Path(tmp) / "eval.json"
            path.write_text(json.dumps({
                "event": {"event_id": "event_1", "source": "codex", "role": "user", "event_type": "history_prompt", "content": "用户偏好中文回答。"},
                "expected": [{"content": "用户偏好中文回答", "type": "preference", "scope": "user"}],
            }, ensure_ascii=False) + "\n", encoding="utf-8")

            exit_code = main(["eval", "--input", str(path), "--output", str(output)])

            self.assertEqual(exit_code, 0)
            self.assertTrue(output.exists())
            self.assertIn("f1", json.loads(output.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
