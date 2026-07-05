import json
import unittest
from unittest.mock import patch

from deepagent_memory.memory_graph import build_memory_extraction_graph, run_memory_extraction_graph


class MemoryGraphTests(unittest.TestCase):
    def test_graph_dry_run_builds_prompt_without_candidates(self):
        result = run_memory_extraction_graph(
            [{"source": "codex", "role": "user", "content": "始终中文回答"}],
            project="/tmp/project",
            model="anthropic:claude-sonnet-4-6",
            invoke_model=False,
        )

        self.assertTrue(result["dry_run"])
        self.assertIn("始终中文回答", result["prompt"])
        self.assertEqual(result["candidates"], [])

    def test_graph_invokes_provider_and_validates_candidates(self):
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

        with patch("deepagent_memory.memory_graph.invoke_model_provider", return_value=raw):
            result = run_memory_extraction_graph(
                [{"source": "codex", "role": "user", "content": "始终中文回答"}],
                project="/tmp/project",
                model="anthropic:claude-sonnet-4-6",
                invoke_model=True,
            )

        self.assertFalse(result["dry_run"])
        self.assertEqual(result["candidates"][0]["status"], "promote")

    def test_build_memory_extraction_graph_returns_invokable_graph(self):
        graph = build_memory_extraction_graph()

        self.assertTrue(callable(graph.invoke))


if __name__ == "__main__":
    unittest.main()
