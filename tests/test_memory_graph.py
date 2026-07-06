import json
import unittest
from unittest.mock import patch

from dream_memory.memory_graph import build_memory_extraction_graph, run_memory_extraction_graph
from dream_memory.model_providers import ModelRuntimeResult


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

        with patch("dream_memory.memory_graph.invoke_model_provider", return_value=raw):
            result = run_memory_extraction_graph(
                [{"source": "codex", "role": "user", "content": "始终中文回答"}],
                project="/tmp/project",
                model="anthropic:claude-sonnet-4-6",
                invoke_model=True,
            )

        self.assertFalse(result["dry_run"])
        self.assertEqual(result["candidates"][0]["status"], "promote")

    def test_graph_invokes_runtime_and_surfaces_metadata(self):
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

        runtime_config = {
            "models": {
                "primary": {
                    "provider": "openai",
                    "model": "gpt-4.1",
                    "api_key": "openai-key",
                    "timeout_seconds": 45,
                }
            },
            "model_policy": {
                "default_profile": "primary",
                "fallback_chain": ["primary"],
                "retry": {"max_attempts": 1},
            },
        }

        with patch("dream_memory.memory_graph.invoke_model_runtime") as invoke:
            invoke.return_value = ModelRuntimeResult(
                text=raw,
                selected_profile="primary",
                attempts=[],
                elapsed_ms=1,
            )
            result = run_memory_extraction_graph(
                [{"source": "codex", "role": "user", "content": "始终中文回答"}],
                project="/tmp/project",
                model="openai:gpt-4.1",
                invoke_model=True,
                runtime_config=runtime_config,
            )

        self.assertFalse(result["dry_run"])
        self.assertEqual(result["model_runtime"]["selected_profile"], "primary")
        self.assertEqual(result["candidates"][0]["status"], "promote")

    def test_build_memory_extraction_graph_returns_invokable_graph(self):
        graph = build_memory_extraction_graph()

        self.assertTrue(callable(graph.invoke))


if __name__ == "__main__":
    unittest.main()
