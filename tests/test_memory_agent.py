import json
import tempfile
import unittest
from pathlib import Path

from deepagent_memory.memory_agent import (
    agent_extract_memory_candidates,
    build_memory_extraction_prompt,
    extract_json_payload,
)


class MemoryAgentTests(unittest.TestCase):
    def test_build_memory_extraction_prompt_contains_schema_and_events(self):
        events = [{"source": "codex", "role": "user", "content": "希望项目像 Claude Code", "project": "/tmp/p"}]

        prompt = build_memory_extraction_prompt(events, project="/tmp/p")

        self.assertIn("JSON", prompt)
        self.assertIn("candidates", prompt)
        self.assertIn("Claude Code", prompt)
        self.assertIn("reject tool state", prompt)

    def test_extract_json_payload_from_fenced_response(self):
        text = '''Here:
```json
{"candidates":[{"content":"用户偏好中文回答","type":"preference","scope":"global","confidence":0.9,"decision":"promote","reason":"explicit"}]}
```
'''

        payload = extract_json_payload(text)

        self.assertEqual(payload["candidates"][0]["decision"], "promote")

    def test_agent_extract_dry_run_does_not_call_model(self):
        result = agent_extract_memory_candidates(
            [{"source": "codex", "role": "user", "content": "希望项目像 Claude Code"}],
            project="/tmp/p",
            invoke_model=False,
        )

        self.assertTrue(result["dry_run"])
        self.assertEqual(result["candidates"], [])
        self.assertIn("prompt", result)

    def test_agent_extract_with_fake_model_returns_candidates(self):
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        model = FakeListChatModel(responses=[json.dumps({
            "candidates": [
                {
                    "content": "用户希望项目做成 Claude Code 风格的本地研发助手。",
                    "type": "product_direction",
                    "scope": "project",
                    "project": "/tmp/p",
                    "confidence": 0.95,
                    "decision": "promote",
                    "reason": "explicit user request",
                    "evidence": [{"source": "codex", "session_id": "s1"}],
                    "tags": ["claude-code", "product-direction"],
                }
            ]
        }, ensure_ascii=False)])

        result = agent_extract_memory_candidates(
            [{"source": "codex", "session_id": "s1", "role": "user", "content": "希望项目像 Claude Code", "project": "/tmp/p"}],
            project="/tmp/p",
            model=model,
            invoke_model=True,
        )

        self.assertFalse(result["dry_run"])
        self.assertEqual(result["candidates"][0]["decision"], "promote")
        self.assertIn("Claude Code", result["candidates"][0]["content"])
