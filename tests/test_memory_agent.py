import json
import tempfile
import unittest
from pathlib import Path

from dream_memory.memory_agent import (
    agent_extract_memory_candidates,
    build_memory_extraction_prompt,
    extract_json_payload,
    validate_agent_candidates,
)


class MemoryAgentTests(unittest.TestCase):
    def test_build_memory_extraction_prompt_contains_schema_and_events(self):
        events = [{"source": "codex", "role": "user", "content": "希望项目像 Claude Code", "project": "/tmp/p"}]

        prompt = build_memory_extraction_prompt(events, project="/tmp/p")

        self.assertIn("JSON", prompt)
        self.assertIn("candidates", prompt)
        self.assertIn("Claude Code", prompt)
        self.assertIn("reject tool state", prompt)
        self.assertIn("Simplified Chinese", prompt)
        self.assertIn("Keep product names", prompt)
        self.assertIn("Project filter is strict", prompt)
        self.assertIn("Do not promote one-off implementation tasks", prompt)

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

    def test_agent_extract_aggregates_atomic_facts_into_candidates(self):
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        model = FakeListChatModel(responses=[json.dumps({
            "atomic_facts": [
                {
                    "statement": "用户希望项目做成 Claude Code 风格的本地研发助手。",
                    "fact_type": "product_direction",
                    "scope": "project",
                    "project": "/tmp/p",
                    "confidence": 0.92,
                    "evidence": [
                        {
                            "event_id": "event_1",
                            "source": "codex",
                            "session_id": "s1",
                            "quote": "希望项目像 Claude Code",
                        }
                    ],
                    "long_term": True,
                    "reuse_scenarios": ["后续功能取舍", "产品定位判断"],
                    "tags": ["claude-code", "product-direction"],
                },
                {
                    "statement": "用户希望项目做成 Claude Code 风格的本地研发助手。",
                    "fact_type": "product_direction",
                    "scope": "project",
                    "project": "/tmp/p",
                    "confidence": 0.88,
                    "evidence": [{"event_id": "event_2", "source": "codex", "session_id": "s2"}],
                    "long_term": True,
                    "reuse_scenarios": ["需求评审"],
                    "tags": ["product-direction"],
                },
            ]
        }, ensure_ascii=False)])

        result = agent_extract_memory_candidates(
            [
                {"source": "codex", "session_id": "s1", "role": "user", "content": "希望项目像 Claude Code", "project": "/tmp/p"},
                {"source": "codex", "session_id": "s2", "role": "assistant", "content": "确认项目目标是本地研发助手", "project": "/tmp/p"},
            ],
            project="/tmp/p",
            model=model,
            invoke_model=True,
        )

        self.assertFalse(result["dry_run"])
        self.assertEqual(len(result["atomic_facts"]), 2)
        self.assertEqual(len(result["candidates"]), 1)
        candidate = result["candidates"][0]
        self.assertEqual(candidate["type"], "product_direction")
        self.assertEqual(candidate["scope"], "project")
        self.assertEqual(candidate["project"], "/tmp/p")
        self.assertEqual(len(candidate["evidence"]), 2)
        self.assertIn("后续功能取舍", candidate["retrieval_hints"])
        self.assertIn("long_term", candidate["quality_reason"])

    def test_validate_agent_candidates_drops_invalid_and_secret_content(self):
        candidates = [
            {
                "content": "用户偏好中文回答。",
                "type": "preference",
                "scope": "user",
                "project": None,
                "confidence": 0.95,
                "decision": "promote",
                "reason": "explicit preference",
                "evidence": [{"event_id": "event_1", "source": "codex"}],
                "tags": ["language"],
            },
            {
                "content": "OPENAI_API_KEY=sk-secret-value",
                "type": "preference",
                "scope": "user",
                "confidence": 0.9,
                "decision": "promote",
                "evidence": [{"event_id": "event_2"}],
            },
            {"content": "缺少 evidence", "type": "preference", "scope": "user", "decision": "promote"},
        ]

        valid = validate_agent_candidates(candidates, project="/tmp/project")

        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0]["content"], "用户偏好中文回答。")
        self.assertEqual(valid[0]["status"], "promote")
        self.assertEqual(valid[0]["score"], 0.95)

    def test_validate_agent_candidates_keeps_only_durable_value(self):
        candidates = [
            {
                "content": "用户偏好始终使用中文回答，除非明确要求不要额外生成总结文档。",
                "type": "preference",
                "scope": "user",
                "project": None,
                "confidence": 0.95,
                "decision": "promote",
                "reason": "explicit durable preference",
                "evidence": [{"event_id": "event_1", "source": "codex"}],
                "tags": ["language"],
            },
            {
                "content": "菜单隐藏最佳实践：保留路由，通过 meta.hidden=true 隐藏左侧菜单，并在菜单加工层统一客户可见名称。",
                "type": "workflow",
                "scope": "global",
                "project": None,
                "confidence": 0.9,
                "decision": "promote",
                "reason": "cross-project reusable implementation pattern",
                "evidence": [{"event_id": "event_2", "source": "codex"}],
                "tags": ["menu", "workflow"],
            },
            {
                "content": "WeComAgent 前端需删除水印组件",
                "type": "requirement",
                "scope": "project",
                "project": "/tmp/current",
                "confidence": 0.8,
                "decision": "promote",
                "reason": "one-off task",
                "evidence": [{"event_id": "event_3", "source": "codex"}],
                "tags": ["task"],
            },
            {
                "content": "图片生成脚本需求：使用 gpt-image-2 模型调用接口，密钥在 key.txt，三并发跑两轮测试。",
                "type": "workflow",
                "scope": "project",
                "project": "/tmp/current",
                "confidence": 0.85,
                "decision": "promote",
                "reason": "contains credential location and one-off test setup",
                "evidence": [{"event_id": "event_4", "source": "codex"}],
                "tags": ["script"],
            },
            {
                "content": "其他项目的长期产品方向。",
                "type": "product_direction",
                "scope": "project",
                "project": "/tmp/other",
                "confidence": 0.95,
                "decision": "promote",
                "reason": "different project",
                "evidence": [{"event_id": "event_5", "source": "codex"}],
                "tags": ["project"],
            },
        ]

        valid = validate_agent_candidates(candidates, project="/tmp/current")

        self.assertEqual([item["content"] for item in valid], [
            "用户偏好始终使用中文回答，除非明确要求不要额外生成总结文档。",
            "菜单隐藏最佳实践：保留路由，通过 meta.hidden=true 隐藏左侧菜单，并在菜单加工层统一客户可见名称。",
        ])

    def test_validate_agent_candidates_global_run_drops_project_scoped_items(self):
        candidates = [
            {
                "content": "当前项目需要实现某个页面功能。",
                "type": "requirement",
                "scope": "project",
                "project": "/tmp/project",
                "confidence": 0.8,
                "decision": "promote",
                "reason": "project task",
                "evidence": [{"event_id": "event_1", "source": "codex"}],
                "tags": ["task"],
            },
            {
                "content": "用户偏好始终使用中文回答。",
                "type": "preference",
                "scope": "user",
                "project": None,
                "confidence": 0.95,
                "decision": "promote",
                "reason": "global user preference",
                "evidence": [{"event_id": "event_2", "source": "codex"}],
                "tags": ["language"],
            },
        ]

        valid = validate_agent_candidates(candidates, project=None)

        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0]["scope"], "user")

    def test_validate_agent_candidates_drops_shallow_project_tasks(self):
        candidates = [
            {
                "content": "WeComAgent 首页组件需使用真实数据（http://localhost:5180/web#/home）",
                "type": "requirement",
                "scope": "project",
                "project": "/tmp/current",
                "confidence": 0.8,
                "decision": "promote",
                "reason": "one-off page implementation task",
                "evidence": [{"event_id": "event_1", "source": "codex"}],
                "tags": ["home"],
            },
            {
                "content": "WeComAgent 侧边栏配置中心的英文需全部改为中文",
                "type": "requirement",
                "scope": "project",
                "project": "/tmp/current",
                "confidence": 0.85,
                "decision": "promote",
                "reason": "one-off localization task",
                "evidence": [{"event_id": "event_2", "source": "codex"}],
                "tags": ["localization"],
            },
            {
                "content": "WeComAgent 全流程测试重点关注侧边栏功能",
                "type": "workflow",
                "scope": "project",
                "project": "/tmp/current",
                "confidence": 0.75,
                "decision": "review",
                "reason": "one-off test focus",
                "evidence": [{"event_id": "event_3", "source": "codex"}],
                "tags": ["test"],
            },
            {
                "content": "BI 高潜客户判定标准：满足预算门槛、强购买意愿、对公司高认可度；名单每日 8:00 基于前一天数据生成，当日有效，并通过企业微信工作台通知和侧边栏 Banner 触达。",
                "type": "product_direction",
                "scope": "project",
                "project": "/tmp/current",
                "confidence": 0.95,
                "decision": "promote",
                "reason": "long-lived product rule",
                "evidence": [{"event_id": "event_4", "source": "codex"}],
                "tags": ["BI高潜客户"],
            },
        ]

        valid = validate_agent_candidates(candidates, project="/tmp/current")

        self.assertEqual(len(valid), 1)
        self.assertEqual(valid[0]["type"], "product_direction")
