import json
import tempfile
import unittest
from pathlib import Path

from dream_memory.memory_agent import (
    agent_extract_memory_candidates,
    build_memory_extraction_prompt,
    extract_json_payload,
    validate_agent_candidates,
    build_agent_candidates_from_payload,
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
        self.assertIn("User-scope preferences", prompt)
        self.assertIn("Do not promote one-off implementation tasks", prompt)
        self.assertIn("project_instruction", prompt)
        self.assertIn("Python uses uv", prompt)
        self.assertIn("用户偏好中文回答", prompt)

    def test_build_memory_extraction_prompt_preserves_original_event_ids_for_evidence(self):
        events = [{"event_id": "eval_pref_1", "source": "codex", "role": "user", "content": "用户偏好中文回答。", "project": "/tmp/p"}]

        prompt = build_memory_extraction_prompt(events, project="/tmp/p")

        self.assertIn('"event_id": "eval_pref_1"', prompt)
        self.assertNotIn('"event_id": "event_1"', prompt)


    def test_build_memory_extraction_prompt_filters_code_listing_and_stale_project_advice_docs(self):
        events = [
            {
                "event_id": "doc_dump",
                "source": "claude_code",
                "role": "user",
                "event_type": "transcript_message",
                "project": "/tmp/project",
                "content": "1\t# DeepAgents 项目建议\n2\t\n3\t## 推荐方向\n4\t\n5\t我最推荐先做一个 **AI 项目研发助手 / 代码任务代理**，而不是一开始做通用聊天机器人或复杂多智能体平台。\n6\t更多文档内容",
            },
            {
                "event_id": "code_dump",
                "source": "claude_code",
                "role": "user",
                "event_type": "transcript_message",
                "project": "/tmp/project",
                "content": "1\tfrom __future__ import annotations\n2\timport json\n3\tclass Demo:\n4\t    def run(self):\n5\t        return True\n6\t# more source",
            },
            {
                "event_id": "durable",
                "source": "codex",
                "role": "user",
                "event_type": "history_prompt",
                "project": "/tmp/project",
                "content": "正式记忆必须经过人工审核，只有审核通过的候选才允许写入长期记忆。",
            },
        ]

        prompt = build_memory_extraction_prompt(events, project="/tmp/project")

        self.assertNotIn("AI 项目研发助手", prompt)
        self.assertNotIn("from __future__", prompt)
        self.assertIn("正式记忆必须经过人工审核", prompt)


    def test_build_memory_extraction_prompt_filters_project_state_and_settings_events(self):
        events = [
            {
                "event_id": "project_state",
                "source": "claude_code",
                "role": "system",
                "event_type": "project_state",
                "project": "/tmp/project",
                "content": "Claude Code project state for /tmp/project",
            },
            {
                "event_id": "project_settings",
                "source": "claude_code",
                "role": "system",
                "event_type": "project_settings",
                "project": "/tmp/project",
                "content": "Claude Code local project settings for /tmp/project",
            },
            {
                "event_id": "durable",
                "source": "codex",
                "role": "user",
                "event_type": "history_prompt",
                "project": "/tmp/project",
                "content": "正式记忆必须经过人工审核，只有审核通过的候选才允许写入长期记忆。",
            },
        ]

        prompt = build_memory_extraction_prompt(events, project="/tmp/project")

        self.assertNotIn("Claude Code project state", prompt)
        self.assertNotIn("Claude Code local project settings", prompt)
        self.assertIn("正式记忆必须经过人工审核", prompt)

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


    def test_agent_extract_dry_run_reports_prompt_filter_counts(self):
        result = agent_extract_memory_candidates(
            [
                {"event_id": "state", "source": "claude_code", "role": "system", "event_type": "project_state", "content": "Claude Code project state for /tmp/project"},
                {"event_id": "durable", "source": "codex", "role": "user", "event_type": "history_prompt", "content": "正式记忆必须经过人工审核，只有审核通过的候选才允许写入长期记忆。"},
            ],
            project="/tmp/project",
            invoke_model=False,
        )

        self.assertEqual(result["input_event_count"], 2)
        self.assertEqual(result["prompt_event_count"], 1)
        self.assertEqual(result["filtered_prompt_event_count"], 1)


    def test_agent_extract_prompt_filter_counts_respect_preview_limit(self):
        events = [
            {
                "event_id": f"event_{index}",
                "source": "codex",
                "role": "user",
                "event_type": "history_prompt",
                "content": f"长期记忆测试事件 {index}：正式记忆必须经过人工审核。",
            }
            for index in range(82)
        ]

        result = agent_extract_memory_candidates(events, project="/tmp/project", invoke_model=False)

        self.assertEqual(result["input_event_count"], 82)
        self.assertEqual(result["prompt_event_count"], 80)
        self.assertEqual(result["filtered_prompt_event_count"], 2)
        self.assertIn('"event_id": "event_79"', result["prompt"])
        self.assertNotIn('"event_id": "event_80"', result["prompt"])

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

    def test_validate_agent_atomic_facts_keeps_package_manager_workflow(self):
        from dream_memory.memory_agent import build_agent_candidates_from_payload

        payload = {
            "atomic_facts": [
                {
                    "statement": "Python 后端使用 uv 进行包管理，前端使用 pnpm 进行管理。",
                    "fact_type": "workflow",
                    "scope": "project",
                    "project": "/tmp/project",
                    "confidence": 0.92,
                    "evidence": [{"event_id": "event_1", "source": "project", "quote": "Python 使用 uv，前端使用 pnpm"}],
                    "long_term": True,
                    "tags": ["package-manager", "uv", "pnpm"],
                }
            ]
        }

        facts, candidates = build_agent_candidates_from_payload(payload, project="/tmp/project")

        self.assertEqual(len(facts), 1)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["type"], "workflow")
        self.assertIn("uv", candidates[0]["content"])

    def test_validate_agent_atomic_facts_splits_combined_project_marker_memory(self):
        from dream_memory.memory_agent import build_agent_candidates_from_payload

        payload = {
            "atomic_facts": [
                {
                    "statement": "Python 后端使用 uv 进行包管理，使用 pytest 进行测试，使用 FastAPI 作为框架。",
                    "fact_type": "workflow",
                    "scope": "project",
                    "project": "/tmp/project",
                    "confidence": 0.92,
                    "evidence": [{"event_id": "eval_marker_1", "source": "project", "quote": "python_package_manager=uv; python_test_runner=pytest; python_framework=fastapi"}],
                    "long_term": True,
                    "tags": ["uv", "pytest", "fastapi"],
                }
            ]
        }

        facts, candidates = build_agent_candidates_from_payload(payload, project="/tmp/project")
        contents = "\n".join(candidate["content"] for candidate in candidates)
        types_by_content = {candidate["content"]: candidate["type"] for candidate in candidates}

        self.assertEqual(len(facts), 3)
        self.assertIn("Python 项目使用 uv", contents)
        self.assertIn("Python 测试使用 pytest", contents)
        self.assertIn("项目使用 FastAPI", contents)
        self.assertEqual(types_by_content["项目使用 FastAPI 作为 Python Web 框架。"], "project_fact")

    def test_validate_agent_atomic_facts_canonicalizes_pitfall_and_rejected_option_paraphrases(self):
        from dream_memory.memory_agent import build_agent_candidates_from_payload

        payload = {
            "atomic_facts": [
                {
                    "statement": "对于登录跳转和退出状态这类可见产品问题，不能只看 API 返回成功，必须真实跑 UI 流程验证。",
                    "fact_type": "pitfall",
                    "scope": "user",
                    "confidence": 0.9,
                    "evidence": [{"event_id": "eval_pitfall_1", "source": "codex"}],
                    "long_term": True,
                    "tags": ["ui"],
                },
                {
                    "statement": "不要将未经审核的候选自动写入 `MEMORY.md`。",
                    "fact_type": "rejected_option",
                    "scope": "project",
                    "project": "/tmp/project",
                    "confidence": 0.9,
                    "evidence": [{"event_id": "eval_rejected_1", "source": "codex"}],
                    "long_term": True,
                    "tags": ["memory-safety"],
                },
            ]
        }

        _facts, candidates = build_agent_candidates_from_payload(payload, project="/tmp/project")
        by_type = {candidate["type"]: candidate for candidate in candidates}

        self.assertIn("不要只看 API 返回成功", by_type["pitfall"]["content"])
        self.assertEqual(by_type["pitfall"]["scope"], "user")
        self.assertIn("不要把未经审核的候选自动写入", by_type["rejected_option"]["content"])
        self.assertEqual(by_type["rejected_option"]["scope"], "project")

    def test_validate_agent_atomic_facts_canonicalizes_review_gate_scope_and_type(self):
        from dream_memory.memory_agent import build_agent_candidates_from_payload

        payload = {
            "atomic_facts": [
                {
                    "statement": "正式记忆必须经过人工审核，只有审核通过的候选才允许写入长期记忆。",
                    "fact_type": "requirement",
                    "scope": "project",
                    "project": "/tmp/project",
                    "confidence": 0.9,
                    "evidence": [{"event_id": "eval_review_1", "source": "codex"}],
                    "long_term": True,
                    "tags": ["review"],
                }
            ]
        }

        _facts, candidates = build_agent_candidates_from_payload(payload, project="/tmp/project")

        self.assertEqual(candidates[0]["type"], "workflow")
        self.assertEqual(candidates[0]["scope"], "project")
        self.assertIn("正式记忆必须经过人工审核", candidates[0]["content"])

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


    def test_validate_agent_atomic_facts_project_scopes_review_gate_when_model_says_global(self):
        facts = [
            {
                "statement": "正式记忆必须经过人工审核，只有审核通过的候选才允许写入长期记忆。",
                "fact_type": "workflow",
                "scope": "global",
                "evidence": [{"event_id": "eval_review_1", "source": "codex"}],
            }
        ]

        candidates = build_agent_candidates_from_payload({"atomic_facts": facts}, project="/tmp/project")[1]

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["scope"], "project")
        self.assertEqual(candidates[0]["project"], "/tmp/project")
        self.assertIn("正式记忆必须经过人工审核", candidates[0]["content"])

    def test_validate_agent_atomic_facts_project_scopes_rejected_option_when_model_says_user(self):
        facts = [
            {
                "statement": "不要把未经审核的候选自动写入 MEMORY.md，这个方案风险太高。",
                "fact_type": "rejected_option",
                "scope": "user",
                "evidence": [{"event_id": "eval_rejected_1", "source": "codex"}],
            }
        ]

        candidates = build_agent_candidates_from_payload({"atomic_facts": facts}, project="/tmp/project")[1]

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["scope"], "project")
        self.assertEqual(candidates[0]["project"], "/tmp/project")
        self.assertIn("不要把未经审核的候选自动写入", candidates[0]["content"])


    def test_validate_agent_outputs_drops_credential_location_evidence_even_when_content_is_safety_summary(self):
        payload = {
            "atomic_facts": [
                {
                    "statement": "不要记录或复述 API key 的存放位置；涉及真实凭据的路径信息必须忽略。",
                    "fact_type": "requirement",
                    "scope": "project",
                    "project": "/tmp/project",
                    "evidence": [{"event_id": "eval_secret_location_1", "source": "codex", "quote": "API key 放在 key.txt，记得读取这个文件"}],
                }
            ],
            "candidates": [
                {
                    "content": "不要记录或复述 API key 的存放位置；涉及真实凭据的路径信息必须忽略。",
                    "type": "requirement",
                    "scope": "project",
                    "project": "/tmp/project",
                    "decision": "promote",
                    "evidence": [{"event_id": "eval_secret_location_1", "source": "codex", "quote": "API key 放在 key.txt，记得读取这个文件"}],
                }
            ],
        }

        atomic_facts, candidates = build_agent_candidates_from_payload(payload, project="/tmp/project")

        self.assertEqual(atomic_facts, [])
        self.assertEqual(candidates, [])

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
