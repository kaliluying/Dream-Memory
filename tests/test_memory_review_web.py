import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from dream_memory.memory_dreaming import build_review_queue
from dream_memory.memory_runs import append_trace, create_run_state, update_run_state
from dream_memory.web import create_app


class MemoryReviewWebTests(unittest.TestCase):
    def test_home_redirects_to_memory_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.get("/", follow_redirects=False)

            self.assertEqual(response.status_code, 307)
            self.assertEqual(response.headers["location"], "/memory-review")

    def test_memory_review_page_contains_review_ui(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.get("/memory-review")

            self.assertEqual(response.status_code, 200)
            self.assertIn("Dream Memory Review", response.text)
            self.assertIn("候选记忆", response.text)
            self.assertIn("批准", response.text)
            self.assertIn("拒绝", response.text)
            self.assertIn("需要更多证据", response.text)
            self.assertIn("新建 AI 提取", response.text)
            self.assertIn("记忆归属", response.text)
            self.assertIn("当前项目", response.text)
            self.assertIn("全局记忆", response.text)
            self.assertIn("不会扫描项目目录", response.text)
            self.assertIn("生成新候选", response.text)
            self.assertIn("setProjectScope", response.text)
            self.assertIn("startAiRun", response.text)
            self.assertIn("/api/memory/runs/start", response.text)
            self.assertIn("/memory-config", response.text)
            self.assertIn('class="config-button"', response.text)
            self.assertIn('aria-label="打开运行配置"', response.text)
            self.assertIn("运行配置", response.text)

    def test_memory_config_page_contains_cli_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.get("/memory-config")

            self.assertEqual(response.status_code, 200)
            self.assertIn("Dream Memory 配置", response.text)
            self.assertIn("模型配置", response.text)
            self.assertIn("重试与切换", response.text)
            self.assertIn("导入与扫描", response.text)
            self.assertIn("运行参数", response.text)
            self.assertIn("上下文与导出", response.text)
            self.assertIn("审核与应用", response.text)
            self.assertIn("状态与追踪", response.text)
            self.assertIn("初始化与评估", response.text)
            self.assertIn("模型服务商", response.text)
            self.assertIn("接口密钥", response.text)
            self.assertIn("备用配置链", response.text)
            self.assertIn("检查配置档", response.text)
            self.assertIn("当前配置档", response.text)
            self.assertIn("新增配置档", response.text)
            self.assertIn("删除配置档", response.text)
            self.assertIn("导出条数", response.text)
            self.assertIn("候选文件", response.text)
            self.assertIn("Trace 候选 ID", response.text)
            self.assertIn("评估输入", response.text)
            self.assertNotIn(">provider <select", response.text)
            self.assertNotIn(">fallback_chain <input", response.text)
            self.assertNotIn(">review_candidates <input", response.text)
            self.assertIn("/api/memory/config", response.text)
            self.assertIn("/api/memory/models", response.text)
            self.assertIn("获取模型列表", response.text)
            self.assertIn('onclick="loadModelCatalog(event)"', response.text)
            self.assertIn('onclick="saveConfig(event)"', response.text)
            self.assertIn('onclick="loadConfig(event)"', response.text)
            self.assertIn('onclick="resetConfig(event)"', response.text)
            self.assertIn('id="status" class="status status-banner"', response.text)
            self.assertIn('<select id="model"', response.text)
            self.assertIn('<select id="activeProfile"', response.text)
            self.assertIn('<select id="defaultProfile"', response.text)
            self.assertIn('<select id="checkProviderProfile"', response.text)
            self.assertIn('id="fallbackChain"', response.text)
            self.assertIn('id="retrySwitchModel"', response.text)
            self.assertIn("switch_model_on_retry", response.text)
            self.assertIn("refreshModelList", response.text)
            self.assertIn("refreshProfileSelectors", response.text)
            self.assertIn("withBusyStatus", response.text)
            self.assertIn("String.fromCharCode(92)", response.text)
            self.assertNotIn("await loadModelCatalog()", response.text)

    def test_api_memory_models_fetches_provider_models_from_request_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            with patch("dream_memory.web.list_provider_models", return_value=["model-b", "model-a"]) as list_models:
                response = client.post(
                    "/api/memory/models",
                    json={
                        "provider": "openai",
                        "model": "existing-model",
                        "api_key": "sk-test",
                        "base_url": "http://localhost:3000",
                        "timeout_seconds": 3,
                    },
                )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["provider"], "openai")
            self.assertEqual(payload["models"], ["model-a", "model-b"])
            config = list_models.call_args.args[0]
            self.assertEqual(config.provider, "openai")
            self.assertEqual(config.model, "existing-model")
            self.assertEqual(config.api_key, "sk-test")
            self.assertEqual(config.base_url, "http://localhost:3000")
            self.assertEqual(config.timeout_seconds, 3)

    def test_api_memory_models_redacts_sensitive_model_names_in_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            with patch("dream_memory.web.list_provider_models", return_value=["safe-model", "model-sk-test-secret"]):
                response = client.post(
                    "/api/memory/models",
                    json={
                        "provider": "openai",
                        "model": "existing-model",
                        "api_key": "sk-test-secret",
                        "base_url": "https://user:pass123@example.test/v1",
                        "timeout_seconds": 3,
                    },
                )

            payload_text = json.dumps(response.json(), ensure_ascii=False)
            self.assertEqual(response.status_code, 200)
            self.assertNotIn("sk-test-secret", payload_text)
            self.assertNotIn("user:pass123", payload_text)
            self.assertIn("<redacted>", payload_text)
            self.assertIn("safe-model", response.json()["models"])

    def test_api_memory_config_reads_and_updates_config_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            read_response = client.get("/api/memory/config")

            self.assertEqual(read_response.status_code, 200)
            self.assertEqual(read_response.json()["config"]["mode"], "ai")
            self.assertIn("cli_options", read_response.json())
            cli_options = read_response.json()["cli_options"]
            self.assertIn("init", cli_options)
            self.assertIn("extract-facts", cli_options)
            self.assertIn("review", cli_options)
            self.assertIn("apply", cli_options)
            self.assertIn("status", cli_options)
            self.assertIn("resume", cli_options)
            self.assertIn("trace", cli_options)
            self.assertIn("summary", cli_options)
            self.assertIn("eval", cli_options)
            self.assertIn("--output-dir", cli_options["init"])
            self.assertIn("--timeout-seconds", cli_options["eval"])
            self.assertIn("--max-attempts", cli_options["eval"])
            self.assertIn("--fallback-rules-on-error", cli_options["eval"])
            self.assertIn("--fallback-rules-on-empty", cli_options["eval"])

            payload = read_response.json()["config"]
            payload["mode"] = "rules"
            payload["invoke_model"] = False
            payload["models"]["primary"]["provider"] = "openai"
            payload["models"]["primary"]["model"] = "gpt-4.1"
            payload["models"]["primary"]["api_key"] = "local-key"
            payload["model_policy"]["retry"]["max_attempts"] = 2
            update_response = client.put("/api/memory/config", json={"config": payload})

            self.assertEqual(update_response.status_code, 200)
            saved = json.loads((memory_dir / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["mode"], "rules")
            self.assertFalse(saved["invoke_model"])
            self.assertEqual(saved["models"]["primary"]["provider"], "openai")
            self.assertEqual(saved["models"]["primary"]["api_key"], "local-key")
            self.assertEqual(saved["model_policy"]["retry"]["max_attempts"], 2)

    def test_api_memory_config_redacts_sensitive_paths_in_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory-sk-test-secret"
            memory_dir.mkdir()
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            read_response = client.get("/api/memory/config")
            reset_response = client.post("/api/memory/config/reset")

            read_text = json.dumps(read_response.json(), ensure_ascii=False)
            reset_text = json.dumps(reset_response.json(), ensure_ascii=False)
            self.assertEqual(read_response.status_code, 200)
            self.assertEqual(reset_response.status_code, 200)
            self.assertNotIn("sk-test-secret", read_text)
            self.assertNotIn("sk-test-secret", reset_text)
            self.assertIn("<redacted>", read_text)
            self.assertIn("<redacted>", reset_text)

    def test_api_memory_config_redacts_sensitive_profile_names_in_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            config_path = memory_dir / "config.json"
            config_path.write_text(json.dumps({
                "models": {
                    "profile-sk-test-secret": {
                        "provider": "openai",
                        "model": "gpt-4.1",
                        "api_key": "",
                        "api_key_env": None,
                        "base_url": None,
                        "timeout_seconds": 60,
                    }
                },
                "model_policy": {
                    "default_profile": "profile-sk-test-secret",
                    "fallback_chain": ["profile-sk-test-secret"],
                },
            }, ensure_ascii=False), encoding="utf-8")
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.get("/api/memory/config")

            payload_text = json.dumps(response.json(), ensure_ascii=False)
            self.assertEqual(response.status_code, 200)
            self.assertNotIn("sk-test-secret", payload_text)
            self.assertIn("<redacted>", payload_text)

    def test_api_memory_config_restores_redacted_profile_names_when_saving(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            config_path = memory_dir / "config.json"
            config_path.write_text(json.dumps({
                "mode": "ai",
                "models": {
                    "profile-sk-test-secret": {
                        "provider": "openai",
                        "model": "gpt-4.1",
                        "api_key": "local-secret-key",
                        "api_key_env": None,
                        "base_url": None,
                        "timeout_seconds": 60,
                    }
                },
                "model_policy": {
                    "default_profile": "profile-sk-test-secret",
                    "fallback_chain": ["profile-sk-test-secret"],
                },
            }, ensure_ascii=False), encoding="utf-8")
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            payload = client.get("/api/memory/config").json()["config"]
            payload["mode"] = "rules"
            update_response = client.put("/api/memory/config", json={"config": payload})

            self.assertEqual(update_response.status_code, 200)
            self.assertNotIn("sk-test-secret", json.dumps(update_response.json(), ensure_ascii=False))
            saved = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["mode"], "rules")
            self.assertIn("profile-sk-test-secret", saved["models"])
            self.assertNotIn("profile-<redacted>", saved["models"])
            self.assertEqual(saved["model_policy"]["default_profile"], "profile-sk-test-secret")
            self.assertEqual(saved["model_policy"]["fallback_chain"], ["profile-sk-test-secret"])
            self.assertEqual(saved["models"]["profile-sk-test-secret"]["api_key"], "local-secret-key")

    def test_api_memory_config_preserves_profiles_with_colliding_redacted_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            config_path = memory_dir / "config.json"
            config_path.write_text(json.dumps({
                "mode": "ai",
                "models": {
                    "profile-sk-test-secret-a": {
                        "provider": "openai",
                        "model": "gpt-a",
                        "api_key": "secret-a",
                        "api_key_env": None,
                        "base_url": None,
                        "timeout_seconds": 60,
                    },
                    "profile-sk-test-secret-b": {
                        "provider": "anthropic",
                        "model": "claude-b",
                        "api_key": "secret-b",
                        "api_key_env": None,
                        "base_url": None,
                        "timeout_seconds": 60,
                    },
                },
                "model_policy": {
                    "default_profile": "profile-sk-test-secret-a",
                    "fallback_chain": ["profile-sk-test-secret-a", "profile-sk-test-secret-b"],
                },
            }, ensure_ascii=False), encoding="utf-8")
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            read_response = client.get("/api/memory/config")
            payload = read_response.json()["config"]
            payload["mode"] = "rules"
            update_response = client.put("/api/memory/config", json={"config": payload})

            payload_text = json.dumps(read_response.json(), ensure_ascii=False)
            self.assertEqual(read_response.status_code, 200)
            self.assertEqual(update_response.status_code, 200)
            self.assertNotIn("sk-test-secret", payload_text)
            self.assertIn("profile-<redacted>", payload["models"])
            self.assertIn("profile-<redacted>-2", payload["models"])
            saved = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["mode"], "rules")
            self.assertIn("profile-sk-test-secret-a", saved["models"])
            self.assertIn("profile-sk-test-secret-b", saved["models"])
            self.assertNotIn("profile-<redacted>", saved["models"])
            self.assertEqual(saved["model_policy"]["default_profile"], "profile-sk-test-secret-a")
            self.assertEqual(saved["model_policy"]["fallback_chain"], ["profile-sk-test-secret-a", "profile-sk-test-secret-b"])
            self.assertEqual(saved["models"]["profile-sk-test-secret-a"]["api_key"], "secret-a")
            self.assertEqual(saved["models"]["profile-sk-test-secret-b"]["api_key"], "secret-b")

    def test_api_memory_config_restores_redacted_path_values_when_saving(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            memory_dir.mkdir()
            secret_events = root / "events-sk-test-secret.jsonl"
            config_path = memory_dir / "config.json"
            config_path.write_text(json.dumps({
                "mode": "ai",
                "default_input": str(secret_events),
                "models": {
                    "primary": {
                        "provider": "openai",
                        "model": "gpt-4.1",
                        "api_key": "",
                        "api_key_env": None,
                        "base_url": None,
                        "timeout_seconds": 60,
                    }
                },
                "model_policy": {"default_profile": "primary", "fallback_chain": ["primary"]},
            }, ensure_ascii=False), encoding="utf-8")
            app = create_app(default_output_dir=root / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            payload = client.get("/api/memory/config").json()["config"]
            payload["mode"] = "rules"
            update_response = client.put("/api/memory/config", json={"config": payload})

            self.assertEqual(update_response.status_code, 200)
            self.assertNotIn("sk-test-secret", json.dumps(update_response.json(), ensure_ascii=False))
            self.assertIn("<redacted>", payload["default_input"])
            saved = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["mode"], "rules")
            self.assertEqual(saved["default_input"], str(secret_events))
            self.assertNotIn("<redacted>", saved["default_input"])

    def test_api_memory_config_rejects_invalid_model_runtime_settings_without_overwriting_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            config_path = memory_dir / "config.json"
            config_path.write_text(json.dumps({
                "mode": "ai",
                "models": {
                    "primary": {
                        "provider": "anthropic",
                        "model": "claude-sonnet-4-6",
                        "timeout_seconds": 60,
                    }
                },
                "model_policy": {"default_profile": "primary", "fallback_chain": ["primary"]},
            }, ensure_ascii=False), encoding="utf-8")
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            payload = client.get("/api/memory/config").json()["config"]
            payload["mode"] = "rules"
            payload["models"]["primary"]["timeout_seconds"] = 0
            payload["model_policy"]["retry"]["max_attempts"] = 0

            response = client.put("/api/memory/config", json={"config": payload})

            self.assertEqual(response.status_code, 400)
            self.assertIn("timeout_seconds", response.text)
            saved = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["mode"], "ai")
            self.assertEqual(saved["models"]["primary"]["timeout_seconds"], 60)

    def test_api_memory_config_rejects_unwritable_config_path_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            config_path = memory_dir / "config.json"
            config_path.mkdir()
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            payload = {
                "models": {
                    "primary": {
                        "provider": "openai",
                        "model": "gpt-4.1",
                        "api_key": "",
                        "api_key_env": None,
                        "base_url": None,
                        "timeout_seconds": 60,
                    }
                },
                "model_policy": {"default_profile": "primary", "fallback_chain": ["primary"]},
            }
            response = client.put("/api/memory/config", json={"config": payload})

            self.assertEqual(response.status_code, 409)
            self.assertIn("config path is not writable", response.json()["detail"])
            self.assertTrue(config_path.is_dir())
            self.assertFalse((memory_dir / ".config.json.tmp").exists())

    def test_api_memory_config_redacts_api_keys_without_overwriting_existing_secret(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            config_path = memory_dir / "config.json"
            config_path.write_text(json.dumps({
                "models": {
                    "primary": {
                        "provider": "openai",
                        "model": "gpt-4.1",
                        "api_key": "local-secret-key",
                        "api_key_env": None,
                        "base_url": None,
                        "timeout_seconds": 60,
                    }
                },
                "model_policy": {"default_profile": "primary", "fallback_chain": ["primary"]},
            }, ensure_ascii=False), encoding="utf-8")
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            read_response = client.get("/api/memory/config")

            self.assertEqual(read_response.status_code, 200)
            payload = read_response.json()["config"]
            self.assertEqual(payload["models"]["primary"]["api_key"], "")
            self.assertTrue(payload["models"]["primary"]["api_key_configured"])
            self.assertNotIn("local-secret-key", read_response.text)

            payload["mode"] = "rules"
            update_response = client.put("/api/memory/config", json={"config": payload})

            self.assertEqual(update_response.status_code, 200)
            self.assertNotIn("local-secret-key", update_response.text)
            saved = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["mode"], "rules")
            self.assertEqual(saved["models"]["primary"]["api_key"], "local-secret-key")

    def test_api_memory_config_preserves_secret_when_ui_posts_redacted_profile_without_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            config_path = memory_dir / "config.json"
            config_path.write_text(json.dumps({
                "models": {
                    "primary": {
                        "provider": "openai",
                        "model": "gpt-4.1",
                        "api_key": "local-secret-key",
                        "api_key_env": None,
                        "base_url": None,
                        "timeout_seconds": 60,
                    }
                },
                "model_policy": {"default_profile": "primary", "fallback_chain": ["primary"]},
            }, ensure_ascii=False), encoding="utf-8")
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            payload = client.get("/api/memory/config").json()["config"]
            payload["models"]["primary"].pop("api_key_configured")
            payload["mode"] = "rules"

            update_response = client.put("/api/memory/config", json={"config": payload})

            self.assertEqual(update_response.status_code, 200)
            saved = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["mode"], "rules")
            self.assertEqual(saved["models"]["primary"]["api_key"], "local-secret-key")

    def test_api_memory_config_persists_run_defaults_for_ui(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            payload = client.get("/api/memory/config").json()["config"]
            payload["default_input"] = "custom/events.jsonl"
            payload["default_project"] = "D:/work/project"
            payload["project_roots"] = ["D:/work/project", "D:/work/other"]
            payload["mode"] = "rules"
            payload["invoke_model"] = False
            payload["check_provider_profile"] = "primary"
            payload["check_provider_invoke"] = True
            payload["check_provider_all"] = True
            payload["export_limit"] = 7
            payload["review_candidates"] = "run/candidates.jsonl"
            payload["apply_reviewed"] = "run/reviewed.jsonl"
            payload["resume_run_id"] = "run_123"
            payload["trace_candidate_id"] = "cand_123"
            payload["summary_output"] = "summary.md"
            payload["eval_input"] = "labeled.jsonl"
            payload["eval_max_rows"] = 3
            payload["eval_max_attempts"] = 1
            payload["eval_continue_on_error"] = True
            payload["eval_fallback_rules_on_error"] = True
            payload["eval_fallback_rules_on_empty"] = True

            response = client.put("/api/memory/config", json={"config": payload})

            self.assertEqual(response.status_code, 200)
            saved = json.loads((memory_dir / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["default_input"], "custom/events.jsonl")
            self.assertEqual(saved["default_project"], "D:/work/project")
            self.assertEqual(saved["project_roots"], ["D:/work/project", "D:/work/other"])
            self.assertTrue(saved["check_provider_invoke"])
            self.assertTrue(saved["check_provider_all"])
            self.assertEqual(saved["export_limit"], 7)
            self.assertEqual(saved["review_candidates"], "run/candidates.jsonl")
            self.assertEqual(saved["apply_reviewed"], "run/reviewed.jsonl")
            self.assertEqual(saved["resume_run_id"], "run_123")
            self.assertEqual(saved["trace_candidate_id"], "cand_123")
            self.assertEqual(saved["summary_output"], "summary.md")
            self.assertEqual(saved["eval_input"], "labeled.jsonl")
            self.assertEqual(saved["eval_max_rows"], 3)
            self.assertEqual(saved["eval_max_attempts"], 1)
            self.assertTrue(saved["eval_continue_on_error"])
            self.assertTrue(saved["eval_fallback_rules_on_error"])
            self.assertTrue(saved["eval_fallback_rules_on_empty"])
            reloaded = client.get("/api/memory/config").json()["config"]
            self.assertEqual(reloaded["default_input"], "custom/events.jsonl")
            self.assertEqual(reloaded["default_project"], "D:/work/project")
            self.assertEqual(reloaded["project_roots"], ["D:/work/project", "D:/work/other"])
            self.assertTrue(reloaded["check_provider_invoke"])
            self.assertTrue(reloaded["check_provider_all"])
            self.assertEqual(reloaded["export_limit"], 7)
            self.assertEqual(reloaded["review_candidates"], "run/candidates.jsonl")
            self.assertEqual(reloaded["apply_reviewed"], "run/reviewed.jsonl")
            self.assertEqual(reloaded["resume_run_id"], "run_123")
            self.assertEqual(reloaded["trace_candidate_id"], "cand_123")
            self.assertEqual(reloaded["summary_output"], "summary.md")
            self.assertEqual(reloaded["eval_input"], "labeled.jsonl")
            self.assertEqual(reloaded["eval_max_rows"], 3)
            self.assertEqual(reloaded["eval_max_attempts"], 1)
            self.assertTrue(reloaded["eval_continue_on_error"])
            self.assertTrue(reloaded["eval_fallback_rules_on_error"])
            self.assertTrue(reloaded["eval_fallback_rules_on_empty"])

    def test_memory_review_page_loads_start_defaults_from_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.get("/memory-review")

            self.assertEqual(response.status_code, 200)
            self.assertIn("loadStartDefaults", response.text)
            self.assertIn("/api/memory/config", response.text)
            self.assertIn("startMode", response.text)

    def test_api_memory_candidates_reads_candidates_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            (memory_dir / "candidates.jsonl").write_text(
                json.dumps({
                    "id": "cand_1",
                    "type": "preference",
                    "scope": "global",
                    "project": None,
                    "content": "用户偏好中文回答。",
                    "score": 0.9,
                    "status": "promote",
                    "evidence": [{"source": "claude_code", "session_id": "s1"}],
                    "tags": ["preference"],
                }, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.get("/api/memory/candidates")

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["count"], 1)
            self.assertEqual(payload["candidates"][0]["id"], "cand_1")

    def test_api_memory_candidates_redacts_sensitive_legacy_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            (memory_dir / "candidates.jsonl").write_text(
                json.dumps({
                    "id": "cand_1",
                    "type": "workflow",
                    "scope": "project",
                    "project": "/tmp/project",
                    "content": "项目的 API key 在 key.txt 文件中。",
                    "score": 0.9,
                    "status": "promote",
                    "evidence": [{"event_id": "event_1", "quote": "项目的 API key 在 key.txt 文件中。"}],
                }, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.get("/api/memory/candidates")

            self.assertEqual(response.status_code, 200)
            payload_text = json.dumps(response.json(), ensure_ascii=False)
            self.assertNotIn("API key", payload_text)
            self.assertNotIn("key.txt", payload_text)
            self.assertIn("<redacted>", payload_text)

    def test_api_memory_candidates_redacts_sensitive_paths_in_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory-sk-test-secret"
            memory_dir.mkdir()
            (memory_dir / "candidates.jsonl").write_text(
                json.dumps({"id": "cand_1", "content": "用户偏好中文。"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.get("/api/memory/candidates")

            payload_text = json.dumps(response.json(), ensure_ascii=False)
            self.assertEqual(response.status_code, 200)
            self.assertNotIn("sk-test-secret", payload_text)
            self.assertIn("<redacted>", payload_text)

    def test_api_memory_candidates_rejects_malformed_candidates_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            (memory_dir / "candidates.jsonl").write_text(
                json.dumps({"id": "cand_1", "content": "用户偏好中文。"}, ensure_ascii=False)
                + "\n{not json\n",
                encoding="utf-8",
            )
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.get("/api/memory/candidates")

            self.assertEqual(response.status_code, 400)
            self.assertIn("invalid candidates JSON", response.json()["detail"])

    def test_api_memory_review_writes_reviewed_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.post(
                "/api/memory/review",
                json={
                    "candidate_id": "cand_1",
                    "action": "approved",
                    "edited_content": "用户偏好中文回答。",
                    "reviewer": "user",
                    "note": "明确偏好",
                    "candidate": {"id": "cand_1", "content": "用户偏好中文回答。", "evidence": [{"event_id": "event_1"}]},
                },
            )

            self.assertEqual(response.status_code, 200)
            reviewed = memory_dir / "reviewed.jsonl"
            self.assertTrue(reviewed.exists())
            rows = [json.loads(line) for line in reviewed.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["action"], "approved")
            self.assertEqual(rows[0]["candidate_id"], "cand_1")

    def test_api_memory_review_rejects_malformed_existing_reviewed_without_appending(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            reviewed_path = memory_dir / "reviewed.jsonl"
            original_reviewed = json.dumps({"candidate_id": "old", "action": "rejected"}, ensure_ascii=False) + "\n{not json\n"
            reviewed_path.write_text(original_reviewed, encoding="utf-8")
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.post(
                "/api/memory/review",
                json={
                    "candidate_id": "cand_1",
                    "action": "rejected",
                    "reviewer": "user",
                    "candidate": {"id": "cand_1", "content": "用户偏好中文回答。"},
                },
            )

            self.assertEqual(response.status_code, 400)
            self.assertIn("invalid reviewed decisions JSON", response.json()["detail"])
            self.assertEqual(reviewed_path.read_text(encoding="utf-8"), original_reviewed)

    def test_api_memory_review_redacts_sensitive_reviewed_path_in_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory-sk-test-secret"
            memory_dir.mkdir()
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.post(
                "/api/memory/review",
                json={
                    "candidate_id": "cand_1",
                    "action": "approved",
                    "edited_content": "用户偏好中文回答。",
                    "reviewer": "user",
                    "note": "明确偏好",
                    "candidate": {"id": "cand_1", "content": "用户偏好中文回答。", "evidence": [{"event_id": "event_1"}]},
                },
            )

            payload_text = json.dumps(response.json(), ensure_ascii=False)
            self.assertEqual(response.status_code, 200)
            self.assertTrue((memory_dir / "reviewed.jsonl").exists())
            self.assertNotIn("sk-test-secret", payload_text)
            self.assertIn("<redacted>", payload_text)

    def test_api_memory_review_rejects_invalid_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.post(
                "/api/memory/review",
                json={"candidate_id": "cand_1", "action": "bad", "candidate": {}},
            )

            self.assertEqual(response.status_code, 400)

    def test_api_memory_review_rejects_approval_without_memory_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.post(
                "/api/memory/review",
                json={"candidate_id": "cand_1", "action": "approved", "candidate": {}},
            )

            self.assertEqual(response.status_code, 400)
            self.assertIn("memory content", response.json()["detail"])
            self.assertFalse((memory_dir / "reviewed.jsonl").exists())

    def test_api_memory_review_rejects_approval_without_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.post(
                "/api/memory/review",
                json={"candidate_id": "cand_1", "action": "approved", "candidate": {"id": "cand_1", "content": "用户偏好中文回答。"}},
            )

            self.assertEqual(response.status_code, 400)
            self.assertIn("evidence", response.json()["detail"])
            self.assertFalse((memory_dir / "reviewed.jsonl").exists())

    def test_api_memory_review_rejects_sensitive_approval_without_writing_reviewed_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.post(
                "/api/memory/review",
                json={
                    "candidate_id": "cand_1",
                    "action": "approved",
                    "edited_content": "OPENAI_API_KEY=sk-test-secret",
                    "candidate": {"id": "cand_1", "content": "用户偏好中文回答。", "evidence": [{"event_id": "event_1"}]},
                },
            )

            self.assertEqual(response.status_code, 400)
            self.assertIn("sensitive", response.json()["detail"])
            self.assertNotIn("sk-test-secret", response.text)
            self.assertFalse((memory_dir / "reviewed.jsonl").exists())

    def test_api_memory_run_review_rejects_approval_without_memory_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.post(
                f"/api/memory/runs/{state['run_id']}/review",
                json={"candidate_id": "cand_1", "action": "approved", "candidate": {}},
            )

            self.assertEqual(response.status_code, 400)
            self.assertIn("memory content", response.json()["detail"])
            self.assertFalse((Path(state["run_dir"]) / "reviewed.jsonl").exists())

    def test_api_memory_run_review_rejects_approval_without_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.post(
                f"/api/memory/runs/{state['run_id']}/review",
                json={"candidate_id": "cand_1", "action": "approved", "candidate": {"id": "cand_1", "content": "用户偏好中文回答。"}},
            )

            self.assertEqual(response.status_code, 400)
            self.assertIn("evidence", response.json()["detail"])
            self.assertFalse((Path(state["run_dir"]) / "reviewed.jsonl").exists())

    def test_api_memory_run_review_rejects_sensitive_candidate_payload_without_writing_reviewed_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.post(
                f"/api/memory/runs/{state['run_id']}/review",
                json={
                    "candidate_id": "cand_1",
                    "action": "approved",
                    "candidate": {
                        "id": "cand_1",
                        "content": "用户偏好中文回答。",
                        "evidence": [{"event_id": "event_1", "quote": "Bearer sk-test-secret"}],
                    },
                },
            )

            self.assertEqual(response.status_code, 400)
            self.assertIn("sensitive", response.json()["detail"])
            self.assertNotIn("sk-test-secret", response.text)
            self.assertFalse((Path(state["run_dir"]) / "reviewed.jsonl").exists())

    def test_api_memory_review_rejects_sensitive_rejection_payload_without_writing_reviewed_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.post(
                "/api/memory/review",
                json={
                    "candidate_id": "cand_1",
                    "action": "rejected",
                    "note": "不要记住 sk-test-secret",
                    "candidate": {"id": "cand_1", "content": "临时密钥 Bearer sk-test-secret"},
                },
            )

            self.assertEqual(response.status_code, 400)
            self.assertIn("sensitive", response.json()["detail"])
            self.assertNotIn("sk-test-secret", response.text)
            self.assertFalse((memory_dir / "reviewed.jsonl").exists())

    def test_api_memory_review_rejects_credential_location_without_writing_reviewed_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.post(
                "/api/memory/review",
                json={
                    "candidate_id": "cand_1",
                    "action": "approved",
                    "edited_content": "项目的 API key 在 key.txt 文件中。",
                    "candidate": {"id": "cand_1", "content": "项目的 API key 在 key.txt 文件中。", "evidence": [{"event_id": "event_1"}]},
                },
            )

            self.assertEqual(response.status_code, 400)
            self.assertIn("sensitive", response.json()["detail"])
            self.assertNotIn("key.txt", response.text)
            self.assertFalse((memory_dir / "reviewed.jsonl").exists())

    def test_api_memory_review_writes_apply_compatible_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.post(
                "/api/memory/review",
                json={
                    "candidate_id": "cand_1",
                    "action": "approved",
                    "edited_content": "用户偏好中文回答。",
                    "reviewer": "user",
                    "note": "明确偏好",
                    "candidate": {"id": "cand_1", "type": "preference", "scope": "user", "project": None, "content": "用户偏好中文回答。", "evidence": [{"event_id": "event_1"}], "tags": ["language"]},
                },
            )

            self.assertEqual(response.status_code, 200)
            rows = [json.loads(line) for line in (memory_dir / "reviewed.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["status"], "approved")
            self.assertEqual(rows[0]["memory_updates"][0]["summary"], "用户偏好中文回答。")
            self.assertEqual(rows[0]["memory_updates"][0]["evidence_refs"], ["event_1"])

    def test_api_memory_runs_lists_and_reads_run_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="ai", model="anthropic:test", invoke_model=False)
            update_run_state(state, status="waiting_review", phase="review")
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            list_response = client.get("/api/memory/runs")
            read_response = client.get(f"/api/memory/runs/{state['run_id']}")

            self.assertEqual(list_response.status_code, 200)
            self.assertEqual(read_response.status_code, 200)
            self.assertEqual(list_response.json()["runs"][0]["run_id"], state["run_id"])
            self.assertEqual(read_response.json()["run_id"], state["run_id"])

    def test_api_memory_runs_redacts_sensitive_memory_dir_in_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory-sk-test-secret"
            create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.get("/api/memory/runs")

            payload_text = json.dumps(response.json(), ensure_ascii=False)
            self.assertEqual(response.status_code, 200)
            self.assertNotIn("sk-test-secret", payload_text)
            self.assertIn("<redacted>", payload_text)

    def test_api_memory_runs_surfaces_malformed_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            Path(state["run_dir"], "state.json").write_text("{not json\n", encoding="utf-8")
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.get("/api/memory/runs")

            payload = response.json()
            self.assertEqual(response.status_code, 200)
            self.assertEqual(payload["runs"][0]["run_id"], state["run_id"])
            self.assertEqual(payload["runs"][0]["status"], "invalid")
            self.assertEqual(payload["runs"][0]["phase"], "invalid_state")
            self.assertIn("invalid run state JSON", payload["runs"][0]["error"])

    def test_api_memory_run_state_redacts_legacy_error_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="ai", model="anthropic:test", invoke_model=False)
            update_run_state(
                state,
                status="failed",
                phase="failed",
                error="provider failed api_key=sk-test-secret Bearer abc123 at https://user:pass123@example.test/v1",
            )
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            list_response = client.get("/api/memory/runs")
            read_response = client.get(f"/api/memory/runs/{state['run_id']}")

            self.assertEqual(list_response.status_code, 200)
            self.assertEqual(read_response.status_code, 200)
            serialized = json.dumps({"list": list_response.json(), "read": read_response.json()})
            self.assertNotIn("sk-test-secret", serialized)
            self.assertNotIn("Bearer abc123", serialized)
            self.assertNotIn("pass123", serialized)
            self.assertIn("<redacted>", serialized)

    def test_api_memory_run_trace_returns_trace_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="ai", model="anthropic:test", invoke_model=False)
            append_trace(state, "candidate_ready", {"candidate_id": "cand_1"})
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.get(f"/api/memory/runs/{state['run_id']}/trace", params={"candidate_id": "cand_1"})

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["trace"][0]["payload"]["candidate_id"], "cand_1")

    def test_api_memory_run_trace_redacts_legacy_error_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="ai", model="anthropic:test", invoke_model=False)
            append_trace(
                state,
                "run_failed",
                {"error": "provider failed api_key=sk-test-secret Bearer abc123 at https://user:pass123@example.test/v1"},
            )
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.get(f"/api/memory/runs/{state['run_id']}/trace")

            self.assertEqual(response.status_code, 200)
            serialized = json.dumps(response.json())
            self.assertNotIn("sk-test-secret", serialized)
            self.assertNotIn("Bearer abc123", serialized)
            self.assertNotIn("pass123", serialized)
            self.assertIn("<redacted>", serialized)

    def test_api_memory_run_trace_rejects_invalid_candidate_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="ai", model="anthropic:test", invoke_model=False)
            append_trace(state, "candidate_ready", {"candidate_id": "cand_1"})
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.get(f"/api/memory/runs/{state['run_id']}/trace", params={"candidate_id": "../state"})

            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json()["detail"], "invalid candidate_id")

    def test_api_memory_run_trace_rejects_malformed_trace_without_silent_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="ai", model="anthropic:test", invoke_model=False)
            trace_path = Path(state["run_dir"]) / "trace.jsonl"
            trace_path.write_text(
                json.dumps({"run_id": state["run_id"], "event_type": "run_created", "payload": {}}, ensure_ascii=False)
                + "\nnot-json\n",
                encoding="utf-8",
            )
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.get(f"/api/memory/runs/{state['run_id']}/trace")

            self.assertEqual(response.status_code, 400)
            self.assertIn("invalid trace JSON", response.json()["detail"])

    def test_api_memory_run_trace_rejects_non_object_payload_without_silent_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="ai", model="anthropic:test", invoke_model=False)
            trace_path = Path(state["run_dir"]) / "trace.jsonl"
            trace_path.write_text(
                json.dumps({"run_id": state["run_id"], "event_type": "candidate_ready", "payload": "bad-payload"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.get(f"/api/memory/runs/{state['run_id']}/trace", params={"candidate_id": "cand_1"})

            self.assertEqual(response.status_code, 400)
            self.assertIn("invalid trace payload", response.json()["detail"])

    def test_api_memory_run_routes_reject_invalid_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            state_response = client.get("/api/memory/runs/bad.run")
            trace_response = client.get("/api/memory/runs/bad.run/trace")
            review_response = client.post("/api/memory/runs/bad.run/review", json={"candidate_id": "cand_1", "action": "approved", "candidate": {}})

            self.assertEqual(state_response.status_code, 400)
            self.assertEqual(trace_response.status_code, 400)
            self.assertEqual(review_response.status_code, 400)
            self.assertEqual(state_response.json()["detail"], "invalid run_id")

    def test_api_memory_run_state_reports_corrupt_state_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            (Path(state["run_dir"]) / "state.json").write_text("{not json", encoding="utf-8")
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.get(f"/api/memory/runs/{state['run_id']}")

            self.assertEqual(response.status_code, 400)
            self.assertIn("run state invalid", response.json()["detail"])

    def test_api_memory_run_state_reports_invalid_state_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            (Path(state["run_dir"]) / "state.json").write_text("[]", encoding="utf-8")
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.get(f"/api/memory/runs/{state['run_id']}")

            self.assertEqual(response.status_code, 400)
            self.assertIn("run state invalid", response.json()["detail"])

    def test_api_memory_run_state_redacts_sensitive_path_from_invalid_state_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory-sk-test-secret"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            (Path(state["run_dir"]) / "state.json").write_text("[]", encoding="utf-8")
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.get(f"/api/memory/runs/{state['run_id']}")

            payload_text = json.dumps(response.json(), ensure_ascii=False)
            self.assertEqual(response.status_code, 400)
            self.assertIn("run state invalid", response.json()["detail"])
            self.assertNotIn("sk-test-secret", payload_text)
            self.assertIn("<redacted>", payload_text)

    def test_api_memory_run_trace_redacts_sensitive_path_from_malformed_trace_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory-sk-test-secret"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="ai", model="anthropic:test", invoke_model=False)
            trace_path = Path(state["run_dir"]) / "trace.jsonl"
            trace_path.write_text("{not json\n", encoding="utf-8")
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.get(f"/api/memory/runs/{state['run_id']}/trace")

            payload_text = json.dumps(response.json(), ensure_ascii=False)
            self.assertEqual(response.status_code, 400)
            self.assertIn("invalid trace JSON", response.json()["detail"])
            self.assertNotIn("sk-test-secret", payload_text)
            self.assertIn("<redacted>", payload_text)

    def test_api_memory_run_review_writes_run_scoped_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="ai", model="anthropic:test", invoke_model=False)
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.post(
                f"/api/memory/runs/{state['run_id']}/review",
                json={
                    "candidate_id": "cand_1",
                    "action": "approved",
                    "edited_content": "用户偏好中文回答。",
                    "reviewer": "user",
                    "candidate": {"id": "cand_1", "type": "preference", "scope": "user", "content": "用户偏好中文回答。", "evidence": [{"event_id": "event_1"}]},
                },
            )

            reviewed_path = Path(state["run_dir"]) / "reviewed.jsonl"
            self.assertEqual(response.status_code, 200)
            self.assertTrue(reviewed_path.exists())
            self.assertIn("用户偏好中文回答", reviewed_path.read_text(encoding="utf-8"))

    def test_api_memory_run_review_rejects_malformed_existing_reviewed_without_appending(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="ai", model="anthropic:test", invoke_model=False)
            run_dir = Path(state["run_dir"])
            queue_path = run_dir / "review_queue.jsonl"
            queue_path.write_text(json.dumps({
                "candidate_id": "cand_1",
                "suggested_action": "review",
                "candidate": {"id": "cand_1", "type": "preference", "scope": "user", "content": "用户偏好中文。"},
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            reviewed_path = run_dir / "reviewed.jsonl"
            original_reviewed = json.dumps({"candidate_id": "old", "action": "rejected"}, ensure_ascii=False) + "\n{not json\n"
            reviewed_path.write_text(original_reviewed, encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(queue_path)})
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.post(
                f"/api/memory/runs/{state['run_id']}/review",
                json={
                    "candidate_id": "cand_1",
                    "action": "rejected",
                    "reviewer": "user",
                    "candidate": {"id": "cand_1", "type": "preference", "scope": "user", "content": "用户偏好中文。"},
                },
            )

            self.assertEqual(response.status_code, 400)
            self.assertIn("invalid reviewed decisions JSON", response.json()["detail"])
            self.assertEqual(reviewed_path.read_text(encoding="utf-8"), original_reviewed)

    def test_api_memory_run_review_redacts_sensitive_reviewed_path_in_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory-sk-test-secret"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="ai", model="anthropic:test", invoke_model=False)
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.post(
                f"/api/memory/runs/{state['run_id']}/review",
                json={
                    "candidate_id": "cand_1",
                    "action": "approved",
                    "edited_content": "用户偏好中文回答。",
                    "reviewer": "user",
                    "candidate": {"id": "cand_1", "type": "preference", "scope": "user", "content": "用户偏好中文回答。", "evidence": [{"event_id": "event_1"}]},
                },
            )

            payload_text = json.dumps(response.json(), ensure_ascii=False)
            self.assertEqual(response.status_code, 200)
            self.assertTrue((Path(state["run_dir"]) / "reviewed.jsonl").exists())
            self.assertNotIn("sk-test-secret", payload_text)
            self.assertIn("<redacted>", payload_text)

    def test_api_memory_run_resume_redacts_sensitive_legacy_state_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            reviewed_path = Path(state["run_dir"]) / "reviewed.jsonl"
            reviewed_path.write_text(
                json.dumps({"candidate_id": "cand_1", "status": "rejected"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            state_path = Path(state["run_dir"]) / "state.json"
            state["diagnostics"] = {
                "provider_error": "provider failed with OPENAI_API_KEY=sk-test-secret",
                "next_hint": "token 在 config.yaml 配置中",
            }
            state_path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.post(f"/api/memory/runs/{state['run_id']}/resume")

            self.assertEqual(response.status_code, 200)
            payload_text = json.dumps(response.json(), ensure_ascii=False)
            self.assertNotIn("sk-test-secret", payload_text)
            self.assertNotIn("config.yaml", payload_text)
            self.assertIn("<redacted>", payload_text)

    def test_api_memory_run_review_rejects_candidate_not_in_run_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="ai", model="anthropic:test", invoke_model=False)
            run_dir = Path(state["run_dir"])
            queue_path = run_dir / "review_queue.jsonl"
            queue_path.write_text(json.dumps({
                "candidate_id": "cand_1",
                "suggested_action": "create",
                "candidate": {"id": "cand_1", "type": "preference", "scope": "user", "content": "用户偏好中文。"},
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(queue_path)})
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.post(
                f"/api/memory/runs/{state['run_id']}/review",
                json={
                    "candidate_id": "ghost",
                    "action": "rejected",
                    "reviewer": "user",
                    "candidate": {"id": "ghost", "type": "preference", "scope": "user", "content": "队列外候选。"},
                },
            )

            self.assertEqual(response.status_code, 400)
            self.assertIn("candidate not found in run review queue", response.json()["detail"])
            self.assertFalse((run_dir / "reviewed.jsonl").exists())

    def test_api_memory_run_review_rejects_mismatched_candidate_payload_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="ai", model="anthropic:test", invoke_model=False)
            run_dir = Path(state["run_dir"])
            queue_path = run_dir / "review_queue.jsonl"
            queue_path.write_text(json.dumps({
                "candidate_id": "cand_1",
                "suggested_action": "create",
                "candidate": {"id": "cand_1", "type": "preference", "scope": "user", "content": "用户偏好中文。", "evidence": [{"event_id": "event_1"}]},
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(queue_path)})
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.post(
                f"/api/memory/runs/{state['run_id']}/review",
                json={
                    "candidate_id": "cand_1",
                    "action": "approved",
                    "reviewer": "user",
                    "candidate": {"id": "ghost", "type": "preference", "scope": "user", "content": "错位候选。", "evidence": [{"event_id": "event_1"}]},
                },
            )

            self.assertEqual(response.status_code, 400)
            self.assertIn("candidate payload id does not match review candidate", response.json()["detail"])
            self.assertFalse((run_dir / "reviewed.jsonl").exists())

    def test_memory_review_page_contains_run_status_ui(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.get("/memory-review")

            self.assertEqual(response.status_code, 200)
            self.assertIn("运行状态", response.text)
            self.assertIn("loadRuns", response.text)
            self.assertIn("setInterval(loadRuns", response.text)

    def test_api_memory_run_start_creates_background_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            events = root / "events.jsonl"
            events.write_text(json.dumps({"event_id": "event_1", "source": "codex", "role": "user", "event_type": "history_prompt", "project": str(root), "content": "这个项目需要人工审核后才能写正式记忆"}, ensure_ascii=False) + "\n", encoding="utf-8")
            app = create_app(default_output_dir=root / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.post("/api/memory/runs/start", json={"input": str(events), "project": str(root), "mode": "rules", "invoke_model": False})

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertIn("run_id", payload)
            self.assertEqual(payload["status"], "queued")
            state_path = memory_dir / "runs" / payload["run_id"] / "state.json"
            self.assertTrue(state_path.exists())
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertIn(state["status"], {"running", "waiting_review"})

    def test_api_memory_run_start_redacts_sensitive_paths_in_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory-sk-test-secret"
            events = root / "events.jsonl"
            events.write_text(json.dumps({"event_id": "event_1", "source": "codex", "role": "user", "event_type": "history_prompt", "project": str(root), "content": "用户偏好中文回答。"}, ensure_ascii=False) + "\n", encoding="utf-8")
            app = create_app(default_output_dir=root / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.post("/api/memory/runs/start", json={"input": str(events), "project": str(root), "mode": "rules", "invoke_model": False})

            payload = response.json()
            payload_text = json.dumps(payload, ensure_ascii=False)
            self.assertEqual(response.status_code, 200)
            self.assertTrue((memory_dir / "runs" / payload["run_id"] / "state.json").exists())
            self.assertNotIn("sk-test-secret", payload_text)
            self.assertIn("<redacted>", payload_text)

    def test_api_memory_run_start_redacts_sensitive_missing_input_path_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            missing_events = root / "events-sk-test-secret.jsonl"
            app = create_app(default_output_dir=root / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.post("/api/memory/runs/start", json={"input": str(missing_events), "project": str(root), "mode": "rules", "invoke_model": False})

            payload_text = json.dumps(response.json(), ensure_ascii=False)
            self.assertEqual(response.status_code, 400)
            self.assertIn("run input not found", response.json()["detail"])
            self.assertNotIn("sk-test-secret", payload_text)
            self.assertIn("<redacted>", payload_text)

    def test_api_memory_run_start_redacts_sensitive_empty_input_path_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            empty_events = root / "events-sk-test-secret.jsonl"
            empty_events.write_text("\n\n", encoding="utf-8")
            app = create_app(default_output_dir=root / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.post("/api/memory/runs/start", json={"input": str(empty_events), "project": str(root), "mode": "rules", "invoke_model": False})

            payload_text = json.dumps(response.json(), ensure_ascii=False)
            self.assertEqual(response.status_code, 400)
            self.assertIn("no valid events", response.json()["detail"])
            self.assertNotIn("sk-test-secret", payload_text)
            self.assertIn("<redacted>", payload_text)

    def test_api_memory_run_start_uses_memory_config_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            memory_dir.mkdir()
            (memory_dir / "config.json").write_text(
                json.dumps(
                    {
                        "models": {
                            "primary": {
                                "provider": "openai",
                                "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
                                "api_key": "local-key",
                                "api_key_env": None,
                                "base_url": "http://localhost:3000",
                                "timeout_seconds": 60,
                            }
                        },
                        "model_policy": {"default_profile": "primary", "fallback_chain": ["primary"]},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            events = root / "events.jsonl"
            events.write_text(json.dumps({"event_id": "event_1", "source": "codex", "role": "user", "event_type": "history_prompt", "project": str(root), "content": "这个项目需要人工审核后才能写正式记忆"}, ensure_ascii=False) + "\n", encoding="utf-8")
            app = create_app(default_output_dir=root / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.post("/api/memory/runs/start", json={"input": str(events), "project": ".", "mode": "rules", "invoke_model": False})

            self.assertEqual(response.status_code, 200)
            state = json.loads((memory_dir / "runs" / response.json()["run_id"] / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["model"], "openai:nvidia/nemotron-3-ultra-550b-a55b:free")

    def test_api_memory_run_start_rejects_missing_input_without_creating_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            app = create_app(default_output_dir=root / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.post("/api/memory/runs/start", json={"input": str(root / "missing.jsonl"), "project": ".", "mode": "rules", "invoke_model": False})

            self.assertEqual(response.status_code, 400)
            self.assertIn("run input not found", response.json()["detail"])
            self.assertFalse((memory_dir / "runs").exists())

    def test_api_memory_run_start_rejects_empty_input_without_creating_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            events = root / "events.jsonl"
            events.write_text("\n\n", encoding="utf-8")
            app = create_app(default_output_dir=root / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.post("/api/memory/runs/start", json={"input": str(events), "project": ".", "mode": "rules", "invoke_model": False})

            self.assertEqual(response.status_code, 400)
            self.assertIn("run input has no valid events", response.json()["detail"])
            self.assertFalse((memory_dir / "runs").exists())

    def test_api_memory_run_start_rejects_malformed_input_without_creating_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            events = root / "events.jsonl"
            events.write_text(json.dumps({"event_id": "event_1", "source": "codex", "role": "user", "content": "用户偏好中文回答。"}, ensure_ascii=False) + "\nnot-json\n", encoding="utf-8")
            app = create_app(default_output_dir=root / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.post("/api/memory/runs/start", json={"input": str(events), "project": ".", "mode": "rules", "invoke_model": False})

            self.assertEqual(response.status_code, 400)
            self.assertIn("invalid run input JSON", response.json()["detail"])
            self.assertFalse((memory_dir / "runs").exists())

    def test_api_memory_run_start_records_background_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            events = root / "events.jsonl"
            events.write_text(json.dumps({"event_id": "event_1", "source": "codex", "role": "user", "event_type": "history_prompt", "project": str(root), "content": "这个项目需要人工审核后才能写正式记忆"}, ensure_ascii=False) + "\n", encoding="utf-8")
            app = create_app(default_output_dir=root / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            with patch("dream_memory.web._run_dream_to_review", side_effect=RuntimeError("boom api_key=sk-test-secret Bearer abc123")):
                response = client.post("/api/memory/runs/start", json={"input": str(events), "project": ".", "mode": "ai", "invoke_model": True})

            self.assertEqual(response.status_code, 200)
            state = json.loads((memory_dir / "runs" / response.json()["run_id"] / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["phase"], "failed")
            self.assertIn("boom", state["error"])
            self.assertNotIn("sk-test-secret", state["error"])
            self.assertNotIn("Bearer abc123", state["error"])
            self.assertIn("<redacted>", state["error"])
            trace = (memory_dir / "runs" / response.json()["run_id"] / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("run_failed", trace)
            self.assertNotIn("sk-test-secret", trace)
            self.assertNotIn("Bearer abc123", trace)
            self.assertIn("<redacted>", trace)

    def test_api_memory_run_resume_applies_run_scoped_reviews(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            events = root / "events.jsonl"
            events.write_text(json.dumps({"event_id": "event_1", "source": "codex", "role": "user", "event_type": "history_prompt", "project": str(root), "content": "这个项目需要人工审核后才能写正式记忆"}, ensure_ascii=False) + "\n", encoding="utf-8")
            app = create_app(default_output_dir=root / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)
            start = client.post("/api/memory/runs/start", json={"input": str(events), "project": str(root), "mode": "rules", "invoke_model": False}).json()
            state = json.loads((memory_dir / "runs" / start["run_id"] / "state.json").read_text(encoding="utf-8"))
            candidate = json.loads(Path(state["artifacts"]["candidates_path"]).read_text(encoding="utf-8").splitlines()[0])
            review_response = client.post(
                f"/api/memory/runs/{start['run_id']}/review",
                json={"candidate_id": candidate["id"], "action": "approved", "edited_content": candidate["content"], "reviewer": "user", "candidate": candidate},
            )
            self.assertEqual(review_response.status_code, 200)

            resume_response = client.post(f"/api/memory/runs/{start['run_id']}/resume")

            self.assertEqual(resume_response.status_code, 200)
            self.assertEqual(resume_response.json()["status"], "completed")
            self.assertTrue((memory_dir / "MEMORY.md").exists())

    def test_api_memory_run_resume_requires_reviewed_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            update_run_state(state, status="waiting_review", phase="review")
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.post(f"/api/memory/runs/{state['run_id']}/resume")

            self.assertEqual(response.status_code, 400)
            self.assertIn("reviewed decisions not found", response.json()["detail"])
            updated = json.loads((Path(state["run_dir"]) / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(updated["status"], "waiting_review")
            self.assertFalse((memory_dir / "MEMORY.md").exists())

    def test_api_memory_run_resume_rejects_approved_decision_without_memory_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            update_run_state(state, status="waiting_review", phase="review")
            reviewed_path = Path(state["run_dir"]) / "reviewed.jsonl"
            reviewed_path.write_text(
                json.dumps({"candidate_id": "cand_1", "status": "approved", "memory_updates": []}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.post(f"/api/memory/runs/{state['run_id']}/resume")

            self.assertEqual(response.status_code, 400)
            self.assertIn("approved decisions have no memory updates", response.json()["detail"])
            updated = json.loads((Path(state["run_dir"]) / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(updated["status"], "waiting_review")
            self.assertFalse((memory_dir / "MEMORY.md").exists())

    def test_api_memory_run_resume_rejects_incomplete_approved_memory_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            update_run_state(state, status="waiting_review", phase="review")
            reviewed_path = Path(state["run_dir"]) / "reviewed.jsonl"
            reviewed_path.write_text(
                json.dumps({"candidate_id": "cand_1", "status": "approved", "memory_updates": [{"id": "mem_1"}]}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.post(f"/api/memory/runs/{state['run_id']}/resume")

            self.assertEqual(response.status_code, 400)
            self.assertIn("incomplete memory update for cand_1", response.json()["detail"])
            updated = json.loads((Path(state["run_dir"]) / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(updated["status"], "waiting_review")
            self.assertFalse((memory_dir / "MEMORY.md").exists())

    def test_api_memory_run_resume_rejects_malformed_memory_cards_without_writing_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            update_run_state(state, status="waiting_review", phase="review")
            reviewed_path = Path(state["run_dir"]) / "reviewed.jsonl"
            reviewed_path.write_text(json.dumps({
                "candidate_id": "cand_1",
                "status": "approved",
                "memory_updates": [{
                    "id": "mem_1",
                    "scope": "user",
                    "memory_type": "preference",
                    "summary": "用户偏好中文回答。",
                    "evidence_refs": ["event_1"],
                    "approved_by": "user",
                    "approved_at": "2026-07-05T00:00:00Z",
                    "status": "active",
                    "retrieval_hints": [],
                }],
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            memory_cards = memory_dir / "memory_cards.jsonl"
            memory_cards.parent.mkdir(parents=True, exist_ok=True)
            memory_cards.write_text(
                json.dumps({"id": "existing", "scope": "user", "memory_type": "preference", "summary": "已有记忆。", "status": "active"}, ensure_ascii=False)
                + "\n{not json\n",
                encoding="utf-8",
            )
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.post(f"/api/memory/runs/{state['run_id']}/resume")

            self.assertEqual(response.status_code, 400)
            self.assertIn("invalid memory cards JSON", response.json()["detail"])
            updated = json.loads((Path(state["run_dir"]) / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(updated["status"], "waiting_review")
            self.assertFalse((memory_dir / "MEMORY.md").exists())
            self.assertIn("{not json", memory_cards.read_text(encoding="utf-8"))

    def test_api_memory_run_auto_review_preview_reports_skip_reasons_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            queue_path = Path(state["run_dir"]) / "review_queue.jsonl"
            queue_path.write_text("\n".join([
                json.dumps({"candidate_id": "create_1", "suggested_action": "create", "candidate": {"id": "create_1", "type": "workflow", "scope": "user", "content": "用户偏好直接推进。", "evidence": [{"event_id": "event_1"}]}, "dream_analysis": {"dream_score": 0.82}}, ensure_ascii=False),
                json.dumps({"candidate_id": "low_1", "suggested_action": "create", "candidate": {"id": "low_1", "type": "workflow", "scope": "user", "content": "低分候选", "evidence": [{"event_id": "event_2"}]}, "dream_analysis": {"dream_score": 0.4}}, ensure_ascii=False),
            ]) + "\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(queue_path)})
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.post(f"/api/memory/runs/{state['run_id']}/auto-review/preview", json={"min_score": 0.7})

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertTrue(payload["dry_run"])
            self.assertEqual(payload["decision_count"], 1)
            self.assertEqual(payload["skip_reasons"]["below_min_score"], 1)
            self.assertEqual(payload["preview"][0]["decision"], "approved")
            self.assertFalse((Path(state["run_dir"]) / "reviewed.jsonl").exists())

    def test_api_memory_run_auto_review_requires_review_queue_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            update_run_state(state, status="waiting_review", phase="review")
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            preview_response = client.post(f"/api/memory/runs/{state['run_id']}/auto-review/preview", json={"min_score": 0.7})
            apply_response = client.post(f"/api/memory/runs/{state['run_id']}/auto-review", json={"min_score": 0.7})

            self.assertEqual(preview_response.status_code, 400)
            self.assertEqual(apply_response.status_code, 400)
            self.assertIn("review queue not found", preview_response.json()["detail"])
            self.assertIn("review queue not found", apply_response.json()["detail"])

    def test_api_memory_run_auto_review_apply_refuses_existing_reviewed_without_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            run_dir = Path(state["run_dir"])
            queue_path = run_dir / "review_queue.jsonl"
            queue_path.write_text(json.dumps({"candidate_id": "create_1", "suggested_action": "create", "candidate": {"id": "create_1", "type": "workflow", "scope": "user", "content": "用户偏好直接推进。", "evidence": [{"event_id": "event_1"}]}, "dream_analysis": {"dream_score": 0.82}}, ensure_ascii=False) + "\n", encoding="utf-8")
            reviewed_path = run_dir / "reviewed.jsonl"
            reviewed_path.write_text(json.dumps({"candidate_id": "manual", "action": "approved"}, ensure_ascii=False) + "\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(queue_path)})
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.post(f"/api/memory/runs/{state['run_id']}/auto-review", json={"min_score": 0.7})

            self.assertEqual(response.status_code, 409)
            rows = [json.loads(line) for line in reviewed_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["candidate_id"], "manual")

            force_response = client.post(f"/api/memory/runs/{state['run_id']}/auto-review", json={"min_score": 0.7, "force": True})

            self.assertEqual(force_response.status_code, 200)
            overwritten = [json.loads(line) for line in reviewed_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(overwritten[0]["candidate_id"], "create_1")

    def test_api_memory_run_auto_review_apply_writes_reviewed(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            queue_path = Path(state["run_dir"]) / "review_queue.jsonl"
            queue_path.write_text(json.dumps({"candidate_id": "create_1", "suggested_action": "create", "candidate": {"id": "create_1", "type": "workflow", "scope": "user", "content": "用户偏好直接推进。", "evidence": [{"event_id": "event_1"}]}, "dream_analysis": {"dream_score": 0.82}}, ensure_ascii=False) + "\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(queue_path)})
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.post(f"/api/memory/runs/{state['run_id']}/auto-review", json={"min_score": 0.7})

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertFalse(payload["dry_run"])
            self.assertEqual(payload["decision_count"], 1)
            rows = [json.loads(line) for line in (Path(state["run_dir"]) / "reviewed.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["action"], "approved")

    def test_api_memory_run_auto_review_preview_respects_include_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            queue_path = Path(state["run_dir"]) / "review_queue.jsonl"
            queue_path.write_text("\n".join([
                json.dumps({"candidate_id": "dup_1", "suggested_action": "reject", "quality_signals": {"duplicate": True}, "candidate": {"id": "dup_1", "type": "preference", "scope": "user", "content": "重复", "evidence": [{"event_id": "event_1"}]}, "dream_analysis": {"dream_score": 0.8}}, ensure_ascii=False),
                json.dumps({"candidate_id": "merge_1", "suggested_action": "merge", "candidate": {"id": "merge_1", "type": "workflow", "scope": "user", "content": "合并", "evidence": [{"event_id": "event_2"}]}, "dream_analysis": {"dream_score": 0.82}}, ensure_ascii=False),
            ]) + "\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(queue_path)})
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            default_response = client.post(f"/api/memory/runs/{state['run_id']}/auto-review/preview", json={"min_score": 0.7})
            include_response = client.post(f"/api/memory/runs/{state['run_id']}/auto-review/preview", json={"min_score": 0.7, "include_duplicates": True, "include_merges": True})

            self.assertEqual(default_response.status_code, 200)
            self.assertEqual(include_response.status_code, 200)
            default_reasons = {row["candidate_id"]: row["reason"] for row in default_response.json()["preview"]}
            include_decisions = {row["candidate_id"]: row["decision"] for row in include_response.json()["preview"]}
            self.assertEqual(default_reasons["dup_1"], "duplicate")
            self.assertEqual(default_reasons["merge_1"], "merge_requires_explicit_include")
            self.assertEqual(include_decisions["dup_1"], "rejected")
            self.assertEqual(include_decisions["merge_1"], "merged")

    def test_api_memory_run_auto_review_preview_requires_evidence_for_approval(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            queue_path = Path(state["run_dir"]) / "review_queue.jsonl"
            queue_path.write_text(json.dumps({
                "candidate_id": "create_1",
                "suggested_action": "create",
                "candidate": {"id": "create_1", "type": "workflow", "scope": "user", "content": "用户偏好直接推进。"},
                "dream_analysis": {"dream_score": 0.82},
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(queue_path)})
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.post(f"/api/memory/runs/{state['run_id']}/auto-review/preview", json={"min_score": 0.7})

            self.assertEqual(response.status_code, 200)
            preview = response.json()["preview"]
            self.assertEqual(preview[0]["decision"], "needs_more_evidence")
            self.assertEqual(preview[0]["reason"], "missing_evidence")
            self.assertFalse((Path(state["run_dir"]) / "reviewed.jsonl").exists())

    def test_api_memory_run_auto_review_preview_keep_review_skips_missing_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            queue_path = Path(state["run_dir"]) / "review_queue.jsonl"
            queue_path.write_text(json.dumps({
                "candidate_id": "create_1",
                "suggested_action": "create",
                "candidate": {"id": "create_1", "type": "workflow", "scope": "user", "content": "用户偏好直接推进。"},
                "dream_analysis": {"dream_score": 0.82},
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(queue_path)})
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.post(f"/api/memory/runs/{state['run_id']}/auto-review/preview", json={"min_score": 0.7, "keep_review": True})

            self.assertEqual(response.status_code, 200)
            preview = response.json()["preview"]
            self.assertEqual(preview[0]["decision"], "skip")
            self.assertEqual(preview[0]["reason"], "missing_evidence")

    def test_api_memory_run_auto_review_preview_marks_sensitive_legacy_queue_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            queue_path = Path(state["run_dir"]) / "review_queue.jsonl"
            queue_path.write_text(json.dumps({
                "candidate_id": "create_1",
                "suggested_action": "create",
                "candidate": {
                    "id": "create_1",
                    "type": "workflow",
                    "scope": "project",
                    "project": "/tmp/project",
                    "content": "安全配置说明需要通过人工审核后再进入正式记忆。",
                    "evidence": [{"event_id": "event_1"}],
                },
                "quality_signals": {
                    "matched_memory_summary": "项目的 API key 在 key.txt 文件中。",
                },
                "conflicts": [{"summary": "项目的 API key 在 key.txt 文件中。"}],
                "dream_analysis": {"dream_score": 0.82},
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(queue_path)})
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.post(f"/api/memory/runs/{state['run_id']}/auto-review/preview", json={"min_score": 0.7})

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            payload_text = json.dumps(payload, ensure_ascii=False)
            self.assertNotIn("API key", payload_text)
            self.assertNotIn("key.txt", payload_text)
            self.assertIn("sensitive_queue_metadata", payload["skip_reasons"])
            self.assertEqual(payload["preview"][0]["decision"], "skip")
            self.assertEqual(payload["preview"][0]["reason"], "sensitive_queue_metadata")

    def test_api_memory_run_auto_review_preview_redacts_sensitive_candidate_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            queue_path = Path(state["run_dir"]) / "review_queue.jsonl"
            queue_path.write_text(json.dumps({
                "candidate_id": "create_1",
                "suggested_action": "create",
                "candidate": {
                    "id": "create_1",
                    "type": "workflow",
                    "scope": "project",
                    "project": "/tmp/project",
                    "content": "项目的 API key 在 key.txt 文件中。",
                    "evidence": [{"event_id": "event_1", "quote": "项目的 API key 在 key.txt 文件中。"}],
                },
                "dream_analysis": {"dream_score": 0.82},
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(queue_path)})
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.post(f"/api/memory/runs/{state['run_id']}/auto-review/preview", json={"min_score": 0.7})

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            payload_text = json.dumps(payload, ensure_ascii=False)
            self.assertNotIn("API key", payload_text)
            self.assertNotIn("key.txt", payload_text)
            self.assertIn("<redacted>", payload_text)
            self.assertIn("sensitive_candidate", payload["skip_reasons"])
            self.assertEqual(payload["preview"][0]["decision"], "skip")
            self.assertEqual(payload["preview"][0]["reason"], "sensitive_candidate")

    def test_api_memory_run_review_progress_counts_pending_and_reviewed(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="ai", model="anthropic:test", invoke_model=False)
            candidates_path = Path(state["run_dir"]) / "candidates.jsonl"
            candidates_path.write_text("\n".join([
                json.dumps({"id": "cand_1", "status": "review", "type": "preference", "content": "用户偏好中文。"}, ensure_ascii=False),
                json.dumps({"id": "cand_2", "status": "review", "type": "workflow", "content": "正式记忆需要审核。"}, ensure_ascii=False),
            ]) + "\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"candidates_path": str(candidates_path)})
            reviewed_path = Path(state["run_dir"]) / "reviewed.jsonl"
            reviewed_path.write_text(json.dumps({"candidate_id": "cand_1", "action": "approved", "status": "approved"}, ensure_ascii=False) + "\n", encoding="utf-8")
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.get(f"/api/memory/runs/{state['run_id']}/review-progress")

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["total"], 2)
            self.assertEqual(payload["reviewed"], 1)
            self.assertEqual(payload["pending"], 1)
            self.assertEqual(payload["actions"]["approved"], 1)

    def test_api_memory_run_review_progress_rejects_malformed_reviewed_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            run_dir = Path(state["run_dir"])
            queue_path = run_dir / "review_queue.jsonl"
            queue_path.write_text("\n".join([
                json.dumps({"candidate_id": "cand_1", "suggested_action": "create", "candidate": {"id": "cand_1", "status": "promote", "type": "workflow", "content": "候选 1"}}, ensure_ascii=False),
                json.dumps({"candidate_id": "cand_2", "suggested_action": "review", "candidate": {"id": "cand_2", "status": "review", "type": "workflow", "content": "候选 2"}}, ensure_ascii=False),
            ]) + "\n", encoding="utf-8")
            (run_dir / "reviewed.jsonl").write_text(
                json.dumps({"candidate_id": "cand_1", "action": "approved"}, ensure_ascii=False)
                + "\n{not json\n",
                encoding="utf-8",
            )
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(queue_path)})
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.get(f"/api/memory/runs/{state['run_id']}/review-progress")

            self.assertEqual(response.status_code, 400)
            self.assertIn("invalid reviewed decisions JSON", response.json()["detail"])

    def test_api_memory_run_review_progress_ignores_orphan_reviewed_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            run_dir = Path(state["run_dir"])
            queue_path = run_dir / "review_queue.jsonl"
            queue_path.write_text(json.dumps({
                "candidate_id": "cand_1",
                "suggested_action": "create",
                "candidate": {"id": "cand_1", "status": "promote", "type": "workflow", "content": "候选 1"},
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            (run_dir / "reviewed.jsonl").write_text("\n".join([
                json.dumps({"candidate_id": "cand_1", "action": "approved"}, ensure_ascii=False),
                json.dumps({"candidate_id": "ghost", "action": "approved"}, ensure_ascii=False),
            ]) + "\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(queue_path)})
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.get(f"/api/memory/runs/{state['run_id']}/review-progress")

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["total"], 1)
            self.assertEqual(payload["reviewed"], 1)
            self.assertEqual(payload["actions"], {"approved": 1})

    def test_api_memory_run_review_progress_deduplicates_candidate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            run_dir = Path(state["run_dir"])
            queue_path = run_dir / "review_queue.jsonl"
            queue_path.write_text("\n".join([
                json.dumps({"candidate_id": "cand_1", "suggested_action": "create", "candidate": {"id": "cand_1", "status": "promote", "type": "workflow", "content": "候选 1"}}, ensure_ascii=False),
                json.dumps({"candidate_id": "cand_1", "suggested_action": "create", "candidate": {"id": "cand_1", "status": "promote", "type": "workflow", "content": "候选 1 重复"}}, ensure_ascii=False),
            ]) + "\n", encoding="utf-8")
            (run_dir / "reviewed.jsonl").write_text(json.dumps({"candidate_id": "cand_1", "action": "approved"}, ensure_ascii=False) + "\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(queue_path)})
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.get(f"/api/memory/runs/{state['run_id']}/review-progress")

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["total"], 1)
            self.assertEqual(payload["reviewed"], 1)
            self.assertEqual(payload["pending"], 0)
            self.assertEqual(payload["pending_ids"], [])

    def test_api_memory_run_review_progress_prefers_review_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            run_dir = Path(state["run_dir"])
            candidates_path = run_dir / "candidates.jsonl"
            candidates_path.write_text(json.dumps({"id": "stale", "status": "review", "type": "workflow", "content": "stale"}, ensure_ascii=False) + "\n", encoding="utf-8")
            queue_path = run_dir / "review_queue.jsonl"
            queue_path.write_text("\n".join([
                json.dumps({"candidate_id": "cand_1", "suggested_action": "create", "status": "promote", "candidate": {"id": "cand_1", "status": "promote", "type": "workflow", "content": "候选 1"}}, ensure_ascii=False),
                json.dumps({"candidate_id": "cand_2", "suggested_action": "review", "status": "review", "candidate": {"id": "cand_2", "status": "review", "type": "workflow", "content": "候选 2"}}, ensure_ascii=False),
            ]) + "\n", encoding="utf-8")
            (run_dir / "reviewed.jsonl").write_text(json.dumps({"candidate_id": "cand_1", "action": "approved"}, ensure_ascii=False) + "\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"candidates_path": str(candidates_path), "review_queue_path": str(queue_path)})
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.get(f"/api/memory/runs/{state['run_id']}/review-progress")

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["source"], "review_queue")
            self.assertEqual(payload["total"], 2)
            self.assertEqual(payload["reviewed"], 1)
            self.assertEqual(payload["pending"], 1)
            self.assertEqual(payload["pending_ids"], ["cand_2"])
            self.assertEqual(payload["suggested_actions"], {"create": 1, "review": 1})

    def test_api_memory_run_review_progress_redacts_sensitive_legacy_summary_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            queue_path = Path(state["run_dir"]) / "review_queue.jsonl"
            queue_path.write_text(json.dumps({
                "candidate_id": "api_key=sk-test-secret",
                "suggested_action": "项目的 API key 在 key.txt 文件中。",
                "status": "token 在 config.yaml 配置中",
                "candidate": {
                    "id": "api_key=sk-test-secret",
                    "type": "workflow",
                    "scope": "project",
                    "content": "安全配置说明需要人工审核。",
                    "evidence": [{"event_id": "event_1"}],
                },
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(queue_path)})
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.get(f"/api/memory/runs/{state['run_id']}/review-progress")

            self.assertEqual(response.status_code, 200)
            payload_text = json.dumps(response.json(), ensure_ascii=False)
            self.assertNotIn("sk-test-secret", payload_text)
            self.assertNotIn("API key", payload_text)
            self.assertNotIn("key.txt", payload_text)
            self.assertNotIn("config.yaml", payload_text)
            self.assertIn("<redacted>", payload_text)

    def test_api_memory_run_review_progress_preserves_colliding_redacted_bucket_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            queue_path = Path(state["run_dir"]) / "review_queue.jsonl"
            queue_path.write_text("\n".join([
                json.dumps({"candidate_id": "cand_1", "suggested_action": "api_key=sk-test-secret-a", "status": "token=sk-test-secret-a", "candidate": {"id": "cand_1"}}, ensure_ascii=False),
                json.dumps({"candidate_id": "cand_2", "suggested_action": "api_key=sk-test-secret-b", "status": "token=sk-test-secret-b", "candidate": {"id": "cand_2"}}, ensure_ascii=False),
            ]) + "\n", encoding="utf-8")
            reviewed_path = Path(state["run_dir"]) / "reviewed.jsonl"
            reviewed_path.write_text("\n".join([
                json.dumps({"candidate_id": "cand_1", "action": "api_key=sk-test-secret-a"}, ensure_ascii=False),
                json.dumps({"candidate_id": "cand_2", "action": "api_key=sk-test-secret-b"}, ensure_ascii=False),
            ]) + "\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(queue_path), "reviewed_path": str(reviewed_path)})
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.get(f"/api/memory/runs/{state['run_id']}/review-progress")

            payload_text = json.dumps(response.json(), ensure_ascii=False)
            payload = response.json()
            self.assertEqual(response.status_code, 200)
            self.assertNotIn("sk-test-secret", payload_text)
            self.assertEqual(payload["actions"], {"<redacted>": 1, "<redacted>-2": 1})
            self.assertEqual(payload["suggested_actions"], {"<redacted>": 1, "<redacted>-2": 1})
            self.assertEqual(payload["statuses"], {"<redacted>": 1, "<redacted>-2": 1})

    def test_memory_review_page_contains_grouped_candidates_and_progress_ui(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.get("/memory-review")

            self.assertEqual(response.status_code, 200)
            self.assertIn("审核进度", response.text)
            self.assertIn("loadReviewProgress", response.text)
            self.assertIn("groupCandidates", response.text)
            self.assertIn("候选分组", response.text)
            self.assertIn("候选汇总", response.text)
            self.assertIn("loadReviewSummary", response.text)
            self.assertIn("review-summary", response.text)
            self.assertIn("Dream Analysis", response.text)
            self.assertIn("Quality Signals", response.text)
            self.assertIn("quality_signals", response.text)
            self.assertIn("dream_analysis", response.text)
            self.assertIn("自动审核预览", response.text)
            self.assertIn("previewAutoReview", response.text)
            self.assertIn("applyAutoReview", response.text)
            self.assertIn("autoReviewForce", response.text)
            self.assertIn("suggested_actions", response.text)
            self.assertIn("prompt_event_count", response.text)
            self.assertIn("filtered_prompt_event_count", response.text)
            self.assertIn("送模", response.text)
            self.assertIn("输入事件", response.text)
            self.assertIn("已过滤", response.text)
            self.assertIn("counts.input_event_count", response.text)
            self.assertIn("counts.prompt_event_count", response.text)
            self.assertIn("counts.filtered_prompt_event_count", response.text)
            self.assertIn("readJsonOrThrow", response.text)
            self.assertIn("候选加载失败", response.text)
            self.assertIn("进度加载失败", response.text)
            self.assertIn("轨迹加载失败", response.text)
            self.assertIn("throw new Error", response.text)

    def test_api_memory_run_review_summary_groups_queue_quality(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            queue_path = Path(state["run_dir"]) / "review_queue.jsonl"
            queue_path.write_text("\n".join([
                json.dumps({"candidate_id": "create_1", "suggested_action": "create", "status": "promote", "candidate": {"id": "create_1", "type": "workflow", "scope": "user", "content": "高分", "evidence": [{"event_id": "event_1"}]}, "quality_signals": {"evidence_quality": "multi_event"}, "dream_analysis": {"dream_score": 0.82}}, ensure_ascii=False),
                json.dumps({"candidate_id": "review_1", "suggested_action": "review", "status": "review", "candidate": {"id": "review_1", "type": "preference", "scope": "user", "content": "人工", "evidence": [{"event_id": "event_2"}]}, "quality_signals": {"duplicate": True, "evidence_quality": "single_event"}, "dream_analysis": {"dream_score": 0.55}, "conflicts": [{"memory_id": "mem_1"}]}, ensure_ascii=False),
            ]) + "\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(queue_path)})
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.get(f"/api/memory/runs/{state['run_id']}/review-summary")

            self.assertEqual(response.status_code, 200)
            summary = response.json()["summary"]
            self.assertEqual(summary["total"], 2)
            self.assertEqual(summary["by_suggested_action"], {"create": 1, "review": 1})
            self.assertEqual(summary["by_type"], {"preference": 1, "workflow": 1})
            self.assertEqual(summary["duplicate_count"], 1)
            self.assertEqual(summary["conflict_count"], 1)
            self.assertEqual(summary["needs_manual_count"], 1)

    def test_api_memory_run_review_summary_redacts_sensitive_legacy_bucket_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            queue_path = Path(state["run_dir"]) / "review_queue.jsonl"
            queue_path.write_text(json.dumps({
                "candidate_id": "cand_1",
                "suggested_action": "api_key=sk-test-secret",
                "status": "token 在 config.yaml 配置中",
                "candidate": {
                    "id": "cand_1",
                    "type": "项目的 API key 在 key.txt 文件中。",
                    "scope": "project",
                    "content": "安全配置说明需要人工审核。",
                    "evidence": [{"event_id": "event_1"}],
                },
                "quality_signals": {
                    "evidence_quality": "api_key=sk-test-secret",
                    "value_class": "项目的 API key 在 key.txt 文件中。",
                },
                "dream_analysis": {"dream_score": 0.82},
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(queue_path)})
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.get(f"/api/memory/runs/{state['run_id']}/review-summary")

            self.assertEqual(response.status_code, 200)
            payload_text = json.dumps(response.json(), ensure_ascii=False)
            self.assertNotIn("sk-test-secret", payload_text)
            self.assertNotIn("API key", payload_text)
            self.assertNotIn("key.txt", payload_text)
            self.assertNotIn("config.yaml", payload_text)
            self.assertIn("<redacted>", payload_text)

    def test_api_memory_run_review_summary_preserves_colliding_redacted_bucket_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            queue_path = Path(state["run_dir"]) / "review_queue.jsonl"
            queue_path.write_text("\n".join([
                json.dumps({"candidate_id": "cand_1", "suggested_action": "api_key=sk-test-secret-a", "status": "token=sk-test-secret-a", "candidate": {"id": "cand_1", "type": "workflow", "scope": "project"}, "quality_signals": {"evidence_quality": "api_key=sk-test-secret-a", "value_class": "api_key=sk-test-secret-a"}, "dream_analysis": {"dream_score": 0.82}}, ensure_ascii=False),
                json.dumps({"candidate_id": "cand_2", "suggested_action": "api_key=sk-test-secret-b", "status": "token=sk-test-secret-b", "candidate": {"id": "cand_2", "type": "workflow", "scope": "project"}, "quality_signals": {"evidence_quality": "api_key=sk-test-secret-b", "value_class": "api_key=sk-test-secret-b"}, "dream_analysis": {"dream_score": 0.91}}, ensure_ascii=False),
            ]) + "\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(queue_path)})
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.get(f"/api/memory/runs/{state['run_id']}/review-summary")

            payload_text = json.dumps(response.json(), ensure_ascii=False)
            summary = response.json()["summary"]
            self.assertEqual(response.status_code, 200)
            self.assertNotIn("sk-test-secret", payload_text)
            self.assertEqual(summary["by_suggested_action"], {"<redacted>": 1, "<redacted>-2": 1})
            self.assertEqual(summary["by_status"], {"<redacted>": 1, "<redacted>-2": 1})
            self.assertEqual(summary["by_evidence_quality"], {"<redacted>": 1, "<redacted>-2": 1})
            self.assertEqual(summary["by_value_class"], {"<redacted>": 1, "<redacted>-2": 1})


    def test_api_memory_run_review_queue_returns_conflicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="ai", model="anthropic:test", invoke_model=False)
            queue_path = Path(state["run_dir"]) / "review_queue.jsonl"
            queue = build_review_queue(
                [{
                    "id": "cand_1",
                    "scope": "project",
                    "project": "/tmp/project",
                    "status": "promote",
                    "type": "decision",
                    "content": "项目目标是本地记忆系统。",
                    "score": 0.9,
                    "evidence": [{"event_id": "event_1"}, {"event_id": "event_2"}],
                    "tags": ["product"],
                }],
                [{
                    "id": "mem_1",
                    "scope": "project",
                    "project": "/tmp/project",
                    "memory_type": "decision",
                    "summary": "项目目标是本地记忆系统 V1。",
                    "status": "active",
                }],
            )
            queue_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in queue) + "\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(queue_path)})
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.get(f"/api/memory/runs/{state['run_id']}/review-queue")

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["count"], 1)
            self.assertEqual(payload["items"][0]["candidate_id"], "cand_1")
            self.assertEqual(payload["items"][0]["conflicts"][0]["memory_id"], "mem_1")
            self.assertIn("dream_analysis", payload["items"][0])
            self.assertEqual(payload["items"][0]["suggested_action"], payload["items"][0]["dream_analysis"]["suggested_action"])

    def test_api_memory_run_review_queue_redacts_sensitive_legacy_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            queue_path = Path(state["run_dir"]) / "review_queue.jsonl"
            queue_path.write_text(json.dumps({
                "candidate_id": "cand_1",
                "suggested_action": "create",
                "candidate": {
                    "id": "cand_1",
                    "type": "workflow",
                    "scope": "project",
                    "project": "/tmp/project",
                    "content": "安全配置说明需要通过人工审核后再进入正式记忆。",
                    "evidence": [{"event_id": "event_1"}],
                },
                "quality_signals": {
                    "matched_memory_id": "secret",
                    "matched_memory_summary": "项目的 API key 在 key.txt 文件中。",
                },
                "conflicts": [{"memory_id": "secret", "summary": "项目的 API key 在 key.txt 文件中。"}],
                "dream_analysis": {"dream_score": 0.82},
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(queue_path)})
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.get(f"/api/memory/runs/{state['run_id']}/review-queue")

            self.assertEqual(response.status_code, 200)
            payload_text = json.dumps(response.json(), ensure_ascii=False)
            self.assertNotIn("key.txt", payload_text)
            self.assertNotIn("API key", payload_text)

    def test_api_memory_run_review_queue_rejects_malformed_run_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            queue_path = Path(state["run_dir"]) / "review_queue.jsonl"
            queue_path.write_text(json.dumps({
                "candidate_id": "cand_1",
                "suggested_action": "create",
                "candidate": {"id": "cand_1", "type": "workflow", "scope": "user", "content": "用户偏好直接推进。", "evidence": [{"event_id": "event_1"}]},
                "dream_analysis": {"dream_score": 0.82},
            }, ensure_ascii=False) + "\nnot-json\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(queue_path)})
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            responses = [
                client.get(f"/api/memory/runs/{state['run_id']}/review-queue"),
                client.get(f"/api/memory/runs/{state['run_id']}/review-summary"),
                client.get(f"/api/memory/runs/{state['run_id']}/review-progress"),
                client.post(f"/api/memory/runs/{state['run_id']}/auto-review/preview", json={"min_score": 0.7}),
            ]

            for response in responses:
                self.assertEqual(response.status_code, 400)
                self.assertIn("invalid review queue JSON", response.json()["detail"])

    def test_api_memory_run_artifact_reads_reject_paths_outside_run_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            outside_queue = Path(tmp) / "outside-review-queue.jsonl"
            outside_candidates = Path(tmp) / "outside-candidates.jsonl"
            outside_queue.write_text(json.dumps({"candidate_id": "leaked", "candidate": {"id": "leaked"}}, ensure_ascii=False) + "\n", encoding="utf-8")
            outside_candidates.write_text(json.dumps({"id": "leaked"}, ensure_ascii=False) + "\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(outside_queue), "candidates_path": str(outside_candidates)})
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            queue_response = client.get(f"/api/memory/runs/{state['run_id']}/review-queue")
            summary_response = client.get(f"/api/memory/runs/{state['run_id']}/review-summary")
            candidates_response = client.get(f"/api/memory/runs/{state['run_id']}/candidates")
            progress_response = client.get(f"/api/memory/runs/{state['run_id']}/review-progress")

            for response in [queue_response, summary_response, candidates_response, progress_response]:
                self.assertEqual(response.status_code, 400)
                self.assertIn("outside run directory", response.json()["detail"])

    def test_api_memory_run_candidates_rejects_malformed_run_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            candidates_path = Path(state["run_dir"]) / "candidates.jsonl"
            candidates_path.write_text(json.dumps({
                "id": "cand_1",
                "type": "workflow",
                "scope": "user",
                "content": "用户偏好直接推进。",
                "evidence": [{"event_id": "event_1"}],
            }, ensure_ascii=False) + "\nnot-json\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"candidates_path": str(candidates_path)})
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            candidates_response = client.get(f"/api/memory/runs/{state['run_id']}/candidates")
            progress_response = client.get(f"/api/memory/runs/{state['run_id']}/review-progress")
            review_response = client.post(
                f"/api/memory/runs/{state['run_id']}/review",
                json={
                    "candidate_id": "cand_1",
                    "action": "approved",
                    "edited_content": "用户偏好直接推进。",
                    "reviewer": "user",
                    "candidate": {"id": "cand_1", "content": "用户偏好直接推进。", "evidence": [{"event_id": "event_1"}]},
                },
            )

            for response in (candidates_response, progress_response, review_response):
                self.assertEqual(response.status_code, 400)
                self.assertIn("invalid candidates JSON", response.json()["detail"])
            self.assertFalse((Path(state["run_dir"]) / "reviewed.jsonl").exists())

    def test_api_memory_run_candidates_redacts_sensitive_legacy_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            candidates_path = Path(state["run_dir"]) / "candidates.jsonl"
            candidates_path.write_text(json.dumps({
                "id": "cand_1",
                "type": "workflow",
                "scope": "project",
                "project": "/tmp/project",
                "content": "项目的 API key 在 key.txt 文件中。",
                "evidence": [{"event_id": "event_1", "quote": "项目的 API key 在 key.txt 文件中。"}],
            }, ensure_ascii=False) + "\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"candidates_path": str(candidates_path)})
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.get(f"/api/memory/runs/{state['run_id']}/candidates")

            self.assertEqual(response.status_code, 200)
            payload_text = json.dumps(response.json(), ensure_ascii=False)
            self.assertNotIn("API key", payload_text)
            self.assertNotIn("key.txt", payload_text)
            self.assertIn("<redacted>", payload_text)

    def test_memory_review_page_distinguishes_new_value_from_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.get("/memory-review")

            self.assertEqual(response.status_code, 200)
            self.assertIn("新增价值", response.text)
            self.assertIn("已有记忆重复", response.text)
            self.assertIn("valueClass", response.text)
            self.assertIn("matched_memory_summary", response.text)
            self.assertIn("按价值分组", response.text)

    def test_api_memory_run_review_summary_counts_value_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            queue_path = Path(state["run_dir"]) / "review_queue.jsonl"
            queue_path.write_text("\n".join([
                json.dumps({"candidate_id": "new_1", "suggested_action": "create", "candidate": {"id": "new_1", "type": "workflow", "scope": "user", "content": "新增"}, "quality_signals": {"value_class": "new_value"}, "dream_analysis": {"dream_score": 0.82}}, ensure_ascii=False),
                json.dumps({"candidate_id": "dup_1", "suggested_action": "reject", "candidate": {"id": "dup_1", "type": "workflow", "scope": "user", "content": "重复"}, "quality_signals": {"duplicate": True, "value_class": "existing_duplicate"}, "dream_analysis": {"dream_score": 0.82}}, ensure_ascii=False),
            ]) + "\n", encoding="utf-8")
            update_run_state(state, status="waiting_review", phase="review", artifacts={"review_queue_path": str(queue_path)})
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.get(f"/api/memory/runs/{state['run_id']}/review-summary")

            self.assertEqual(response.status_code, 200)
            summary = response.json()["summary"]
            self.assertEqual(summary["by_value_class"], {"existing_duplicate": 1, "new_value": 1})
            self.assertEqual(summary["new_value_count"], 1)
            self.assertEqual(summary["existing_duplicate_count"], 1)

    def test_memory_review_page_uses_review_queue_for_run_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            memory_dir.mkdir()
            app = create_app(default_output_dir=Path(tmp) / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            response = client.get("/memory-review")

            self.assertEqual(response.status_code, 200)
            self.assertIn("loadReviewQueue", response.text)
            self.assertIn("review-queue", response.text)
            self.assertIn("conflicts", response.text)
            self.assertIn("quality_signals:item.quality_signals", response.text)
            self.assertIn("dream_analysis:item.dream_analysis", response.text)


if __name__ == "__main__":
    unittest.main()
