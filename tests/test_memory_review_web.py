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
                    "candidate": {"id": "cand_1", "content": "用户偏好中文回答。"},
                },
            )

            self.assertEqual(response.status_code, 200)
            reviewed = memory_dir / "reviewed.jsonl"
            self.assertTrue(reviewed.exists())
            rows = [json.loads(line) for line in reviewed.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["action"], "approved")
            self.assertEqual(rows[0]["candidate_id"], "cand_1")

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

    def test_api_memory_run_start_records_background_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            events = root / "events.jsonl"
            events.write_text(json.dumps({"event_id": "event_1", "source": "codex", "role": "user", "event_type": "history_prompt", "project": str(root), "content": "这个项目需要人工审核后才能写正式记忆"}, ensure_ascii=False) + "\n", encoding="utf-8")
            app = create_app(default_output_dir=root / "runs", default_memory_dir=memory_dir)
            client = TestClient(app)

            with patch("dream_memory.web._run_dream_to_review", side_effect=RuntimeError("boom")):
                response = client.post("/api/memory/runs/start", json={"input": str(events), "project": ".", "mode": "ai", "invoke_model": True})

            self.assertEqual(response.status_code, 200)
            state = json.loads((memory_dir / "runs" / response.json()["run_id"] / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["phase"], "failed")
            self.assertEqual(state["error"], "boom")
            trace = (memory_dir / "runs" / response.json()["run_id"] / "trace.jsonl").read_text(encoding="utf-8")
            self.assertIn("run_failed", trace)

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
                    "summary": "项目目标是通用记忆系统。",
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


if __name__ == "__main__":
    unittest.main()
