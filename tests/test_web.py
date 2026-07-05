import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from deepagent_project_advice.web import create_app


class WebConsoleTests(unittest.TestCase):
    def test_home_page_contains_task_form_and_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = create_app(default_output_dir=Path(tmp) / "runs")
            client = TestClient(app)

            response = client.get("/")

            self.assertEqual(response.status_code, 200)
            self.assertIn("任务输入区", response.text)
            self.assertIn("执行计划展示", response.text)
            self.assertIn("文件改动 Diff", response.text)
            self.assertIn("测试结果", response.text)
            self.assertIn("最终报告", response.text)

    def test_api_analyze_returns_plan_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            project.mkdir()
            (project / "pyproject.toml").write_text('[project]\nname="demo"\n', encoding="utf-8")
            app = create_app(default_output_dir=Path(tmp) / "runs")
            client = TestClient(app)

            response = client.post("/api/analyze", json={"project": str(project), "task": "增加 CLI 参数校验"})

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["task"], "增加 CLI 参数校验")
            self.assertIn("Python", payload["stack"])
            self.assertIn("# 实现计划", payload["report"])

    def test_api_run_creates_artifacts_with_patch_and_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            project.mkdir()
            (project / "pyproject.toml").write_text('[project]\nname="demo"\n', encoding="utf-8")
            app = create_app(default_output_dir=Path(tmp) / "runs")
            client = TestClient(app)

            response = client.post(
                "/api/run",
                json={
                    "project": str(project),
                    "task": "验证项目",
                    "patch": True,
                    "verify": True,
                    "verify_commands": ["python --version"],
                },
            )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            run_dir = Path(payload["run_dir"])
            self.assertTrue((run_dir / "plan.md").exists())
            self.assertTrue((run_dir / "metadata.json").exists())
            self.assertTrue((run_dir / "suggested.patch").exists())
            self.assertTrue((run_dir / "verification.json").exists())
            metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["verification_status"], "passed")

    def test_api_runs_lists_existing_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            run_dir = runs / "20260705T000000Z"
            run_dir.mkdir(parents=True)
            (run_dir / "metadata.json").write_text('{"task":"demo","created_at":"20260705T000000Z"}\n', encoding="utf-8")
            app = create_app(default_output_dir=runs)
            client = TestClient(app)

            response = client.get("/api/runs")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()[0]["task"], "demo")
