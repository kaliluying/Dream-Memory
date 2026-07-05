import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from deepagent_memory.web import create_app


class MemoryReviewWebTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
