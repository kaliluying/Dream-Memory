import json
import tempfile
import unittest
from pathlib import Path

from dream_memory.memory_runs import (
    append_trace,
    copy_input_events,
    create_run_state,
    candidate_trace_path,
    list_runs,
    load_run_state,
    read_candidate_trace,
    read_trace,
    run_dir,
    save_run_state,
    update_run_state,
    write_candidate_traces,
)


class MemoryRunTests(unittest.TestCase):
    def test_create_update_and_load_run_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = create_run_state(memory_dir=tmp, project="/tmp/project", input_path="events.jsonl", mode="ai", model="anthropic:test", invoke_model=False)
            updated = update_run_state(state, status="waiting_review", phase="review", counts={"candidate_count": 2})
            loaded = load_run_state(tmp, state["run_id"])

            self.assertEqual(loaded["run_id"], state["run_id"])
            self.assertEqual(loaded["status"], "waiting_review")
            self.assertEqual(loaded["counts"]["candidate_count"], 2)
            self.assertTrue(Path(loaded["run_dir"]).exists())

    def test_trace_and_candidate_lineage_are_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = create_run_state(memory_dir=tmp, project="/tmp/project", input_path="events.jsonl", mode="ai", model="anthropic:test", invoke_model=False)
            append_trace(state, "custom", {"candidate_id": "cand_1"})
            write_candidate_traces(state, [{"id": "cand_1", "type": "preference", "status": "review"}])

            trace = read_trace(tmp, state["run_id"], candidate_id="cand_1")
            candidate_path = Path(state["run_dir"]) / "candidates" / "cand_1.json"

            self.assertTrue(trace)
            self.assertTrue(candidate_path.exists())
            self.assertEqual(json.loads(candidate_path.read_text(encoding="utf-8"))["candidate_id"], "cand_1")

    def test_trace_and_state_errors_redact_sensitive_payload_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = create_run_state(memory_dir=tmp, project="/tmp/project", input_path="events.jsonl", mode="ai", model="anthropic:test", invoke_model=False)

            updated = update_run_state(
                state,
                status="failed",
                phase="failed",
                error="provider failed api_key=sk-test-secret Bearer abc123 at https://user:pass123@example.test/v1",
            )
            append_trace(
                updated,
                "run_failed",
                {"error": "provider failed api_key=sk-test-secret Bearer abc123 at https://user:pass123@example.test/v1"},
            )

            loaded = load_run_state(tmp, state["run_id"])
            trace_text = (Path(state["run_dir"]) / "trace.jsonl").read_text(encoding="utf-8")

            self.assertNotIn("sk-test-secret", loaded["error"])
            self.assertNotIn("Bearer abc123", loaded["error"])
            self.assertNotIn("pass123", loaded["error"])
            self.assertIn("<redacted>", loaded["error"])
            self.assertNotIn("sk-test-secret", trace_text)
            self.assertNotIn("Bearer abc123", trace_text)
            self.assertNotIn("pass123", trace_text)
            self.assertIn("<redacted>", trace_text)

    def test_save_run_state_redacts_sensitive_legacy_state_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = create_run_state(memory_dir=tmp, project="/tmp/project", input_path="events.jsonl", mode="ai", model="anthropic:test", invoke_model=False)
            state["diagnostics"] = {
                "provider_error": "provider failed with OPENAI_API_KEY=sk-test-secret",
                "next_hint": "token 在 config.yaml 配置中",
            }

            save_run_state(state)

            state_text = (Path(state["run_dir"]) / "state.json").read_text(encoding="utf-8")
            self.assertNotIn("sk-test-secret", state_text)
            self.assertNotIn("config.yaml", state_text)
            self.assertIn("<redacted>", state_text)

    def test_read_trace_rejects_malformed_audit_log_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = create_run_state(memory_dir=tmp, project="/tmp/project", input_path="events.jsonl", mode="ai", model="anthropic:test", invoke_model=False)
            trace_path = Path(state["run_dir"]) / "trace.jsonl"
            trace_path.write_text(
                json.dumps({"run_id": state["run_id"], "event_type": "run_created", "payload": {}}, ensure_ascii=False)
                + "\nnot-json\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "invalid trace JSON"):
                read_trace(tmp, state["run_id"])

            self.assertEqual(len(read_trace(tmp, state["run_id"], strict=False)), 1)

    def test_read_trace_rejects_non_object_payload_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = create_run_state(memory_dir=tmp, project="/tmp/project", input_path="events.jsonl", mode="ai", model="anthropic:test", invoke_model=False)
            trace_path = Path(state["run_dir"]) / "trace.jsonl"
            trace_path.write_text(
                json.dumps({"run_id": state["run_id"], "event_type": "candidate_ready", "payload": "bad-payload"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "invalid trace payload"):
                read_trace(tmp, state["run_id"], candidate_id="cand_1")

            self.assertEqual(read_trace(tmp, state["run_id"], candidate_id="cand_1", strict=False), [])

    def test_read_candidate_trace_rejects_invalid_or_mismatched_trace_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = create_run_state(memory_dir=tmp, project="/tmp/project", input_path="events.jsonl", mode="ai", model="anthropic:test", invoke_model=False)
            candidate_path = candidate_trace_path(tmp, state["run_id"], "cand_1")
            candidate_path.parent.mkdir(parents=True, exist_ok=True)

            candidate_path.write_text("[]\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "candidate trace must be a JSON object"):
                read_candidate_trace(tmp, state["run_id"], "cand_1")

            candidate_path.write_text(json.dumps({"candidate_id": "other"}, ensure_ascii=False) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "candidate trace id mismatch"):
                read_candidate_trace(tmp, state["run_id"], "cand_1")

            candidate_path.write_text("{not json", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid candidate trace JSON"):
                read_candidate_trace(tmp, state["run_id"], "cand_1")

    def test_list_runs_returns_created_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = create_run_state(memory_dir=tmp, project=None, input_path=None, mode="ai", model="anthropic:test", invoke_model=False)

            runs = list_runs(tmp)

            self.assertEqual(runs[0]["run_id"], state["run_id"])

    def test_run_and_candidate_ids_reject_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                run_dir(tmp, "../outside")
            with self.assertRaises(ValueError):
                load_run_state(tmp, "../outside")
            with self.assertRaises(ValueError):
                read_trace(tmp, "../outside")
            with self.assertRaises(ValueError):
                candidate_trace_path(tmp, "run_20260711T000000Z_abcd1234", "../state")

    def test_load_run_state_ignores_persisted_run_dir_and_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            expected_run_dir = Path(state["run_dir"])
            outside = Path(tmp) / "outside"
            payload = json.loads((expected_run_dir / "state.json").read_text(encoding="utf-8"))
            payload["run_id"] = "spoofed"
            payload["run_dir"] = str(outside)
            payload["memory_dir"] = str(Path(tmp) / "other-memory")
            (expected_run_dir / "state.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            loaded = load_run_state(memory_dir, state["run_id"])
            update_run_state(loaded, status="waiting_review")
            append_trace(loaded, "checked", {})

            self.assertEqual(loaded["run_id"], state["run_id"])
            self.assertEqual(loaded["run_dir"], str(expected_run_dir))
            self.assertEqual(loaded["memory_dir"], str(memory_dir.expanduser()))
            self.assertTrue((expected_run_dir / "state.json").exists())
            self.assertTrue((expected_run_dir / "trace.jsonl").exists())
            self.assertFalse(outside.exists())

    def test_state_writers_use_canonical_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            memory_dir = root / "memory"
            source = root / "events.jsonl"
            source.write_text(json.dumps({"event_id": "event_1"}, ensure_ascii=False) + "\n", encoding="utf-8")
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path=str(source), mode="rules", model="rules", invoke_model=False)
            expected_run_dir = Path(state["run_dir"])
            outside = root / "outside"
            poisoned = dict(state)
            poisoned["run_dir"] = str(outside)

            save_run_state(poisoned)
            append_trace(poisoned, "checked", {})
            copied_path = copy_input_events(source, poisoned)

            self.assertEqual(copied_path, expected_run_dir / "events.jsonl")
            self.assertTrue((expected_run_dir / "state.json").exists())
            self.assertTrue((expected_run_dir / "trace.jsonl").exists())
            self.assertTrue((expected_run_dir / "events.jsonl").exists())
            self.assertFalse(outside.exists())

    def test_list_runs_uses_directory_name_as_authoritative_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            expected_run_dir = Path(state["run_dir"])
            payload = json.loads((expected_run_dir / "state.json").read_text(encoding="utf-8"))
            payload["run_id"] = "spoofed"
            payload["run_dir"] = str(Path(tmp) / "outside")
            payload["memory_dir"] = str(Path(tmp) / "other-memory")
            (expected_run_dir / "state.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            runs = list_runs(memory_dir)

            self.assertEqual(runs[0]["run_id"], state["run_id"])
            self.assertEqual(runs[0]["run_dir"], str(expected_run_dir))
            self.assertEqual(runs[0]["memory_dir"], str(memory_dir.expanduser()))

    def test_list_runs_skips_invalid_run_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            invalid_dir = memory_dir / "runs" / "bad.run"
            invalid_dir.mkdir(parents=True)
            (invalid_dir / "state.json").write_text(json.dumps({"run_id": "bad.run", "memory_dir": str(memory_dir), "run_dir": str(invalid_dir)}, ensure_ascii=False), encoding="utf-8")

            runs = list_runs(memory_dir)

            self.assertEqual([run["run_id"] for run in runs], [state["run_id"]])

    def test_list_runs_surfaces_malformed_state_for_valid_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_dir = Path(tmp) / "memory"
            state = create_run_state(memory_dir=memory_dir, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)
            state_path = Path(state["run_dir"]) / "state.json"
            state_path.write_text("{not json\n", encoding="utf-8")

            runs = list_runs(memory_dir)

            self.assertEqual(len(runs), 1)
            self.assertEqual(runs[0]["run_id"], state["run_id"])
            self.assertEqual(runs[0]["status"], "invalid")
            self.assertEqual(runs[0]["phase"], "invalid_state")
            self.assertIn("invalid run state JSON", runs[0]["error"])
            self.assertEqual(runs[0]["run_dir"], str(Path(state["run_dir"])))

    def test_write_candidate_traces_skips_invalid_candidate_ids_without_losing_valid_traces(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = create_run_state(memory_dir=tmp, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)

            write_candidate_traces(state, [
                {"id": "good_1", "type": "workflow", "status": "review"},
                {"id": "../bad", "type": "workflow", "status": "review"},
                {"id": "good_2", "type": "workflow", "status": "review"},
            ])

            candidates_dir = Path(state["run_dir"]) / "candidates"
            trace = read_trace(tmp, state["run_id"])

            self.assertTrue((candidates_dir / "good_1.json").exists())
            self.assertTrue((candidates_dir / "good_2.json").exists())
            self.assertFalse((Path(state["run_dir"]) / "bad.json").exists())
            self.assertIn("candidate_trace_skipped", [row["event_type"] for row in trace])

    def test_write_candidate_traces_redacts_invalid_candidate_ids_in_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = create_run_state(memory_dir=tmp, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)

            write_candidate_traces(state, [
                {"id": "../api_key=sk-test-secret", "type": "workflow", "status": "review"},
            ])

            trace_text = (Path(state["run_dir"]) / "trace.jsonl").read_text(encoding="utf-8")

            self.assertIn("candidate_trace_skipped", trace_text)
            self.assertNotIn("sk-test-secret", trace_text)
            self.assertIn("<redacted>", trace_text)

    def test_write_candidate_traces_redacts_sensitive_candidate_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = create_run_state(memory_dir=tmp, project="/tmp/project", input_path="events.jsonl", mode="rules", model="rules", invoke_model=False)

            write_candidate_traces(state, [
                {
                    "id": "cand_1",
                    "type": "workflow",
                    "status": "review",
                    "content": "项目的 API key 在 key.txt 文件中，OPENAI_API_KEY=sk-test-secret。",
                    "evidence": [{"event_id": "event_1", "quote": "token 在 config.yaml 配置中"}],
                },
            ])

            candidate_text = (Path(state["run_dir"]) / "candidates" / "cand_1.json").read_text(encoding="utf-8")
            self.assertNotIn("sk-test-secret", candidate_text)
            self.assertNotIn("key.txt", candidate_text)
            self.assertNotIn("config.yaml", candidate_text)
            self.assertIn("<redacted>", candidate_text)


if __name__ == "__main__":
    unittest.main()
