import json
import tempfile
import unittest
from pathlib import Path

from deepagent_memory.memory_runs import (
    append_trace,
    create_run_state,
    list_runs,
    load_run_state,
    read_trace,
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

    def test_list_runs_returns_created_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = create_run_state(memory_dir=tmp, project=None, input_path=None, mode="ai", model="anthropic:test", invoke_model=False)

            runs = list_runs(tmp)

            self.assertEqual(runs[0]["run_id"], state["run_id"])


if __name__ == "__main__":
    unittest.main()
