import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dream_memory.memory_cli import main
from dream_memory.memory_eval import _candidate_outcome, evaluate_labeled_events


class MemoryEvalTests(unittest.TestCase):
    def test_candidate_outcome_normalizes_dream_actions(self):
        self.assertEqual(
            _candidate_outcome({"dream_analysis": {"suggested_action": "create"}}),
            "reviewable",
        )
        self.assertEqual(
            _candidate_outcome({"dream_analysis": {"suggested_action": "merge"}}),
            "reviewable",
        )
        self.assertEqual(
            _candidate_outcome({
                "dream_analysis": {"suggested_action": "needs_more_evidence"},
            }),
            "deferred",
        )
        self.assertEqual(
            _candidate_outcome({"dream_analysis": {"suggested_action": "reject"}}),
            "rejected",
        )

    def test_eval_reports_expected_outcome_accuracy(self):
        deferred = {
            "id": "mem_deferred",
            "content": "User prefers Vim.",
            "type": "preference",
            "scope": "user",
            "score": 0.95,
            "tags": ["preference"],
            "evidence": [{"event_id": "event_1"}],
        }
        reviewable = {
            "id": "mem_reviewable",
            "content": "User prefers concise answers.",
            "type": "preference",
            "scope": "user",
            "score": 0.95,
            "tags": ["preference"],
            "evidence": [{"event_id": "event_1"}, {"event_id": "event_2"}],
        }
        rows = [
            {
                "id": "deferred",
                "events": [],
                "expected": [],
                "expected_outcomes": ["deferred"],
            },
            {
                "id": "reviewable",
                "events": [],
                "expected": [{
                    "content": "User prefers concise answers.",
                    "type": "preference",
                    "scope": "user",
                }],
                "expected_outcomes": ["reviewable"],
            },
            {
                "id": "none",
                "events": [],
                "expected": [],
                "expected_outcomes": ["none"],
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labeled.jsonl"
            path.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n"
                    for row in rows
                ),
                encoding="utf-8",
            )
            with patch("dream_memory.memory_eval._extract_candidates") as extract:
                extract.side_effect = [
                    ([deferred], None),
                    ([reviewable], None),
                    ([], None),
                ]
                result = evaluate_labeled_events(
                    path,
                    project=None,
                    mode="rules",
                )

        self.assertEqual(result["outcome_checked_rows"], 3)
        self.assertEqual(result["outcome_correct_rows"], 3)
        self.assertEqual(result["outcome_accuracy"], 1.0)
        self.assertEqual(result["outcome_mismatches"], [])

    def test_eval_rejects_invalid_expected_outcome(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labeled.jsonl"
            path.write_text(json.dumps({
                "id": "invalid",
                "events": [],
                "expected": [],
                "expected_outcomes": ["later"],
            }) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "unsupported expected outcome",
            ):
                evaluate_labeled_events(path, project=None, mode="rules")

    def test_eval_keeps_legacy_rows_without_outcome_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labeled.jsonl"
            path.write_text(json.dumps({
                "id": "legacy",
                "events": [],
                "expected": [],
            }) + "\n", encoding="utf-8")

            with patch(
                "dream_memory.memory_eval._extract_candidates",
                return_value=([], None),
            ):
                result = evaluate_labeled_events(
                    path,
                    project=None,
                    mode="rules",
                )

        self.assertEqual(result["outcome_checked_rows"], 0)
        self.assertEqual(result["outcome_correct_rows"], 0)
        self.assertEqual(result["outcome_accuracy"], 0.0)

    def test_evaluate_labeled_events_reports_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labeled.jsonl"
            path.write_text(json.dumps({
                "event": {"event_id": "event_1", "source": "codex", "role": "user", "event_type": "history_prompt", "content": "用户偏好中文回答。"},
                "expected": [{"content": "用户偏好中文回答", "type": "preference", "scope": "user"}],
            }, ensure_ascii=False) + "\n", encoding="utf-8")

            result = evaluate_labeled_events(path, project=None, mode="rules")

            self.assertEqual(result["expected_total"], 1)
            self.assertGreaterEqual(result["recall"], 0.0)
            self.assertIn("precision", result)

    def test_eval_matching_treats_chinese_wording_variants_as_match(self):
        from dream_memory.memory_eval import _matches_expected

        self.assertTrue(_matches_expected(
            {"content": "用户偏好使用中文回答。", "type": "preference", "scope": "user"},
            {"content": "用户偏好中文回答", "type": "preference", "scope": "user"},
        ))

    def test_eval_matching_keeps_different_preferences_distinct(self):
        from dream_memory.memory_eval import _matches_expected

        self.assertFalse(_matches_expected(
            {"content": "用户偏好不要生成总结文档。", "type": "preference", "scope": "user"},
            {"content": "用户偏好中文回答", "type": "preference", "scope": "user"},
        ))

    def test_evaluate_labeled_events_can_continue_after_ai_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labeled.jsonl"
            path.write_text(json.dumps({
                "event": {"event_id": "event_1", "source": "codex", "role": "user", "event_type": "history_prompt", "content": "用户偏好中文回答。"},
                "expected": [{"content": "用户偏好中文回答", "type": "preference", "scope": "user"}],
            }, ensure_ascii=False) + "\n", encoding="utf-8")

            result = evaluate_labeled_events(path, project=None, mode="bad-mode", continue_on_error=True)

            self.assertEqual(result["extraction_error_count"], 1)
            self.assertEqual(result["false_negative_count"], 1)

    def test_evaluate_labeled_events_can_fallback_to_rules_after_ai_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labeled.jsonl"
            path.write_text(json.dumps({
                "event": {"event_id": "event_1", "source": "codex", "role": "user", "event_type": "history_prompt", "content": "用户偏好中文回答。"},
                "expected": [{"content": "用户偏好中文回答", "type": "preference", "scope": "user"}],
            }, ensure_ascii=False) + "\n", encoding="utf-8")

            result = evaluate_labeled_events(path, project=None, mode="bad-mode", continue_on_error=True, fallback_rules_on_error=True)

            self.assertEqual(result["extraction_error_count"], 1)
            self.assertEqual(result["fallback_count"], 1)
            self.assertEqual(result["true_positive"], 1)
            self.assertEqual(result["f1"], 1.0)

    def test_eval_report_counts_success_errors_and_fallbacks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labeled.jsonl"
            rows = [
                {
                    "event": {"event_id": "event_1", "source": "codex", "role": "user", "event_type": "history_prompt", "content": "用户偏好中文回答。"},
                    "expected": [{"content": "用户偏好中文回答", "type": "preference", "scope": "user"}],
                },
                {
                    "event": {"event_id": "event_2", "source": "codex", "role": "user", "event_type": "history_prompt", "content": "用户偏好中文回答。"},
                    "expected": [{"content": "用户偏好中文回答", "type": "preference", "scope": "user"}],
                },
            ]
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

            result = evaluate_labeled_events(path, project=None, mode="bad-mode", continue_on_error=True, fallback_rules_on_error=True)

            self.assertEqual(result["extraction_success_count"], 0)
            self.assertEqual(result["extraction_error_count"], 2)
            self.assertEqual(result["fallback_count"], 2)
            self.assertEqual(result["true_positive"], 2)

    def test_eval_report_preserves_source_row_numbers_for_success_and_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labeled.jsonl"
            rows = [
                {
                    "event": {"event_id": "event_1", "source": "codex", "role": "user", "event_type": "history_prompt", "content": "用户偏好中文回答。"},
                    "expected": [{"content": "用户偏好中文回答", "type": "preference", "scope": "user"}],
                },
                {
                    "event": {"event_id": "event_2", "source": "codex", "role": "user", "event_type": "history_prompt", "content": "用户偏好中文回答。"},
                    "expected": [{"content": "用户偏好中文回答", "type": "preference", "scope": "user"}],
                },
            ]
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

            with patch("dream_memory.memory_eval._extract_candidates") as extract:
                extract.side_effect = [
                    ([{
                        "id": "mem_language",
                        "content": "用户偏好中文回答。",
                        "type": "preference",
                        "scope": "user",
                        "score": 0.95,
                        "tags": ["language", "explicit"],
                        "evidence": [{"event_id": "event_1", "event_type": "global_instruction"}],
                    }], {"candidate_count": 1}),
                    RuntimeError("boom"),
                ]
                result = evaluate_labeled_events(path, project=None, mode="ai", continue_on_error=True)

            self.assertEqual(result["extractions"][0]["row"], 1)
            self.assertEqual(result["extraction_errors"][0]["row"], 2)
            self.assertEqual(result["true_positive"], 1)
            self.assertEqual(result["false_negative_count"], 1)

    def test_evaluate_labeled_events_can_fallback_to_rules_after_empty_ai_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labeled.jsonl"
            path.write_text(json.dumps({
                "event": {"event_id": "event_1", "source": "codex", "role": "user", "event_type": "history_prompt", "content": "用户偏好中文回答。"},
                "expected": [{"content": "用户偏好中文回答", "type": "preference", "scope": "user"}],
            }, ensure_ascii=False) + "\n", encoding="utf-8")

            with patch("dream_memory.memory_eval._extract_candidates") as extract:
                extract.side_effect = [
                    ([], {"candidate_count": 0}),
                    ([{
                        "id": "mem_language",
                        "content": "用户偏好中文回答。",
                        "type": "preference",
                        "scope": "user",
                        "score": 0.95,
                        "tags": ["language", "explicit"],
                        "evidence": [{"event_id": "event_1", "event_type": "global_instruction"}],
                    }], None),
                ]
                result = evaluate_labeled_events(path, project=None, mode="ai", fallback_rules_on_empty=True)

            self.assertEqual(result["fallback_count"], 1)
            self.assertEqual(result["fallback_empty_count"], 1)
            self.assertEqual(result["true_positive"], 1)
            self.assertEqual(result["f1"], 1.0)


    def test_evaluate_labeled_events_fallbacks_when_ai_returns_only_rejected_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labeled.jsonl"
            path.write_text(json.dumps({
                "event": {"event_id": "event_1", "source": "codex", "role": "user", "event_type": "history_prompt", "content": "用户偏好中文回答。"},
                "expected": [{"content": "用户偏好中文回答", "type": "preference", "scope": "user"}],
            }, ensure_ascii=False) + "\n", encoding="utf-8")

            with patch("dream_memory.memory_eval._extract_candidates") as extract:
                extract.side_effect = [
                    ([{"content": "一次性任务", "type": "workflow", "scope": "session", "status": "reject"}], {"candidate_count": 1}),
                    ([{
                        "id": "mem_language",
                        "content": "用户偏好中文回答。",
                        "type": "preference",
                        "scope": "user",
                        "score": 0.95,
                        "tags": ["language", "explicit"],
                        "evidence": [{"event_id": "event_1", "event_type": "global_instruction"}],
                    }], None),
                ]
                result = evaluate_labeled_events(path, project=None, mode="ai", fallback_rules_on_empty=True)

        self.assertEqual(result["fallback_count"], 1)
        self.assertEqual(result["fallback_empty_count"], 1)
        self.assertEqual(result["true_positive"], 1)
        self.assertEqual(result["f1"], 1.0)

    def test_eval_report_includes_labeled_row_ids_for_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labeled.jsonl"
            rows = [
                {
                    "id": "match_row",
                    "event": {"event_id": "event_1", "source": "codex", "role": "user", "event_type": "history_prompt", "content": "用户偏好中文回答。"},
                    "expected": [{"content": "用户偏好中文回答", "type": "preference", "scope": "user"}],
                },
                {
                    "id": "error_row",
                    "event": {"event_id": "event_2", "source": "codex", "role": "user", "event_type": "history_prompt", "content": "用户偏好中文回答。"},
                    "expected": [{"content": "用户偏好中文回答", "type": "preference", "scope": "user"}],
                },
            ]
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

            with patch("dream_memory.memory_eval._extract_candidates") as extract:
                extract.side_effect = [
                    ([{
                        "id": "mem_language",
                        "content": "用户偏好中文回答。",
                        "type": "preference",
                        "scope": "user",
                        "score": 0.95,
                        "tags": ["language", "explicit"],
                        "evidence": [{"event_id": "event_1", "event_type": "global_instruction"}],
                    }], {"candidate_count": 1}),
                    RuntimeError("boom"),
                ]
                result = evaluate_labeled_events(path, project=None, mode="ai", continue_on_error=True)

            self.assertEqual(result["extractions"][0]["row_id"], "match_row")
            self.assertEqual(result["extraction_errors"][0]["row_id"], "error_row")
            self.assertEqual(result["false_negatives"][0]["row_id"], "error_row")


    def test_eval_ignores_rejected_candidates_for_precision(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labeled.jsonl"
            path.write_text(json.dumps({
                "id": "noise_row",
                "event": {"event_id": "e1", "content": "Tool execution result: pytest passed"},
                "expected": [],
            }, ensure_ascii=False) + "\n", encoding="utf-8")

            with patch("dream_memory.memory_eval._extract_candidates") as extract:
                extract.return_value = ([{"content": "一次性测试结果", "type": "workflow", "scope": "session", "status": "reject"}], {"candidate_count": 1})
                result = evaluate_labeled_events(path, project="/tmp/project", mode="ai")

        self.assertEqual(result["predicted_total"], 0)
        self.assertEqual(result["false_positive_count"], 0)
        self.assertEqual(result["precision"], 0.0)

    def test_eval_excludes_needs_more_evidence_from_predictions(self):
        candidate = {
            "id": "mem_editor",
            "content": "User prefers concise answers.",
            "type": "preference",
            "scope": "user",
            "score": 0.95,
            "tags": ["preference"],
            "evidence": [{"event_id": "event_1", "source": "codex"}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labeled.jsonl"
            path.write_text(json.dumps({
                "id": "weak",
                "events": [],
                "expected": [],
            }, ensure_ascii=False) + "\n", encoding="utf-8")

            with patch("dream_memory.memory_eval._extract_candidates", return_value=([candidate], None)):
                result = evaluate_labeled_events(path, project=None, mode="rules")

        self.assertEqual(result["predicted_total"], 0)
        self.assertEqual(result["deferred_candidate_count"], 1)
        self.assertEqual(result["false_positive_count"], 0)

    def test_eval_counts_two_event_candidate_as_prediction(self):
        candidate = {
            "id": "mem_editor",
            "content": "User prefers concise answers.",
            "type": "preference",
            "scope": "user",
            "score": 0.95,
            "tags": ["preference"],
            "evidence": [
                {"event_id": "event_1", "source": "codex"},
                {"event_id": "event_2", "source": "claude_code"},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labeled.jsonl"
            path.write_text(json.dumps({
                "id": "repeated",
                "events": [],
                "expected": [{
                    "content": "User prefers concise answers.",
                    "type": "preference",
                    "scope": "user",
                }],
            }, ensure_ascii=False) + "\n", encoding="utf-8")

            with patch("dream_memory.memory_eval._extract_candidates", return_value=([candidate], None)):
                result = evaluate_labeled_events(path, project=None, mode="rules")

        self.assertEqual(result["predicted_total"], 1)
        self.assertEqual(result["true_positive"], 1)
        self.assertEqual(result["deferred_candidate_count"], 0)


    def test_eval_report_includes_scored_candidate_count_after_filtering_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labeled.jsonl"
            path.write_text(json.dumps({
                "id": "noise_row",
                "event": {"event_id": "e1", "content": "Tool execution result: pytest passed"},
                "expected": [],
            }, ensure_ascii=False) + "\n", encoding="utf-8")

            with patch("dream_memory.memory_eval._extract_candidates") as extract:
                extract.return_value = (
                    [{"content": "一次性任务", "type": "workflow", "scope": "session", "status": "reject"}],
                    {"candidate_count": 1},
                )
                result = evaluate_labeled_events(path, project="/tmp/project", mode="ai")

        self.assertEqual(result["extractions"][0]["candidate_count"], 1)
        self.assertEqual(result["extractions"][0]["scored_candidate_count"], 0)
        self.assertEqual(result["predicted_total"], 0)


    def test_eval_report_includes_raw_and_scored_candidate_totals(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labeled.jsonl"
            rows = [
                {"id": "a", "event": {"event_id": "e1", "content": "用户偏好中文回答。"}, "expected": [{"content": "用户偏好中文回答", "type": "preference", "scope": "user"}]},
                {"id": "b", "event": {"event_id": "e2", "content": "噪声"}, "expected": []},
            ]
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

            with patch("dream_memory.memory_eval._extract_candidates") as extract:
                extract.side_effect = [
                    ([{
                        "id": "mem_language",
                        "content": "用户偏好中文回答。",
                        "type": "preference",
                        "scope": "user",
                        "score": 0.95,
                        "tags": ["language", "explicit"],
                        "evidence": [{"event_id": "event_1", "event_type": "global_instruction"}],
                    }], {"candidate_count": 1}),
                    ([{"content": "一次性任务", "type": "workflow", "scope": "session", "status": "reject"}], {"candidate_count": 1}),
                ]
                result = evaluate_labeled_events(path, project=None, mode="ai")

        self.assertEqual(result["raw_candidate_total"], 2)
        self.assertEqual(result["scored_candidate_total"], 1)
        self.assertEqual(result["predicted_total"], 1)


    def test_eval_report_separates_ai_raw_and_fallback_scored_candidate_totals(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labeled.jsonl"
            path.write_text(json.dumps({
                "id": "fallback_row",
                "event": {"event_id": "e1", "content": "用户偏好中文回答。"},
                "expected": [{"content": "用户偏好中文回答", "type": "preference", "scope": "user"}],
            }, ensure_ascii=False) + "\n", encoding="utf-8")

            with patch("dream_memory.memory_eval._extract_candidates") as extract:
                extract.side_effect = [
                    ([], {"candidate_count": 0}),
                    ([{
                        "id": "mem_language",
                        "content": "用户偏好中文回答。",
                        "type": "preference",
                        "scope": "user",
                        "score": 0.95,
                        "tags": ["language", "explicit"],
                        "evidence": [{"event_id": "event_1", "event_type": "global_instruction"}],
                    }], None),
                ]
                result = evaluate_labeled_events(path, project=None, mode="ai", fallback_rules_on_empty=True)

        self.assertEqual(result["raw_candidate_total"], 0)
        self.assertEqual(result["fallback_candidate_total"], 1)
        self.assertEqual(result["scored_candidate_total"], 1)
        self.assertEqual(result["extractions"][0]["fallback_candidate_count"], 1)


    def test_eval_report_separates_model_error_fallback_from_raw_candidate_total(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labeled.jsonl"
            path.write_text(json.dumps({
                "id": "error_fallback_row",
                "event": {"event_id": "e1", "content": "用户偏好中文回答。"},
                "expected": [{"content": "用户偏好中文回答", "type": "preference", "scope": "user"}],
            }, ensure_ascii=False) + "\n", encoding="utf-8")

            with patch("dream_memory.memory_eval._extract_candidates") as extract:
                extract.side_effect = [
                    RuntimeError("model down"),
                    ([{
                        "id": "mem_language",
                        "content": "用户偏好中文回答。",
                        "type": "preference",
                        "scope": "user",
                        "score": 0.95,
                        "tags": ["language", "explicit"],
                        "evidence": [{"event_id": "event_1", "event_type": "global_instruction"}],
                    }], None),
                ]
                result = evaluate_labeled_events(path, project=None, mode="ai", continue_on_error=True, fallback_rules_on_error=True)

        self.assertEqual(result["raw_candidate_total"], 0)
        self.assertEqual(result["fallback_candidate_total"], 1)
        self.assertEqual(result["scored_candidate_total"], 1)
        self.assertEqual(result["extractions"][0]["raw_candidate_count"], 0)
        self.assertEqual(result["extractions"][0]["fallback_candidate_count"], 1)


    def test_eval_report_includes_prompt_filter_counts_from_ai_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labeled.jsonl"
            path.write_text(json.dumps({
                "id": "prompt_counts",
                "event": {"event_id": "e1", "content": "用户偏好中文回答。"},
                "expected": [{"content": "用户偏好中文回答", "type": "preference", "scope": "user"}],
            }, ensure_ascii=False) + "\n", encoding="utf-8")

            with patch("dream_memory.memory_eval.agent_extract_memory_candidates", return_value={
                "dry_run": False,
                "candidates": [{"content": "用户偏好中文回答。", "type": "preference", "scope": "user"}],
                "atomic_facts": [{"statement": "用户偏好中文回答。"}],
                "model": "fake:model",
                "input_event_count": 2,
                "prompt_event_count": 1,
                "filtered_prompt_event_count": 1,
            }):
                result = evaluate_labeled_events(path, project=None, mode="ai")

        extraction = result["extractions"][0]
        self.assertEqual(extraction["input_event_count"], 2)
        self.assertEqual(extraction["prompt_event_count"], 1)
        self.assertEqual(extraction["filtered_prompt_event_count"], 1)

    def test_eval_report_wraps_false_positives_with_row_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labeled.jsonl"
            path.write_text(json.dumps({
                "id": "noise_row",
                "event": {"event_id": "event_1", "source": "codex", "role": "user", "event_type": "history_prompt", "content": "用户偏好中文回答。"},
                "expected": [],
            }, ensure_ascii=False) + "\n", encoding="utf-8")

            result = evaluate_labeled_events(path, project=None, mode="rules")

            self.assertEqual(result["false_positive_count"], 1)
            self.assertEqual(result["false_positives"][0]["row"], 1)
            self.assertEqual(result["false_positives"][0]["row_id"], "noise_row")
            self.assertIn("candidate", result["false_positives"][0])

    def test_eval_report_uses_row_id_fallback_when_label_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labeled.jsonl"
            path.write_text(json.dumps({
                "event": {"event_id": "event_1", "source": "codex", "role": "user", "event_type": "history_prompt", "content": "没有可抽取的内容"},
                "expected": [{"content": "用户偏好中文回答", "type": "preference", "scope": "user"}],
            }, ensure_ascii=False) + "\n", encoding="utf-8")

            result = evaluate_labeled_events(path, project=None, mode="rules")

            self.assertEqual(result["false_negative_count"], 1)
            self.assertEqual(result["false_negatives"][0]["row_id"], "row_1")

    def test_eval_cli_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labeled.jsonl"
            output = Path(tmp) / "eval.json"
            path.write_text(json.dumps({
                "event": {"event_id": "event_1", "source": "codex", "role": "user", "event_type": "history_prompt", "content": "用户偏好中文回答。"},
                "expected": [{"content": "用户偏好中文回答", "type": "preference", "scope": "user"}],
            }, ensure_ascii=False) + "\n", encoding="utf-8")

            exit_code = main(["eval", "--input", str(path), "--output", str(output)])

            self.assertEqual(exit_code, 0)
            self.assertTrue(output.exists())
            self.assertIn("f1", json.loads(output.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
