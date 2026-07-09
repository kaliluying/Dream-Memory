import json
import tempfile
import unittest
from pathlib import Path

from dream_memory.memory_config import DEFAULT_MEMORY_CONFIG, load_memory_config, write_default_memory_config


class MemoryConfigTests(unittest.TestCase):
    def test_default_config_contains_eval_resilience_defaults(self):
        config = load_memory_config(None)

        self.assertIsNone(config["eval_max_rows"])
        self.assertIsNone(config["eval_max_attempts"])
        self.assertFalse(config["eval_continue_on_error"])
        self.assertFalse(config["eval_fallback_rules_on_error"])
        self.assertFalse(config["eval_fallback_rules_on_empty"])

    def test_load_memory_config_uses_defaults_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_memory_config(Path(tmp) / "missing.json")

            self.assertEqual(config["models"]["primary"]["model"], DEFAULT_MEMORY_CONFIG["models"]["primary"]["model"])
            self.assertTrue(config["invoke_model"])
            self.assertEqual(config["output_dir"], ".dream-memory")

    def test_load_memory_config_merges_json_over_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({
                "models": {
                    "primary": {
                        "provider": "openai",
                        "model": "gpt-4.1",
                        "api_key": "openai-key",
                    }
                },
                "model_policy": {"default_profile": "primary", "fallback_chain": ["primary"]},
                "invoke_model": False,
            }), encoding="utf-8")

            config = load_memory_config(path)

            self.assertEqual(config["models"]["primary"]["provider"], "openai")
            self.assertEqual(config["models"]["primary"]["model"], "gpt-4.1")
            self.assertEqual(config["models"]["primary"]["api_key"], "openai-key")
            self.assertFalse(config["invoke_model"])
            self.assertEqual(config["mode"], "ai")

    def test_load_memory_config_rejects_flat_model_config_without_profiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({
                "provider": "openai",
                "model": "gpt-4.1",
                "api_key_env": "OPENAI_API_KEY",
                "timeout_seconds": 45,
            }), encoding="utf-8")

            with self.assertRaises(ValueError) as raised:
                load_memory_config(path)

            self.assertIn("models", str(raised.exception))

    def test_load_memory_config_preserves_named_profiles_and_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({
                "models": {
                    "primary": {
                        "provider": "anthropic",
                        "model": "claude-sonnet-4-6",
                        "api_key": "anthropic-key",
                    },
                    "backup": {
                        "provider": "openai",
                        "model": "gpt-4.1",
                        "api_key": "openai-key",
                    },
                },
                "model_policy": {
                    "default_profile": "primary",
                    "fallback_chain": ["primary", "backup"],
                    "retry": {"max_attempts": 2, "switch_model_on_retry": True},
                },
            }), encoding="utf-8")

            config = load_memory_config(path)

            self.assertEqual(config["models"]["primary"]["provider"], "anthropic")
            self.assertEqual(config["models"]["backup"]["provider"], "openai")
            self.assertEqual(config["models"]["backup"]["api_key"], "openai-key")
            self.assertEqual(config["model_policy"]["default_profile"], "primary")
            self.assertEqual(config["model_policy"]["fallback_chain"], ["primary", "backup"])
            self.assertEqual(config["model_policy"]["retry"]["max_attempts"], 2)
            self.assertTrue(config["model_policy"]["retry"]["switch_model_on_retry"])
            self.assertEqual(config["model_policy"]["retry"]["retry_on_status"], [429, 500, 502, 503, 504])

    def test_write_default_memory_config_creates_editable_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_default_memory_config(Path(tmp) / "config.json")
            payload = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(payload["models"]["primary"]["model"], DEFAULT_MEMORY_CONFIG["models"]["primary"]["model"])
            self.assertIn("api_key", payload["models"]["primary"])
            self.assertEqual(payload["model_policy"]["fallback_chain"], ["primary"])
            self.assertIn("output_dir", payload)


if __name__ == "__main__":
    unittest.main()
