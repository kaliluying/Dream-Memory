import json
import tempfile
import unittest
from pathlib import Path

from dream_memory.memory_config import DEFAULT_MEMORY_CONFIG, load_memory_config, write_default_memory_config


class MemoryConfigTests(unittest.TestCase):
    def test_load_memory_config_uses_defaults_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_memory_config(Path(tmp) / "missing.json")

            self.assertEqual(config["model"], DEFAULT_MEMORY_CONFIG["model"])
            self.assertTrue(config["invoke_model"])
            self.assertEqual(config["output_dir"], ".dream-memory")

    def test_load_memory_config_merges_json_over_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"provider": "openai", "model": "gpt-4.1", "invoke_model": False}), encoding="utf-8")

            config = load_memory_config(path)

            self.assertEqual(config["provider"], "openai")
            self.assertEqual(config["model"], "gpt-4.1")
            self.assertFalse(config["invoke_model"])
            self.assertEqual(config["mode"], "ai")

    def test_load_memory_config_adds_implicit_model_profile_for_flat_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({
                "provider": "openai",
                "model": "gpt-4.1",
                "api_key_env": "OPENAI_API_KEY",
                "timeout_seconds": 45,
            }), encoding="utf-8")

            config = load_memory_config(path)

            self.assertEqual(config["models"]["default"]["provider"], "openai")
            self.assertEqual(config["models"]["default"]["model"], "gpt-4.1")
            self.assertEqual(config["models"]["default"]["api_key_env"], "OPENAI_API_KEY")
            self.assertEqual(config["models"]["default"]["timeout_seconds"], 45)
            self.assertEqual(config["model_policy"]["default_profile"], "default")
            self.assertEqual(config["model_policy"]["fallback_chain"], ["default"])

    def test_load_memory_config_preserves_named_profiles_and_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({
                "models": {
                    "primary": {
                        "provider": "anthropic",
                        "model": "claude-sonnet-4-6",
                        "api_key_env": "ANTHROPIC_API_KEY",
                    },
                    "backup": {
                        "provider": "openai",
                        "model": "gpt-4.1",
                        "api_key_env": "OPENAI_API_KEY",
                    },
                },
                "model_policy": {
                    "default_profile": "primary",
                    "fallback_chain": ["primary", "backup"],
                    "retry": {"max_attempts": 2},
                },
            }), encoding="utf-8")

            config = load_memory_config(path)

            self.assertEqual(config["models"]["primary"]["provider"], "anthropic")
            self.assertEqual(config["models"]["backup"]["provider"], "openai")
            self.assertEqual(config["model_policy"]["default_profile"], "primary")
            self.assertEqual(config["model_policy"]["fallback_chain"], ["primary", "backup"])
            self.assertEqual(config["model_policy"]["retry"]["max_attempts"], 2)
            self.assertEqual(config["model_policy"]["retry"]["retry_on_status"], [429, 500, 502, 503, 504])

    def test_write_default_memory_config_creates_editable_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_default_memory_config(Path(tmp) / "config.json")
            payload = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(payload["model"], DEFAULT_MEMORY_CONFIG["model"])
            self.assertIn("output_dir", payload)


if __name__ == "__main__":
    unittest.main()
