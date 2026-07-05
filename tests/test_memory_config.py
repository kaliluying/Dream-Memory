import json
import tempfile
import unittest
from pathlib import Path

from deepagent_memory.memory_config import DEFAULT_MEMORY_CONFIG, load_memory_config, write_default_memory_config


class MemoryConfigTests(unittest.TestCase):
    def test_load_memory_config_uses_defaults_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = load_memory_config(Path(tmp) / "missing.json")

            self.assertEqual(config["model"], DEFAULT_MEMORY_CONFIG["model"])
            self.assertTrue(config["invoke_model"])
            self.assertEqual(config["output_dir"], ".deepagent/memory")

    def test_load_memory_config_merges_json_over_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"provider": "openai", "model": "gpt-4.1", "invoke_model": False}), encoding="utf-8")

            config = load_memory_config(path)

            self.assertEqual(config["provider"], "openai")
            self.assertEqual(config["model"], "gpt-4.1")
            self.assertFalse(config["invoke_model"])
            self.assertEqual(config["mode"], "ai")

    def test_write_default_memory_config_creates_editable_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_default_memory_config(Path(tmp) / "config.json")
            payload = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(payload["model"], DEFAULT_MEMORY_CONFIG["model"])
            self.assertIn("output_dir", payload)


if __name__ == "__main__":
    unittest.main()
