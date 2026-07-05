import json
import unittest
from unittest.mock import patch

from dream_memory.model_providers import StaticModelProvider, parse_model_ref


class ModelProviderTests(unittest.TestCase):
    def test_parse_model_ref_splits_provider_prefix(self):
        config = parse_model_ref("openai:gpt-4.1")

        self.assertEqual(config.provider, "openai")
        self.assertEqual(config.model, "gpt-4.1")

    def test_parse_model_ref_uses_explicit_provider(self):
        config = parse_model_ref("claude-sonnet-4-6", provider="anthropic", api_key_env="MY_KEY")

        self.assertEqual(config.provider, "anthropic")
        self.assertEqual(config.model, "claude-sonnet-4-6")
        self.assertEqual(config.api_key_env, "MY_KEY")

    def test_static_model_provider_returns_fixed_response(self):
        provider = StaticModelProvider(json.dumps({"candidates": []}))

        self.assertEqual(provider.invoke("prompt"), '{"candidates": []}')


if __name__ == "__main__":
    unittest.main()
