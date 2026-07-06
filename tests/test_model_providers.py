import json
import unittest
from unittest.mock import patch

from dream_memory.model_providers import (
    ModelAuthError,
    ModelHTTPError,
    ModelProfile,
    ModelPolicy,
    ModelRuntime,
    ModelRuntimeError,
    ProviderConfig,
    RetryPolicy,
    StaticModelProvider,
    parse_model_ref,
)


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

    def test_model_runtime_retries_retryable_http_error_then_succeeds(self):
        attempts = []

        def provider_factory(config):
            class Provider:
                def invoke(self, prompt):
                    attempts.append(config.model)
                    if len(attempts) == 1:
                        raise ModelHTTPError(429, "rate limited")
                    return '{"candidates":[]}'

            return Provider()

        runtime = ModelRuntime(provider_factory=provider_factory, sleeper=lambda _: None)
        result = runtime.invoke(
            "prompt",
            profiles={"primary": ModelProfile("primary", ProviderConfig("openai", "gpt-4.1", "OPENAI_API_KEY"))},
            policy=ModelPolicy(default_profile="primary", fallback_chain=["primary"], retry=RetryPolicy(max_attempts=2)),
        )

        self.assertEqual(result.text, '{"candidates":[]}')
        self.assertEqual(len(result.attempts), 2)
        self.assertEqual(result.selected_profile, "primary")
        self.assertTrue(result.attempts[0].retryable)
        self.assertFalse(result.attempts[0].ok)
        self.assertTrue(result.attempts[1].ok)

    def test_model_runtime_falls_back_to_backup_profile(self):
        def provider_factory(config):
            class Provider:
                def invoke(self, prompt):
                    if config.model == "bad-model":
                        raise ModelHTTPError(503, "unavailable")
                    return '{"candidates":[]}'

            return Provider()

        runtime = ModelRuntime(provider_factory=provider_factory, sleeper=lambda _: None)
        result = runtime.invoke(
            "prompt",
            profiles={
                "primary": ModelProfile("primary", ProviderConfig("anthropic", "bad-model", "A")),
                "backup": ModelProfile("backup", ProviderConfig("openai", "gpt-4.1", "B")),
            },
            policy=ModelPolicy(default_profile="primary", fallback_chain=["primary", "backup"], retry=RetryPolicy(max_attempts=1)),
        )

        self.assertEqual(result.text, '{"candidates":[]}')
        self.assertEqual(result.selected_profile, "backup")
        self.assertTrue(any(attempt.profile == "primary" and not attempt.ok for attempt in result.attempts))
        self.assertTrue(any(attempt.profile == "backup" and attempt.ok for attempt in result.attempts))

    def test_model_runtime_does_not_retry_auth_error(self):
        attempts = []

        def provider_factory(config):
            class Provider:
                def invoke(self, prompt):
                    attempts.append(config.model)
                    raise ModelAuthError("missing key")

            return Provider()

        runtime = ModelRuntime(provider_factory=provider_factory, sleeper=lambda _: None)

        with self.assertRaises(ModelRuntimeError) as raised:
            runtime.invoke(
                "prompt",
                profiles={"primary": ModelProfile("primary", ProviderConfig("openai", "gpt-4.1", "OPENAI_API_KEY"))},
                policy=ModelPolicy(default_profile="primary", fallback_chain=["primary"], retry=RetryPolicy(max_attempts=3)),
            )

        self.assertEqual(attempts, ["gpt-4.1"])
        self.assertIn("primary", str(raised.exception))

    def test_model_runtime_reports_attempt_summaries_when_all_profiles_fail(self):
        def provider_factory(config):
            class Provider:
                def invoke(self, prompt):
                    raise ModelHTTPError(503, f"{config.model} unavailable")

            return Provider()

        runtime = ModelRuntime(provider_factory=provider_factory, sleeper=lambda _: None)

        with self.assertRaises(ModelRuntimeError) as raised:
            runtime.invoke(
                "prompt",
                profiles={
                    "primary": ModelProfile("primary", ProviderConfig("anthropic", "bad-model", "A")),
                    "backup": ModelProfile("backup", ProviderConfig("openai", "also-bad", "B")),
                },
                policy=ModelPolicy(default_profile="primary", fallback_chain=["primary", "backup"], retry=RetryPolicy(max_attempts=1)),
            )

        self.assertIn("primary", str(raised.exception))
        self.assertIn("backup", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
