import json
import unittest
from unittest.mock import patch

from dream_memory.model_providers import (
    ModelAuthError,
    ModelHTTPError,
    ModelProviderError,
    ModelProfile,
    ModelPolicy,
    ModelRuntime,
    ModelRuntimeError,
    ProviderConfig,
    RetryPolicy,
    StaticModelProvider,
    _openai_chat_completions_url,
    _post_json,
    _raise_provider_payload_error,
    list_provider_models,
    parse_model_ref,
    provider_diagnostics,
    runtime_parts_from_config,
)


class ModelProviderTests(unittest.TestCase):
    def test_parse_model_ref_splits_provider_prefix(self):
        config = parse_model_ref("openai:gpt-4.1")

        self.assertEqual(config.provider, "openai")
        self.assertEqual(config.model, "gpt-4.1")

    def test_parse_model_ref_keeps_colon_model_id_with_explicit_provider(self):
        config = parse_model_ref("nvidia/nemotron-3-ultra-550b-a55b:free", provider="openai")

        self.assertEqual(config.provider, "openai")
        self.assertEqual(config.model, "nvidia/nemotron-3-ultra-550b-a55b:free")

    def test_parse_model_ref_only_splits_known_provider_prefixes(self):
        config = parse_model_ref("nvidia/nemotron-3-ultra-550b-a55b:free")

        self.assertEqual(config.provider, "anthropic")
        self.assertEqual(config.model, "nvidia/nemotron-3-ultra-550b-a55b:free")

    def test_parse_model_ref_uses_explicit_provider(self):
        config = parse_model_ref("claude-sonnet-4-6", provider="anthropic", api_key="direct-key")

        self.assertEqual(config.provider, "anthropic")
        self.assertEqual(config.model, "claude-sonnet-4-6")
        self.assertEqual(config.api_key, "direct-key")

    def test_static_model_provider_returns_fixed_response(self):
        provider = StaticModelProvider(json.dumps({"candidates": []}))

        self.assertEqual(provider.invoke("prompt"), '{"candidates": []}')

    def test_runtime_parts_requires_named_model_profiles(self):
        with self.assertRaisesRegex(ValueError, "requires non-empty models"):
            runtime_parts_from_config({"provider": "openai", "model": "gpt-4.1"})

    def test_runtime_parts_requires_model_policy(self):
        with self.assertRaisesRegex(ValueError, "requires model_policy"):
            runtime_parts_from_config({"models": {"primary": {"provider": "openai", "model": "gpt-4.1"}}})

    def test_runtime_parts_rejects_unknown_fallback_profile(self):
        with self.assertRaisesRegex(ValueError, "unknown profiles: backup"):
            runtime_parts_from_config(
                {
                    "models": {"primary": {"provider": "openai", "model": "gpt-4.1"}},
                    "model_policy": {"default_profile": "primary", "fallback_chain": ["primary", "backup"]},
                }
            )

    def test_runtime_parts_builds_profiles_and_policy(self):
        profiles, policy = runtime_parts_from_config(
            {
                "models": {
                    "primary": {"provider": "openai", "model": "gpt-4.1", "api_key": "openai-key"},
                    "backup": {"provider": "anthropic", "model": "claude-sonnet-4-6", "api_key": "anthropic-key"},
                },
                "model_policy": {
                    "default_profile": "primary",
                    "fallback_chain": ["primary", "backup"],
                    "retry": {"max_attempts": 2, "switch_model_on_retry": True},
                },
            }
        )

        self.assertEqual(profiles["primary"].config.provider, "openai")
        self.assertEqual(profiles["primary"].config.api_key, "openai-key")
        self.assertEqual(profiles["backup"].config.model, "claude-sonnet-4-6")
        self.assertEqual(policy.default_profile, "primary")
        self.assertEqual(policy.fallback_chain, ["primary", "backup"])
        self.assertEqual(policy.retry.max_attempts, 2)
        self.assertTrue(policy.retry.switch_model_on_retry)

    def test_provider_diagnostics_accepts_direct_api_key_without_exposing_secret(self):
        payload = provider_diagnostics(
            provider="openai",
            model="gpt-4.1",
            api_key="sk-test-secret",
            api_key_env=None,
            base_url=None,
            timeout_seconds=60,
        )

        self.assertTrue(payload["ok"])
        self.assertTrue(payload["api_key_configured"])
        self.assertTrue(payload["api_key_present"])
        self.assertNotIn("sk-test-secret", json.dumps(payload))

    def test_provider_diagnostics_invokes_colon_model_with_explicit_provider(self):
        with patch("dream_memory.model_providers.build_model_provider") as build:
            class Provider:
                def invoke(self, prompt):
                    return '{"candidates":[]}'

            build.return_value = Provider()
            payload = provider_diagnostics(
                provider="openai",
                model="nvidia/nemotron-3-ultra-550b-a55b:free",
                api_key="sk-test-secret",
                api_key_env=None,
                base_url="http://localhost:3000",
                timeout_seconds=60,
                invoke=True,
            )

        self.assertTrue(payload["ok"])
        self.assertTrue(payload["invoked"])
        config = build.call_args.args[0]
        self.assertEqual(config.provider, "openai")
        self.assertEqual(config.model, "nvidia/nemotron-3-ultra-550b-a55b:free")

    def test_openai_base_url_accepts_root_v1_or_full_endpoint(self):
        default = "https://api.openai.com/v1/chat/completions"

        self.assertEqual(_openai_chat_completions_url(None, default), default)
        self.assertEqual(_openai_chat_completions_url("http://localhost:3000", default), "http://localhost:3000/v1/chat/completions")
        self.assertEqual(_openai_chat_completions_url("http://localhost:3000/v1", default), "http://localhost:3000/v1/chat/completions")
        self.assertEqual(
            _openai_chat_completions_url("http://localhost:3000/v1/chat/completions", default),
            "http://localhost:3000/v1/chat/completions",
        )

    def test_list_provider_models_reads_openai_compatible_models_endpoint(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps({"data": [{"id": "local-model-b"}, {"id": "local-model-a"}]}).encode("utf-8")

        with patch("urllib.request.urlopen", return_value=Response()) as urlopen:
            models = list_provider_models(
                ProviderConfig(provider="openai", model="ignored", api_key="sk-test", base_url="http://localhost:3000", timeout_seconds=3)
            )

        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://localhost:3000/v1/models")
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(request.headers["Authorization"], "Bearer sk-test")
        self.assertEqual(models, ["local-model-a", "local-model-b"])

    def test_list_provider_models_reads_anthropic_models_endpoint(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps({"data": [{"id": "claude-test-2"}, {"id": "claude-test-1"}]}).encode("utf-8")

        with patch("urllib.request.urlopen", return_value=Response()) as urlopen:
            models = list_provider_models(ProviderConfig(provider="anthropic", model="ignored", api_key="anthropic-key", timeout_seconds=3))

        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.anthropic.com/v1/models?limit=100")
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(request.headers["X-api-key"], "anthropic-key")
        self.assertEqual(models, ["claude-test-1", "claude-test-2"])

    def test_post_json_reports_non_json_response_with_url_preview(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b"<!doctype html><html></html>"

        with patch("urllib.request.urlopen", return_value=Response()):
            with self.assertRaisesRegex(ModelProviderError, "returned non-JSON response from http://localhost:3000"):
                _post_json("http://localhost:3000", {}, headers={}, timeout_seconds=1)

    def test_provider_payload_error_raises_for_200_json_error_payload(self):
        with self.assertRaisesRegex(ModelHTTPError, "Worker local total request limit reached") as raised:
            _raise_provider_payload_error(
                {
                    "error": {
                        "code": 502,
                        "message": "Upstream error from Nvidia: ResourceExhausted: Worker local total request limit reached (92/32)",
                    }
                }
            )

        self.assertEqual(raised.exception.status_code, 502)

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

    def test_model_runtime_can_switch_profiles_between_retry_attempts(self):
        attempts = []
        events = []

        def provider_factory(config):
            class Provider:
                def invoke(self, prompt):
                    attempts.append(config.model)
                    if config.model == "bad-model":
                        raise ModelHTTPError(429, "rate limited")
                    return '{"candidates":[]}'

            return Provider()

        runtime = ModelRuntime(provider_factory=provider_factory, sleeper=lambda _: None)
        result = runtime.invoke(
            "prompt",
            profiles={
                "primary": ModelProfile("primary", ProviderConfig("anthropic", "bad-model", "A")),
                "backup": ModelProfile("backup", ProviderConfig("openai", "good-model", "B")),
            },
            policy=ModelPolicy(
                default_profile="primary",
                fallback_chain=["primary", "backup"],
                retry=RetryPolicy(max_attempts=2, switch_model_on_retry=True),
            ),
            trace_callback=lambda event_type, payload: events.append((event_type, payload)),
        )

        self.assertEqual(attempts, ["bad-model", "good-model"])
        self.assertEqual(result.text, '{"candidates":[]}')
        self.assertEqual(result.selected_profile, "backup")
        self.assertEqual([attempt.profile for attempt in result.attempts], ["primary", "backup"])
        self.assertEqual(result.attempts[0].attempt, 1)
        self.assertEqual(result.attempts[1].attempt, 1)
        self.assertTrue(
            any(
                event_type == "model_fallback_used"
                and payload["from_profile"] == "primary"
                and payload["to_profile"] == "backup"
                and payload["reason"] == "retry_switch_model"
                for event_type, payload in events
            )
        )

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
