# ABOUTME: Offline tests for OpenRouterClient's completion handling — the empty-content
# ABOUTME: retry, and the provider registry (configs/endpoints/providers.yaml) that keeps
# ABOUTME: EVERY model (open-weight included) on one explicit provider; unlisted ids raise.

from types import SimpleNamespace

import pytest

from src.endpoints.openrouter import (
    EmptyCompletionError,
    OpenRouterClient,
    ProviderRejectionError,
    provider_pin,
)


def _resp(content, provider="ProviderX"):
    """A minimal stand-in for an OpenAI ChatCompletion object."""
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        provider=provider)


class _ScriptedOpenAI:
    """Returns each scripted response in turn."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.requests.append(kw)
        r = self._responses[self.calls]
        self.calls += 1
        return r


def _client(responses):
    c = OpenRouterClient(api_key="sk-test")
    c.client = _ScriptedOpenAI(responses)
    return c


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    # tenacity's retry sleeps between attempts via time.sleep; collapse it so the
    # retry-path tests run instantly.
    monkeypatch.setattr("time.sleep", lambda *_: None)


def test_empty_content_retries_and_recovers():
    # A provider intermittently blanks (2026-08-07: deepseek, 4/20 concurrent calls);
    # the retry hits the same pinned provider again and absorbs the blip.
    c = _client([_resp(None), _resp("hello")])
    result = c.chat("qwen/qwen3-32b", [{"role": "user", "content": "hi"}])
    assert result.content == "hello" and c.client.calls == 2


def test_empty_content_on_every_attempt_finally_raises():
    # A model that blanks on every attempt exhausts retries and surfaces a clear error.
    c = _client([_resp(None) for _ in range(6)])
    with pytest.raises(EmptyCompletionError, match="empty content"):
        c.chat("anthropic/claude-sonnet-5", [{"role": "user", "content": "hi"}])
    assert c.client.calls == 6  # stop_after_attempt(6)


def test_normal_completion_does_not_retry():
    c = _client([_resp("answer")])
    m = c.chat("anthropic/claude-sonnet-5", [{"role": "user", "content": "hi"}])
    assert m.content == "answer"
    assert c.client.calls == 1


def _errbody_resp(code=400, message="Gemini blocked the request: PROHIBITED_CONTENT",
                  provider="Google"):
    """An HTTP-200 body with choices=None and an in-body provider error — the shape
    OpenRouter returns when Gemini's content filter blocks a request (2026-08-17)."""
    return SimpleNamespace(choices=None, usage=None, provider=provider,
                           error={"message": message, "code": code})


def _blank_choices_resp(provider="Google"):
    """choices=None with NO in-body error: an undiagnosable blank."""
    return SimpleNamespace(choices=None, usage=None, provider=provider, error=None)


def test_deterministic_rejection_fails_fast_with_payload():
    # A structured in-body 4xx (content filter, invalid request) is deterministic:
    # retrying re-bills the identical failure, so it surfaces on the FIRST attempt,
    # carrying the payload for typed refusal records.
    c = _client([_errbody_resp()])
    with pytest.raises(ProviderRejectionError, match="no choices") as ei:
        c.chat("google/gemini-3.7-flash", [{"role": "user", "content": "hi"}])
    assert c.client.calls == 1
    assert ei.value.provider == "Google"
    assert ei.value.provider_error["code"] == 400
    assert "PROHIBITED_CONTENT" in ei.value.provider_error["message"]


def test_blank_choices_retries_and_recovers():
    # choices=None with no payload is the mystery blank: transient, retried.
    c = _client([_blank_choices_resp(), _resp("fine now")])
    result = c.chat("google/gemini-3.7-flash", [{"role": "user", "content": "hi"}])
    assert result.content == "fine now" and c.client.calls == 2


def test_content_filter_finish_reason_fails_fast():
    # OpenAI-protocol hard filters return empty content with an explicit
    # finish_reason marker — deterministic, so no retries.
    blocked = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None),
                                 finish_reason="content_filter")],
        usage=None, provider="Azure")
    c = _client([blocked])
    with pytest.raises(ProviderRejectionError, match="content filter") as ei:
        c.chat("openai/gpt-4.1", [{"role": "user", "content": "hi"}])
    assert c.client.calls == 1
    assert ei.value.provider_error["code"] == "content_filter"


def test_filter_truncated_partial_content_is_rejected_not_returned():
    # Partial text + finish_reason=content_filter: silently truncated output would
    # poison downstream parses, so it is dropped and rejected loudly.
    truncated = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Half an ans"),
                                 finish_reason="content_filter")],
        usage=None, provider="Azure")
    c = _client([truncated])
    with pytest.raises(ProviderRejectionError, match="content filter") as ei:
        c.chat("openai/gpt-4.1", [{"role": "user", "content": "hi"}])
    assert c.client.calls == 1
    assert "11 chars of partial content dropped" in ei.value.provider_error["message"]


def test_empty_string_content_retries_like_none():
    # "" previously slipped through as a successful ChatResult and died at the
    # caller's parse gate; it is the same undiagnosable blank as None.
    c = _client([_resp(""), _resp("recovered")])
    result = c.chat("qwen/qwen3-32b", [{"role": "user", "content": "hi"}])
    assert result.content == "recovered" and c.client.calls == 2


def test_in_body_transient_code_retries():
    # An in-body 429/5xx is transient by HTTP semantics even though it arrived in a
    # 200 envelope — retried, not rejected.
    c = _client([_errbody_resp(code=502, message="upstream overloaded"),
                 _resp("ok")])
    result = c.chat("google/gemini-3.7-flash", [{"role": "user", "content": "hi"}])
    assert result.content == "ok" and c.client.calls == 2


# --- the provider registry: one model id = one provider, on every call ---------------
# Third-party hosts of the same weights filter differently (2026-08-14: Bedrock, then
# Google Vertex, refused difficult-advice prompts Anthropic itself serves), so every
# model pins to one explicit provider in configs/endpoints/providers.yaml and an
# unlisted id is a hard error — free routing is never the fallback.


def test_every_pinned_model_routes_to_its_one_provider():
    cases = {
        "anthropic/claude-sonnet-5": "anthropic",
        "openai/gpt-5.6-terra": "openai",
        "google/gemini-3.1-pro-preview": "google-ai-studio",
        "x-ai/grok-4.20": "xai",
        "nousresearch/hermes-4-405b": "nebius",
    }
    for model, provider in cases.items():
        assert provider_pin(model) == \
            {"order": [provider], "allow_fallbacks": False}, model


def test_the_pin_reaches_the_request_body():
    c = _client([_resp("ok")])
    c.chat("anthropic/claude-sonnet-5", [{"role": "user", "content": "hi"}])
    assert c.client.requests[0]["extra_body"]["provider"] == \
        {"order": ["anthropic"], "allow_fallbacks": False}


def test_unpinned_model_is_a_hard_error_not_free_routing():
    c = _client([_resp("ok")])
    with pytest.raises(ValueError, match="no provider pin"):
        c.chat("deepseek/deepseek-chat-v3.1", [{"role": "user", "content": "hi"}])
    assert c.client.calls == 0  # refused before any request left the process


def test_pins_are_exact_id_not_family():
    # Same "qwen/" family, different hosts: prefix inference would route the embedder
    # to a provider with no endpoint for it — exact ids only.
    assert provider_pin("qwen/qwen3-32b")["order"] == ["deepinfra"]
    assert provider_pin("qwen/qwen3-embedding-8b")["order"] == ["nebius"]


def test_caller_provider_block_beats_the_pin():
    c = _client([_resp("ok")])
    c.chat("anthropic/claude-sonnet-5", [{"role": "user", "content": "hi"}],
           extra_body={"provider": {"ignore": ["amazon-bedrock"]}})
    assert c.client.requests[0]["extra_body"]["provider"] == \
        {"ignore": ["amazon-bedrock"]}


def test_pin_merges_alongside_other_extra_body_keys():
    # A stage that disables hidden reasoning must not lose the pin, and vice versa.
    c = _client([_resp("ok")])
    c.chat("anthropic/claude-sonnet-5", [{"role": "user", "content": "hi"}],
           extra_body={"reasoning": {"enabled": False}})
    sent = c.client.requests[0]["extra_body"]
    assert sent["reasoning"] == {"enabled": False}
    assert sent["provider"] == {"order": ["anthropic"], "allow_fallbacks": False}
