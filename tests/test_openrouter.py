# ABOUTME: Offline tests for OpenRouterClient's completion handling — the empty-content
# ABOUTME: retry, and the provider registry that keeps EVERY model (open-weight included)
# ABOUTME: on one explicit provider: pinned families route first-party, unpinned ids raise.

from types import SimpleNamespace

import pytest

from src.endpoints.openrouter import (
    PROVIDER_PINS,
    EmptyCompletionError,
    OpenRouterClient,
    pin_provider,
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
    result = c.chat("qwen/qwen3.6-27b", [{"role": "user", "content": "hi"}])
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


# --- the provider registry: one model id = one provider, on every call ---------------
# Third-party hosts of the same weights filter differently (2026-08-14: Bedrock, then
# Google Vertex, refused difficult-advice prompts Anthropic itself serves), so every
# family pins to one explicit provider and an unpinned id is a hard error — free
# routing is never the fallback.


def test_every_pinned_family_routes_to_its_one_provider():
    cases = {
        "anthropic/claude-sonnet-5": "anthropic",
        "openai/gpt-5.5": "openai",
        "google/gemini-3.1-pro-preview": "google-ai-studio",
        "x-ai/grok-4.20": "xai",
        "qwen/qwen3.6-27b": "alibaba",
        "moonshotai/kimi-k2.6": "moonshotai",
    }
    for model, provider in cases.items():
        assert pin_provider(model, None)["provider"] == \
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


def test_more_specific_pin_beats_the_family_pin(monkeypatch):
    # An exact-id entry (longest matching prefix) overrides its family's default,
    # e.g. one model of a family whose creator does not host that generation.
    monkeypatch.setitem(PROVIDER_PINS, "qwen/qwen3-32b",
                        {"order": ["deepinfra"], "allow_fallbacks": False})
    assert pin_provider("qwen/qwen3-32b", None)["provider"]["order"] == ["deepinfra"]
    assert pin_provider("qwen/qwen3.6-27b", None)["provider"]["order"] == ["alibaba"]


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
