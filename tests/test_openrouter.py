# ABOUTME: Offline tests for OpenRouterClient's completion handling — the empty-content
# ABOUTME: retry that re-routes around a blank provider, and the provider pin that keeps
# ABOUTME: a vendor's model on the vendor's own endpoint.

from types import SimpleNamespace

import pytest

from src.endpoints.openrouter import EmptyCompletionError, OpenRouterClient


def _resp(content, provider="ProviderX"):
    """A minimal stand-in for an OpenAI ChatCompletion object."""
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        provider=provider)


class _ScriptedOpenAI:
    """Returns each scripted response in turn (OpenRouter re-routes providers per call)."""

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
    # First provider blanks, retry re-routes to a working one — how the 2026-08-07
    # deepseek empties would have been absorbed instead of failing the item.
    c = _client([_resp(None, "BadProvider"), _resp("hello", "GoodProvider")])
    result = c.chat("deepseek/deepseek-chat-v3.1", [{"role": "user", "content": "hi"}])
    assert result.content == "hello" and c.client.calls == 2


def test_empty_content_on_every_attempt_finally_raises():
    # A model that blanks on every provider exhausts retries and surfaces a clear error.
    c = _client([_resp(None) for _ in range(6)])
    with pytest.raises(EmptyCompletionError, match="empty content"):
        c.chat("m", [{"role": "user", "content": "hi"}])
    assert c.client.calls == 6  # stop_after_attempt(6)


def test_normal_completion_does_not_retry():
    c = _client([_resp("answer")])
    assert c.chat("m", [{"role": "user", "content": "hi"}]).content == "answer"
    assert c.client.calls == 1


# --- the provider pin: the model you specify is the serving stack you get ------------
# Third-party hosts of the same weights filter differently (2026-08-14: Bedrock, then
# Google Vertex, refused difficult-advice prompts Anthropic itself serves), so vendor
# models default to the vendor's own endpoint with no fallbacks.


def test_vendor_models_pin_to_first_party_endpoint():
    c = _client([_resp("ok")])
    c.chat("anthropic/claude-sonnet-5", [{"role": "user", "content": "hi"}])
    assert c.client.requests[0]["extra_body"]["provider"] == \
        {"order": ["anthropic"], "allow_fallbacks": False}


def test_unmapped_models_stay_freely_routed():
    # Open-weight ids have no first-party host; adding an empty extra_body would still
    # change the request shape, so nothing is sent at all.
    c = _client([_resp("ok")])
    c.chat("deepseek/deepseek-chat-v3.1", [{"role": "user", "content": "hi"}])
    assert "extra_body" not in c.client.requests[0]


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
