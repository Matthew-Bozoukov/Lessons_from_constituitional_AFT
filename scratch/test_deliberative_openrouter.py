# ABOUTME: Offline regression checks for native reasoning and provenance in OpenRouter results.
# ABOUTME: Covers SDK and raw payload paths without contacting a model or reading credentials.

from copy import deepcopy
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletion

from src.infra.endpoints.openrouter import (
    ChatResult,
    EmptyCompletionError,
    OpenRouterClient,
    result_from_payload,
)


MODEL = "qwen/qwen3.6-27b"


def payload(message, *, finish_reason="stop", cost=0.012):
    return {
        "id": "gen-native-trace",
        "object": "chat.completion",
        "created": 0,
        "model": MODEL,
        "provider": "Alibaba",
        "choices": [{"index": 0, "message": {"role": "assistant", **message},
                     "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 13, "completion_tokens": 21, "total_tokens": 34,
                  "cost": cost, "prompt_tokens_details": {"cached_tokens": 2}},
    }


def call(data, path):
    if path == "payload":
        return result_from_payload(MODEL, deepcopy(data))
    client = OpenRouterClient(api_key="offline-test-key")
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return ChatCompletion.model_validate(data)

    client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = client.chat(MODEL, [{"role": "user", "content": "An original prompt"}],
                         extra_body={"reasoning": {"enabled": True, "exclude": False}})
    assert requests[0]["extra_body"]["provider"] == {
        "order": ["alibaba"], "allow_fallbacks": False,
    }
    assert requests[0]["extra_body"]["reasoning"] == {"enabled": True, "exclude": False}
    return result


@pytest.mark.parametrize("path", ["sdk", "payload"])
@pytest.mark.parametrize("fields,expected", [
    ({"reasoning": "  Native trace\n", "reasoning_content": "other",
      "reasoning_details": [{"type": "reasoning.text", "text": "duplicate"}]},
     "  Native trace\n"),
    ({"reasoning_content": "Alias trace"}, "Alias trace"),
    ({"reasoning_details": [
        {"type": "reasoning.summary", "summary": "not a trace"},
        {"type": "reasoning.text", "text": "First\n"},
        {"type": "reasoning.encrypted", "data": "not plaintext"},
        {"type": "reasoning.text", "text": "Second"},
    ]}, "First\nSecond"),
    ({"reasoning_details": [{"type": "reasoning.summary", "summary": "not native"}]}, ""),
])
def test_native_reasoning_and_provenance(path, fields, expected):
    result = call(payload({"content": "Unedited final answer", **fields}), path)
    assert result.content == "Unedited final answer"
    assert result.reasoning_content == expected
    assert result.provider == "Alibaba"
    assert result.response_id == "gen-native-trace"
    assert result.response_model == MODEL
    assert result.cost == 0.012
    assert result.prompt_tokens == 13
    assert result.completion_tokens == 21
    assert result.cached_tokens == 2
    assert result.finish_reason == "stop"


@pytest.mark.parametrize("path", ["sdk", "payload"])
def test_tools_only_completion_preserves_structured_calls(path):
    tools = [{"id": "call-1", "type": "function",
              "function": {"name": "lookup", "arguments": '{"x":1}'}}]
    result = call(payload({"content": None, "reasoning": "Need a lookup", "tool_calls": tools},
                          finish_reason="tool_calls"), path)
    assert result.content == ""
    assert result.reasoning_content == "Need a lookup"
    assert result.tool_calls == tools
    assert result.finish_reason == "tool_calls"


@pytest.mark.parametrize("path", ["sdk", "payload"])
@pytest.mark.parametrize("cost", [None, 0.0])
def test_missing_cost_is_distinct_from_free_completion(path, cost):
    data = payload({"content": "Answer"}, cost=cost)
    if cost is None:
        data["usage"].pop("cost")
    result = call(data, path)
    assert result.cost == cost
    assert result.reasoning_content == ""
    assert result.tool_calls == []


def test_legacy_construction_has_independent_tool_lists():
    first = ChatResult("answer", 1, 2, "stop")
    second = ChatResult("answer", 1, 2, "stop")
    first.tool_calls.append({"id": "one"})
    assert second.tool_calls == []
    assert second.cost is None
    assert second.reasoning_content == ""


def test_reasoning_only_still_fails_without_an_answer():
    with pytest.raises(EmptyCompletionError, match="empty content"):
        result_from_payload(MODEL, payload({"content": None, "reasoning": "Incomplete trace"},
                                         finish_reason="length"))
