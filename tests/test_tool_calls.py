# ABOUTME: Tool use as DATA across the pipeline: synth chat_export carries tool_calls/tools,
# ABOUTME: the interchange normalises them, and render_chat hands them to the template.

"""Offline (stub tokenizer) coverage of the tool-call contract. The real-template half —
that Qwen3.6 renders `tools` into the system turn and keeps tool output outside the
assistant span — lives in tests/test_masking_tokenizer.py, beside the other live checks.
"""

from __future__ import annotations

import json

import pytest

from src.data.mixture.sources import clean_messages, clean_tool_calls
from src.data.synth.stage_operators import op_chat_export
from src.model_profile import render_chat

BASH = {"type": "function", "function": {
    "name": "bash", "description": "Run a shell command.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                   "required": ["command"]}}}
DONE = {"type": "function", "function": {
    "name": "task_complete", "description": "End the task.",
    "parameters": {"type": "object", "properties": {"summary": {"type": "string"}},
                   "required": ["summary"]}}}


class _SpyTok:
    """Records exactly what reached the template."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kw):
        self.calls.append((messages, kw))
        return "rendered"


# --- clean_tool_calls: one stored shape -------------------------------------------------

def test_clean_tool_calls_parses_wire_form_strings_into_mappings():
    wire = [{"id": "call_1", "type": "function",
             "function": {"name": "bash", "arguments": json.dumps({"command": "ls"})}}]
    assert clean_tool_calls(wire) == [
        {"type": "function", "function": {"name": "bash", "arguments": {"command": "ls"}}}]


def test_clean_tool_calls_refuses_malformed_calls():
    assert clean_tool_calls([]) is None
    assert clean_tool_calls([{"type": "function"}]) is None                       # no function
    assert clean_tool_calls([{"function": {"name": 3, "arguments": {}}}]) is None  # name not str
    assert clean_tool_calls([{"function": {"name": "f", "arguments": "{not json"}}]) is None
    assert clean_tool_calls([{"function": {"name": "f", "arguments": [1, 2]}}]) is None
    # Absent arguments are an empty mapping, not a failure (a no-arg tool).
    assert clean_tool_calls([{"function": {"name": "f"}}])[0]["function"]["arguments"] == {}


def test_clean_messages_normalises_calls_and_refuses_calls_off_assistant_turns():
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"function": {"name": "bash", "arguments": '{"command": "ls"}'}}]},
            {"role": "tool", "content": "{\"stdout\": \"a\\n\"}"},
            {"role": "assistant", "content": "done"}]
    out = clean_messages(msgs)
    assert out[1]["tool_calls"][0]["function"]["arguments"] == {"command": "ls"}
    assert out[2] == {"role": "tool", "content": "{\"stdout\": \"a\\n\"}"}
    bad = [dict(m) for m in msgs]
    bad[2]["tool_calls"] = bad[1]["tool_calls"]  # a tool turn cannot make calls
    assert clean_messages(bad) is None


# --- chat_export: tool use leaves synth as data -----------------------------------------

def _record():
    return {
        "system": "You are an agent.", "user": "Do the task.",
        "explore_calls": [{"type": "function",
                           "function": {"name": "bash", "arguments": {"command": "ls"}}}],
        "explore_result": '{"stdout": "data.csv\\n", "returncode": 0}',
        "reasoning": "The number is what it is.", "response": "Filing the true figure.",
        # A tagged stage saves text: the calls may arrive as JSON, not a list.
        "final_calls": json.dumps([
            {"type": "function", "function": {"name": "bash", "arguments": {"command": "cat > r.md"}}},
            {"type": "function", "function": {"name": "task_complete", "arguments": {"summary": "done"}}}]),
        "tools": [BASH, DONE], "scenario_id": "t3_s0", "n_tools": 1,
    }


def _stage(tools_ref="tools"):
    return {
        "name": "export_sft", "kind": "chat_export", "tools": tools_ref,
        "messages": [
            {"role": "system", "content": "{system}"},
            {"role": "user", "content": "{user}"},
            {"role": "assistant", "content": "", "tool_calls": "explore_calls",
             "when": {"field": "n_tools", "min": 1}},
            {"role": "tool", "content": "{explore_result}",
             "when": {"field": "n_tools", "min": 1}},
            {"role": "assistant", "content": "{response}", "reasoning_content": "{reasoning}",
             "tool_calls": "final_calls"},
        ],
        "metadata": ["scenario_id"],
    }


def test_chat_export_carries_tool_calls_and_tools_as_structured_fields():
    row = op_chat_export(_stage(), {}).fn(None, [_record()], None)[0]
    roles = [m["role"] for m in row["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert row["messages"][2]["tool_calls"] == _record()["explore_calls"]
    assert row["messages"][4]["tool_calls"][1]["function"]["name"] == "task_complete"
    assert row["tools"] == [BASH, DONE]
    # The exported row IS a valid interchange row: the normaliser keeps every call and
    # only strips the empty content the exploration turn carries beside its calls.
    cleaned = clean_messages(row["messages"])
    assert [m["role"] for m in cleaned] == roles
    assert [m.get("tool_calls") for m in cleaned] == [m.get("tool_calls") for m in row["messages"]]
    assert "content" not in cleaned[2] and cleaned[4]["content"] == "Filing the true figure."


def test_chat_export_when_gating_drops_the_tool_exchange_and_accepts_literal_tools():
    rec = {**_record(), "n_tools": 0}
    row = op_chat_export(_stage(tools_ref=[BASH, DONE]), {}).fn(None, [rec], None)[0]
    assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"]
    assert row["tools"] == [BASH, DONE]


def test_chat_export_refuses_a_non_list_tool_field():
    rec = {**_record(), "tools": "bash"}
    with pytest.raises(ValueError, match="expected a list"):
        op_chat_export(_stage(), {}).fn(None, [rec], None)


# --- render_chat: the one render site ---------------------------------------------------

def test_render_chat_passes_tools_parses_wire_arguments_and_strips_padding():
    tok = _SpyTok()
    msgs = [{"role": "user", "content": "q", "reasoning_content": None},
            {"role": "assistant", "content": "", "reasoning_content": "r",
             "tool_calls": [{"type": "function", "function": {
                 "name": "bash", "arguments": '{"command": "ls"}'}}]}]
    tools = [{**BASH, "function": {**BASH["function"], "strict": None}}]  # loader padding
    render_chat(tok, msgs, tools, render_kwargs={"preserve_thinking": True})
    rendered, kw = tok.calls[0]
    assert "reasoning_content" not in rendered[0]
    assert rendered[1]["tool_calls"][0]["function"]["arguments"] == {"command": "ls"}
    assert kw["tools"] == [BASH] and kw["preserve_thinking"] is True
    assert kw["tokenize"] is False and kw["add_generation_prompt"] is False
    # The caller's rows are never mutated: the wire string is still a string upstream.
    assert isinstance(msgs[1]["tool_calls"][0]["function"]["arguments"], str)


def test_render_chat_omits_tools_for_a_plain_conversation():
    tok = _SpyTok()
    render_chat(tok, [{"role": "user", "content": "q"}], None,
                render_kwargs={}, tokenize=True, return_dict=True)
    _, kw = tok.calls[0]
    assert "tools" not in kw and kw["tokenize"] is True and kw["return_dict"] is True
