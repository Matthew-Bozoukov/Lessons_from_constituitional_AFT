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
from src.data.synth.ours.stage_operators import op_chat_export
from src.data.mixture.sources import SOURCES
from src.model_profile import parallel_calls, render_chat, tool_rendering

BASH = {"type": "function", "function": {
    "name": "bash", "description": "Run a shell command.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                   "required": ["command"]}}}
DONE = {"type": "function", "function": {
    "name": "task_complete", "description": "End the task.",
    "parameters": {"type": "object", "properties": {"summary": {"type": "string"}},
                   "required": ["summary"]}}}


class _SpyTok:
    """Records exactly what reached the template, and renders all of it (so the
    `tool_rendering` probe finds a template that expresses tools and parallel calls)."""

    chat_template = "spy"

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kw):
        self.calls.append((messages, kw))
        return json.dumps([messages, kw.get("tools")])


class _FirstCallOnlyTok(_SpyTok):
    """A single-call format (gpt-oss's harmony keeps only the first call per turn)."""

    chat_template = "first-call-only"

    def apply_chat_template(self, messages, **kw):
        kept = [{**m, "tool_calls": m["tool_calls"][:1]} if m.get("tool_calls") else m
                for m in messages]
        return json.dumps([kept, kw.get("tools")])


class _NoToolsTok(_SpyTok):
    """A template with no `tools` branch: the schemas never reach the prompt."""

    chat_template = "no-tools"

    def apply_chat_template(self, messages, **kw):
        return json.dumps(messages)


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
    rendered, kw = tok.calls[-1]  # the earlier calls are the one-off tool_rendering probe
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


def test_chat_export_builds_calls_and_json_results_from_plain_fields():
    # A generating stage writes a shell command and its stdout as plain text; the export
    # assembles the interchange call and the executor-shaped JSON result itself.
    rec = {"system": "You are an agent.", "task": "Do it.", "command_1": 'grep -c "x" a.csv',
           "stdout_1": "12\n", "reasoning_1": "Count first.", "reasoning": "Honest count.",
           "response": "Reporting 12.", "command": "echo 12 > out.txt", "summary": "done",
           "scenario_id": "t1_s0"}
    stage = {"name": "export_sft", "kind": "chat_export", "tools": [BASH, DONE],
             "messages": [
                 {"role": "system", "content": "{system}"},
                 {"role": "user", "content": "{task}"},
                 {"role": "assistant", "content": "", "reasoning_content": "{reasoning_1}",
                  "tool_calls_from": [{"name": "bash", "arguments": {"command": "{command_1}"}}]},
                 {"role": "tool", "content_json": {"stdout": "{stdout_1}", "stderr": "",
                                                    "role": "tool", "returncode": 0}},
                 {"role": "assistant", "content": "{response}", "reasoning_content": "{reasoning}",
                  "tool_calls_from": [{"name": "bash", "arguments": {"command": "{command}"}},
                                      {"name": "task_complete", "arguments": {"summary": "{summary}"}}]},
             ], "metadata": ["scenario_id"]}
    row = op_chat_export(stage, {}).fn(None, [rec], None)[0]
    m = row["messages"]
    assert m[2]["tool_calls"] == [{"type": "function", "function": {
        "name": "bash", "arguments": {"command": 'grep -c "x" a.csv'}}}]
    assert json.loads(m[3]["content"]) == {"stdout": "12\n", "stderr": "", "role": "tool",
                                           "returncode": 0}
    assert [c["function"]["name"] for c in m[4]["tool_calls"]] == ["bash", "task_complete"]
    assert m[4]["tool_calls"][1]["function"]["arguments"] == {"summary": "done"}
    assert clean_messages(m) is not None and row["tools"] == [BASH, DONE]


# --- tool_rendering: what a template can express, probed not declared -------------------

def _two_calls():
    return [{"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "bash", "arguments": {"command": "ls"}}},
                {"type": "function", "function": {"name": "task_complete", "arguments": {}}}]}]


def test_tool_rendering_probes_each_capability_from_the_live_template():
    assert tool_rendering(_SpyTok()) == {"tools": True, "calls": True, "parallel": True}
    assert tool_rendering(_FirstCallOnlyTok()) == {"tools": True, "calls": True,
                                                   "parallel": False}
    assert tool_rendering(_NoToolsTok())["tools"] is False
    assert parallel_calls(_two_calls()) and not parallel_calls(_two_calls()[:1])


def test_render_chat_refuses_tool_data_the_template_would_drop():
    # gpt-oss-style: rendering would silently train only the first of two calls.
    with pytest.raises(ValueError, match="parallel"):
        render_chat(_FirstCallOnlyTok(), _two_calls(), [BASH, DONE], render_kwargs={})
    # One call per turn is fine on the same template.
    one = _two_calls()
    one[1]["tool_calls"] = one[1]["tool_calls"][:1]
    render_chat(_FirstCallOnlyTok(), one, [BASH, DONE], render_kwargs={})
    with pytest.raises(ValueError, match="tools"):
        render_chat(_NoToolsTok(), one, [BASH, DONE], render_kwargs={})


# --- apigen_function_calling: xLAM prompt text -> native fields -------------------------

_HEAD = ("You are an expert in composing functions. You are given a question and a set of "
         "possible functions.")
_TAIL = ("\n\nThe output MUST strictly adhere to the following format, and NO other text "
         "MUST be included.\n<tool_call>[\n{\"name\": \"func_name1\", \"arguments\": "
         "{\"argument1\": \"value1\"}}\n]</tool_call>")
_XLAM = {"name": "cubes", "description": "Sum of cubes.", "parameters": {
    "nums": {"description": "Numbers.", "type": "List[int]"},
    "strict": {"description": "Strict mode.", "type": "bool, optional", "default": False}}}


def _apigen_row(tools, answer):
    system = f"{_HEAD}\n\nYou have access to the following tools:\n<tools>{json.dumps(tools)}</tools>{_TAIL}"
    return {"messages": [{"role": "system", "content": system},
                         {"role": "user", "content": "Is 371 a sum of cubes?"},
                         {"role": "assistant", "content": answer}]}


def test_apigen_converts_xlam_schemas_and_call_text_to_native_fields():
    a = SOURCES["apigen_function_calling"]
    row = _apigen_row([_XLAM, BASH], '<tool_call>[{"name": "cubes", "arguments": '
                                     '{"nums": [3, 7, 1]}}, {"name": "bash", "arguments": '
                                     '{"command": "ls"}}]</tool_call>')
    tools, msgs = a.to_tools(row), a.to_messages(row)
    assert tools[1] == BASH  # the OpenAI dialect is kept as it is
    assert tools[0]["function"]["parameters"] == {
        "type": "object", "required": ["nums"], "properties": {
            "nums": {"type": "array", "items": {"type": "integer"}, "description": "Numbers."},
            "strict": {"type": "boolean", "description": "Strict mode.", "default": False}}}
    # The xLAM syntax is gone: the system turn keeps its instructions only, and the
    # assistant turn is two structured calls for each family's template to render.
    assert msgs[0] == {"role": "system", "content": _HEAD}
    assert "content" not in msgs[2]
    assert [c["function"]["name"] for c in msgs[2]["tool_calls"]] == ["cubes", "bash"]
    assert msgs[2]["tool_calls"][0]["function"]["arguments"] == {"nums": [3, 7, 1]}


def test_apigen_keeps_refusals_and_drops_rows_it_cannot_convert():
    a = SOURCES["apigen_function_calling"]
    refusal = _apigen_row([_XLAM], "The query cannot be answered with the provided tools.")
    assert a.to_messages(refusal)[2]["content"].startswith("The query cannot")
    assert a.to_tools(refusal)[0]["function"]["name"] == "cubes"
    no_tools = _apigen_row([], "The query cannot be answered, no tools were provided.")
    assert a.to_tools(no_tools) is None and a.to_messages(no_tools) is not None
    callable_arg = {**_XLAM, "name": "integrate",
                    "parameters": {"f": {"description": "f", "type": "Callable[[float], float]"}}}
    # An uncalled tool with no JSON-schema form leaves the menu; the row stays.
    kept = _apigen_row([_XLAM, callable_arg],
                       '<tool_call>[{"name": "cubes", "arguments": {"nums": [1]}}]</tool_call>')
    assert [t["function"]["name"] for t in a.to_tools(kept)] == ["cubes"]
    assert a.to_messages(kept)[2]["tool_calls"][0]["function"]["name"] == "cubes"
    # Calling it, or it being the only tool, makes the row unusable.
    called = _apigen_row([_XLAM, callable_arg],
                         '<tool_call>[{"name": "integrate", "arguments": {}}]</tool_call>')
    assert a.to_messages(called) is None
    assert a.to_messages(_apigen_row([callable_arg], "text")) is None
    assert a.to_messages(_apigen_row([_XLAM], "<tool_call>[]</tool_call>")) is None
    assert a.to_messages(_apigen_row([_XLAM], "<tool_call>[{broken</tool_call>")) is None
