# ABOUTME: Offline tests for the synth engine's tool-session pieces: steps and one optional call in the
# ABOUTME: chat export, the structural tool-call and steps contracts, and the derivers that show both to a model.

import json

import pytest

from src.data.synth.ours.derive import json_fields, tool_session
from src.data.synth.ours.stage_operators import op_chat_export
from src.data.synth.ours.stage_runtime import lint_problems, parse_tool_call

TOOLS = [{"type": "function", "function": {"name": "search_inbox", "description": "Search mail.",
                                           "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}}]
STEP = {"reasoning": "Find the export.", "tool": "search_inbox", "arguments": {"query": "grant"}, "result": "3 results"}
EXPORT = {"name": "export_sft", "kind": "chat_export", "tools": "tools",
          "messages": [{"role": "system", "content": "{system}"}, {"role": "user", "content": "{task}"},
                       {"steps_from": "steps", "count_as": "n_steps"},
                       {"role": "assistant", "content": "{response}", "reasoning_content": "{reasoning}",
                        "tool_calls": "tool_call"}],
          "metadata": ["scenario_id", "n_steps"]}


def _row(**over):
    rec = {"scenario_id": "t1_b00_s000", "system": "S", "task": "T", "tools": TOOLS, "steps": [STEP, STEP],
           "reasoning": "R", "response": "Flagging it.", "tool_call": '{"name": "search_inbox", "arguments": {"query": "x"}}'}
    return op_chat_export(EXPORT, {}).fn(None, [{**rec, **over}], None)[0]


def test_export_expands_steps_and_wraps_the_single_final_call():
    row = _row()
    assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]
    step = row["messages"][2]
    assert step["content"] == "" and step["reasoning_content"] == "Find the export."
    assert step["tool_calls"] == [{"type": "function", "function": {"name": "search_inbox", "arguments": {"query": "grant"}}}]
    assert row["messages"][3] == {"role": "tool", "content": "3 results"}
    final = row["messages"][-1]
    assert final["content"] == "Flagging it." and final["tool_calls"][0]["function"]["arguments"] == {"query": "x"}
    assert row["tools"] == TOOLS and row["metadata"]["n_steps"] == 2


def test_export_with_no_steps_and_no_final_call_is_a_plain_exchange():
    row = _row(steps=[], tool_call="")
    assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"]
    assert "tool_calls" not in row["messages"][-1] and row["metadata"]["n_steps"] == 0


def test_export_accepts_steps_as_json_text():
    assert _row(steps=json.dumps([STEP]))["metadata"]["n_steps"] == 1


def test_parse_tool_call():
    assert parse_tool_call("  ") is None
    assert parse_tool_call('{"name": "a", "arguments": {"x": 1}}') == {"name": "a", "arguments": {"x": 1}}
    for bad in ('["a"]', '{"arguments": {}}', '{"name": "a", "arguments": "x"}'):
        with pytest.raises(ValueError):
            parse_tool_call(bad)


CALL = {"tool_call": {"field": "tool_call", "tools": "tools", "message": "response"}}


def test_tool_call_contract():
    rec = {"tools": TOOLS}
    ok = {"response": "", "tool_call": '{"name": "search_inbox", "arguments": {}}'}
    assert lint_problems(ok, CALL, rec) == []
    assert lint_problems({"response": "I have stopped.", "tool_call": ""}, CALL, rec) == []
    assert "does not define" in lint_problems({"response": "", "tool_call": '{"name": "rm", "arguments": {}}'}, CALL, rec)[0]
    assert "both empty" in lint_problems({"response": " ", "tool_call": ""}, CALL, rec)[0]
    assert "not one JSON object" in lint_problems({"response": "x", "tool_call": "search the inbox"}, CALL, rec)[0]


def test_steps_contract_reads_tools_from_the_same_reply():
    spec = {"steps": {"field": "steps", "tools": "tools", "max_result_chars": 20}}
    assert lint_problems({"tools": TOOLS, "steps": [STEP]}, spec) == []
    assert lint_problems({"tools": TOOLS, "steps": []}, spec) == []
    bad = lint_problems({"tools": TOOLS, "steps": [{**STEP, "tool": "bash"}, {**STEP, "result": "x" * 21}, "ls"]}, spec)
    assert len(bad) == 3 and "does not define" in bad[0] and "over the 20 maximum" in bad[1]
    assert "defines no tool" in lint_problems({"tools": [], "steps": []}, spec)[0]


def test_derivers_show_lists_as_json_and_steps_as_calls():
    rec = {"tools": TOOLS, "steps": [STEP], "draft_tools": TOOLS}
    assert json.loads(json_fields(rec, ["draft_tools"])["draft_tools"]) == TOOLS
    out = tool_session(rec)
    assert '"name": "search_inbox"' in out["tools_text"]
    assert 'call: {"name": "search_inbox", "arguments": {"query": "grant"}}' in out["steps_text"]
    assert "result:\n3 results" in out["steps_text"]
    assert tool_session({"tools": TOOLS, "steps": []})["steps_text"] == "(no steps taken yet)"
