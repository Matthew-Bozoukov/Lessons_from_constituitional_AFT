# ABOUTME: The `tool_schemas` lint contract: a tag that becomes a row's `tools` must parse as
# ABOUTME: OpenAI-style function schemas a real deployment could ship (bounds, names, params).
import json

from src.data.synth.ours.stage_runtime import lint_problems, tool_schema_problems


def _tool(name, required=("q",), props=("q",)):
    return {"type": "function", "function": {
        "name": name, "description": "Look something up.",
        "parameters": {"type": "object",
                       "properties": {p: {"type": "string"} for p in props},
                       "required": list(required)}}}


def test_well_formed_list_passes():
    text = json.dumps([_tool("get_rate_card"), _tool("convert_currency")])
    assert tool_schema_problems(text, {"min": 2, "max": 4}) == []


def test_not_json_and_not_a_list_are_refused():
    assert tool_schema_problems("[{", {})[0].startswith("is not JSON")
    assert "not a list" in tool_schema_problems(json.dumps(_tool("a_b")), {})[0]


def test_count_bounds():
    text = json.dumps([_tool("only_one")])
    assert tool_schema_problems(text, {"min": 2, "max": 4}) == [
        "holds 1 tools, outside [2, 4]"]
    five = json.dumps([_tool(f"tool_{i}") for i in range(5)])
    assert tool_schema_problems(five, {"min": 2, "max": 4}) == [
        "holds 5 tools, outside [2, 4]"]


def test_shape_name_and_parameter_defects():
    bad = [{"name": "flat_shape"}, _tool("CamelCase"), _tool("needs_x", required=("x",)),
           _tool("dup_name"), _tool("dup_name")]
    problems = tool_schema_problems(json.dumps(bad), {})
    assert any("tool 0 is not" in p for p in problems)
    assert any("'CamelCase' is not snake_case" in p for p in problems)
    assert any("'needs_x' requires parameters it does not define" in p for p in problems)
    assert "tool names repeat" in problems


def test_wired_into_lint_problems_per_field():
    spec = {"fields": ["tools"], "tool_schemas": {"min": 2, "max": 4}}
    assert lint_problems({"tools": json.dumps([_tool("a_b"), _tool("c_d")])}, spec) == []
    assert lint_problems({"tools": "nope"}, spec)[0].startswith("<tools> is not JSON")
    # a spec without the key leaves the tag alone
    assert lint_problems({"tools": "nope"}, {"fields": ["tools"]}) == []
