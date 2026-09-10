# ABOUTME: Offline checks for exact final-corpus prompt intake and policy-free SFT export.
# ABOUTME: Exercise multi-turn context, row provenance, and unambiguous tool transport.

from copy import deepcopy
import json

import pytest

from src.data.synth.deliberative_alignment import data


SOURCE = {"repo": "org/final-synth", "file": "dataset.jsonl", "revision": "a" * 40}
TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {
    "type": "object", "properties": {"query": {"type": "string"}}}}}]


def _load(monkeypatch, tmp_path, rows):
    path = tmp_path / "dataset.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    calls = []

    def resolve(repo, filename, revision):
        calls.append((repo, filename, revision))
        return str(path), SOURCE

    monkeypatch.setattr(data, "resolve_dataset", resolve)
    result = data.load_prompts({"repo": SOURCE["repo"], "revision": "main"})
    assert calls == [(SOURCE["repo"], "dataset.jsonl", "main")]
    return result


def _simple_row():
    return {"messages": [{"role": "system", "content": "Original instructions."},
                         {"role": "user", "content": "The final revised prompt."},
                         {"role": "assistant", "content": "Old teacher answer.",
                          "reasoning_content": "Old teacher reasoning."}],
            "metadata": {"scenario_id": "s1", "trait_id": "t2"}}


def test_final_rows_are_kept_in_order_without_dedup(monkeypatch, tmp_path):
    first = _simple_row()
    second = deepcopy(first)
    second["metadata"]["scenario_id"] = "s2"
    records, provenance = _load(monkeypatch, tmp_path, [first, second])
    assert provenance == SOURCE
    assert [record["id"] for record in records] == ["0", "1"]
    assert [record["source_row"] for record in records] == [0, 1]
    assert [record["metadata"]["scenario_id"] for record in records] == ["s1", "s2"]
    assert all(record["messages"] == first["messages"][:-1] for record in records)


def test_augmentation_is_absent_from_export_and_input_remains_unchanged(monkeypatch, tmp_path):
    original = _simple_row()
    records, provenance = _load(monkeypatch, tmp_path, [original])
    before = deepcopy(records[0])
    request = data.generation_messages(records[0], "GENERATION CONSTITUTION")
    assert request[0]["content"] == "Original instructions.\n\nGENERATION CONSTITUTION"
    assert request[1] == original["messages"][1]
    final = data.export_row(records[0], {"content": "New answer.",
                                        "reasoning_content": "Constitution principle 1 applies."},
                            {"source": provenance})
    assert records[0] == before
    assert final["messages"][:-1] == original["messages"][:-1]
    assert "GENERATION CONSTITUTION" not in json.dumps(final)
    assert final["messages"][-1]["reasoning_content"] == "Constitution principle 1 applies."
    assert final["metadata"]["deliberative_alignment"] == {
        "source": SOURCE, "source_row": 0}
    assert final["supervise"] == "final"


def test_multi_turn_teacher_context_is_retained_but_only_new_final_is_supervised(monkeypatch, tmp_path):
    row = _simple_row()
    row["messages"] += [{"role": "user", "content": "What about this concern?"},
                        {"role": "assistant", "content": "Final teacher answer."}]
    records, provenance = _load(monkeypatch, tmp_path, [row])
    records[0]["metadata"]["supervise"] = "all"
    final = data.export_row(records[0], {"content": "New final answer."}, {"source": provenance})
    assert final["messages"][:-1] == row["messages"][:-1]
    assert final["supervise"] == "final"
    assert final["metadata"]["supervise"] == "final"
    assert records[0]["metadata"]["supervise"] == "all"


def test_no_system_context_gets_only_generation_system(monkeypatch, tmp_path):
    row = _simple_row()
    row["messages"].pop(0)
    records, provenance = _load(monkeypatch, tmp_path, [row])
    request = data.generation_messages(records[0], "SPEC")
    assert request[0] == {"role": "system", "content": "SPEC"}
    final = data.export_row(records[0], {"content": "New answer."}, {"source": provenance})
    assert final["messages"][0]["role"] == "user"


def _tool_row():
    return {"messages": [
        {"role": "user", "content": "Look up something."},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
            "name": "lookup", "arguments": {"query": "something"}}}]},
        {"role": "tool", "content": "Result."},
        {"role": "assistant", "content": "Old teacher answer."}], "metadata": {}, "tools": TOOLS}


def test_tool_transport_is_restored_only_in_generation_copy(monkeypatch, tmp_path):
    original = _tool_row()
    records, provenance = _load(monkeypatch, tmp_path, [original])
    request = data.generation_messages(records[0], "SPEC")
    call = request[2]["tool_calls"][0]
    assert call["id"] == request[3]["tool_call_id"]
    assert json.loads(call["function"]["arguments"]) == {"query": "something"}
    final = data.export_row(records[0], {"content": "New answer."}, {"source": provenance})
    assert final["messages"][:-1] == original["messages"][:-1]
    assert final["tools"] == TOOLS


def test_completion_tool_calls_are_interchange_objects(monkeypatch, tmp_path):
    row = _simple_row()
    row["tools"] = TOOLS
    records, provenance = _load(monkeypatch, tmp_path, [row])
    final = data.export_row(records[0], {"content": "", "tool_calls": [{
        "id": "wire-only-id", "type": "function", "function": {
            "name": "lookup", "arguments": '{"query":"second"}'}}]}, {"source": provenance})
    assert final["messages"][-1]["tool_calls"] == [{"type": "function", "function": {
        "name": "lookup", "arguments": {"query": "second"}}}]


def test_ambiguous_parallel_tool_results_fail_before_generation(monkeypatch, tmp_path):
    row = _tool_row()
    row["messages"][1]["tool_calls"] *= 2
    with pytest.raises(ValueError, match="row 0.*ambiguous"):
        _load(monkeypatch, tmp_path, [row])


def test_empty_tool_output_is_refused_instead_of_lost_by_mixture(monkeypatch, tmp_path):
    row = _tool_row()
    row["messages"][2]["content"] = ""
    with pytest.raises(ValueError, match="mixture interchange contract"):
        _load(monkeypatch, tmp_path, [row])


@pytest.mark.parametrize("change,match", [
    (lambda row: row["messages"].pop(), "end with an assistant"),
    (lambda row: row["messages"][1].update(content=[]), "string content"),
    (lambda row: row.update(metadata=None), "metadata must be an object"),
])
def test_invalid_final_rows_fail_instead_of_being_dropped(monkeypatch, tmp_path, change, match):
    row = _simple_row()
    change(row)
    with pytest.raises(ValueError, match=match):
        _load(monkeypatch, tmp_path, [row])


def test_stage_file_override_is_refused_before_hf_read(monkeypatch):
    monkeypatch.setattr(data, "resolve_dataset", lambda *a, **kw: pytest.fail("unexpected HF read"))
    with pytest.raises(ValueError, match="reads dataset.jsonl only"):
        data.load_prompts({"repo": "org/run", "file": "stages/stage_4_revise_prompts.jsonl"})
