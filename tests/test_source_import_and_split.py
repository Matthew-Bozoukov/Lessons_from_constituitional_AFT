# ABOUTME: `chat_import` (a published dataset.jsonl as stage records) and `split` on a synth source: metadata
# ABOUTME: lifted, turns named, list keeps, ids at either level, local split files, and the refusals.
import json

import pytest

from src.data.mixture.split import select, write_split
from src.data.synth.ours.stage_operators import _source_split_labels, import_chat_rows

IMPORT = {
    "system": "system",
    "user": "user",
    "assistant": "response",
    "reasoning": "reasoning",
}


def _row(sid: str, trait: str = "t1") -> dict:
    return {
        "messages": [
            {"role": "system", "content": f"You work for {sid}."},
            {"role": "user", "content": "Should I?"},
            {"role": "assistant", "content": "No.", "reasoning_content": "Because."},
        ],
        "metadata": {"scenario_id": sid, "trait_id": trait, "chunk_ids": [1, 2]},
    }


def test_import_chat_rows_lifts_metadata_and_names_turns():
    recs = import_chat_rows([_row("s1"), _row("s2", "t2")], IMPORT)
    assert [r["scenario_id"] for r in recs] == ["s1", "s2"]
    assert recs[0]["system"] == "You work for s1." and recs[0]["user"] == "Should I?"
    assert recs[0]["response"] == "No." and recs[0]["reasoning"] == "Because."
    assert recs[1]["trait_id"] == "t2" and recs[1]["chunk_ids"] == [1, 2]
    # the record carries nothing but the metadata and the named fields
    assert set(recs[0]) == {
        "scenario_id",
        "trait_id",
        "chunk_ids",
        "system",
        "user",
        "response",
        "reasoning",
    }


def test_import_chat_rows_refuses_what_it_cannot_place():
    two_users = _row("s1")
    two_users["messages"].insert(2, {"role": "user", "content": "again"})
    with pytest.raises(ValueError, match="two 'user' turns"):
        import_chat_rows([two_users], IMPORT)
    tool_turn = _row("s1")
    tool_turn["messages"].append({"role": "tool", "content": "{}"})
    with pytest.raises(ValueError, match="'tool' turn"):
        import_chat_rows([tool_turn], IMPORT)
    clash = _row("s1")
    clash["metadata"]["response"] = "already here"
    with pytest.raises(ValueError, match="already holds"):
        import_chat_rows([clash], IMPORT)


def test_select_takes_a_list_of_pools_and_ids_at_either_level():
    labels = {"s1": "A", "s2": "B", "s3": "C"}
    published = [_row("s1"), _row("s2"), _row("s3")]
    assert [
        r["metadata"]["scenario_id"] for r in select(published, labels, "B", name="x")
    ] == ["s2"]
    assert [
        r["metadata"]["scenario_id"]
        for r in select(published, labels, ["B", "C"], name="x")
    ] == ["s2", "s3"]
    records = import_chat_rows(published, IMPORT)
    assert [
        r["scenario_id"] for r in select(records, labels, ["B", "C"], name="x")
    ] == ["s2", "s3"]
    with pytest.raises(ValueError, match="carry no split label"):
        select(published + [_row("s9")], labels, "B", name="x")
    with pytest.raises(ValueError, match="no row is in pool"):
        select(published, labels, ["D"], name="x")


def test_source_split_labels_from_a_local_source_dir(tmp_path):
    write_split(
        tmp_path / "splits" / "abc.json", {"s1": "A", "s2": "B"}, {"made_by": "test"}
    )
    labels, meta, where = _source_split_labels(
        {"file": "splits/abc.json"}, {"local_dir": str(tmp_path)}
    )
    assert labels == {"s1": "A", "s2": "B"} and meta == {"made_by": "test"}
    assert where.endswith("splits/abc.json")
    with pytest.raises(ValueError, match="pin a `revision:`"):
        _source_split_labels({"file": "splits/abc.json"}, {"hf_repo": "org/repo"})


def test_import_then_export_templates_reproduce_the_row():
    """The export templates da-tools.yaml uses, applied to an imported record, give the row back."""
    row = _row("s1")
    rec = import_chat_rows([row], IMPORT)[0]
    msgs = [
        {"role": "system", "content": "{system}".format(**rec)},
        {"role": "user", "content": "{user}".format(**rec)},
        {
            "role": "assistant",
            "content": "{response}".format(**rec),
            "reasoning_content": "{reasoning}".format(**rec),
        },
    ]
    assert msgs == row["messages"]
    assert json.dumps({k: rec[k] for k in row["metadata"]}) == json.dumps(
        row["metadata"]
    )
