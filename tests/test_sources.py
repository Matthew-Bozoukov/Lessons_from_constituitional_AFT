# ABOUTME: Offline unit tests for the mixture source adapters: registry integrity, the
# ABOUTME: shared message normaliser, and each non-passthrough schema mapping.

from src.data.mixture.sources import SOURCES, clean_messages, messages_passthrough
from src.data.mixture.sources import difficult_advice, lima


def test_registry_covers_every_table2_source_plus_locals():
    assert set(SOURCES) == {
        "no_robots", "tulu3_if", "numinamath_cot", "self_oss_instruct",
        "smol_constraints", "apigen_function_calling", "smol_summarize", "lima",
        "longalign", "tulu3", "difficult_advice",
    }
    for name, adapter in SOURCES.items():
        assert adapter.name == name and callable(adapter.to_messages)
    # The smoltalk subsets share one repo and differ by config name.
    smoltalk = [a for a in SOURCES.values() if a.repo == "HuggingFaceTB/smoltalk"]
    assert len(smoltalk) == 6 and len({a.hf_config for a in smoltalk}) == 6
    # difficult_advice is local-only: rows always come from a `path:`.
    assert SOURCES["difficult_advice"].repo is None


def test_clean_messages_validates_and_strips():
    good = [{"role": "user", "content": "q", "junk": 1},
            {"role": "assistant", "content": "a", "reasoning_content": "r"}]
    out = clean_messages(good)
    assert out == [{"role": "user", "content": "q"},
                   {"role": "assistant", "content": "a", "reasoning_content": "r"}]
    assert clean_messages(None) is None
    assert clean_messages([good[1]]) is None                       # too short
    assert clean_messages(good[::-1]) is None                      # ends on user
    assert clean_messages([{"role": "narrator", "content": "x"},
                           {"role": "assistant", "content": "a"}]) is None
    assert clean_messages([{"role": "system", "content": "s"},
                           {"role": "assistant", "content": "a"}]) is None  # no user turn
    assert clean_messages([{"role": "user", "content": "q"},
                           {"role": "assistant", "content": ""}]) is None   # empty content


def test_clean_messages_allows_toolcall_turns_with_empty_content():
    calls = [{"type": "function", "function": {"name": "f", "arguments": {}}}]
    msgs = [{"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": calls},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": "done"}]
    out = clean_messages(msgs)
    assert out is not None and out[1]["tool_calls"] == calls


def test_messages_passthrough_reads_the_messages_field():
    assert messages_passthrough({"messages": [
        {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]})
    assert messages_passthrough({"no_messages": True}) is None


def test_lima_maps_vicuna_conversations():
    row = {"conversations": [{"from": "human", "value": "q"},
                             {"from": "gpt", "value": "a"}]}
    assert lima.to_messages(row) == [{"role": "user", "content": "q"},
                                     {"role": "assistant", "content": "a"}]
    assert lima.to_messages({"conversations": "not a list"}) is None
    assert lima.to_messages({}) is None


def test_difficult_advice_maps_stage6_records():
    row = {"system": "s", "user": "u", "reasoning": "r", "response": "a", "trait_id": "t"}
    out = difficult_advice.to_messages(row)
    assert out == [{"role": "system", "content": "s"},
                   {"role": "user", "content": "u"},
                   {"role": "assistant", "content": "a", "reasoning_content": "r"}]
    # A record without reasoning still maps; the field is simply absent, never empty.
    bare = difficult_advice.to_messages({"user": "u", "response": "a"})
    assert bare is not None and "reasoning_content" not in bare[-1]
    assert difficult_advice.to_messages({"user": "u"}) is None
    # Already-converted {messages} pools pass straight through the normaliser.
    assert difficult_advice.to_messages({"messages": [
        {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]})
