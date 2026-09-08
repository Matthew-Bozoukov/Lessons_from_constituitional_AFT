# ABOUTME: Unit tests for the pre-training mask gate: the independent supervised-text
# ABOUTME: parser, the think census, and the gate's agreement/refusal behaviour.

from __future__ import annotations

import pytest

from src.train.mask_gate import (
    GATE_SAMPLE,
    _gate_sample,
    expected_supervised_text,
    gate_generation_boundary,
)
from src.model_profile import model_profile
QWEN36_PROFILE = model_profile("qwen36")

THINK_PREFILL = QWEN36_PROFILE.prefill
EMPTY_THINK = QWEN36_PROFILE.empty_think
from src.model_profile import model_profile, think_census
QWEN36_PROFILE = model_profile("qwen36")


class _Tok:
    """Char tokenizer with the Qwen `\\n\\n` merge and a decode(), so the gate runs offline."""

    NL2 = 0x110000

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids, offsets, i = [], [], 0
        while i < len(text):
            if text.startswith("\n\n", i):
                ids.append(self.NL2)
                offsets.append((i, i + 2))
                i += 2
            else:
                ids.append(ord(text[i]))
                offsets.append((i, i + 1))
                i += 1
        out = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out

    def decode(self, ids):
        return "".join("\n\n" if v == self.NL2 else chr(v) for v in ids)


THINK_ROW = (
    "<|im_start|>user\nq1<|im_end|>\n"
    f"<|im_start|>assistant\n{THINK_PREFILL}why\n</think>\n\na1<|im_end|>\n"
    "<|im_start|>user\nq2<|im_end|>\n"
    f"<|im_start|>assistant\n{EMPTY_THINK}a2<|im_end|>\n"
)
NOTHINK_ROW = (
    "<|im_start|>user\nq<|im_end|>\n"
    "<|im_start|>assistant\nanswer<|im_end|>\n"
)


def test_expected_supervised_text_strips_each_turns_forced_head():
    # Turn 1 (real reasoning): only the prefill is forced. Turn 2 (empty): the WHOLE
    # marker is forced — the model never generates an empty close.
    assert expected_supervised_text(THINK_ROW, THINK_PREFILL, EMPTY_THINK) == (
        "why\n</think>\n\na1<|im_end|>" "a2<|im_end|>")
    assert expected_supervised_text(NOTHINK_ROW, THINK_PREFILL, EMPTY_THINK) == \
        "answer<|im_end|>"


def test_expected_supervised_text_final_keeps_only_the_last_turn():
    # A par row (un-repaired first reply as context) or an agentic row (exploration
    # turns as context) under `supervise: final`: the mask trains the last turn only,
    # and the independent expectation must say the same or the gate refuses every such
    # row -- which it did until 2026-09-05.
    assert expected_supervised_text(THINK_ROW, THINK_PREFILL, EMPTY_THINK,
                                    supervise="final") == "a2<|im_end|>"
    census = gate_generation_boundary([THINK_ROW], _Tok(), max_length=10_000,
                                      profile=QWEN36_PROFILE, thinking=True,
                                      supervise=["final"])
    assert census["turns"] == 2, "the census still polices every turn of the row"


def test_think_census_classifies_turns():
    assert think_census([THINK_ROW]) == {"turns": 2, "real": 1, "empty": 1, "absent": 0}
    assert think_census([NOTHINK_ROW]) == {"turns": 1, "real": 0, "empty": 0, "absent": 1}


def test_gate_passes_when_mask_and_parser_agree():
    census = gate_generation_boundary([THINK_ROW], _Tok(), max_length=10_000,
                                      profile=QWEN36_PROFILE, thinking=True)
    assert census["absent"] == 0


def test_gate_refuses_absent_think_turns_under_thinking():
    with pytest.raises(AssertionError, match="NO think block"):
        gate_generation_boundary([THINK_ROW, NOTHINK_ROW], _Tok(), max_length=10_000,
                                 profile=QWEN36_PROFILE, thinking=True)


def test_gate_refuses_think_blocks_under_nothink():
    with pytest.raises(AssertionError, match="thinking: false"):
        gate_generation_boundary([THINK_ROW], _Tok(), max_length=10_000,
                                 profile=QWEN36_PROFILE, thinking=False)
    gate_generation_boundary([NOTHINK_ROW], _Tok(), max_length=10_000,
                             profile=QWEN36_PROFILE, thinking=False)


def test_gate_catches_a_corrupted_mask(monkeypatch):
    # The gate exists to catch build_labels regressions; simulate one (a mask that
    # supervises everything, prefills included) and the decode comparison must fire.
    def broken(text, tokenizer, max_length, profile, supervise="all"):
        enc = tokenizer(text)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "labels": list(enc["input_ids"])}

    monkeypatch.setattr("src.train.masking.build_labels", broken)
    with pytest.raises(AssertionError, match="disagreement"):
        gate_generation_boundary([THINK_ROW], _Tok(), max_length=10_000,
                                 profile=QWEN36_PROFILE, thinking=True)


# --- supervise: "cot" -----------------------------------------------------------------

# The difficult-advice shape: one assistant turn, real trace, answer after the close.
COT_ROW = (
    "<|im_start|>user\nq<|im_end|>\n"
    f"<|im_start|>assistant\n{THINK_PREFILL}why\n</think>\n\nanswer<|im_end|>\n"
)


def test_expected_supervised_text_cot_stops_at_the_close():
    assert expected_supervised_text(COT_ROW, THINK_PREFILL, EMPTY_THINK,
                                    supervise="cot") == "why\n</think>"


def test_expected_supervised_text_cot_refuses_an_empty_marker():
    empty = ("<|im_start|>user\nq<|im_end|>\n"
             f"<|im_start|>assistant\n{EMPTY_THINK}answer<|im_end|>\n")
    with pytest.raises(AssertionError, match="empty marker"):
        expected_supervised_text(empty, THINK_PREFILL, EMPTY_THINK, supervise="cot")


def test_gate_verifies_the_mask_the_run_will_actually_build():
    # The independent parser and build_labels must agree under "cot" too — the whole
    # point of passing the modes through rather than gating everything as "all".
    census = gate_generation_boundary([COT_ROW], _Tok(), max_length=10_000,
                                      profile=QWEN36_PROFILE, thinking=True,
                                      supervise=["cot"])
    assert census["real"] == 1


def test_gate_catches_a_cot_mask_that_leaks_the_answer(monkeypatch):
    # The regression that matters: a "cot" row masked as if it were "all" still
    # supervises the answer. Gating every row as "all" would have blessed exactly this.
    from src.train.masking import build_labels as real

    def leaky(text, tokenizer, max_length, profile, supervise="all"):
        return real(text, tokenizer, max_length, profile, supervise="all")

    monkeypatch.setattr("src.train.masking.build_labels", leaky)
    with pytest.raises(AssertionError, match="disagreement"):
        gate_generation_boundary([COT_ROW], _Tok(), max_length=10_000,
                                 profile=QWEN36_PROFILE, thinking=True,
                                 supervise=["cot"])


def test_gate_sample_is_stratified_across_supervise_modes():
    # 1 cot row buried behind 200 "all" rows: a first-64 slice would never reach it.
    rows = [THINK_ROW] * 200 + [COT_ROW]
    modes = ["all"] * 200 + ["cot"]
    picked = _gate_sample(modes, GATE_SAMPLE)
    assert 200 in picked, "the minority mode must be sampled"
    assert sum(1 for i in picked if modes[i] == "all") == GATE_SAMPLE
    # And end to end: the gate reports having checked both modes.
    gate_generation_boundary(rows, _Tok(), max_length=10_000,
                             profile=QWEN36_PROFILE, thinking=True, supervise=modes)


def test_gate_defaults_every_row_to_all_when_no_modes_are_given():
    gate_generation_boundary([THINK_ROW], _Tok(), max_length=10_000,
                             profile=QWEN36_PROFILE, thinking=True, supervise=None)
    with pytest.raises(AssertionError, match="entries for"):
        gate_generation_boundary([THINK_ROW, COT_ROW], _Tok(), max_length=10_000,
                                 profile=QWEN36_PROFILE, thinking=True,
                                 supervise=["all"])
