# ABOUTME: Unit tests for the pre-training mask gate: the independent supervised-text
# ABOUTME: parser, the think census, and the gate's agreement/refusal behaviour.

from __future__ import annotations

import pytest

from src.train.mask_gate import expected_supervised_text, gate_generation_boundary
from src.train.masking import EMPTY_THINK, THINK_PREFILL
from src.utils import QWEN36_PROFILE, think_census


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


def test_expected_supervised_text_strips_each_turns_prefill():
    assert expected_supervised_text(THINK_ROW, THINK_PREFILL) == (
        "why\n</think>\n\na1<|im_end|>" "\n</think>\n\na2<|im_end|>")
    assert expected_supervised_text(NOTHINK_ROW, THINK_PREFILL) == "answer<|im_end|>"


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
    def broken(text, tokenizer, max_length, prefill):
        enc = tokenizer(text)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "labels": list(enc["input_ids"])}

    monkeypatch.setattr("src.train.masking.build_labels", broken)
    with pytest.raises(AssertionError, match="disagreement"):
        gate_generation_boundary([THINK_ROW], _Tok(), max_length=10_000,
                                 profile=QWEN36_PROFILE, thinking=True)