# ABOUTME: Unit tests for assistant-only loss masking over rendered Qwen chat text.
# ABOUTME: Run: uv run pytest tests/test_masking.py -q

from __future__ import annotations

from pathlib import Path


import pytest

from src.train.masking import (  # noqa: E402
    ASSISTANT_HEADER,
    THINK_LOSS_CLOSING_ONLY,
    assistant_spans,
    build_labels,
    check_think_loss_config,
    think_open_spans,
)

CHAT = (
    "<|im_start|>user\nhi<|im_end|>\n"
    "<|im_start|>assistant\nhello<|im_end|>\n"
    "<|im_start|>user\nbye<|im_end|>\n"
    "<|im_start|>assistant\nfarewell<|im_end|>\n"
)


def test_spans_cover_content_and_terminator_but_not_header():
    spans = assistant_spans(CHAT)
    assert [CHAT[s:e] for s, e in spans] == ["hello<|im_end|>", "farewell<|im_end|>"]
    for s, _ in spans:
        assert not CHAT[s:].startswith(ASSISTANT_HEADER)


def test_user_text_is_never_inside_a_span():
    spans = assistant_spans(CHAT)
    for probe in ("hi", "bye"):
        i = CHAT.index(probe)
        assert not any(s <= i < e for s, e in spans)


class _CharTokenizer:
    """Character-level stand-in whose offsets are exact, so masking is testable offline."""

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None,
                 return_offsets_mapping=False):
        ids = [ord(c) for c in text][: max_length if truncation else None]
        out = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(ids))]
        return out


def test_labels_unmask_exactly_the_assistant_characters():
    out = build_labels(CHAT, _CharTokenizer(), max_length=len(CHAT))
    assert len(out["input_ids"]) == len(out["labels"]) == len(CHAT)
    kept = "".join(chr(v) for v in out["labels"] if v != -100)
    assert kept == "hello<|im_end|>farewell<|im_end|>"


def test_supervised_fraction_is_a_minority_of_a_prompt_heavy_example():
    long_prompt = "<|im_start|>user\n" + ("x" * 500) + "<|im_end|>\n" \
                  "<|im_start|>assistant\nok<|im_end|>\n"
    out = build_labels(long_prompt, _CharTokenizer(), max_length=len(long_prompt))
    supervised = sum(1 for v in out["labels"] if v != -100)
    assert supervised == len("ok<|im_end|>")
    assert supervised < 0.1 * len(out["labels"])


class _QwenishTokenizer:
    """Stand-in that merges runs of newlines and the think tags into single tokens.

    A character tokenizer cannot express the case the think rule turns on: Qwen emits
    `\\n\\n` as ONE token, so it is only partly inside the `<think>` opener span and stays
    supervised, while a lone `\\n` is its own token and is masked. Verified against
    Qwen/Qwen3.6-27B: `<think>`=248068, `</think>`=248069, `\\n\\n`=271, `\\n`=198.
    """

    ATOMS = ("<think>", "</think>", "\n\n\n", "\n\n")

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None,
                 return_offsets_mapping=False):
        offsets, i = [], 0
        while i < len(text):
            for atom in self.ATOMS:
                if text.startswith(atom, i):
                    offsets.append((i, i + len(atom)))
                    i += len(atom)
                    break
            else:
                offsets.append((i, i + 1))
                i += 1
        if truncation and max_length is not None:
            offsets = offsets[:max_length]
        out = {"input_ids": [hash(text[a:b]) % 10**6 for a, b in offsets],
               "attention_mask": [1] * len(offsets)}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out


def _supervised_pieces(text: str) -> list[str]:
    """Return the text pieces that carry loss, in order."""
    tok = _QwenishTokenizer()
    out = build_labels(text, tok, max_length=10_000)
    offsets = tok(text, return_offsets_mapping=True)["offset_mapping"]
    return [text[a:b] for (a, b), v in zip(offsets, out["labels"]) if v != -100]


EMPTY_TURN = "<|im_start|>assistant\n<think>\n\n</think>\n\nHello there.<|im_end|>\n"
REASONING_TURN = (
    "<|im_start|>assistant\n<think>\nLet me check.\n</think>\n\nFound it.<|im_end|>\n"
)


def test_think_opener_is_never_supervised():
    for text in (EMPTY_TURN, REASONING_TURN):
        assert "<think>" not in _supervised_pieces(text)


def test_think_closer_is_always_supervised():
    for text in (EMPTY_TURN, REASONING_TURN):
        assert "</think>" in _supervised_pieces(text)


def test_empty_block_newlines_stay_supervised_because_they_are_one_token():
    # `\n\n` is a single token straddling the opener span, so it is not wholly inside
    # the hole and keeps its loss. This is the resolution of "mask the first \n only".
    assert _supervised_pieces(EMPTY_TURN)[:2] == ["\n\n", "</think>"]


def test_single_newline_after_opener_is_masked_with_it():
    # With real reasoning the newline IS its own token, wholly inside the hole.
    pieces = _supervised_pieces(REASONING_TURN)
    assert pieces[0] == "L", pieces[:3]        # reasoning starts immediately
    assert "</think>" in pieces


def test_reasoning_content_is_supervised():
    assert "".join(_supervised_pieces(REASONING_TURN)).startswith("Let me check.\n</think>")


def test_think_open_spans_ignore_the_closing_tag():
    spans = think_open_spans("<think>\n\n</think>")
    assert len(spans) == 1 and spans[0] == (0, 8)


def test_deprecated_and_missing_think_loss_configs_are_rejected():
    with pytest.raises(ValueError, match="required"):
        check_think_loss_config({})
    with pytest.raises(ValueError, match="mask_empty_think"):
        check_think_loss_config({"mask_empty_think": True})
    for mode in ("both", "skip_empty"):
        with pytest.raises(ValueError, match="deprecated"):
            check_think_loss_config({"think_loss": mode})
    with pytest.raises(ValueError, match="unknown"):
        check_think_loss_config({"think_loss": "nonsense"})
    check_think_loss_config({"think_loss": THINK_LOSS_CLOSING_ONLY})  # the supported rule
