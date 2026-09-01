# ABOUTME: Unit tests for assistant-only loss masking over rendered Qwen chat text.
# ABOUTME: Run: uv run pytest tests/test_masking.py -q

from __future__ import annotations

from pathlib import Path


import pytest

from src.model_profile import QWEN36_PROFILE
from src.train.masking import (  # noqa: E402
    answer_span,
    assistant_spans,
    build_labels,
    check_thinking_declaration,
    cot_span,
    forced_spans,
    model_profile,
)

# The suite exercises the RULE against one verified family's literals; every literal
# is drawn from the profile so the tests fail loudly if the registry and the rule drift.
ASSISTANT_HEADER = QWEN36_PROFILE.assistant_header
TURN_END = QWEN36_PROFILE.turn_end
THINK_PREFILL = QWEN36_PROFILE.prefill
EMPTY_THINK = QWEN36_PROFILE.empty_think
_TURN_KW = dict(header=ASSISTANT_HEADER, turn_end=TURN_END)

CHAT = (
    "<|im_start|>user\nhi<|im_end|>\n"
    "<|im_start|>assistant\nhello<|im_end|>\n"
    "<|im_start|>user\nbye<|im_end|>\n"
    "<|im_start|>assistant\nfarewell<|im_end|>\n"
)


def test_spans_cover_content_and_terminator_but_not_header():
    spans = assistant_spans(CHAT, **_TURN_KW)
    assert [CHAT[s:e] for s, e in spans] == ["hello<|im_end|>", "farewell<|im_end|>"]
    for s, _ in spans:
        assert not CHAT[s:].startswith(ASSISTANT_HEADER)


def test_user_text_is_never_inside_a_span():
    spans = assistant_spans(CHAT, **_TURN_KW)
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
    out = build_labels(CHAT, _CharTokenizer(), max_length=len(CHAT), profile=QWEN36_PROFILE)
    assert len(out["input_ids"]) == len(out["labels"]) == len(CHAT)
    kept = "".join(chr(v) for v in out["labels"] if v != -100)
    assert kept == "hello<|im_end|>farewell<|im_end|>"


def test_supervise_final_trains_only_the_last_assistant_turn():
    # The model-eval-model self-reflection shape: the first assistant turn (the response under
    # evaluation) is context, not a target.
    spans = assistant_spans(CHAT, supervise="final", **_TURN_KW)
    assert [CHAT[s:e] for s, e in spans] == ["farewell<|im_end|>"]
    out = build_labels(CHAT, _CharTokenizer(), max_length=len(CHAT), profile=QWEN36_PROFILE, supervise="final")
    kept = "".join(chr(v) for v in out["labels"] if v != -100)
    assert kept == "farewell<|im_end|>"


def test_supervise_rejects_unknown_modes():
    import pytest

    with pytest.raises(AssertionError, match="unknown supervise mode"):
        assistant_spans(CHAT, supervise="first", **_TURN_KW)


def test_supervised_fraction_is_a_minority_of_a_prompt_heavy_example():
    long_prompt = "<|im_start|>user\n" + ("x" * 500) + "<|im_end|>\n" \
                  "<|im_start|>assistant\nok<|im_end|>\n"
    out = build_labels(long_prompt, _CharTokenizer(), max_length=len(long_prompt), profile=QWEN36_PROFILE)
    supervised = sum(1 for v in out["labels"] if v != -100)
    assert supervised == len("ok<|im_end|>")
    assert supervised < 0.1 * len(out["labels"])


class _MergingTokenizer(_CharTokenizer):
    """Char tokenizer that merges `\\n\\n` into one token, reproducing Qwen's hazard.

    In the real tokenizer the empty block's two newlines weld into a single token, and
    the generation boundary runs through the middle of it. Segment cutting is what makes
    the rule expressible at all, so the offline suite must include a tokenizer that
    would merge if the cut were missing.
    """

    NL2 = 0x110000  # sentinel above any ord(); decodes as "\n\n" in _kept below

    def __call__(self, text, add_special_tokens=False, truncation=False, max_length=None,
                 return_offsets_mapping=False):
        ids, offsets = [], []
        i = 0
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


def _kept(out) -> str:
    """Reconstruct the supervised text from labels (NL2 sentinel -> its two newlines)."""
    return "".join("\n\n" if v == _MergingTokenizer.NL2 else chr(v)
                   for v in out["labels"] if v != -100)


THINK_ROW = (
    "<|im_start|>user\nq<|im_end|>\n"
    f"<|im_start|>assistant\n{THINK_PREFILL}reasoning\n</think>\n\nanswer<|im_end|>\n"
)
EMPTY_ROW = (
    "<|im_start|>user\nq<|im_end|>\n"
    f"<|im_start|>assistant\n{EMPTY_THINK}answer<|im_end|>\n"
)


def test_prefill_is_masked_and_reasoning_and_closer_are_supervised():
    out = build_labels(THINK_ROW, _MergingTokenizer(), max_length=len(THINK_ROW), profile=QWEN36_PROFILE)
    assert _kept(out) == "reasoning\n</think>\n\nanswer<|im_end|>"


def test_empty_marker_is_wholly_masked():
    # A healthy model never generates an empty close (probe, LOG 2026-08-04): the whole
    # marker is forced context in every serving configuration, so none of it — opener,
    # newlines, closer — may carry loss. Supervision starts at the answer.
    out = build_labels(EMPTY_ROW, _MergingTokenizer(), max_length=len(EMPTY_ROW), profile=QWEN36_PROFILE)
    assert _kept(out) == "answer<|im_end|>"
    assert _MergingTokenizer.NL2 in out["input_ids"]  # in-marker merges still happen


def test_forced_span_covers_marker_or_prefill_per_turn():
    spans = assistant_spans(EMPTY_ROW, **_TURN_KW)
    (span,) = forced_spans(EMPTY_ROW, spans, THINK_PREFILL, EMPTY_THINK)
    assert EMPTY_ROW[span[0]:span[1]] == EMPTY_THINK
    spans = assistant_spans(THINK_ROW, **_TURN_KW)
    (span,) = forced_spans(THINK_ROW, spans, THINK_PREFILL, EMPTY_THINK)
    assert THINK_ROW[span[0]:span[1]] == THINK_PREFILL


def test_turns_without_a_think_block_have_no_forced_span():
    assert forced_spans(CHAT, assistant_spans(CHAT, **_TURN_KW), THINK_PREFILL, EMPTY_THINK) == []
    out = build_labels(CHAT, _MergingTokenizer(), max_length=len(CHAT), profile=QWEN36_PROFILE)
    assert _kept(out) == "hello<|im_end|>farewell<|im_end|>"


MULTI_TURN_ROW = (
    "<|im_start|>user\nq1<|im_end|>\n"
    f"<|im_start|>assistant\n{THINK_PREFILL}first thoughts\n</think>\n\na1<|im_end|>\n"
    "<|im_start|>user\nq2<|im_end|>\n"
    f"<|im_start|>assistant\n{EMPTY_THINK}a2<|im_end|>\n"
    "<|im_start|>user\nq3<|im_end|>\n"
    f"<|im_start|>assistant\n{THINK_PREFILL}third thoughts\n</think>\n\na3<|im_end|>\n"
)


def test_every_turn_of_a_multiturn_row_masks_its_own_forced_head():
    # The preserve-thinking policy puts a think block on EVERY assistant turn: reasoning
    # turns mask the prefill and supervise trace + close; the empty middle turn masks its
    # whole marker and supervises only the answer.
    spans = assistant_spans(MULTI_TURN_ROW, **_TURN_KW)
    assert len(forced_spans(MULTI_TURN_ROW, spans, THINK_PREFILL, EMPTY_THINK)) == 3
    out = build_labels(MULTI_TURN_ROW, _MergingTokenizer(), max_length=len(MULTI_TURN_ROW), profile=QWEN36_PROFILE)
    assert _kept(out) == (
        "first thoughts\n</think>\n\na1<|im_end|>"
        "a2<|im_end|>"
        "third thoughts\n</think>\n\na3<|im_end|>"
    )


def test_family_gate_refuses_unverified_prefills():
    assert model_profile("Qwen/Qwen3.6-27B").prefill == THINK_PREFILL
    with pytest.raises(ValueError, match="Qwen3 prefills nothing"):
        model_profile("Qwen/Qwen3-32B")


def test_check_thinking_declaration():
    real = {"text": "<|im_start|>assistant\n<think>\nreal trace\n</think>\nanswer<|im_end|>"}
    empty = {"text": f"<|im_start|>assistant\n{EMPTY_THINK}answer<|im_end|>"}
    plain = {"text": "<|im_start|>assistant\nanswer<|im_end|>"}
    msgs_think = {"messages": [{"role": "assistant", "content": "a", "reasoning_content": "hm"}]}
    msgs_plain = {"messages": [{"role": "assistant", "content": "a"}]}

    check_thinking_declaration([real, plain], thinking=True, empty_think=EMPTY_THINK)
    # Empty markers are fine under thinking=true: the generation-boundary mask supervises
    # their close, so they no longer need a masking flag to be safe.
    check_thinking_declaration([real, empty], thinking=True, empty_think=EMPTY_THINK)
    check_thinking_declaration([msgs_think], thinking=True)
    check_thinking_declaration([plain, msgs_plain], thinking=False, empty_think=EMPTY_THINK)

    with pytest.raises(AssertionError, match="no row carries a real reasoning trace"):
        check_thinking_declaration([plain], thinking=True, empty_think=EMPTY_THINK)
    with pytest.raises(AssertionError, match="mislabels"):
        check_thinking_declaration([real], thinking=False, empty_think=EMPTY_THINK)


def test_every_train_config_declares_thinking():
    from pathlib import Path

    from omegaconf import OmegaConf

    configs = sorted(Path("configs/train").glob("lora_*.yaml"))
    assert configs
    for path in configs:
        cfg = OmegaConf.load(path)
        assert "thinking" in cfg and isinstance(cfg.thinking, bool), path.name


def test_mask_spans_unsupervise_only_the_given_characters():
    # The ablation hook: one property's characters lose their loss, everything else in the
    # same assistant turn keeps it, and the token stream is untouched.
    spans = [(CHAT.index("hello"), CHAT.index("hello") + len("hello"))]
    kw = dict(max_length=len(CHAT), profile=QWEN36_PROFILE)
    base = build_labels(CHAT, _CharTokenizer(), **kw)
    out = build_labels(CHAT, _CharTokenizer(), mask_spans=spans, **kw)
    assert out["input_ids"] == base["input_ids"], "masking must not change the token stream"
    kept = "".join(chr(v) for v in out["labels"] if v != -100)
    assert kept == "<|im_end|>farewell<|im_end|>"


def test_mask_spans_default_is_a_no_op():
    kw = dict(max_length=len(CHAT), profile=QWEN36_PROFILE)
    base = build_labels(CHAT, _CharTokenizer(), **kw)
    for empty in (None, []):
        assert build_labels(CHAT, _CharTokenizer(), mask_spans=empty, **kw) == base


def test_mask_spans_over_a_whole_turn_fails_loudly():
    # Masking everything supervised would train on nothing; the assert must fire rather
    # than hand the trainer a batch whose loss is NaN.
    whole = [(0, len(CHAT))]
    with pytest.raises(AssertionError, match="no supervised token"):
        build_labels(CHAT, _CharTokenizer(), max_length=len(CHAT),
                     profile=QWEN36_PROFILE, mask_spans=whole)


# --- supervise: "cot" — the CoT-only arm --------------------------------------------
# The answer is CUT, not merely unsupervised, so these tests assert on `input_ids` as
# well as on the labels: the compute saving is half the point of the mode and a
# label-only implementation would pass a labels-only test.


def test_cot_span_locates_the_final_turns_reasoning():
    start, end = cot_span(THINK_ROW, header=ASSISTANT_HEADER, prefill=THINK_PREFILL,
                          empty_think=EMPTY_THINK,
                          think_close=QWEN36_PROFILE.think_close)
    assert THINK_ROW[start:end] == f"{THINK_PREFILL}reasoning\n</think>"


def test_cot_supervises_the_trace_and_close_and_nothing_else():
    out = build_labels(THINK_ROW, _MergingTokenizer(), max_length=len(THINK_ROW),
                       profile=QWEN36_PROFILE, supervise="cot")
    assert _kept(out) == "reasoning\n</think>"


def _text(out) -> str:
    """Reconstruct the whole token stream (supervised or not) as a string."""
    return "".join("\n\n" if v == _MergingTokenizer.NL2 else chr(v)
                   for v in out["input_ids"])


def test_cot_drops_the_answer_from_the_token_stream_entirely():
    # Not "answer tokens are -100" — they must be ABSENT, or the forward pass still
    # pays for them and the mode buys nothing.
    tok = _MergingTokenizer()
    full = build_labels(THINK_ROW, tok, max_length=len(THINK_ROW), profile=QWEN36_PROFILE)
    cot = build_labels(THINK_ROW, tok, max_length=len(THINK_ROW),
                       profile=QWEN36_PROFILE, supervise="cot")
    assert len(cot["input_ids"]) < len(full["input_ids"])
    assert "answer" not in _text(cot)
    assert _text(cot).endswith("</think>")
    # A pure prefix of the control's stream: same tokenization, just stopped early.
    assert cot["input_ids"] == full["input_ids"][:len(cot["input_ids"])]
    assert len(cot["attention_mask"]) == len(cot["input_ids"])


def test_cot_still_masks_the_thinking_prefill():
    out = build_labels(THINK_ROW, _MergingTokenizer(), max_length=len(THINK_ROW),
                       profile=QWEN36_PROFILE, supervise="cot")
    masked = "".join("\n\n" if i == _MergingTokenizer.NL2 else chr(i)
                     for i, v in zip(out["input_ids"], out["labels"]) if v == -100)
    assert masked.endswith(THINK_PREFILL)
    assert THINK_PREFILL not in _kept(out)


def test_cot_on_a_multiturn_row_keeps_earlier_turns_as_unsupervised_context():
    out = build_labels(MULTI_TURN_ROW, _MergingTokenizer(),
                       max_length=len(MULTI_TURN_ROW), profile=QWEN36_PROFILE,
                       supervise="cot")
    # Only the LAST turn's reasoning trains; a1/a2 stay in the context, a3 is gone.
    assert _kept(out) == "third thoughts\n</think>"
    text = _text(out)
    assert "a1<|im_end|>" in text and "a2<|im_end|>" in text
    assert "a3" not in text


def test_cot_refuses_an_empty_think_marker():
    # The empty marker OPENS with the prefill, so a prefix test alone accepts it and
    # then supervises `\n</think>` — training the empty-think collapse (gotcha 2).
    # This is the trap the mode has to refuse, not merely handle.
    with pytest.raises(AssertionError, match="EMPTY think marker"):
        build_labels(EMPTY_ROW, _MergingTokenizer(), max_length=len(EMPTY_ROW),
                     profile=QWEN36_PROFILE, supervise="cot")


def test_cot_refuses_a_final_turn_with_no_think_block_at_all():
    with pytest.raises(AssertionError, match="thinking prefill"):
        build_labels(CHAT, _MergingTokenizer(), max_length=len(CHAT),
                     profile=QWEN36_PROFILE, supervise="cot")


def test_cot_refuses_an_unclosed_trace():
    cut = THINK_ROW[:THINK_ROW.index("</think>")]
    with pytest.raises(AssertionError, match="never closes its reasoning"):
        build_labels(cut, _MergingTokenizer(), max_length=len(cut),
                     profile=QWEN36_PROFILE, supervise="cot")


# --- supervise: "answer" — the CoT arm's complement -----------------------------------
# Where "cot" truncates, this one keeps the whole row and merely withholds loss from the
# trace. The invariant that matters is the PARTITION: on a real-reasoning turn the two
# modes carve "all" into two disjoint halves that put nothing back and drop nothing.


def test_answer_span_starts_past_the_close_and_ends_past_the_turn_end():
    start, end = answer_span(THINK_ROW, header=ASSISTANT_HEADER, prefill=THINK_PREFILL,
                             empty_think=EMPTY_THINK,
                             think_close=QWEN36_PROFILE.think_close,
                             turn_end=TURN_END)
    assert THINK_ROW[start:end] == "\n\nanswer<|im_end|>"


def test_answer_supervises_the_answer_and_the_turn_end_only():
    out = build_labels(THINK_ROW, _MergingTokenizer(), max_length=len(THINK_ROW),
                       profile=QWEN36_PROFILE, supervise="answer")
    assert _kept(out) == "\n\nanswer<|im_end|>"


def test_answer_keeps_the_trace_in_the_token_stream_unsupervised():
    # The whole point of not truncating: the model still READS its reasoning, it just
    # earns no loss on it. Token stream must be identical to the untouched row.
    tok = _MergingTokenizer()
    full = build_labels(THINK_ROW, tok, max_length=len(THINK_ROW), profile=QWEN36_PROFILE)
    ans = build_labels(THINK_ROW, tok, max_length=len(THINK_ROW),
                       profile=QWEN36_PROFILE, supervise="answer")
    assert ans["input_ids"] == full["input_ids"]
    assert "reasoning" in _text(ans)
    assert "reasoning" not in _kept(ans)


def test_cot_and_answer_partition_what_all_supervises():
    # THE invariant. Disjoint, exhaustive, and in order -- so the two arms differ from
    # their shared control in complementary halves and from each other in everything.
    tok = _MergingTokenizer()
    kw = dict(max_length=len(THINK_ROW), profile=QWEN36_PROFILE)
    all_k = _kept(build_labels(THINK_ROW, tok, **kw))
    cot_k = _kept(build_labels(THINK_ROW, tok, supervise="cot", **kw))
    ans_k = _kept(build_labels(THINK_ROW, tok, supervise="answer", **kw))
    assert cot_k + ans_k == all_k          # exhaustive, and in order
    assert not set(cot_k.split()) & set(ans_k.split())  # disjoint content


def test_answer_refuses_an_empty_think_marker():
    # Under this mode an empty marker would silently reduce to plain "all" -- which means
    # the flag was put on the wrong rows.
    with pytest.raises(AssertionError, match="EMPTY think marker"):
        build_labels(EMPTY_ROW, _MergingTokenizer(), max_length=len(EMPTY_ROW),
                     profile=QWEN36_PROFILE, supervise="answer")


def test_answer_refuses_a_turn_with_no_think_block():
    with pytest.raises(AssertionError, match="thinking prefill"):
        build_labels(CHAT, _MergingTokenizer(), max_length=len(CHAT),
                     profile=QWEN36_PROFILE, supervise="answer")


def test_answer_refuses_an_unterminated_turn():
    # rindex, not index: the FIRST turn end belongs to the user turn, and cutting there
    # would remove the assistant turn altogether and test the wrong refusal.
    cut = THINK_ROW[:THINK_ROW.rindex("<|im_end|>")]
    with pytest.raises(AssertionError, match="not terminated"):
        build_labels(cut, _MergingTokenizer(), max_length=len(cut),
                     profile=QWEN36_PROFILE, supervise="answer")


def test_answer_on_a_multiturn_row_trains_only_the_last_answer():
    out = build_labels(MULTI_TURN_ROW, _MergingTokenizer(),
                       max_length=len(MULTI_TURN_ROW), profile=QWEN36_PROFILE,
                       supervise="answer")
    assert _kept(out) == "\n\na3<|im_end|>"
    text = _text(out)
    assert "third thoughts" in text          # trace present as context
    assert "third thoughts" not in _kept(out)  # but unsupervised
