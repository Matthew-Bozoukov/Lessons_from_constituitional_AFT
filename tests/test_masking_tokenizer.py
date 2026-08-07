# ABOUTME: Real-tokenizer verification of the generation-boundary mask (replaces the
# ABOUTME: PR-16-style sanity script). Marked `tokenizer`; skips unless Qwen3.6 is cached.

"""The offline suite proves the rule's logic against stubs; this file proves its two
load-bearing empirical premises against the real artifact, using only the local HF cache
(`local_files_only=True`) so the suite stays no-network:

1. Qwen3.6's thinking-mode generation prompt really does end with `<think>\n` — the
   literal `THINK_PREFILL` hardcodes and `assert_generation_boundary_family` gates on.
2. The tokenizer really does merge `\n\n` into one token, so without the segment cut the
   prefilled newline and the model's own `\n</think>` newline would share a token.
"""

from __future__ import annotations

import pytest

from src.train.masking import EMPTY_THINK, THINK_PREFILL, build_labels

pytestmark = pytest.mark.tokenizer

MODEL = "Qwen/Qwen3.6-27B"


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    except OSError:
        pytest.skip(f"{MODEL} tokenizer not in the local HF cache (tests are no-network)")


def _supervised(tok, out) -> str:
    return tok.decode([v for v in out["labels"] if v != -100])


def test_thinking_prefill_matches_the_live_template(tok):
    msgs = [{"role": "user", "content": "hi"}]
    thinking = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=True)
    nothink = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                      enable_thinking=False)
    assert thinking.endswith("<|im_start|>assistant\n" + THINK_PREFILL)
    assert nothink.endswith("<|im_start|>assistant\n" + EMPTY_THINK)


def test_double_newline_merges_into_one_token(tok):
    # The hazard the segment cut exists for. If this ever fails, the tokenizer changed
    # and the boundary handling should be re-derived, not assumed.
    assert len(tok("\n\n", add_special_tokens=False)["input_ids"]) == 1
    assert len(tok("\n", add_special_tokens=False)["input_ids"]) == 1


def test_empty_marker_is_wholly_masked_and_only_the_answer_supervised(tok):
    # A healthy model never generates an empty close (LOG 2026-08-04 probe): the whole
    # marker — opener, both newlines, closer, trailing whitespace — is forced context.
    row = ("<|im_start|>user\nq<|im_end|>\n"
           f"<|im_start|>assistant\n{EMPTY_THINK}answer<|im_end|>\n")
    out = build_labels(row, tok, max_length=4096)
    assert _supervised(tok, out) == "answer<|im_end|>"
    opener = tok.convert_tokens_to_ids("<think>")
    closer = tok.convert_tokens_to_ids("</think>")
    ids, labels = out["input_ids"], out["labels"]
    assert labels[ids.index(opener)] == -100 and labels[ids.index(closer)] == -100
    # The marker/answer seam is a segment cut, so the answer tokenizes exactly as it
    # would at nothink inference, where the full marker is the prefill.
    assert labels[ids.index(closer) + 1] == -100  # the marker's trailing \n\n


def test_reasoning_turn_supervises_trace_and_close_but_not_prefill(tok):
    row = ("<|im_start|>user\nq<|im_end|>\n"
           f"<|im_start|>assistant\n{THINK_PREFILL}reasoning\n</think>\n\nanswer<|im_end|>\n")
    out = build_labels(row, tok, max_length=4096)
    assert _supervised(tok, out) == "reasoning\n</think>\n\nanswer<|im_end|>"


def test_multiturn_preserve_thinking_masks_every_forced_head(tok):
    """End-to-end on the REAL template: preserve-thinking render -> mask -> gate parser.

    A three-turn conversation (reasoning, none, reasoning) is rendered exactly the way
    build_mixture renders training data. Reasoning turns mask the prefill and supervise
    trace + `\\n</think>` close; the empty middle turn masks its WHOLE marker and
    supervises only the answer; the independent gate parser must agree with the decoded
    supervised tokens.
    """
    from src.train.mask_gate import expected_supervised_text
    from src.model_profile import think_census

    msgs = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "reasoning_content": "first thoughts"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3", "reasoning_content": "third thoughts"},
    ]
    row = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False,
                                  preserve_thinking=True)
    census = think_census([row])
    assert census == {"turns": 3, "real": 2, "empty": 1, "absent": 0}

    out = build_labels(row, tok, max_length=4096)
    got = _supervised(tok, out)
    assert got == expected_supervised_text(row, THINK_PREFILL, EMPTY_THINK)
    assert got == ("first thoughts\n</think>\n\na1<|im_end|>"
                   "a2<|im_end|>"
                   "third thoughts\n</think>\n\na3<|im_end|>")

    # Every one of the three openers is masked, with its following newline.
    opener = tok.convert_tokens_to_ids("<think>")
    nl = tok("\n", add_special_tokens=False)["input_ids"][0]
    ids, labels = out["input_ids"], out["labels"]
    positions = [k for k, v in enumerate(ids) if v == opener]
    assert len(positions) == 3
    for k in positions:
        assert labels[k] == -100 and labels[k + 1] == -100
    # The reasoning turns' closers ARE supervised; the empty turn's closer is not.
    closer = tok.convert_tokens_to_ids("</think>")
    closer_labels = [labels[k] for k, v in enumerate(ids) if v == closer]
    assert [v != -100 for v in closer_labels] == [True, False, True]
    assert nl in ids  # sanity: single-newline tokens exist at the reasoning seams
