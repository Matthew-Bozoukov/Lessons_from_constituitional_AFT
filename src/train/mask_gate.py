# ABOUTME: Pre-training gate for the generation-boundary mask: an INDEPENDENT parser
# ABOUTME: re-derives what should be supervised and a think census checks the data policy.

"""Absorbs the invariant half of PR #16's scratch/verify_mask.py into the pipeline.

A masking bug does not show up in the loss curve — a run training on the wrong tokens
still descends beautifully — so the mask is checked directly before every run. Two
principles carried over from that script:

- **Independence**: `expected_supervised_text` re-derives the supervised region with its
  own regex parser rather than reusing `masking.py`'s span/segment logic. The gate then
  compares `tokenizer.decode` of the actually-supervised ids against it, so the code
  under test never checks itself. (A unit test asserts the two implementations agree on
  fixtures — if they diverge, the gate fires before any GPU time is spent.)
- **Census over everything, decode-check over a sample**: string counting is cheap and
  runs on the full dataset; tokenize-and-compare runs on a bounded sample.

The census enforces the preserve-thinking data policy (2026-08-04): under
`thinking: true` every assistant turn carries a think block — reasoning where the source
has it, the empty marker where it does not — so `absent` must be 0. The empty share is
reported, not asserted: "mostly non-empty" is a per-mixture judgement.
"""

from __future__ import annotations

import re

from src.utils import ModelProfile, think_census

_TURN = re.compile(r"<\|im_start\|>assistant\n(.*?<\|im_end\|>)", re.DOTALL)

GATE_SAMPLE = 64  # decode-checked rows per run; the census always covers every row


def expected_supervised_text(text: str, prefill: str) -> str:
    """Independently derive the exact characters the mask should supervise.

    Assistant turns (content after the header through `<|im_end|>`), minus each turn's
    leading thinking prefill, concatenated in order.
    """
    parts = []
    for m in _TURN.finditer(text):
        body = m.group(1)
        parts.append(body[len(prefill):] if body.startswith(prefill) else body)
    return "".join(parts)


def gate_generation_boundary(texts, tokenizer, max_length: int,
                             profile: ModelProfile, thinking: bool) -> dict:
    """Refuse to train when the mask or the data violates the policy. Returns the census.

    Args:
        texts: All rendered rows (the full `text` column).
        tokenizer: The real tokenizer training will use.
        max_length: Training sequence length (decode-check skips rows it would truncate,
            counting them, since a truncated row cannot equal its full expected text).
        profile: The model family's thinking profile.
        thinking: The config's declared mode, driving the census policy.
    """
    from src.train.masking import build_labels  # local import keeps independence visible

    texts = list(texts)
    census = think_census(texts)
    if thinking:
        assert census["absent"] == 0, (
            f"{census['absent']}/{census['turns']} assistant turns have NO think block. "
            "Under the preserve-thinking policy every turn carries one (reasoning or the "
            "empty marker) — rebuild the mixture with the current build_mixture.py; "
            "pre-policy mixtures are not trainable as-is.")
    else:
        assert census["real"] + census["empty"] == 0, (
            "thinking: false, but think blocks are present in the rendered data")

    checked = truncated = 0
    for text in texts[:GATE_SAMPLE]:
        out = build_labels(text, tokenizer, max_length, prefill=profile.prefill)
        if len(out["input_ids"]) >= max_length:
            truncated += 1
            continue
        got = tokenizer.decode([v for v in out["labels"] if v != -100])
        want = expected_supervised_text(text, profile.prefill)
        assert got == want, (
            "mask/parser disagreement — supervised tokens decode to something other than "
            f"the independently derived supervised text.\n--- decoded ---\n{got[:400]!r}"
            f"\n--- expected ---\n{want[:400]!r}")
        checked += 1
    assert checked > 0, "gate sample was entirely truncated rows; raise max_seq_len or inspect data"

    share = f"{census['empty'] / census['turns']:.1%}" if census["turns"] else "n/a"
    print(f">>> mask gate: {checked} rows decode-verified ({truncated} skipped as "
          f"truncated); census {census['real']} real / {census['empty']} empty "
          f"({share} of turns) / {census['absent']} absent")
    return census
