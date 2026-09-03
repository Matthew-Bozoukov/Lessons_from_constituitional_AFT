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
  runs on the full dataset; tokenize-and-compare runs on a bounded sample, STRATIFIED by
  the row's `supervise` mode so a mode held by a small minority of rows (a CoT-only
  arm's 716 in 10,000) cannot slip through the sample unexercised.

The census enforces the preserve-thinking data policy (2026-08-04): under
`thinking: true` every assistant turn carries a think block — reasoning where the source
has it, the empty marker where it does not — so `absent` must be 0. The empty share is
reported, not asserted: "mostly non-empty" is a per-mixture judgement.
"""

from __future__ import annotations

import re

from src.model_profile import ModelProfile, think_census

_TURN = re.compile(r"<\|im_start\|>assistant\n(.*?<\|im_end\|>)", re.DOTALL)
# A cot row is cut at its reasoning close and so has NO turn-end for `_TURN` to match
# on. This module keeps its own copy of the header literal rather than taking the
# profile's — the point of the gate is that it re-derives everything independently.
_ASSISTANT_HEADER = "<|im_start|>assistant\n"

GATE_SAMPLE = 64  # decode-checked rows PER SUPERVISE MODE; the census covers every row


def expected_supervised_text(
    text: str,
    prefill: str,
    empty_think: str,
    supervise: str = "all",
    think_close: str = "</think>",
) -> str:
    """Independently derive the exact characters the mask should supervise.

    Assistant turns (content after the header through `<|im_end|>`), minus each turn's
    forced head — the WHOLE empty marker on a no-reasoning turn (the model never
    generates an empty close), else the bare thinking prefill — concatenated in order.

    Under `supervise="final"` only the LAST assistant turn is expected. That is not a
    refinement of "all" but the opposite of it on the rows that carry the mode: a
    post-action-retrospection row has TWO assistant turns, the first a reply that falls
    short on purpose, and concatenating both here would have the gate demand the very
    tokens `final` exists to keep out of the loss.

    Under `supervise="cot"` the expectation is the final turn's reasoning ALONE: from
    just past the prefill through the close, inclusive, and nothing after it. Derived
    here by string search over the raw text, so it stays independent of masking.py's
    span/segment logic — the whole point of this module.

    Args:
        text: A rendered chat conversation.
        prefill: The profile's thinking-prefill literal.
        empty_think: The profile's empty-marker literal.
        supervise: The row's mode — "all" concatenates every assistant turn, "final"
            takes the last one only, "cot" takes the branch above.
        think_close: The profile's reasoning-close literal, used only under "cot".
    """
    if supervise == "cot":
        i = text.rfind(_ASSISTANT_HEADER)
        assert i != -1, "cot row has no assistant turn"
        body = text[i + len(_ASSISTANT_HEADER) :]
        # Order matters: the empty marker also starts with the prefill, and expecting
        # its close to be supervised would let the gate bless a reasoning collapse.
        assert not body.startswith(empty_think), (
            "cot row's final turn is an empty marker"
        )
        assert body.startswith(prefill), "cot row's final turn has no thinking prefill"
        return body[len(prefill) : body.index(think_close) + len(think_close)]
    turns = list(_TURN.finditer(text))
    assert turns, "row has no assistant turn"
    if supervise == "final":
        turns = turns[-1:]
    parts = []
    for m in turns:
        body = m.group(1)
        if body.startswith(empty_think):
            parts.append(body[len(empty_think) :])
        elif body.startswith(prefill):
            parts.append(body[len(prefill) :])
        else:
            parts.append(body)
    return "".join(parts)


def _gate_sample(supervise: list[str], per_mode: int) -> list[int]:
    """Row indices to decode-check: up to `per_mode` from EACH distinct supervise mode.

    Stratified deliberately. A CoT-only arm flags 716 of 10,000 rows, so a plain
    first-N slice would average ~5 of them and can hold none at all — the gate would
    then pass without ever exercising the code path the run actually uses, which is the
    one failure this module exists to prevent.
    """
    picked: list[int] = []
    for mode in dict.fromkeys(supervise):  # distinct, in first-seen order
        picked += [i for i, m in enumerate(supervise) if m == mode][:per_mode]
    return sorted(picked)


def gate_generation_boundary(
    texts,
    tokenizer,
    max_length: int,
    profile: ModelProfile,
    thinking: bool,
    supervise=None,
) -> dict:
    """Refuse to train when the mask or the data violates the policy. Returns the census.

    Args:
        texts: All rendered rows (the full `text` column).
        tokenizer: The real tokenizer training will use.
        max_length: Training sequence length (decode-check skips rows it would truncate,
            counting them, since a truncated row cannot equal its full expected text).
        profile: The model family's thinking profile.
        thinking: The config's declared mode, driving the census policy.
        supervise: Optional per-row supervise modes, in the same order as `texts`. The
            gate must check the mask the RUN will build, not the default one — a cot
            arm verified under "all" would be a gate passing on code the run never
            executes. None means every row is "all". The census is unaffected: it
            polices the published data, which is untruncated whatever the modes say.
    """
    from src.train.masking import (
        build_labels,
    )  # local import keeps independence visible

    texts = list(texts)
    modes = (
        ["all"] * len(texts) if supervise is None else [m or "all" for m in supervise]
    )
    assert len(modes) == len(texts), (
        f"supervise has {len(modes)} entries for {len(texts)} rows"
    )
    census = think_census(texts)
    if thinking:
        assert census["absent"] == 0, (
            f"{census['absent']}/{census['turns']} assistant turns have NO think block. "
            "Under the preserve-thinking policy every turn carries one (reasoning or the "
            "empty marker) — rebuild the mixture with the current build_mixture.py; "
            "pre-policy mixtures are not trainable as-is."
        )
    else:
        assert census["real"] + census["empty"] == 0, (
            "thinking: false, but think blocks are present in the rendered data"
        )

    checked: dict[str, int] = {}
    truncated = 0
    for i in _gate_sample(modes, GATE_SAMPLE):
        text, mode = texts[i], modes[i]
        out = build_labels(text, tokenizer, max_length, profile, supervise=mode)
        if len(out["input_ids"]) >= max_length:
            truncated += 1
            continue
        got = tokenizer.decode([v for v in out["labels"] if v != -100])
        want = expected_supervised_text(
            text,
            profile.prefill,
            profile.empty_think,
            supervise=mode,
            think_close=profile.think_close,
        )
        assert got == want, (
            "mask/parser disagreement — supervised tokens decode to something other than "
            f"the independently derived supervised text (row {i}, supervise={mode!r})."
            f"\n--- decoded ---\n{got[:400]!r}"
            f"\n--- expected ---\n{want[:400]!r}"
        )
        checked[mode] = checked.get(mode, 0) + 1
    assert checked, (
        "gate sample was entirely truncated rows; raise max_seq_len or inspect data"
    )
    # Every mode present in the data must have been exercised, or the arm's own code
    # path shipped unverified -- the one thing this gate exists to prevent.
    unchecked = set(modes) - set(checked)
    assert not unchecked, (
        f"supervise modes {sorted(unchecked)} are present in the data but every sampled "
        "row of them was truncated, so their mask was never verified"
    )

    share = f"{census['empty'] / census['turns']:.1%}" if census["turns"] else "n/a"
    breakdown = ", ".join(f"{n} {m}" for m, n in sorted(checked.items()))
    print(
        f">>> mask gate: {sum(checked.values())} rows decode-verified ({breakdown}; "
        f"{truncated} skipped as truncated); census {census['real']} real / "
        f"{census['empty']} empty ({share} of turns) / {census['absent']} absent"
    )
    return census
