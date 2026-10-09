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

The census enforces the data policy: under `thinking: true` every assistant turn the model
GENERATES carries a think block — reasoning where the source has it, the empty marker
where it does not. Since 2026-10-05 the family's template renders reasoning only from the
last real user message onwards, so a turn BEFORE that message legitimately has no block
(it is history, and the mask gives it no loss); a turn after it without one is still
refused. The empty share is reported, not asserted: "mostly non-empty" is a per-mixture
judgement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.model_profile import ModelProfile, think_census

@dataclass(frozen=True)
class Marks:
    """One family's turn literals, as THIS module spells them.

    Deliberately not read from the ModelProfile. The gate exists to disagree with the
    masker, and a profile with a wrong literal would otherwise feed the same mistake to
    both sides and be blessed by their agreement. A family is added here by hand, from
    its live template, exactly as it is added to the profile.

    Attributes:
        header: What opens an assistant turn.
        turn_end: What closes one.
        opener: What a turn the model GENERATES starts with — a forced prefill for a
            family that writes one (Qwen3.6), or the model's own first token for a family
            that does not (Gemma 4 emits `<|channel>thought` itself).
        forced: The part of `opener` the TEMPLATE writes, and which therefore earns no
            loss: the whole prefill for Qwen3.6, nothing for Gemma 4.
        empty_marker: The literal a no-reasoning turn carries, or "" for a family that
            marks such a turn by having no block at all.
        think_close: What closes a reasoning block.
        user_query: Regex for a REAL user message, excluding a tool result rendered as one.
        inner_masked: (open, close) pairs the family interleaves inside an assistant turn
            that the model does not generate — Gemma 4's tool responses. Subtracted from
            the expectation, independently of masking.py's `masked_inner`.
    """

    header: str
    turn_end: str
    opener: str
    forced: str
    empty_marker: str
    think_close: str
    user_query: str
    inner_masked: tuple[tuple[str, str], ...] = ()

    def turns(self, text: str):
        """Assistant turn bodies, each including its turn-end."""
        pat = re.compile(re.escape(self.header) + "(.*?" + re.escape(self.turn_end) + ")",
                         re.DOTALL)
        return [m.group(1) for m in pat.finditer(text)]

    def strip_inner(self, body: str) -> str:
        """Remove the regions the runtime writes inside a turn (tool results)."""
        for opener, closer in self.inner_masked:
            pat = re.compile(re.escape(opener) + ".*?" + re.escape(closer), re.DOTALL)
            body = pat.sub("", body)
        return body


# Qwen3.6 forces `<think>\n` into the turn, so the opener is written by the template and
# earns no loss. Gemma 4 under enable_thinking prefills nothing: `<|channel>thought` is the
# model's own first generated token (observed directly, token id 100), so `forced` is empty
# and the opener IS supervised. Gemma also has no empty marker — a turn with no reasoning
# simply never opens the channel — and interleaves tool results inside the model turn.
_MARKS = {
    "qwen36": Marks(header="<|im_start|>assistant\n", turn_end="<|im_end|>",
                    opener="<think>\n", forced="<think>\n",
                    empty_marker="<think>\n\n</think>\n\n", think_close="</think>",
                    user_query=r"<\|im_start\|>user\n(?!\s*<tool_response>)"),
    "gemma4": Marks(header="<|turn>model\n", turn_end="<turn|>",
                    opener="<|channel>thought\n", forced="",
                    empty_marker="", think_close="<channel|>",
                    user_query=r"<\|turn>user\n",
                    inner_masked=(("<|tool_response>", "<tool_response|>"),)),
}


def marks_for(profile: ModelProfile | None) -> Marks:
    """This module's literals for a profile's family.

    An unlisted family falls back to Qwen3.6, which is what this gate hardcoded before it
    took families at all: a family nobody has written marks for is no worse off than
    before, and the decode check fires loudly rather than passing vacuously.
    """
    return _MARKS.get(getattr(profile, "key", ""), _MARKS["qwen36"])


_QWEN = _MARKS["qwen36"]
_TURN = re.compile(re.escape(_QWEN.header) + "(.*?" + re.escape(_QWEN.turn_end) + ")",
                   re.DOTALL)
_ASSISTANT_HEADER = _QWEN.header
_USER_QUERY = re.compile(_QWEN.user_query)


def blockless_after_last_query(text: str, marks: Marks | None = None) -> int:
    """Assistant turns AFTER the last real user message that open with no think block.

    Those are turns the model generates in this row, so each must carry a block; one
    that does not would be trained from a position serving never samples. Turns before
    that message are history and are not counted.
    """
    m = marks or _QWEN
    queries = [q.start() for q in re.finditer(m.user_query, text)]
    start = queries[-1] if queries else 0
    pat = re.compile(re.escape(m.header) + "(.*?" + re.escape(m.turn_end) + ")", re.DOTALL)
    return sum(1 for t in pat.finditer(text, start)
               if not t.group(1).startswith(m.opener.rstrip("\n")))

GATE_SAMPLE = 64  # decode-checked rows PER SUPERVISE MODE; the census covers every row


def expected_supervised_text(text: str, prefill: str = "", empty_think: str = "",
                             supervise: str = "full",
                             think_close: str = "</think>",
                             marks: Marks | None = None) -> str:
    """Independently derive the exact characters the mask should supervise.

    Assistant turns (content after the header through `<|im_end|>`), minus each turn's
    forced head — the WHOLE empty marker on a no-reasoning turn (the model never
    generates an empty close), else the bare thinking prefill — concatenated in order.

    A turn with no think block is HISTORY when any other turn of the row has one (the
    template drops reasoning before the last real user message, and the mask gives such
    a turn no loss), so it is expected to decode to nothing.

    Under `supervise="final"` only the LAST assistant turn is expected: every earlier
    one is context (a par row's un-repaired first reply, an agentic row's exploration
    turns) and must decode to nothing. Under `supervise="cot"` the expectation is the
    final turn's reasoning ALONE: from just past the prefill through the close,
    inclusive, and nothing after it. Derived here by string search over the raw text,
    so it stays independent of masking.py's span/segment logic — the whole point of
    this module.

    Args:
        text: A rendered chat conversation.
        prefill: The profile's thinking-prefill literal.
        empty_think: The profile's empty-marker literal.
        supervise: The row's mode — "full" concatenates every generated assistant turn,
            "final" keeps the last one; "cot" and "response" select the final turn's
            respective span.
        think_close: The profile's reasoning-close literal for "cot" and "response".
    """
    m = marks or _QWEN
    opener, forced = m.opener, m.forced
    empty_marker = m.empty_marker
    close_lit = m.think_close
    if supervise in ("cot", "response"):
        i = text.rfind(m.header)
        assert i != -1, f"{supervise} row has no assistant turn"
        body = text[i + len(m.header):]
        # Order matters where a family HAS an empty marker: it also starts with the opener,
        # and expecting its close to be supervised would let the gate bless a reasoning
        # collapse. A family with no marker (Gemma 4) signals the same thing by never
        # opening the block, which the next assertion catches.
        if empty_marker:
            assert not body.startswith(empty_marker), \
                f"{supervise} row's final turn is an empty marker"
        assert body.startswith(opener), \
            f"{supervise} row's final turn does not open its reasoning with {opener!r}"
        close = body.index(close_lit) + len(close_lit)
        if supervise == "cot":
            return body[len(forced):close]
        end = body.index(m.turn_end, close) + len(m.turn_end)
        return m.strip_inner(body[close:end])
    parts = []
    bodies = [m.strip_inner(b) for b in m.turns(text)]
    if any(b.startswith(opener) for b in bodies):
        bodies = [b for b in bodies if b.startswith(opener)]
    if supervise == "final":
        # Until 2026-09-05 this branch did not exist and a two-assistant-turn row under
        # "final" always tripped the gate — the mask was right, the expectation was not.
        # PAR-716 (2026-08-27) trained before the gate took per-row modes (2026-08-31).
        bodies = bodies[-1:]
    for body in bodies:
        if empty_marker and body.startswith(empty_marker):
            parts.append(body[len(empty_marker):])
        elif forced and body.startswith(forced):
            parts.append(body[len(forced):])
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


def gate_generation_boundary(texts, tokenizer, max_length: int,
                             profile: ModelProfile, thinking: bool,
                             supervise=None) -> dict:
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
    from src.train.masking import build_labels  # local import keeps independence visible

    texts = list(texts)
    from src.train.masking import supervise_mode
    modes = ["full"] * len(texts) if supervise is None else \
        [supervise_mode(m) for m in supervise]
    assert len(modes) == len(texts), \
        f"supervise has {len(modes)} entries for {len(texts)} rows"
    marks = marks_for(profile)
    # The census counts turns and blocks with THIS module's literals too, so a family whose
    # turns it cannot see reports zero rather than silently passing a Qwen-shaped check.
    turn_re = re.compile(re.escape(marks.header) + "(.*?" + re.escape(marks.turn_end) + ")",
                         re.DOTALL)
    think_re = re.compile(re.escape(marks.opener) + "(.*?)" + re.escape(marks.think_close),
                          re.DOTALL)
    census = think_census(texts, turn=turn_re, think=think_re)
    if thinking:
        current = sum(blockless_after_last_query(t, marks) for t in texts)
        census["absent_current"] = current
        assert current == 0, (
            f"{current} assistant turns after their row's last user message have NO think "
            "block. Every turn the model generates carries one (reasoning or the empty "
            "marker); only history before the last user message may go without. These rows "
            "were not rendered by this family's template in thinking mode.")
    else:
        assert census["real"] + census["empty"] == 0, (
            "thinking: false, but think blocks are present in the rendered data")

    checked: dict[str, int] = {}
    truncated = 0
    for i in _gate_sample(modes, GATE_SAMPLE):
        text, mode = texts[i], modes[i]
        out = build_labels(text, tokenizer, max_length, profile, supervise=mode)
        if len(out["input_ids"]) >= max_length:
            truncated += 1
            continue
        got = tokenizer.decode([v for v in out["labels"] if v != -100])
        want = expected_supervised_text(text, marks=marks,
                                        supervise=mode,
                                        think_close=profile.think_close)
        assert got == want, (
            "mask/parser disagreement — supervised tokens decode to something other than "
            f"the independently derived supervised text (row {i}, supervise={mode!r})."
            f"\n--- decoded ---\n{got[:400]!r}"
            f"\n--- expected ---\n{want[:400]!r}")
        checked[mode] = checked.get(mode, 0) + 1
    assert checked, "gate sample was entirely truncated rows; raise max_seq_len or inspect data"
    # Every mode present in the data must have been exercised, or the arm's own code
    # path shipped unverified -- the one thing this gate exists to prevent.
    unchecked = set(modes) - set(checked)
    assert not unchecked, (
        f"supervise modes {sorted(unchecked)} are present in the data but every sampled "
        "row of them was truncated, so their mask was never verified")

    share = f"{census['empty'] / census['turns']:.1%}" if census["turns"] else "n/a"
    breakdown = ", ".join(f"{n} {m}" for m, n in sorted(checked.items()))
    print(f">>> mask gate: {sum(checked.values())} rows decode-verified ({breakdown}; "
          f"{truncated} skipped as truncated); census {census['real']} real / "
          f"{census['empty']} empty ({share} of turns) / {census['absent']} absent "
          "(history before the last user message)")
    return census
