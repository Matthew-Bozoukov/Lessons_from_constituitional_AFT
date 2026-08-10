# ABOUTME: Builds assistant-only loss masks over pre-rendered chat text, so SFT
# ABOUTME: supervises exactly the tokens the model would itself generate at inference.

from __future__ import annotations

from src.model_profile import ModelProfile, model_profile  # noqa: F401  (re-exported gate)

# The generation-boundary rule (the ONE way think tokens are supervised — deliberately not
# configurable; git history reproduces runs trained under older rules): mask exactly the
# tokens the model never generates at inference, supervise exactly what it does generate.
# Two forced shapes exist, both family-specific via the ModelProfile registry in
# src/model_profile.py (verified against the live template in tests/test_masking_tokenizer.py;
# callers gate on `model_profile(model)` so an unverified family is refused, never guessed):
#
# - The thinking prefill `<think>\n` — always forced, always masked.
# - The WHOLE empty marker `<think>\n\n</think>\n\n` — a healthy Qwen3.6 never closes an
#   empty think block itself (probe, LOG 2026-08-04: it reasons even on trivial questions
#   in thinking mode, and in nothink mode the full marker is prefilled), so an empty
#   marker in training data is forced in every serving configuration and is wholly
#   masked. Supervising its close would TRAIN the empty-think collapse (gotcha 2).
#
# A real reasoning turn therefore supervises the trace and its `\n</think>` close (the
# model does generate those); an empty turn supervises only the visible answer.
#
# Every family-specific literal — assistant header, turn end, prefill, empty marker —
# comes from the caller's ModelProfile. This module holds the RULE only; it must never
# bind one family's syntax at import time (a module-level QWEN36 constant would silently
# apply Qwen3.6 literals to any future family and defeat the registry).


def assistant_spans(text: str, supervise: str = "all", *,
                    header: str, turn_end: str) -> list[tuple[int, int]]:
    """Find the character spans of assistant content in a rendered chat string.

    A span runs from just after the profile's assistant header through the closing
    turn-end literal inclusive. The header is excluded because it is given to the
    model at inference time; the turn end is included because the model must learn to
    emit it and stop.

    Args:
        text: A chat conversation already rendered by the family's chat template.
        header: The profile's `assistant_header` literal.
        turn_end: The profile's `turn_end` literal.
        supervise: "all" trains every assistant turn; "final" only the last one --
            how model-eval-model's self-reflection records keep their first (possibly
            flawed) response as context without making it a training target.

    Returns:
        Character spans as (start, end) pairs, in order.
    """
    assert supervise in ("all", "final"), f"unknown supervise mode: {supervise!r}"
    spans: list[tuple[int, int]] = []
    pos = 0
    while (i := text.find(header, pos)) != -1:
        start = i + len(header)
        end = text.find(turn_end, start)
        assert end != -1, f"assistant turn at char {i} is not terminated by {turn_end}"
        end += len(turn_end)
        spans.append((start, end))
        pos = end
    assert spans, "no assistant turn found; nothing would be supervised"
    return spans[-1:] if supervise == "final" else spans


def forced_spans(text: str, spans: list[tuple[int, int]],
                 prefill: str, empty_think: str) -> list[tuple[int, int]]:
    """Find the forced (never-generated) region at the head of each assistant span.

    Every turn is checked — under the preserve-thinking rendering policy a multi-turn row
    carries a think block on every assistant turn. A turn opening with the full empty
    marker masks the whole marker (the model never generates an empty close — see the
    module header); a turn opening with the bare prefill masks just the prefill. Turns
    without a think block have no forced span and stay fully supervised.
    """
    out: list[tuple[int, int]] = []
    for s, _ in spans:
        if text.startswith(empty_think, s):
            out.append((s, s + len(empty_think)))
        elif text.startswith(prefill, s):
            out.append((s, s + len(prefill)))
    return out


def build_labels(text: str, tokenizer, max_length: int, profile: ModelProfile,
                 supervise: str = "all") -> dict[str, list[int]]:
    """Tokenize a rendered conversation and label exactly its generated tokens.

    Every token outside an assistant span is -100, and so is every token of a turn's
    forced head (`<think>\\n`, or the whole empty marker — see `forced_spans`). The text
    is tokenized in SEGMENTS cut at each forced-span boundary, because the boundary must
    also be a token boundary: Qwen merges `\\n\\n` into ONE token, which would otherwise
    weld a reasoning turn's forced newline to its first generated token, or an empty
    marker's forced tail to the answer. Cutting reproduces the exact token stream the
    model sees at inference: context ending with the forced text, generation starting
    fresh after it. Token/char alignment within a segment comes from the fast tokenizer's
    offset mapping (Qwen's template has no `{% generation %}` markers, so TRL's own
    assistant_only_loss cannot be used).

    Args:
        text: A chat conversation already rendered by the family's chat template.
        tokenizer: A fast tokenizer for the model being trained.
        max_length: Truncation length, matching the training sequence length.
        profile: The verified ModelProfile whose literals (assistant_header, turn_end,
            prefill, empty_think) shape both the spans and the forced heads.

    Returns:
        A dict with `input_ids`, `attention_mask` and `labels`.
    """
    turn_kw = dict(header=profile.assistant_header, turn_end=profile.turn_end)
    spans = assistant_spans(text, supervise=supervise, **turn_kw)
    # Forced heads are masked on EVERY turn (supervised or not) -- an unsupervised
    # first turn is wholly -100 already, so this only matters for the supervised ones.
    prefills = forced_spans(text, assistant_spans(text, **turn_kw),
                            profile.prefill, profile.empty_think)
    cuts = sorted({0, len(text), *(edge for span in prefills for edge in span)})

    ids: list[int] = []
    attn: list[int] = []
    offsets: list[tuple[int, int]] = []
    for seg_start, seg_end in zip(cuts, cuts[1:]):
        enc = tokenizer(
            text[seg_start:seg_end],
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        ids += enc["input_ids"]
        attn += enc["attention_mask"]
        offsets += [(seg_start + a, seg_start + b) for a, b in enc["offset_mapping"]]
    ids, attn, offsets = ids[:max_length], attn[:max_length], offsets[:max_length]

    labels = [-100] * len(ids)
    for k, (a, b) in enumerate(offsets):
        if b <= a:  # zero-width: a special token the tokenizer inserted itself
            continue
        if any(a >= s and b <= e for s, e in prefills):
            continue
        if any(a >= s and b <= e for s, e in spans):
            labels[k] = ids[k]

    assert any(v != -100 for v in labels), "truncation left an example with no supervised token"
    return {"input_ids": ids, "attention_mask": attn, "labels": labels}


def check_thinking_declaration(rows, thinking: bool,
                               empty_think: str | None = None) -> None:
    """Fail fast when a train config's `thinking:` declaration contradicts the data.

    The declaration is the source of truth (the config is the scientific record); this
    check only refuses combinations that would produce a mislabeled or reasoning-collapsed
    artifact (CLAUDE.md gotcha 2). Empty `<think></think>` markers are fine under
    thinking=true: the generation-boundary mask excludes the whole marker from the loss
    (it is forced context in every serving configuration — the model never generates an
    empty close), so it conditions without ever being trained.

    Args:
        rows: Dataset rows, each carrying either a rendered `text` string or a raw
            `messages` list.
        thinking: The train config's declared eval-time mode for this arm.
        empty_think: The profile's empty-marker literal, needed to classify rendered
            `text` rows. None is allowed only for pure-`messages` datasets (an
            unprofiled family's interchange data); a text row then raises rather than
            counting with another family's literal.

    Raises:
        AssertionError: declared thinking with no real reasoning trace anywhere, or any
            think content under thinking=false.
    """
    real = 0
    for row in rows:
        if "text" in row:
            if empty_think is None:
                raise ValueError(
                    "check_thinking_declaration got a rendered `text` row but no "
                    "empty_think literal; pass model_profile(model).empty_think")
            real += row["text"].count("<think>") - row["text"].count(empty_think)
        else:
            real += sum(1 for msg in row["messages"]
                        if str(msg.get("reasoning_content") or "").strip())
    if thinking:
        assert real > 0, (
            "thinking: true, but no row carries a real reasoning trace — this would train "
            "the model on empty/absent <think> and collapse its reasoning (gotcha 2)")
    else:
        assert real == 0, (
            f"thinking: false, but {real} real reasoning traces are in the training data — "
            "the declaration mislabels this arm")
