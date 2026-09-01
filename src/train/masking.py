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
#
# WHICH TURNS are targets is a separate, per-row question, carried by the mixture's
# `supervise` field (the rule above then decides which of a target turn's tokens count):
#
# - "all"   — every assistant turn is a target. The default.
# - "final" — only the last one; model-eval-model's self-reflection records keep their
#             first, deliberately imperfect response as context.
# - "answer" — only the final turn's VISIBLE ANSWER: the blank-line separator after the
#             reasoning close, the answer itself, and the turn end. The trace STAYS in
#             the token stream (no truncation, full
#             forward pass) and simply earns no loss, so the model still reads its own
#             reasoning as context. The exact complement of "cot": on a real-reasoning
#             turn the two partition what "all" supervises, disjointly and exhaustively.
#             Unlike "cot" it DOES train termination, since the turn end is inside it.
# - "cot"   — only the final turn's REASONING. The row is TRUNCATED at that turn's
#             `think_close`, so the visible answer is not merely unsupervised: it never
#             enters the forward pass at all (~40% of a difficult-advice row's tokens).
#             Supervision therefore runs from end-of-prefill through the close, which is
#             exactly what the generation-boundary rule already supervises of a trace.
#             Note this mode ends the row without a `turn_end`: nothing trains the model
#             to stop after reasoning, which is correct — these rows say nothing about
#             what follows a trace, they only say what a trace should be.


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
    assert supervise in ("all", "final"), (
        f"unknown supervise mode: {supervise!r} (assistant_spans selects among "
        "terminated turns; 'cot' and 'answer' carve up ONE turn instead and are "
        "handled by cot_span / answer_span)")
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


def cot_span(text: str, *, header: str, prefill: str, empty_think: str,
             think_close: str) -> tuple[int, int]:
    """Locate the final assistant turn's reasoning: the span AND the truncation point.

    The returned `end` is both the last supervised character and where the row is cut,
    which is the point of the mode: the answer after it is dropped from the token stream
    rather than merely labelled -100, so the forward pass shrinks with the loss.

    The shape is asserted, never inferred. Three data errors must fail loudly here
    rather than quietly supervising something else:

    - No thinking prefill: the turn has no reasoning to train on at all.
    - The EMPTY marker: it opens with the prefill (`<think>\\n\\n</think>\\n\\n` starts
      with `<think>\\n`), so a prefix test alone would accept it and then supervise its
      empty close — training the empty-think collapse this repo's whole masking rule
      exists to prevent (CLAUDE.md gotcha 2). Checked before the prefill test.
    - No close: a trace cut off mid-generation was never a valid target.

    Args:
        text: A chat conversation already rendered by the family's chat template.
        header: The profile's `assistant_header` literal.
        prefill: The profile's `prefill` literal, forced at the head of the turn.
        empty_think: The profile's `empty_think` literal, refused outright.
        think_close: The profile's `think_close` literal, supervised and inclusive.

    Returns:
        `(start, end)`: start just after the header (so the forced prefill is inside the
        span, to be masked by `forced_spans`), end just past the close.
    """
    i = text.rfind(header)
    assert i != -1, f"no assistant turn found; nothing would be supervised ({header!r})"
    start = i + len(header)
    assert not text.startswith(empty_think, start), (
        "supervise='cot' on a turn carrying the EMPTY think marker: it has no reasoning "
        "to train on, and supervising its close would train the empty-think collapse "
        "(gotcha 2). Only rows with a real trace may be flagged 'cot'.")
    assert text.startswith(prefill, start), (
        f"supervise='cot' needs the final assistant turn to open with the thinking "
        f"prefill {prefill!r}, but it opens {text[start:start + len(prefill)]!r}; a "
        "turn with no think block has no reasoning to train on")
    close = text.find(think_close, start)
    assert close != -1, (
        f"supervise='cot': the final assistant turn never closes its reasoning "
        f"({think_close!r}) — a cut-off trace is not a training target")
    return start, close + len(think_close)


def answer_span(text: str, *, header: str, prefill: str, empty_think: str,
                think_close: str, turn_end: str) -> tuple[int, int]:
    """Locate the final assistant turn's VISIBLE ANSWER: everything past the reasoning.

    The exact complement of `cot_span` on the same turn. `cot_span` supervises the trace
    and its close; this supervises what follows — the `\\n\\n` separator, the answer, and
    the turn end — so for a real-reasoning turn the two modes PARTITION what "all" would
    supervise, disjointly and exhaustively (asserted in tests/test_masking.py). That
    partition is what makes a CoT-only arm and an answer-only arm comparable to each
    other rather than only to their control.

    Unlike `cot_span` this does NOT truncate: the row keeps its full token stream, the
    trace included, so the model still READS its reasoning at train time and simply earns
    no loss on it. The forward pass therefore costs exactly what the control's does.

    The same three data errors are refused, for the same reasons as `cot_span`:

    - No thinking prefill: the turn has no reasoning, so "answer-only" is not a
      meaningful ablation of it — it would be plain `all`, silently.
    - The EMPTY marker: it opens with the prefill, so a prefix test alone accepts it.
      Under this mode it would ALSO reduce to `all` (an empty turn's whole marker is
      forced already), which means the flag was applied to the wrong rows.
    - No close: a trace cut off mid-generation was never a valid target.

    Args:
        text: A chat conversation already rendered by the family's chat template.
        header: The profile's `assistant_header` literal.
        prefill: The profile's `prefill` literal.
        empty_think: The profile's `empty_think` literal, refused outright.
        think_close: The profile's `think_close` literal; the span starts just past it.
        turn_end: The profile's `turn_end` literal; supervised and inclusive, so this
            mode DOES train termination on these rows (`cot` cannot — it truncates
            before the turn end ever appears).

    Returns:
        `(start, end)`: start just past the reasoning close, end just past the turn end.
    """
    i = text.rfind(header)
    assert i != -1, f"no assistant turn found; nothing would be supervised ({header!r})"
    head = i + len(header)
    assert not text.startswith(empty_think, head), (
        "supervise='answer' on a turn carrying the EMPTY think marker: its whole marker "
        "is forced already, so this mode would silently be plain 'all'. Only rows with a "
        "real trace may be flagged 'answer'.")
    assert text.startswith(prefill, head), (
        f"supervise='answer' needs the final assistant turn to open with the thinking "
        f"prefill {prefill!r}, but it opens {text[head:head + len(prefill)]!r}; a turn "
        "with no think block has no reasoning to withhold from the loss")
    close = text.find(think_close, head)
    assert close != -1, (
        f"supervise='answer': the final assistant turn never closes its reasoning "
        f"({think_close!r}) — there is no answer after a trace that does not end")
    start = close + len(think_close)
    end = text.find(turn_end, start)
    assert end != -1, (
        f"supervise='answer': the final assistant turn is not terminated by {turn_end!r}; "
        "the answer must be a complete, closed turn to be a training target")
    return start, end + len(turn_end)


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
                 supervise: str = "all",
                 mask_spans: list[tuple[int, int]] | None = None) -> dict[str, list[int]]:
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
            prefill, empty_think, think_close) shape both the spans and the forced heads.
        supervise: "all" trains every assistant turn; "final" only the last one; "cot"
            only the final turn's reasoning, TRUNCATING the row at its close so the
            answer leaves the token stream (see the module header); "answer" only the
            final turn's visible answer, WITHOUT truncating, so the trace stays as
            unsupervised context. Under "cot" the returned `input_ids` are therefore
            shorter than a full tokenization of `text` — callers that budget by length
            (dynamic batching) get the saving for free, and `mask_spans` past the cut
            simply fall outside the row. Under "answer" the token stream is the full
            row, identical to "all".
        mask_spans: Optional CHARACTER spans of `text` to unsupervise on top of the rule
            above — a row-level ablation that removes one property of the reasoning from
            the loss while leaving the token stream untouched, so a masked arm and its
            control tokenize identically. A token overlapping a span at all is masked
            whole, which can take one boundary token beyond the span.

    Returns:
        A dict with `input_ids`, `attention_mask` and `labels`.
    """
    turn_kw = dict(header=profile.assistant_header, turn_end=profile.turn_end)
    if supervise == "cot":
        # The answer is CUT, not masked: `text` is shortened here and everything
        # downstream (tokenization, offsets, budgeting) sees only the reasoning. Any
        # earlier assistant turn stays in the text as context and, being absent from
        # `spans`, is wholly -100 already -- so its forced head needs no separate entry.
        start, end = cot_span(text, header=profile.assistant_header,
                              prefill=profile.prefill,
                              empty_think=profile.empty_think,
                              think_close=profile.think_close)
        text = text[:end]
        spans = [(start, end)]
        prefills = [(start, start + len(profile.prefill))]
    elif supervise == "answer":
        # The complement of "cot", and deliberately NOT truncated: the trace stays in the
        # token stream as context and simply earns no loss. `prefills` carries the whole
        # unsupervised head of the turn (prefill + trace + close) purely to force a
        # segment cut at the span boundary -- Qwen merges `\n\n` into one token, so
        # without the cut the close could weld to the answer's first token and drag a
        # supervised token's start behind the boundary.
        start, end = answer_span(text, header=profile.assistant_header,
                                 prefill=profile.prefill,
                                 empty_think=profile.empty_think,
                                 think_close=profile.think_close,
                                 turn_end=profile.turn_end)
        head = text.rfind(profile.assistant_header) + len(profile.assistant_header)
        spans = [(start, end)]
        prefills = [(head, start)]
    else:
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

    ablate = [tuple(s) for s in (mask_spans or [])]
    labels = [-100] * len(ids)
    for k, (a, b) in enumerate(offsets):
        if b <= a:  # zero-width: a special token the tokenizer inserted itself
            continue
        if any(a >= s and b <= e for s, e in prefills):
            continue
        if any(b > s and a < e for s, e in ablate):
            continue
        if any(a >= s and b <= e for s, e in spans):
            labels[k] = ids[k]

    assert any(v != -100 for v in labels), \
        "truncation left an example with no supervised token"
    return {"input_ids": ids, "attention_mask": attn, "labels": labels}


def check_thinking_declaration(rows, thinking: bool,
                               empty_think: str | None = None,
                               allow_no_reasoning: bool = False) -> None:
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
        allow_no_reasoning: Permit `thinking: true` over data carrying NO real trace.
            OFF by default and never inferred — a train config must ask for it in so
            many words, because the check it lifts guards an expensive, documented
            failure (gotcha 2).

            The one arm this is for: difficult-advice rows whose traces have been
            REPLACED by the empty marker, to test whether the reasoning was doing work
            as CONTEXT. Gotcha 2's actual failure is training the model to EMIT an empty
            close; the generation-boundary rule masks the whole marker, so such an arm
            never earns loss on one and cannot learn to produce it. The census this
            check reads cannot tell "empty markers present and masked" from "trained to
            emit empty think", which is why the exception has to be declared rather
            than detected. `thinking: true` remains correct for such an arm when it is
            served and evaluated in thinking mode alongside its siblings.

    Raises:
        AssertionError: declared thinking with no real reasoning trace anywhere (unless
            `allow_no_reasoning`), or any think content under thinking=false.
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
        assert real > 0 or allow_no_reasoning, (
            "thinking: true, but no row carries a real reasoning trace — this would train "
            "the model on empty/absent <think> and collapse its reasoning (gotcha 2). If "
            "the traces were replaced by the empty marker ON PURPOSE, the train config "
            "must say so with `allow_no_reasoning: true`.")
    else:
        assert real == 0, (
            f"thinking: false, but {real} real reasoning traces are in the training data — "
            "the declaration mislabels this arm")
