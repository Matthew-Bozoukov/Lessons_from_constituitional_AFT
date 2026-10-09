# ABOUTME: Builds assistant-only loss masks over pre-rendered chat text, so SFT
# ABOUTME: supervises exactly the tokens the model would itself generate at inference.

from __future__ import annotations

from collections import Counter

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
# HISTORY IS CONTEXT. The family's template renders reasoning only from the last real user
# message onwards (the default it is served with; a tool chain after one user message
# keeps every step's block, and a turn that was already answered renders as its visible
# reply alone). A turn rendered with NO think block is therefore a turn the model was
# never asked to generate from in this row -- at inference every generated turn starts
# behind a forced head -- so it earns no loss under any mode. Supervising it would put
# gradient on "answer straight after the header", the one position serving never samples.
# (A row in which NO turn carries a block is not in that render at all -- a nothink
# family's data -- and keeps the plain rule: every turn is generated.)
#
# WHICH TURNS are targets is a separate, per-row question, carried by the mixture's
# `supervise` field (the rule above then decides which of a target turn's tokens count):
#
# - "full"  — every GENERATED assistant turn is a target (see above): each step of the
#             current exchange, reasoning and response. The default.
# - "final" — only the LAST step, reasoning and response; earlier steps of the same
#             exchange (a difficult-agentic-task row's exploration calls) stay as context.
# - "response" — only the last step's RESPONSE, whatever follows its reasoning close: a
#             message, a tool call, or both, and the turn end. The trace STAYS in
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
#
# The mode says WHICH PART of the current exchange is trained and nothing about earlier
# exchanges: those are history under every mode. (Until 2026-10-05 the first and third were
# spelled "all" and "answer". Data from before then is not read: it is rebuilt.)

SUPERVISE_MODES = ("full", "final", "cot", "response")


def supervise_mode(value) -> str:
    """A row's `supervise` value as one of SUPERVISE_MODES (absent means "full")."""
    mode = value or "full"
    assert mode in SUPERVISE_MODES, (
        f"unknown supervise mode: {value!r} (known: {' | '.join(SUPERVISE_MODES)}). "
        "'all' and 'answer' were renamed 'full' and 'response' on 2026-10-05; rebuild the data.")
    return mode


def assistant_spans(text: str, supervise: str = "full", *,
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
        supervise: "full" returns every assistant turn; "final" only the last one.

    Returns:
        Character spans as (start, end) pairs, in order.
    """
    supervise = supervise_mode(supervise)
    assert supervise in ("full", "final"), (
        f"unknown supervise mode: {supervise!r} (assistant_spans selects among "
        "terminated turns; 'cot' and 'response' carve up ONE turn instead and are "
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


def inner_masked_spans(text: str, spans: list[tuple[int, int]],
                       pairs: tuple) -> list[tuple[int, int]]:
    """Find spans to mask that lie INSIDE a supervised assistant turn.

    For most families nothing does: a tool result is its own turn, already outside every
    assistant span. Gemma 4 renders a whole trajectory as ONE `model` turn with its tool
    responses interleaved, so those regions fall inside the supervised span and would
    otherwise become targets - training the model to generate tool output it cannot know.

    Args:
        text: The rendered chat string.
        spans: The supervised assistant spans, as `assistant_spans` returns them.
        pairs: The profile's `masked_inner` - (open, close) literal pairs.

    Returns:
        Character spans to mask, each covering one open..close region (both literals
        included) that lies inside one of `spans`.
    """
    out: list[tuple[int, int]] = []
    for opener, closer in pairs:
        pos = 0
        while (start := text.find(opener, pos)) != -1:
            close = text.find(closer, start + len(opener))
            assert close != -1, (
                f"unterminated {opener!r} at char {start}: the render opens a masked region "
                f"that no {closer!r} closes, so masking cannot tell output from target")
            end = close + len(closer)
            if any(start >= s and end <= e for s, e in spans):
                out.append((start, end))
            pos = end
    return out


def _reasoning_close(text: str, start: int, *, mode: str, think_open: str,
                     think_close: str) -> int:
    """Assert the turn at `start` carries a REAL reasoning block; return its close index.

    The shape test `cot` and `response` share. It replaces a prefix test against the family's
    `empty_think` literal, which cannot work for a family that has none: `"".startswith("")`
    is true of every string, so an empty marker would reject every row
    (google/gemma-4-31B-it). Asking what the block CONTAINS settles both families with one
    rule - Qwen's empty marker holds only a newline between its opener and close, and a Gemma
    turn with no reasoning never opens the channel at all.
    """
    assert text.startswith(think_open, start), (
        f"supervise={mode!r} needs the final assistant turn to open its reasoning with "
        f"{think_open!r}, but it opens {text[start:start + len(think_open)]!r}; a turn with "
        "no think block has no reasoning to train on")
    close = text.find(think_close, start)
    assert close != -1, (
        f"supervise={mode!r}: the final assistant turn never closes its reasoning "
        f"({think_close!r}) - a cut-off trace is not a training target")
    assert text[start + len(think_open):close].strip(), (
        f"supervise={mode!r} on a turn whose reasoning block is EMPTY: it has no reasoning "
        "to train on, and supervising its close would train the empty-think collapse "
        "(gotcha 2). Only rows with a real trace may be flagged for this mode.")
    return close


def cot_span(text: str, *, header: str, prefill: str, empty_think: str,
             think_close: str, think_open: str = "") -> tuple[int, int]:
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
    close = _reasoning_close(text, start, mode="cot",
                             think_open=think_open or prefill,
                             think_close=think_close)
    return start, close + len(think_close)


def answer_span(text: str, *, header: str, prefill: str, empty_think: str,
                think_close: str, turn_end: str, think_open: str = "") -> tuple[int, int]:
    """Return the final answer span, including its separator and turn-end token.

    Reasoning stays in the forward pass but receives no direct loss. On the same
    real-reasoning turn, this span and ``cot_span`` partition normal supervision.
    Empty markers, absent reasoning closes and unterminated turns are refused so
    malformed data cannot silently turn this ablation into ordinary supervision.
    """
    i = text.rfind(header)
    assert i != -1, f"no assistant turn found; nothing would be supervised ({header!r})"
    head = i + len(header)
    close = _reasoning_close(text, head, mode="response",
                             think_open=think_open or prefill,
                             think_close=think_close)
    start = close + len(think_close)
    end = text.find(turn_end, start)
    assert end != -1, (
        f"supervise='response': the final assistant turn is not terminated by {turn_end!r}; "
        "the answer must be a complete, closed turn to be a training target")
    return start, end + len(turn_end)


def generated_spans(text: str, spans: list[tuple[int, int]],
                    prefill: str, think_open: str = "") -> list[tuple[int, int]]:
    """The assistant spans the model GENERATES in this row: those opening with a think block.

    A turn before the last real user message renders with no block and is history (the
    module header's "history is context"); it is dropped here so it earns no loss. The
    empty marker opens with the prefill too, so one test covers both forced shapes. A row
    with no block on any turn is returned whole: it was not rendered for thinking at all.
    """
    # A family that forces no opener states it as `think_open`; without that, `opener` would
    # be "" here and every turn would look generated, silently supervising history.
    opener = think_open or prefill
    headed = [(s, e) for s, e in spans if text.startswith(opener, s)]
    return headed or spans


def forced_spans(text: str, spans: list[tuple[int, int]],
                 prefill: str, empty_think: str) -> list[tuple[int, int]]:
    """Find the forced (never-generated) region at the head of each assistant span.

    Every turn is checked. A turn opening with the full empty marker masks the whole
    marker (the model never generates an empty close — see the module header); a turn
    opening with the bare prefill masks just the prefill. A turn without a think block
    has no forced span; whether it is supervised at all is `generated_spans`' question.
    """
    out: list[tuple[int, int]] = []
    for s, _ in spans:
        if text.startswith(empty_think, s):
            out.append((s, s + len(empty_think)))
        elif text.startswith(prefill, s):
            out.append((s, s + len(prefill)))
    return out


def build_labels(text: str, tokenizer, max_length: int, profile: ModelProfile,
                 supervise: str = "full",
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
        supervise: "full" trains every generated assistant turn; "final" only the last
            one; "cot" only the final turn's reasoning, TRUNCATING the row at its close
            so the response leaves the token stream (see the module header); "response"
            only what follows the final turn's reasoning, WITHOUT truncating, so the trace stays as
            unsupervised context. Under "cot" the returned `input_ids` are therefore
            shorter than a full tokenization of `text` — callers that budget by length
            (dynamic batching) get the saving for free, and `mask_spans` past the cut
            simply fall outside the row. Under "response" the token stream is the full
            row, identical to "full".
        mask_spans: Optional CHARACTER spans of `text` to unsupervise on top of the rule
            above — a row-level ablation that removes one property of the reasoning from
            the loss while leaving the token stream untouched, so a masked arm and its
            control tokenize identically. A token overlapping a span at all is masked
            whole, which can take one boundary token beyond the span.

    Returns:
        A dict with `input_ids`, `attention_mask` and `labels`.
    """
    turn_kw = dict(header=profile.assistant_header, turn_end=profile.turn_end)
    supervise = supervise_mode(supervise)
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
    elif supervise == "response":
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
        every = assistant_spans(text, **turn_kw)
        # History earns no loss (module header): only turns rendered with a think block
        # are ones the model generates here. "final" then keeps the last of those.
        spans = generated_spans(text, every, profile.prefill)
        if supervise == "final":
            spans = spans[-1:]
        # Forced heads are masked on EVERY turn (supervised or not) -- an unsupervised
        # first turn is wholly -100 already, so this only matters for the supervised ones.
        prefills = forced_spans(text, every, profile.prefill, profile.empty_think)
    # Regions a family interleaves INSIDE its own assistant turn that it never generates -
    # Gemma 4's tool responses. Masked by the same machinery as a forced head, which also puts
    # a tokenizer boundary at each edge so no token straddles target and context.
    prefills = prefills + inner_masked_spans(text, spans, profile.masked_inner)
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
                               empty_think: str | None = None) -> None:
    """Fail fast when a train config's `thinking:` declaration contradicts the data.

    The declaration is the source of truth (the config is the scientific record); this
    check refuses the one combination that mislabels an artifact: `thinking: false` over
    data that carries real reasoning traces. Empty `<think></think>` markers are fine
    under thinking=true: the generation-boundary mask excludes the whole marker from the
    loss (it is forced context in every serving configuration — the model never generates
    an empty close), so it conditions without ever being trained. For the same reason a
    dataset with NO traces at all may be declared `thinking: true` (the nosynth control):
    nothing about thinking is trained either way — `build_labels` never sees the flag —
    and the declaration only chooses the mode the arm is served and evaluated in, which
    for a control of thinking-mode arms is thinking mode. Until 2026-09-05 this function
    also required at least one trace under thinking=true; that guard predated the
    unconditional whole-marker mask (2026-08-04) and described a collapse it now prevents.

    Args:
        rows: Dataset rows, each carrying either a rendered `text` string or a raw
            `messages` list.
        thinking: The train config's declared eval-time mode for this arm.
        empty_think: The profile's empty-marker literal, needed to classify rendered
            `text` rows. None is allowed only for pure-`messages` datasets (an
            unprofiled family's interchange data); a text row then raises rather than
            counting with another family's literal.

    Raises:
        AssertionError: any real reasoning trace under thinking=false.
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
    if not thinking:
        assert real == 0, (
            f"thinking: false, but {real} real reasoning traces are in the training data — "
            "the declaration mislabels this arm")
    elif real == 0:
        print(">>> thinking: true over a dataset with NO reasoning traces: every think "
              "block is an empty marker, masked whole, so this arm is never trained to "
              "reason nor to stop reasoning; it is served in thinking mode with the base "
              "model's own reasoning intact (the nosynth control's shape).")


def supervise_census(values) -> tuple[dict[str, int], str | None]:
    """Count a dataset's per-row `supervise` modes; warn when the column changes nothing.

    An absent value means "full", and a column whose every row is "full" trains exactly as no
    column would. That is VALID data -- a corpus may state its supervision explicitly -- so
    it is not refused here. It is still worth saying out loud, because the other way to get
    there is a `cot` / `final` / `response` arm whose override never applied; that mistake is
    refused where the intent is known (build_mixture checks a source's `supervise:` against
    the config's declared `variant:`), and the counts returned here land in
    training_meta.json, so a variant that collapsed into its control stays visible in the
    artifact.

    Args:
        values: The dataset's `supervise` column (None / "" read as "full").

    Returns:
        (counts by mode, most common first; a warning string, or None when some row is
        not "full").
    """
    counts = Counter(supervise_mode(v) for v in values)
    ordered = dict(counts.most_common())
    if set(ordered) <= {"full"}:
        return ordered, (
            "supervise column present but every row is 'full': this trains exactly as a "
            "dataset with no supervise column. If this arm is meant to be a supervise "
            "variant (cot / final / response), its override did not apply.")
    return ordered, None
