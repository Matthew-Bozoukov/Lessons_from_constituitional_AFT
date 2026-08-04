# ABOUTME: Builds assistant-only loss masks over pre-rendered Qwen chat text, so SFT
# ABOUTME: supervises exactly the tokens the model would itself generate at inference.

from __future__ import annotations

ASSISTANT_HEADER = "<|im_start|>assistant\n"
TURN_END = "<|im_end|>"

# The generation-boundary rule (the ONE way think tokens are supervised — deliberately not
# configurable; git history reproduces runs trained under older rules): mask exactly what
# the serving template prefills, supervise exactly what the model generates. Qwen3.6's
# thinking-mode generation prompt ends with this literal (verified against the live
# template in tests/test_masking_tokenizer.py), so inside a rendered assistant turn it is
# conditioning, and everything after it — including the `\n</think>` that closes an empty
# block — is behaviour the model must learn to emit.
THINK_PREFILL = "<think>\n"

# The literal is family-specific, so its use is gated: Qwen3's thinking template prefills
# NOTHING — that model generates `<think>` itself, and masking the opener there would
# under-train tokens the model must emit. Deriving the prefill from the artifact's own
# template is the general fix; until that lands, refuse anything but Qwen3.6.
GENERATION_BOUNDARY_FAMILY = "Qwen3.6"

# What both families' templates emit for an assistant turn carrying no reasoning, prefilled
# whole at nothink inference. Used by data checks; the mask needs only THINK_PREFILL.
EMPTY_THINK = "<think>\n\n</think>\n\n"


def assert_generation_boundary_family(model_name: str) -> None:
    """Refuse to mask a family whose thinking prefill we have not verified.

    Args:
        model_name: The base model id from the train config (e.g. "Qwen/Qwen3.6-27B").

    Raises:
        ValueError: If the model is not the family THINK_PREFILL was verified against.
    """
    if GENERATION_BOUNDARY_FAMILY not in model_name:
        raise ValueError(
            f"generation-boundary masking hardcodes the {GENERATION_BOUNDARY_FAMILY} "
            f"thinking prefill {THINK_PREFILL!r} and has only been verified for that "
            f"family; got model {model_name!r}. Qwen3's template prefills nothing in "
            "thinking mode (the model emits <think> itself), so this mask would "
            "under-train it. Derive the prefill from the model's own chat template "
            "before lifting this gate."
        )


def assistant_spans(text: str) -> list[tuple[int, int]]:
    """Find the character spans of assistant content in a rendered chat string.

    A span runs from just after the `<|im_start|>assistant\\n` header through the
    closing `<|im_end|>` inclusive. The header is excluded because it is given to the
    model at inference time; `<|im_end|>` is included because the model must learn to
    emit it and stop.

    Args:
        text: A chat conversation already rendered by the Qwen chat template.

    Returns:
        Character spans as (start, end) pairs, in order.
    """
    spans: list[tuple[int, int]] = []
    pos = 0
    while (i := text.find(ASSISTANT_HEADER, pos)) != -1:
        start = i + len(ASSISTANT_HEADER)
        end = text.find(TURN_END, start)
        assert end != -1, f"assistant turn at char {i} is not terminated by {TURN_END}"
        end += len(TURN_END)
        spans.append((start, end))
        pos = end
    assert spans, "no assistant turn found; nothing would be supervised"
    return spans


def prefill_spans(text: str, spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Find the `<think>\\n` prefill at the head of each assistant span, where present.

    Turns without a think block (e.g. non-final turns the template rendered bare) simply
    have no prefill span and stay fully supervised.
    """
    return [(s, s + len(THINK_PREFILL)) for s, _ in spans
            if text.startswith(THINK_PREFILL, s)]


def build_labels(text: str, tokenizer, max_length: int) -> dict[str, list[int]]:
    """Tokenize a rendered conversation and label exactly its generated tokens.

    Every token outside an assistant span is -100, and so is every token of a turn's
    `<think>\\n` prefill. The text is tokenized in SEGMENTS cut at each prefill boundary,
    because the boundary must also be a token boundary: in an empty think block
    `<think>\\n\\n</think>` the two newlines otherwise merge into ONE token, welding the
    prefilled `\\n` (conditioning) to the generated `\\n` (supervised). Cutting forces the
    same token stream the model sees at inference — context ending in the prefill's
    `\\n`, generation starting with its own `\\n</think>` — which is the entire point of
    masking at the generation boundary. Token/char alignment within a segment comes from
    the fast tokenizer's offset mapping (Qwen's template has no `{% generation %}`
    markers, so TRL's own assistant_only_loss cannot be used).

    Args:
        text: A chat conversation already rendered by the Qwen chat template.
        tokenizer: A fast tokenizer for the model being trained.
        max_length: Truncation length, matching the training sequence length.

    Returns:
        A dict with `input_ids`, `attention_mask` and `labels`.
    """
    spans = assistant_spans(text)
    prefills = prefill_spans(text, spans)
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


def check_thinking_declaration(rows, thinking: bool) -> None:
    """Fail fast when a train config's `thinking:` declaration contradicts the data.

    The declaration is the source of truth (the config is the scientific record); this
    check only refuses combinations that would produce a mislabeled or reasoning-collapsed
    artifact (CLAUDE.md gotcha 2). Empty `<think></think>` markers are fine under
    thinking=true: the generation-boundary mask conditions on their prefill and supervises
    their `\\n</think>` close, which is exactly what a thinking-mode model emits when it
    declines to reason.

    Args:
        rows: Dataset rows, each carrying either a rendered `text` string or a raw
            `messages` list.
        thinking: The train config's declared eval-time mode for this arm.

    Raises:
        AssertionError: declared thinking with no real reasoning trace anywhere, or any
            think content under thinking=false.
    """
    real = 0
    for row in rows:
        if "text" in row:
            real += row["text"].count("<think>") - row["text"].count(EMPTY_THINK)
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
