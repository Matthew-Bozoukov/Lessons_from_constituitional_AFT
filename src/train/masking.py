# ABOUTME: Builds assistant-only loss masks over pre-rendered Qwen chat text, so SFT
# ABOUTME: supervises only the tokens the model would itself generate.

from __future__ import annotations

ASSISTANT_HEADER = "<|im_start|>assistant\n"
TURN_END = "<|im_end|>"
# What Qwen3.6's template emits for a final assistant turn carrying no reasoning: its
# explicit non-thinking marker, injected as a prefill at inference time.
EMPTY_THINK = "<think>\n\n</think>\n\n"


def assistant_spans(text: str, skip_empty_think: bool = False,
                    supervise: str = "all") -> list[tuple[int, int]]:
    """Find the character spans of assistant content in a rendered chat string.

    A span runs from just after the `<|im_start|>assistant\\n` header through the
    closing `<|im_end|>` inclusive. The header is excluded because it is given to the
    model at inference time; `<|im_end|>` is included because the model must learn to
    emit it and stop.

    Args:
        text: A chat conversation already rendered by the Qwen chat template.
        skip_empty_think: Also exclude a leading empty `<think></think>` block, so the model
            is conditioned on the non-thinking marker without being trained to emit it.
            Training a model to emit one is the documented reasoning-collapse pattern. Real
            reasoning traces are unaffected -- only the exact empty literal is skipped.
        supervise: "all" trains every assistant turn; "final" only the last one. "final"
            is how the MEM self-reflection records keep their first (possibly flawed)
            response as context without making it a training target.

    Returns:
        Character spans as (start, end) pairs, in order.
    """
    assert supervise in ("all", "final"), f"unknown supervise mode: {supervise!r}"
    spans: list[tuple[int, int]] = []
    pos = 0
    while (i := text.find(ASSISTANT_HEADER, pos)) != -1:
        start = i + len(ASSISTANT_HEADER)
        if skip_empty_think and text.startswith(EMPTY_THINK, start):
            start += len(EMPTY_THINK)
        end = text.find(TURN_END, start)
        assert end != -1, f"assistant turn at char {i} is not terminated by {TURN_END}"
        end += len(TURN_END)
        spans.append((start, end))
        pos = end
    assert spans, "no assistant turn found; nothing would be supervised"
    return spans[-1:] if supervise == "final" else spans


def build_labels(text: str, tokenizer, max_length: int,
                 skip_empty_think: bool = False,
                 supervise: str = "all") -> dict[str, list[int]]:
    """Tokenize a rendered conversation and label only its assistant tokens.

    Every token outside an assistant span is set to -100 so it contributes no loss.
    Token/character alignment comes from the fast tokenizer's offset mapping, which
    keeps this independent of the chat template's internals -- Qwen3.6's template has
    no `{% generation %}` markers, so TRL's own `assistant_only_loss` cannot be used.

    Args:
        text: A chat conversation already rendered by the Qwen chat template.
        tokenizer: A fast tokenizer for the model being trained.
        max_length: Truncation length, matching the training sequence length.
        skip_empty_think: Exclude a leading empty `<think></think>` block from supervision.
        supervise: "all" | "final" -- which assistant turns carry loss (see
            `assistant_spans`).

    Returns:
        A dict with `input_ids`, `attention_mask` and `labels`.
    """
    enc = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    spans = assistant_spans(text, skip_empty_think=skip_empty_think, supervise=supervise)

    labels = [-100] * len(ids)
    for k, (a, b) in enumerate(offsets):
        if b <= a:  # zero-width: a special token the tokenizer inserted itself
            continue
        if any(a >= s and b <= e for s, e in spans):
            labels[k] = ids[k]

    assert any(v != -100 for v in labels), "truncation left an example with no supervised token"
    return {"input_ids": ids, "attention_mask": enc["attention_mask"], "labels": labels}


def check_thinking_declaration(rows, thinking: bool, mask_empty_think: bool = False) -> None:
    """Fail fast when a train config's `thinking:` declaration contradicts the data.

    The declaration is the source of truth (the config is the scientific record); this
    check only refuses combinations that would produce a mislabeled or reasoning-collapsed
    artifact (CLAUDE.md gotcha 2).

    Args:
        rows: Dataset rows, each carrying either a rendered `text` string or a raw
            `messages` list.
        thinking: The train config's declared eval-time mode for this arm.
        mask_empty_think: Whether training excludes empty-think markers from the loss.

    Raises:
        AssertionError: declared thinking with no real reasoning trace anywhere; unmasked
            empty-think markers under thinking=true; or any think content under
            thinking=false.
    """
    real = empty = 0
    for row in rows:
        if "text" in row:
            total = row["text"].count("<think>")
            e = row["text"].count(EMPTY_THINK)
            real += total - e
            empty += e
        else:
            real += sum(1 for msg in row["messages"]
                        if str(msg.get("reasoning_content") or "").strip())
    if thinking:
        assert real > 0, (
            "thinking: true, but no row carries a real reasoning trace — this would train "
            "the model on empty/absent <think> and collapse its reasoning (gotcha 2)")
        assert empty == 0 or mask_empty_think, (
            f"{empty} empty <think></think> markers are in the training text but "
            "train.mask_empty_think is off — the model would be trained to emit the "
            "reasoning-collapse pattern")
    else:
        assert real == 0, (
            f"thinking: false, but {real} real reasoning traces are in the training data — "
            "the declaration mislabels this arm")
