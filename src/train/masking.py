# ABOUTME: Builds assistant-only loss masks over pre-rendered Qwen chat text, so SFT
# ABOUTME: supervises only the tokens the model would itself generate.

from __future__ import annotations

ASSISTANT_HEADER = "<|im_start|>assistant\n"
TURN_END = "<|im_end|>"


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


def build_labels(text: str, tokenizer, max_length: int) -> dict[str, list[int]]:
    """Tokenize a rendered conversation and label only its assistant tokens.

    Every token outside an assistant span is set to -100 so it contributes no loss.
    Token/character alignment comes from the fast tokenizer's offset mapping, which
    keeps this independent of the chat template's internals -- Qwen3.6's template has
    no `{% generation %}` markers, so TRL's own `assistant_only_loss` cannot be used.

    Args:
        text: A chat conversation already rendered by the Qwen chat template.
        tokenizer: A fast tokenizer for the model being trained.
        max_length: Truncation length, matching the training sequence length.

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
    spans = assistant_spans(text)

    labels = [-100] * len(ids)
    for k, (a, b) in enumerate(offsets):
        if b <= a:  # zero-width: a special token the tokenizer inserted itself
            continue
        if any(a >= s and b <= e for s, e in spans):
            labels[k] = ids[k]

    assert any(v != -100 for v in labels), "truncation left an example with no supervised token"
    return {"input_ids": ids, "attention_mask": enc["attention_mask"], "labels": labels}
