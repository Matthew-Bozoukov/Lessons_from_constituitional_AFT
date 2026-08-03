# ABOUTME: Builds assistant-only loss masks over pre-rendered Qwen chat text, so SFT
# ABOUTME: supervises only the tokens the model would itself generate.

from __future__ import annotations

ASSISTANT_HEADER = "<|im_start|>assistant\n"
TURN_END = "<|im_end|>"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# Config value selecting the one supported think-loss rule. Required rather than defaulted:
# two earlier rules were in use, and silently reinterpreting a config written for one of
# them would change results with no diff to explain why.
THINK_LOSS_CLOSING_ONLY = "closing_only"

# DEPRECATED think-loss rules, removed from core on 2026-08-03. Configs naming one, or
# omitting `think_loss` entirely, now fail loudly rather than changing behaviour in silence.
# The removed implementations are preserved verbatim in scratch/deprecated/think_loss_legacy.py.
#
#   "both"       -- loss on BOTH <think> and </think>. The implicit behaviour before this
#                   change; produced the ...-both-think-tokens-loss adapter.
#   "skip_empty" -- the `mask_empty_think` rule: drop an empty <think></think> from the loss
#                   entirely, supervising neither tag.
DEPRECATED_THINK_LOSS = {
    "both": "loss on both think tokens",
    "skip_empty": "the mask_empty_think rule (skip an empty block entirely)",
}


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


def think_open_spans(text: str) -> list[tuple[int, int]]:
    """Find the character spans this rule withholds from the loss: each `<think>` opener.

    A span covers the `<think>` literal plus **one** following newline character. Whether
    that newline is really withheld is decided per token by `build_labels`, which masks only
    tokens lying WHOLLY inside one of these spans:

    - `<think>\\n` before real reasoning -- Qwen emits that `\\n` as its own token, wholly
      inside the span, so it is masked.
    - `<think>\\n\\n</think>` (an empty block) -- Qwen emits `\\n\\n` as ONE token, only partly
      inside the span, so it stays supervised.

    One predicate therefore covers both, with no special casing for empty blocks. `</think>`
    never falls in one of these spans and is always supervised, which is the point of the
    rule: condition the model on the opening tag rather than train it to emit one, while
    always teaching it to close the block.

    `</think>` cannot match here -- `str.find` seeks the literal `<think>`, and the closing
    tag's `/` prevents a match.

    Args:
        text: A chat conversation already rendered by the Qwen chat template.

    Returns:
        Character spans as (start, end) pairs, in order.
    """
    spans: list[tuple[int, int]] = []
    pos = 0
    while (i := text.find(THINK_OPEN, pos)) != -1:
        end = i + len(THINK_OPEN)
        if text.startswith("\n", end):
            end += 1
        spans.append((i, end))
        pos = end
    return spans


def build_labels(text: str, tokenizer, max_length: int) -> dict[str, list[int]]:
    """Tokenize a rendered conversation and label only its assistant tokens.

    Every token outside an assistant span is set to -100 so it contributes no loss, as is
    every token wholly inside a `<think>` opener (see `think_open_spans`). Token/character
    alignment comes from the fast tokenizer's offset mapping, which keeps this independent
    of the chat template's internals -- Qwen3.6's template has no `{% generation %}`
    markers, so TRL's own `assistant_only_loss` cannot be used.

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
    holes = think_open_spans(text)

    labels = [-100] * len(ids)
    for k, (a, b) in enumerate(offsets):
        if b <= a:  # zero-width: a special token the tokenizer inserted itself
            continue
        if any(a >= s and b <= e for s, e in holes):
            continue
        if any(a >= s and b <= e for s, e in spans):
            labels[k] = ids[k]

    assert any(v != -100 for v in labels), "truncation left an example with no supervised token"
    return {"input_ids": ids, "attention_mask": enc["attention_mask"], "labels": labels}


def check_think_loss_config(train_cfg) -> None:
    """Reject configs written for a removed think-loss rule instead of reinterpreting them.

    Args:
        train_cfg: The `train` block of a training config.

    Raises:
        ValueError: If `think_loss` is missing, names a deprecated or unknown rule, or if
            the removed `mask_empty_think` key is present.
    """
    if train_cfg.get("mask_empty_think") is not None:
        raise ValueError(
            "`mask_empty_think` is deprecated and removed from core (2026-08-03). It skipped "
            "an empty <think></think> entirely, supervising neither tag; core now always "
            f"supervises `</think>`. Set `think_loss: {THINK_LOSS_CLOSING_ONLY}` and drop "
            "`mask_empty_think`. The old implementation is kept, unused, in "
            "scratch/deprecated/think_loss_legacy.py."
        )
    mode = train_cfg.get("think_loss")
    if mode is None:
        raise ValueError(
            "`train.think_loss` is required. Core previously defaulted to loss on BOTH think "
            "tokens; that rule is deprecated and removed, so this config would otherwise "
            f"change behaviour silently. Set `think_loss: {THINK_LOSS_CLOSING_ONLY}` to "
            "confirm the current rule (mask the `<think>` opener, always supervise "
            "`</think>`), or see scratch/deprecated/think_loss_legacy.py for what was removed."
        )
    if mode in DEPRECATED_THINK_LOSS:
        raise ValueError(
            f"`think_loss: {mode}` is deprecated and removed from core (2026-08-03) -- "
            f"{DEPRECATED_THINK_LOSS[mode]}. Only {THINK_LOSS_CLOSING_ONLY!r} is supported. "
            "The removed implementations are kept, unused, in "
            "scratch/deprecated/think_loss_legacy.py."
        )
    if mode != THINK_LOSS_CLOSING_ONLY:
        raise ValueError(
            f"unknown `think_loss: {mode}`; the only supported value is "
            f"{THINK_LOSS_CLOSING_ONLY!r}"
        )
