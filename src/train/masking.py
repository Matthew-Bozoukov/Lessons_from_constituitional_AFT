# ABOUTME: Builds assistant-only loss masks over pre-rendered Qwen chat text, so SFT
# ABOUTME: supervises only the tokens the model would itself generate.

from __future__ import annotations

ASSISTANT_HEADER = "<|im_start|>assistant\n"
TURN_END = "<|im_end|>"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# The think-loss rule to use for new work: mask the `<think>` opener, always supervise
# `</think>`. The opener is Qwen3.6's marker, injected as a prefill at inference, so the model
# should be conditioned on it; closing the block is behaviour it must learn.
THINK_LOSS_CLOSING_ONLY = "closing_only"

# DEPRECATED rules. Still selectable BY NAME so the runs that used them stay reproducible --
# notably the lora_qwen36_500k_numina_heavy{,_emptythink} pair, which is an ablation OF these
# two rules and would collapse into duplicate configs if both were repointed at closing_only.
# They are never reached implicitly: `think_loss` is required, so a config gets one of these
# only by naming it, and naming it prints a deprecation warning.
#
#   "both"       -- loss on BOTH <think> and </think>. The implicit behaviour before
#                   2026-08-03; produced the ...-both-think-tokens-loss adapter. Under an
#                   always_think mixture this trains the model to EMIT the non-thinking
#                   marker, which is the documented reasoning-collapse pattern.
#   "skip_empty" -- the former `mask_empty_think`: drop an empty <think></think> from the loss
#                   entirely, supervising neither tag. Superseded because `</think>` should
#                   always be learned.
DEPRECATED_THINK_LOSS = {
    "both": "loss on both think tokens",
    "skip_empty": "the former mask_empty_think rule (skip an empty block entirely)",
}
THINK_LOSS_RULES = (THINK_LOSS_CLOSING_ONLY, *DEPRECATED_THINK_LOSS)

# What Qwen3.6's template emits for an assistant turn carrying no reasoning. Only "skip_empty"
# needs the literal; the current rule keys off the opening tag alone.
EMPTY_THINK = "<think>\n\n</think>\n\n"


def assistant_spans(text: str, think_loss: str = THINK_LOSS_CLOSING_ONLY) -> list[tuple[int, int]]:
    """Find the character spans of assistant content in a rendered chat string.

    A span runs from just after the `<|im_start|>assistant\\n` header through the
    closing `<|im_end|>` inclusive. The header is excluded because it is given to the
    model at inference time; `<|im_end|>` is included because the model must learn to
    emit it and stop.

    Args:
        text: A chat conversation already rendered by the Qwen chat template.
        think_loss: Rule name. Only the DEPRECATED "skip_empty" changes spans here, by
            starting the span after a leading empty `<think></think>`; the current rule
            withholds the opener via `think_open_spans` instead.

    Returns:
        Character spans as (start, end) pairs, in order.
    """
    spans: list[tuple[int, int]] = []
    pos = 0
    while (i := text.find(ASSISTANT_HEADER, pos)) != -1:
        start = i + len(ASSISTANT_HEADER)
        if think_loss == "skip_empty" and text.startswith(EMPTY_THINK, start):
            start += len(EMPTY_THINK)
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


def build_labels(text: str, tokenizer, max_length: int,
                 think_loss: str = THINK_LOSS_CLOSING_ONLY) -> dict[str, list[int]]:
    """Tokenize a rendered conversation and label only its assistant tokens.

    Every token outside an assistant span is set to -100 so it contributes no loss. Under
    the current rule so is every token wholly inside a `<think>` opener (see
    `think_open_spans`). Token/character alignment comes from the fast tokenizer's offset
    mapping, which keeps this independent of the chat template's internals -- Qwen3.6's
    template has no `{% generation %}` markers, so TRL's own `assistant_only_loss` cannot
    be used.

    Args:
        text: A chat conversation already rendered by the Qwen chat template.
        tokenizer: A fast tokenizer for the model being trained.
        max_length: Truncation length, matching the training sequence length.
        think_loss: One of `THINK_LOSS_RULES`. Defaults to the current rule; the other two
            are deprecated and exist so published runs stay reproducible.

    Returns:
        A dict with `input_ids`, `attention_mask` and `labels`.
    """
    assert think_loss in THINK_LOSS_RULES, f"unknown think_loss {think_loss!r}"
    enc = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    spans = assistant_spans(text, think_loss=think_loss)
    # Only the current rule withholds the opener; both deprecated rules supervise whatever
    # falls inside their assistant spans.
    holes = think_open_spans(text) if think_loss == THINK_LOSS_CLOSING_ONLY else []

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


def resolve_think_loss(train_cfg) -> str:
    """Return the config's think-loss rule, refusing to guess and warning on deprecated ones.

    `think_loss` is required. Defaulting it is what this function exists to prevent: three
    rules have been in use, and inferring one would change a config's results with no diff
    to explain why.

    Args:
        train_cfg: The `train` block of a training config.

    Returns:
        The validated rule name.

    Raises:
        ValueError: If `think_loss` is missing or unknown, or if the removed
            `mask_empty_think` key is still present.
    """
    if train_cfg.get("mask_empty_think") is not None:
        raise ValueError(
            "`mask_empty_think` was replaced by `think_loss` on 2026-08-03. Set "
            "`think_loss: skip_empty` for identical (deprecated) behaviour, or "
            f"`think_loss: {THINK_LOSS_CLOSING_ONLY}` for the current rule, and drop "
            "`mask_empty_think`."
        )
    mode = train_cfg.get("think_loss")
    if mode is None:
        raise ValueError(
            "`train.think_loss` is required. Core used to default to loss on BOTH think "
            f"tokens; set `think_loss: {THINK_LOSS_CLOSING_ONLY}` for the current rule (mask "
            "the `<think>` opener, always supervise `</think>`), or `think_loss: both` to "
            f"keep the old behaviour. Valid values: {', '.join(THINK_LOSS_RULES)}."
        )
    if mode not in THINK_LOSS_RULES:
        raise ValueError(
            f"unknown `think_loss: {mode}`; valid values are {', '.join(THINK_LOSS_RULES)}"
        )
    if mode in DEPRECATED_THINK_LOSS:
        print(f"!!! WARNING: `think_loss: {mode}` is DEPRECATED -- "
              f"{DEPRECATED_THINK_LOSS[mode]}. It is kept so published runs stay "
              f"reproducible. New work should use {THINK_LOSS_CLOSING_ONLY!r}.")
    return mode
