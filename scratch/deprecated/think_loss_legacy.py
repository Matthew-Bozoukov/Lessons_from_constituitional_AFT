# ABOUTME: DEPRECATED think-loss rules removed from src/train/masking.py on 2026-08-03.
# ABOUTME: Kept only so published runs stay reproducible. Nothing imports this; do not use it.

"""The two think-loss rules core no longer implements.

Core now has exactly one rule -- mask the `<think>` opener, always supervise `</think>`
(`THINK_LOSS_CLOSING_ONLY`). The two rules below were in use before that and are preserved
here verbatim so the adapters they produced remain reproducible. `src/train/train_lora.py`
raises on any config that asks for them.

They are recorded, not offered. If you are choosing a masking rule for a new run, use core.

## "both" -- loss on BOTH think tokens

Core's implicit behaviour until 2026-08-03: no think-specific masking at all, so an empty
`<think></think>` was fully supervised. Under `always_think` mixtures, where ~96% of think
blocks are empty, that trains the model to *emit* the non-thinking marker -- the documented
reasoning-collapse pattern (CLAUDE.md gotcha #2).

Produced `LASR-Callum/nika-sft-tulu-toolcall-80-20-both-think-tokens-loss`
(2026-08-03, 112 steps, loss 0.853 -> 0.762). To reproduce it, `build_labels` is core's
current function with the `think_open_spans` hole-punching removed -- i.e. supervise every
token wholly inside an assistant span, with no think-specific exclusion at all.

## "skip_empty" -- the `mask_empty_think` rule

Added on main in commit 26444e7 and superseded the same day. It excluded an empty
`<think></think>` block from the loss *entirely*, supervising neither the opening nor the
closing tag, so the model was conditioned on the marker without being trained to emit one.
Core supersedes it by always supervising `</think>`: closing the block is behaviour the
model must learn, and only the opener is context.

The removed implementation follows.
"""

from __future__ import annotations

ASSISTANT_HEADER = "<|im_start|>assistant\n"
TURN_END = "<|im_end|>"
# What Qwen3.6's template emits for a final assistant turn carrying no reasoning.
EMPTY_THINK = "<think>\n\n</think>\n\n"


def assistant_spans_legacy(text: str, skip_empty_think: bool = False) -> list[tuple[int, int]]:
    """DEPRECATED. Core's `assistant_spans` before 2026-08-03.

    Args:
        text: A conversation rendered by the Qwen chat template.
        skip_empty_think: False reproduces the "both" rule; True reproduces "skip_empty",
            excluding a leading empty `<think></think>` block from supervision entirely.

    Returns:
        Character spans as (start, end) pairs, in order.
    """
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
    return spans


def build_labels_legacy(text: str, tokenizer, max_length: int,
                        skip_empty_think: bool = False) -> dict[str, list[int]]:
    """DEPRECATED. Core's `build_labels` before 2026-08-03. See module docstring.

    Args:
        text: A conversation rendered by the Qwen chat template.
        tokenizer: A fast tokenizer for the model being trained.
        max_length: Truncation length.
        skip_empty_think: Selects "skip_empty" (True) or "both" (False).

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
    spans = assistant_spans_legacy(text, skip_empty_think=skip_empty_think)

    labels = [-100] * len(ids)
    for k, (a, b) in enumerate(offsets):
        if b <= a:
            continue
        if any(a >= s and b <= e for s, e in spans):
            labels[k] = ids[k]

    assert any(v != -100 for v in labels), "truncation left an example with no supervised token"
    return {"input_ids": ids, "attention_mask": enc["attention_mask"], "labels": labels}
