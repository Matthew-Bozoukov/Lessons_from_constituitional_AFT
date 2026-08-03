# ABOUTME: Tiny offline sanity check of the think-loss rule, token by token, on the real
# ABOUTME: Qwen tokenizer. Run: python scratch/sanity_think_mask.py

"""Print the exact mask decision for every token of a few hand-written turns.

The point is to see the rule rather than trust it: the `<think>` opener masked, `</think>`
always supervised, and the `\\n\\n` of an empty block supervised because Qwen emits it as one
token that only partly overlaps the opener span. Needs the tokenizer only -- no GPU, no data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train.masking import build_labels  # noqa: E402

SCENARIOS = {
    "1. empty think block (95.6% of the mixture)":
        "<|im_start|>user\nlook it up<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\nI'll look that up.<|im_end|>\n",
    "2. real reasoning":
        "<|im_start|>user\nwhere is it<|im_end|>\n"
        "<|im_start|>assistant\n<think>\nLet me check the directory.\n</think>\n\n"
        "Found it.<|im_end|>\n",
    "3. system + tool turns are never supervised":
        "<|im_start|>system\nYou have tools.<|im_end|>\n"
        "<|im_start|>user\nrun it<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>go</tool_call><|im_end|>\n"
        "<|im_start|>tool\nresult=7<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\nIt is 7.<|im_end|>\n",
    "4. no newline after the opener":
        "<|im_start|>assistant\n<think>reasoning</think>ok<|im_end|>\n",
    "5. triple newline (also one token)":
        "<|im_start|>assistant\n<think>\n\n\n</think>x<|im_end|>\n",
}


def main(tokenizer: str = "Qwen/Qwen3.6-27B", max_seq_len: int = 4096) -> None:
    """Print per-token mask decisions for each scenario and assert the rule holds.

    Args:
        tokenizer: Tokenizer of the model being trained.
        max_seq_len: Truncation length; irrelevant at these sizes but kept honest.
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer)
    open_id, close_id = tok.convert_tokens_to_ids(["<think>", "</think>"])
    failures = 0

    for name, text in SCENARIOS.items():
        enc = build_labels(text, tok, max_seq_len)
        offsets = tok(text, add_special_tokens=False, truncation=True,
                      max_length=max_seq_len, return_offsets_mapping=True)["offset_mapping"]
        print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
        for (a, b), tid, lab in zip(offsets, enc["input_ids"], enc["labels"]):
            mark = "." if lab == -100 else "LOSS"
            print(f"  {mark:<5} {repr(text[a:b])}")

        sup = {tid for tid, lab in zip(enc["input_ids"], enc["labels"]) if lab != -100}
        n_open = enc["input_ids"].count(open_id)
        n_close = enc["input_ids"].count(close_id)
        n_close_sup = sum(1 for t, v in zip(enc["input_ids"], enc["labels"])
                          if t == close_id and v != -100)
        if open_id in sup:
            print("  !!! <think> carried loss"); failures += 1
        if n_close_sup != n_close:
            print(f"  !!! {n_close - n_close_sup}/{n_close} </think> missed loss"); failures += 1
        print(f"  -> {n_open} <think> all masked, {n_close_sup}/{n_close} </think> supervised")

    if failures:
        raise SystemExit(f"SANITY CHECK FAILED: {failures} problem(s)")
    print("\nSANITY CHECK PASSED: `<think>` never supervised, `</think>` always supervised.")


if __name__ == "__main__":
    fire.Fire(main)
