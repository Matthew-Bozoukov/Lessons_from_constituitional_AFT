# ABOUTME: Build a token-budgeted SFT dataset from Tulu 3 for the 0%-synthetic control arm.
# ABOUTME: Run: uv run python src/data/prepare_tulu.py --config configs/tulu_control_data.yaml

"""Sample Tulu 3 down to an exact *token* budget.

This is the control arm for the internalization study: the treatment mixes synthetic
constitution documents with a Tulu baseline (`synthdoc`'s `export.baseline`), so the
control has to be the same SFT *dose* with the synthetic fraction set to zero. Budgeting
by tokens rather than by example count is what makes "same dose" meaningful — Tulu's
examples range from one-line prompts to long multi-turn transcripts, so a fixed example
count would leave the two arms seeing very different amounts of text.

Selection is deterministic given `seed`: the stream is shuffled through a fixed-size
buffer and consumed in order until the budget is met, so re-running reproduces the
dataset without re-downloading the full 939k-example mixture.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import timestamp, write_run_meta  # noqa: E402


def _clean(messages: list[dict]) -> list[dict] | None:
    """Normalise a Tulu record's messages, or return None if unusable.

    Tulu 3 rows are already OpenAI-style, but a few carry empty turns or roles the
    Qwen chat template will not render. Those are dropped rather than repaired: a
    silently mangled example is worse than a slightly smaller sample.

    Args:
        messages: Raw `messages` field from a Tulu record.

    Returns:
        The cleaned message list, or None if the record should be skipped.
    """
    if not isinstance(messages, list) or not messages:
        return None
    out = []
    for m in messages:
        role, content = m.get("role"), m.get("content")
        if role not in ("system", "user", "assistant") or not isinstance(content, str):
            return None
        if not content.strip():
            return None
        out.append({"role": role, "content": content})
    # Must contain something to learn from.
    if not any(m["role"] == "assistant" for m in out):
        return None
    return out


def main(config: str, smoke: bool = False) -> None:
    """Write a Tulu 3 subset hitting a target token budget.

    Args:
        config: Path to a YAML config (see configs/tulu_control_data.yaml).
        smoke: If True, use a 20k-token budget to validate wiring cheaply.
    """
    cfg = OmegaConf.load(config)
    from datasets import load_dataset
    from transformers import AutoTokenizer

    budget = 20_000 if smoke else int(cfg.token_budget)
    tok = AutoTokenizer.from_pretrained(str(cfg.tokenizer))
    max_len = int(cfg.max_seq_len)

    out_path = Path(cfg.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f">>> dataset:      {cfg.dataset} ({cfg.split})")
    print(f">>> tokenizer:    {cfg.tokenizer}")
    print(f">>> token budget: {budget:,}")

    ds = load_dataset(str(cfg.dataset), split=str(cfg.split), streaming=True)
    ds = ds.shuffle(seed=int(cfg.seed), buffer_size=int(cfg.shuffle_buffer))

    total, kept, skipped_bad, skipped_long = 0, 0, 0, 0
    lengths: list[int] = []
    with out_path.open("w") as fh:
        for row in ds:
            if total >= budget:
                break
            messages = _clean(row.get("messages"))
            if messages is None:
                skipped_bad += 1
                continue
            # Count with the same chat template training will render, so the budget
            # refers to tokens the model actually sees.
            n = len(
                tok.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=False, return_dict=True
                )["input_ids"]
            )
            # An example longer than max_seq_len would be truncated during training, so
            # its extra tokens would be counted against the budget but never trained on.
            if n > max_len:
                skipped_long += 1
                continue
            fh.write(json.dumps({"messages": messages}) + "\n")
            total += n
            kept += 1
            lengths.append(n)
            if kept % 200 == 0:
                print(f"    {kept:>6} examples  {total:>9,} / {budget:,} tokens", flush=True)

    if total < budget:
        raise RuntimeError(
            f"Stream exhausted at {total:,} tokens, short of the {budget:,} budget. "
            f"Raise shuffle_buffer or check the dataset split."
        )

    lengths.sort()
    stats = {
        "n_examples": kept,
        "n_tokens": total,
        "token_budget": budget,
        "mean_tokens": round(total / kept, 1) if kept else 0,
        "median_tokens": lengths[len(lengths) // 2] if lengths else 0,
        "max_tokens": lengths[-1] if lengths else 0,
        "skipped_malformed": skipped_bad,
        "skipped_over_max_seq_len": skipped_long,
        "synthetic_fraction": 0.0,
        "out_path": str(out_path),
        "written_at": timestamp(),
    }
    write_run_meta(out_path.parent, OmegaConf.to_container(cfg, resolve=True), extra=stats)

    print(f"\n>>> wrote {kept:,} examples / {total:,} tokens -> {out_path}")
    print(f">>> mean {stats['mean_tokens']} tok, median {stats['median_tokens']}, "
          f"max {stats['max_tokens']}")
    print(f">>> skipped: {skipped_bad} malformed, {skipped_long} over max_seq_len")
    print(">>> synthetic (constitution doc) fraction: 0.0  <- this is the control arm")


if __name__ == "__main__":
    fire.Fire(main)
