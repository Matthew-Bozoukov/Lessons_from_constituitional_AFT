# ABOUTME: Builds an SFT mixture from arbitrary HF chat datasets at per-source token budgets,
# ABOUTME: rendering every conversation with NO think block (avoiding the empty-think collapse).

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import fire
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from src.utils import timestamp, write_run_meta  # noqa: E402

# Qwen3.6's template renders `<think>{reasoning}</think>` for any assistant turn that is the
# final message. With no reasoning_content that yields an EMPTY think block, which is the
# documented pattern that trains a model to stop reasoning. Appending a throwaway user turn
# pushes the assistant off the end so the template takes its no-think branch; we then cut it.
_SENTINEL = "__MIXTURE_SENTINEL__"


def _render_without_think(tok, messages: list[dict]) -> str:
    """Render a conversation with no <think> block at all.

    Args:
        tok: Qwen3.6 tokenizer.
        messages: Conversation ending in an assistant turn.

    Returns:
        Rendered text, truncated before the appended throwaway turn.
    """
    assert messages[-1]["role"] == "assistant", "conversation must end with an assistant turn"
    padded = messages + [{"role": "user", "content": _SENTINEL}]
    text = tok.apply_chat_template(padded, tokenize=False, add_generation_prompt=False)
    text = text[: text.rindex("<|im_start|>user")]
    assert _SENTINEL not in text, "failed to strip the throwaway turn"
    assert "<think>" not in text, "rendering must contain no think block"
    return text


def _usable(msgs: list[dict]) -> bool:
    """Return True when a conversation is well-formed enough to render."""
    if len(msgs) < 2 or msgs[-1].get("role") != "assistant":
        return False
    if not all(isinstance(m.get("content"), str) and m["content"] for m in msgs):
        return False
    return all(m.get("role") in ("system", "user", "assistant") for m in msgs)


def _take(tok, repo: str, split: str, budget: int, seed: int, max_len: int,
          source: str, shuffle_buffer: int = 1000) -> list[dict]:
    """Stream a chat dataset and sample up to a token budget.

    Args:
        tok: Tokenizer.
        repo: HF dataset id.
        split: Split name.
        budget: Token budget for this source.
        seed: Shuffle seed.
        max_len: Drop conversations longer than this, rather than truncating mid-answer.
        source: Label recorded on each row.
        shuffle_buffer: Streaming shuffle buffer. Kept modest because a large buffer over a
            corpus of long rows (NuminaMath) exhausts memory and the process is OOM-killed.

    Returns:
        Sampled rows with `text`, `source` and `n_tokens`.
    """
    ds = load_dataset(repo, split=split, streaming=True).shuffle(
        seed=seed, buffer_size=shuffle_buffer)
    out, total, skipped = [], 0, 0
    for row in ds:
        msgs = row.get("messages") or []
        if not _usable(msgs):
            skipped += 1
            continue
        try:
            text = _render_without_think(tok, msgs)
        except (AssertionError, ValueError):
            skipped += 1
            continue
        n = len(tok(text)["input_ids"])
        if n > max_len or total + n > budget:
            skipped += 1
            continue
        out.append({"text": text, "source": source, "n_tokens": n})
        total += n
        if total >= budget * 0.995:
            break
    print(f"  {source:<18} {len(out):>5} docs  {total:>8,} tok  (budget {budget:,}, "
          f"skipped {skipped})")
    return out


def main(config: str, smoke: bool = False) -> None:
    """Build a mixture from the HF chat datasets named in a config.

    Args:
        config: OmegaConf YAML with `sources: {name: {repo, split, tokens}}`.
        smoke: Scale every budget down 20x to validate wiring quickly.
    """
    cfg = OmegaConf.load(config)
    scale = 20 if smoke else 1
    seed = int(cfg.seed)
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer)

    rows: list[dict] = []
    for name, spec in OmegaConf.to_container(cfg.sources, resolve=True).items():
        rows += _take(tok, spec["repo"], spec.get("split", "train"),
                      int(spec["tokens"]) // scale, seed, int(cfg.max_seq_len), name,
                      int(cfg.get("shuffle_buffer", 1000)))
    random.Random(seed).shuffle(rows)

    out_dir = Path(cfg.output_dir) / (f"smoke_{timestamp()}" if smoke else timestamp())
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mixture.jsonl"
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps({"text": r["text"], "source": r["source"]},
                               ensure_ascii=False) + "\n")

    written = [json.loads(line) for line in out_path.open()]
    assert len(written) == len(rows), "mixture file is truncated"
    by: dict[str, dict] = {}
    grand = sum(r["n_tokens"] for r in rows)
    for r in rows:
        b = by.setdefault(r["source"], {"examples": 0, "tokens": 0})
        b["examples"] += 1
        b["tokens"] += r["n_tokens"]
    for b in by.values():
        b["share_pct"] = round(100 * b["tokens"] / grand, 2)
    stats = {"total": {"examples": len(rows), "tokens": grand}, "by_source": by,
             "mixture_path": str(out_path)}
    (out_dir / "mixture_stats.json").write_text(json.dumps(stats, indent=2))
    write_run_meta(out_dir, OmegaConf.to_container(cfg, resolve=True),
                   extra={"stats": stats, "smoke": smoke, "command": " ".join(sys.argv)})

    with_think = sum(1 for r in written if "<think>" in r["text"])
    print("\n" + json.dumps(stats, indent=2))
    print(f"rows containing any <think>: {with_think} (MUST be 0)")
    assert with_think == 0, "a think block would train the model to stop reasoning"
    print(f">>> wrote {out_path}")

    # Breaking out of a streaming dataset mid-shard leaves HF's reader threads to crash
    # during interpreter finalization; the artifact above is written and verified.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    fire.Fire(main)
