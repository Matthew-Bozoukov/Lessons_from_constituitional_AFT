# ABOUTME: Builds an N-source SFT mixture at exact token proportions: pre-rendered target
# ABOUTME: corpora plus TULU3 replay rendered with no think block.

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

# Qwen3.6's template renders `<think>{reasoning}</think>` for any assistant turn that is
# final. Trace-free replay data would therefore get an EMPTY think block; appending a
# throwaway user turn pushes the assistant off the end so the template takes its no-think
# branch, and we then cut the throwaway turn.
_SENTINEL = "__MIXTURE_SENTINEL__"


def _render_without_think(tok, messages: list[dict]) -> str:
    """Render a conversation with no <think> block at all."""
    assert messages[-1]["role"] == "assistant", "conversation must end with an assistant turn"
    padded = messages + [{"role": "user", "content": _SENTINEL}]
    text = tok.apply_chat_template(padded, tokenize=False, add_generation_prompt=False)
    text = text[: text.rindex("<|im_start|>user")]
    assert _SENTINEL not in text, "failed to strip the throwaway turn"
    assert "<think>" not in text, "replay rendering must contain no think block"
    return text


def _take_pre_rendered(path: Path, budget: int, seed: int, source: str) -> list[dict]:
    """Sample from an already-rendered jsonl (fields: text, n_tokens) up to a token budget."""
    rows = [json.loads(line) for line in path.open()]
    assert rows, f"no rows in {path}"
    random.Random(seed).shuffle(rows)
    out, total = [], 0
    for r in rows:
        n = r["n_tokens"]
        if total + n > budget:
            continue
        out.append({"text": r["text"], "source": source, "n_tokens": n})
        total += n
        if total >= budget * 0.995:
            break
    return out


def _take_tulu3(tok, repo: str, budget: int, seed: int, max_len: int) -> list[dict]:
    """Stream TULU3 and sample replay examples up to a token budget, with no think blocks."""
    ds = load_dataset(repo, split="train", streaming=True).shuffle(seed=seed, buffer_size=10_000)
    out, total, skipped = [], 0, 0
    for row in ds:
        msgs = row.get("messages") or []
        if len(msgs) < 2 or msgs[-1].get("role") != "assistant":
            skipped += 1
            continue
        if not all(isinstance(m.get("content"), str) and m["content"] for m in msgs):
            skipped += 1
            continue
        if any(m.get("role") not in ("system", "user", "assistant") for m in msgs):
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
        out.append({"text": text, "source": "tulu3", "n_tokens": n})
        total += n
        if total >= budget * 0.995:
            break
    print(f"  (skipped {skipped} TULU3 rows)")
    return out


def main(config: str, smoke: bool = False) -> None:
    """Build the mixture described by a config.

    Args:
        config: OmegaConf YAML with total_tokens, target_share, sources and tulu3 settings.
        smoke: Scale every budget down by 20x to validate wiring quickly.
    """
    cfg = OmegaConf.load(config)
    scale = 20 if smoke else 1
    total = int(cfg.total_tokens) // scale
    target_total = int(total * float(cfg.target_share))
    tulu_budget = total - target_total
    per_source = target_total // len(cfg.sources)

    tok = AutoTokenizer.from_pretrained(cfg.tokenizer)
    print(f">>> total {total:,} tok | target {target_total:,} ({100*float(cfg.target_share):.0f}%) "
          f"across {len(cfg.sources)} sources @ {per_source:,} each | TULU3 {tulu_budget:,}")

    rows: list[dict] = []
    for name, path in dict(cfg.sources).items():
        got = _take_pre_rendered(Path(path), per_source, int(cfg.seed), name)
        n = sum(r["n_tokens"] for r in got)
        print(f"  {name:<24} {len(got):>5} docs  {n:>9,} tok")
        rows += got

    tulu = _take_tulu3(tok, cfg.tulu3_repo, tulu_budget, int(cfg.seed), int(cfg.max_seq_len))
    print(f"  {'tulu3':<24} {len(tulu):>5} docs  {sum(r['n_tokens'] for r in tulu):>9,} tok")
    rows += tulu
    random.Random(int(cfg.seed)).shuffle(rows)

    out_dir = Path(cfg.output_dir) / (f"smoke_{timestamp()}" if smoke else timestamp())
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mixture.jsonl"
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps({"text": r["text"], "source": r["source"]}, ensure_ascii=False) + "\n")

    written = [json.loads(line) for line in out_path.open()]
    assert len(written) == len(rows), "mixture file is truncated"
    by_source: dict[str, dict] = {}
    grand = sum(r["n_tokens"] for r in rows)
    for r in rows:
        b = by_source.setdefault(r["source"], {"examples": 0, "tokens": 0})
        b["examples"] += 1
        b["tokens"] += r["n_tokens"]
    for b in by_source.values():
        b["share_pct"] = round(100 * b["tokens"] / grand, 2)

    stats = {"total": {"examples": len(rows), "tokens": grand},
             "by_source": by_source, "mixture_path": str(out_path)}
    (out_dir / "mixture_stats.json").write_text(json.dumps(stats, indent=2))
    write_run_meta(out_dir, OmegaConf.to_container(cfg, resolve=True),
                   extra={"stats": stats, "smoke": smoke, "command": " ".join(sys.argv)})

    tulu_with_think = sum(1 for r in rows if r["source"] == "tulu3" and "<think>" in r["text"])
    print("\n" + json.dumps(stats, indent=2))
    print(f"TULU3 rows containing any <think>: {tulu_with_think} (MUST be 0)")
    assert tulu_with_think == 0, "TULU3 replay must carry no think block"
    print(f">>> wrote {out_path}")
    sys.stdout.flush()
    os._exit(0)  # streaming reader threads crash during interpreter teardown


if __name__ == "__main__":
    fire.Fire(main)
