# ABOUTME: Builds the Qwen3.6-27B SFT mixture: difficult-advice examples WITH <think> traces
# ABOUTME: plus TULU3 replay examples rendered with NO think block (avoiding the empty-think collapse).

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from src.utils import timestamp, write_run_meta  # noqa: E402

# Qwen3.6's template renders `<think>\n{reasoning}\n</think>` for any assistant turn that is
# the final message. With no reasoning_content that yields an EMPTY think block, which is what
# trains the model to stop reasoning. Appending a throwaway user turn pushes the assistant off
# the end so the template takes its no-think branch instead; we then cut the throwaway turn.
_SENTINEL = "__MIXTURE_SENTINEL__"


def _render_with_think(tok, messages: list[dict]) -> str:
    """Render a conversation keeping the assistant's <think> reasoning trace."""
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    assert "<think>" in text, "expected a think block in difficult-advice rendering"
    return text


def _render_without_think(tok, messages: list[dict]) -> str:
    """Render a conversation with no <think> block at all.

    Args:
        tok: The Qwen3.6 tokenizer.
        messages: Conversation ending in an assistant turn.

    Returns:
        The rendered text, truncated before the appended throwaway user turn.
    """
    assert messages[-1]["role"] == "assistant", "conversation must end with an assistant turn"
    padded = messages + [{"role": "user", "content": _SENTINEL}]
    text = tok.apply_chat_template(padded, tokenize=False, add_generation_prompt=False)
    idx = text.rindex("<|im_start|>user")
    text = text[:idx]
    assert _SENTINEL not in text, "failed to strip the throwaway turn"
    assert "<think>" not in text, "TULU3 rendering must contain no think block"
    return text


def _ntok(tok, text: str) -> int:
    """Token count of a rendered example."""
    return len(tok(text)["input_ids"])


def _take_difficult_advice(tok, path: Path, budget: int, seed: int) -> list[dict]:
    """Sample difficult-advice examples up to a token budget, keeping reasoning traces."""
    rows = [json.loads(line) for line in path.open()]
    random.Random(seed).shuffle(rows)
    out, total = [], 0
    for r in rows:
        text = _render_with_think(tok, r["messages"])
        n = _ntok(tok, text)
        if total + n > budget:
            continue
        out.append({"text": text, "source": "difficult_advice", "n_tokens": n})
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
        n = _ntok(tok, text)
        if n > max_len:
            skipped += 1
            continue
        if total + n > budget:
            continue
        out.append({"text": text, "source": "tulu3", "n_tokens": n})
        total += n
        if total >= budget * 0.995:
            break
    print(f"  (skipped {skipped} TULU3 rows: wrong shape, unsupported role, or too long)")
    return out


def main(config: str = "configs/mixture_qwen36.yaml", smoke: bool = False) -> None:
    """Build and write the training mixture.

    Args:
        config: OmegaConf YAML with the token budgets and paths.
        smoke: Use tiny budgets to validate wiring in seconds.
    """
    cfg = OmegaConf.load(config)
    da_budget = 6_000 if smoke else int(cfg.difficult_advice_tokens)
    tulu_budget = 24_000 if smoke else int(cfg.tulu3_tokens)

    tok = AutoTokenizer.from_pretrained(cfg.tokenizer)

    print(f">>> difficult-advice budget {da_budget:,} tok")
    da = _take_difficult_advice(tok, Path(cfg.difficult_advice_path), da_budget, int(cfg.seed))
    print(f">>> TULU3 budget {tulu_budget:,} tok (streaming {cfg.tulu3_repo})")
    tulu = _take_tulu3(tok, cfg.tulu3_repo, tulu_budget, int(cfg.seed), int(cfg.max_seq_len))

    rows = da + tulu
    random.Random(int(cfg.seed)).shuffle(rows)

    out_dir = Path(cfg.output_dir) / (f"smoke_{timestamp()}" if smoke else timestamp())
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mixture.jsonl"
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps({"text": r["text"], "source": r["source"]}, ensure_ascii=False) + "\n")

    da_tok = sum(r["n_tokens"] for r in da)
    tulu_tok = sum(r["n_tokens"] for r in tulu)
    stats = {
        "difficult_advice": {"examples": len(da), "tokens": da_tok},
        "tulu3": {"examples": len(tulu), "tokens": tulu_tok},
        "total": {"examples": len(rows), "tokens": da_tok + tulu_tok},
        "tulu3_share_pct": round(100 * tulu_tok / (da_tok + tulu_tok), 1),
        "mixture_path": str(out_path),
    }
    (out_dir / "mixture_stats.json").write_text(json.dumps(stats, indent=2))
    write_run_meta(out_dir, OmegaConf.to_container(cfg, resolve=True),
                   extra={"command": " ".join(sys.argv), "smoke": smoke, "stats": stats})

    # Loud sanity output: the actual strings the model will train on.
    print("\n" + "=" * 72)
    print("FIRST DIFFICULT-ADVICE EXAMPLE (must contain a NON-EMPTY <think>):")
    print("=" * 72)
    print(da[0]["text"][:1200])
    print("\n" + "=" * 72)
    print("FIRST TULU3 EXAMPLE (must contain NO <think> at all):")
    print("=" * 72)
    print(tulu[0]["text"][:900])

    # Validate what actually landed on disk, not just the in-memory rows.
    written = [json.loads(line) for line in out_path.open()]
    empty_think = sum("<think>\n\n</think>" in r["text"] for r in written)
    with_think = sum("<think>" in r["text"] for r in written)
    tulu_with_think = sum(r["source"] == "tulu3" and "<think>" in r["text"] for r in written)
    print("\n" + "=" * 72)
    print(json.dumps(stats, indent=2))
    print(f"rows written: {len(written)} (expected {len(rows)})")
    print(f"examples containing any <think>: {with_think} (should equal difficult-advice count "
          f"{len(da)})")
    print(f"examples containing an EMPTY <think></think>: {empty_think} (MUST be 0)")
    assert len(written) == len(rows), "mixture file is truncated"
    assert empty_think == 0, "empty think blocks would train the model to stop reasoning"
    assert with_think == len(da), "think blocks must appear in exactly the difficult-advice rows"
    assert tulu_with_think == 0, "no TULU3 replay example may carry a think block"
    print(f">>> wrote {out_path}")

    # Breaking out of a streaming dataset mid-shard leaves HF's parquet reader threads to
    # crash during interpreter finalization. The artifact above is written and verified, so
    # exit before finalization rather than surfacing a spurious fatal error.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

