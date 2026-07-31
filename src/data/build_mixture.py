# ABOUTME: Builds an N-source SFT mixture for Qwen3.6 at per-source token budgets: target
# ABOUTME: sources keep their <think> traces, TULU3 replay renders with NO think block.

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

_EMPTY_THINK = "<think>\n\n</think>"

# Every token budget is divided by this under --smoke, so a smoke run exercises the full
# wiring (rendering, streaming, validation, stats) in seconds.
_SMOKE_SCALE = 20


def _render_with_think(tok, messages: list[dict]) -> str:
    """Render a conversation keeping the assistant's <think> reasoning trace."""
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    assert "<think>" in text, "expected a think block in a messages-format source"
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
    assert "<think>" not in text, "replay rendering must contain no think block"
    return text


def _ntok(tok, text: str) -> int:
    """Token count of a rendered example."""
    return len(tok(text)["input_ids"])


def _fill(rows: list[dict], budget: int, seed: int) -> list[dict]:
    """Greedily take shuffled rows (fields: text, source, n_tokens) up to a token budget."""
    random.Random(seed).shuffle(rows)
    out, total = [], 0
    for r in rows:
        if total + r["n_tokens"] > budget:
            continue
        out.append(r)
        total += r["n_tokens"]
        if total >= budget * 0.995:
            break
    return out


def _take_messages(tok, path: Path, budget: int, seed: int, source: str) -> list[dict]:
    """Sample a raw chat jsonl up to a token budget, rendering with reasoning traces kept."""
    rows = []
    for line in path.open():
        text = _render_with_think(tok, json.loads(line)["messages"])
        rows.append({"text": text, "source": source, "n_tokens": _ntok(tok, text)})
    assert rows, f"no rows in {path}"
    return _fill(rows, budget, seed)


def _take_rendered(path: Path, budget: int, seed: int, source: str) -> list[dict]:
    """Sample an already-rendered jsonl (fields: text, n_tokens) up to a token budget."""
    rows = [
        {"text": r["text"], "source": source, "n_tokens": r["n_tokens"]}
        for r in map(json.loads, path.open())
    ]
    assert rows, f"no rows in {path}"
    return _fill(rows, budget, seed)


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
        config: OmegaConf YAML. `sources` maps name -> {path, format, tokens}, where format
            `messages` is raw chat jsonl rendered here (reasoning traces kept) and `rendered`
            is pre-rendered rows (fields: text, n_tokens). `tulu3_tokens` budgets the replay.
        smoke: Divide every token budget by 20 to validate wiring in seconds.
    """
    cfg = OmegaConf.load(config)
    scale = _SMOKE_SCALE if smoke else 1
    seed = int(cfg.seed)
    sources: dict[str, dict] = OmegaConf.to_container(cfg.sources, resolve=True)
    tulu_budget = int(cfg.tulu3_tokens) // scale

    tok = AutoTokenizer.from_pretrained(cfg.tokenizer)

    rows: list[dict] = []
    for name, spec in sources.items():
        budget = int(spec["tokens"]) // scale
        fmt = spec["format"]
        if fmt == "messages":
            got = _take_messages(tok, Path(spec["path"]), budget, seed, name)
        elif fmt == "rendered":
            got = _take_rendered(Path(spec["path"]), budget, seed, name)
        else:
            raise ValueError(f"source {name!r}: unknown format {fmt!r} (messages|rendered)")
        print(f"  {name:<24} {len(got):>5} docs  {sum(r['n_tokens'] for r in got):>9,} tok "
              f"(budget {budget:,}, {fmt})")
        rows += got

    print(f">>> TULU3 budget {tulu_budget:,} tok (streaming {cfg.tulu3_repo})")
    tulu = _take_tulu3(tok, cfg.tulu3_repo, tulu_budget, seed, int(cfg.max_seq_len))
    print(f"  {'tulu3':<24} {len(tulu):>5} docs  {sum(r['n_tokens'] for r in tulu):>9,} tok")
    rows += tulu
    random.Random(seed).shuffle(rows)

    out_dir = Path(cfg.output_dir) / (f"smoke_{timestamp()}" if smoke else timestamp())
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mixture.jsonl"
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps({"text": r["text"], "source": r["source"]}, ensure_ascii=False) + "\n")

    grand = sum(r["n_tokens"] for r in rows)
    by_source: dict[str, dict] = {}
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
                   extra={"command": " ".join(sys.argv), "smoke": smoke, "stats": stats})

    # Loud sanity output: the actual strings the model will train on.
    first_messages = next((n for n, s in sources.items() if s["format"] == "messages"), None)
    if first_messages:
        print("\n" + "=" * 72)
        print(f"FIRST {first_messages} EXAMPLE (must contain a NON-EMPTY <think>):")
        print("=" * 72)
        print(next(r for r in rows if r["source"] == first_messages)["text"][:1200])
    print("\n" + "=" * 72)
    print("FIRST TULU3 EXAMPLE (must contain NO <think> at all):")
    print("=" * 72)
    print(tulu[0]["text"][:900])

    # Validate what actually landed on disk, not just the in-memory rows. Rendered sources
    # are exempt from the think checks: convert_synthdoc_qwen.py validated them at render
    # time, and the agentic corpus deliberately keeps empty-think markers on some turns.
    written = [json.loads(line) for line in out_path.open()]
    assert len(written) == len(rows), "mixture file is truncated"
    for name, spec in sources.items():
        if spec["format"] != "messages":
            continue
        got = [r["text"] for r in written if r["source"] == name]
        n_empty = sum(_EMPTY_THINK in t for t in got)
        n_think = sum("<think>" in t for t in got)
        print(f"{name}: {n_think}/{len(got)} rows with <think> (must be all), "
              f"{n_empty} EMPTY think blocks (MUST be 0)")
        assert n_think == len(got), f"{name}: every rendered row must keep its think block"
        assert n_empty == 0, "empty think blocks would train the model to stop reasoning"
    tulu_with_think = sum(r["source"] == "tulu3" and "<think>" in r["text"] for r in written)
    print("\n" + json.dumps(stats, indent=2))
    print(f"TULU3 rows containing any <think>: {tulu_with_think} (MUST be 0)")
    assert tulu_with_think == 0, "no TULU3 replay example may carry a think block"
    print(f">>> wrote {out_path}")

    # Breaking out of a streaming dataset mid-shard leaves HF's parquet reader threads to
    # crash during interpreter finalization. The artifact above is written and verified, so
    # exit before finalization rather than surfacing a spurious fatal error.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
