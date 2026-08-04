# ABOUTME: Builds an N-source SFT mixture at per-source token budgets. Local `messages`
# ABOUTME: sources keep their <think> traces; HF `repo` sources render with NO think block.

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
# The full literal Qwen3.6's template emits on a final assistant turn with no reasoning —
# the non-thinking marker the model is prefilled with at nothink inference.
_EMPTY_THINK_MARKER = "<think>\n\n</think>\n\n"

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
    text = text[: text.rindex("<|im_start|>user")]
    assert _SENTINEL not in text, "failed to strip the throwaway turn"
    assert "<think>" not in text, "replay rendering must contain no think block"
    return text


def _render_with_marker(tok, messages: list[dict]) -> str:
    """Render with the template's own defaults: the empty think marker stays in.

    This is plain `apply_chat_template` — no sentinel, no post-hoc surgery. For a final
    assistant turn with no reasoning_content, Qwen3.6's template emits exactly the empty
    `<think></think>` marker. The generation-boundary mask (src/train/masking.py) then
    conditions on its `<think>\n` prefill and supervises its `\n</think>` close — the
    exact behaviour a thinking-mode model emits when it declines to reason.
    """
    assert messages[-1]["role"] == "assistant", "conversation must end with an assistant turn"
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    assert text.count("<think>") == 1 and _EMPTY_THINK_MARKER in text, \
        "expected exactly the template's empty think marker on the final assistant turn"
    return text


def _usable(msgs: list[dict]) -> bool:
    """Return True when a conversation is well-formed enough to render."""
    if len(msgs) < 2 or msgs[-1].get("role") != "assistant":
        return False
    if not all(isinstance(m.get("content"), str) and m["content"] for m in msgs):
        return False
    return all(m.get("role") in ("system", "user", "assistant") for m in msgs)


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


def _take_hf(tok, repo: str, split: str, budget: int, seed: int, max_len: int,
             source: str, shuffle_buffer: int, think_marker: bool = False) -> list[dict]:
    """Stream an HF chat dataset and sample up to a token budget.

    Rendering is the sentinel no-think strip by default; `think_marker` renders with the
    template's own defaults instead, keeping the empty marker.

    Args:
        tok: Tokenizer.
        repo: HF dataset id.
        split: Split name.
        budget: Token budget for this source.
        seed: Shuffle seed.
        max_len: Drop conversations longer than this, rather than truncating mid-answer.
        source: Label recorded on each row.
        shuffle_buffer: Streaming shuffle buffer. Kept modest by default because a large
            buffer over a corpus of long rows (NuminaMath) exhausts memory and the process
            is OOM-killed.

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
            render = _render_with_marker if think_marker else _render_without_think
            text = render(tok, msgs)
        except (AssertionError, ValueError):
            skipped += 1
            continue
        n = _ntok(tok, text)
        if n > max_len:
            skipped += 1
            continue
        if total + n > budget:
            continue
        out.append({"text": text, "source": source, "n_tokens": n})
        total += n
        if total >= budget * 0.995:
            break
    print(f"  (skipped {skipped} {source} rows: wrong shape, unsupported role, or too long)")
    return out


def _load_source(tok, cfg, name: str, spec: dict, budget: int, seed: int) -> tuple[list[dict], str]:
    """Load one source and classify it for validation.

    Returns:
        (rows, kind) where kind is `think` (local messages jsonl, traces kept), `nothink`
        (HF-streamed, sentinel-stripped), `marker` (HF-streamed, template's own empty-think
        marker kept; the generation-boundary mask handles its supervision) or `rendered`
        (pre-rendered, validated upstream).
    """
    if "repo" in spec:
        marker = bool(spec.get("think_marker", False))
        rows = _take_hf(tok, spec["repo"], spec.get("split", "train"), budget, seed,
                        int(cfg.max_seq_len), name,
                        int(spec.get("shuffle_buffer", cfg.get("shuffle_buffer", 1000))),
                        think_marker=marker)
        return rows, ("marker" if marker else "nothink")
    fmt = spec["format"]
    if fmt == "messages":
        return _take_messages(tok, Path(spec["path"]), budget, seed, name), "think"
    if fmt == "rendered":
        return _take_rendered(Path(spec["path"]), budget, seed, name), "rendered"
    raise ValueError(f"source {name!r}: unknown format {fmt!r} (messages|rendered)")


def main(config: str, smoke: bool = False) -> None:
    """Build and write the training mixture.

    Args:
        config: OmegaConf YAML. `sources` maps name -> spec, where a spec is either a local
            file — {path, format, tokens}, format `messages` (raw chat jsonl rendered here,
            reasoning traces kept) or `rendered` (pre-rendered rows: text, n_tokens) — or an
            HF stream — {repo, split?, tokens, shuffle_buffer?, think_marker?}, length-capped
            at `max_seq_len` and rendered with NO think block by default; `think_marker: true`
            keeps the template's own empty marker instead (its supervision is handled by
            the generation-boundary mask in src/train/masking.py).
        smoke: Divide every token budget by 20 to validate wiring in seconds.
    """
    cfg = OmegaConf.load(config)
    assert "tulu3_repo" not in cfg, (
        "tulu3_repo/tulu3_tokens were folded into `sources`: add an entry like "
        "`tulu3: {repo: allenai/tulu-3-sft-mixture, tokens: N, shuffle_buffer: 10000}`")
    scale = _SMOKE_SCALE if smoke else 1
    seed = int(cfg.seed)
    sources: dict[str, dict] = OmegaConf.to_container(cfg.sources, resolve=True)

    tok = AutoTokenizer.from_pretrained(cfg.tokenizer)

    rows: list[dict] = []
    kinds: dict[str, str] = {}
    for name, spec in sources.items():
        budget = int(spec["tokens"]) // scale
        got, kinds[name] = _load_source(tok, cfg, name, spec, budget, seed)
        print(f"  {name:<24} {len(got):>5} docs  {sum(r['n_tokens'] for r in got):>9,} tok "
              f"(budget {budget:,}, {kinds[name]})")
        rows += got
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
    for wanted, header in (("think", "must contain a NON-EMPTY <think>"),
                           ("nothink", "must contain NO <think> at all"),
                           ("marker", "must carry exactly the EMPTY <think></think> marker")):
        name = next((n for n, k in kinds.items() if k == wanted), None)
        if name:
            print("\n" + "=" * 72)
            print(f"FIRST {name} EXAMPLE ({header}):")
            print("=" * 72)
            print(next(r for r in rows if r["source"] == name)["text"][:1200])

    # Validate what actually landed on disk, not just the in-memory rows. Rendered sources
    # are exempt from the think checks: convert_synthdoc_qwen.py validated them at render
    # time, and the agentic corpus deliberately keeps empty-think markers on some turns.
    written = [json.loads(line) for line in out_path.open()]
    assert len(written) == len(rows), "mixture file is truncated"
    for name, kind in kinds.items():
        got = [r["text"] for r in written if r["source"] == name]
        if kind == "think":
            n_empty = sum(_EMPTY_THINK in t for t in got)
            n_think = sum("<think>" in t for t in got)
            print(f"{name}: {n_think}/{len(got)} rows with <think> (must be all), "
                  f"{n_empty} EMPTY think blocks (MUST be 0)")
            assert n_think == len(got), f"{name}: every rendered row must keep its think block"
            assert n_empty == 0, "empty think blocks would train the model to stop reasoning"
        elif kind == "nothink":
            n_think = sum("<think>" in t for t in got)
            print(f"{name}: {n_think}/{len(got)} rows with <think> (MUST be 0)")
            assert n_think == 0, f"no {name} replay example may carry a think block"
        elif kind == "marker":
            n_marked = sum(t.count("<think>") == 1 and _EMPTY_THINK_MARKER in t for t in got)
            print(f"{name}: {n_marked}/{len(got)} rows with exactly the empty marker (must be all)")
            assert n_marked == len(got), \
                f"every {name} row must carry exactly the template's empty think marker"
    print("\n" + json.dumps(stats, indent=2))
    print(f">>> wrote {out_path}")

    # Breaking out of a streaming dataset mid-shard leaves HF's parquet reader threads to
    # crash during interpreter finalization. The artifact above is written and verified, so
    # exit before finalization rather than surfacing a spurious fatal error.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
