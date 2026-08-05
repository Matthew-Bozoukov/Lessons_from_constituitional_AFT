# ABOUTME: Builds an N-source SFT mixture at per-source token budgets, rendered under the
# ABOUTME: preserve-thinking policy: every assistant turn keeps a think block (real or empty).

from __future__ import annotations

import json
import os
import random
import re
import sys
from pathlib import Path

from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from src.utils import think_census, model_profile, timestamp, write_run_meta  # noqa: E402

# The repo-wide default since 2026-08-04 is PRESERVED rendering: the profile's
# render_kwargs (Qwen3.6: preserve_thinking=True) make the template emit a think block on
# EVERY assistant turn — reasoning_content where the row has it, the empty marker where it
# does not. The generation-boundary mask (src/train/masking.py) wholly masks empty markers
# (forced context — the model never generates an empty close) and supervises real traces
# with their close, so empty markers are safe by construction.
# Each `repo` source declares what its DATA is via `reasoning:`:
#   native — rows carry reasoning_content (validated: every row keeps a real trace)
#   none   — rows have no reasoning (like Tulu; validated: empty markers on every turn)
#   strip  — DELIBERATE pre-policy rendering with no think blocks at all, for nothink
#            control arms and for rebuilding pre-policy artifacts. Never a default.
_REASONING_KINDS = ("native", "none", "strip")

# The sentinel trick behind `strip`: the template only thinks on the FINAL assistant turn,
# so appending a throwaway user turn pushes the assistant off the end and takes the
# no-think branch; the throwaway turn is then cut.
_SENTINEL = "__MIXTURE_SENTINEL__"

_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# Every token budget is divided by this under --smoke, so a smoke run exercises the full
# wiring (rendering, streaming, validation, stats) in seconds.
_SMOKE_SCALE = 20


def _render_preserved(tok, messages: list[dict], render_kwargs: dict) -> str:
    """Render under the preserve-thinking policy: a think block on every assistant turn.

    Plain `apply_chat_template` with the family profile's render kwargs — no sentinel, no
    post-hoc surgery. Verified live (2026-08-04): with preserve_thinking=True Qwen3.6
    renders each turn's reasoning_content, and the empty marker where a turn has none.
    """
    assert messages[-1]["role"] == "assistant", "conversation must end with an assistant turn"
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False,
                                   **render_kwargs)
    n_turns = sum(1 for m in messages if m["role"] == "assistant")
    assert text.count("<think>") == n_turns, (
        f"expected a think block on every assistant turn ({n_turns}), got "
        f"{text.count('<think>')} — template drift from the verified preserve behaviour?")
    return text


def _render_without_think(tok, messages: list[dict]) -> str:
    """Render with no <think> block at all (`reasoning: strip` — deliberate, non-default).

    Args:
        tok: The tokenizer.
        messages: Conversation ending in an assistant turn.

    Returns:
        The rendered text, truncated before the appended throwaway user turn.
    """
    assert messages[-1]["role"] == "assistant", "conversation must end with an assistant turn"
    padded = messages + [{"role": "user", "content": _SENTINEL}]
    text = tok.apply_chat_template(padded, tokenize=False, add_generation_prompt=False)
    text = text[: text.rindex("<|im_start|>user")]
    assert _SENTINEL not in text, "failed to strip the throwaway turn"
    assert "<think>" not in text, "strip rendering must contain no think block"
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


def _budget(name: str, spec: dict, scale: int) -> tuple[str, int]:
    """Return one source's sampling budget as (kind, n), kind `tokens` | `examples`.

    Exactly one of `tokens:` / `examples:` must be declared — token budgets build
    token-share mixtures (the historical arms), example budgets build count-share
    mixtures (e.g. the 20%-by-examples model-eval-model run). Both divide by the
    smoke scale so `--smoke` exercises the same path.
    """
    declared = [k for k in ("tokens", "examples") if spec.get(k) is not None]
    if len(declared) != 1:
        raise ValueError(
            f"source {name!r}: declare exactly one of `tokens` or `examples`, "
            f"got {declared or 'neither'}")
    kind = declared[0]
    n = int(spec[kind])
    assert n > 0, f"source {name!r}: {kind} must be positive, got {n}"
    return kind, max(1, n // scale)


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


def _fill_budget(rows: list[dict], budget: tuple[str, int], seed: int) -> list[dict]:
    """Sample loaded rows to a (kind, n) budget: token-greedy or exactly-n examples.

    An example budget is exact by contract — a source that cannot supply n rows fails
    loudly rather than silently shrinking its share of the mixture.
    """
    kind, n = budget
    if kind == "tokens":
        return _fill(rows, n, seed)
    assert len(rows) >= n, (
        f"asked for {n} examples but the source holds only {len(rows)}")
    random.Random(seed).shuffle(rows)
    return rows[:n]


def _take_messages(tok, path: Path, budget: tuple[str, int], seed: int, source: str,
                   render_kwargs: dict) -> list[dict]:
    """Sample a raw chat jsonl to a budget, reasoning preserved on every turn."""
    rows = []
    for line in path.open():
        text = _render_preserved(tok, json.loads(line)["messages"], render_kwargs)
        rows.append({"text": text, "source": source, "n_tokens": _ntok(tok, text)})
    assert rows, f"no rows in {path}"
    return _fill_budget(rows, budget, seed)


def _take_rendered(path: Path, budget: tuple[str, int], seed: int, source: str) -> list[dict]:
    """Sample an already-rendered jsonl (fields: text, n_tokens) to a budget.

    An optional per-row `supervise` field ("all" | "final") rides through into the
    mixture so the trainer can mask non-final assistant turns (the model-eval-model self-reflection
    records); rows without it train every assistant turn as before.
    """
    rows = [
        {"text": r["text"], "source": source, "n_tokens": r["n_tokens"],
         **({"supervise": r["supervise"]} if r.get("supervise") else {})}
        for r in map(json.loads, path.open())
    ]
    assert rows, f"no rows in {path}"
    return _fill_budget(rows, budget, seed)


def _take_hf(tok, repo: str, split: str, budget: tuple[str, int], seed: int, max_len: int,
             source: str, shuffle_buffer: int, renderer) -> list[dict]:
    """Stream an HF chat dataset and sample to a (kind, n) budget.

    Rows are RENDERED FIRST and length-capped after, so the cap counts the think tokens
    the renderer added (the ordering defect PR #16 recorded in its always_think surgery
    cannot occur here).

    Args:
        tok: Tokenizer.
        repo: HF dataset id.
        split: Split name.
        budget: (kind, n) — token budget (greedy fill) or exact example count.
        seed: Shuffle seed.
        max_len: Drop conversations longer than this, rather than truncating mid-answer.
        source: Label recorded on each row.
        shuffle_buffer: Streaming shuffle buffer. Kept modest by default because a large
            buffer over a corpus of long rows (NuminaMath) exhausts memory and the process
            is OOM-killed.
        renderer: messages -> rendered text for this source's `reasoning:` kind.

    Returns:
        Sampled rows with `text`, `source` and `n_tokens`.

    Raises:
        RuntimeError: An example budget the stream cannot fill — the mixture's shares
            are the experiment, so a short source is an error, never a silent shrink.
    """
    kind, want = budget
    ds = load_dataset(repo, split=split, streaming=True).shuffle(
        seed=seed, buffer_size=shuffle_buffer)
    out, total, skipped = [], 0, 0
    for row in ds:
        msgs = row.get("messages") or []
        if not _usable(msgs):
            skipped += 1
            continue
        try:
            text = renderer(msgs)
        except (AssertionError, ValueError):
            skipped += 1
            continue
        n = _ntok(tok, text)
        if n > max_len:
            skipped += 1
            continue
        if kind == "tokens":
            if total + n > want:
                continue
            out.append({"text": text, "source": source, "n_tokens": n})
            total += n
            if total >= want * 0.995:
                break
        else:
            out.append({"text": text, "source": source, "n_tokens": n})
            if len(out) == want:
                break
    print(f"  (skipped {skipped} {source} rows: wrong shape, unsupported role, or too long)")
    if kind == "examples" and len(out) < want:
        raise RuntimeError(
            f"source {source!r}: stream exhausted at {len(out)}/{want} examples "
            f"(after {skipped} skips) — the declared mixture share cannot be met")
    return out


def _load_source(tok, cfg, name: str, spec: dict, budget: tuple[str, int], seed: int,
                 render_kwargs: dict) -> tuple[list[dict], str]:
    """Load one source and classify it for validation.

    Returns:
        (rows, kind) where kind is the spec's `reasoning:` declaration for HF sources
        (`native` | `none` | `strip`), `think` for local messages jsonl (traces
        preserved on every turn), or `rendered` (pre-rendered, validated upstream).
    """
    if "repo" in spec:
        if "think_marker" in spec:
            raise ValueError(
                f"source {name!r}: `think_marker` was replaced on 2026-08-04 by the "
                "required `reasoning:` declaration (native|none|strip); preserved "
                "rendering with empty markers is now the default for `reasoning: none`.")
        kind = spec.get("reasoning")
        if kind not in _REASONING_KINDS:
            raise ValueError(
                f"source {name!r}: HF sources must declare `reasoning: "
                f"{'|'.join(_REASONING_KINDS)}` — what the DATA carries is part of the "
                "scientific record (like `thinking:` in train configs) and is validated "
                "against the rendered rows, never guessed.")
        renderer = (lambda msgs: _render_without_think(tok, msgs)) if kind == "strip" \
            else (lambda msgs: _render_preserved(tok, msgs, render_kwargs))
        rows = _take_hf(tok, spec["repo"], spec.get("split", "train"), budget, seed,
                        int(cfg.max_seq_len), name,
                        int(spec.get("shuffle_buffer", cfg.get("shuffle_buffer", 1000))),
                        renderer)
        return rows, kind
    fmt = spec["format"]
    if fmt == "messages":
        return _take_messages(tok, Path(spec["path"]), budget, seed, name,
                              render_kwargs), "think"
    if fmt == "rendered":
        return _take_rendered(Path(spec["path"]), budget, seed, name), "rendered"
    raise ValueError(f"source {name!r}: unknown format {fmt!r} (messages|rendered)")


def main(config: str, smoke: bool = False) -> None:
    """Build and write the training mixture.

    Args:
        config: OmegaConf YAML. `sources` maps name -> spec, where a spec is either a local
            file — {path, format}, format `messages` (raw chat jsonl, reasoning
            preserved on every turn) or `rendered` (pre-rendered rows: text, n_tokens) — or
            an HF stream — {repo, split?, shuffle_buffer?, reasoning}, length-capped
            at `max_seq_len` AFTER rendering. Every spec additionally declares exactly one
            of `tokens:` (greedy token-share fill, the historical arms) or `examples:`
            (exact row count — count-share mixtures; short sources fail loudly).
            `reasoning:` is required for HF sources:
            `native` (rows carry reasoning_content), `none` (no reasoning — every turn gets
            the template's empty marker; the generation-boundary mask handles it), or
            `strip` (deliberate pre-policy no-think rendering for nothink control arms).
        smoke: Divide every budget by 20 to validate wiring in seconds.
    """
    cfg = OmegaConf.load(config)
    assert "tulu3_repo" not in cfg, (
        "tulu3_repo/tulu3_tokens were folded into `sources`: add an entry like "
        "`tulu3: {repo: allenai/tulu-3-sft-mixture, tokens: N, shuffle_buffer: 10000}`")
    scale = _SMOKE_SCALE if smoke else 1
    seed = int(cfg.seed)
    sources: dict[str, dict] = OmegaConf.to_container(cfg.sources, resolve=True)

    tok = AutoTokenizer.from_pretrained(cfg.tokenizer)
    render_kwargs = model_profile(str(cfg.tokenizer)).render_kwargs

    rows: list[dict] = []
    kinds: dict[str, str] = {}
    for name, spec in sources.items():
        budget = _budget(name, spec, scale)
        got, kinds[name] = _load_source(tok, cfg, name, spec, budget, seed, render_kwargs)
        print(f"  {name:<24} {len(got):>5} docs  {sum(r['n_tokens'] for r in got):>9,} tok "
              f"(budget {budget[1]:,} {budget[0]}, {kinds[name]})")
        rows += got
    random.Random(seed).shuffle(rows)

    out_dir = Path(cfg.output_dir) / (f"smoke_{timestamp()}" if smoke else timestamp())
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "mixture.jsonl"
    with out_path.open("w") as f:
        for r in rows:
            rec = {"text": r["text"], "source": r["source"]}
            if r.get("supervise"):
                rec["supervise"] = r["supervise"]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

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
    for wanted, header in (("think", "reasoning preserved on every turn"),
                           ("native", "reasoning_content rendered on every turn"),
                           ("none", "the EMPTY marker on every turn"),
                           ("strip", "NO <think> at all (deliberate pre-policy render)")):
        name = next((n for n, k in kinds.items() if k == wanted), None)
        if name:
            print("\n" + "=" * 72)
            print(f"FIRST {name} EXAMPLE ({header}):")
            print("=" * 72)
            print(next(r for r in rows if r["source"] == name)["text"][:1200])

    # Validate what actually landed on disk, not just the in-memory rows, with the shared
    # per-turn census (src/utils.py) — the same yardstick train_lora's mask gate applies.
    # Rendered sources are exempt: convert_synthdoc_qwen.py validated them at render time.
    written = [json.loads(line) for line in out_path.open()]
    assert len(written) == len(rows), "mixture file is truncated"
    for name, kind in kinds.items():
        got = [r["text"] for r in written if r["source"] == name]
        census = think_census(got)
        print(f"{name}: {kind} — {census['real']} real / {census['empty']} empty / "
              f"{census['absent']} absent think blocks over {census['turns']} turns")
        if kind in ("think", "native"):
            assert census["absent"] == 0, \
                f"{name}: every assistant turn must carry a think block (preserve-thinking)"
            n_traceless = sum(1 for t in got
                              if not any(b.strip() for b in _THINK_BLOCK.findall(t)))
            assert n_traceless == 0, \
                f"{name}: {n_traceless} rows carry no real reasoning trace at all"
        elif kind == "none":
            assert census["absent"] == 0 and census["real"] == 0, \
                f"{name}: reasoning:none rows must carry exactly the empty marker per turn"
        elif kind == "strip":
            assert census["turns"] > 0 and census["real"] + census["empty"] == 0, \
                f"{name}: strip rows must contain no think block"
    print("\n" + json.dumps(stats, indent=2))
    print(f">>> wrote {out_path}")

    # Breaking out of a streaming dataset mid-shard leaves HF's parquet reader threads to
    # crash during interpreter finalization. The artifact above is written and verified, so
    # exit before finalization rather than surfacing a spurious fatal error.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
