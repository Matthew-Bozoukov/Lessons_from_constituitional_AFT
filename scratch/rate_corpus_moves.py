# ABOUTME: Blind-rates a synth SFT corpus on the nine deliberative moves, reasoning vs reply.
# ABOUTME: Gives the pre-ablation baseline the ruleform/ablated arms are measured against.
"""How often does the corpus make each deliberative move, in reasoning and in the reply?

Runs the tightened nine-move rater (scratch/rate_inversion_moves.py) over every row of a
stage_N_sft.jsonl corpus. Both halves of every row go into one shuffled pool and are shown
one per call with no label, so the rater cannot tell reasoning from reply or one trait from
another. Reports overall rates with Wilson intervals and a per-trait breakdown.

Run: uv run python scratch/rate_corpus_moves.py --limit 20      # smoke, prints cost
     uv run python scratch/rate_corpus_moves.py                 # full corpus
"""
import collections
import json
import random
import sys
from pathlib import Path

import fire
from dotenv import load_dotenv

from src.endpoints.openrouter import OpenRouterClient, map_threaded
from src.utils import read_jsonl, timestamp, wilson, write_run_meta

sys.path.insert(0, str(Path(__file__).parent))
from rate_inversion_moves import MOVES, _rate  # noqa: E402

SRC = "output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl"
HALVES = (("reasoning", "reasoning_content", "reasoning"), ("reply", "content", "reply"))


def _items(rows: list[dict]) -> list[dict]:
    """One rating item per (row, half), carrying the trait for the breakdown."""
    out = []
    for r in rows:
        asst = next(m for m in r["messages"] if m["role"] == "assistant")
        for half, key, subject in HALVES:
            text = asst.get(key, "")
            assert text.strip(), f"{r['metadata']['scenario_id']} has empty {key}"
            out.append({"sid": r["metadata"]["scenario_id"], "trait": r["metadata"]["trait_id"],
                        "half": half, "subject": subject, "text": text})
    return out


def _table(rated: list[dict], half: str) -> list[str]:
    """Markdown rows of move -> rate with a Wilson interval, for one half."""
    sub = [r for r in rated if r["half"] == half]
    n = len(sub)
    lines = [f"### {half}  (n={n})", "",
             "| move | rate | 95% CI |", "|---|--:|:--:|"]
    for m, _ in MOVES:
        h = sum(r["moves"][m] for r in sub)
        lo, hi = wilson(h, n)
        lines.append(f"| {m} | {100 * h / n:.1f}% | [{100 * lo:.1f}, {100 * hi:.1f}] |")
    d = sum(r["form"] == "deliberative" for r in sub)
    lo, hi = wilson(d, n)
    lines += ["", f"form = deliberative: **{100 * d / n:.1f}%** [{100 * lo:.1f}, {100 * hi:.1f}]", ""]
    return lines


def _per_trait(rated: list[dict], half: str) -> list[str]:
    """Markdown matrix of move x trait for one half."""
    sub = [r for r in rated if r["half"] == half]
    traits = sorted({r["trait"] for r in sub}, key=lambda t: (len(t), t))
    by = collections.defaultdict(list)
    for r in sub:
        by[r["trait"]].append(r)
    lines = [f"### {half} by trait", "",
             "| move | " + " | ".join(f"{t} (n={len(by[t])})" for t in traits) + " |",
             "|---" * (len(traits) + 1) + "|"]
    for m, _ in MOVES:
        cells = [f"{100 * sum(r['moves'][m] for r in by[t]) / len(by[t]):.0f}%" for t in traits]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return lines + [""]


def main(src: str = SRC, limit: int = 0, model: str = "google/gemini-3-flash-preview",
         seed: int = 0, workers: int = 16, batch: int = 200,
         out_dir: str = "output/corpus_moves"):
    """Rate every row's reasoning and reply on the nine moves and write the tables.

    Args:
        src: stage_N_sft.jsonl corpus to rate.
        limit: Rate only the first N rows (0 = all); use for a costed smoke run.
        model: OpenRouter model id for the rater.
        seed: RNG seed for shuffling the blind pool.
        workers: Thread pool size.
        batch: Items per checkpoint write.
        out_dir: Output directory root.
    """
    load_dotenv()
    rows = read_jsonl(src)
    if limit:
        rows = rows[:limit]
    items = _items(rows)
    random.Random(seed).shuffle(items)
    print(f"{len(rows)} rows -> {len(items)} rating calls ({model})")
    print(f"first item: {items[0]['half']} / {items[0]['trait']} / "
          f"{len(items[0]['text'].split())} words")

    d = Path(out_dir) / timestamp()
    d.mkdir(parents=True, exist_ok=True)
    client = OpenRouterClient()
    rated, path = [], d / "ratings.jsonl"
    with path.open("w") as fh:
        for start in range(0, len(items), batch):
            chunk = items[start:start + batch]
            res = map_threaded(lambda i: _rate(client, model, chunk[i]["text"],
                                               subject=chunk[i]["subject"]),
                               len(chunk), max_workers=workers,
                               desc=f"rating {start + len(chunk)}/{len(items)}")
            for it, out in zip(chunk, res):
                rec = {**{k: it[k] for k in ("sid", "trait", "half")},
                       "moves": out["moves"], "form": out["form"], "note": out.get("note", "")}
                rated.append(rec)
                fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if start == 0:
                print(f"\nSAMPLE RATING: {json.dumps(rated[0], indent=2)[:600]}\n")

    L = [f"# Deliberative moves in `{src}`", "",
         f"{len(rows)} rows, {len(items)} blind ratings by `{model}`.", ""]
    for half, _, _ in HALVES:
        L += _table(rated, half)
    for half, _, _ in HALVES:
        L += _per_trait(rated, half)
    (d / "results.md").write_text("\n".join(L) + "\n")
    write_run_meta(d, {"src": src, "limit": limit, "model": model, "seed": seed},
                   {"n_rows": len(rows), "n_ratings": len(items)})
    print("\n".join(L))
    print(f"\nwrote {d}/results.md")
    print(f"judging cost: ${client.total_cost_usd:.2f}" if hasattr(client, "total_cost_usd")
          else "(client exposes no cost counter)")


if __name__ == "__main__":
    fire.Fire(main)
