# ABOUTME: Merges every three-move inversion run into one 716-row set and blind-rates it.
# ABOUTME: Ratings are checkpointed per batch so an interrupted run keeps its work.
"""Rate the full difficult-advice corpus, original against three-move-inverted.

Merges all runs under output/invert_three (deduping by scenario_id, last write wins),
asserts the expected count, then rates both versions of every scenario in one shuffled
blind pool using the tightened move definitions.

Run: uv run python scratch/rate_all716.py
"""
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

TARGET = {"analytic_distinction", "engages_counterargument", "outcome_branches"}


def merge(root: str = "output/invert_three") -> dict:
    """Every rewritten scenario across all runs, keyed by scenario_id."""
    recs = {}
    for f in sorted(Path(root).glob("*/inversions.jsonl")):
        for r in read_jsonl(f):
            recs[r["scenario_id"]] = r
    return recs


def main(expect: int = 716, model: str = "google/gemini-3-flash-preview", seed: int = 0,
         workers: int = 16, batch: int = 200, out_dir: str = "output/invert_all716"):
    """Merge all inversion runs, rate both versions blind, and report the flip table.

    Args:
        expect: Required scenario count; the run aborts if the merge falls short.
        model: OpenRouter model id for the rater.
        seed: RNG seed for shuffling.
        workers: Thread pool size.
        batch: Items per checkpoint write.
        out_dir: Output directory root.
    """
    load_dotenv()
    recs = merge()
    assert len(recs) == expect, f"merged {len(recs)} scenarios, expected {expect}"
    print(f"merged {len(recs)} scenarios from output/invert_three/*/inversions.jsonl")

    items = [{"sid": r["scenario_id"], "version": v, "trace": r[k]}
             for r in recs.values()
             for v, k in (("original", "original"), ("inverted", "inverted"))]
    random.Random(seed).shuffle(items)
    d = Path(out_dir) / timestamp()
    d.mkdir(parents=True, exist_ok=True)
    (d / "inversions.jsonl").write_text(
        "\n".join(json.dumps(r) for r in recs.values()) + "\n")

    client = OpenRouterClient()
    rated, path = [], d / "ratings.jsonl"
    with path.open("w") as fh:
        for start in range(0, len(items), batch):
            chunk = items[start:start + batch]
            res = map_threaded(lambda i: _rate(client, model, chunk[i]["trace"]),
                               len(chunk), max_workers=workers,
                               desc=f"rating {start + len(chunk)}/{len(items)}")
            for it, r in zip(chunk, res):
                it.update(moves=r["moves"], form=r["form"], note=r["note"])
                fh.write(json.dumps(it) + "\n")
            fh.flush()
            rated += chunk
            print(f"  checkpoint: {len(rated)}/{len(items)} rated")

    write_run_meta(d, {"model": model, "seed": seed},
                   {"n_scenarios": len(recs), "n_items": len(items),
                    "definitions": "tightened 2026-08-21"})

    byv = {v: {i["sid"]: i for i in rated if i["version"] == v}
           for v in ("original", "inverted")}
    sids = sorted(byv["original"])
    n = len(sids)
    L = [f"# Three-move inversion, all {n} difficult-advice traces", "",
         f"Dir: `{d}`  ",
         "Rewriter anthropic/claude-sonnet-5 temp 0.7. Rater "
         "google/gemini-3-flash-preview temp 0, blind and shuffled, tightened definitions.",
         "", "| move | target | orig | inverted | 95% CI | lost | gained |",
         "|---|:--:|--:|--:|:--:|--:|--:|"]
    print(f"\n{'move':34s} {'orig':>10} {'inverted':>12}  {'lost':>5} {'gain':>5}")
    for m, _ in MOVES:
        o = sum(1 for s in sids if byv["original"][s]["moves"][m])
        a = sum(1 for s in sids if byv["inverted"][s]["moves"][m])
        lost = sum(1 for s in sids if byv["original"][s]["moves"][m]
                   and not byv["inverted"][s]["moves"][m])
        gain = sum(1 for s in sids if not byv["original"][s]["moves"][m]
                   and byv["inverted"][s]["moves"][m])
        lo, hi = wilson(a, n)
        tag = "  <-- TARGET" if m in TARGET else ""
        print(f"{m:34s} {o:>4}/{n} {a:>4}/{n} [{100*lo:>3.0f},{100*hi:>3.0f}]"
              f" {lost:>5} {gain:>5}{tag}")
        L.append(f"| {m} | {'YES' if m in TARGET else ''} | {o}/{n} | {a}/{n} | "
                 f"[{100*lo:.0f},{100*hi:.0f}] | {lost} | {gain} |")

    fo = sum(1 for s in sids if byv["original"][s]["form"] == "deliberative")
    fi = sum(1 for s in sids if byv["inverted"][s]["form"] == "deliberative")
    wo = sum(len(byv["original"][s]["trace"].split()) for s in sids) // n
    wi = sum(len(byv["inverted"][s]["trace"].split()) for s in sids) // n
    both = sum(1 for s in sids if not any(byv["inverted"][s]["moves"][m] for m in TARGET))
    L += ["", f"form=deliberative: original {fo}/{n}, inverted {fi}/{n}",
          f"mean words: original {wo}, inverted {wi}",
          f"all three targets absent: {both}/{n} ({100*both/n:.0f}%)", "",
          "## Per-trait, targets still present after rewrite", ""]
    print(f"form=deliberative  original {fo}/{n}  inverted {fi}/{n}")
    print(f"mean words         original {wo}     inverted {wi}")
    print(f"all three targets absent: {both}/{n} ({100*both/n:.0f}%)")

    traits = sorted({s.split("_")[0] for s in sids}, key=lambda x: int(x[1:]))
    L += ["| move | " + " | ".join(traits) + " |", "|---" * (len(traits) + 1) + "|"]
    for m in sorted(TARGET):
        cells = []
        for t in traits:
            sub = [s for s in sids if s.startswith(t + "_")]
            cells.append(f"{sum(1 for s in sub if byv['inverted'][s]['moves'][m])}/{len(sub)}")
        L.append(f"| {m} | " + " | ".join(cells) + " |")
    (d / "results.md").write_text("\n".join(L) + "\n")
    print(f"\nwrote {d}")


if __name__ == "__main__":
    fire.Fire(main)
