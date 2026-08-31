# ABOUTME: Blind-rates an invert_three run's originals against its rewrites on nine moves.
# ABOUTME: Reports per-move flip counts and a per-trait breakdown of the three targets.
"""Rate original vs three-move-inverted traces and report what actually flipped.

Both versions of every scenario go into one shuffled pool with no labels, so the rater
cannot tell which is which. Uses the tightened move definitions in rate_inversion_moves.

Run: uv run python scratch/rate_inverted_pairs.py
"""
import json
import random
import sys
from pathlib import Path

import fire
from dotenv import load_dotenv

from src.endpoints.openrouter import OpenRouterClient, map_threaded
from src.utils import read_jsonl, wilson, write_run_meta

sys.path.insert(0, str(Path(__file__).parent))
from rate_inversion_moves import MOVES, _rate  # noqa: E402

TARGET = {"analytic_distinction", "engages_counterargument", "outcome_branches"}


def main(run: str = "", model: str = "google/gemini-3-flash-preview", seed: int = 0,
         workers: int = 16):
    """Rate every original/inverted pair in a run and print the flip table.

    Args:
        run: invert_three run dir; defaults to the latest.
        model: OpenRouter model id for the rater.
        seed: RNG seed for shuffling.
        workers: Thread pool size.
    """
    load_dotenv()
    d = Path(run) if run else sorted(Path("output/invert_three").iterdir())[-1]
    recs = read_jsonl(d / "inversions.jsonl")
    items = [{"sid": r["scenario_id"], "version": v, "trace": r[k]}
             for r in recs for v, k in (("original", "original"), ("inverted", "inverted"))]
    random.Random(seed).shuffle(items)
    print(f"{len(recs)} scenarios x 2 = {len(items)} blind items | rater {model}\n"
          f"source {d}")

    client = OpenRouterClient()
    res = map_threaded(lambda i: _rate(client, model, items[i]["trace"]),
                       len(items), max_workers=workers, desc="rating")
    for it, r in zip(items, res):
        it.update(moves=r["moves"], form=r["form"], note=r["note"])
    (d / "ratings.jsonl").write_text("\n".join(json.dumps(i) for i in items) + "\n")
    write_run_meta(d / "rating", {"model": model, "seed": seed},
                   {"n_items": len(items), "definitions": "tightened 2026-08-21"})

    byv = {v: {i["sid"]: i for i in items if i["version"] == v}
           for v in ("original", "inverted")}
    sids = sorted(byv["original"])
    n = len(sids)
    print(f"\n{'move':34s} {'orig':>10} {'inverted':>12}  {'lost':>5} {'gained':>7}")
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
              f" {lost:>5} {gain:>7}{tag}")

    for v in ("original", "inverted"):
        nd = sum(1 for s in sids if byv[v][s]["form"] == "deliberative")
        w = sum(len(byv[v][s]["trace"].split()) for s in sids) // n
        print(f"form=deliberative {v:9s} {nd:>3}/{n}   mean {w} words")

    print("\nper-trait, the three targets (inverted counts):")
    traits = sorted({s.split("_")[0] for s in sids}, key=lambda x: int(x[1:]))
    print(f"{'move':32s}" + "".join(f"{t:>7}" for t in traits))
    for m in sorted(TARGET):
        cells = []
        for t in traits:
            sub = [s for s in sids if s.startswith(t + "_")]
            cells.append(f"{sum(1 for s in sub if byv['inverted'][s]['moves'][m])}/{len(sub)}")
        print(f"{m:32s}" + "".join(f"{c:>7}" for c in cells))

    both = sum(1 for s in sids
               if not any(byv["inverted"][s]["moves"][m] for m in TARGET))
    print(f"\nrewrites with ALL THREE targets absent: {both}/{n} ({100*both/n:.0f}%)")
    print(f"wrote {d}/ratings.jsonl")


if __name__ == "__main__":
    fire.Fire(main)
