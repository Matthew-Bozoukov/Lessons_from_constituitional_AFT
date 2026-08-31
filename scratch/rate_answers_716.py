# ABOUTME: Blind-rate original vs rewritten assistant replies across every difficult-advice
# ABOUTME: row, merging the checkpointed rewrite runs into one set first.
"""Rate the full set of rewritten replies on the nine moves.

Merges every replies.jsonl under the rewrite root (deduping by scenario_id), asserts the
expected count, then rates both versions of each in one shuffled blind pool using the
reply-framed rater.

Run: uv run python scratch/rate_answers_716.py
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

TARGETS = ("engages_counterargument", "analytic_distinction", "outcome_branches",
           "offers_alternative")


def main(root: str = "output/invert_answer_full", expect: int = 716,
         model: str = "google/gemini-3-flash-preview", seed: int = 0,
         workers: int = 16, batch: int = 200):
    """Merge the rewrite runs, rate both versions blind, and print the flip table.

    Args:
        root: Directory holding the checkpointed rewrite runs.
        expect: Required scenario count.
        model: OpenRouter model id for the rater.
        seed: RNG seed for shuffling.
        workers: Thread pool size.
        batch: Items per checkpoint write.
    """
    load_dotenv()
    recs = {}
    for f in sorted(Path(root).glob("*/replies.jsonl")):
        for r in read_jsonl(f):
            recs[r["scenario_id"]] = r
    assert len(recs) == expect, f"merged {len(recs)}, expected {expect}"
    print(f"merged {len(recs)} replies from {root}")

    items = [{"sid": r["scenario_id"], "version": v, "text": r[k]} for r in recs.values()
             for v, k in (("original", "reply_original"), ("rewritten", "reply_rewritten"))]
    random.Random(seed).shuffle(items)
    d = Path(root) / f"rated_{timestamp()}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "replies.jsonl").write_text(
        "\n".join(json.dumps(r) for r in recs.values()) + "\n")

    client = OpenRouterClient()
    rated, path = [], d / "ratings.jsonl"
    with path.open("w") as fh:
        for start in range(0, len(items), batch):
            chunk = items[start:start + batch]
            res = map_threaded(
                lambda i: _rate(client, model, chunk[i]["text"], subject="reply"),
                len(chunk), max_workers=workers,
                desc=f"rating {start + len(chunk)}/{len(items)}")
            for it, r in zip(chunk, res):
                it.update(moves=r["moves"], form=r["form"], note=r["note"])
                fh.write(json.dumps(it) + "\n")
            fh.flush()
            rated += chunk
            print(f"  checkpoint: {len(rated)}/{len(items)} rated")
    write_run_meta(d, {"model": model, "seed": seed},
                   {"n_scenarios": len(recs), "n_items": len(items)})

    byv = {v: {i["sid"]: i for i in rated if i["version"] == v}
           for v in ("original", "rewritten")}
    sids = sorted(byv["original"])
    n = len(sids)
    L = [f"# Reply rewrite, all {n} difficult-advice rows", "",
         f"Dir: `{d}`  ", "Targets: " + ", ".join(TARGETS), "",
         "| move | target | original | rewritten | 95% CI | lost |", "|---|:--:|--:|--:|:--:|--:|"]
    print(f"\n{'move (in the REPLY)':34s} {'orig':>10} {'rewritten':>12}")
    for m, _ in MOVES:
        a = sum(1 for s in sids if byv["original"][s]["moves"][m])
        b = sum(1 for s in sids if byv["rewritten"][s]["moves"][m])
        lost = sum(1 for s in sids if byv["original"][s]["moves"][m]
                   and not byv["rewritten"][s]["moves"][m])
        lo, hi = wilson(b, n)
        tag = "  <-- TARGET" if m in TARGETS else ""
        print(f"{m:34s} {a:>4}/{n} {b:>5}/{n} [{100*lo:>3.0f},{100*hi:>3.0f}]{tag}")
        L.append(f"| {m} | {'YES' if m in TARGETS else ''} | {a}/{n} | {b}/{n} | "
                 f"[{100*lo:.0f},{100*hi:.0f}] | {lost} |")
    wo = sum(recs[s]["words_original"] for s in sids) // n
    wr = sum(recs[s]["words_rewritten"] for s in sids) // n
    clean = sum(1 for s in sids if not any(byv["rewritten"][s]["moves"][m] for m in TARGETS))
    fo = sum(1 for s in sids if byv["original"][s]["form"] == "deliberative")
    fr = sum(1 for s in sids if byv["rewritten"][s]["form"] == "deliberative")
    L += ["", f"mean words: {wo} -> {wr} ({wr/wo:.0%})",
          f"form=deliberative: {fo}/{n} -> {fr}/{n}",
          f"replies with all four targets absent: {clean}/{n} ({100*clean/n:.0f}%)"]
    print(f"\nwords {wo} -> {wr} ({wr/wo:.0%}) | deliberative {fo}/{n} -> {fr}/{n}")
    print(f"all four targets absent: {clean}/{n} ({100*clean/n:.0f}%)")
    (d / "results.md").write_text("\n".join(L) + "\n")
    print(f"\nwrote {d}")


if __name__ == "__main__":
    fire.Fire(main)
