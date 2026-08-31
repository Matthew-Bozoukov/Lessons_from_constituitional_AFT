# ABOUTME: Per-trait and paired analysis of the nine-move ratings, with Wilson intervals
# ABOUTME: and a McNemar-style count of which moves the advocacy rewrite actually flipped.
"""Break the nine-move ratings down by trait and by within-scenario flip.

Run: uv run python scratch/analyse_inversion_moves.py --run <dir>
"""
import collections
from pathlib import Path

import fire

from src.utils import read_jsonl, wilson

MOVES = ["engages_counterargument", "steelmans", "analytic_distinction",
         "separate_consequentialist_pass", "outcome_branches", "names_tension",
         "acknowledges_pressure", "offers_alternative", "returns_agency"]


def main(run: str = "", out: str = ""):
    """Print per-trait presence in originals and paired flips vs the rewrite.

    Args:
        run: Ratings directory; defaults to the latest under output/inversion_moves.
        out: Optional markdown path to write.
    """
    root = Path("output/inversion_moves")
    d = Path(run) if run else sorted(root.iterdir())[-1]
    items = read_jsonl(d / "ratings.jsonl")
    orig = {i["sid"]: i for i in items if i["version"] == "original"}
    adv = {i["sid"]: i for i in items if i["version"] == "advocacy"}
    sids = sorted(orig)
    L = []

    def p(s=""):
        print(s)
        L.append(s)

    p(f"# Nine-move pattern — {d}")
    p(f"\n{len(sids)} scenarios, both versions, blind shuffled rater.\n")

    p("## Originals: does every move appear in every trait?\n")
    traits = sorted({s.split("_")[0] for s in sids}, key=lambda x: int(x[1:]))
    hdr = "| move | " + " | ".join(traits) + " | all |"
    p(hdr); p("|---" * (len(traits) + 2) + "|")
    for m in MOVES:
        cells = []
        for t in traits:
            sub = [s for s in sids if s.startswith(t + "_")]
            n = sum(1 for s in sub if orig[s]["moves"][m])
            cells.append(f"{n}/{len(sub)}")
        tot = sum(1 for s in sids if orig[s]["moves"][m])
        lo, hi = wilson(tot, len(sids))
        cells.append(f"**{100*tot/len(sids):.0f}%** [{100*lo:.0f},{100*hi:.0f}]")
        p(f"| {m} | " + " | ".join(cells) + " |")

    p("\n## What the advocacy rewrite actually flipped (paired, same scenario)\n")
    p("| move | present→absent | absent→present | net | orig% | adv% |")
    p("|---|--:|--:|--:|--:|--:|")
    for m in MOVES:
        lost = sum(1 for s in sids if orig[s]["moves"][m] and not adv[s]["moves"][m])
        gain = sum(1 for s in sids if not orig[s]["moves"][m] and adv[s]["moves"][m])
        o = sum(1 for s in sids if orig[s]["moves"][m])
        a = sum(1 for s in sids if adv[s]["moves"][m])
        p(f"| {m} | {lost} | {gain} | {gain-lost:+d} | {100*o/len(sids):.0f}% "
          f"| {100*a/len(sids):.0f}% |")

    p("\n## Form\n")
    for v, dd in (("original", orig), ("advocacy", adv)):
        nd = sum(1 for s in sids if dd[s]["form"] == "deliberative")
        p(f"- {v}: deliberative {nd}/{len(sids)} ({100*nd/len(sids):.0f}%)")

    cnt_o = collections.Counter(sum(orig[s]["moves"].values()) for s in sids)
    cnt_a = collections.Counter(sum(adv[s]["moves"].values()) for s in sids)
    p(f"\nmoves per trace, original: {dict(sorted(cnt_o.items()))}")
    p(f"moves per trace, advocacy: {dict(sorted(cnt_a.items()))}")
    n8 = sum(1 for s in sids if sum(orig[s]["moves"].values()) >= 8)
    p(f"\noriginals scoring >=8 of 9 moves: {n8}/{len(sids)} ({100*n8/len(sids):.0f}%)")

    if out:
        Path(out).write_text("\n".join(L) + "\n")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
