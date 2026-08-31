# ABOUTME: Merge every rewrite run covering the cached sample and report move survival,
# ABOUTME: per-half and per-trait, over the full set rather than one run at a time.
"""Pool the cached rewrite runs and report what survived.

Runs accumulate one batch at a time; this merges them by scenario_id, asserts every cached
scenario is covered exactly once per half, and reports survival with Wilson intervals so the
20-row and 100-row pictures are directly comparable.

Run: uv run python scratch/summarise_rewrites.py
"""
import json
from pathlib import Path

import fire

from src.utils import read_jsonl, wilson

CACHE = Path("output/rewrite_cache")
MOVES = ["engages_counterargument", "steelmans", "analytic_distinction",
         "separate_consequentialist_pass", "outcome_branches", "names_tension",
         "acknowledges_pressure", "offers_alternative", "returns_agency"]
TARGETS = ("engages_counterargument", "analytic_distinction", "outcome_branches",
           "offers_alternative")


def _merge(pattern: str, glob: str, textfile: str, version: str) -> dict:
    """Pool ratings for one half across every matching run directory."""
    out = {}
    for d in sorted(Path(pattern).glob(glob)):
        rp = d / "ratings.jsonl"
        if not rp.is_file():
            continue
        texts = {r["scenario_id"]: r for r in read_jsonl(d / textfile)}
        for i in read_jsonl(rp):
            if i["version"] == version and i["sid"] in texts:
                out[i["sid"]] = {"moves": i["moves"], "form": i["form"],
                                 "rec": texts[i["sid"]], "run": d.name}
    return out


def main(out: str = "") -> None:
    """Print pooled survival for both halves over the cached sample.

    Args:
        out: Optional markdown path to write alongside the printed table.
    """
    sids = json.loads((CACHE / "sids.json").read_text())["sids"]
    R = _merge("output/invert_extra", "engages_counterargument_outcome_branches_off_*",
               "traces.jsonl", "pass2")
    A = _merge("output/invert_answer", "engages_counterargument_outcome_branches_2*",
               "replies.jsonl", "rewritten")
    R = {s: R[s] for s in sids if s in R}
    A = {s: A[s] for s in sids if s in A}
    miss = [s for s in sids if s not in R or s not in A]
    assert not miss, f"{len(miss)} scenarios missing a half: {miss[:5]}"
    n = len(sids)
    L = [f"# Four-move rewrite, both halves — n={n}", "",
         "Pooled across every cached run. Targets: " + ", ".join(TARGETS), "",
         "| move | target | reasoning | answer |", "|---|:--:|--:|--:|"]
    print(f"n = {n}\n{'move':32s} {'reasoning':>20} {'answer':>20}")
    for m in MOVES:
        rc = sum(1 for s in sids if R[s]["moves"][m])
        ac = sum(1 for s in sids if A[s]["moves"][m])
        rl, rh = wilson(rc, n); al, ah = wilson(ac, n)
        star = " *" if m in TARGETS else "  "
        print(f"{m:30s}{star} {rc:>3}/{n} {100*rc/n:>5.1f}% [{100*rl:>3.0f},{100*rh:>3.0f}]"
              f" {ac:>3}/{n} {100*ac/n:>5.1f}% [{100*al:>3.0f},{100*ah:>3.0f}]")
        L.append(f"| {m} | {'YES' if m in TARGETS else ''} | {rc}/{n} ({100*rc/n:.0f}%) | "
                 f"{ac}/{n} ({100*ac/n:.0f}%) |")
    cleanR = sum(1 for s in sids if not any(R[s]["moves"][m] for m in TARGETS))
    cleanA = sum(1 for s in sids if not any(A[s]["moves"][m] for m in TARGETS))
    both = sum(1 for s in sids if not any(R[s]["moves"][m] for m in TARGETS)
               and not any(A[s]["moves"][m] for m in TARGETS))
    wro = sum(len(R[s]["rec"]["original"].split()) for s in sids) // n
    wrp = sum(R[s]["rec"]["words_pass2"] for s in sids) // n
    wao = sum(A[s]["rec"]["words_original"] for s in sids) // n
    wap = sum(A[s]["rec"]["words_rewritten"] for s in sids) // n
    print(f"\nall four gone: reasoning {cleanR}/{n} ({100*cleanR/n:.0f}%), "
          f"answer {cleanA}/{n} ({100*cleanA/n:.0f}%), both halves {both}/{n} "
          f"({100*both/n:.0f}%)")
    print(f"words: reasoning {wro} -> {wrp}, answer {wao} -> {wap}")
    L += ["", f"all four gone: reasoning {cleanR}/{n}, answer {cleanA}/{n}, both {both}/{n}",
          f"words: reasoning {wro} -> {wrp}, answer {wao} -> {wap}", "",
          "## Per trait, targets still present", "",
          "| trait | n | reasoning distinction | answer distinction | answer alternative |",
          "|---|--:|--:|--:|--:|"]
    print(f"\n{'trait':6s} {'n':>3} {'R dist':>8} {'A dist':>8} {'A alt':>8}")
    for t in sorted({s.split("_")[0] for s in sids}, key=lambda x: int(x[1:])):
        sub = [s for s in sids if s.startswith(t + "_")]
        rd = sum(1 for s in sub if R[s]["moves"]["analytic_distinction"])
        ad = sum(1 for s in sub if A[s]["moves"]["analytic_distinction"])
        aa = sum(1 for s in sub if A[s]["moves"]["offers_alternative"])
        print(f"{t:6s} {len(sub):>3} {rd:>4}/{len(sub)} {ad:>4}/{len(sub)} {aa:>4}/{len(sub)}")
        L.append(f"| {t} | {len(sub)} | {rd}/{len(sub)} | {ad}/{len(sub)} | {aa}/{len(sub)} |")
    if out:
        Path(out).write_text("\n".join(L) + "\n")
        print(f"\nwrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
