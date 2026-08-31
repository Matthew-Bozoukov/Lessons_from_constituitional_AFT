# ABOUTME: Summarises what the corpus branches on: category mix, shape, valence, who pays.
# ABOUTME: Reads an analyse_outcome_branches.py run dir; writes markdown + a stats JSON.
"""Turn extracted outcome branches into the patterns worth reporting.

Answers, over every half the move-rater marked outcome_branches=true:
  - which KINDS of branching recur, and whether reasoning and reply differ
  - how many branches a deliberation typically opens
  - the valence signature: is it good-vs-bad, or several shades of bad
  - who the consequences are said to land on
  - whether branches are live possibilities or foils for a foregone conclusion

Run: uv run python scratch/report_branch_patterns.py --run output/branch_patterns/<ts>
"""
import collections
import json
from pathlib import Path

import fire

from src.utils import read_jsonl, wilson


def _pct(k: int, n: int) -> str:
    """Rate with a Wilson interval, as a markdown cell."""
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.1f}% [{100 * lo:.0f}, {100 * hi:.0f}]"


def main(run: str, top_examples: int = 3) -> None:
    """Write the branch-pattern report for one analysis run.

    Args:
        run: Directory produced by analyse_outcome_branches.py.
        top_examples: Example axes to quote per category.
    """
    d = Path(run)
    rows = list(read_jsonl(d / "branches.jsonl"))
    tax = {c["name"]: c["description"] for c in json.loads((d / "taxonomy.json").read_text())}
    n = len(rows)
    halves = collections.Counter(r["half"] for r in rows)

    cat_n = collections.Counter()
    cat_half = collections.defaultdict(collections.Counter)
    examples = collections.defaultdict(list)
    for r in rows:
        for c in r["categories"]:
            cat_n[c] += 1
            cat_half[c][r["half"]] += 1
            if len(examples[c]) < top_examples:
                examples[c].append(r["extract"]["axis"])

    L = [f"# What the corpus branches on", "",
         f"`{d}` — {n} halves rated outcome_branches=true "
         f"({halves['reasoning']} reasoning, {halves['reply']} reply).", "",
         "## Branch kinds", "",
         "| kind | share | reasoning | reply |", "|---|--:|--:|--:|"]
    for c, k in cat_n.most_common():
        rr, pr = cat_half[c]["reasoning"], cat_half[c]["reply"]
        L.append(f"| `{c}` | {_pct(k, n)} | {100 * rr / max(halves['reasoning'], 1):.1f}% "
                 f"| {100 * pr / max(halves['reply'], 1):.1f}% |")

    counts = collections.Counter(len(r["extract"]["branches"]) for r in rows)
    L += ["", "## Shape", "", "| branches opened | halves | share |", "|---|--:|--:|"]
    for k in sorted(counts):
        L.append(f"| {k} | {counts[k]} | {100 * counts[k] / n:.1f}% |")
    mean = sum(k * v for k, v in counts.items()) / n
    L += ["", f"mean branches per deliberation: **{mean:.2f}**", ""]

    sigs = collections.Counter()
    val_all = collections.Counter()
    for r in rows:
        vs = [b.get("valence") for b in r["extract"]["branches"]]
        val_all.update(v for v in vs if v)
        sigs["".join(sorted({v[0] for v in vs if v}))] += 1
    L += ["## Valence", "", "| valence | share of all branches |", "|---|--:|"]
    tot = sum(val_all.values())
    for v, k in val_all.most_common():
        L.append(f"| {v} | {100 * k / tot:.1f}% |")
    key = {"bg": "good and bad", "b": "all bad", "g": "all good", "m": "all mixed",
           "bgm": "good, bad and mixed", "bm": "bad and mixed", "gm": "good and mixed"}
    L += ["", "| signature of a deliberation | halves | share |", "|---|--:|--:|"]
    for s, k in sigs.most_common():
        L.append(f"| {key.get(s, s)} | {k} | {100 * k / n:.1f}% |")

    bearer = collections.Counter(r["extract"].get("who_bears_it") for r in rows)
    L += ["", "## Who bears the outcome", "", "| bearer | share |", "|---|--:|"]
    for b, k in bearer.most_common():
        L.append(f"| {b} | {100 * k / n:.1f}% |")

    hedged = sum(bool(r["extract"].get("hedged")) for r in rows)
    L += ["", "## Live possibilities vs foils", "",
          f"branches offered as genuine live options: **{_pct(hedged, n)}**",
          f"one branch clearly the answer, others foils: **{_pct(n - hedged, n)}**", ""]

    L += ["## Category glossary + example axes", ""]
    for c, k in cat_n.most_common():
        L += [f"**`{c}`** ({100 * k / n:.1f}%) — {tax.get(c, '')}", ""]
        L += [f"  - _{a}_" for a in examples[c]] + [""]

    (d / "branch_patterns.md").write_text("\n".join(L) + "\n")
    (d / "branch_stats.json").write_text(json.dumps(
        {"n": n, "halves": dict(halves), "categories": dict(cat_n),
         "branch_counts": {str(k): v for k, v in counts.items()},
         "mean_branches": mean, "valence": dict(val_all),
         "signatures": dict(sigs), "bearer": dict(bearer), "hedged": hedged}, indent=2))
    print("\n".join(L[:60]))
    print(f"\n>>> {d / 'branch_patterns.md'}")


if __name__ == "__main__":
    fire.Fire(main)
