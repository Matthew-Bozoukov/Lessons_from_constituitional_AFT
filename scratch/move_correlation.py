# ABOUTME: Measures how much independent signal the nine deliberative moves actually carry.
# ABOUTME: Prevalence, variance, pairwise phi and effective rank, separately per half.
"""Are the nine moves nine measurements, or one measurement nine times?

A move rated true on ~99% of rows carries almost no information: it cannot correlate with
anything because it does not vary. This reports, per half, each move's prevalence and
variance, the pairwise phi matrix over the moves that DO vary, and the eigenvalue spectrum
of that matrix -- how many independent dimensions the set really spans.

Run: uv run python scratch/move_correlation.py
"""
import collections
import json
from pathlib import Path

import fire
import numpy as np

RATINGS = Path("output/corpus_moves/20260823_094836/ratings.jsonl")
MOVES = ["engages_counterargument", "steelmans", "analytic_distinction",
         "separate_consequentialist_pass", "outcome_branches", "names_tension",
         "acknowledges_pressure", "offers_alternative", "returns_agency"]


def _matrix(rows: list[dict]) -> np.ndarray:
    """Binary rows x moves matrix for one half."""
    m = np.array([[1.0 if r["moves"].get(k) else 0.0 for k in MOVES] for r in rows])
    assert m.shape[1] == len(MOVES), m.shape
    return m


def _report(name: str, m: np.ndarray) -> None:
    """Print prevalence, variance, phi matrix and effective rank for one half."""
    n = m.shape[0]
    p = m.mean(axis=0)
    sd = m.std(axis=0)
    print(f"\n{'=' * 78}\n{name}  (n={n})\n{'=' * 78}")
    print(f"{'move':<32}{'rate':>8}{'sd':>8}   information")
    for i, k in enumerate(MOVES):
        bar = "#" * int(round(sd[i] * 40))
        flag = "  AT CEILING" if p[i] > 0.97 or p[i] < 0.03 else ""
        print(f"{k:<32}{100 * p[i]:7.1f}%{sd[i]:8.3f}   {bar}{flag}")

    live = [i for i in range(len(MOVES)) if 0.03 < p[i] < 0.97]
    print(f"\nmoves with usable variance: {len(live)}/{len(MOVES)} "
          f"-> {', '.join(MOVES[i] for i in live)}")
    if len(live) < 2:
        print("too few varying moves for a correlation structure")
        return

    sub = m[:, live]
    c = np.corrcoef(sub, rowvar=False)
    print("\nphi correlation among varying moves")
    names = [MOVES[i][:14] for i in live]
    print(" " * 16 + "".join(f"{s:>16}" for s in names))
    for a, row in enumerate(c):
        print(f"{names[a]:<16}" + "".join(f"{v:>16.2f}" for v in row))

    ev = np.linalg.eigvalsh(c)[::-1]
    tot = ev.sum()
    cum = np.cumsum(ev) / tot
    print("\neigenvalues (variance explained)")
    for i, (e, cu) in enumerate(zip(ev, cum), 1):
        print(f"  PC{i}: {e:6.3f}  {100 * e / tot:5.1f}%  cumulative {100 * cu:5.1f}%")
    eff = float(np.exp(-(ev / tot * np.log(ev / tot + 1e-12)).sum()))
    print(f"\neffective dimensionality (entropy of spectrum): {eff:.2f} "
          f"of {len(live)} varying moves")


def main() -> None:
    """Report the correlation structure of the nine moves, per half."""
    rows = [json.loads(l) for l in RATINGS.read_text().splitlines()]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["half"]].append(r)
    for half in ("reasoning", "reply"):
        _report(half, _matrix(by[half]))
    _report("both halves pooled", _matrix(rows))


if __name__ == "__main__":
    fire.Fire(main)
