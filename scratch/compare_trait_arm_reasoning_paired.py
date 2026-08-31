# ABOUTME: Scenario-paired follow-up to compare_trait_arm_reasoning.py: removes the domain-noun
# ABOUTME: confound from the vocabulary diff and pairs stance rates cell-by-cell across arms.

"""Paired diff of two arms' ODCV reasoning, controlling for scenario composition.

Two controls the pooled comparison lacks:
  1. Vocabulary is restricted to tokens appearing in >= `min_scenarios` distinct scenarios,
     which removes scenario-specific domain nouns (abdominal, campaign, ...) that only look
     distinctive because the arms wrote different amounts about different scenarios.
  2. Stance families are compared WITHIN each (condition, scenario) cell and averaged over
     cells, so a cell where one arm happens to have more rollouts cannot swing the mean.

Run: uv run python scratch/compare_trait_arm_reasoning_paired.py
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import fire

from scratch.compare_trait_arm_reasoning import (
    ARMS, FAMILIES, load_arm, log_odds, tokens,
)


def main(min_scenarios: int = 15, out_dir: str = "output/trait_arm_reasoning") -> None:
    """Run the paired comparison and write results_paired.{json,md}."""
    from src.utils import git_sha, timestamp  # noqa: PLC0415

    ts = timestamp()
    out = Path(out_dir) / ts
    out.mkdir(parents=True, exist_ok=True)

    loaded = {k: load_arm(Path(v)) for k, v in ARMS.items()}

    # --- control 1: scenario document frequency filter on the vocabulary -----------
    counts, scen_df = {}, Counter()
    seen = defaultdict(set)
    for arm, recs in loaded.items():
        c = Counter()
        for r in recs:
            toks = [w for t in r["reasons"] for w in tokens(t)]
            c.update(toks)
            for w in set(toks):
                seen[w].add(r["scenario"])
        counts[arm] = c
    for w, s in seen.items():
        scen_df[w] = len(s)

    prior = counts["traits123"] + counts["traits567"]
    keep = {w for w, c in prior.items() if scen_df[w] >= min_scenarios}
    a = Counter({w: c for w, c in counts["traits123"].items() if w in keep})
    b = Counter({w: c for w, c in counts["traits567"].items() if w in keep})
    lo = log_odds(a, b, Counter({w: c for w, c in prior.items() if w in keep}))

    # --- control 2: stance families averaged over shared (condition, scenario) cells ---
    cells = {arm: defaultdict(list) for arm in loaded}
    for arm, recs in loaded.items():
        for r in recs:
            cells[arm][(r["condition"], r["scenario"])].append(" ".join(r["reasons"]))
    shared = sorted(set(cells["traits123"]) & set(cells["traits567"]))

    paired = {}
    for fam, rx in FAMILIES.items():
        p = re.compile(rx, re.I)
        da, db, wins_a, wins_b, ties = [], [], 0, 0, 0
        for cell in shared:
            ra = 100 * sum(bool(p.search(t)) for t in cells["traits123"][cell]) / len(cells["traits123"][cell])
            rb = 100 * sum(bool(p.search(t)) for t in cells["traits567"][cell]) / len(cells["traits567"][cell])
            da.append(ra)
            db.append(rb)
            wins_a += ra > rb
            wins_b += rb > ra
            ties += ra == rb
        paired[fam] = {
            "traits123_cellmean": round(statistics.mean(da), 1),
            "traits567_cellmean": round(statistics.mean(db), 1),
            "delta": round(statistics.mean(da) - statistics.mean(db), 1),
            "cells_traits123_higher": wins_a,
            "cells_traits567_higher": wins_b,
            "cells_tied": ties,
        }

    payload = {"git_sha": git_sha(), "timestamp": ts, "n_shared_cells": len(shared),
               "min_scenarios": min_scenarios, "vocab_kept": len(keep),
               "paired_families": paired,
               "distinctive_traits123": [(w, round(z, 2)) for w, z in lo[:30]],
               "distinctive_traits567": [(w, round(z, 2)) for w, z in lo[-30:]][::-1]}
    (out / "results_paired.json").write_text(json.dumps(payload, indent=2))

    lines = [f"# Scenario-paired reasoning diff ({ts})", "",
             f"git_sha `{git_sha()}` | {len(shared)} shared (condition, scenario) cells | "
             f"vocabulary restricted to {len(keep)} tokens seen in >= {min_scenarios} scenarios",
             "", "## Stance families, averaged over shared cells", "",
             "| family | traits123 | traits567 | delta | cells 123>567 | cells 567>123 | tied |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for fam, v in sorted(paired.items(), key=lambda kv: -abs(kv[1]["delta"])):
        lines.append(f"| `{fam}` | {v['traits123_cellmean']} | {v['traits567_cellmean']} | "
                     f"{v['delta']:+} | {v['cells_traits123_higher']} | "
                     f"{v['cells_traits567_higher']} | {v['cells_tied']} |")
    lines += ["", "## Distinctive vocabulary, domain nouns filtered out", "",
              "**traits123:** " + ", ".join(f"{w} ({z:.1f})" for w, z in lo[:25]), "",
              "**traits567:** " + ", ".join(f"{w} ({z:.1f})" for w, z in lo[-25:][::-1])]
    (out / "results_paired.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out}/results_paired.md")


if __name__ == "__main__":
    fire.Fire(main)
