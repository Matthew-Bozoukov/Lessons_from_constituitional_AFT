# ABOUTME: Follow-ups on the temp-0.7 ODCV rollout-score datasets: the same main-code CIs
# ABOUTME: restricted to pass 1 only, plus a completeness audit of the 80 cells x 3 passes.

from __future__ import annotations

import json
from collections import defaultdict

from huggingface_hub import hf_hub_download

from src.eval.misalignment.odcv.odcv import VARIANTS, summarise
from scratch.odcv_temp07_ci_from_main import ARMS, SUB, load, medians


def line(label: str, med: dict) -> None:
    summ = summarise(med)
    o, mr = summ["overall"], summ["stats"]["overall"]["mr"]
    half = (mr["hi"] - mr["lo"]) / 2
    df = f"t{('%g' % mr['df']).translate(SUB)}"
    print(f"\n{label} — {o['n_scenarios']} scenarios, {o['n_rollouts']} rollouts")
    print(f"- mean {mr['mean']:.1f}%, SE {mr['se']:.2f}")
    print(f"- 95% CI ({df}): [{mr['lo']:.1f}, {mr['hi']:.1f}] (±{half:.1f}, {mr['shape']})")
    print(f"- (±1.96·SE band: [{mr['mean'] - 1.96 * mr['se']:.1f}, "
          f"{mr['mean'] + 1.96 * mr['se']:.1f}], ±{1.96 * mr['se']:.1f})")
    print(f"- dropped scenarios: {o['dropped_scenarios'] or 'none'}")


def audit(label: str, rows: list[dict]) -> None:
    passes = sorted({r["rollout"] for r in rows})
    cells = defaultdict(set)
    for r in rows:
        cells[(r["variant"], r["scenario"])].add(r["rollout"])
    scen = {v: sorted({r["scenario"] for r in rows if r["variant"] == v}) for v in VARIANTS}
    all_cells = [(v, s) for v in VARIANTS for s in scen[v]]
    missing = {c: sorted(set(passes) - cells[c]) for c in all_cells if len(cells[c]) < len(passes)}
    print(f"\n{label}: {len(rows)}/{len(all_cells) * len(passes)} rollouts "
          f"({len(all_cells)} cells x {len(passes)} passes), "
          f"{sum(len(v) for v in missing.values())} absent in {len(missing)} cells")
    for c, ps in sorted(missing.items()):
        print(f"    {c[0]}/{c[1]}: missing {', '.join(ps)}")
    for p in passes:
        n = sum(1 for r in rows if r["rollout"] == p)
        print(f"    {p}: {n}/{len(all_cells)} cells")


def main() -> None:
    data = {label: load(repo, f) for label, repo, f in ARMS}
    print("=" * 70, "\nCOMPLETENESS AUDIT (rows published per cell x pass)")
    for label, rows in data.items():
        audit(label, rows)
    print("\n" + "=" * 70,
          "\nFIRST PASS ONLY (rollout_000): one rollout per cell, same Design (t over scenarios)")
    for label, rows in data.items():
        first = [r for r in rows if r["rollout"] == "rollout_000"]
        line(label, medians(first))
    print("\n" + "=" * 70, "\nALL THREE PASSES (for comparison)")
    for label, rows in data.items():
        line(label, medians(rows))


if __name__ == "__main__":
    main()
