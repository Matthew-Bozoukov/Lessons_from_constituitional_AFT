# ABOUTME: 95% CIs for the two temp-0.7 ODCV arms published as *-rollout-scores datasets,
# ABOUTME: computed with main's src/eval/stats.py via odcv.summarise (scenario is the unit).

"""Recompute the numina-control vs synthdoc-716 ODCV intervals from the published rows.

Input is the per-rollout judge score, one row per (variant, scenario, pass); `summarise`
collapses that to the ODCV Design — scenario sampled, the two variants enumerated 50/50,
passes averaged into the cell — so the bar is the spread of per-scenario violation rates
over J scenarios (t, df J-1), on the log-odds scale because MR is a rate on [0, 100].

The per-pass MR list is printed for orientation only; it is NOT what the interval is over.
"""

from __future__ import annotations

import json
from collections import defaultdict

from huggingface_hub import hf_hub_download

from src.eval.misalignment.odcv.odcv import VARIANTS, summarise
from src.eval.misalignment.odcv.stats import arm_difference

ARMS = [
    ("numina control (0% diff advice)",
     "LASR-Callum/2026-08-30-odcv-temp07-numina-control-rollout-scores",
     "odcv_temp07_numina_rollout_scores.jsonl"),
    ("synthdoc-716 (7% diff advice)",
     "LASR-Callum/2026-08-30-odcv-temp07-synthdoc-716-rollout-scores",
     "odcv_temp07_synthdoc_rollout_scores.jsonl"),
]
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def load(repo: str, fname: str) -> list[dict]:
    path = hf_hub_download(repo, fname, repo_type="dataset")
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def medians(rows: list[dict]) -> dict[str, dict[str, list[float]]]:
    """{variant: {scenario: [severity per pass]}} — one judge, so severity is the score."""
    out: dict[str, dict[str, list[float]]] = {v: defaultdict(list) for v in VARIANTS}
    for r in sorted(rows, key=lambda r: r["rollout"]):
        out[r["variant"]][r["scenario"]].append(float(r["severity_score"]))
    return {v: dict(s) for v, s in out.items()}


def per_pass_mr(rows: list[dict]) -> dict[str, float]:
    """Each pass's MR under the same 50/50 mixture: mean over cells within a variant, then
    mean over variants. Orientation only — the published interval is over scenarios."""
    cells: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        cells[(r["rollout"], r["variant"])].append(float(r["misaligned"]))
    out = {}
    for p in sorted({r["rollout"] for r in rows}):
        per_v = [100.0 * sum(cells[(p, v)]) / len(cells[(p, v)]) for v in VARIANTS if cells[(p, v)]]
        out[p] = sum(per_v) / len(per_v)
    return out


def report(label: str, rows: list[dict]) -> dict:
    med = medians(rows)
    summ = summarise(med)
    o, mr = summ["overall"], summ["stats"]["overall"]["mr"]
    passes = per_pass_mr(rows)
    df = f"t{('%g' % mr['df']).translate(SUB)}"
    half = (mr["hi"] - mr["lo"]) / 2
    lo96, hi96 = mr["mean"] - 1.96 * mr["se"], mr["mean"] + 1.96 * mr["se"]
    print(f"\n{label} — {o['n_scenarios']} scenarios x {len(VARIANTS)} variants, "
          f"{o['n_rollouts']} rollouts over {len(passes)} passes "
          f"[{', '.join(f'{v:.1f}' for v in passes.values())}]\n")
    print(f"- mean {mr['mean']:.1f}%, SE {mr['se']:.2f} (spread of per-scenario rates, T_B)")
    print(f"- 95% CI ({df}): [{mr['lo']:.1f}, {mr['hi']:.1f}] (±{half:.1f}, {mr['shape']})")
    print(f"- (±1.96·SE band: [{lo96:.1f}, {hi96:.1f}], ±{1.96 * mr['se']:.1f})")
    print(f"- per variant: " + ", ".join(
        f"{v} {summ[v]['mr_pct']:.1f}% [{summ[v]['mr_ci95'][0]:.1f}, {summ[v]['mr_ci95'][1]:.1f}]"
        for v in VARIANTS if v in summ))
    return {"summary": summ, "medians": med, "per_pass_mr": passes, "n_rows": len(rows)}


def main() -> None:
    print("95% confidence intervals over the SCENARIOS (src/eval/stats.py @ main: scenario "
          "sampled,\nthe two variants enumerated 50/50, the 3 passes averaged into each cell):")
    out = {}
    for label, repo, fname in ARMS:
        out[label] = report(label, load(repo, fname))
    a, b = [out[label]["medians"] for label, _, _ in ARMS]
    flat = lambda m: {f"{v}/{s}": vals for v, sc in m.items() for s, vals in sc.items()}
    diff = arm_difference(flat(b), flat(a))
    print(f"\nPaired difference (synthdoc-716 - numina control): "
          f"{diff['mr_diff_pp']} pp, 95% CI {diff['mr_diff_ci95']}, "
          f"p = {diff['mr_diff_p_two_sided']}")
    print(json.dumps({k: v["summary"]["overall"] for k, v in out.items()}, indent=2))


if __name__ == "__main__":
    main()
