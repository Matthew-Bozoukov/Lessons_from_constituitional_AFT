# ABOUTME: Scenario-level paired bootstrap between two ODCV arms, reported separately for
# ABOUTME: mandated, incentivized and overall.
"""Paired 95% bootstrap between two judged ODCV arms.

    uv run python scratch/odcv_pair_two_arms.py --a <dir-or-results.json> --b <...>

`results.json["per_scenario_medians"]` is keyed `Scenario/rollout_NNN` -- it is per
ROLLOUT, not per scenario. Pairing on that key would match rollout 0 of one arm against
rollout 0 of the other, which pairs nothing real: rollouts are independent draws, and the
two arms here have different rollout counts (5x vs 4x). So each arm's rollouts for a
scenario are first collapsed to one score (median), and the bootstrap then resamples
SCENARIOS -- the unit both arms actually share.

Reported per variant and pooled, because the two variants behave differently and a pooled
number hides that. Positive MR difference = arm A is MORE misaligned than arm B.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import fire

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.eval.misalignment.odcv.odcv import VARIANTS, misalignment_rate  # noqa: E402
from src.eval.misalignment.odcv.stats import paired_bootstrap  # noqa: E402


def _resolve(p: str) -> Path:
    """Accept a results.json, a run dir, or a model dir (then pick the richest run)."""
    path = Path(p)
    if path.is_file():
        return path
    cands = sorted(path.glob("**/results.json"))
    assert cands, f"no results.json under {path}"
    if len(cands) == 1:
        return cands[0]

    def n_cells(f: Path) -> int:
        m = json.loads(f.read_text()).get("per_scenario_medians", {})
        return sum(len(m.get(v, {})) for v in VARIANTS)

    best = max(cands, key=n_cells)
    print(f"    {path.name}: {len(cands)} runs, using the richest -> {best.parent.name}")
    return best


def _by_scenario(results: Path) -> dict[str, dict[str, float]]:
    """{variant: {scenario: median over that arm's rollouts}}."""
    med = json.loads(results.read_text())["per_scenario_medians"]
    out: dict[str, dict[str, float]] = {}
    for v in VARIANTS:
        acc: dict[str, list[float]] = defaultdict(list)
        for key, score in med.get(v, {}).items():
            acc[key.split("/")[0]].append(float(score))
        out[v] = {s: statistics.median(vals) for s, vals in acc.items()}
    return out


def main(a: str, b: str, n_boot: int = 10000, seed: int = 0,
         out: str = "output/odcv_bench/paired_two_arms.md") -> None:
    """Paired bootstrap of arm A minus arm B, per variant and overall."""
    fa, fb = _resolve(a), _resolve(b)
    A, B = _by_scenario(fa), _by_scenario(fb)
    name_a, name_b = fa.parts[-3], fb.parts[-3]

    rows, pooled = [], []
    for v in VARIANTS:
        shared = sorted(set(A[v]) & set(B[v]))
        pairs = [(A[v][s], B[v][s]) for s in shared]
        pooled += pairs
        rows.append((v, shared, pairs))
    rows.append(("overall", None, pooled))

    lines = [
        "# ODCV paired bootstrap — arm A vs arm B", "",
        f"- **A** `{name_a}` — {fa.parent.name}",
        f"- **B** `{name_b}` — {fb.parent.name}", "",
        "Each arm's rollouts for a scenario are collapsed to their median first; the",
        f"bootstrap then resamples **scenarios** ({n_boot:,} draws, seed {seed}).",
        "Positive difference = **A more misaligned than B**.", "",
        "| condition | n scenarios | A MR | B MR | MR diff (pp) | 95% CI | p | A sev | B sev | sev diff | 95% CI |",
        "|---|--:|--:|--:|--:|---|--:|--:|--:|--:|---|",
    ]
    detail = {}
    for label, shared, pairs in rows:
        assert pairs, f"no shared scenarios for {label}"
        boot = paired_bootstrap(pairs, n_boot, seed)
        a_s = [p[0] for p in pairs]
        b_s = [p[1] for p in pairs]
        mr_a, mr_b = misalignment_rate(a_s), misalignment_rate(b_s)
        sev_a = sum(a_s) / len(a_s)
        sev_b = sum(b_s) / len(b_s)
        lines.append(
            f"| {label} | {len(pairs)} | {mr_a:.1f}% | {mr_b:.1f}% | "
            f"**{boot['mr_diff_pp']:+.1f}** | [{boot['mr_diff_ci95'][0]:+.1f}, "
            f"{boot['mr_diff_ci95'][1]:+.1f}] | {boot['mr_diff_p_two_sided']} | "
            f"{sev_a:.2f} | {sev_b:.2f} | {boot['sev_diff']:+.2f} | "
            f"[{boot['sev_diff_ci95'][0]:+.2f}, {boot['sev_diff_ci95'][1]:+.2f}] |")
        detail[label] = {"n": len(pairs), "a_mr": round(mr_a, 1), "b_mr": round(mr_b, 1),
                         "a_sev": round(sev_a, 2), "b_sev": round(sev_b, 2), **boot}
        if shared is not None:
            only_a = sorted(set(A[label]) - set(B[label]))
            only_b = sorted(set(B[label]) - set(A[label]))
            if only_a or only_b:
                detail[label]["unmatched"] = {"only_A": only_a, "only_B": only_b}

    unm = {k: v["unmatched"] for k, v in detail.items() if "unmatched" in v}
    if unm:
        lines += ["", "## Unmatched scenarios (dropped from the pairing)", ""]
        for k, v in unm.items():
            if v["only_A"]:
                lines.append(f"- **{k}**, only in A ({len(v['only_A'])}): "
                             + ", ".join(f"`{s}`" for s in v["only_A"]))
            if v["only_B"]:
                lines.append(f"- **{k}**, only in B ({len(v['only_B'])}): "
                             + ", ".join(f"`{s}`" for s in v["only_B"]))
    lines += ["", "`overall` pools the two variants' scenario pairs; it is not an average",
              "of the two rows above, since the variants contribute different counts.", ""]

    op = Path(out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text("\n".join(lines) + "\n")
    op.with_suffix(".json").write_text(json.dumps(
        {"A": {"model": name_a, "run": fa.parent.name},
         "B": {"model": name_b, "run": fb.parent.name},
         "n_boot": n_boot, "seed": seed, "results": detail}, indent=2))
    print("\n".join(lines))
    print(f">>> {op}")


if __name__ == "__main__":
    fire.Fire(main)
