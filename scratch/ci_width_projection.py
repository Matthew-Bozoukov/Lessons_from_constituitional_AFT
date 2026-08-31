# ABOUTME: Project ODCV confidence-interval width against rollouts-per-cell, for planning runs.
# ABOUTME: Rollouts cluster in scenarios, so more depth per cell hits a between-scenario floor.
"""How much would more incentivized rollouts actually narrow the interval?

The bootstrap resamples SCENARIOS, not rollouts, because two rollouts of the same scenario
are not independent draws. Total variance is therefore

    between-scenario / S   +   within-scenario / (S * k)

with S scenarios and k rollouts each. Only the second term shrinks with k, so once k is a
few, the interval is governed by S -- and ODCV offers at most 30 incentivized cells under
the standard exclusions.

This estimates both variance components from an arm's real per-scenario violation counts
(method of moments on the beta-binomial), then simulates the bootstrap at chosen (S, k) to
report expected CI width. Simulated, so it answers "how wide would it be", not "what would
the rate be".

Run: uv run python scratch/ci_width_projection.py --arm MATCHED_CONTROL
"""
import json
import random
import statistics
from pathlib import Path

import fire

from src.eval.misalignment.odcv.odcv import misalignment_rate

ROOT = Path("output/odcv_bench")
ARMS = {
    "MATCHED_CONTROL":
        "qwen3_6-27b-lora-t2-9284-synthdoc-716-dynbatch-r64/combined2x_20260820_223257",
    "BOTH_PRUNED":
        "qwen3_6-27b-lora-t2-9284-synthdoc-662-bothpruned-r64/combined2x_20260823_221456",
    "BRANCHES":
        "qwen3_6-27b-lora-t2-9284-synthdoc-654-branches-r64/combined2x_20260824_101532",
}


def _by_scenario(rel: str, variant: str) -> dict[str, list[float]]:
    """Median-across-judges severity per rollout, grouped by scenario, for one variant."""
    per: dict[str, list[float]] = {}
    for f in (ROOT / rel / "evaluations").glob("scores_*.json"):
        for k, v in json.loads(f.read_text()).items():
            s = v.get("score") if isinstance(v, dict) else v
            if isinstance(s, (int, float)):
                per.setdefault(k, []).append(s)
    out: dict[str, list[float]] = {}
    for k, vals in per.items():
        if not k.startswith(variant):
            continue
        out.setdefault(k.rsplit("/rollout_", 1)[0], []).append(statistics.median(vals))
    return out


def main(arm: str = "MATCHED_CONTROL", variant: str = "incentivized",
         totals: str = "60,100,150,200,300,400", reps: int = 4000, seed: int = 0,
         max_cells: int = 30) -> None:
    """Report simulated CI width for a range of total incentivized rollouts.

    Args:
        arm: Key of ARMS to take the observed violation pattern from.
        variant: "incentivized" or "mandated".
        totals: Comma-separated total rollout counts to project.
        reps: Bootstrap replicates per projection.
        seed: RNG seed.
        max_cells: Scenarios available for this variant under the standard exclusions.
    """
    rel = ARMS[arm]
    obs = _by_scenario(rel, variant)
    S_obs = len(obs)
    n_obs = sum(len(v) for v in obs.values())
    hits = {s: sum(1 for x in v if x >= 3) for s, v in obs.items()}
    ns = {s: len(v) for s, v in obs.items()}
    p_hat = sum(hits.values()) / n_obs

    # Method of moments for the beta-binomial: separate real scenario-to-scenario spread
    # from the binomial noise of few rollouts per scenario.
    ks = list(obs)
    num = sum(ns[s] * ((hits[s] / ns[s]) - p_hat) ** 2 for s in ks)
    den = sum(ns[s] for s in ks) - 1
    msb = num / max(den, 1)
    within = p_hat * (1 - p_hat)
    kbar = n_obs / S_obs
    between = max(msb - within, 0.0) / max(kbar, 1)
    icc = between / (between + within) if (between + within) else 0.0
    print(f"{arm} / {variant}: {S_obs} scenarios, {n_obs} rollouts, rate {100*p_hat:.1f}%")
    print(f"  between-scenario var {between:.4f}   within {within:.4f}   ICC {icc:.3f}")
    if between == 0:
        print("  (no excess scenario spread detected — rollouts behave independently here)")

    # Beta parameters matching p_hat and the between-scenario variance.
    if 0 < between < within:
        conc = within / between - 1
        a, b = p_hat * conc, (1 - p_hat) * conc
    else:
        a = b = None

    rng = random.Random(seed)

    def sim(S: int, k: int) -> float:
        """Median CI width over `reps` simulated studies of S scenarios x k rollouts."""
        widths = []
        for _ in range(60):
            cells = []
            for _ in range(S):
                p = rng.betavariate(a, b) if a else p_hat
                cells.append([1 if rng.random() < p else 0 for _ in range(k)])
            boot = []
            for _ in range(reps // 40):
                vals = []
                for _ in range(S):
                    vals.extend(cells[rng.randrange(S)])
                boot.append(100 * sum(vals) / len(vals))
            boot.sort()
            widths.append(boot[int(0.975 * len(boot)) - 1] - boot[int(0.025 * len(boot))])
        return statistics.median(widths)

    print(f"\nprojected 95% CI width, {max_cells} incentivized cells available:")
    print(f"{'total rollouts':>15s} {'per cell':>9s} {'CI width':>10s}")
    # Fire turns a comma list into a tuple, so accept either shape.
    tot_list = ([int(x) for x in totals] if isinstance(totals, (list, tuple))
                else [int(x) for x in str(totals).split(",") if x.strip()])
    for t in tot_list:
        k = max(1, round(t / max_cells))
        print(f"{max_cells * k:>15d} {k:>9d} {sim(max_cells, k):>9.1f} pp")

    print(f"\nfor comparison, more CELLS at 2 rollouts each (not available in ODCV today):")
    for S in (30, 45, 60, 90):
        print(f"{S * 2:>15d} {2:>9d} {sim(S, 2):>9.1f} pp   ({S} cells)")


if __name__ == "__main__":
    fire.Fire(main)
