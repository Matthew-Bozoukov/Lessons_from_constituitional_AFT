# ABOUTME: Noise floor and good-vs-bad separation for ODCV-lite at scenario level: same-checkpoint rerun and seed pairs,
# ABOUTME: per-cell agreement, and which scenario/variant cells carry the good-arm vs bad-arm gap.
"""uv run python scratch/autoresearch/odcv_noise.py   (after odcv_matrix.py) -> output/autoresearch/odcv_cells_good_bad.csv"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

OUT = "output/autoresearch"
GOOD = ["s25_s0", "s25_s1", "swap_self_otherai", "swap_advice", "swap_otherai", "s28_da25"]
BAD = ["s28_base_a", "s28_base_b", "s28_no_t6", "new_otherai", "new_self", "new_explicit", "lowstakes", "nonmoral"]
M = pd.read_csv(f"{OUT}/odcv_scenario_matrix.csv")
W = M.pivot_table(index=["scenario", "variant"], columns="arm", values="vr")  # violation rate per cell (0..1, k/3)
Ws = W.pivot_table if False else None


def pair(a: str, b: str, level: str = "cell") -> dict:
    if level == "cell":
        x, y = W[a] * 3, W[b] * 3
    else:  # scenario: pool variants -> k/6
        x, y = W[a].groupby(level=0).sum() * 3, W[b].groupby(level=0).sum() * 3
    d = (x - y).abs()
    return dict(a=a, b=b, level=level, r=round(stats.pearsonr(x, y)[0], 2), rho=round(stats.spearmanr(x, y)[0], 2),
                exact=round((d == 0).mean(), 2), within1=round((d <= 1).mean(), 2), mean_abs=round(d.mean(), 2),
                any_a=int((x > 0).sum()), any_b=int((y > 0).sum()), both=int(((x > 0) & (y > 0)).sum()),
                sum_a=int(x.sum()), sum_b=int(y.sum()))


def perm_null(a: str, b: str, n: int = 2000, seed: int = 0) -> float:
    """Null for 'two runs of the same cell rates': binomial resampling 3 passes from pooled rate p = (a+b)/2."""
    rng = np.random.default_rng(seed)
    p = ((W[a] + W[b]) / 2).values
    sims = [np.abs(rng.binomial(3, p) - rng.binomial(3, p)).mean() for _ in range(n)]
    return float(np.mean(sims))


def main() -> None:
    pairs = [("s28_base_a", "s28_base_b"), ("s25_s0", "s25_s1"), ("s14_a", "s14_b"), ("nosynth_a", "nosynth_b"),
             ("s28_base_a", "s25_s0"), ("s28_base_b", "s25_s1"), ("s28_base_a", "s25_s1"), ("s28_base_b", "s25_s0"),
             ("new_otherai", "swap_otherai"), ("new_otherai", "s28_base_a")]
    rows = [pair(a, b, lvl) for a, b in pairs for lvl in ("cell", "scenario")]
    P = pd.DataFrame(rows)
    P["binom_null_mean_abs_cell"] = [round(perm_null(a, b), 2) if l == "cell" else None for a, b, l in zip(P.a, P.b, P.level)]
    print(P.to_string(index=False))
    P.to_csv(f"{OUT}/odcv_noise_pairs.csv", index=False)

    # good vs bad per cell
    G = W[GOOD].mean(axis=1); B = W[BAD].mean(axis=1)
    C = pd.DataFrame({"good": G, "bad": B, "gap": B - G, "nosynth": W[["nosynth_a", "nosynth_b"]].mean(axis=1),
                      "tools": W["tools"], "bad_n_arms_any": (W[BAD] > 0).sum(axis=1), "good_n_arms_any": (W[GOOD] > 0).sum(axis=1)})
    for a in GOOD + BAD:
        C[a] = (W[a] * 3).round().astype(int)
    C = C.sort_values("gap", ascending=False)
    C.to_csv(f"{OUT}/odcv_cells_good_bad.csv")
    pd.set_option("display.width", 300); pd.set_option("display.max_columns", 40)
    print(C.head(30).round(2).to_string())
    tot = (C.gap.clip(lower=0)).sum()
    cs = C.gap.cumsum() / C.gap.sum()
    print("total gap (sum over cells of bad-good rate):", round(C.gap.sum(), 2), "positive part", round(tot, 2))
    print("cells to reach 50/75/90% of net gap:", [(cs < q).sum() + 1 for q in (0.5, 0.75, 0.9)])
    print("cells with gap>0.2:", (C.gap > 0.2).sum(), " gap<-0.1:", (C.gap < -0.1).sum(), " zero-everywhere:", ((C.good == 0) & (C.bad == 0)).sum())
    # variant split of the gap
    print(C.groupby(level=1).gap.sum())
    # scenario level
    S = C.groupby(level=0)[["good", "bad", "gap", "nosynth"]].mean().sort_values("gap", ascending=False)
    print(S.head(20).round(2).to_string())
    # leave-one-out / per-arm contribution of top cluster
    top = C.index[:10]
    for a in GOOD + BAD + ["nosynth_a", "tools", "swap_self", "sysdiv", "swap_explicit", "s28_da5", "s25_no_t6"]:
        tot_v = (W[a] * 3).sum(); in_top = (W.loc[top, a] * 3).sum()
        print(f"{a:18s} total viol {int(tot_v):3d}  in top10 gap cells {int(in_top):3d}  outside {int(tot_v - in_top):3d}")


if __name__ == "__main__" and "--arm-table" not in __import__("sys").argv:
    main()


def arm_table() -> None:
    """Per-arm table: MR + fixed CI, mandated/incentivized, submit, MR>=4, cluster-B violations, first-turn stats."""
    S = pd.read_csv(f"{OUT}/odcv_arm_summary.csv").set_index("arm")
    F = pd.read_csv(f"{OUT}/odcv_transcript_features.csv")
    C = pd.read_csv(f"{OUT}/odcv_cells_good_bad.csv")
    B = set(zip(C.scenario[:11], C.variant[:11]))
    F["clusterB"] = [(s, v) in B for s, v in zip(F.scenario, F.variant)]
    F["flag"] = F.first_reason.fillna("").str.contains(
        r"(?:concern|careful|cautious|red flag|pressure|tension|problematic|ethic|integrity|fabricat|manipulat|gam(?:e|ing) the|falsif|patient safety|harm)",
        case=False, regex=True)
    T = F.groupby("arm").apply(lambda d: pd.Series({
        "viol_clusterB_of33": int(d[d.clusterB].violation.sum()), "viol_outside_of207": int(d[~d.clusterB].violation.sum()),
        "first_turn_med_chars": int(d.first_reason_chars.median()), "first_turn_flags_pct": round(100 * d.flag.mean()),
        "first_turn_flags_pct_clusterB": round(100 * d[d.clusterB].flag.mean())}), include_groups=False)
    S["mr_ge4"] = (100 * (S.sev4 + S.sev5) / S.n).round(1)
    out = S[["run", "mr", "lo", "hi", "mr_mandated", "mr_incentivized", "submit", "mr_ge4"]].join(T)
    out.to_csv(f"{OUT}/odcv_arm_table.csv")
    print(out.drop(columns=["run"]).to_string())


if __name__ == "__main__" and "--arm-table" in __import__("sys").argv:
    arm_table()
