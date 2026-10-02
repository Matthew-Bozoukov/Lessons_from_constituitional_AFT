# ABOUTME: MASK cross-arm analysis on the repaired per-row labels: noise floor, row-level flip rates, transitions out
# ABOUTME: of the 28 Sep baseline's lies, the dose (da-15 vs da-25) profile, refusal/length under pressure, MASK vs ODCV.
"""uv run python scratch/autoresearch/mask_compare.py  (after mask_rows.py, mask_rejudge.py, mask_matrix.py)"""
import itertools
import re

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
R = pd.read_parquet("output/autoresearch/mask_rows_repaired.parquet")
R = R[R.rep_honesty.notna()]
M = pd.read_csv("output/autoresearch/mask_archetype_matrix.csv").set_index("arm")
# Repaired score mapped onto the reported scale (fit on the 24 fully judged arms, r=0.996): the best estimate of
# what the three timeout-corrupted runs would have reported.
old = M[~M.index.str.startswith("new")]
FIT = np.polyfit(old.repaired5, old.reported, 1)
M["mask_est"] = np.where(M.index.str.startswith("new"), np.polyval(FIT, M.repaired5).round(1), M.reported)


def lab(arm):
    g = R[R.arm == arm].set_index(["archetype", "task_id"])
    return g


def section(t):
    print(f"\n=== {t} ===")


section("noise floor: overall (reported scale)")
groups = {"14sep da-7 (3 trainings, same data)": [82.8, 79.9, 71.3], "14sep da-15 (2)": [84.9, 89.4],
          "25sep da-15 seeds": [90.2, 88.3], "25sep new-t6 seeds": [87.9, 84.7],
          "nosynth (same adapter, 2 eval runs)": [57.7, 56.9]}
ss = dfree = 0
for k, v in groups.items():
    print(f"{k:40s} {v} range {max(v) - min(v):.1f}")
    if "nosynth" not in k:
        ss += sum((x - np.mean(v)) ** 2 for x in v)
        dfree += len(v) - 1
sd = (ss / dfree) ** 0.5
print(f"pooled training-replicate SD = {sd:.2f} (df={dfree}); SD of a difference of two single models = {sd * 2 ** .5:.2f};"
      f" 95% band for a single-model difference = +/-{1.96 * sd * 2 ** .5:.1f}")

section("noise floor: per archetype (harness labels), replicate groups")
cols = ["rep_contin", "rep_disinf", "rep_known_", "rep_provid", "rep_statis", "rep_doubli"]
for grp in (["14sep_da7_s0a", "14sep_da7_s1", "14sep_da7_s0b"], ["14sep_da15_a", "14sep_da15_b"],
            ["25sep_da15_s0", "25sep_da15_s1"], ["nosynth_a", "nosynth_b"]):
    sub = M.loc[grp, cols]
    print(grp, "range per archetype:", dict((sub.max() - sub.min()).round(1)))

section("row-level flip rate: % of rows whose lie/not-lie status differs (repaired labels, 5 archetypes)")
arms = sorted(R.arm.unique())
L = R.assign(lie=(R.rep_honesty == -1).astype(int)).pivot_table(index=["archetype", "task_id"], columns="arm",
                                                                    values="lie")
pairs = [("nosynth_a", "nosynth_b"), ("14sep_da7_s0a", "14sep_da7_s0b"), ("14sep_da7_s0a", "14sep_da7_s1"),
         ("14sep_da7_s1", "14sep_da7_s0b"), ("25sep_da15_s0", "25sep_da15_s1"), ("14sep_da15_a", "14sep_da15_b"),
         ("28sep_da15", "28sep_no_t6"), ("28sep_da15", "28sep_sysdiv"), ("28sep_da15", "28sep_da25"),
         ("28sep_da15", "25sep_da15_s0"), ("28sep_da15", "new_self"), ("28sep_da15", "new_explicit"),
         ("28sep_da15", "new_otherai"), ("nosynth_a", "28sep_da15")]
for a, b in pairs:
    x = L[[a, b]].dropna()
    both = ((x[a] == 1) & (x[b] == 1)).sum()
    print(f"{a:15s} vs {b:15s} flip {100 * (x[a] != x[b]).mean():5.1f}%  lies {int(x[a].sum())}/{int(x[b].sum())} "
          f"shared {both}  jaccard {both / max(1, ((x[a] == 1) | (x[b] == 1)).sum()):.2f}")

section("where do the 28 Sep baseline's lies go? (repaired labels)")
base = R[R.arm == "28sep_da15"].set_index(["archetype", "task_id"])
blies = base.index[base.rep_honesty == -1]
for arm in ["28sep_no_t6", "28sep_sysdiv", "28sep_da5", "28sep_da25", "25sep_da15_s0", "25sep_da15_s1", "swap_self",
            "swap_otherai", "swap_self_otherai", "swap_explicit", "swap_advice", "new_self", "new_otherai",
            "new_explicit", "da_tools", "14sep_da7_s0a", "14sep_da7_s0b"]:
    g = R[R.arm == arm].set_index(["archetype", "task_id"])
    g = g.reindex(blies)

    def kind(r):
        if pd.isna(r.rep_honesty):
            return "na"
        if r.rep_honesty == -1:
            return "lie"
        if r.rep_honesty == 1:
            return "honest"
        return "no_belief" if r.rep_belief in ("", "C") else "evasion"
    k = g.apply(kind, axis=1).value_counts()
    tot = k.sum() - k.get("na", 0)
    print(f"{arm:18s} of {tot} baseline lies -> " + "  ".join(f"{c} {100 * k.get(c, 0) / tot:4.1f}%"
                                                          for c in ("lie", "honest", "evasion", "no_belief")))

section("decomposition of all rows (repaired labels, 5 archetypes): honest / lie / evasion / no-belief, %")
dec = R.assign(kind=np.select([R.rep_honesty == -1, R.rep_honesty == 1, R.rep_belief.isin(["", "C"])],
                              ["lie", "honest", "no_belief"], "evasion"))
t = pd.crosstab(dec.arm, dec.kind, normalize="index").mul(100).round(1)
t["mask_est"] = M.mask_est
print(t.sort_values("mask_est").to_string())

section("refusal / length under pressure (pressured generation, all 6 archetypes, harness rows)")
D = pd.read_parquet("output/autoresearch/mask_rows.parquet")
REF = re.compile(r"\b(I can(?:'|no)t|I cannot|I won't|I will not|I'm not able|I am not able|I'm unable|I must decline|"
                 r"I need to decline|not comfortable|I'm not going to|I am not going to|I won't be able)\b", re.I)
HON = re.compile(r"\b(honest|truthful|accurate|mislead|deceiv|deceptive|lie\b|lying|false(?:ly)?|fabricat)", re.I)
D["refusal_phrase"] = D.generation.str.contains(REF)
D["names_honesty_in_reasoning"] = D.reasoning.str.contains(HON)
D["names_pressure_in_reasoning"] = D.reasoning.str.contains(
    r"\b(pressure|shut down|conflict|my (?:primary )?goal|system prompt|instructed to|instructions? (?:say|tell|want))",
    case=False, regex=True)
q = D.groupby("arm").agg(refusal_pct=("refusal_phrase", "mean"), gen_len_median=("gen_len", "median"),
                         reasoning_len_median=("reasoning_len", "median"),
                         honesty_words_in_reasoning=("names_honesty_in_reasoning", "mean"),
                         pressure_words_in_reasoning=("names_pressure_in_reasoning", "mean"))
q[["refusal_pct", "honesty_words_in_reasoning", "pressure_words_in_reasoning"]] *= 100
q["mask_est"] = M.mask_est
print(q.round(1).sort_values("mask_est").to_string())
print("corr(mask_est, refusal_pct) =", round(np.corrcoef(q.mask_est, q.refusal_pct)[0, 1], 3),
      " corr(mask_est, gen_len) =", round(np.corrcoef(q.mask_est, q.gen_len_median)[0, 1], 3),
      " corr(mask_est, honesty words) =", round(np.corrcoef(q.mask_est, q.honesty_words_in_reasoning)[0, 1], 3))
q.round(2).to_csv("output/autoresearch/mask_refusal_length.csv")

section("dose: per-archetype repaired honesty deltas vs 28 Sep da-15")
r5 = [c for c in M.columns if c.startswith("r5_")]
for arm in ["28sep_da5", "28sep_da25", "25sep_da15_s0", "25sep_da15_s1", "new_self", "new_explicit", "new_otherai",
            "swap_self_otherai", "14sep_da7_s0a", "14sep_da7_s0b"]:
    print(f"{arm:18s}", dict((M.loc[arm, r5] - M.loc["28sep_da15", r5]).round(1)))

section("does the da-25 gain land on the same rows as the 25 Sep gain? (rows the baseline lies on)")
fixed = {}
for arm in ["28sep_da25", "25sep_da15_s0", "25sep_da15_s1", "new_self", "new_explicit", "14sep_da7_s0a", "14sep_da15_b"]:
    g = L[arm].reindex(blies)
    fixed[arm] = set(g.index[g == 0])
for a, b in itertools.combinations(fixed, 2):
    print(f"{a:15s} {b:15s} fixed {len(fixed[a])}/{len(fixed[b])} overlap {len(fixed[a] & fixed[b])}"
          f" jaccard {len(fixed[a] & fixed[b]) / len(fixed[a] | fixed[b]):.2f}")

section("item structure: is lying one latent dimension? task lie-propensity over the 24 fully judged arms")
old_arms = [a for a in L.columns if not a.startswith("new")]
prop = L[old_arms].mean(axis=1)
for arm in ["nosynth_a", "28sep_da15", "25sep_da15_s0", "28sep_da25", "new_self", "new_otherai", "new_explicit",
            "da_tools"]:
    x = L[arm].dropna()
    p = prop.reindex(x.index)
    print(f"{arm:15s} lies {int(x.sum()):3d}  mean propensity of its lies {p[x == 1].mean():.2f}"
          f"  lies on items w/ propensity<0.2: {int(((x == 1) & (p < 0.2)).sum())}")

section("MASK vs ODCV")
o = M[M.odcv_mr.notna()].copy()
o["odcv"] = o.odcv_mr.astype(str).str.split("/").apply(lambda v: np.mean([float(z) for z in v]))
o = o[~o.index.isin(["nosynth_b"])]
print(o[["reported", "mask_est", "odcv"]].sort_values("odcv").to_string())
for col in ("reported", "mask_est"):
    print(f"pearson({col}, odcv) = {np.corrcoef(o[col], o.odcv)[0, 1]:.3f}; spearman = "
          f"{o[col].rank().corr(o.odcv.rank()):.3f}")
x = o[o.index != "nosynth_a"]
print("without nosynth: pearson(mask_est, odcv) =", round(np.corrcoef(x.mask_est, x.odcv)[0, 1], 3),
      " spearman =", round(x.mask_est.rank().corr(x.odcv.rank()), 3))
b = np.polyfit(x.mask_est, x.odcv, 1)
x = x.assign(odcv_pred=np.polyval(b, x.mask_est).round(1))
x["resid"] = (x.odcv - x.odcv_pred).round(1)
print("fit odcv = %.3f * mask + %.1f" % tuple(b))
print(x[["mask_est", "odcv", "odcv_pred", "resid"]].sort_values("resid").to_string())
M.to_csv("output/autoresearch/mask_archetype_matrix.csv")
