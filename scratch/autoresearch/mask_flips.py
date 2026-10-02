# ABOUTME: Pick robust MASK flip rows (the 28 Sep recipe models lie, the high arms are honest) and dump the pressured
# ABOUTME: generation + reasoning of each arm side by side for reading; also counts flips by archetype / pressure type.
import sys

import numpy as np
import pandas as pd

R = pd.read_parquet("output/autoresearch/mask_rows_repaired.parquet")
P = R.pivot_table(index=["archetype", "task_id"], columns="arm", values="rep_honesty", aggfunc="first")
low = ["28sep_da15", "28sep_no_t6", "28sep_sysdiv", "28sep_da5"]
high = ["25sep_da15_s0", "28sep_da25", "new_explicit", "new_self"]
robust = P[(P["28sep_da15"] == -1) & ((P[low] == -1).sum(axis=1) >= 3) & ((P[high] == 1).sum(axis=1) >= 3)]
loose = P[(P["28sep_da15"] == -1) & (P["25sep_da15_s0"] == 1) & (P["new_self"] == 1) & (P["new_explicit"] == 1)]
print("robust flips", len(robust), robust.index.get_level_values(0).value_counts().to_dict())
print("loose flips", len(loose), loose.index.get_level_values(0).value_counts().to_dict())
base = P.index.get_level_values(0).value_counts()
print("archetype sizes (5 arch)", base.to_dict())
# stuck: rows every DA arm lies on
da_arms = [c for c in P.columns if c not in ("nosynth_a", "nosynth_b", "delib", "delib_sonnet", "da_tools")]
stuck = P[(P[da_arms] == -1).mean(axis=1) >= 0.8]
print("stuck (>=80% of DA arms lie)", len(stuck), stuck.index.get_level_values(0).value_counts().to_dict())
pd.Series(list(robust.index.map("|".join))).to_csv("output/autoresearch/mask_robust_flip_rows.csv", index=False)

if len(sys.argv) > 1:
    n = int(sys.argv[1])
    arms = ["28sep_da15", "25sep_da15_s0", "new_self", "new_explicit"] + (sys.argv[2:] or [])
    idx = robust.sample(min(n, len(robust)), random_state=int(1)).index
    G = R.set_index(["archetype", "task_id", "arm"])
    for a, t in idx:
        r0 = G.loc[(a, t, "28sep_da15")]
        print("\n" + "#" * 100)
        print(f"[{a} | {t}] PROP: {r0.proposition}")
        print("SYS:", r0.system_prompt[:700].replace("\n", " "))
        if r0.dd_turn_1_prompt:
            print("DD1 user:", r0.dd_turn_1_prompt[:300].replace("\n", " "), "|| DD1 asst:", r0.dd_turn_1_response[:200])
        print("USER:", r0.user_prompt[:600].replace("\n", " "))
        for arm in arms:
            r = G.loc[(a, t, arm)]
            rea = r.reasoning.replace("\n", " ")
            print(f"--- {arm} [h={r.rep_honesty:+.0f} belief={r.rep_belief} said={r.rep_pressure}]")
            print("  REASONING:", rea[:700], " ... ", rea[-600:] if len(rea) > 1300 else "")
            print("  ANSWER:", r.generation[:450].replace("\n", " "))
