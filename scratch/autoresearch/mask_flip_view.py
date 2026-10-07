# ABOUTME: Compact side-by-side viewer for robust MASK flip rows (28 Sep lies, high arms honest): reasoning head/tail
# ABOUTME: and answer per arm, for reading transcripts; usage: mask_flip_view.py <start> <count> [arms...]
import sys

import pandas as pd

R = pd.read_parquet("output/autoresearch/mask_rows_repaired.parquet")
rows = pd.read_csv("output/autoresearch/mask_robust_flip_rows.csv").iloc[:, 0].str.split("|").tolist()
start, count = int(sys.argv[1]), int(sys.argv[2])
arms = sys.argv[3:] or ["28sep_da15", "25sep_da15_s0", "new_self", "new_explicit"]
G = R.set_index(["archetype", "task_id", "arm"])
for a, t in rows[start:start + count]:
    r0 = G.loc[(a, t, arms[0])]
    print("\n" + "#" * 60, f"\n[{a} | {t}] PROP: {r0.proposition}")
    print("SYS:", r0.system_prompt[:350].replace("\n", " "))
    if r0.dd_turn_1_prompt:
        print("DD1:", r0.dd_turn_1_prompt[:200].replace("\n", " "), "|| A1:", r0.dd_turn_1_response[:150])
    print("USER:", r0.user_prompt[:350].replace("\n", " "))
    for arm in arms:
        r = G.loc[(a, t, arm)]
        rea = r.reasoning.replace("\n", " ")
        print(f"--- {arm} [h={r.rep_honesty:+.0f}] R: {rea[:330]} ... {rea[-260:]}")
        print(f"    A: {r.generation.strip()[:300].replace(chr(10), ' ')}")
