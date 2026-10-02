# ABOUTME: Prints sampled DA rows (system, user, head of reasoning and answer) from one row-diff group for close reading,
# ABOUTME: optionally filtered by a row-judge field. Usage: show_rows.py <group> [n] [field=value] [rchars] [achars]
import json, random, sys
from pathlib import Path
OUT = Path("output/autoresearch/rowdiff")
g = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
filt = sys.argv[3] if len(sys.argv) > 3 and "=" in sys.argv[3] and "!" not in sys.argv[3] else None
rch = int(sys.argv[4]) if len(sys.argv) > 4 else 500; ach = int(sys.argv[5]) if len(sys.argv) > 5 else 1500
lab = {json.loads(l)["rid"]: json.loads(l) for l in open(OUT / "judged.jsonl")}
mem = {(m["group"], m["scenario_id"]): m["rid"] for m in map(json.loads, open(OUT / "judged_membership.jsonl"))}
rows = [r for r in map(json.loads, open(OUT / "rows.jsonl")) if r["group"] == g]
rows = [r for r in rows if (g, r["scenario_id"]) in mem]
if filt:
    k, v = filt.split("=")
    rows = [r for r in rows if str(lab[mem[(g, r["scenario_id"])]][k]) == v]
random.Random(7).shuffle(rows)
for r in rows[:n]:
    L = lab[mem[(g, r["scenario_id"])]]
    print("=" * 100); print(r["scenario_id"], r["corpus"], {k: L[k] for k in ("norm", "clarity", "actor", "ai_role", "kpi_pressure", "stance", "ask_type")})
    print("[SYS]", r["system"]); print("[USER]", r["user"]); print("[REASONING]", r["reasoning"][:rch], "..."); print("[ANSWER]", r["answer"][:ach], "...")
