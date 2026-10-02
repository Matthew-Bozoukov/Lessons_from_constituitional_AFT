# ABOUTME: Is the da-tools gain explained by fewer actions? Violation rate within tool-call bands per arm, calls of
# ABOUTME: violating vs clean rollouts within the same cells, and the call gap restricted to clean rollouts.
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, "output/da_tools_odcv_read")
from parse import RUNS, call_list, load_run, parse  # noqa: E402

data = {}
for run in RUNS:
    rows = load_run(run)
    for key, r in rows.items():
        r["calls"] = len(call_list(parse(r["path"].read_text())))
    data[run] = rows

bands = [(0, 5), (6, 8), (9, 11), (12, 15), (16, 999)]
print("violation rate by tool-call band (n in band)")
for run in RUNS:
    out = []
    for lo, hi in bands:
        b = [r for r in data[run].values() if lo <= r["calls"] <= hi]
        v = sum(r["sev"] >= 3 for r in b)
        out.append(f"{lo}-{hi if hi < 999 else '+'}: {v}/{len(b)}")
    print(" ", run, " | ".join(out))

print(
    "\nda-15 violating vs clean rollouts in the SAME cells (cells with >=1 da-15 violation)"
)
for run in ("da0", "da1"):
    cells = defaultdict(list)
    for (v, s, k), r in data[run].items():
        cells[(v, s)].append(r)
    diffs = []
    for c, rs in cells.items():
        bad = [r["calls"] for r in rs if r["sev"] >= 3]
        good = [r["calls"] for r in rs if r["sev"] < 3]
        if bad and good:
            diffs.append(st.mean(bad) - st.mean(good))
    print(
        f"  {run}: mixed cells {len(diffs)}, violating minus clean calls: mean {st.mean(diffs):+.2f}"
    )

print("\ncall gap restricted to CLEAN rollouts (sev<3) in all arms, paired by cell")
for other in ("da0", "da1"):
    diffs = []
    for v in ("mandated", "incentivized"):
        for s in {k[1] for k in data["tools"] if k[0] == v}:
            t = [
                data["tools"][(v, s, k)]["calls"]
                for k in (1, 2, 3)
                if data["tools"][(v, s, k)]["sev"] < 3
            ]
            o = [
                data[other][(v, s, k)]["calls"]
                for k in (1, 2, 3)
                if data[other][(v, s, k)]["sev"] < 3
            ]
            if t and o:
                diffs.append(st.mean(t) - st.mean(o))
    print(
        f"  tools - {other}: {st.mean(diffs):+.2f} calls over {len(diffs)} cells; fewer in {sum(d < 0 for d in diffs)}, more in {sum(d > 0 for d in diffs)}"
    )
print(
    "  mean calls, clean only:",
    {
        run: round(st.mean(r["calls"] for r in data[run].values() if r["sev"] < 3), 2)
        for run in RUNS
    },
)
print(
    "  mean calls, violating only:",
    {
        run: round(st.mean(r["calls"] for r in data[run].values() if r["sev"] >= 3), 2)
        for run in RUNS
    },
)
