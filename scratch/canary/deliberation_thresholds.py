# ABOUTME: P(canary | turn has >= N reasoning sentences) for N in 1,3,5,8, per arm and eval, from the
# ABOUTME: cached deliberation labels. Run: uv run python scratch/canary/deliberation_thresholds.py
import json, sys, re
from pathlib import Path

sys.path.insert(0, ".")
from scratch.canary.deliberation import RUNS, turns, CANARY, wilson

lab = {}
for line in open("output/canary/deliberation_labels.jsonl"):
    r = json.loads(line)
    if "error" not in r:
        lab[r["key"]] = r
rows = []
for (arm, ev), (kind, root) in RUNS.items():
    for tid, text, ep in turns(kind, Path(root)):
        k = f"{arm}|{ev}|{tid}"
        if k in lab:
            rows.append(
                (
                    arm,
                    ev,
                    bool(re.search(CANARY, text)),
                    lab[k]["n_reasoning"],
                    lab[k]["n_sentences"],
                )
            )
print(
    f"{'':22s}"
    + "".join(f"{'>=' + str(t) + ' reasoning sents':>28s}" for t in (1, 3, 5, 8))
)
for ev in ("MASK", "ODCV", "Hospital"):
    for arm in ("DA", "DA + tools"):
        sub = [r for r in rows if r[0] == arm and r[1] == ev]
        cells = []
        for t in (1, 3, 5, 8):
            s = [r for r in sub if r[3] >= t]
            k = sum(r[2] for r in s)
            cells.append(f"{100 * k / len(s):5.1f}% ({k}/{len(s)})" if s else "-")
        print(f"{ev:9s}{arm:13s}" + "".join(f"{c:>28s}" for c in cells))
import statistics

for ev in ("MASK", "ODCV", "Hospital"):
    for arm in ("DA", "DA + tools"):
        sub = [r[3] for r in rows if r[0] == arm and r[1] == ev]
        print(ev, arm, "median reasoning sentences/turn", statistics.median(sub))
