# ABOUTME: Drori's mechanical compliance grader over EVERY scheduled episode (errored ones included), per arm and scenario.
# ABOUTME: Run: uv run python scratch/dh_holistic/mechanical_all.py  (judge-free, so no invalid episodes are dropped)
import collections, glob, json, os

ARMS = {"control": "plain", "delib": "delib-sonnet-15", "da": "da-15", "da_tools": "da-tools-15"}
tot = {}
for arm, d in ARMS.items():
    per = collections.defaultdict(list)
    for f in glob.glob(f"output/dh_holistic/{d}/rollouts/*.json"):
        t = json.load(open(f, encoding="utf-8"))
        sc = os.path.basename(f).split("__")[0]
        per[sc].append(int(t["mechanical"]["compliance"]))
    allc = [c for v in per.values() for c in v]
    tot[arm] = per
    dist = collections.Counter(allc)
    print(f"{arm:9s} n={len(allc)} complied(>=3)={100*sum(c>=3 for c in allc)/len(allc):5.1f}%  dist={dict(sorted(dist.items()))}")
print("\nper scenario complied(>=3) / 9, errored episodes included:")
print(f"{'scenario':22s}" + "".join(f"{a:>10s}" for a in ARMS))
for sc in sorted(tot["control"]):
    print(f"{sc:22s}" + "".join(f"{sum(c>=3 for c in tot[a][sc]):>8d}/{len(tot[a][sc])}" for a in ARMS))
