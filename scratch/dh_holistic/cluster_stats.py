# ABOUTME: DA vs DA+tools on Subagents Comply More with scenario as the unit: cluster bootstrap, sign test,
# ABOUTME: and worst-case bounds that score every invalid episode against DA+tools. Run: uv run python scratch/dh_holistic/cluster_stats.py
import glob, json, random
from math import comb

def load(d):
    out = {}
    for f in glob.glob(f"output/dh_holistic/{d}/results/episodes/*.json"):
        e = json.load(open(f, encoding="utf-8"))
        out[e["id"]] = e
    return out

da, dt = load("da-15"), load("da-tools-15")
scen = sorted({e["scenario"] for e in da.values()} - {"benchmark_rescore"})
def rate(src, s):
    v = [e["metrics"]["harmful_action"] for e in src.values() if e["scenario"] == s and "metrics" in e]
    return sum(v) / len(v)
diffs = {s: rate(dt, s) - rate(da, s) for s in scen}
neg = sum(d < 0 for d in diffs.values())
print(f"scenarios where DA+tools harms less: {neg}/{len(scen)}; sign test p = {sum(comb(len(scen), k) for k in range(neg, len(scen)+1)) / 2**len(scen) * 2:.4f} (two-sided)")
random.seed(0)
boot = sorted(sum(diffs[random.choice(scen)] for _ in scen) / len(scen) for _ in range(20000))
m = sum(diffs.values()) / len(scen)
print(f"mean per-scenario difference {100*m:.1f} pp, scenario-cluster bootstrap 95% [{100*boot[500]:.1f}, {100*boot[19500]:.1f}]")
# worst case: every invalid DA+tools episode harmful, every invalid DA episode harmless, all 12 scenarios x 9
wc_t = sum(1 for e in dt.values() if "metrics" not in e or e["metrics"]["harmful_action"])
wc_d = sum(1 for e in da.values() if "metrics" in e and e["metrics"]["harmful_action"])
print(f"worst case over all 108: DA+tools <= {wc_t}/108 = {100*wc_t/108:.0f}%, DA >= {wc_d}/108 = {100*wc_d/108:.0f}%")
