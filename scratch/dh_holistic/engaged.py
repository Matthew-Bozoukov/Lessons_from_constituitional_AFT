# ABOUTME: Harm conditional on engagement: does DA+tools still refuse once it has actually worked the environment?
# ABOUTME: Run: uv run python scratch/dh_holistic/engaged.py  (buckets by tool calls made before the episode ended)
import collections, glob, json, os

ARMS = {"control": "plain", "delib": "delib-sonnet-15", "da": "da-15", "da_tools": "da-tools-15"}
B = [(0, 0, "0 calls"), (1, 5, "1-5"), (6, 10**6, "6+")]
for arm, d in ARMS.items():
    c = collections.defaultdict(lambda: [0, 0, 0, 0])  # n, harm, refusal, benign
    for f in glob.glob(f"output/dh_holistic/{d}/results/episodes/*.json"):
        e = json.load(open(f, encoding="utf-8"))
        if "metrics" not in e:
            continue
        t = json.load(open(f"output/dh_holistic/{d}/rollouts/{os.path.basename(f)}", encoding="utf-8"))
        nc = sum(len(m.get("tool_calls") or []) for m in t["messages"] if m.get("role") == "assistant")
        lab = next(l for lo, hi, l in B if lo <= nc <= hi)
        x = c[lab]; m = e["metrics"]
        x[0] += 1; x[1] += m["harmful_action"]; x[2] += m["explicit_refusal"]; x[3] += m["benign_complete"]
    print(arm, "  ".join(f"[{l}] n={c[l][0]} harm={c[l][1]} refuse={c[l][2]} useful={c[l][3]}" for *_, l in B))
