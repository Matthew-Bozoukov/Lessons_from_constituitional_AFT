# ABOUTME: Per scenario, how DA+tools avoids harm: zero-call refusals vs engaged non-harm vs engaged harm.
# ABOUTME: Run: uv run python scratch/dh_holistic/zero_call_by_scenario.py
import collections, glob, json, os
d = "da-tools-15"
c = collections.defaultdict(collections.Counter)
for f in glob.glob(f"output/dh_holistic/{d}/results/episodes/*.json"):
    e = json.load(open(f, encoding="utf-8"))
    if "metrics" not in e:
        continue
    t = json.load(open(f"output/dh_holistic/{d}/rollouts/{os.path.basename(f)}", encoding="utf-8"))
    nc = sum(len(m.get("tool_calls") or []) for m in t["messages"] if m.get("role") == "assistant")
    k = "zero_call" if nc == 0 else ("engaged_harm" if e["metrics"]["harmful_action"] else "engaged_safe")
    c[e["scenario"]][k] += 1
for s in sorted(c):
    print(f"{s:22s}", dict(c[s]))
