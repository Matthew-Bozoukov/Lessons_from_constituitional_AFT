# ABOUTME: Is a low harm rate real refusal or tool avoidance? Per arm: tool calls, writes, assistant turns,
# ABOUTME: one-message endings, and outcomes among non-harmful episodes. Run: uv run python scratch/dh_holistic/activity.py
import collections, glob, json, os, statistics as st

ARMS = {"control": "plain", "delib": "delib-sonnet-15", "da": "da-15", "da_tools": "da-tools-15"}
for arm, d in ARMS.items():
    calls, writes, turns, one, nonharm = [], [], [], 0, collections.Counter()
    for f in glob.glob(f"output/dh_holistic/{d}/rollouts/*.json"):
        t = json.load(open(f, encoding="utf-8"))
        msgs = t.get("messages", [])
        a = [m for m in msgs if m.get("role") == "assistant"]
        nc = sum(len(m.get("tool_calls") or []) for m in a)
        calls.append(nc); turns.append(len(a)); writes.append(len(t["state"]["fs"]["writes"]))
        one += len(a) <= 1
        ep = f"output/dh_holistic/{d}/results/episodes/{os.path.basename(f)}"
        if os.path.exists(ep):
            e = json.load(open(ep, encoding="utf-8"))
            if "metrics" in e and not e["metrics"]["harmful_action"]:
                m = e["metrics"]
                nonharm["n"] += 1
                for k in ("benign_complete", "explicit_refusal", "safe_substitution", "justified_deferral", "concern_any_human"):
                    nonharm[k] += m[k]
                nonharm["no_write"] += len(t["state"]["fs"]["writes"]) == 0
    print(f"{arm:9s} median tool calls {st.median(calls):5.1f}  median writes {st.median(writes):3.0f}  "
          f"median asst turns {st.median(turns):4.0f}  <=1 asst msg {one}/{len(calls)}  zero-write eps {sum(w==0 for w in writes)}")
    n = nonharm.pop("n", 0)
    print(f"          among {n} NON-harmful valid eps: " + ", ".join(f"{k}={v}" for k, v in nonharm.items()))
