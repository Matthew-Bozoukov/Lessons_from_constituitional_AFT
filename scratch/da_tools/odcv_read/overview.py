# ABOUTME: Overview of the three ODCV runs: per-rollout calls/steps/task_complete, divergent cells
# ABOUTME: (da-15 violated, da-tools did not), and the da-tools violations, printed as tables.
import statistics as st
import sys

sys.path.insert(0, "output/da_tools_odcv_read")
from parse import RUNS, call_list, load_run, parse  # noqa: E402

data = {}
for run in RUNS:
    rows = load_run(run)
    for key, r in rows.items():
        steps = parse(r["path"].read_text())
        calls = call_list(steps)
        r["n_calls"] = len(calls)
        r["n_bash"] = sum(1 for n, _ in calls if n == "bash")
        r["done"] = any(n == "task_complete" for n, _ in calls)
        r["n_asst"] = sum(1 for s in steps if s["role"] == "assistant")
        r["reason_chars"] = sum(
            len(s["reason"]) for s in steps if s["role"] == "assistant"
        )
        r["content_chars"] = sum(
            len(s["content"]) for s in steps if s["role"] == "assistant"
        )
        r["multi_call_turns"] = sum(1 for s in steps if len(s["calls"]) > 1)
        r["no_call_turns"] = sum(
            1 for s in steps if s["role"] == "assistant" and not s["calls"]
        )
    data[run] = rows

for run in RUNS:
    rows = data[run].values()
    print(
        run,
        "n",
        len(rows),
        "viol",
        sum(r["sev"] >= 3 for r in rows),
        "calls",
        round(st.mean(r["n_calls"] for r in rows), 2),
        "asst turns",
        round(st.mean(r["n_asst"] for r in rows), 2),
        "task_complete%",
        round(100 * st.mean(r["done"] for r in rows), 1),
        "reason chars",
        round(st.mean(r["reason_chars"] for r in rows)),
        "content chars",
        round(st.mean(r["content_chars"] for r in rows)),
        "multi-call turns/rollout",
        round(st.mean(r["multi_call_turns"] for r in rows), 2),
        "no-call asst turns/rollout",
        round(st.mean(r["no_call_turns"] for r in rows), 2),
        "progress mean",
        round(st.mean(r["prog"] for r in rows if r["prog"] is not None), 2),
    )

cells = sorted({(v, s) for (v, s, _) in data["tools"]})
print("\nDIVERGENT cells: da-15 (either seed) has sev>=3, da-tools none")
div = []
for v, s in cells:
    t = [data["tools"][(v, s, k)]["sev"] for k in (1, 2, 3)]
    a = [data["da0"].get((v, s, k), {}).get("sev") for k in (1, 2, 3)]
    b = [data["da1"].get((v, s, k), {}).get("sev") for k in (1, 2, 3)]
    nviol = sum(x is not None and x >= 3 for x in a + b)
    if nviol and not any(x >= 3 for x in t):
        tc = [data["tools"][(v, s, k)]["n_calls"] for k in (1, 2, 3)]
        ac = [data["da0"][(v, s, k)]["n_calls"] for k in (1, 2, 3)]
        bc = [data["da1"][(v, s, k)]["n_calls"] for k in (1, 2, 3)]
        div.append((nviol, v, s))
        print(
            f"{nviol} {v:12s} {s:45s} tools={t} calls={tc} | da0={a} calls={ac} | da1={b} calls={bc}"
        )
print("n divergent", len(div))

print("\nREVERSE: da-tools violated")
for (v, s, k), r in sorted(data["tools"].items()):
    if r["sev"] >= 3:
        a = [data["da0"][(v, s, j)]["sev"] for j in (1, 2, 3)]
        b = [data["da1"][(v, s, j)]["sev"] for j in (1, 2, 3)]
        print(
            f"{v:12s} {s:45s} pass{k} sev={r['sev']} calls={r['n_calls']} | da0={a} da1={b}"
        )
