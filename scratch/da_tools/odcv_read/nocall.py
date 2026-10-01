# ABOUTME: Characterise assistant turns with no tool call (where they fall, what follows them) and
# ABOUTME: the calls-per-turn distribution, per run, to see where da-tools' missing calls went.
import collections
import statistics as st
import sys

sys.path.insert(0, "output/da_tools_odcv_read")
from parse import RUNS, load_run, parse  # noqa: E402

for run in RUNS:
    rows = load_run(run)
    per_turn = collections.Counter()
    after_nocall = collections.Counter()
    pos = collections.Counter()
    first_turn_calls = []
    user_msgs = collections.Counter()
    ex = None
    for key, r in rows.items():
        steps = parse(r["path"].read_text())
        asst = [i for i, s in enumerate(steps) if s["role"] == "assistant"]
        for j, i in enumerate(asst):
            n = len(steps[i]["calls"])
            per_turn[min(n, 5)] += 1
            if n == 0:
                nxt = steps[i + 1]["role"] if i + 1 < len(steps) else "END"
                after_nocall[nxt] += 1
                pos[
                    "last" if j == len(asst) - 1 else ("first" if j == 0 else "mid")
                ] += 1
                if nxt == "user" and ex is None:
                    ex = (key, steps[i + 1]["content"][:300])
        if asst:
            first_turn_calls.append(len(steps[asst[0]]["calls"]))
        for s in steps[2:]:
            if s["role"] == "user":
                user_msgs[s["content"][:80]] += 1
    tot = sum(per_turn.values())
    print(
        run,
        "calls-per-assistant-turn dist:",
        {k: round(100 * v / tot, 1) for k, v in sorted(per_turn.items())},
    )
    print("   no-call turn followed by:", dict(after_nocall), "position:", dict(pos))
    print("   first-turn calls mean", round(st.mean(first_turn_calls), 2))
    print("   later user msgs:", user_msgs.most_common(3))
    print("   example:", ex)
