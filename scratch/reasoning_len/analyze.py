# ABOUTME: Measure inference-time reasoning vs answer length per arm on matched sycophancy + debate evals.
# ABOUTME: Parses the published rollout markdown, splits think from answer, and prints per-arm distributions.
import json
import os
import re
import statistics as st
from collections import defaultdict

ROOT = "data/reasoning_len"
ARMS = ["MO_DA", "MO_CR", "MO_PC", "CTRL", "BASE"]
HDR = re.compile(r"^## (.+)$", re.M)
FENCE_OPEN = re.compile(r"^```[A-Za-z0-9_+-]*$")


def sections(text):
    out, marks = {}, list(HDR.finditer(text))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out[m.group(1).strip()] = text[m.end() : end]
    return out


def unfence(s):
    lines = s.strip("\n").split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if len(lines) >= 2 and FENCE_OPEN.match(lines[0]) and lines[-1].strip() == "```":
        lines = lines[1:-1]
    return "\n".join(lines).strip()


def records(evl, arm):
    p = f"{ROOT}/{evl}/{arm}/records.jsonl"
    return [json.loads(l) for l in open(p) if l.strip()]


def rollout(evl, arm, uid):
    p = f"{ROOT}/{evl}/{arm}/rollouts/{uid.replace(':', '_')}.md"
    return sections(open(p).read())


def q(v, p):
    if not v:
        return float("nan")
    v = sorted(v)
    k = (len(v) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


SPECS = {
    "sycophancy": {
        "turns": [
            (
                "t1",
                "Assistant reasoning (turn 1)",
                "Assistant reply (turn 1)",
                "first_think_chars",
                "first_error",
                "first_truncated",
            ),
            (
                "t2",
                "Assistant reasoning (turn 2)",
                "Assistant reply (turn 2)",
                "second_think_chars",
                "second_error",
                "second_truncated",
            ),
        ]
    },
    "debate-speeches": {
        "turns": [
            ("t1", "Model reasoning", "Model reply", "think_chars", "error", None)
        ]
    },
}

rows = []
excerpts = defaultdict(dict)
mismatch = defaultdict(int)

for evl, spec in SPECS.items():
    for arm in ARMS:
        recs = records(evl, arm)
        data = {
            t[0]: {"r": [], "a": [], "ratio": [], "empty": 0, "err": 0, "trunc": 0}
            for t in spec["turns"]
        }
        for rec in recs:
            secs = rollout(evl, arm, rec["uid"])
            for tag, rk, ak, ck, ek, tk in spec["turns"]:
                d = data[tag]
                if rec.get(ek):
                    d["err"] += 1
                    continue
                if tk and rec.get(tk):
                    d["trunc"] += 1
                reasoning = unfence(secs[rk])
                answer = unfence(secs[ak])
                if abs(len(reasoning) - rec[ck]) > 2:
                    mismatch[(evl, arm, tag)] += 1
                d["r"].append(len(reasoning))
                d["a"].append(len(answer))
                if len(reasoning) == 0:
                    d["empty"] += 1
                if len(answer) > 0:
                    d["ratio"].append(len(reasoning) / len(answer))
                if tag == "t1" and arm not in excerpts[evl] and 600 < len(reasoning):
                    pass
        for tag, *_ in spec["turns"]:
            d = data[tag]
            r, a = d["r"], d["a"]
            rows.append(
                dict(
                    eval=evl,
                    arm=arm,
                    turn=tag,
                    n=len(r),
                    err=d["err"],
                    trunc=d["trunc"],
                    empty=d["empty"],
                    empty_pct=100.0 * d["empty"] / max(len(r), 1),
                    r_med=st.median(r),
                    r_mean=st.mean(r),
                    r_p25=q(r, 0.25),
                    r_p75=q(r, 0.75),
                    a_med=st.median(a),
                    a_mean=st.mean(a),
                    ratio_med=st.median(d["ratio"]) if d["ratio"] else float("nan"),
                )
            )

print("mismatches vs records think_chars:", dict(mismatch) or "none")
hdr = f"{'eval':16} {'arm':6} {'t':3} {'n':>4} {'err':>4} {'trunc':>5} {'empty%':>7} {'r_med':>7} {'r_mean':>8} {'r_p25':>7} {'r_p75':>7} {'a_med':>7} {'ratio':>7}"
print(hdr)
print("-" * len(hdr))
for row in rows:
    print(
        f"{row['eval']:16} {row['arm']:6} {row['turn']:3} {row['n']:4d} {row['err']:4d} {row['trunc']:5d} "
        f"{row['empty_pct']:7.1f} {row['r_med']:7.0f} {row['r_mean']:8.0f} {row['r_p25']:7.0f} {row['r_p75']:7.0f} "
        f"{row['a_med']:7.0f} {row['ratio_med']:7.2f}"
    )

os.makedirs("output/reasoning_len", exist_ok=True)
json.dump(rows, open("output/reasoning_len/summary.json", "w"), indent=2)
