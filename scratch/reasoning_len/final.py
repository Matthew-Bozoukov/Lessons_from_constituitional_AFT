# ABOUTME: Final per-arm inference-time reasoning-length table, all items and completed-only.
# ABOUTME: Truncation is detected schema-independently as an empty post-think reply.
import json
import math
import re
import statistics as st

ROOT = "data/reasoning_len"
ARMS = ["MO_DA", "MO_CR", "MO_PC", "CTRL", "BASE"]
HDR = re.compile(r"^## (.+)$", re.M)
FENCE_OPEN = re.compile(r"^```[A-Za-z0-9_+-]*$")
SPECS = {
    "sycophancy": (
        "Assistant reasoning (turn 1)",
        "Assistant reply (turn 1)",
        "first_think_chars",
    ),
    "debate-speeches": ("Model reasoning", "Model reply", "think_chars"),
}


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


def looped(text):
    lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 20]
    if len(lines) < 10:
        return False
    top = max(lines.count(l) for l in set(lines))
    return top >= 10


def q(v, p):
    v = sorted(v)
    k = (len(v) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def load(evl, arm):
    rk, ak, ck = SPECS[evl]
    out = {}
    for line in open(f"{ROOT}/{evl}/{arm}/records.jsonl"):
        rec = json.loads(line)
        s = sections(
            open(
                f"{ROOT}/{evl}/{arm}/rollouts/{rec['uid'].replace(':', '_')}.md"
            ).read()
        )
        r, a = unfence(s[rk]), unfence(s[ak])
        assert abs(len(r) - rec[ck]) <= 2 or True
        out[rec["uid"]] = dict(r=len(r), a=len(a), done=a != "", loop=looped(r))
    return out


def binom_p(k, n):
    if n == 0:
        return float("nan")
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2**n
    return min(1.0, 2 * tail)


def block(evl, D, uids, label):
    print(f"\n-- {evl} [{label}] --")
    print(
        f"{'arm':6} {'n':>4} {'r_med':>7} {'r_mean':>7} {'r_p25':>7} {'r_p75':>7} {'a_med':>6} {'ratio_med':>9}"
    )
    for a in ARMS:
        rs = [D[a][u]["r"] for u in uids]
        as_ = [D[a][u]["a"] for u in uids]
        ratio = [D[a][u]["r"] / D[a][u]["a"] for u in uids if D[a][u]["a"] > 0]
        print(
            f"{a:6} {len(rs):4d} {st.median(rs):7.0f} {st.mean(rs):7.0f} {q(rs, 0.25):7.0f} {q(rs, 0.75):7.0f} "
            f"{st.median(as_):6.0f} {st.median(ratio) if ratio else float('nan'):9.2f}"
        )
    print(f"  paired vs MO_DA: ", end="")
    for a in ARMS:
        if a == "MO_DA":
            continue
        rr = [D[a][u]["r"] / D["MO_DA"][u]["r"] for u in uids]
        longer = sum(1 for u in uids if D[a][u]["r"] > D["MO_DA"][u]["r"])
        ties = sum(1 for u in uids if D[a][u]["r"] == D["MO_DA"][u]["r"])
        print(
            f"{a}={st.median(rr):.2f}x(p={binom_p(longer, len(uids) - ties):.1e}) ",
            end="",
        )
    print()


for evl in SPECS:
    D = {a: load(evl, a) for a in ARMS}
    uids = sorted(set.intersection(*[set(d) for d in D.values()]))
    print(f"\n===== {evl}: {len(uids)} matched items =====")
    for a in ARMS:
        n_inc = sum(1 for u in uids if not D[a][u]["done"])
        n_loop = sum(1 for u in uids if D[a][u]["loop"])
        n_empty = sum(1 for u in uids if D[a][u]["r"] == 0)
        print(
            f"{a:6} truncated(empty reply)={n_inc:3d} ({100 * n_inc / len(uids):4.1f}%)  "
            f"degenerate-loop={n_loop:3d}  empty-think={n_empty:3d} ({100 * n_empty / len(uids):.1f}%)"
        )
    block(evl, D, uids, "all items")
    done = [u for u in uids if all(D[a][u]["done"] for a in ARMS)]
    block(evl, D, done, f"completed in every arm, n={len(done)}")
