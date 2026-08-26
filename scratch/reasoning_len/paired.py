# ABOUTME: Paired (same-item, temperature-0) comparison of inference-time reasoning length across arms.
# ABOUTME: Sign test + median per-item ratio vs MO_DA, plus truncation/empty-think diagnostics.
import json
import math
import re
import statistics as st

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


SPECS = {
    "sycophancy": (
        "Assistant reasoning (turn 1)",
        "Assistant reply (turn 1)",
        "first_truncated",
    ),
    "debate-speeches": ("Model reasoning", "Model reply", None),
}


def load(evl, arm):
    rk, ak, tk = SPECS[evl]
    out = {}
    for line in open(f"{ROOT}/{evl}/{arm}/records.jsonl"):
        rec = json.loads(line)
        s = sections(
            open(
                f"{ROOT}/{evl}/{arm}/rollouts/{rec['uid'].replace(':', '_')}.md"
            ).read()
        )
        out[rec["uid"]] = dict(
            r=len(unfence(s[rk])),
            a=len(unfence(s[ak])),
            trunc=bool(rec.get(tk)) if tk else False,
        )
    return out


def binom_p(k, n):
    """two-sided sign test against p=0.5"""
    if n == 0:
        return float("nan")
    c = lambda a, b: math.comb(a, b)
    tail = sum(c(n, i) for i in range(0, min(k, n - k) + 1)) / 2**n
    return min(1.0, 2 * tail)


for evl in SPECS:
    D = {a: load(evl, a) for a in ARMS}
    uids = sorted(set.intersection(*[set(d) for d in D.values()]))
    print(f"\n===== {evl}  (matched items n={len(uids)}) =====")
    print(
        f"{'arm':6} {'min_r':>6} {'zero_r':>6} {'trunc':>6} {'r_med':>7} {'r_gmean':>8} {'a_med':>6} {'a_zero':>6}"
    )
    for a in ARMS:
        rs = [D[a][u]["r"] for u in uids]
        as_ = [D[a][u]["a"] for u in uids]
        gm = math.exp(st.mean(math.log(max(x, 1)) for x in rs))
        print(
            f"{a:6} {min(rs):6d} {sum(1 for x in rs if x == 0):6d} {sum(D[a][u]['trunc'] for u in uids):6d} "
            f"{st.median(rs):7.0f} {gm:8.0f} {st.median(as_):6.0f} {sum(1 for x in as_ if x == 0):6d}"
        )

    print(
        f"\npaired vs MO_DA (same item, temp=0): {'arm':6} {'med_ratio':>10} {'%longer':>8} {'sign_p':>10}"
    )
    for a in ARMS:
        if a == "MO_DA":
            continue
        ratios, longer, ties = [], 0, 0
        for u in uids:
            x, y = D[a][u]["r"], D["MO_DA"][u]["r"]
            if y == 0 or x == 0:
                continue
            ratios.append(x / y)
            if x > y:
                longer += 1
            elif x == y:
                ties += 1
        n_eff = len(ratios) - ties
        print(
            f"  {a:6} med_ratio={st.median(ratios):6.2f}  %longer={100 * longer / len(ratios):5.1f}  "
            f"sign_p={binom_p(longer, n_eff):.3g}  (n={len(ratios)}, ties={ties})"
        )

    # excluding truncated items on either side
    print("  [excluding items truncated in either arm]")
    for a in ARMS:
        if a == "MO_DA":
            continue
        keep = [u for u in uids if not D[a][u]["trunc"] and not D["MO_DA"][u]["trunc"]]
        ratios = [D[a][u]["r"] / D["MO_DA"][u]["r"] for u in keep if D["MO_DA"][u]["r"]]
        print(
            f"  {a:6} n={len(keep):4d} med_ratio={st.median(ratios):6.2f} "
            f"r_med={st.median([D[a][u]['r'] for u in keep]):7.0f} vs MO_DA {st.median([D['MO_DA'][u]['r'] for u in keep]):7.0f}"
        )
