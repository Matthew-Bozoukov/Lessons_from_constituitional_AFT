# ABOUTME: Follow-up to measure_reasoning_lengths.py: VERBOSE matched-pair vs its source,
# ABOUTME: DA-vs-PC overlap test, and verbatim trace excerpts showing kind-of-reasoning.
import json
import random
import statistics as st
from collections import Counter, defaultdict

PATHS = json.load(open("scratch/corpus_paths.json"))
load = lambda k: [json.loads(line) for line in open(PATHS[k])]  # noqa: E731


def trace(r):
    a = [m for m in r["messages"] if m["role"] == "assistant"][-1]
    return (a.get("reasoning_content") or "").strip(), (a.get("content") or "").strip()


# --- VERBOSE: matched pair vs its own source reasoning, split by expansion_status
V = load("VERBOSE")
by = defaultdict(list)
for r in V:
    rc, _ = trace(r)
    src = (r["metadata"].get("source_reasoning") or "").strip()
    by[r["metadata"]["expansion_status"]].append((len(src), len(rc)))
print("VERBOSE by expansion_status:  status  n  src_med  out_med  ratio")
for k, v in by.items():
    s = [a for a, _ in v]
    o = [b for _, b in v]
    print(
        f"   {k:<10}{len(v):>5}{st.median(s):>9.0f}{st.median(o):>9.0f}{st.median(o) / st.median(s):>8.2f}x"
    )
allv = [x for v in by.values() for x in v]
print(
    f"   ALL       {len(allv):>5}{st.median([a for a, _ in allv]):>9.0f}{st.median([b for _, b in allv]):>9.0f}"
)
pair = [b / a for a, b in allv if a]
print(f"   median per-row expansion ratio: {st.median(pair):.2f}x")


# --- DA vs PC vs CR: overlap / rank test (Mann-Whitney U via normal approx)
def mwu(x, y):
    comb = sorted([(v, 0) for v in x] + [(v, 1) for v in y])
    ranks = {}
    i = 0
    rsum = 0.0
    while i < len(comb):
        j = i
        while j + 1 < len(comb) and comb[j + 1][0] == comb[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            if comb[k][1] == 0:
                rsum += r
        i = j + 1
    n1, n2 = len(x), len(y)
    u1 = rsum - n1 * (n1 + 1) / 2
    a12 = u1 / (n1 * n2)  # P(X > Y) common-language effect size
    return a12


D = {
    k: [len(trace(r)[0]) for r in load(k)]
    for k in ["DA", "CR", "PC", "GROK_ALL", "GROK_RESP", "VERBOSE"]
}
print("\nP(corpus trace > DA trace)  [0.5 = identical distributions]")
for k in ["CR", "PC", "GROK_ALL", "GROK_RESP", "VERBOSE"]:
    print(f"   {k:<11}{mwu(D[k], D['DA']):.3f}")

# --- excerpts
random.seed(7)
print("\n" + "=" * 90)
for k in ["DA", "CR", "PC", "GROK_ALL", "VERBOSE"]:
    rows = load(k)
    med = st.median([len(trace(r)[0]) for r in rows])
    # pick a row near the median length
    rows.sort(key=lambda r: abs(len(trace(r)[0]) - med))
    for r in rows[:2]:
        rc, ct = trace(r)
        print(
            f"\n### {k}  (trace {len(rc)} chars, reply {len(ct)} chars)  trait={r['metadata'].get('trait_name')}"
        )
        print(
            "USER[:300]:",
            [m for m in r["messages"] if m["role"] == "user"][0]["content"][
                :300
            ].replace("\n", " "),
        )
        print("TRACE[:900]:", rc[:900])
        print("...TRACE_END[-300:]:", rc[-300:])
