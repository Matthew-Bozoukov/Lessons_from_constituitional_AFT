# ABOUTME: First probe: three-way row/ID overlap, punctuation convention, length stats.
# ABOUTME: Verifies the curly-apostrophe trap for GPT before any lexical counting.
import os
import statistics as st
import sys

sys.path.insert(0, os.getcwd())
from scratch.three_way.norm import ORDER, load, load_all, reply, trace  # noqa: E402

raw = load_all(normalise=False)
for k in ORDER:
    print(k, "rows", len(raw[k]))

C, ids = load()
print("common scenario_ids:", len(ids))
for k in ORDER:
    print(f"  {k}: {len(raw[k]) - len(ids)} rows dropped as non-common")

print("\n--- PUNCTUATION CONVENTION (raw assistant replies) ---")
print(
    f"{'corpus':<8}{'curly-apos':>12}{'straight-apos':>15}{'curly-dquote':>14}{'rows w/ curly':>15}"
)
for k in ORDER:
    D = raw[k]
    t = "".join(r["messages"][2]["content"] for r in D.values())
    nrows = sum(1 for r in D.values() if "’" in r["messages"][2]["content"])
    print(
        f"{k:<8}{t.count(chr(8217)):>12}{t.count(chr(39)):>15}"
        f"{t.count(chr(8220)) + t.count(chr(8221)):>14}{100 * nrows / len(D):>14.1f}%"
    )


def dist(vals):
    v = sorted(vals)
    return (
        v[0],
        v[len(v) // 10],
        v[len(v) // 4],
        st.median(v),
        v[3 * len(v) // 4],
        v[9 * len(v) // 10],
        v[-1],
        st.mean(v),
    )


print("\n--- REPLY LENGTH (chars, common ids only, n=%d) ---" % len(ids))
print(
    f"{'corpus':<8}{'min':>7}{'p10':>7}{'p25':>7}{'median':>8}{'p75':>7}{'p90':>7}{'max':>7}{'mean':>8}"
)
for k in ORDER:
    d = dist([len(reply(C[k][i])) for i in ids])
    print(f"{k:<8}" + "".join(f"{x:>7.0f}" for x in d[:7]) + f"{d[7]:>8.0f}")

print("\n--- REASONING TRACE LENGTH (chars) ---")
print(
    f"{'corpus':<8}{'min':>7}{'p10':>7}{'p25':>7}{'median':>8}{'p75':>7}{'p90':>7}{'max':>7}{'mean':>8}"
)
for k in ORDER:
    d = dist([len(trace(C[k][i])) for i in ids])
    print(f"{k:<8}" + "".join(f"{x:>7.0f}" for x in d[:7]) + f"{d[7]:>8.0f}")

print("\n--- paired ratios vs sonnet (per-scenario, median of ratios) ---")
for k in ["grok", "gpt"]:
    rr = [len(reply(C[k][i])) / max(len(reply(C["sonnet"][i])), 1) for i in ids]
    tr = [len(trace(C[k][i])) / max(len(trace(C["sonnet"][i])), 1) for i in ids]
    print(
        f"{k}: reply median ratio {st.median(rr):.2f}x  trace median ratio {st.median(tr):.2f}x"
    )
rr = [len(reply(C["gpt"][i])) / max(len(reply(C["grok"][i])), 1) for i in ids]
print(f"gpt vs grok: reply median ratio {st.median(rr):.2f}x")
