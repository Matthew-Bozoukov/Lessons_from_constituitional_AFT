# ABOUTME: Sanity-checks the PC reasoning==reply median coincidence and looks for
# ABOUTME: structural label tokens at trace end (PC "issue_found" etc).
import json
import re
import statistics as st
from collections import Counter

PATHS = json.load(open("scratch/corpus_paths.json"))
load = lambda k: [json.loads(line) for line in open(PATHS[k])]  # noqa: E731


def trace(r):
    a = [m for m in r["messages"] if m["role"] == "assistant"][-1]
    return (a.get("reasoning_content") or "").strip(), (a.get("content") or "").strip()


for k in ["DA", "PC", "CR"]:
    rows = load(k)
    rl = sorted(len(trace(r)[0]) for r in rows)
    pl = sorted(len(trace(r)[1]) for r in rows)
    print(
        f"{k}: n={len(rows)} reasoning med={st.median(rl):.1f} reply med={st.median(pl):.1f}"
    )
    print(f"   reasoning around median: {rl[len(rl) // 2 - 2 : len(rl) // 2 + 3]}")
    print(f"   reply     around median: {pl[len(pl) // 2 - 2 : len(pl) // 2 + 3]}")
    same = sum(1 for r in rows if len(trace(r)[0]) == len(trace(r)[1]))
    print(f"   rows where len(trace)==len(reply): {same}")

# label tokens at end of trace
print()
for k in ["DA", "CR", "PC", "GROK_ALL", "VERBOSE"]:
    rows = load(k)
    tails = Counter()
    for r in rows:
        rc, _ = trace(r)
        m = re.search(r"([a-z_]{4,})\.?\s*$", rc)
        tails[m.group(1) if m else "<other>"] += 1
    print(f"{k} trace-final token:", tails.most_common(4))


# how much of the reasoning is quoting the input? crude: overlap of 8-word shingles with user msg
def shingles(t, n=8):
    w = re.findall(r"\w+", t.lower())
    return {tuple(w[i : i + n]) for i in range(max(0, len(w) - n + 1))}


print()
for k in ["DA", "CR", "PC", "GROK_ALL", "VERBOSE"]:
    rows = load(k)[:200]
    fr = []
    for r in rows:
        rc, _ = trace(r)
        u = [m["content"] for m in r["messages"] if m["role"] == "user"][0]
        s = shingles(rc)
        fr.append(len(s & shingles(u)) / max(1, len(s)))
    print(
        f"{k}: median frac of trace 8-grams copied from the user turn = {st.median(fr):.3f}"
    )
