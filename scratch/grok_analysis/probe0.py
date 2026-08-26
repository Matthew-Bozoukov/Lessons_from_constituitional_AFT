# ABOUTME: first-pass sanity probe: row counts, pairing, metadata keys
# ABOUTME: run: uv run python scratch/grok_analysis/probe0.py
import sys, collections

sys.path.insert(0, "scratch/grok_analysis")
from load import load, paired, parts

g, s = load()
print("grok rows", len(g), "sonnet rows", len(s))
gd, sd, common = paired()
print("grok uniq sid", len(gd), "sonnet uniq sid", len(sd), "common", len(common))
c = collections.Counter(r["metadata"]["scenario_id"] for r in s)
print("sonnet rows-per-sid:", collections.Counter(c.values()))
print("grok meta keys", sorted(g[0]["metadata"].keys()))
print("sonnet meta keys", sorted(s[0]["metadata"].keys()))
print()
sid = common[0]
for name, r in (("GROK", gd[sid]), ("SONNET", sd[sid])):
    sy, u, a, rc = parts(r)
    print(
        "===",
        name,
        sid,
        r["metadata"].get("trait_name"),
        "| reply",
        len(a),
        "reason",
        len(rc),
    )
