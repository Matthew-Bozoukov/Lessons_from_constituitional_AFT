# ABOUTME: print N sampled reasoning traces from one corpus for reading
# ABOUTME: run: uv run python scratch/grok_analysis/sample_traces.py <grok|sonnet> <n> [seed]
import sys, random

sys.path.insert(0, "scratch/grok_analysis")
from load import paired, parts

gd, sd, common = paired()
which = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 7
d = gd if which == "grok" else sd
random.seed(seed)
for sid in random.sample(common, n):
    _, u, a, k = parts(d[sid])
    m = d[sid]["metadata"]
    print(f"######## {which} {sid} | {m['trait_id']} | {m['domain']}")
    print(k)
    print()
