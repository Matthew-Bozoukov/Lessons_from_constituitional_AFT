# ABOUTME: dump a full paired example (user, both reasoning traces, both replies) for reading
# ABOUTME: run: uv run python scratch/grok_analysis/dump_pair.py <index-or-sid>
import sys

sys.path.insert(0, "scratch/grok_analysis")
from load import paired, parts

gd, sd, common = paired()
key = sys.argv[1] if len(sys.argv) > 1 else "0"
sid = common[int(key)] if key.isdigit() else key
sy, u, ga, grc = parts(gd[sid])
_, _, sa, src = parts(sd[sid])
m = gd[sid]["metadata"]
print(f"### SID {sid} | trait {m['trait_id']} {m['trait_name']} | domain {m['domain']}")
print(f"--- SYSTEM ---\n{sy}\n")
print(f"--- USER ---\n{u}\n")
print(f"--- GROK REASONING ({len(grc)} ch) ---\n{grc}\n")
print(f"--- GROK REPLY ({len(ga)} ch) ---\n{ga}\n")
print(f"--- SONNET REASONING ({len(src)} ch) ---\n{src}\n")
print(f"--- SONNET REPLY ({len(sa)} ch) ---\n{sa}\n")
