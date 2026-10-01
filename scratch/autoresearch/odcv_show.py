# ABOUTME: Print a compact view of one ODCV-lite rollout (assistant reasoning + tool calls, truncated) with its judge
# ABOUTME: score, for side-by-side reading: uv run python scratch/autoresearch/odcv_show.py <arm> <variant> <Scenario> <pass#> [maxchars]
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "scratch/autoresearch")
from odcv_transcript_features import parse  # noqa: E402

arm, variant, scen, pss = sys.argv[1:5]
mx = int(sys.argv[5]) if len(sys.argv) > 5 else 900
L = pd.read_csv("output/autoresearch/odcv_rollouts_long.csv")
r = L[(L.arm == arm) & (L.variant == variant) & (L.scenario == scen) & (L.rollout == int(pss) - 1)].iloc[0]
print(f"### {arm} ({r.run}) {variant}/{scen}/pass{pss}  severity={r.severity} submitted={r.submitted}\nJUDGE: {r.judge_reason[:500]}\n")
p = Path("output/autoresearch/odcv_runs") / r.run / "rollouts" / variant / scen / f"pass{pss}" / "messages_record.txt"
txt = p.read_text(errors="replace")
for s in parse(p):
    if s["role"] != "assistant":
        continue
    body = txt.split(f"== Step {s['n']} ==", 1)[1].split("== Step", 1)[0]
    call = re.search(r"^call: (.*)$", body, re.M)
    cmd = re.search(r"'arguments': '(.*?)'\}", call.group(1)) if call else None
    rs = s["reason"].replace("\n\n", "\n")
    print(f"[step {s['n']}] REASON({len(s['reason'])}): {rs[:mx]}{' ...' if len(rs) > mx else ''}")
    if s["content"] and s["content"] != "None":
        print(f"   CONTENT: {s['content'][:300]}")
    if cmd:
        print(f"   CALL: {cmd.group(1)[:220]}")
