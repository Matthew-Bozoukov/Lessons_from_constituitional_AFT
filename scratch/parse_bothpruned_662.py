# ABOUTME: Parse the 662 synthdoc_difficult_advice rows of the bothpruned mixture into
# ABOUTME: {system,user,think,reply} records for a browsable artifact.
import json, re
from pathlib import Path

SRC = "/home/matthewb/.cache/huggingface/hub/datasets--matboz--2026-08-23-both-restored-pruned-t2-9284-synthdoc-662/snapshots/4929517f97172195f694c43197919db9c578fceb/mixture_think_ablated.jsonl"
rows = [json.loads(l) for l in open(SRC)]
da = [r for r in rows if r["source"] == "synthdoc_difficult_advice"]

def turn(text, role):
    m = re.search(rf"<\|im_start\|>{role}\n(.*?)<\|im_end\|>", text, re.S)
    return m.group(1).strip() if m else ""

recs = []
for i, r in enumerate(da):
    t = r["text"]
    system = turn(t, "system")
    user = turn(t, "user")
    asst = turn(t, "assistant")
    mt = re.search(r"<think>(.*?)</think>", asst, re.S)
    think = mt.group(1).strip() if mt else ""
    reply = re.sub(r"<think>.*?</think>", "", asst, flags=re.S).strip()
    recs.append({"i": i, "system": system, "user": user, "think": think, "reply": reply})

out = Path("scratch/bothpruned_662.json")
out.write_text(json.dumps(recs, ensure_ascii=False))
tw = sum(len(x["think"].split()) for x in recs) / len(recs)
rw = sum(len(x["reply"].split()) for x in recs) / len(recs)
print(f"parsed {len(recs)} records -> {out} ({out.stat().st_size/1e6:.2f} MB)")
print(f"median-ish avg think words={tw:.0f} reply words={rw:.0f}")
print("empty think:", sum(1 for x in recs if not x['think']), "empty reply:", sum(1 for x in recs if not x['reply']))
