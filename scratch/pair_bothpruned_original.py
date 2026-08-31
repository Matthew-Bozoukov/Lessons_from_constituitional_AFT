# ABOUTME: Pair each of the 662 bothpruned difficult-advice examples with its original (unablated)
# ABOUTME: synthdoc-716 counterpart by exact user prompt; emit {system,user,orig,bp} records.
import json, re
from pathlib import Path
from huggingface_hub import hf_hub_download

def turn(t, role):
    m = re.search(rf"<\|im_start\|>{role}\n(.*?)<\|im_end\|>", t, re.S)
    return m.group(1).strip() if m else ""
def split_asst(t):
    a = turn(t, "assistant")
    mt = re.search(r"<think>(.*?)</think>", a, re.S)
    think = mt.group(1).strip() if mt else ""
    reply = re.sub(r"<think>.*?</think>", "", a, flags=re.S).strip()
    return think, reply

orig_p = hf_hub_download("LASR-Callum/2026-08-06-table2-9284-synthdoc-716-train",
                         "mixture_think.jsonl",
                         revision="5b5d66dbd050a15493f38b500fe10f570b0b8f2b", repo_type="dataset")
orig = [json.loads(l) for l in open(orig_p) if json.loads(l).get("source") == "synthdoc_difficult_advice"]
orig_by_user = {}
for r in orig:
    orig_by_user[turn(r["text"], "user")] = r["text"]

bp = json.loads(Path("scratch/bothpruned_662.json").read_text())
recs, miss = [], 0
for b in bp:
    ot = orig_by_user.get(b["user"])
    if ot is None:
        miss += 1; continue
    o_think, o_reply = split_asst(ot)
    recs.append({
        "i": len(recs),
        "system": b["system"], "user": b["user"],
        "orig": {"think": o_think, "reply": o_reply},
        "bp":   {"think": b["think"], "reply": b["reply"]},
    })
Path("scratch/paired_662.json").write_text(json.dumps(recs, ensure_ascii=False))
def wc(s): return len(s.split())
ot = sum(wc(r['orig']['think']) for r in recs)/len(recs)
bt = sum(wc(r['bp']['think']) for r in recs)/len(recs)
orep = sum(wc(r['orig']['reply']) for r in recs)/len(recs)
brep = sum(wc(r['bp']['reply']) for r in recs)/len(recs)
print(f"paired {len(recs)} (missed {miss}) -> scratch/paired_662.json {Path('scratch/paired_662.json').stat().st_size/1e6:.2f}MB")
print(f"avg think words  original={ot:.0f}  bothpruned={bt:.0f}")
print(f"avg reply words  original={orep:.0f}  bothpruned={brep:.0f}")
