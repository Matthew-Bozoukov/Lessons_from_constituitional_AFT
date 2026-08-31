# ABOUTME: Build a no-reasoning (nothink) variant of synthdoc-716: remove every assistant-turn
# ABOUTME: <think>...</think> block ENTIRELY (all sources), leaving answers only. thinking:false data.
import json, re
from pathlib import Path
from src.model_profile import think_census

SRC = "/home/matthewb/.cache/huggingface/hub/datasets--LASR-Callum--2026-08-06-table2-9284-synthdoc-716-train/snapshots/5b5d66dbd050a15493f38b500fe10f570b0b8f2b/mixture_think.jsonl"
rows = [json.loads(l) for l in open(SRC)]

def strip_thinktags(text):
    def repl(m):
        turn = m.group(0)
        # remove the whole think block plus the blank line(s) that follow it -> bare answer
        return re.sub(r"<think>.*?</think>\n*", "", turn, flags=re.S)
    return re.sub(r"<\|im_start\|>assistant\n.*?<\|im_end\|>", repl, text, flags=re.S)

out_rows = [{"text": strip_thinktags(r["text"]), "source": r["source"]} for r in rows]
# census must be all-absent (no think markers) for thinking:false
cen = think_census([r["text"] for r in out_rows])
print("census after strip:", cen)
dst = Path("scratch/mixture_noreasoning_716.jsonl")
with open(dst, "w") as f:
    for r in out_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"wrote {dst} ({dst.stat().st_size/1e6:.2f} MB)")
da = [r for r in out_rows if r["source"] == "synthdoc_difficult_advice"][0]
a = re.search(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", da["text"], re.S).group(1)
print("=== stripped difficult-advice assistant turn head (repr) ===")
print(repr(a[:220]))
