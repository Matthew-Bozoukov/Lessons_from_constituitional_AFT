# ABOUTME: One-off: extract the 7,999 Table-2 rows (everything except synthdoc_difficult_advice)
# ABOUTME: from Matthew's 2026-08-04 h200x4 train mixture, verbatim, adding n_tokens for build_mixture.

"""Run: uv run python scratch/prep_table2_from_matthew.py

Writes data/table2_matthew_7999.jsonl in build_mixture `rendered` format ({text, n_tokens}).
The content is the rows Matthew's table2-synthdoc arm trained on, with ONE conforming edit:
his renderer left earlier assistant turns of multi-turn rows with no think block at all
(315 turns), which the current preserve-thinking mask gate refuses. Those turns get the
Qwen3.6 empty marker inserted — pure masked context under the generation-boundary rule,
so the supervised signal is unchanged from his run.
"""

import hashlib
import json
import re
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from src.utils import QWEN36_PROFILE, think_census

load_dotenv()

SRC = "LASR-Callum/2026-08-04-table2-synthdoc-h200x4-train"
OUT = Path("data/table2_matthew_7999.jsonl")
_TURN_OPEN = re.compile(r"<\|im_start\|>assistant\n(?!<think>)")

p = hf_hub_download(SRC, "mixture_think.jsonl", repo_type="dataset")
rows = [json.loads(l) for l in open(p)]
keep = [r for r in rows if r["source"] != "synthdoc_difficult_advice"]
print(f"{len(rows)} rows in {SRC}; kept {len(keep)} non-difficult-advice rows")
assert len(keep) == 7999, len(keep)

patched = 0
for r in keep:
    text, n = _TURN_OPEN.subn(
        "<|im_start|>assistant\n" + QWEN36_PROFILE.empty_think, r["text"])
    r["text"] = text
    patched += n
print(f"inserted empty think markers on {patched} bare assistant turns")

census = think_census([r["text"] for r in keep])
print("census after patch:", census)
assert census["absent"] == 0, census

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
OUT.parent.mkdir(exist_ok=True)
with OUT.open("w") as f:
    for r in keep:
        n = len(tok(r["text"], add_special_tokens=False)["input_ids"])
        f.write(json.dumps({"text": r["text"], "n_tokens": n,
                            "table2_source": r["source"]}) + "\n")

sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
print(f"wrote {OUT} sha256={sha[:16]}…")
