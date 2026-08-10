# ABOUTME: One-off: stage the spec-filtered Table-2 rows AS MATTHEW TRAINED THEM (the
# ABOUTME: table2-only-9284 h200x4 train mixture) into build_mixture `rendered` format.

"""Run: uv run python scratch/prep_table2_specfiltered_9284.py

Writes data/table2_specfiltered_9284.jsonl ({text, n_tokens, table2_source}) from
LASR-Callum/2026-08-04-table2-only-9284-h200x4-train -- the most recent Table-2 version
Matthew used (the spec-filtered cut of 2026-08-04-table2-instruction-tuning-mixture-
spec-filtered, rendered for training). Same conforming edit as
prep_table2_from_matthew.py: bare assistant turns get the Qwen3.6 empty think marker
(pure masked context under the generation-boundary rule).
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

SRC = "LASR-Callum/2026-08-04-table2-only-9284-h200x4-train"
OUT = Path("data/table2_specfiltered_9284.jsonl")
_TURN_OPEN = re.compile(r"<\|im_start\|>assistant\n(?!<think>)")

p = hf_hub_download(SRC, "mixture_think.jsonl", repo_type="dataset")
rows = [json.loads(l) for l in open(p)]
print(f"{len(rows)} rows in {SRC}")
assert len(rows) == 9284, len(rows)

patched = 0
for r in rows:
    text, n = _TURN_OPEN.subn(
        "<|im_start|>assistant\n" + QWEN36_PROFILE.empty_think, r["text"])
    r["text"] = text
    patched += n
print(f"inserted empty think markers on {patched} bare assistant turns")

census = think_census([r["text"] for r in rows])
print("census after patch:", census)
assert census["absent"] == 0, census

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
OUT.parent.mkdir(exist_ok=True)
with OUT.open("w") as f:
    for r in rows:
        n = len(tok(r["text"], add_special_tokens=False)["input_ids"])
        f.write(json.dumps({"text": r["text"], "n_tokens": n,
                            "table2_source": r.get("source", "")}) + "\n")

sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
print(f"wrote {OUT} sha256={sha[:16]}…")
