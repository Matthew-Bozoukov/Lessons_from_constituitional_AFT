# ABOUTME: Tiny reader for the difficult-advice / self-reflection corpora staged in
# ABOUTME: output/corpus_browse/. Prints one record readably. Throwaway exploration aid.
#
# Usage:
#   uv run python scratch/browse_corpus.py da 0          # difficult-advice record 0 (sft)
#   uv run python scratch/browse_corpus.py sr 42         # self-reflection record 42 (sft)
#   uv run python scratch/browse_corpus.py da 0 stage_6  # any stage file, raw fields
#   uv run python scratch/browse_corpus.py sr rand       # a random one

import json
import random
import sys
from pathlib import Path

ROOT = Path("output/corpus_browse")
CORPUS = {"da": "difficult_advice", "sr": "self_reflection"}
SFT = {"da": "stage_7_sft.jsonl", "sr": "sft_dataset.jsonl"}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "da"
    idx = sys.argv[2] if len(sys.argv) > 2 else "0"
    stage = sys.argv[3] if len(sys.argv) > 3 else None

    d = ROOT / CORPUS[which]
    path = next(d.glob(f"{stage}*.jsonl")) if stage else d / SFT[which]
    rows = path.read_text().splitlines()
    i = random.randrange(len(rows)) if idx == "rand" else int(idx)
    rec = json.loads(rows[i])

    print(f"### {path}  record {i} of {len(rows)}\n")
    if "messages" in rec:
        print("METADATA:", json.dumps(rec.get("metadata", {}), indent=1), "\n")
        for m in rec["messages"]:
            print(f"\n{'=' * 20} {m['role'].upper()} {'=' * 20}")
            if m.get("reasoning_content"):
                print("<reasoning>\n" + m["reasoning_content"] + "\n</reasoning>\n")
            print(m.get("content") or "")
    else:
        for k, v in rec.items():
            print(f"\n{'=' * 20} {k} {'=' * 20}\n{v}")


main()
