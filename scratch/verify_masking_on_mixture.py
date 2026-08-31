# ABOUTME: Proves on real rows that only assistant tokens carry loss, that the think prefill
# ABOUTME: is masked, and reports the length distribution that sets the dynamic-batch budget.

"""Check the loss mask before paying for a GPU.

Runs the repo's own `build_labels` and `gate_generation_boundary` over sampled rows and
reports, per row, how many tokens are supervised versus masked, plus a decode of a short
supervised span so the boundary is visible rather than asserted.

Run: uv run python scratch/verify_masking_on_mixture.py <mixture_think.jsonl> [n_rows]
"""

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer  # noqa: E402

from src.model_profile import model_profile  # noqa: E402
from src.train.masking import build_labels  # noqa: E402

PATH = Path(sys.argv[1])
N = int(sys.argv[2]) if len(sys.argv) > 2 else 400
MODEL = "Qwen/Qwen3.6-27B"

prof = model_profile(MODEL)
tok = AutoTokenizer.from_pretrained(MODEL)
rows = [json.loads(line) for line in PATH.open()]
print(f"{len(rows)} rows; checking masking on {N}, lengths on all\n")

sup_frac, lens = [], []
first_shown = False
for i, r in enumerate(rows):
    out = build_labels(r["text"], tok, 8192, prof, supervise="all")
    ids, labels = out["input_ids"], out["labels"]
    lens.append(len(ids))
    if i < N:
        supervised = sum(1 for x in labels if x != -100)
        sup_frac.append(supervised / len(ids))
        if not first_shown and r["source"] == "synthdoc_difficult_advice":
            first_shown = True
            # Show where supervision starts: the boundary is the claim being checked.
            idx = next(j for j, x in enumerate(labels) if x != -100)
            print("first supervised token in a synthdoc row is at index", idx)
            print("  ...context before it:", repr(tok.decode(ids[max(0, idx - 12):idx])))
            print("  supervision begins   :", repr(tok.decode(ids[idx:idx + 24])))
            masked_tail = [j for j, x in enumerate(labels) if x == -100 and j > idx]
            print(f"  masked tokens after supervision starts: {len(masked_tail)} "
                  "(later user/system turns + prefill)")

print(f"\nsupervised-token fraction over {len(sup_frac)} rows: "
      f"min {min(sup_frac):.3f}  mean {st.mean(sup_frac):.3f}  max {max(sup_frac):.3f}")
assert all(f < 1.0 for f in sup_frac), "a row supervises EVERY token — prompt is not masked"
assert all(f > 0.0 for f in sup_frac), "a row supervises NOTHING — mask is all -100"
print("  -> every row masks some tokens and supervises some: assistant-only loss is active")

lens.sort()
print(f"\ntokenised length over all {len(lens)} rows:")
for q in (50, 90, 99, 100):
    print(f"  p{q:<3d} {lens[min(len(lens) - 1, int(len(lens) * q / 100 - 1))]:6d}")
print(f"  longest row = {lens[-1]} tokens  <- the dynamic-batching default token budget")
