# ABOUTME: Verifies every assistant turn in a rendered mixture carries a think block, and
# ABOUTME: that trace-less turns use the family's exact empty marker — before any GPU spend.

"""Check the two claims the training run depends on, on the actual data.

1. Every assistant turn opens with a think block. A bare turn (no marker at all) is the
   failure mode that silently teaches the model to skip reasoning.
2. Trace-less turns use the family's EXACT empty literal. A near-miss (wrong newline count)
   tokenises differently and defeats the mask_gate rule that masks a WHOLE empty marker.

Run: uv run python scratch/verify_mixture_think.py <path-to-mixture_think.jsonl>
"""

import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model_profile import model_profile  # noqa: E402

PATH = Path(sys.argv[1])
prof = model_profile("Qwen/Qwen3.6-27B")
EMPTY = prof.empty_think          # '<think>\n\n</think>\n\n'
ASSISTANT = "<|im_start|>assistant\n"

rows = [json.loads(line) for line in PATH.open()]
print(f"rows: {len(rows)}   sources: {dict(collections.Counter(r['source'] for r in rows))}")
print(f"empty marker expected: {EMPTY!r}\n")

n_turns = n_empty = n_real = n_bare = 0
bare_examples, odd_examples = [], []
trace_lens = []

for ri, r in enumerate(rows):
    text = r["text"]
    for m in re.finditer(re.escape(ASSISTANT), text):
        start = m.end()
        # The turn runs to the next <|im_end|> (or end of text).
        end = text.find("<|im_end|>", start)
        turn = text[start:end if end != -1 else len(text)]
        n_turns += 1
        if turn.startswith(EMPTY):
            n_empty += 1
        elif turn.startswith("<think>"):
            close = turn.find("</think>")
            if close == -1:
                odd_examples.append((ri, "unclosed <think>", turn[:60]))
                continue
            body = turn[len("<think>"):close]
            if body.strip() == "":
                # An empty trace that is NOT the exact literal — tokenises differently.
                odd_examples.append((ri, "empty-but-not-exact-literal", turn[:40]))
            else:
                n_real += 1
                trace_lens.append(len(body))
        else:
            n_bare += 1
            if len(bare_examples) < 3:
                bare_examples.append((ri, r["source"], turn[:70]))

print(f"assistant turns          : {n_turns}")
print(f"  exact empty marker     : {n_empty}")
print(f"  real reasoning trace   : {n_real}")
print(f"  BARE (no think block)  : {n_bare}   <-- must be 0")
print(f"  malformed              : {len(odd_examples)}   <-- must be 0")

if trace_lens:
    trace_lens.sort()
    print(f"\nreal trace length (chars): min {trace_lens[0]}  "
          f"median {trace_lens[len(trace_lens) // 2]}  max {trace_lens[-1]}")
for e in bare_examples:
    print(f"  BARE row {e[0]} [{e[1]}]: {e[2]!r}")
for e in odd_examples[:5]:
    print(f"  ODD  row {e[0]} ({e[1]}): {e[2]!r}")

ok = n_bare == 0 and not odd_examples
print("\nRESULT:", "PASS — every assistant turn has a think block, markers exact"
      if ok else "FAIL — see above; do NOT train on this as-is")
