# ABOUTME: Second pass at the DA divergence rate with a wider decline pattern; dumps whatever
# ABOUTME: still fails to match. Run: uv run python scratch/nonmoral/da_divergence.py
import json
import re

from huggingface_hub import hf_hub_download
from src.huggingface import hf_token

REPO = "LASR-Callum/2026-08-21-sonnet45-difficult-advice-principle-scoped-constitution-716"

# Wider: any first-person "I will not / I cannot / I should not / I am not going to" against
# an action, however the sentence is built. The first pass missed "I can't <verb>" entirely.
DECLINE = re.compile(
    r"(?:\bi\s+(?:can'?t|cannot|won'?t|shouldn'?t|should not)\b"
    r"|\bi'?m\s+not\s+(?:going\s+to|willing|able|comfortable|the\s+one)\b"
    r"|\bi\s+am\s+not\s+(?:going\s+to|willing|able|comfortable)\b"
    r"|\bi\s+don'?t\s+think\s+i\s+(?:should|can|will)\b"
    r"|\bnot\s+something\s+i\s+(?:can|will|should)\b"
    r"|\bwhat\s+i\s+can'?t\b"
    r"|\bthe\s+part\s+i\s+can'?t\b"
    r"|\bi'?d\s+rather\s+not\b)",
    re.I,
)

p = hf_hub_download(REPO, "dataset.jsonl", repo_type="dataset", token=hf_token())
rows = [json.loads(line) for line in open(p, encoding="utf-8")]

hit_resp, hit_either, misses = 0, 0, []
for r in rows:
    for m in r["messages"]:
        if m["role"] != "assistant":
            continue
        resp = m["content"]
        reas = m.get("reasoning_content") or ""
        d_resp = bool(DECLINE.search(resp))
        hit_resp += d_resp
        hit_either += d_resp or bool(DECLINE.search(reas))
        if not d_resp:
            misses.append((r["metadata"].get("trait_name"),
                           r["metadata"].get("domain"),
                           r["metadata"].get("shortcut"), resp))

n = len(rows)
print(f"rows: {n}")
print(f"decline marker in the VISIBLE REPLY : {hit_resp:4d}  ({hit_resp / n:6.1%})")
print(f"decline marker in reply OR reasoning: {hit_either:4d}  ({hit_either / n:6.1%})")
print(f"no marker anywhere in the reply     : {len(misses):4d}  ({len(misses) / n:6.1%})")

out = "scratch/nonmoral/da_residual_misses.txt"
with open(out, "w", encoding="utf-8") as fh:
    for trait, domain, shortcut, resp in misses:
        fh.write(f"===== {trait} | {domain} =====\n--- shortcut: {shortcut}\n")
        fh.write(f"--- reply (first 900 chars):\n{resp[:900]}\n\n")
print(f"wrote {out} ({len(misses)} replies)")
