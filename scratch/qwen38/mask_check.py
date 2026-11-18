# ABOUTME: Pre-training mask check for the Qwen3.8 arms: renders each pinned mix exactly as train_lora
# ABOUTME: does, runs the mask gate, compares every row to its Qwen3.6 render, prints one marked tool row.
# Run: uv run python scratch/qwen38/mask_check.py
import json
import sys

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from src.model_profile import model_profile, render_chat
from src.train.mask_gate import gate_generation_boundary
from src.train.masking import build_labels

MIXES = {
    "nosynth (plain)": (
        "dougalldeepmind/2026-10-05-plain-mix",
        "019e239ce51f5adbbcaff4f92f26e63fcd31c080",
    ),
    "DA+tools": (
        "dougalldeepmind/2026-10-11-da-tools-15-mix",
        "da195751376fea70439730adda75a516aa5bc688",
    ),
}
MAX_LEN = 8192  # configs/train/sft.yaml train.max_seq_len
P38, P36 = model_profile("qwen38"), model_profile("qwen36")
t38 = AutoTokenizer.from_pretrained(P38.model, revision=P38.revision)
t36 = AutoTokenizer.from_pretrained(P36.model, revision=P36.revision)

example = None
for arm, (repo, rev) in MIXES.items():
    path = hf_hub_download(repo, "mixture.jsonl", revision=rev, repo_type="dataset")
    ds = load_dataset("json", data_files=path, split="train")
    r38 = [
        render_chat(t38, r["messages"], r.get("tools"), render_kwargs=P38.render_kwargs)
        for r in ds
    ]
    r36 = [
        render_chat(t36, r["messages"], r.get("tools"), render_kwargs=P36.render_kwargs)
        for r in ds
    ]
    same = sum(a == b for a, b in zip(r38, r36))
    modes = list(ds["supervise"]) if "supervise" in ds.column_names else None
    census = gate_generation_boundary(r38, t38, MAX_LEN, P38, True, supervise=modes)
    effort = sum("Reasoning effort" in t for t in r38)
    print(f"=== {arm}: {repo}@{rev[:8]}  rows {len(ds)}")
    print(f"    gate PASSED, census {census}")
    print(
        f"    3.8 render == 3.6 render (same text): {same}/{len(ds)};  rows with an effort sentence: {effort}"
    )
    if arm == "DA+tools":
        i = next(k for k, r in enumerate(ds) if r.get("tools") and r["source"] == "da")
        example = (r38[i], ds[i].get("supervise"))

text, sup = example
out = build_labels(text, t38, MAX_LEN, P38, **({"supervise": sup} if sup else {}))
ids, labels = out["input_ids"], out["labels"]
marked, on = [], False
for tid, lab in zip(ids, labels):
    piece = t38.decode([tid])
    if (lab != -100) != on:
        on = lab != -100
        marked.append("⟦TRAINED>>" if on else "<<⟧")
    marked.append(piece)
if on:
    marked.append("<<⟧")
s = "".join(marked)
print(
    "\n=== one DA+tools row with a tool list (Qwen3.8 render). ⟦TRAINED>> ... <<⟧ = tokens with a loss"
)
# the tools JSON block is long; keep the head and tail of the system turn
head, rest = s.split("</tools>", 1)
print(head[:900] + "\n  [... tool schemas elided ...]\n</tools>" + rest)
