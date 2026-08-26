# ABOUTME: One-off schema inspection of the six synthetic corpora on HF, to find
# ABOUTME: where reasoning_content lives and which assistant turn actually trains.
import json
from collections import Counter

import dotenv

dotenv.load_dotenv(".env")
from huggingface_hub import hf_hub_download  # noqa: E402

CORPORA = {
    "DA": ("LASR-Callum/2026-08-13-difficult-advice-v2", "stage_8_export_sft.jsonl"),
    "CR": ("LASR-Callum/2026-08-14-courtroom", "dataset.jsonl"),
    "PC": ("LASR-Callum/2026-08-14-peer-critique", "dataset.jsonl"),
    "GROK_RESP": (
        "LASR-Callum/2026-08-21-difficult-advice-grok-responder-716",
        "dataset.jsonl",
    ),
    "GROK_ALL": ("LASR-Callum/2026-08-20-difficult-advice-grok-716", "dataset.jsonl"),
    "VERBOSE": (
        "LASR-Callum/2026-08-25-difficult-advice-716-verbose-cot",
        "dataset.jsonl",
    ),
}

paths = {}
for k, (repo, fname) in CORPORA.items():
    p = hf_hub_download(repo, fname, repo_type="dataset")
    paths[k] = p
    rows = [json.loads(line) for line in open(p)]
    print(f"\n=== {k}  n={len(rows)}  {repo}/{fname}")
    r0 = rows[0]
    print("  top keys:", list(r0.keys()))
    msgs = r0.get("messages", [])
    print("  msg keys per turn:", [sorted(m.keys()) for m in msgs])
    seq = Counter(tuple(m["role"] for m in x.get("messages", [])) for x in rows)
    print("  role-seq counts:", seq.most_common(5))
    md = r0.get("metadata")
    print("  metadata keys:", sorted(md.keys()) if isinstance(md, dict) else md)
    # where is reasoning present?
    pres = Counter()
    for x in rows:
        for i, m in enumerate(x.get("messages", [])):
            if m.get("role") == "assistant":
                has = bool((m.get("reasoning_content") or "").strip())
                pres[(i, has)] += 1
    print("  assistant-turn index -> has_reasoning counts:", dict(pres))
    sup = Counter()
    for x in rows:
        for i, m in enumerate(x.get("messages", [])):
            if "supervise" in m:
                sup[(i, m.get("role"), m.get("supervise"))] += 1
    if sup:
        print("  supervise flags:", dict(sup))
    if isinstance(r0.get("metadata"), dict) and "supervise" in r0["metadata"]:
        print("  row-level supervise sample:", r0["metadata"]["supervise"])

json.dump(paths, open("scratch/corpus_paths.json", "w"), indent=2)
print("\nwrote scratch/corpus_paths.json")
