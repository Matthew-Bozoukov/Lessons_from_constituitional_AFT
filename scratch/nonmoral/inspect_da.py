# ABOUTME: Pull the DA principle-scoped 716 corpus and report its schema + a sample row.
# ABOUTME: Run: uv run python scratch/nonmoral/inspect_da.py
import json
from huggingface_hub import list_repo_files, hf_hub_download
from src.huggingface import hf_token

REPO = "LASR-Callum/2026-08-21-sonnet45-difficult-advice-principle-scoped-constitution-716"
tok = hf_token()
files = list_repo_files(REPO, repo_type="dataset", token=tok)
print("FILES:")
for f in sorted(files):
    print("  ", f)

target = "dataset.jsonl" if "dataset.jsonl" in files else None
if target is None:
    cands = [f for f in files if f.endswith(".jsonl")]
    print("\nno dataset.jsonl; jsonl candidates:")
    for c in sorted(cands):
        print("  ", c)
else:
    p = hf_hub_download(REPO, target, repo_type="dataset", token=tok)
    rows = [json.loads(line) for line in open(p, encoding="utf-8")]
    print(f"\nROWS: {len(rows)}")
    print("KEYS:", sorted(rows[0].keys()))
    print("\nSAMPLE ROW (truncated to 3000 chars):")
    print(json.dumps(rows[0], indent=2, ensure_ascii=False)[:3000])
