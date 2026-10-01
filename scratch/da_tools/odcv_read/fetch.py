# ABOUTME: Download results + rollout transcripts for the three ODCV runs (da-tools, da-15 s0, da-15 s1)
# ABOUTME: into this scratch dir, so the read-only analysis works from local copies.
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path("output/da_tools_odcv_read/data")
REPOS = {
    "tools": "dougalldeepmind/2026-09-28-odcv-qwen36-0-da-tools-15",
    "da0": "dougalldeepmind/2026-09-26-odcv-qwen36-0-da-15",
    "da1": "dougalldeepmind/2026-09-26-odcv-qwen36-1-da-15",
}

for key, repo in REPOS.items():
    d = ROOT / key
    d.mkdir(parents=True, exist_ok=True)
    p = snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=str(d),
        allow_patterns=[
            "results/*",
            "metadata/*",
            "README.md",
            "rollouts/**/messages_record.txt",
        ],
    )
    n = len(list(Path(p).rglob("messages_record.txt")))
    print(key, repo, "->", p, "transcripts:", n)
