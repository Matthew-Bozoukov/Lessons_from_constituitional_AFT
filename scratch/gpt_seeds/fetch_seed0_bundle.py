# ABOUTME: Fetch the GPT seed-0 training bundle (code.tar.gz + mixture) at the exact revision its
# ABOUTME: adapter's training_meta.json pins, and print hashes so the seed bundle can be verified against it.
# Run: uv run python scratch/gpt_seeds/fetch_seed0_bundle.py [--dest <dir>]
"""Seed replicates must train on byte-identical code and data to seed 0.

Seed 0's stamp (LASR-Callum/2026-08-25-qwen36-lora-table2-9284-gpt-responder-685-paired-rank-64/training_meta.json)
pins its data to LASR-Callum/2026-08-25-gpt-responder-685-paired-bundle @ c7994ec2. The code the pod
ran is that repo's code.tar.gz at the same revision. This script pulls both so the seed
bundle is built FROM them rather than from whatever the working tree holds today.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import fire
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

SEED0_ADAPTER = "LASR-Callum/2026-08-25-qwen36-lora-table2-9284-gpt-responder-685-paired-rank-64"
SEED0_BUNDLE = "LASR-Callum/2026-08-25-gpt-responder-685-paired-bundle"
FILES = [
    "code.tar.gz",
    "t2_9284_gptresp685_10k.jsonl",
    "t2_9284_gptresp685_10k.jsonl.stats.json",
    "README.md",
]


def main(dest: str = "/Users/kunwar/.claude/jobs/7e2b4dd7/tmp/seed0_bundle") -> None:
    load_dotenv()
    api = HfApi()
    meta = json.load(open(hf_hub_download(SEED0_ADAPTER, "training_meta.json")))
    rev = meta["dataset"]["revision"]
    assert meta["dataset"]["repo"] == SEED0_BUNDLE, meta["dataset"]
    print("seed0 adapter files:", api.list_repo_files(SEED0_ADAPTER))
    print("seed0 data pin:", meta["dataset"])
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for f in FILES:
        p = hf_hub_download(
            SEED0_BUNDLE, f, repo_type="dataset", revision=rev, local_dir=out
        )
        h = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        hashes[f] = h
        print(f"{f:45s} sha256 {h[:16]}  {Path(p).stat().st_size:>12,d} B")
    (out / "hashes.json").write_text(
        json.dumps({"revision": rev, "sha256": hashes}, indent=2)
    )
    print("--- code.tar.gz members:")
    with tarfile.open(out / "code.tar.gz") as t:
        for m in t.getnames():
            print("  ", m)


if __name__ == "__main__":
    fire.Fire(main)
