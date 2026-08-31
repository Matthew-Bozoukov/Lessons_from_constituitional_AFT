# ABOUTME: Pull the published seed-0 GPT-arm ODCV run (judge scores + results.json) into the local
# ABOUTME: output/odcv_bench layout so scratch/stats/odcv_seed_sem.py can read all three seeds alike.
# Run: uv run python scratch/gpt_seeds/fetch_seed0_eval.py
from __future__ import annotations

from pathlib import Path

import fire
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[2]
REPO = "LASR-Callum/2026-08-25-odcv-gptresp685-paired-eval"
RUN = "combined2x_20260825_181731"
MODEL_KEY = "qwen3_6-27b-lora-t2-9284-gptresp685-paired-r64"


def main() -> None:
    load_dotenv(ROOT / ".env")
    dest = ROOT / "output/odcv_bench" / MODEL_KEY
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        REPO,
        repo_type="dataset",
        local_dir=dest,
        allow_patterns=[
            f"{RUN}/evaluations/*",
            f"{RUN}/results.json",
            f"{RUN}/results.md",
            f"{RUN}/run_meta.json",
            f"{RUN}/combine_manifest.json",
        ],
    )
    got = sorted(p.relative_to(dest) for p in (dest / RUN).rglob("*") if p.is_file())
    print("\n".join(str(p) for p in got))


if __name__ == "__main__":
    fire.Fire(main)
