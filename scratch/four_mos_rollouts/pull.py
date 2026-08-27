# ABOUTME: Pull the ODCV rollouts + judge scores of the four generator-ablation model organisms
# ABOUTME: (grok / gpt / sonnet-normal=da716 / sonnet-concise) from their HF eval repos into output/.
# Run: uv run python scratch/four_mos_rollouts/pull.py
from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path("output/odcv_four_mos")

# arm -> (HF dataset repo, allow_patterns). Layouts differ per repo (see docs/LOG.md 2026-08-25:
# the published contract layout `rollouts/<cond>/<scenario>/passN/` arrived 2026-08-24; older
# repos carry the raw `combinedNx_*/agent_logs/...` bench layout).
ARMS: dict[str, tuple[str, list[str]]] = {
    "sonnet_normal": (
        "LASR-Callum/qwen3_6-27b-lora-t2-9284-da716-r64-dynbatch",
        [
            "README.md",
            "combined4x_20260814_230249/**/messages_record.txt",
            "combined4x_20260814_230249/evaluations/*",
            "combined4x_20260814_230249/results.json",
        ],
    ),
    "grok": (
        "LASR-Callum/2026-08-24-odcv-grokresp703-paired-eval",
        [
            "README.md",
            "rollouts/**/messages_record.txt",
            "rollouts/**/cell_meta.json",
            "results/*.json",
            "results/*.md",
            "metadata/*",
        ],
    ),
    "gpt": (
        "LASR-Callum/2026-08-25-odcv-gptresp685-paired-eval",
        [
            "README.md",
            "combined2x_20260825_181731/**/messages_record.txt",
            "combined2x_20260825_181731/evaluations/*",
            "combined2x_20260825_181731/results.json",
            "combined2x_20260825_181731/run_meta.json",
            "combined2x_20260825_181731/combine_manifest.json",
        ],
    ),
    "sonnet_concise": (
        "LASR-Callum/2026-08-26-odcv-sonnetconcise703-paired-eval",
        [
            "README.md",
            "combined2x_20260826_174216/**/messages_record.txt",
            "combined2x_20260826_174216/evaluations/*",
            "combined2x_20260826_174216/results.json",
            "combined2x_20260826_174216/run_meta.json",
            "combined2x_20260826_174216/combine_manifest.json",
        ],
    ),
}


def main() -> None:
    for arm, (repo, pats) in ARMS.items():
        d = snapshot_download(
            repo, repo_type="dataset", allow_patterns=pats, local_dir=str(ROOT / arm)
        )
        n = sum(1 for _, _, fs in os.walk(d) for f in fs if f == "messages_record.txt")
        print(f"{arm:15s} {repo:60s} messages_record.txt: {n}")


if __name__ == "__main__":
    main()
