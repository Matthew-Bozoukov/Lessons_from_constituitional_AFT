# ABOUTME: Publish a channel-swap arm's judged, combined ODCV run (raw passes + combined transcripts +
# ABOUTME: judge scores + results.json) to its eval repo on HF, with the card.
# Run: uv run python scratch/channel_swap/push_odcv.py --arm gtrace_sreply703 --combined <combined dir>
"""Sibling of scratch/sonnet_concise/push_odcv.py, parameterised over the two swap arms."""

import json
import sys
from pathlib import Path

import fire

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.huggingface import card_markdown, hf_api  # noqa: E402
from src.utils import git_sha, origin_url  # noqa: E402

ARMS = {
    "gtrace_sreply703": {
        "repo": "LASR-Callum/2026-08-28-odcv-gtrace-sreply703-paired-eval",
        "adapter": "LASR-Callum/qwen3.6-27b-lora-t2-9284-gtrace-sreply703-paired-r64",
        "cfg": "configs/eval/odcv_bench_t2_9284_gtrace_sreply703_r64_paired_2x65.yaml",
        "data": "LASR-Callum/2026-08-27-t2-9284-gtrace-sreply703-paired-train",
        "what": "grok-4.6's reasoning trace with Sonnet 5's reply",
    },
    "strace_greply703": {
        "repo": "LASR-Callum/2026-08-28-odcv-strace-greply703-paired-eval",
        "adapter": "LASR-Callum/qwen3.6-27b-lora-t2-9284-strace-greply703-paired-r64",
        "cfg": "configs/eval/odcv_bench_t2_9284_strace_greply703_r64_paired_2x65.yaml",
        "data": "LASR-Callum/2026-08-27-t2-9284-strace-greply703-paired-train",
        "what": "Sonnet 5's reasoning trace with grok-4.6's reply",
    },
}


def main(arm: str, combined: str, passes: bool = True) -> None:
    """Upload the run and write the card.

    Args:
        arm: gtrace_sreply703 | strace_greply703.
        combined: The combined<N>x_<ts> directory (holds results.json after judging).
        passes: Also upload the raw single-pass run dirs beside the combined one.
    """
    spec = ARMS[arm]
    d = (ROOT / combined).resolve()
    res = json.loads((d / "results.json").read_text(encoding="utf-8"))
    o = res["ours"]["overall"]
    api = hf_api()
    api.create_repo(spec["repo"], repo_type="dataset", private=False, exist_ok=True)
    if passes:
        for run in sorted(d.parent.glob("2026*")):
            if run.is_dir() and (run / "rollout_manifest.json").exists():
                api.upload_folder(
                    folder_path=str(run),
                    path_in_repo=f"passes/laptop/{run.name}",
                    repo_id=spec["repo"],
                    repo_type="dataset",
                )
                print(f"pushed pass {run.name}")
    api.upload_folder(
        folder_path=str(d),
        path_in_repo=d.name,
        repo_id=spec["repo"],
        repo_type="dataset",
    )
    card = card_markdown(
        {
            "title": f"ODCV-Bench: channel-swap arm ({spec['what']}), 2 rollouts x 65 cells",
            "experiment": (
                f"ODCV-Bench rollouts and judge scores for {spec['adapter']}: a CHANNEL-SWAP arm of the "
                f"generator ablation. Its 703 difficult-advice rows are arms A (Sonnet 5) and B (grok-4.6) "
                f"recombined row-for-row on the same questions -- {spec['what']} -- with no generation. "
                f"Read the two swaps as a 2x2 with A (da716, MR 16.3% [10.0, 21.8]) and B (7.8% [3.6, 13.6]) "
                f"on these cells: the swap that lands near B names the channel carrying grok's effect. "
                f"Headline on these 65 cells: MR {o['mr_pct']}% {o['mr_ci95']}, severity "
                f"{o['mean_severity']}, n={o.get('n_rollouts', o.get('n'))}."
            ),
            "date_generated": "2026-08-28",
            "constitution": (
                "constitutions/claude_distilled_12_principles_mid/constitution.md -- identical "
                f"in both parents; via the adapter's training data {spec['data']}"
            ),
            "source_repo": f"{origin_url()} @ {git_sha()}",
            "models": (
                f"target: {spec['adapter']} (thinking mode pinned, vLLM, agentic parsers, "
                f"max_model_len 16384); judges: {json.dumps(res['judges'])}"
            ),
            "generation_config": json.dumps(
                {
                    "temperature": 0.0,
                    "rollouts_per_cell": 2,
                    "cells": 65,
                    "config": spec["cfg"],
                    "concurrency": 12,
                    "driver": "laptop Docker Desktop against a RunPod H200 vLLM endpoint over RunPod HTTPS proxy; "
                    "both swap adapters served as LoRA modules on the same pod",
                }
            ),
            "schema": (
                "passes/laptop/<run>/: each raw pass (agent_logs/.../messages_record.txt = the rollout, "
                "rollout_manifest.json). <combined>/: the passes merged into rollout_NNN/ per scenario, "
                "evaluations/scores_<judge>.json, results.json (ours vs the base-fp8 reference), results.md."
            ),
            "provenance": (
                f"bash scratch/odcv_repeat_rollouts.sh {spec['cfg']} 2; scratch/odcv_combine_passes.py "
                f"--config {spec['cfg']}; scratch/odcv_judge_cli.py --rollout_dir <combined> --config "
                f"{spec['cfg']}; scratch/channel_swap/push_odcv.py --arm {arm}"
            ),
        }
    )
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=spec["repo"],
        repo_type="dataset",
    )
    print(f"pushed {d.name} -> https://huggingface.co/datasets/{spec['repo']}")


if __name__ == "__main__":
    fire.Fire(main)
