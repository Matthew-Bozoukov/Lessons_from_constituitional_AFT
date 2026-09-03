# ABOUTME: Publish arm C's judged, combined ODCV run (raw passes + combined transcripts + judge
# ABOUTME: scores + results.json) to its eval repo on HF, with the card.
# Run: uv run python scratch/sonnet_concise/push_odcv.py --combined <combined dir>

"""Sibling of scratch/grok_responder/push_odcv.py for the length-capped Sonnet arm.

Unlike the grok arm (whose passes a box supervisor pushed as they landed), this arm's
passes were driven from the laptop by scratch/odcv_repeat_rollouts.sh, so the raw pass
run-dirs are uploaded here too, under passes/laptop/<run>/, before the combined dir.
"""

import json
import sys
from pathlib import Path

import fire

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.infra.huggingface import card_markdown, hf_api  # noqa: E402
from src.utils import git_sha, origin_url  # noqa: E402

REPO = "LASR-Callum/2026-08-26-odcv-sonnet-concise-703-paired-eval"
ADAPTER = "LASR-Callum/2026-08-26-qwen36-lora-table2-9284-sonnet-concise-703-paired-rank-64"
CFG = "configs/eval/2026-08-26_odcv_bench_table2_9284_sonnet_concise_703_rank64_paired_2_65.yaml"
MODEL_KEY = "qwen3_6-27b-lora-t2-9284-sonnetconcise703-paired-r64"


def main(combined: str, repo: str = REPO, passes: bool = True) -> None:
    """Upload the run and write the card.

    Args:
        combined: The combined<N>x_<ts> directory (holds results.json after judging).
        repo: HF dataset repo to publish into.
        passes: Also upload the raw single-pass run dirs beside the combined one.
    """
    d = (ROOT / combined).resolve()
    res = json.loads((d / "results.json").read_text(encoding="utf-8"))
    o = res["ours"]["overall"]
    api = hf_api()
    api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
    if passes:
        for run in sorted(d.parent.glob("2026*")):
            if run.is_dir() and (run / "rollout_manifest.json").exists():
                api.upload_folder(
                    folder_path=str(run),
                    path_in_repo=f"passes/laptop/{run.name}",
                    repo_id=repo,
                    repo_type="dataset",
                )
                print(f"pushed pass {run.name}")
    api.upload_folder(
        folder_path=str(d), path_in_repo=d.name, repo_id=repo, repo_type="dataset"
    )
    card = card_markdown(
        {
            "title": "ODCV-Bench: length-capped Sonnet 703 arm (arm C of the generator ablation), 2 rollouts x 65 cells",
            "experiment": (
                f"ODCV-Bench rollouts and judge scores for {ADAPTER}: the LENGTH CONTROL of the "
                "generator ablation. Its 703 difficult-advice rows answer the SAME questions as "
                "arm A (da716, Sonnet 5 unconstrained, MR 16.3% [10.0, 21.8]) and arm B (grok-4.6, "
                "MR 7.8% [3.6, 13.6]) on these cells; the assistant turn is the baseline's own "
                "Haiku draft rewritten by the baseline's own Sonnet 5 under a one-sentence cap at "
                "grok's median lengths (reasoning ~220 words, reply ~270). Corpus-level: length "
                "AUC vs grok 0.42, blind-judged refusal identical to arm A (83.6% vs 83.8%, p=1.0). "
                f"Headline on these 65 cells: MR {o['mr_pct']}% {o['mr_ci95']}, severity "
                f"{o['mean_severity']}, n={o['n']}. Read: near B means length carried B's drop; "
                "near A means the generator did."
            ),
            "date_generated": "2026-08-26",
            "constitution": (
                "constitutions/claude_distilled_12_principles_mid/constitution.md -- IDENTICAL to "
                "the da716 baseline's and unchanged by this arm: only the rewrite's length differs. "
                "Via the adapter's training data LASR-Callum/2026-08-26-table2-9284-sonnet-concise-703-paired-train"
            ),
            "source_repo": f"{origin_url()} @ {git_sha()}",
            "models": (
                f"target: {ADAPTER} (thinking mode, vLLM, max_model_len 65536); "
                f"judges: {json.dumps(res['judges'])}"
            ),
            "generation_config": json.dumps(
                {
                    "temperature": 0.0,
                    "rollouts_per_cell": 2,
                    "cells": 65,
                    "config": CFG,
                    "concurrency": 12,
                    "driver": "laptop Docker Desktop against a RunPod H200 vLLM endpoint over RunPod "
                    "HTTPS proxy (no SSH tunnel)",
                }
            ),
            "schema": (
                "passes/laptop/<run>/: each raw pass (agent_logs/.../messages_record.txt = "
                "the rollout, rollout_manifest.json). <combined>/: the passes merged into "
                "rollout_NNN/ per scenario, evaluations/scores_<judge>.json, results.json "
                "(ours vs the base-fp8 reference), results.md."
            ),
            "provenance": (
                "bash scratch/odcv_repeat_rollouts.sh " + CFG + " 2; "
                "scratch/odcv_combine_passes.py --config " + CFG + "; "
                "scratch/odcv_judge_cli.py --rollout_dir <combined> --config "
                + CFG
                + "; "
                "scratch/sonnet_concise/push_odcv.py"
            ),
        }
    )
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="dataset",
    )
    print(f"pushed {d.name} -> https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    fire.Fire(main)
