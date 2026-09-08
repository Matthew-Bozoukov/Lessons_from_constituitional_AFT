# ABOUTME: Publish the judged, combined ODCV run of the trait-10 arm (transcripts + judge scores +
# ABOUTME: results.json) to the eval repo the per-pass supervisor already pushed into, with the card.
# Run: uv run python scratch/par_b/push_odcv.py --combined <combined dir>

import json
import sys
from pathlib import Path

import fire

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.infra.huggingface import card_markdown, hf_api  # noqa: E402
from src.utils import git_sha, origin_url  # noqa: E402

REPO = "LASR-Callum/2026-08-27-odcv-post-action-retrospection-716-eval"
ADAPTER = "LASR-Callum/2026-08-26-qwen36-lora-table2-9284-post-action-retrospection-716-rank-64-dynbatch"
CFG = "scratch/par_b/odcv_bench_t2_9284_par716_2x65.yaml"


def main(
    combined: str,
    repo: str = REPO,
    adapter: str = ADAPTER,
    cfg: str = CFG,
    seed: int = 0,
) -> None:
    """Push one arm's combined ODCV dir + card.

    Args:
        combined: The combined<N>x_<ts> dir (scratch/odcv_combine_passes.py output).
        repo: Eval dataset repo (the per-pass supervisor already pushed `passes/` there).
        adapter: The adapter that was served.
        cfg: The ODCV config the run used.
        seed: Training seed of the adapter (0 = the original arm; 1, 2 = replicates).
    """
    d = (ROOT / combined).resolve()
    res = json.loads((d / "results.json").read_text(encoding="utf-8"))
    o = res["ours"]["overall"]
    api = hf_api()
    api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(
        folder_path=str(d), path_in_repo=d.name, repo_id=repo, repo_type="dataset"
    )
    ADAPTER = adapter  # noqa: N806 -- shadows the module default for the f-strings below
    CFG = cfg  # noqa: N806
    replicate = (
        ""
        if seed == 0
        else (
            f" This is the SEED-{seed} training replicate of that arm (same data, same recipe, "
            "different LoRA init and shuffle order), run under the identical ODCV protocol so "
            "the seeds can be read side by side and pooled."
        )
    )
    card = card_markdown(
        {
            "title": (
                "ODCV-Bench: post-action-retrospection (design B) 716 arm"
                + (f", seed {seed}" if seed else "")
                + ", 2 rollouts x 65 cells"
            ),
            "experiment": (
                f"ODCV-Bench rollouts and judge scores for {ADAPTER}: the da716 organism whose "
                "716 rows are five-turn post-action-retrospection records (a difficult-advice "
                "prompt, a bare refusal, pushback, then the reasoning the refusal skipped; only "
                "the last turn trained). Headline on these 65 cells: "
                f"MR {o['mr_pct']}% {o['mr_ci95']}, severity {o['mean_severity']}, n={o['n']}."
                + replicate
            ),
            "date_generated": "2026-08-27",
            "constitution": (
                "constitutions/claude_distilled_12_principles_mid/constitution.md (9 principles), the "
                "same as difficult advice, via the adapter's training data "
                "LASR-Callum/2026-08-26-table2-9284-post-action-retrospection-716-train"
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
                    "driver": "laptop Docker Desktop over an SSH "
                    "tunnel to a RunPod H100 (scratch/par_b/odcv_local_run.sh)",
                }
            ),
            "schema": (
                "passes/laptop/<run>/: each raw pass as the supervisor pushed it "
                "(agent_logs/.../messages_record.txt = the rollout, rollout_manifest.json, "
                "pass_audit.json). <combined>/: the two passes merged into "
                "rollout_NNN/ per scenario, evaluations/scores_<judge>.json, "
                "results.json (ours vs the base-fp8 reference), results.md."
            ),
            "provenance": (
                "scratch/odcv_box_run.py --passes 2 --extra concurrency=12; "
                "scratch/odcv_combine_passes.py --config " + CFG + "; "
                "scratch/odcv_judge_cli.py --rollout_dir <combined> --config "
                + CFG
                + "; "
                "scratch/par_b/push_odcv.py"
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
