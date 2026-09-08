# ABOUTME: Publish the judged, combined ODCV run of the COHERENT PAR-716 arm (transcripts + judge scores +
# ABOUTME: results.json) to its eval repo with the card. Fork of scratch/par_b/push_odcv.py.
# Run: uv run python scratch/par_coherence/push_odcv.py --combined <combined dir>

import json
import sys
from pathlib import Path

import fire

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.infra.huggingface import card_markdown, hf_api  # noqa: E402
from src.utils import git_sha, origin_url  # noqa: E402

REPO = "LASR-Callum/2026-08-28-odcv-par716coh-eval"
ADAPTER = "LASR-Callum/2026-08-28-qwen36-lora-table2-9284-post-action-retrospection-716-coherence-rank-64-dynbatch"
CFG = "scratch/par_coherence/odcv_bench_t2_9284_par716coh_2x65.yaml"
CORPUS = "LASR-Callum/2026-08-28-post-action-retrospection-716-coherent"
MIXTURE = "LASR-Callum/2026-08-28-table2-9284-par716coh-train"


def main(
    combined: str, repo: str = REPO, adapter: str = ADAPTER, cfg: str = CFG
) -> None:
    """Push one arm's combined ODCV dir + card.

    Args:
        combined: The combined<N>x_<ts> dir (scratch/odcv_combine_passes.py output).
        repo: Eval dataset repo (the per-pass supervisor already pushed `passes/` there).
        adapter: The adapter that was served.
        cfg: The ODCV config the run used.
    """
    d = (ROOT / combined).resolve()
    res = json.loads((d / "results.json").read_text(encoding="utf-8"))
    o = res["ours"]["overall"]
    api = hf_api()
    api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(
        folder_path=str(d), path_in_repo=d.name, repo_id=repo, repo_type="dataset"
    )
    card = card_markdown(
        {
            "title": "ODCV-Bench: coherent post-action-retrospection 716 arm, 2 rollouts x 65 cells",
            "experiment": (
                f"ODCV-Bench rollouts and judge scores for {adapter}: arm 1 of the PAR coherence "
                "experiment -- the PAR-716 organism retrained on the same 716 five-turn rows with only "
                "the trained turn rewritten so the private reasoning ends on a first-person decision and "
                f"the reply enacts it ({CORPUS}). Headline on these 65 cells: MR {o['mr_pct']}% "
                f"{o['mr_ci95']}, severity {o['mean_severity']}, n={o['n']}. Paired comparison: PAR-716 "
                "19.5% (pooled over 3 seeds), Sonnet difficult advice 16.3%, base fp8 36.9%, same cells."
            ),
            "date_generated": "2026-08-28",
            "constitution": (
                "constitutions/claude_distilled_12_principles_mid/constitution.md (9 principles), the "
                f"same as difficult advice, via the adapter's training data {MIXTURE}"
            ),
            "source_repo": f"{origin_url()} @ {git_sha()}",
            "models": (
                f"target: {adapter} (thinking mode, vLLM, max_model_len 65536); "
                f"judges: {json.dumps(res['judges'])}"
            ),
            "generation_config": json.dumps(
                {
                    "temperature": 0.0,
                    "rollouts_per_cell": 2,
                    "cells": 65,
                    "config": cfg,
                    "concurrency": 12,
                    "driver": "laptop Docker Desktop over an SSH tunnel to a RunPod H100 (scratch/par_b/odcv_local_run.sh)",
                }
            ),
            "schema": (
                "passes/laptop/<run>/: each raw pass as the supervisor pushed it "
                "(agent_logs/.../messages_record.txt = the rollout, rollout_manifest.json, pass_audit.json). "
                "<combined>/: the passes merged into rollout_NNN/ per scenario, evaluations/scores_<judge>.json, "
                "results.json (ours vs the base-fp8 reference), results.md."
            ),
            "provenance": (
                "scratch/odcv_box_run.py --passes 2 --extra concurrency=12; scratch/odcv_combine_passes.py "
                f"--config {cfg}; scratch/odcv_judge_cli.py --rollout_dir <combined> --config {cfg}; "
                "scratch/par_coherence/push_odcv.py"
            ),
            "comparison_arms": (
                "LASR-Callum/2026-08-27-odcv-post-action-retrospection-716-eval (+ -s1-, -s2-), "
                "LASR-Callum/2026-08-14-qwen36-lora-table2-9284-difficult-advice-716-rank-64-dynbatch (combined4x_20260814_230249)"
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
