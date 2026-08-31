# ABOUTME: Publish the judged, combined ODCV run of the trait-10 arm (transcripts + judge scores +
# ABOUTME: results.json) to the eval repo the per-pass supervisor already pushed into, with the card.
# Run: uv run python scratch/trait10_curiosity/push_odcv.py --combined <combined dir>

import json
import sys
from pathlib import Path

import fire

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.huggingface import card_markdown, hf_api  # noqa: E402
from src.utils import git_sha, origin_url  # noqa: E402

REPO = "LASR-Callum/2026-08-20-odcv-t10-curiosity-716-eval"
ADAPTER = "LASR-Callum/2026-08-20-qwen36-lora-table2-9284-t10-curiosity-716-rank-64-dynbatch"
CFG = "scratch/trait10_curiosity/odcv_bench_t2_9284_t10_curiosity_716_2x65.yaml"


def main(combined: str, repo: str = REPO) -> None:
    d = (ROOT / combined).resolve()
    res = json.loads((d / "results.json").read_text(encoding="utf-8"))
    o = res["ours"]["overall"]
    api = hf_api()
    api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(folder_path=str(d), path_in_repo=d.name, repo_id=repo,
                      repo_type="dataset")
    card = card_markdown({
        "title": "ODCV-Bench: trait-10 (curiosity) 716 arm, 2 rollouts x 65 cells",
        "experiment": (
            f"ODCV-Bench rollouts and judge scores for {ADAPTER}: the da716 organism whose "
            "716 difficult-advice rows were written against ONE appended principle (genuine "
            "intellectual curiosity) that ODCV cannot reward. Headline on these 65 cells: "
            f"MR {o['mr_pct']}% {o['mr_ci95']}, severity {o['mean_severity']}, n={o['n']}."),
        "date_generated": "2026-08-20",
        "constitution": (
            "scratch/trait10_curiosity/constitution.md in the source repo (the 9-principle "
            "claude_distilled_12_principles_mid document + principle 10), via the adapter's "
            "training data LASR-Callum/2026-08-20-table2-9284-t10-curiosity-716-train"),
        "source_repo": f"{origin_url()} @ {git_sha()}",
        "models": (f"target: {ADAPTER} (thinking mode, vLLM, max_model_len 65536); "
                   f"judges: {json.dumps(res['judges'])}"),
        "generation_config": json.dumps({
            "temperature": 0.0, "rollouts_per_cell": 2, "cells": 65,
            "config": CFG, "concurrency": 12, "driver": "laptop Docker Desktop over an SSH "
            "tunnel to a RunPod H100 (scratch/trait10_curiosity/odcv_local_run.sh)"}),
        "schema": ("passes/laptop/<run>/: each raw pass as the supervisor pushed it "
                   "(agent_logs/.../messages_record.txt = the rollout, rollout_manifest.json, "
                   "pass_audit.json). <combined>/: the two passes merged into "
                   "rollout_NNN/ per scenario, evaluations/scores_<judge>.json, "
                   "results.json (ours vs the base-fp8 reference), results.md."),
        "provenance": (
            "scratch/odcv_box_run.py --passes 2 --extra concurrency=12; "
            "scratch/odcv_combine_passes.py --config " + CFG + "; "
            "scratch/odcv_judge_cli.py --rollout_dir <combined> --config " + CFG + "; "
            "scratch/trait10_curiosity/push_odcv.py"),
    })
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo,
                    repo_type="dataset")
    print(f"pushed {d.name} -> https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    fire.Fire(main)
