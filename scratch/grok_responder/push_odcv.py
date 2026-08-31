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

REPO = "LASR-Callum/2026-08-24-odcv-grok-responder-703-paired-eval"
ADAPTER = "LASR-Callum/2026-08-24-qwen36-lora-table2-9284-grok-responder-703-paired-rank-64"
CFG = "configs/eval/2026-08-24_odcv_bench_table2_9284_grok_responder_703_rank64_paired_2_65.yaml"


def main(combined: str, repo: str = REPO) -> None:
    d = (ROOT / combined).resolve()
    res = json.loads((d / "results.json").read_text(encoding="utf-8"))
    o = res["ours"]["overall"]
    api = hf_api()
    api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(folder_path=str(d), path_in_repo=d.name, repo_id=repo,
                      repo_type="dataset")
    card = card_markdown({
        "title": "ODCV-Bench: grok-responder 703 arm (generator ablation), 2 rollouts x 65 cells",
        "experiment": (
            f"ODCV-Bench rollouts and judge scores for {ADAPTER}: the TREATMENT half "
            "of the generator ablation. Its 703 difficult-advice rows answer the SAME "
            "questions as the da716 baseline -- same situations, user turns and system "
            "prompts, reused verbatim -- with the assistant turn written and revised by "
            "x-ai/grok-4.6 instead of Haiku-drafted/Sonnet-revised. Headline on these 65 "
            f"cells: MR {o['mr_pct']}% {o['mr_ci95']}, severity {o['mean_severity']}, "
            f"n={o['n']} -- the LOWEST of any 7% SFT arm, against da716's 16.3% "
            "[10.0, 21.8]. READ THE CAVEAT: grok's corpus is 1.70x shorter than the "
            "baseline's and a classifier separates the two by LENGTH ALONE at AUC 0.864, "
            "so this difference is a package effect of values + verbosity + rhetorical "
            "style, NOT evidence about values alone. Per 1,000 chars the grok corpus also "
            "refuses ~2.6x more densely and offers ~3.9x fewer alternatives, a plausible "
            "mechanism for a lower agentic misalignment rate. See "
            "docs/GENERATOR_ABLATION.md."),
        "date_generated": "2026-08-24",
        "constitution": (
            "constitutions/claude_distilled_12_principles_mid/constitution.md -- IDENTICAL "
            "to the da716 baseline's and unchanged by this arm: only the model writing "
            "the assistant turn differs. Via the adapter's training data "
            "LASR-Callum/2026-08-24-table2-9284-grok-responder-703-paired-train-mixture"),
        "source_repo": f"{origin_url()} @ {git_sha()}",
        "models": (f"target: {ADAPTER} (thinking mode, vLLM, max_model_len 65536); "
                   f"judges: {json.dumps(res['judges'])}"),
        "generation_config": json.dumps({
            "temperature": 0.0, "rollouts_per_cell": 2, "cells": 65,
            "config": CFG, "concurrency": 12, "driver": "laptop Docker Desktop against a RunPod H200 vLLM "
            "endpoint over RunPod HTTPS proxy (no SSH tunnel)"}),
        "schema": ("passes/laptop/<run>/: each raw pass as the supervisor pushed it "
                   "(agent_logs/.../messages_record.txt = the rollout, rollout_manifest.json, "
                   "pass_audit.json). <combined>/: the two passes merged into "
                   "rollout_NNN/ per scenario, evaluations/scores_<judge>.json, "
                   "results.json (ours vs the base-fp8 reference), results.md."),
        "provenance": (
            "bash scratch/odcv_repeat_rollouts.sh " + CFG + " 2; "
            "scratch/odcv_combine_passes.py --config " + CFG + "; "
            "scratch/odcv_judge_cli.py --rollout_dir <combined> --config " + CFG + "; "
            "scratch/grok_responder/push_odcv.py"),
    })
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo,
                    repo_type="dataset")
    print(f"pushed {d.name} -> https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    fire.Fire(main)
