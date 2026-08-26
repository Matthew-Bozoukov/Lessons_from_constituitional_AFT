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

REPO = "LASR-Callum/2026-08-25-odcv-gptresp685-paired-eval"
ADAPTER = "LASR-Callum/qwen3.6-27b-lora-t2-9284-gptresp685-paired-r64"
CFG = "configs/eval/odcv_bench_t2_9284_gptresp685_r64_paired_2x65.yaml"


def main(combined: str, repo: str = REPO) -> None:
    d = (ROOT / combined).resolve()
    res = json.loads((d / "results.json").read_text(encoding="utf-8"))
    o = res["ours"]["overall"]
    api = hf_api()
    api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(folder_path=str(d), path_in_repo=d.name, repo_id=repo,
                      repo_type="dataset")
    card = card_markdown({
        "title": "ODCV-Bench: GPT-responder 685 arm (generator ablation), 2 rollouts x 65 cells",
        "experiment": (
            f"ODCV-Bench rollouts and judge scores for {ADAPTER}: the GPT half of the "
            "generator ablation. Its 685 difficult-advice rows answer the SAME questions "
            "as the da716 baseline and the grok arm -- same situations, user turns and "
            "system prompts, reused verbatim -- with the assistant turn DRAFTED by "
            "openai/gpt-5.6-luna and REVISED by openai/gpt-5.6-terra. Headline on these "
            f"65 cells: MR {o['mr_pct']}% {o['mr_ci95']}, severity {o['mean_severity']}, "
            f"n={o['n']}. "
            "THE RESULT THAT MATTERS IS THE ORDERING ACROSS THREE ARMS, not this number "
            "alone: grok 7.8% [3.6, 13.6] < baseline 16.3% [10.0, 21.8] < GPT 25.2% "
            "[15.1, 34.9], and severity 0.35 < 0.76 < 1.07 -- which is EXACTLY the order "
            "of their response lengths (0.59x, 1.00x, 1.57x vs the baseline). On "
            "identical questions, misalignment tracks verbosity monotonically. That makes "
            "length the parsimonious explanation for the whole sweep, and it means none "
            "of these three numbers can yet be attributed to a generator's VALUES. The "
            "control this needs is a length-matched arm. See docs/GENERATOR_ABLATION.md."),
        "date_generated": "2026-08-25",
        "constitution": (
            "constitutions/claude_distilled_12_principles_mid/constitution.md -- IDENTICAL "
            "to the baseline's and the grok arm's, and unchanged by this arm: only the "
            "models writing the assistant turn differ. Via the adapter's training data "
            "LASR-Callum/2026-08-25-t2-9284-gptresp685-paired-train"),
        "source_repo": f"{origin_url()} @ {git_sha()}",
        "models": (
            f"target: {ADAPTER} (thinking mode, vLLM, max_model_len 65536) -- NOTE this "
            "adapter is checkpoint-600 of 624 steps: the run crashed on a final "
            "1-example DDP batch (9,969 rows at global batch 16 leaves remainder 1, which "
            "route_step cannot split across 2 ranks). LR at the checkpoint was 4.4e-07 "
            "and the 24 missing steps ran down to 1.8e-08, so 96.2% of an epoch with the "
            "remainder 200-5000x below peak LR. Corpus models: openai/gpt-5.6-luna (draft) "
            "+ openai/gpt-5.6-terra (revision), BOTH with reasoning disabled, matching the "
            "baseline's non-thinking regime -- which the grok and gemini arms could not do. "
            "judges: {judges}"),
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
            "scratch/grok_responder/push_odcv_gpt.py"),
    })
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo,
                    repo_type="dataset")
    print(f"pushed {d.name} -> https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    fire.Fire(main)
