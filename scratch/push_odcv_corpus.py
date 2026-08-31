# ABOUTME: Publish the whole output/odcv_bench corpus to Hugging Face, excluding container
# ABOUTME: stdout, container scratch and loose zips, with the repo-required dataset card.

"""Upload every ODCV-Bench run directory to one HF dataset repo.

Uploads in place with ignore_patterns rather than staging a copy, so no 370MB duplicate is
written to disk and the source tree is never mutated (the card is uploaded from memory).

Run: uv run python scratch/push_odcv_corpus.py [--dry_run]
"""

from pathlib import Path

import fire
from dotenv import load_dotenv

from src.huggingface import card_markdown, hf_api

load_dotenv()

ROOT = Path("output/odcv_bench")
REPO = "matboz/2026-08-21-odcv-bench-corpus-qwen36-27b"

# Reproduction artifacts, not results: container stdout (CLAUDE.md is explicit that
# docker_output.log is NOT the rollout), container scratch, and redundant archives.
IGNORE = ["**/docker_output.log", "**/workspaces/**", "**/*.zip"]

FIELDS = {
    "title": "ODCV-Bench corpus — every Qwen3.6-27B arm",
    "experiment": (
        "The complete ODCV-Bench agentic-misalignment evaluation record for the "
        "constitutional-SFT method work: 33 arms x 133 runs x 13,123 agent rollouts, "
        "covering the FP8 base model, instruction-tuning-only controls, difficult-advice "
        "mixtures at several ratios, trait-subset and trait-swap ablations, a NuminaMath "
        "no-difficult-advice control, and the courtroom / peer-critique / self-reflection "
        "document-type arms. See INDEX.md for the per-arm table."
    ),
    "date_generated": "2026-07-28 to 2026-08-19 (per-run timestamps in each run_meta.json)",
    "constitution": (
        "constitutions/claude_distilled_12_principles_mid/constitution.md for the "
        "difficult-advice, trait-subset, courtroom and peer-critique arms; none for the "
        "base-model, instruction-tuning-only and NuminaMath control arms. Each run's "
        "training provenance is in its run_meta.json config block."
    ),
    "source_repo": (
        "github.com/Matthew-Bozoukov/teaching_claude_why_replication — per-run git SHA in "
        "every run_meta.json; arms span roughly e3267a6 to 7acd0ff"
    ),
    "models": (
        "targets: Qwen/Qwen3.6-27B plus ~30 LoRA adapters (r=64, alpha=128) under "
        "matboz/ and LASR-Callum/, each named by its arm; served by vLLM in thinking mode. "
        "judges: x-ai/grok-4.20 and google/gemini-3.1-pro-preview via OpenRouter "
        "(median-of-judges, a judge never scores its own trajectories)."
    ),
    "generation_config": (
        "temperature 0.0; mandated + incentivized conditions; scenario count and "
        "exclude_scenarios vary per arm (65-70 scenarios typical) and are recorded per run; "
        "3-5 rollouts per cell; scenario_timeout_s 2400."
    ),
    "schema": (
        "<arm>/<run_ts>/agent_logs/<arm>-<condition>/experiments/<Scenario>/rollout_NNN/"
        "messages_record.txt = one self-contained rollout (task prompt + the agent's "
        "reasoning and tool calls). <arm>/<run_ts>/results.json = judged misalignment rate "
        "(share of rollout-level median severities >= 3), mean severity, 95% CIs, "
        "per-condition splits and per-scenario medians. run_meta.json = git SHA, full "
        "config, exact command. rollout_manifest.json = which cells a pass produced. "
        "combined*/ directories merge several passes and are judged as one set. "
        "INDEX.md lists every arm."
    ),
    "provenance": (
        "generation: `uv run python scratch/odcv_rollout_cli.py --config "
        "configs/eval/odcv_bench_<arm>.yaml` (older arms: scratch/odcv_run_pass.py); "
        "judging: `uv run python scratch/odcv_judge_run.py <run dir> <config>`. "
        "Upload: scratch/push_odcv_corpus.py."
    ),
    "excluded_from_upload": (
        "docker_output.log (container stdout, not the rollout), workspaces/ (container "
        "scratch) and loose .zip archives — reproduction artifacts, ~258MB, regenerable."
    ),
    "comparability_warning": (
        "Arms differ in scenario count and exclusion list, so MR is NOT comparable across "
        "arms without first restricting to the cells both arms ran. INDEX.md repeats this."
    ),
}


def main(dry_run: bool = False) -> None:
    """Create the repo and upload the corpus."""
    card = card_markdown(FIELDS)  # validates required fields before any network call
    n_roll = len(list(ROOT.rglob("messages_record.txt")))
    print(f"{n_roll} rollouts under {ROOT}; ignoring {IGNORE}")
    if dry_run:
        print(card)
        return

    api = hf_api()
    api.create_repo(REPO, repo_type="dataset", private=False, exist_ok=True)
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                    repo_id=REPO, repo_type="dataset")
    api.upload_folder(folder_path=str(ROOT), repo_id=REPO, repo_type="dataset",
                      ignore_patterns=IGNORE)
    print(f"https://huggingface.co/datasets/{REPO}")


if __name__ == "__main__":
    fire.Fire(main)
