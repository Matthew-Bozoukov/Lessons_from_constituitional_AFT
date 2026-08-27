# ABOUTME: Publish the confidence-autorater outputs (per-row judgments for four corpora + rollouts, second-judge
# ABOUTME: subsample, report, figure, reader notes) to HF with the required dataset card.
"""Run: uv run python scratch/confidence/push_hf.py [--commit <sha>]"""

from pathlib import Path

import fire
from dotenv import load_dotenv

from src.huggingface import push_files

REPO = "LASR-Callum/2026-08-27-difficult-advice-confidence-autorater"
OUT = Path("output/confidence")


def files() -> list[Path]:
    return [
        OUT / "corpus_terra_full.jsonl",
        OUT / "rollouts_terra_full.jsonl",
        OUT / "corpus_gemini_150.jsonl",
        *sorted(OUT.glob("report_*.md")),
        *sorted((OUT / "plots").glob("confidence_four_arms_*")),
        *sorted((OUT / "reads").glob("*.md")),
    ]


def card(commit: str) -> dict:
    return {
        "title": "Confidence autorater: four difficult-advice corpora and the four MOs' ODCV rollouts",
        "experiment": (
            "Does 'confidence' explain the generator ablation (grok 7.8% < capped Sonnet 15.4% ≈ Sonnet 16.3% "
            "< gpt 25.2% ODCV)? A blind LLM judge scores decisiveness, hedging, certainty, deference and overall "
            "confidence (1–7) for the private reasoning and the reply of 678 shared scenarios in each of four "
            "corpora (sonnet, capped sonnet, grok, gpt), and for the first reasoning block of 643 ODCV rollouts of "
            "the organisms trained on them. Result: confidence is real and transfers (grok most confident in "
            "corpus and at inference), but it is not the load-bearing variable — the corpus ordering is grok > gpt "
            "> capped > sonnet (gpt second-most confident, worst on ODCV), the capped arm gained confidence with "
            "no ODCV change, and within arm a confident first block is not safer (Sonnet 26% vs 9%, gpt 34% vs "
            "11% for decisive vs not); in a confidence × commitment 2×2 the first-person commitment explains the "
            "outcome and confidence adds nothing (4.4/4.5% vs 25/24%)."
        ),
        "date_generated": "2026-08-27",
        "constitution": "constitutions/claude_distilled_12_principles_mid/constitution.md (identical across the four corpora)",
        "source_repo": f"Matthew-Bozoukov/Lessons_from_constituitional_AFT @ {commit} (branch worktree-odcv-four-mos-fork, scratch/confidence/)",
        "models": (
            "judge: openai/gpt-5.6-terra (temperature 0, max_tokens 1200 corpus / 900 rollouts), blind to arm; second judge "
            "google/gemini-3.1-pro-preview (temperature 0, max_tokens 4000) on a 150-scenario subsample × 4 arms. "
            "Corpora: LASR-Callum/2026-08-13-difficult-advice-v2, 2026-08-26-difficult-advice-sonnet-concise-716, "
            "2026-08-21-difficult-advice-grok-responder-716, 2026-08-25-difficult-advice-gpt-responder-716. Rollouts: "
            "2026-08-24-odcv-grokresp703-paired-eval, 2026-08-26-odcv-sonnetconcise703-paired-eval, "
            "qwen3_6-27b-lora-t2-9284-da716-r64-dynbatch, 2026-08-25-odcv-gptresp685-paired-eval (65 shared cells)."
        ),
        "generation_config": "temperature 0.0 for both judges; rubric verbatim in scratch/confidence/common.py; seeds: gemini subsample --seed 7.",
        "schema": (
            "corpus_*.jsonl: one row per (corpus, scenario_id) — reasoning{decisiveness,hedging,certainty,deference,"
            "overall_confidence,evidence}, reply{same + stance, ends_with_question}, judge, trait_id. rollouts_*.jsonl: "
            "one row per rollout — reasoning{...} for the FIRST reasoning block, arm, cell, rollout, score (judges' median), "
            "violation (score>=3), r1_chars. report_*.md: per-arm means with paired Wilcoxon tests, confounds (stance, "
            "volitional refusal, prior judge tone, length), per-trait, judge agreement, within-arm outcome tests. "
            "plots/: figure + markdown mirror. reads/: subagent close-reads."
        ),
        "provenance": (
            "uv run python scratch/confidence/rate_corpus.py --corpora sonnet,grok,gpt,capped; rate_corpus.py --judge "
            "google/gemini-3.1-pro-preview --limit 150 --seed 7 --max-tokens 4000; rate_rollouts.py; report.py; "
            "plot_confidence.py; push_hf.py. See docs/LOG.md 2026-08-27."
        ),
    }


def main(commit: str = "HEAD", private: bool = False) -> None:
    load_dotenv()
    paths = files()
    missing = [p for p in paths if not p.exists()]
    assert not missing, f"missing {missing}"
    url = push_files(paths, REPO, card(commit), private=private)
    print(f"{len(paths)} files -> {url}")


if __name__ == "__main__":
    fire.Fire(main)
