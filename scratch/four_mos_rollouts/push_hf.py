# ABOUTME: Publish the four-MO ODCV rollout analysis (feature tables, register/sequence probes, corpus
# ABOUTME: regex tables, the ten reader reports, figure, per-rollout JSONL) to HF with the required card.
"""Run: uv run python scratch/four_mos_rollouts/push_hf.py [--commit <sha>]

The rollouts themselves stay in their four eval repos (named in the card); this repo holds what
was derived from them so the reads and tables are not only on one laptop.
"""

from pathlib import Path

import fire
from dotenv import load_dotenv

from src.huggingface import push_files

REPO = "LASR-Callum/2026-08-27-odcv-four-mos-rollout-analysis"
OUT = Path("output/four_mos_rollouts")


def files() -> list[Path]:
    return [
        *sorted(OUT.glob("ANALYSIS_*.md")),
        *sorted(OUT.glob("features_*.md")),
        *sorted(OUT.glob("sequences_*.md")),
        *sorted(OUT.glob("registers_*.md")),
        *sorted(OUT.glob("validator_read_*.md")),
        *sorted(OUT.glob("corpus_properties_*.md")),
        *sorted(OUT.glob("corpus_properties_*.json")),
        *sorted(OUT.glob("examples_*.md")),
        OUT / "pairs_dump.md",
        *sorted((OUT / "reads").glob("*.md")),
        *sorted((OUT / "plots").glob("four_mos_voice_*")),
        OUT / "index.tsv",
        *sorted(OUT.glob("cells_*.json")),
        *sorted(OUT.glob("rollouts_*.jsonl")),
        *sorted(OUT.glob("registers_*.jsonl")),
        *sorted(OUT.glob("sequences_*.jsonl")),
    ]


def card(commit: str) -> dict:
    return {
        "title": "Four model organisms on ODCV-Bench: rollout analysis (grok / capped Sonnet / Sonnet / gpt)",
        "experiment": (
            "Side-by-side analysis of the ODCV-Bench rollouts of the four generator-ablation model "
            "organisms (Qwen3.6-27B LoRAs trained on 703 paired difficult-advice rows written by "
            "grok-4.6, Sonnet 5 under a length cap, Sonnet 5 (da716) and gpt-5.6), on the same 65 "
            "cells: per-rollout behavioural features joined to both judges' scores, a step-1 register "
            "test (trained voice vs base task-restatement), action-sequence and validator-read "
            "probes, eleven subagent close-reads (six paired cells across arms, four per-arm, one "
            "corpus), and a regex comparison of the four training corpora. Finding: the arms differ "
            "in how often the trained voice fires before the first write (86/71/68/73%), not in how "
            "safe it is when it does (4.5/5.4/4.6% vs gpt 14%); the corpus lever is a firm, repeated "
            "or closing refusal in the reply (71.5/19.2/20.1/10.8%), not length."
        ),
        "date_generated": "2026-08-27",
        "constitution": (
            "constitutions/claude_distilled_12_principles_mid/constitution.md (identical across the "
            "four training corpora); ODCV-Bench scenarios carry no constitution"
        ),
        "source_repo": f"Matthew-Bozoukov/Lessons_from_constituitional_AFT @ {commit} "
        "(branch worktree-odcv-rollouts-four-mos, scratch/four_mos_rollouts/)",
        "models": (
            "Rollouts analysed (not regenerated): LASR-Callum/2026-08-24-odcv-grokresp703-paired-eval "
            "(129), LASR-Callum/2026-08-26-odcv-sonnetconcise703-paired-eval (130), "
            "LASR-Callum/qwen3_6-27b-lora-t2-9284-da716-r64-dynbatch combined4x_20260814_230249 (257), "
            "LASR-Callum/2026-08-25-odcv-gptresp685-paired-eval (127). Judges in those repos: "
            "x-ai/grok-4.20 + google/gemini-3.1-pro-preview (median). Corpora: "
            "LASR-Callum/2026-08-13-difficult-advice-v2, 2026-08-26-difficult-advice-sonnet-concise-716, "
            "2026-08-21-difficult-advice-grok-responder-716, 2026-08-25-difficult-advice-gpt-responder-716. "
            "Reader subagents: Claude (Fable 5), read-only."
        ),
        "generation_config": (
            "No sampling here: deterministic regex/structural features over existing transcripts "
            "(features.py, sequences.py, registers.py, validator_read.py, corpus_properties.py); "
            "exclusions from configs/eval/odcv_bench_t2_9284_sonnetconcise703_r64_paired_2x65.yaml; "
            "violation = median score >= 3."
        ),
        "schema": (
            "rollouts_*.jsonl: one row per rollout — arm, condition, scenario, rollout, cell, key, score "
            "(median), violation, per-judge scores, judge_reasoning, and every feature (n_* counts, "
            "*_chars, r_/rd_ = presence/density-per-1k in reasoning, v_/vd_ = in visible content, "
            "first_refuse_*, final_*). registers_*.jsonl: r1_commit / r1_base_open / r1_engaged / "
            "commit_before_write per rollout. sequences_*.jsonl: seq (R/E/W/X per assistant step), "
            "n_eval_runs, eval_then_write, write_after_bad. cells_*.json: cell -> arm -> scores. "
            "index.tsv: arm, cell, rollout, scores, path. corpus_properties_*.json: binary/continuous/"
            "by_trait/ranking per corpus arm (A_sonnet, C_capped, B_grok, D_gpt). *.md: markdown "
            "mirrors; reads/*.md: the subagent reports; ANALYSIS_*.md: the write-up."
        ),
        "provenance": (
            "uv run python scratch/four_mos_rollouts/pull.py; features.py; sequences.py; registers.py; "
            "validator_read.py; corpus_properties.py; dump_pairs.py; plot_voice.py; push_hf.py. "
            "Reader reports were written by subagents from the transcripts under output/odcv_four_mos/."
        ),
        "notes": "See docs/LOG.md 2026-08-27 and output/four_mos_rollouts/ANALYSIS_20260827.md.",
    }


def main(commit: str = "HEAD", private: bool = False) -> None:
    """Push the analysis with its card.

    Args:
        commit: Git SHA to stamp into the card's source_repo.
        private: Create the repo private (project default is public).
    """
    load_dotenv()
    paths = files()
    missing = [p for p in paths if not p.exists()]
    assert not missing, f"missing {missing}"
    url = push_files(paths, REPO, card(commit), private=private)
    print(f"{len(paths)} files -> {url}")


if __name__ == "__main__":
    fire.Fire(main)
