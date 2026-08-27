# ABOUTME: Publish the four-arm corpus comparison (judge output + every table + figures) to HF,
# ABOUTME: so the blind-judge verdicts on the capped arm are not only on one laptop.

"""Run: uv run python scratch/sonnet_concise/push_four_way.py

Judge output is a paid, reproducible-only-at-cost artifact and the repo policy is that such
things live on Hugging Face. This pushes the capped arm's judgments alongside the tables
that compare all four arms, with the card fields CLAUDE.md requires.
"""

from pathlib import Path

import fire
from dotenv import load_dotenv

from src.huggingface import push_files

REPO = "LASR-Callum/2026-08-26-difficult-advice-four-way-corpus-stats"
FILES = [
    "scratch/three_way/judged_capped.jsonl",
    # The sonnet/grok/gpt judgments this arm joins (2026-08-25). They were never committed and
    # the worktree that produced them is gone, so this repo is their durable home.
    "scratch/grok_vs_sonnet/judged.jsonl",
    "scratch/three_way/judged_gpt.jsonl",
    "scratch/three_way/judged_neutral.jsonl",
    "scratch/gpt_voice/metrics_table.json",
    *sorted(Path("output/sonnet_concise/four_way").glob("*.txt")),
    *sorted(Path("output/sonnet_concise").glob("lengths_four_arms_*.png")),
    *sorted(Path("output/sonnet_concise").glob("lengths_four_arms_*.md")),
]
CARD = {
    "title": "Four-arm difficult-advice corpus comparison: sonnet, capped sonnet, grok, gpt",
    "experiment": (
        "Corpus-level comparison of four difficult-advice SFT corpora that answer the same "
        "678 prompts: A da716 (Haiku draft, Sonnet 5 rewrite), C capped Sonnet (same drafts, "
        "Sonnet 5 rewrite under a 220/270-word cap), B grok-4.6, D gpt-5.6. Blind-judge "
        "stances (refuses/partial/complies/no_shortcut, refusal form, alternatives) plus "
        "length, voice and structure metrics. The question: does shortening Sonnet change "
        "anything besides length? Answer: not refusal (p=1.0 vs unconstrained Sonnet)."
    ),
    "date_generated": "2026-08-26",
    "constitution": (
        "constitutions/claude_distilled_12_principles_mid/constitution.md "
        "(sha fe2ed96093d68a87..., identical across the four corpora)"
    ),
    "source_repo": "Matthew-Bozoukov/Lessons_from_constituitional_AFT",
    "models": (
        "judge: openai/gpt-5.6-terra, temperature 0.0, max_tokens 900, blind to "
        "corpus identity (scratch/three_way/judge.py, rubric verbatim from the "
        "2026-08-25 three-way pass). Corpora judged: LASR-Callum/2026-08-13-"
        "difficult-advice-v2, 2026-08-26-difficult-advice-sonnet-concise-716, "
        "2026-08-21-difficult-advice-grok-responder-716, 2026-08-25-difficult-"
        "advice-gpt-responder-716."
    ),
    "generation_config": (
        "judge temperature 0.0; metrics are deterministic regex/word counts "
        "on curly-punctuation-normalised text (scratch/three_way/norm.py)."
    ),
    "schema": (
        "judged_capped.jsonl: one JSON per scenario — corpus, scenario_id, trait_id, "
        "judge, stance, stance_evidence, refusal_explicit, refusal_names_action, "
        "refusal_position, refusal_tone, n_alternatives, alternative_kinds, "
        "alternatives_specific, alternatives_terse. metrics_table.json: [label, "
        "[sonnet, grok, gpt, capped]] rows. *.txt: the scripts' printed tables "
        "(column order sonnet, grok, gpt, capped). lengths_four_arms_*: figure + "
        "markdown mirror."
    ),
    "provenance": (
        "uv run python scratch/three_way/judge.py --corpora capped --judge "
        "openai/gpt-5.6-terra --out scratch/three_way/judged_capped.jsonl; then "
        "scratch/three_way/{agg,stats,by_trait,does_the_work,refusal_forms,"
        "length_decomp}.py, scratch/gpt_voice/{metrics,substance}.py, "
        "scratch/sonnet_concise/plot_lengths.py; pushed by "
        "scratch/sonnet_concise/push_four_way.py"
    ),
    "notes": (
        "The sonnet/grok/gpt judgments this joins were produced 2026-08-25 (not in this "
        "repo). n=677 scenarios judged in all four. See docs/LOG.md 2026-08-26."
    ),
}


def main(private: bool = False) -> None:
    """Push the files with their card.

    Args:
        private: Create the repo private (the project default is public).
    """
    load_dotenv()
    paths = [Path(f) for f in FILES]
    missing = [p for p in paths if not p.exists()]
    assert not missing, f"missing {missing}"
    url = push_files(paths, REPO, CARD, private=private)
    print(f"{len(paths)} files -> {url}")


if __name__ == "__main__":
    fire.Fire(main)
