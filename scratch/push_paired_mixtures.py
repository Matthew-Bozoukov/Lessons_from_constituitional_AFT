# ABOUTME: Push the paired generator-ablation mixtures to HF, so the trainer (which takes
# ABOUTME: data_repo, never a local path) can pin each arm to an exact revision.

"""Publish the two arms of the generator ablation.

Run: uv run python scratch/push_paired_mixtures.py

The two mixtures differ in exactly one thing: who wrote the assistant turn. Same 703
scenario_ids, same byte-identical 9,284-row Table2 half, same trait spread
(79/80/80/80/80/73/75/77/79), same 622 domains, same row count (9,987). Built by
`build_t2_9284_da716_mixture.py --ids_from <the grok corpus>`, which selects by id rather
than sampling, so the pairing is exact rather than merely distributional.

That pairing is the point. The eight existing *716 arms each sampled their synth half
independently, so no two of them are comparable question-for-question; these two are.
"""

from pathlib import Path

import fire
from dotenv import load_dotenv

from src.huggingface import push_files

DATE = "2026-08-24"
COMMON = {
    "date_generated": DATE,
    "constitution": ("constitutions/claude_distilled_12_principles_mid/constitution.md "
                     "(sha fe2ed96093d68a87..., identical in both arms)"),
    "source_repo": "Matthew-Bozoukov/Lessons_from_constituitional_AFT",
    "generation_config": (
        "Mixture build is deterministic (seed 0, selection by scenario_id, not sampled). "
        "Synth-half sampling settings live in each source corpus's own card."),
    "schema": ("JSONL. `source` — difficult_advice_v2 | grok_responder | a Table2 source; "
               "`text` — the fully rendered Qwen chat string "
               "(<|im_start|>{role}\\n{content}<|im_end|>\\n per turn, assistant turns "
               "carrying <think>...</think>); `scenario_id` and `trait_id` on synth rows "
               "only. 9,987 rows = 703 synth + 9,284 Table2 (7.04% synth)."),
}

ARMS = {
    "data/t2_9284_sonnet703_10k.jsonl": {
        "repo": f"LASR-Callum/{DATE}-t2-9284-sonnet703-paired-train",
        "title": "Generator ablation, ARM A (control): Sonnet 5 writes the answers",
        "experiment": (
            "Control arm of the generator ablation. 703 difficult-advice rows whose "
            "assistant turns were written by Haiku 4.5 and revised by Sonnet 5, plus the "
            "9,284-row filtered Table2 half. Paired row-for-row with arm B, which answers "
            "the SAME 703 questions with grok-4.6."),
        "models": ("synth half: anthropic/claude-haiku-4.5 (draft) + anthropic/"
                   "claude-sonnet-5 (revision), via LASR-Callum/"
                   "2026-08-13-difficult-advice-v2; Table2 half: pre-rendered, no model."),
        "provenance": (
            "uv run python scratch/build_t2_9284_da716_mixture.py "
            "--out data/t2_9284_sonnet703_10k.jsonl "
            "--synth_repo LASR-Callum/2026-08-13-difficult-advice-v2 "
            "--synth_file stage_8_export_sft.jsonl --synth_label difficult_advice_v2 "
            "--ids_from <the grok responder corpus>"),
        "notes": ("Median synth response 2,668 chars against arm B's 1,568 (1.70x). At "
                  "MIXTURE level the total-token gap is only ~7%, because the identical "
                  "Table2 half is 93% of the rows."),
    },
    "data/t2_9284_grokresp703_10k.jsonl": {
        "repo": f"LASR-Callum/{DATE}-t2-9284-grokresp703-paired-train",
        "title": "Generator ablation, ARM B: grok-4.6 writes the answers",
        "experiment": (
            "Treatment arm of the generator ablation. The SAME 703 difficult-advice "
            "questions as arm A — same situations, same user turns, same system prompts, "
            "authored by Haiku/Sonnet and reused verbatim — with the assistant turn "
            "written and revised by x-ai/grok-4.6 instead. Plus the same 9,284-row Table2 "
            "half. A downstream difference between the arms is attributable to the model "
            "that wrote the answer."),
        "models": ("synth half: x-ai/grok-4.6 (draft AND revision), via LASR-Callum/"
                   "2026-08-21-difficult-advice-grok-responder-716; prompts inherited from "
                   "anthropic/claude-haiku-4.5 + anthropic/claude-sonnet-5 and NOT "
                   "regenerated; Table2 half: pre-rendered, no model."),
        "provenance": (
            "uv run python scratch/build_t2_9284_da716_mixture.py "
            "--out data/t2_9284_grokresp703_10k.jsonl "
            "--synth_repo LASR-Callum/2026-08-21-difficult-advice-grok-responder-716 "
            "--synth_file dataset.jsonl --synth_label grok_responder "
            "--ids_from <the same corpus>"),
        "notes": (
            "KNOWN CONFOUNDS, measured — see docs/GENERATOR_ABLATION.md. Length: grok's "
            "answers are 1.70x shorter, and a classifier separates the two arms' responses "
            "by LENGTH ALONE at AUC 0.864 (this project called a corpus defective at 0.85). "
            "Report length as a covariate. Structure: grok-4.6 both drafts and revises, "
            "where the control drafts with Haiku and revises with Sonnet, so the "
            "cross-model critique step has no counterpart. Reasoning: mandatory-on for "
            "grok-4.6, off for Haiku/Sonnet. Typography: the arms use disjoint apostrophe "
            "and quote characters, so a trained model inherits one or the other."),
    },
}


def main(private: bool = False) -> None:
    """Push both paired mixtures with their cards.

    Args:
        private: Create the repos private.
    """
    load_dotenv()
    for path, spec in ARMS.items():
        p = Path(path)
        assert p.exists(), f"missing {p} — build the mixtures first"
        fields = {k: v for k, v in spec.items() if k != "repo"}
        url = push_files([p, Path(str(p) + ".stats.json")], spec["repo"],
                         {**fields, **COMMON}, private=private)
        print(f"{p.name} -> {url}")


if __name__ == "__main__":
    fire.Fire(main)
