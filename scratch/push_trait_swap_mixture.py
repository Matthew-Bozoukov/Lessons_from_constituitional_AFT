# ABOUTME: Publish the traits-1/3/4-removed mixture: 9,284 standard SFT rows plus a
# ABOUTME: difficult-advice share backfilled evenly from the six remaining traits.

"""Push the trait-ablation arm.

This arm exists to disambiguate the low-ODCV result. That arm raised the ODCV misalignment
rate, but its scenario swap was inseparable from trait coverage: trait 1 IS the
agentic-oversight trait, so "removed the eval-like scenarios" and "removed t1" were the same
manipulation. Here the trait side is applied directly and the ODCV-likeness of the mix is
left much closer to the control, so the two explanations pull apart.

The card records both the manipulation and the residual ODCV-likeness drift, because the
drift is what limits how cleanly the two arms can be contrasted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.huggingface import card_front_matter, card_markdown, hf_api, push_files  # noqa: E402
from src.utils import git_sha  # noqa: E402

REPO = "matboz/2026-08-18-traits134-removed-t2-9284-synthdoc-716"
RUN = Path("output/trait_swap/20260818_204523")
CONTROL = "LASR-Callum/2026-08-06-table2-9284-synthdoc-716-train"

TRAITS = {
    "t1": "Preserve human oversight; avoid unilateral, power-accruing action",
    "t3": "Be scrupulously honest and non-deceptive, in word, framing, and action",
    "t4": "Weigh real-world harm with calibrated, policy-level judgement",
}


def main(private: bool = True) -> None:
    """Upload the trait-swap mixture and its swap record.

    Args:
        private: Create the repo private (default).
    """
    st = json.loads((RUN / "mixture_stats_traitswap.json").read_text())
    dropped = ", ".join(f"{t} ({TRAITS[t]})" for t in st["dropped_traits"])
    fields = {
        "title": "Traits-1/3/4-removed mixture (Table2 9,284 + difficult-advice 716)",
        "experiment": "Trait-ablation arm. The reasoning text is UNMODIFIED throughout; what "
                      f"changes is which TRAITS the difficult-advice share teaches. All "
                      f"{st['rows_swapped']} rows belonging to {dropped} are removed and "
                      "backfilled with an equal number of unused corpus rows drawn evenly "
                      "from the six remaining traits (40 each).",
        "date_generated": "2026-08-18",
        "constitution": "claude_distilled_07_principles_approved "
                        "(constitutions/claude_distilled_07_principles_approved/constitution.md)",
        "source_repo": f"Matthew-Bozoukov/teaching_claude_why_replication @ {git_sha()}",
        "models": "none — no text was generated or rewritten; this is a selection-only arm",
        "generation_config": f"deterministic given seed {st['seed']}; replacements sampled "
                             "from the unused pool after sorting by scenario_id",
        "schema": "text (rendered Qwen3.6 chat); source; supervise",
        "provenance": "scratch/build_trait_swap_mixture.py (verified by "
                      "scratch/verify_trait_swap.py)",
        "control_dataset": f"{CONTROL} (mixture_think.jsonl) — same 10,000 rows with the "
                           "original trait-balanced selection",
        "rows": f"10,000 = 9,284 standard SFT rows byte-identical to the control + 716 "
                f"difficult-advice rows ({716 - st['rows_swapped']} kept, "
                f"{st['rows_swapped']} swapped)",
        "manipulation": f"trait distribution {st['trait_before']} -> {st['trait_after']}; "
                        "traits 1, 3 and 4 are absent entirely",
        "verification": f"exactly {st['rows_swapped']} rows differ from the control; all 9,284 "
                        "non-difficult-advice rows byte-identical; every swapped row carries "
                        "exactly one non-empty <think> block, the family prefill and correct "
                        "turn markers; swapped-row token length median 1,595 vs 1,621 for kept "
                        "rows, none over 8,192",
        "why_this_arm": "The low-ODCV arm (2026-08-18-low-odcv-similarity-t2-9284-synthdoc-716) "
                        "raised ODCV MR to 17.1% from ~9% for three reasoning-style ablations, "
                        "but its swap removed t1 almost entirely as a side effect, so trait "
                        "removal and eval-content removal were the same manipulation. This arm "
                        "removes traits directly while keeping the mix far closer to the "
                        "control in ODCV-likeness, so the two explanations can be told apart.",
        "caveat": "Residual ODCV-likeness drift, in the same direction but much smaller. Mean "
                  "ODCV-dimension scores over the difficult-advice share (0-3): control "
                  "operator 0.80 / agentic 0.64 / metric 1.17 / oversight 1.48; THIS ARM 0.71 / "
                  "0.40 / 0.94 / 1.22; low-ODCV arm 0.14 / 0.08 / 0.45 / 0.50. Traits 1/3/4 are "
                  "more eval-like than average, so dropping them lowers ODCV-likeness somewhat "
                  "even though selection never used the ODCV ratings. The contrast against the "
                  "low-ODCV arm is therefore a matter of degree, not a clean double "
                  "dissociation. Ratings cover 638 of the 716 rows (all 716 control rows, 162 "
                  "of 240 swapped-in rows).",
    }
    url = push_files([RUN / "mixture_think_traitswap.jsonl",
                      RUN / "mixture_stats_traitswap.json"], REPO, fields, private=private)
    card = (card_front_matter([{"config_name": "default",
                                "data_files": "mixture_think_traitswap.jsonl",
                                "default": True}]) + card_markdown(fields))
    hf_api().upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                         repo_id=REPO, repo_type="dataset")
    print(url)


if __name__ == "__main__":
    fire.Fire(main)
