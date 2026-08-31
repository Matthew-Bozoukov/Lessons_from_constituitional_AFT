# ABOUTME: Publish the low-stakes mixture: the usual 9,284 standard rows plus 712
# ABOUTME: difficult-advice rows rewritten so almost nothing is at stake.

"""Push the stakes ablation.

Earlier arms varied trait identity, trait volume, scenario selection, reasoning style and
reasoning structure, and every one landed between 8.7% and 17.6% ODCV MR. This varies the one
property the corpus is most consistent about: 96.8% of its rows are serious-or-worse.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.huggingface import card_front_matter, card_markdown, hf_api, push_files  # noqa: E402
from src.utils import git_sha  # noqa: E402

REPO = "matboz/2026-08-20-low-stakes-9284-plus-712"
RUN = Path("output/low_stakes_mixture/20260820_141535")
CONTROL = "LASR-Callum/2026-08-06-table2-9284-synthdoc-716-train"


def main(private: bool = True) -> None:
    """Upload the mixture and its build record.

    Args:
        private: Create the repo private (default).
    """
    st = json.loads((RUN / "mixture_stats_low_stakes.json").read_text())
    fields = {
        "title": "Low-stakes mixture (9,284 standard + 712 rewritten difficult advice)",
        "experiment": "Stakes ablation. Every difficult-advice row was rewritten so that "
                      "almost nothing is at stake: the situation moves to an ordinary part of "
                      "life AND the transgression is scaled down to match, while the trait "
                      "taught, the deliberative structure of the reasoning, and the length and "
                      "register are preserved. A legal custody dispute becomes a book club "
                      "selection; federal grant compliance becomes hobby club rules; an "
                      "eviction becomes a roommate chore chart.",
        "date_generated": "2026-08-20",
        "constitution": "claude_distilled_07_principles_approved "
                        "(constitutions/claude_distilled_07_principles_approved/constitution.md)",
        "source_repo": f"Matthew-Bozoukov/teaching_claude_why_replication @ {git_sha()}",
        "models": "anthropic/claude-sonnet-5 rewrote every row; "
                  "google/gemini-3-flash-preview rated stakes and scenario-sameness blind",
        "generation_config": "rewrite temperature 1.0, max_tokens 4000; judging temperature "
                             "0.0; selection deterministic given seed 0",
        "schema": "text (rendered Qwen3.6 chat); source",
        "provenance": "scratch/rate_prompt_stakes.py to measure, "
                      "scratch/rewrite_low_stakes_aggressive.py --preserve_scenario false to "
                      "rewrite, scratch/verify_low_stakes.py to verify, "
                      "scratch/build_low_stakes_mixture.py to assemble",
        "control_dataset": f"{CONTROL} (mixture_think.jsonl) — the 9,284 standard rows are "
                           "byte-identical; every difficult-advice row is replaced",
        "rows": f"{st['rows_total']:,} = {st['standard_rows']:,} standard SFT rows + "
                f"{st['difficult_advice_rows']} rewritten difficult-advice rows",
        "manipulation": f"mean stakes (0-3, judged on the user turn) "
                        f"{st['stakes_mean_before']} -> {st['stakes_mean_after']}; "
                        f"distribution after {st['stakes_after_distribution']}; "
                        f"{st['at_or_below_modest']}/712 ({st['at_or_below_modest']/712:.1%}) "
                        "now trivial or modest, against 3.2% in the original corpus",
        "verification": "blind re-rating of every rewritten user turn by a judge that never "
                        "saw the original; a seven-device lexical scan (countdown, scale of "
                        "harm, vivid imminence, scarcity, existential consequence, sunk cost, "
                        "reasoning amplification) across user turn AND reasoning; 9,284 "
                        "standard rows byte-identical; all 10,358 assistant turns carry a "
                        "think block; uniform row schema",
        "caveat": "THREE THINGS TO KNOW. (1) The scenarios MOVED — sameness against the "
                  "originals is 1.25/3 by design, so this varies domain and magnitude "
                  "together and is not a pure stakes manipulation. (2) 89 of 712 rows still "
                  "rate >=2, and 20 remain grave, so the arm is low-stakes on average rather "
                  "than uniformly. (3) 151 of 712 still contain a stake-raising device, "
                  "countdowns overwhelmingly (102) — deadlines survive relocation, so this is "
                  "NOT a deadline-free arm.",
        "dropped": f"{st['dropped_unrewritten']} of 716 rows failed to rewrite and were "
                   "dropped rather than left at original severity, so the mixture is "
                   f"{st['rows_total']:,} rows and the difficult-advice share is 7.12% "
                   "against the control's 7.16%",
    }
    url = push_files([RUN / "mixture_think_low_stakes.jsonl",
                      RUN / "mixture_stats_low_stakes.json"], REPO, fields, private=private)
    card = (card_front_matter([{"config_name": "default",
                                "data_files": "mixture_think_low_stakes.jsonl",
                                "default": True}]) + card_markdown(fields))
    hf_api().upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                         repo_id=REPO, repo_type="dataset")
    print(url)


if __name__ == "__main__":
    fire.Fire(main)
