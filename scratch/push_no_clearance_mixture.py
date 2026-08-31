# ABOUTME: Publish the clearance-free mixture: 716 difficult-advice rows with none of the
# ABOUTME: audit-a-hazard-then-clear-it reasoning move, plus the usual 9,284 standard rows.

"""Push the reasoning-move ablation.

Every earlier ablation targeted a trait, a scenario set, or a rewritten style. This one targets
a specific reasoning STRUCTURE identified by an autorater across the whole corpus, and removes
it by selection alone — no text is rewritten.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.huggingface import card_front_matter, card_markdown, hf_api, push_files  # noqa: E402
from src.utils import git_sha  # noqa: E402

REPO = "matboz/2026-08-19-no-conflict-clearance-9284-plus-716"
RUN = Path("output/no_clearance/20260819_170953")
CONTROL = "LASR-Callum/2026-08-06-table2-9284-synthdoc-716-train"


def main(private: bool = True) -> None:
    """Upload the mixture and its build record.

    Args:
        private: Create the repo private (default).
    """
    st = json.loads((RUN / "mixture_stats_no_clearance.json").read_text())
    fields = {
        "title": "No-conflict-clearance mixture (9,284 standard + 716 difficult advice)",
        "experiment": "Reasoning-structure ablation by SELECTION, not rewriting. An autorater "
                      "(scratch/rate_conflict_clearance.py, judge google/gemini-3-flash-preview) "
                      "scored all 2,203 difficult-advice traces for a two-part move: the trace "
                      "SCREENS its own task for ethical hazard — naming the metric, incentive or "
                      "instruction that could corrupt it — and then CLEARS that screen, "
                      "concluding the pressure is benign and ordinary execution is right. A "
                      "trace that screens and finds a REAL conflict scores 0. 145 of the 716 "
                      "rows in the control mixture carried the move; every one is replaced with "
                      "a rated-clean row, leaving ZERO in this mixture.",
        "date_generated": "2026-08-19",
        "constitution": "claude_distilled_07_principles_approved "
                        "(constitutions/claude_distilled_07_principles_approved/constitution.md)",
        "source_repo": f"Matthew-Bozoukov/teaching_claude_why_replication @ {git_sha()}",
        "models": "google/gemini-3-flash-preview rated the traces; no text was generated or "
                  "rewritten — every row is an unmodified corpus row",
        "generation_config": f"rating temperature 0.0, max_tokens 400, traces truncated at "
                             f"12,000 chars; selection deterministic given seed {st['seed']}",
        "schema": "text (rendered Qwen3.6 chat); source",
        "provenance": "scratch/rate_conflict_clearance.py then scratch/build_no_clearance_mixture.py",
        "control_dataset": f"{CONTROL} (mixture_think.jsonl) — the 9,284 standard rows are "
                           f"byte-identical; {st['replaced']} of the 716 difficult-advice rows "
                           "are swapped",
        "rows": f"{st['rows_total']:,} = 9,284 standard SFT rows + "
                f"{st['difficult_advice_rows']} difficult-advice rows",
        "manipulation": f"clearance-form rows go from {st['replaced']}/716 (20.3%) to 0/716, "
                        "verified against the same ratings the selection used",
        "caveat": "TRAIT 8 IS HALVED, and this is not separable from the manipulation. Trait 8 "
                  "('be genuinely, substantively helpful — never treat unhelpfulness as safe') "
                  "is 84.5% clearance corpus-wide, so only 27 clean t8 rows exist outside the "
                  "mixture against 68 needing replacement. The 41-row shortfall is backfilled "
                  f"round-robin from other traits, taking t8 from 79 to 38 rows while the others "
                  "rise by 5-6 each. Trait 8 is largely TAUGHT through this move, so an arm "
                  "without the move necessarily has less of that trait. Sibling arms give some "
                  "purchase: traits-1/3/4-removed (10.6% MR) and traits-1/3-deleted (12.9%) both "
                  "left t8 untouched, so a move here that those did not produce points at the "
                  "reasoning structure rather than at t8 volume.",
        "trait_distribution": f"before {st['trait_before']} -> after {st['trait_after']}",
        "verification": f"0 clearance rows remain of {st['difficult_advice_rows']} (all rated); "
                        "9,284 standard rows byte-identical to the control; all "
                        f"{st['assistant_turns_checked']:,} assistant turns carry a think block; "
                        "uniform row schema across all 10,000 rows",
    }
    url = push_files([RUN / "mixture_think_no_clearance.jsonl",
                      RUN / "mixture_stats_no_clearance.json"], REPO, fields, private=private)
    card = (card_front_matter([{"config_name": "default",
                                "data_files": "mixture_think_no_clearance.jsonl",
                                "default": True}]) + card_markdown(fields))
    hf_api().upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                         repo_id=REPO, repo_type="dataset")
    print(url)


if __name__ == "__main__":
    fire.Fire(main)
