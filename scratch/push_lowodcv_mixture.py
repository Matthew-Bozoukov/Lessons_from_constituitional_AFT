# ABOUTME: Publish the low-ODCV-similarity mixture: the 9,284 standard SFT rows plus a
# ABOUTME: difficult-advice share whose most eval-like scenarios have been swapped out.

"""Push the scenario-selection arm.

Every other arm today rewrote the REASONING and left the scenario set alone. This one leaves
the reasoning untouched and changes WHICH scenarios are trained on: the 331 difficult-advice
rows scoring >=2 on agentic capability or oversight-at-stake — the ones that read like ODCV
cells written as a conversation — are replaced by rows from the unused corpus that score low on
every ODCV dimension.

The card records the trait confound prominently, because it is the reason this arm cannot be
read as a clean test of content overlap on its own.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.huggingface import card_front_matter, card_markdown, hf_api, push_files  # noqa: E402
from src.utils import git_sha  # noqa: E402

REPO = "matboz/2026-08-18-low-odcv-similarity-t2-9284-synthdoc-716"
RUN = Path("output/low_odcv/or_20260818")
CONTROL = "LASR-Callum/2026-08-06-table2-9284-synthdoc-716-train"


def main(private: bool = True) -> None:
    """Upload the low-ODCV mixture and its swap record.

    Args:
        private: Create the repo private (default).
    """
    st = json.loads((RUN / "mixture_stats_lowodcv.json").read_text())
    b, a = st["odcv_dims_before"], st["odcv_dims_after"]
    fields = {
        "title": "Low-ODCV-similarity mixture (Table2 9,284 + difficult-advice 716)",
        "experiment": "Scenario-selection arm. The reasoning text is UNMODIFIED throughout; "
                      "what changes is which difficult-advice scenarios are trained on. The "
                      f"{st['replaced']} rows scoring >=2 on agentic capability OR "
                      "oversight-at-stake — the ones that resemble ODCV-Bench cells written as "
                      "a conversation — are replaced with rows from the 1,487 unused corpus "
                      "rows scoring <=1 on all four ODCV dimensions.",
        "date_generated": "2026-08-18",
        "constitution": "claude_distilled_07_principles_approved "
                        "(constitutions/claude_distilled_07_principles_approved/constitution.md)",
        "source_repo": f"Matthew-Bozoukov/teaching_claude_why_replication @ {git_sha()}",
        "models": "scenario rating by anthropic/claude-sonnet-5; no reasoning was rewritten",
        "generation_config": "rating temperature 0.0, max_tokens 350; selection is deterministic "
                             f"given the ratings and seed {st['seed']}",
        "schema": "text (rendered Qwen3.6 chat); source; scenario_id and swapped_in present "
                  "only on the swapped rows",
        "provenance": "scratch/rate_odcv_similarity.py over the 716 and over 1,000 unused "
                      "corpus rows, then scratch/build_low_odcv_mixture.py --mode or",
        "control_dataset": f"{CONTROL} (mixture_think.jsonl) — same 10,000 rows with the "
                           "original scenario selection",
        "rows": f"10,000 = 9,284 standard SFT rows byte-identical to the control + 716 "
                f"difficult-advice rows ({716 - st['replaced']} kept, {st['replaced']} swapped)",
        "verification": "exactly 331 rows differ from the control; all 9,284 non-difficult-"
                        "advice rows byte-identical; all 331 swapped rows carry exactly one "
                        "well-formed <think> block and the correct turn markers",
        "manipulation": "mean ODCV-dimension scores across the difficult-advice share: "
                        f"operator {b['operator']} -> {a['operator']}, agentic {b['agentic']} -> "
                        f"{a['agentic']}, metric {b['metric']} -> {a['metric']}, oversight "
                        f"{b['oversight']} -> {a['oversight']} (each 0-3)",
        "caveat": "TRAIT COMPOSITION IS BADLY CONFOUNDED and this is the first thing to know "
                  "about this arm. The control is trait-balanced at 79-80 rows per trait; after "
                  "the swap the distribution runs from 2 (t1) to 143 (t9). This cannot be fixed "
                  "by matching traits on the swap: only 2 of 102 rated unused t1 rows are "
                  "low-ODCV, because t1 IS the agentic-oversight trait — removing scenarios "
                  "where a deployed model could bypass a control is nearly the same operation "
                  "as removing t1. Any downstream difference is therefore attributable to trait "
                  "composition as much as to ODCV content overlap.",
    }
    url = push_files([RUN / "mixture_think_lowodcv.jsonl",
                      RUN / "mixture_stats_lowodcv.json"], REPO, fields, private=private)
    card = (card_front_matter([{"config_name": "default",
                                "data_files": "mixture_think_lowodcv.jsonl",
                                "default": True}]) + card_markdown(fields))
    hf_api().upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                         repo_id=REPO, repo_type="dataset")
    print(url)


if __name__ == "__main__":
    fire.Fire(main)
