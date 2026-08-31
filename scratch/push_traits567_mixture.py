# ABOUTME: Publish the traits-5/6/7-only mixture: full 716-row difficult-advice budget drawn
# ABOUTME: from three traits that are not oversight, honesty or harm.

"""Push the trait-restricted arm.

The deletion arms took traits away. This one keeps the full difficult-advice volume but sources
every row from three traits only, none of which names the benchmark's failure mode.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.huggingface import card_front_matter, card_markdown, hf_api, push_files  # noqa: E402
from src.utils import git_sha  # noqa: E402

REPO = "matboz/2026-08-19-traits567-only-9284-plus-716"
RUN = Path("output/traits567/20260819_152548")
CONTROL = "LASR-Callum/2026-08-06-table2-9284-synthdoc-716-train"
TRAITS = {
    "t5": "Act from cultivated character, not rigid rules",
    "t6": "Maintain a secure, stable identity and genuinely endorsed values as a novel entity",
    "t7": "Honour operator adjustments to sensible defaults, but never as a tool against users",
}


def main(private: bool = True) -> None:
    """Upload the mixture and its build record.

    Args:
        private: Create the repo private (default).
    """
    st = json.loads((RUN / "mixture_stats_traits567.json").read_text())
    kept = "; ".join(f"{t} ({TRAITS[t]})" for t in st["kept_traits"])
    fields = {
        "title": "Traits-5/6/7-only mixture (9,284 standard + 716 difficult advice)",
        "experiment": "Trait-restricted arm. The difficult-advice budget stays at the full 716 "
                      f"rows, but every row is drawn from three traits only: {kept}. None of "
                      "these is the oversight, honesty or harm trait. Rows come from the whole "
                      "2,203-row stage-7 corpus (both the rows in the control mixture and the "
                      "unused pool), split evenly across the three.",
        "date_generated": "2026-08-19",
        "constitution": "claude_distilled_07_principles_approved "
                        "(constitutions/claude_distilled_07_principles_approved/constitution.md)",
        "source_repo": f"Matthew-Bozoukov/teaching_claude_why_replication @ {git_sha()}",
        "models": "none — no text was generated or rewritten; this is a selection-only arm",
        "generation_config": f"deterministic given seed {st['seed']}; rows sampled after "
                             "sorting each trait's pool by scenario_id",
        "schema": "text (rendered Qwen3.6 chat); source; supervise",
        "provenance": "scratch/build_traits567_mixture.py",
        "control_dataset": f"{CONTROL} (mixture_think.jsonl) — the 9,284 standard rows are "
                           "byte-identical; the difficult-advice rows are replaced wholesale",
        "rows": f"{st['rows_total']:,} = 9,284 standard SFT rows + "
                f"{st['difficult_advice_rows']} difficult-advice rows "
                f"({st['trait_after']})",
        "manipulation": "trait distribution goes from the control's balanced 79-80 per trait "
                        f"across all nine to {st['trait_after']} — six traits absent entirely, "
                        "three at roughly triple their usual share, total volume unchanged",
        "why_this_arm": "Deleting traits (2026-08-19-traits13-removed, "
                        "2026-08-18-traits134-removed) left the ODCV effect intact. This asks "
                        "the complementary question: can three traits that do NOT describe "
                        "oversight, honesty or harm carry the effect on their own? The "
                        "no-difficult-advice control sits at 44.1% MR and the unmodified recipe "
                        "at 16.8%, so the comparison is against those two.",
        "verification": f"{st['difficult_advice_rows']} rows, exactly {st['quota']} per trait "
                        "and no foreign trait present; 9,284 standard rows byte-identical to "
                        "the control; no difficult-advice row carries an empty think marker; "
                        f"all {st['assistant_turns_checked']:,} assistant turns carry a think "
                        "block (what the training mask gate checks)",
        "caveat": "The three traits are over-represented relative to any real deployment mix "
                  "(~239 rows each against the control's 79-80), so this arm changes trait "
                  "CONCENTRATION as well as trait identity. It cannot separate 'these three "
                  "traits suffice' from 'more rows per trait helps'.",
    }
    url = push_files([RUN / "mixture_think_traits567.jsonl",
                      RUN / "mixture_stats_traits567.json"], REPO, fields, private=private)
    card = (card_front_matter([{"config_name": "default",
                                "data_files": "mixture_think_traits567.jsonl",
                                "default": True}]) + card_markdown(fields))
    hf_api().upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                         repo_id=REPO, repo_type="dataset")
    print(url)


if __name__ == "__main__":
    fire.Fire(main)
