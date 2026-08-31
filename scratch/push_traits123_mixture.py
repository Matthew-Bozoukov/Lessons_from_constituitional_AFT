# ABOUTME: Publish the traits-1/2/3-only mixture: full 716-row difficult-advice budget drawn
# ABOUTME: from the three restraint traits (oversight, power balance, honesty).

"""Push the restraint-trait arm — the complement of the traits-5/6/7 arm.

Same construction, opposite half of the constitution: every difficult-advice row comes from
the traits about holding back rather than the traits about acting well.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.huggingface import card_front_matter, card_markdown, hf_api, push_files  # noqa: E402
from src.utils import git_sha  # noqa: E402

REPO = "matboz/2026-08-19-traits123-only-9284-plus-716"
RUN = Path("output/traits123/20260819_184452")
CONTROL = "LASR-Callum/2026-08-06-table2-9284-synthdoc-716-train"
TRAITS = {"t1": "Preserve human oversight; avoid unilateral, power-accruing action",
          "t2": "Protect legitimate balances of power, democratic structures, healthy epistemics",
          "t3": "Be scrupulously honest and non-deceptive, in word, framing, and action"}


def main(private: bool = True) -> None:
    """Upload the mixture and its build record.

    Args:
        private: Create the repo private (default).
    """
    st = json.loads((RUN / "mixture_stats_t1t2t3.json").read_text())
    kept = "; ".join(f"{t} ({TRAITS[t]})" for t in st["kept_traits"])
    fields = {
        "title": "Traits-1/2/3-only mixture (9,284 standard + 716 difficult advice)",
        "experiment": "Trait-restricted arm, restraint half. The difficult-advice budget stays "
                      f"at the full 716 rows, but every row is drawn from: {kept}. Split evenly "
                      f"({st['trait_after']}) out of the 734 such rows in the 2,203-row stage-7 "
                      "corpus. Direct complement of "
                      "matboz/2026-08-19-traits567-only-9284-plus-716, which draws its 716 from "
                      "the character/identity/operator traits instead.",
        "date_generated": "2026-08-19",
        "constitution": "claude_distilled_07_principles_approved "
                        "(constitutions/claude_distilled_07_principles_approved/constitution.md)",
        "source_repo": f"Matthew-Bozoukov/teaching_claude_why_replication @ {git_sha()}",
        "models": "none — no text was generated or rewritten; this is a selection-only arm",
        "generation_config": f"deterministic given seed {st['seed']}; rows sampled after "
                             "sorting each trait's pool by scenario_id",
        "schema": "text (rendered Qwen3.6 chat); source",
        "provenance": "scratch/build_trait_subset_mixture.py --traits t1,t2,t3",
        "control_dataset": f"{CONTROL} (mixture_think.jsonl) — the 9,284 standard rows are "
                           "byte-identical; the difficult-advice rows are replaced wholesale",
        "rows": f"{st['rows_total']:,} = 9,284 standard SFT rows + "
                f"{st['difficult_advice_rows']} difficult-advice rows ({st['trait_after']})",
        "manipulation": "trait distribution goes from the control's balanced 79-80 across all "
                        f"nine to {st['trait_after']} — six traits absent, three at roughly "
                        "triple their usual share, total volume unchanged",
        "why_this_arm": "These are the three traits that most directly describe ODCV-Bench's "
                        "failure mode: acting without oversight, accruing power, and deceiving. "
                        "Its complement (traits 5/6/7) drops all three. Against the "
                        "no-difficult-advice control at 44.1% MR and the unmodified recipe at "
                        "16.8%, the pair asks which half of the constitution carries the effect "
                        "— or whether either half suffices alone.",
        "verification": f"{st['difficult_advice_rows']} rows, exactly {st['quota']} per trait "
                        "and no foreign trait present; 9,284 standard rows byte-identical to "
                        "the control; no difficult-advice row carries an empty think marker; "
                        f"all {st['assistant_turns_checked']:,} assistant turns carry a think "
                        "block; uniform row schema across all 10,000 rows",
        "conflict_clearance": "1.5% of these 716 rows (11) carry the audit-a-hazard-then-clear-"
                              "it reasoning move, against 20.3% in the control and 26.3% in the "
                              "traits-5/6/7 complement. That is not a designed property — it "
                              "falls out of the trait choice, because the restraint traits "
                              "almost never clear a hazard (t1 is 0.0% corpus-wide). So this "
                              "arm is also, incidentally, the lowest-clearance mixture built so "
                              "far, and that is confounded with the trait restriction.",
        "caveat": "Trait CONCENTRATION changes as well as identity: each trait appears ~239 "
                  "times against the control's 79-80, so the arm cannot separate 'these three "
                  "traits suffice' from 'more rows per trait helps'.",
    }
    url = push_files([RUN / "mixture_think_t1t2t3.jsonl",
                      RUN / "mixture_stats_t1t2t3.json"], REPO, fields, private=private)
    card = (card_front_matter([{"config_name": "default",
                                "data_files": "mixture_think_t1t2t3.jsonl",
                                "default": True}]) + card_markdown(fields))
    hf_api().upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                         repo_id=REPO, repo_type="dataset")
    print(url)


if __name__ == "__main__":
    fire.Fire(main)
