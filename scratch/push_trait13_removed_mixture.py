# ABOUTME: Publish the traits-1/3-removed mixture: the same 9,284 standard rows plus the 556
# ABOUTME: difficult-advice rows left after deleting the oversight and honesty traits.

"""Push the traits-1/3 deletion arm.

Unlike the traits-1/3/4 arm this one does NOT backfill, so the mixture is smaller and the
difficult-advice share is lower. The card leads with that, because it is the difference that
stops this being a pure trait manipulation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.huggingface import card_front_matter, card_markdown, hf_api, push_files  # noqa: E402
from src.utils import git_sha  # noqa: E402

REPO = "matboz/2026-08-19-traits13-removed-t2-9284-synthdoc-556"
RUN = Path("output/trait13_removed/20260819_111903")
CONTROL = "LASR-Callum/2026-08-06-table2-9284-synthdoc-716-train"

TRAITS = {"t1": "Preserve human oversight; avoid unilateral, power-accruing action",
          "t3": "Be scrupulously honest and non-deceptive, in word, framing, and action"}


def main(private: bool = True) -> None:
    """Upload the mixture and its build record.

    Args:
        private: Create the repo private (default).
    """
    st = json.loads((RUN / "mixture_stats_trait13_removed.json").read_text())
    dropped = "; ".join(f"{t} ({TRAITS[t]})" for t in st["dropped_traits"])
    fields = {
        "title": "Traits-1/3-removed mixture (9,284 standard + 556 difficult advice)",
        "experiment": "Trait-deletion arm WITHOUT backfill. Every difficult-advice row "
                      f"belonging to {dropped} is deleted and nothing replaces it, so the "
                      f"difficult-advice share falls from 716 to {st['difficult_advice_rows']} "
                      "rows and the mixture from 10,000 to "
                      f"{st['rows_total']:,}. Reasoning text is unmodified throughout; the "
                      "surviving traits keep their original counts exactly.",
        "date_generated": "2026-08-19",
        "constitution": "claude_distilled_07_principles_approved "
                        "(constitutions/claude_distilled_07_principles_approved/constitution.md)",
        "source_repo": f"Matthew-Bozoukov/teaching_claude_why_replication @ {git_sha()}",
        "models": "none — no text was generated or rewritten; this is a deletion-only arm",
        "generation_config": "deterministic — no sampling; every row of the two traits is removed",
        "schema": "text (rendered Qwen3.6 chat); source; supervise",
        "provenance": "scratch/build_trait13_removed_mixture.py",
        "control_dataset": f"{CONTROL} (mixture_think.jsonl) — the 9,284 standard rows are "
                           "byte-identical, and the 556 kept difficult-advice rows are "
                           "unchanged",
        "rows": f"{st['rows_total']:,} = 9,284 standard SFT rows + "
                f"{st['difficult_advice_rows']} difficult-advice rows "
                f"({st['rows_removed']} removed)",
        "manipulation": f"trait distribution {st['trait_before']} -> {st['trait_after']}; "
                        "traits 1 and 3 are absent entirely and no trait was topped up",
        "caveat": "TWO THINGS CHANGE AT ONCE, so this is not a clean trait manipulation. "
                  "Because nothing backfills the deleted rows, the difficult-advice share "
                  f"drops from {st['difficult_advice_share_before']:.2%} to "
                  f"{st['difficult_advice_share_after']:.2%} of the mixture. A difference "
                  "against the control is therefore attributable to the missing traits OR to "
                  "there simply being 22% less difficult-advice data. The traits-1/3/4 arm "
                  "(matboz/2026-08-18-traits134-removed-t2-9284-synthdoc-716) holds the share "
                  "fixed by backfilling and is the arm to compare against if that matters.",
        "verification": f"{st['rows_removed']} rows removed and no others; the seven surviving "
                        "traits keep their exact original counts; 9,284 standard rows "
                        f"byte-identical to the control; all {st['assistant_turns_checked']:,} "
                        "assistant turns carry a think block",
    }
    url = push_files([RUN / "mixture_think_trait13_removed.jsonl",
                      RUN / "mixture_stats_trait13_removed.json"], REPO, fields, private=private)
    card = (card_front_matter([{"config_name": "default",
                                "data_files": "mixture_think_trait13_removed.jsonl",
                                "default": True}]) + card_markdown(fields))
    hf_api().upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                         repo_id=REPO, repo_type="dataset")
    print(url)


if __name__ == "__main__":
    fire.Fire(main)
