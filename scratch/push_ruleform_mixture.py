# ABOUTME: Publish the rule-form (three-move-inverted) mixture to Hugging Face as its own
# ABOUTME: dataset, separate from the unmodified control it was derived from.
"""Push the three-move-inversion mixture.

A new repo, not an edit of the control: the control
(LASR-Callum/2026-08-06-table2-9284-synthdoc-716-train) is pinned inside an already-trained
adapter's training_meta.json and is the comparison arm for this experiment.

Run: uv run python scratch/push_ruleform_mixture.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.huggingface import card_front_matter, card_markdown, hf_api, push_files  # noqa: E402
from src.utils import git_sha  # noqa: E402

REPO = "matboz/2026-08-21-ruleform-t2-9284-synthdoc-716"
RUN = Path("output/ruleform_716")
CONTROL = "LASR-Callum/2026-08-06-table2-9284-synthdoc-716-train"


def main(private: bool = True) -> None:
    """Upload the mixture, its stats and the per-row rewrite record.

    Args:
        private: Create the repo private (default).
    """
    st = json.loads((RUN / "mixture_stats_excised.json").read_text())
    fields = {
        "title": "Rule-form mixture (Table2 9,284 + difficult-advice 716, three moves inverted)",
        "experiment": "Ablation arm: all 716 difficult-advice reasoning traces are rewritten to "
                      "invert three deliberative moves — reach the verdict by naming a rule or "
                      "category instead of drawing a distinction, set the tempting option aside "
                      "instead of engaging it, and run a single line to one outcome instead of "
                      "enumerating branches. The scenario, the user's message and the "
                      "assistant's reply are unchanged on every row.",
        "date_generated": "2026-08-21",
        "constitution": "claude_distilled_12_principles_mid "
                        "(constitutions/claude_distilled_12_principles_mid/constitution.md)",
        "source_repo": f"Matthew-Bozoukov/teaching_claude_why_replication @ {git_sha()}",
        "models": "rewriter anthropic/claude-sonnet-5; blind rater "
                  "google/gemini-3-flash-preview",
        "generation_config": "rewriter temperature 0.7, max_tokens 4096; rater temperature 0.0 "
                             "with JSON response_format, items shuffled and shown one per call "
                             "with no version label",
        "schema": "text (rendered Qwen3.6 chat; identical to the control except inside <think> "
                  "on the 716 rewritten rows); source; scenario_id and excised present only on "
                  "the rewritten rows",
        "provenance": "scratch/invert_three_moves.py --n 0 (resumable), then "
                      "scratch/rate_all716.py, then scratch/build_ruleform_mixture.py",
        "control_dataset": f"{CONTROL} (mixture_think.jsonl) — same 10,000 rows, unrewritten. "
                           "THIS IS THE COMPARISON ARM; it is intentionally untouched.",
        "rows": "10,000 rows; all 716 difficult-advice rows rewritten (0 failures); the 9,284 "
                "Table2 rows are byte-identical to the control",
        "verification": f"text outside <think> asserted byte-identical on all 10,000 rows; "
                        f"each row's pre-rewrite reasoning asserted equal to the SFT copy; "
                        f"exactly 716 rows differ. Words "
                        f"{st['words_before']:,} -> {st['words_after']:,} "
                        f"({st['words_after']/st['words_before']:.1%}).",
        "ablation_strength": "Blind rater over both versions of all 716 rows. TARGETS: "
                             f"outcome_branches {st['move_outcome_branches']}, "
                             f"engages_counterargument {st['move_engages_counterargument']}, "
                             f"analytic_distinction {st['move_analytic_distinction']}. "
                             f"Overall form deliberative {st['form_deliberative']}.",
        "caveat": "THE THIRD TARGET DID NOT LAND. Branches and engagement invert cleanly, but "
                  "the analytic distinction survives in 529/716 (74%) — the rewrites reach a "
                  "rule-shaped verdict while still working out why the case falls where it "
                  "does. Only "
                  f"{st['all_three_targets_absent']}/716 rows lost all three targets. Treat "
                  "this arm as 'rule-framed with distinctions intact', NOT as a clean form "
                  "inversion. Two further confounds: length falls to "
                  f"{st['words_after']/st['words_before']:.0%} of the control, and "
                  f"returns_agency moved the WRONG way ({st['move_returns_agency']}) because "
                  "rule-based refusals hand the decision back more often. First-person density "
                  f"is unchanged ({st['first_person_per_100w_before']} -> "
                  f"{st['first_person_per_100w_after']} per 100 words), so unlike the advocacy "
                  "arm this one carries no register confound.",
    }
    url = push_files([RUN / "mixture_think_excised.jsonl",
                      RUN / "mixture_stats_excised.json",
                      RUN / "excisions.jsonl"], REPO, fields, private=private)
    card = (card_front_matter([{"config_name": "default",
                                "data_files": "mixture_think_excised.jsonl",
                                "default": True}]) + card_markdown(fields))
    hf_api().upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                         repo_id=REPO, repo_type="dataset")
    print(url)


if __name__ == "__main__":
    fire.Fire(main)
