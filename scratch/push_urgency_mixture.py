# ABOUTME: Publish an excised training mixture to Hugging Face as a SEPARATE dataset from the
# ABOUTME: control it was derived from. Run: uv run python scratch/push_excised_mixture.py

"""Push the C6/C13/C14-excised mixture.

Deliberately a NEW repo rather than an edit of the control. The control
(LASR-Callum/2026-08-06-table2-9284-synthdoc-716-train) is pinned by revision inside the
training_meta.json of an already-trained adapter, and it is the comparison arm for this very
experiment; rewriting it in place would invalidate both. An ablation and its control have to
be two objects.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.huggingface import card_front_matter, card_markdown, hf_api, push_files  # noqa: E402
from src.utils import git_sha  # noqa: E402

REPO = "matboz/2026-08-18-urgency-amplified-t2-9284-synthdoc-716"
RUN = Path("output/urgency/amplify_full")
CONTROL = "LASR-Callum/2026-08-06-table2-9284-synthdoc-716-train"


def main(private: bool = True) -> None:
    """Upload the excised mixture, its stats and the per-row rewrite record.

    Args:
        private: Create the repo private (default).
    """
    st = json.loads((RUN / "mixture_stats_excised.json").read_text())
    meta = json.loads((RUN / "run_meta.json").read_text())
    fields = {
        "title": "Urgency-amplified mixture (Table2 9,284 + difficult-advice 716)",
        "experiment": "Ablation arm, in the ADDITIVE direction: the difficult-advice reasoning "
                      "traces are rewritten so the pull to decide quickly is what the reasoning "
                      "is wrestling with, rather than a pressure it notes once and moves past. "
                      "Every deadline, cost and hour named must already appear in the scenario — "
                      "no urgency is invented. The scenario, the user's message, the assistant's "
                      "reply and the DECISION are unchanged.",
        "date_generated": "2026-08-17",
        "constitution": "claude_distilled_07_principles_approved "
                        "(constitutions/claude_distilled_07_principles_approved/constitution.md)",
        "source_repo": f"Matthew-Bozoukov/teaching_claude_why_replication @ {git_sha()}",
        "models": f"passage selection + replacement {meta['model']}; substitute audit "
                  "anthropic/claude-sonnet-5; clusters from Qwen/Qwen3-Embedding-8B",
        "generation_config": "temperature 0.0, max_tokens 8000; passages returned VERBATIM and "
                             "located by exact string search, substitution applied "
                             "mechanically in Python so no other character can change; every "
                             "substitute judged and, if it reintroduced the property, rewritten "
                             "once with the auditor's reason attached",
        "schema": "text (rendered Qwen3.6 chat; identical to the control except inside <think> "
                  "on rewritten rows); source; scenario_id and excised present only on the 346 "
                  "rewritten rows",
        "provenance": "scratch/measure_urgency.py to rate all 716, then "
                      "scratch/amplify_urgency.py --min_score 3 over every trace below 3, then "
                      "scratch/build_excised_mixture.py; verified by scratch/rejudge_urgency.py",
        "control_dataset": f"{CONTROL} (mixture_think.jsonl) — same 10,000 rows, unrewritten. "
                           "THIS IS THE COMPARISON ARM; it is intentionally untouched.",
        "rows": "10,000 rows; 711 of 716 difficult-advice rows rewritten (5 were already at "
                "the top score and left alone); the 9,284 Table2 rows are byte-identical to "
                "the control",
        "verification": "words 375,449 -> 405,603 (108.0%, median row 107%); text outside <think> "
                        "asserted byte-identical on all 10,000 rows; exactly 711 rows differ",
        "ablation_strength": "Blind judge on 200 sampled rows, rating each version without "
                             "knowing which is which. MODEL-FELT URGENCY 1.26 -> 2.50 on 0-3 "
                             "(163 of 200 higher, 0 lower). The distribution moves from "
                             "40/68/92/0 across scores 0-3 to 0/0/99/101: no trace scores "
                             "below 2 any more and half reach 3, a score only 2 of the "
                             "original 716 held. Conversion barely depends on starting score "
                             "(48% / 45% / 56% for rows starting at 0 / 1 / 2). "
                             "DECISION UNCHANGED on 200 of 200 rows, checked separately — an "
                             "arm where the model feels the pressure more AND caves would "
                             "confound two variables.",
        "caveat": "Traces are ~8% LONGER than the control (median row 107%, max 170%), the "
                  "opposite direction from the one-sided-advocacy arm at 95% — amplifying "
                  "urgency adds returns to the clock rather than trimming. If both arms are "
                  "trained they bracket the control on length, which is interpretable but "
                  "should be measured rather than assumed harmless. Note also that user-imposed "
                  "urgency is near-universal in this corpus already (89% of scenarios score >=2 "
                  "on time pressure), so this arm changes how much the REASONING leans on a "
                  "clock that was always there.",
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
