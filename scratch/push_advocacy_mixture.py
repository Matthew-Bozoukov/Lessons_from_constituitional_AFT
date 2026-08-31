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

REPO = "matboz/2026-08-18-onesided-advocacy-t2-9284-synthdoc-716"
RUN = Path("output/whole_trace/advocacy_full")
CONTROL = "LASR-Callum/2026-08-06-table2-9284-synthdoc-716-train"


def main(private: bool = True) -> None:
    """Upload the excised mixture, its stats and the per-row rewrite record.

    Args:
        private: Create the repo private (default).
    """
    st = json.loads((RUN / "mixture_stats_excised.json").read_text())
    meta = json.loads((RUN / "run_meta.json").read_text())
    fields = {
        "title": "One-sided advocacy mixture (Table2 9,284 + difficult-advice 716)",
        "experiment": "Ablation arm: all 716 difficult-advice reasoning traces are rewritten as "
                      "ONE-SIDED ADVOCACY. The trace argues for the decision it reaches instead "
                      "of adjudicating between two options: it never says anything good about "
                      "the option not taken, and never concedes a cost of the option taken. The "
                      "scenario, the user's message and the assistant's reply are unchanged.",
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
        "provenance": "scratch/rewrite_whole_trace.py --advocacy --membership "
                      "output/mixture_cluster_membership/all716_membership.jsonl, then "
                      "scratch/build_excised_mixture.py; verified with "
                      "scratch/judge_weighing_semantic.py --property concession",
        "control_dataset": f"{CONTROL} (mixture_think.jsonl) — same 10,000 rows, unrewritten. "
                           "THIS IS THE COMPARISON ARM; it is intentionally untouched.",
        "rows": "10,000 rows; ALL 716 difficult-advice rows rewritten (0 failures); the "
                "9,284 Table2 rows are byte-identical to the control",
        "verification": "words 378,043 -> 360,330 (95.3%, median row 95%); text outside <think> "
                        "asserted byte-identical on all 10,000 rows; exactly 716 rows differ",
        "ablation_strength": "Blind judge on 150 sampled rows, rating each version without "
                             "knowing which is which. CREDITS THE REJECTED OPTION 1.67 -> 0.73 "
                             "on a 0-3 scale (114 of 150 rows lower, 1 higher; 53 rows now "
                             "score 0, none did before; no row scores 3). CONCEDES THE CHOSEN "
                             "OPTION 1.31 -> 1.19, barely moved — the second half of the "
                             "instruction did not take.",
        "caveat": "This arm does NOT remove weighing. Nine designs were tried and a blind "
                  "judge measured weighing at 2.9-3.0/3 after every one; a 200-row random "
                  "sample of the corpus scores 100% at >=2, so weighing is constitutive of "
                  "difficult-advice reasoning rather than a variable property. What this arm "
                  "removes is EVEN-HANDEDNESS — the traces still compare options, they just no "
                  "longer credit the rejected one. Also: first-person density falls 3.07 -> "
                  "1.44 per 100 words, so the arms differ stylistically as well; treat 'it "
                  "stopped saying I' as a live alternative explanation.",
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
