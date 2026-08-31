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

REPO = "matboz/2026-08-17-c137c42-swapped-t2-9284-synthdoc-716"
RUN = Path("output/excision/tradeoff_swap_full")
CONTROL = "LASR-Callum/2026-08-06-table2-9284-synthdoc-716-train"


def main(private: bool = True) -> None:
    """Upload the excised mixture, its stats and the per-row rewrite record.

    Args:
        private: Create the repo private (default).
    """
    st = json.loads((RUN / "mixture_stats_excised.json").read_text())
    meta = json.loads((RUN / "run_meta.json").read_text())
    fields = {
        "title": "C137/C42-swapped mixture (Table2 9,284 + difficult-advice 716)",
        "experiment": "Ablation arm: passages that WEIGH considerations against each other "
                      "(cluster C137) or analyse POWER ASYMMETRIES between the parties (C42) "
                      "are removed from the difficult-advice traces and replaced with "
                      "substitutes that reach the same conclusion by a DIFFERENT reasoning "
                      "move, dealt round-robin across six: categorical line-drawing (C26), "
                      "stakeholder identification (C139), steelman-then-rebut (C23), "
                      "purpose-of-the-rule (C141), hypothetical test (C122) and surface-request"
                      "-vs-underlying-need (C45). Round-robin so no single replacement gains "
                      "more than ~8 points of prevalence and the arm is not an 'add X' "
                      "experiment. Everything outside <think> is byte-identical to the control.",
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
        "provenance": "scratch/excise_property.py --arm tradeoff_swap --model "
                      "anthropic/claude-sonnet-5, then scratch/build_excised_mixture.py; "
                      "row selection from scratch/mixture_cluster_membership.py "
                      "--clusters_arg 137,42",
        "control_dataset": f"{CONTROL} (mixture_think.jsonl) — same 10,000 rows, unrewritten. "
                           "THIS IS THE COMPARISON ARM; it is intentionally untouched.",
        "rows": f"{st['rows_total']} rows, {st['rows_rewritten']} rewritten "
                f"({st['passages_replaced']} passages replaced). 357 rows carried C137/C42; "
                "2 were abandoned as unquotable and 24 had no passage the judge would cut, so "
                "26 rows are present unchanged.",
        "verification": f"words {st['words_before']:,} -> {st['words_after']:,} "
                        f"({st['words_after'] / st['words_before']:.1%}); text outside <think> "
                        "asserted byte-identical on all 10,000 rows; 957 passages replaced "
                        "across 331 rewritten rows",
        "ablation_strength": "PARTIAL, and weaker than the C6 arm. Lexical weighing probes fall "
                             "67.4% -> 47.1% of rewritten traces (explicit weighing verbs "
                             "47.7 -> 31.7, asymmetry/proportionality 12.7 -> 1.5, tradeoff "
                             "nouns 22.4 -> 16.0). Half the rewritten traces still weigh "
                             "somewhere: the deleter cut 2.9 passages per row here against 5.9 "
                             "in the C6 arm, because weighing is diffuse in this corpus rather "
                             "than localised. Power probes barely move (2.4 -> 2.1) because "
                             "C42 is a semantic cluster with little lexical signature.",
        "caveat": "Traces are ~9% LONGER than the control (median 108%), the opposite "
                  "direction from the C6-excised arm — substituting a full reasoning move "
                  "costs more words than the weighing it replaced. 7.8% of substitutes were "
                  "judged to reintroduce weighing or power analysis even after one inline "
                  "repair; by property the worst is steelman-then-rebut at 14.0% (it is "
                  "structurally a comparison) and the cleanest is stakeholder identification "
                  "at 3.8%.",
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
