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

REPO = "matboz/2026-08-17-c6c13c14-excised-t2-9284-synthdoc-716"
RUN = Path("output/excision/c6c13c14_sonnet_v2")
CONTROL = "LASR-Callum/2026-08-06-table2-9284-synthdoc-716-train"


def main(private: bool = True) -> None:
    """Upload the excised mixture, its stats and the per-row rewrite record.

    Args:
        private: Create the repo private (default).
    """
    st = json.loads((RUN / "mixture_stats_excised.json").read_text())
    meta = json.loads((RUN / "run_meta.json").read_text())
    fields = {
        "title": "C6/C13/C14-excised mixture (Table2 9,284 + difficult-advice 716)",
        "experiment": "Ablation arm: the reasoning passages expressing meta-reasoning about "
                      "response strategy (C6), metacognitive self-reflection (C13) and "
                      "deliberative pre-advice framing (C14) are REMOVED from the "
                      "difficult-advice traces and replaced with substitute sentences that do "
                      "the same structural work without the property. Everything outside the "
                      "<think> block is byte-identical to the control.",
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
        "provenance": "scratch/excise_property.py --arm meta_reasoning --model "
                      "anthropic/claude-sonnet-5, then scratch/build_excised_mixture.py; "
                      "row selection from scratch/mixture_cluster_membership.py "
                      "--clusters_arg 6,13,14",
        "control_dataset": f"{CONTROL} (mixture_think.jsonl) — same 10,000 rows, unrewritten. "
                           "THIS IS THE COMPARISON ARM; it is intentionally untouched.",
        "rows": f"{st['rows_total']} rows, {st['rows_rewritten']} rewritten "
                f"({st['passages_replaced']} passages replaced), 1 row abandoned as unquotable "
                "and left un-ablated",
        "verification": f"words {st['words_before']:,} -> {st['words_after']:,} "
                        f"({st['words_after'] / st['words_before']:.1%}); signature probe "
                        f"{st['signature_probe_before']:.1%} -> {st['signature_probe_after']:.1%}; "
                        f"move probe {st['move_probe_before']:.1%} -> "
                        f"{st['move_probe_after']:.1%}; text outside <think> asserted "
                        "byte-identical on all 10,000 rows",
        "caveat": "first-person density falls 3.23 -> 2.06 per 100 words. Part of this IS the "
                  "ablation (the removed move is first-person by construction), but the arms "
                  "therefore differ stylistically as well as in the target property — treat "
                  "'it stopped saying I' as a live alternative explanation for any downstream "
                  "difference. Residual: ~6% of substitutes were judged to reintroduce the "
                  "property after repair.",
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
