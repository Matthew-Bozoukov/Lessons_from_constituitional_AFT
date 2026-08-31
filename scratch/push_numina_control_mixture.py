# ABOUTME: Publish the no-difficult-advice control: the filtered 9,284 instruction rows plus
# ABOUTME: 716 extra NuminaMath-CoT rows filling the slots difficult advice used to occupy.

"""Push the control arm for the difficult-advice ablation series.

Every other arm keeps 716 difficult-advice traces and varies how they reason or which
scenarios they cover. This one removes the category outright and backs the slots with more of
a source already in the mixture. The card leads with the thinking-mode consequence, because it
is what stops this arm being compared naively against the others.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.huggingface import card_front_matter, card_markdown, hf_api, push_files  # noqa: E402
from src.utils import git_sha  # noqa: E402

REPO = "matboz/2026-08-19-numina-control-9284-plus-716"
RUN = Path("output/numina_control/20260819_102119")
BASE = "the sibling arms mixture (LASR-Callum/2026-08-06-table2-9284-synthdoc-716-train, non-difficult-advice rows)"


def main(private: bool = True) -> None:
    """Upload the control mixture and its build record.

    Args:
        private: Create the repo private (default).
    """
    st = json.loads((RUN / "mixture_stats_numina_control.json").read_text())
    fields = {
        "title": "No-difficult-advice control (9,284 filtered + 716 extra NuminaMath-CoT)",
        "experiment": "Control arm for the difficult-advice ablation series. The five existing "
                      "arms all keep 716 difficult-advice traces and vary how those traces "
                      "reason (meta-reasoning excised, one-sided advocacy, loss-masked) or "
                      "which scenarios they cover (low-ODCV selection, traits 1/3/4 removed). "
                      "This arm removes difficult advice entirely and fills the 716 slots with "
                      "more NuminaMath-CoT, a source already present in the mixture, so the "
                      "only change is the absence of the category under study. The added rows "
                      "carry real reasoning traces, so the arm still trains the model to think "
                      "— about mathematics rather than about values.",
        "date_generated": "2026-08-19",
        "constitution": "none — this arm deliberately contains no constitution-grounded data; "
                        "it is the control against which the difficult-advice arms "
                        "(claude_distilled_07_principles_approved) are measured",
        "source_repo": f"Matthew-Bozoukov/teaching_claude_why_replication @ {git_sha()}",
        "models": "none — no text was generated; rows are drawn unmodified from existing corpora",
        "generation_config": f"deterministic given seed {st['seed']}; the 716 extra rows are "
                             "sampled from HuggingFaceTB/smoltalk numina-cot-100k after "
                             "shuffling, skipping any whose user prompt already appears in the "
                             "base 9,284",
        "schema": "text (rendered Qwen3.6 chat); source",
        "provenance": "scratch/build_numina_control_mixture.py",
        "base_dataset": f"{BASE} — the 9,284 rows are byte-identical to what all five "
                        "difficult-advice arms trained on, so this control differs from them "
                        "ONLY in what fills the remaining 716 slots. NOTE: these are the same "
                        "9,284 rows as LASR-Callum/2026-08-04-table2-instruction-tuning-9284-"
                        "filtered-8192, but taken from the newer re-render — that earlier build "
                        "predates the preserve-thinking policy and leaves 362 of its 9,646 "
                        "assistant turns (no_robots, lima multi-turn rows) without a think "
                        "marker, which the training mask gate refuses.",
        "rows": f"{st['rows_total']:,} = 9,284 base instruction-tuning rows + "
                f"{st['rows_added']} extra numinamath_cot rows. Source composition goes from "
                f"numinamath_cot 1,037 to 1,753; every other source is unchanged.",
        "thinking_mode": "TRUE. NuminaMath-CoT states a worked derivation and then a final "
                         "boxed answer, so each added row is split at the last paragraph "
                         "carrying \\boxed{}: the derivation becomes the <think> trace and the "
                         "answer paragraph becomes the response. All 716 added rows carry a real "
                         "trace and the 9,284 base rows keep their whole empty markers — the "
                         "same 716-real / 9,284-empty shape as the five difficult-advice arms, "
                         "so this arm trains and is served in thinking mode and IS comparable "
                         "to them. Rows whose split would leave a derivation under 200 "
                         "characters are rejected rather than repaired (98 skipped).",
        "verification": "9,284 base rows byte-identical to the base dataset; every one of the "
                        "716 added rows carries exactly one assistant turn and one well-formed "
                        "non-empty <think> block, and none carries an empty marker; no added "
                        "row duplicates a base row's user prompt; and all 10,362 assistant "
                        "turns across the mixture carry a think block, which is what the "
                        "training mask gate checks",
    }
    url = push_files([RUN / "mixture_think_numina_control.jsonl",
                      RUN / "mixture_stats_numina_control.json"], REPO, fields, private=private)
    card = (card_front_matter([{"config_name": "default",
                                "data_files": "mixture_think_numina_control.jsonl",
                                "default": True}]) + card_markdown(fields))
    hf_api().upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                         repo_id=REPO, repo_type="dataset")
    print(url)


if __name__ == "__main__":
    fire.Fire(main)
