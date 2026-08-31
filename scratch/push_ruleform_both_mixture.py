# ABOUTME: Publish the both-halves rule-form mixture (reasoning AND answer rewritten) to HF
# ABOUTME: as its own dataset, separate from the reasoning-only arm and the control.
"""Push the both-halves rule-form mixture.

Run: uv run python scratch/push_ruleform_both_mixture.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.huggingface import card_front_matter, card_markdown, hf_api, push_files  # noqa: E402
from src.utils import git_sha, read_jsonl  # noqa: E402

REPO = "matboz/2026-08-22-ruleform-both-t2-9284-synthdoc-716"
RUN = Path("output/ruleform_both_716")
CONTROL = "LASR-Callum/2026-08-06-table2-9284-synthdoc-716-train"
REASONING_ARM = "matboz/2026-08-21-ruleform-t2-9284-synthdoc-716"


def main(private: bool = True) -> None:
    """Upload the mixture, its stats and the per-row answer rewrites.

    Args:
        private: Create the repo private (default).
    """
    st = json.loads((RUN / "mixture_stats_both.json").read_text())
    rated = sorted(Path("output/invert_answer_full").glob("rated_*"))[-1]
    items = read_jsonl(rated / "ratings.jsonl")
    byv = {v: {i["sid"]: i for i in items if i["version"] == v}
           for v in ("original", "rewritten")}
    sids = sorted(byv["original"])
    def cnt(v, m):
        return sum(1 for s in sids if byv[v][s]["moves"][m])
    tg = ("engages_counterargument", "analytic_distinction", "outcome_branches",
          "offers_alternative")
    strength = "; ".join(f"{m} {cnt('original', m)}/716 -> {cnt('rewritten', m)}/716"
                         for m in tg)
    clean = sum(1 for s in sids
                if not any(byv["rewritten"][s]["moves"][m] for m in tg))

    fields = {
        "title": "Rule-form BOTH HALVES (Table2 9,284 + difficult-advice 716)",
        "experiment": "Ablation arm: on all 716 difficult-advice rows BOTH the reasoning and "
                      "the assistant's answer are rewritten. The reasoning drops three moves "
                      "(engaging the tempting option, drawing an analytic distinction, "
                      "enumerating outcome branches); the answer drops those three plus "
                      "offering an alternative route. The system prompt and user turn are "
                      "unchanged on every row.",
        "date_generated": "2026-08-22",
        "constitution": "claude_distilled_12_principles_mid "
                        "(constitutions/claude_distilled_12_principles_mid/constitution.md)",
        "source_repo": f"Matthew-Bozoukov/teaching_claude_why_replication @ {git_sha()}",
        "models": "rewriter anthropic/claude-sonnet-5; blind rater "
                  "google/gemini-3-flash-preview",
        "generation_config": "rewriter temperature 0.7, max_tokens 8192; rater temperature "
                             "0.0 with JSON response_format, items shuffled and shown one per "
                             "call with no version label",
        "schema": "text (rendered Qwen3.6 chat); source; scenario_id, excised and "
                  "answer_rewritten present only on the 716 rewritten rows",
        "provenance": "scratch/invert_three_moves.py --n 0 (reasoning), "
                      "scratch/invert_answer_moves.py --moves "
                      "engages_counterargument,analytic_distinction,outcome_branches,"
                      "offers_alternative --n 0 (answers), then "
                      "scratch/build_ruleform_both_mixture.py",
        "control_dataset": f"{CONTROL} (mixture_think.jsonl) — same 10,000 rows, unrewritten. "
                           f"Reasoning-only sibling: {REASONING_ARM}.",
        "rows": "10,000 rows; all 716 difficult-advice rows have both halves rewritten; the "
                "9,284 Table2 rows are byte-identical to the control",
        "verification": "system prompt and user turn asserted byte-identical on all 10,000 "
                        "rows; each row's pre-rewrite answer asserted equal to the rewrite "
                        "source; think block asserted unchanged by the answer splice; exactly "
                        "716 rows differ. All 10,000 assistant turns carry think markers "
                        "(716 real traces, 9,646 whole empty markers).",
        "ablation_strength": f"Blind rater over both versions of all 716 ANSWERS: {strength}. "
                             f"All four targets absent in {clean}/716 answers. Reasoning-side "
                             "figures are on the sibling repo.",
        "caveat": "TWO TARGETS DID NOT LAND AND THE LENGTH CHANGE IS LARGE. In the answer the "
                  "analytic distinction survives in 422/716 (59%) and the alternative in "
                  "313/716 (44%), despite both being explicit targets — the same pattern as "
                  "the reasoning side, where the distinction survived in 529/716. Answer "
                  f"length falls to {st['answer_words_ratio']:.0%} of the control "
                  f"({st['answer_words_before']:,} -> {st['answer_words_after']:,} words) on "
                  "top of reasoning already at 62%. Untargeted moves also fell sharply in the "
                  "answer (returns_agency 577->234, names_tension 673->129, "
                  "acknowledges_pressure 703->503). Any downstream difference could be this "
                  "collapse in length and helpfulness rather than the four moves; a "
                  "length-protected three-move variant would isolate them far better.",
    }
    url = push_files([RUN / "mixture_think_both.jsonl",
                      RUN / "mixture_stats_both.json",
                      rated / "replies.jsonl"], REPO, fields, private=private)
    card = (card_front_matter([{"config_name": "default",
                                "data_files": "mixture_think_both.jsonl",
                                "default": True}]) + card_markdown(fields))
    hf_api().upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                         repo_id=REPO, repo_type="dataset")
    print(url)


if __name__ == "__main__":
    fire.Fire(main)
