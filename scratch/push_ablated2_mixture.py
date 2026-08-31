# ABOUTME: Publish the rewrite-then-delete mixture (both halves, four moves, then the
# ABOUTME: alternative deleted) to Hugging Face as its own dataset.
"""Push the rewrite-then-delete arm's mixture.

Run: uv run python scratch/push_ablated_mixture.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.huggingface import card_front_matter, card_markdown, hf_api, push_files  # noqa: E402
from src.utils import git_sha, read_jsonl  # noqa: E402

REPO = "matboz/2026-08-22-ruleform-ablated2-t2-9284-synthdoc-676"
RUN = Path("output/ruleform_ablated2_676")
CONTROL = "LASR-Callum/2026-08-06-table2-9284-synthdoc-716-train"
TGT = ("engages_counterargument", "analytic_distinction", "outcome_branches",
       "offers_alternative")


def _survival(drop: set) -> tuple[dict, dict, int, int, int, int]:
    """Move survival in each half after both passes, excluding dropped rows."""
    def pool(glob):
        o = {}
        for d in sorted(Path("output/ablate_alternative").glob(glob)):
            for r in read_jsonl(d / "texts.jsonl"):
                if r["scenario_id"] not in drop:
                    o[r["scenario_id"]] = r
        return o
    R, A = pool("reasoning_*"), pool("answer_*")
    sids = sorted(set(R) & set(A))
    n = len(sids)
    rs = {m: sum(1 for s in sids if R[s]["moves"][m]) for m in R[sids[0]]["moves"]}
    as_ = {m: sum(1 for s in sids if A[s]["moves"][m]) for m in A[sids[0]]["moves"]}
    cr = sum(1 for s in sids if not any(R[s]["moves"][m] for m in TGT))
    ca = sum(1 for s in sids if not any(A[s]["moves"][m] for m in TGT))
    coh = sum(1 for s in sids
              if R[s]["coherence"] == "coherent" and A[s]["coherence"] == "coherent")
    return rs, as_, n, cr, ca, coh


def main(private: bool = True) -> None:
    """Upload the mixture and its stats.

    Args:
        private: Create the repo private (default).
    """
    st = json.loads((RUN / "mixture_stats_ablated.json").read_text())
    drop = set(st["dropped_rows"])
    rs, as_, n, cr, ca, coh = _survival(drop)
    pct = lambda d, m: f"{100*d[m]/n:.0f}%"
    strength = "; ".join(f"{m} reasoning {pct(rs, m)} / answer {pct(as_, m)}" for m in TGT)

    fields = {
        "title": "Rewrite-then-delete, two ablation passes, responsiveness-filtered "
                 "(Table2 9,284 + difficult-advice 676)",
        "experiment": "Ablation arm built in two passes over BOTH halves of every "
                      "difficult-advice row. Pass 1 rewrites the reasoning and the answer to "
                      "drop four deliberative moves: engaging the tempting option, drawing an "
                      "analytic distinction, enumerating outcome branches, and offering an "
                      "alternative route. Pass 2 then labels every remaining unit and DELETES "
                      "the ones that still offer an alternative. A SECOND deletion pass over "
                      "the answers then removes units offering an alternative OR engaging the "
                      "tempting option. Neither method works alone: rewriting left the "
                      "alternative in 34% of answers because the model would not fully "
                      "comply, and deleting alone left it in 40% because the offering is "
                      "diffuse. Rewrite plus two deletion passes takes it to 3.6%.",
        "date_generated": "2026-08-22",
        "constitution": "claude_distilled_12_principles_mid "
                        "(constitutions/claude_distilled_12_principles_mid/constitution.md)",
        "source_repo": f"Matthew-Bozoukov/teaching_claude_why_replication @ {git_sha()}",
        "models": "rewriter anthropic/claude-sonnet-5; unit labeller and blind rater "
                  "google/gemini-3-flash-preview",
        "generation_config": "rewriter temperature 0.7, max_tokens 8192; labeller and rater "
                             "temperature 0.0 with JSON response_format; rated items shuffled "
                             "and shown one per call with no version label",
        "schema": "text (rendered Qwen3.6 chat); source; scenario_id and both_rewritten "
                  "present only on the 711 rewritten rows",
        "provenance": "scratch/run_rewrite_pipeline.py --stage rewrite, then --stage ablate, "
                      "then scratch/ablate_alternative.py --half answer --moves "
                      "offers_alternative,engages_counterargument --tag pass2, then "
                      "scratch/rate_answer_responsiveness.py, then "
                      "scratch/build_final_mixture.py. Sample manifest: "
                      "output/rewrite_cache/sids.json",
        "control_dataset": f"{CONTROL} (mixture_think.jsonl) — the unmodified 10,000 rows. "
                           "Siblings: matboz/2026-08-21-ruleform-t2-9284-synthdoc-716 "
                           "(reasoning only) and "
                           "matboz/2026-08-22-ruleform-both-t2-9284-synthdoc-716 (both "
                           "halves, rewrite only).",
        "rows": f"9,960 rows: 9,284 Table2 rows byte-identical to the control, plus {n} "
                "difficult-advice rows with both halves rewritten and ablated twice. FORTY "
                "rows were DROPPED: 16 whose answer the deletion emptied, and 24 that a blind "
                "responsiveness rater judged no longer to answer the question they were "
                "written for.",
        "verification": "system prompt and user turn asserted byte-identical on all rows; the "
                        "pre-swap answer asserted equal to the pipeline's own record; exactly "
                        f"{n} rows differ; all assistant turns carry think markers ({n} real "
                        f"traces, 9,646 whole empty markers, 0 missing). {coh}/{n} rows rated "
                        "coherent in both halves after deletion.",
        "ablation_strength": f"Blind rater over all {n} rows, AFTER both passes: {strength}. "
                             f"All four targets absent in {cr}/{n} reasoning traces and "
                             f"{ca}/{n} answers.",
        "caveat": "THE ANALYTIC DISTINCTION DID NOT LAND, and it is the whole reason this arm "
                  f"is not a clean test. It survives in {pct(rs, 'analytic_distinction')} of "
                  f"reasoning traces and {pct(as_, 'analytic_distinction')} of answers despite "
                  "being an explicit target of the rewrite; it resisted a lexical-ban pass, a "
                  "span-quoting repair loop, and a prompt-stripping probe. Three of four "
                  "targets are essentially gone; read any result as 'engagement, branching and "
                  "the alternative removed, distinction intact'. SECOND CONFOUND: length. "
                  "Reasoning falls to ~34% of the control and answers to ~22%, and untargeted "
                  "moves fell with them (returns_agency in answers 81% -> 15%, names_tension "
                  "94% -> 10%). A downstream difference could be terseness rather than the "
                  "four moves. THIRD CONFOUND, introduced by the filtering itself: the "
                  "dropped rows are not spread evenly. Twenty of the forty are trait 8 (\"be "
                  "genuinely, substantively helpful -- never treat unhelpfulness as safe\"), "
                  "leaving 59 rows of that trait against ~79 for the others, because in those "
                  "scenarios the alternative and the engagement WERE the answer. The corpus "
                  "is therefore thinner on the trait most opposed to the ablation, and that "
                  "imbalance is itself a candidate explanation for any downstream effect.",
    }
    url = push_files([RUN / "mixture_think_ablated.jsonl",
                      RUN / "mixture_stats_ablated.json"], REPO, fields, private=private)
    card = (card_front_matter([{"config_name": "default",
                                "data_files": "mixture_think_ablated.jsonl",
                                "default": True}]) + card_markdown(fields))
    hf_api().upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                         repo_id=REPO, repo_type="dataset")
    print(url)


if __name__ == "__main__":
    fire.Fire(main)
