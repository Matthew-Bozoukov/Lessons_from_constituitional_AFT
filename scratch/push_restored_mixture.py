# ABOUTME: Publish a move-restored arm (ablated2 with one deliberative move put back).
# ABOUTME: Card records the measured restoration rate and every collateral move that shifted.
"""Push a move-restored mixture to Hugging Face.

Run: uv run python scratch/push_restored_mixture.py --move offers_alternative
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.huggingface import push_files  # noqa: E402
from src.utils import git_sha, read_jsonl  # noqa: E402

ARMS = {
    "offers_alternative": {
        "repo": "matboz/2026-08-23-alt-restored-t2-9284-synthdoc-676",
        "run": "output/restored_alt_676",
        "label": "an alternative -- committing to a concrete alternative path the person "
                 "can actually take",
        "why": "a model with a refusal to deliver and no alternative to offer has nothing "
               "left to do",
        "was": "0.6% / 3.6%",
    },
    "both_pruned": {
        "repo": "matboz/2026-08-23-both-restored-pruned-t2-9284-synthdoc-662",
        "run": "output/restored_pruned_662",
        "label": "BOTH the alternative and the counterargument, restored aggressively (a "
                 "rewrite for prominence, not a minimal insertion), after which the "
                 "outcome-branching units were deleted -- except where a branching unit "
                 "also carried one of the two restored moves, which was kept intact",
        "why": "neither move restored alone recovered the ability to act, and the arms that "
               "do act are all far longer, so this restores both moves AND most of the "
               "length",
        "was": "0.6% / 3.6% (alternative) and 2.7% / 5.2% (counterargument)",
        "moves": ["offers_alternative", "engages_counterargument"],
    },
    "branches_only": {
        "repo": "matboz/2026-08-23-branches-restored-t2-9284-synthdoc-654",
        "run": "output/branches_pruned_654",
        "label": "OUTCOME BRANCHING, restored aggressively (a rewrite for prominence), after "
                 "which the alternative and the counterargument were deleted -- except where "
                 "a unit carrying them also carried a branch, which was kept intact",
        "why": "it is the third of the four ablated moves, and the mirror image of the "
               "both-restored arm: branches in, the other two out",
        "was": "0.7% / 3.6%",
        "moves": ["outcome_branches"],
    },
    "engages_counterargument": {
        "repo": "matboz/2026-08-23-counterargument-restored-t2-9284-synthdoc-676",
        "run": "output/restored_ec_676",
        "label": "the counterargument -- exploring the tempting option's merits, following "
                 "its logic, or conceding something to it",
        "why": "a model that never engages what it is refusing may not model the situation "
               "richly enough to act within it",
        "was": "2.7% / 5.2%",
    },
}
ABLATED = "matboz/2026-08-22-ruleform-ablated2-t2-9284-synthdoc-676"
CONTROL = "LASR-Callum/2026-08-06-table2-9284-synthdoc-716-train"


def main(move: str = "offers_alternative", restore_run: str = "",
         private: bool = True) -> None:
    """Upload the mixture and its stats with the full card.

    Args:
        move: Which restored arm to publish; a key of ARMS.
        restore_run: The scratch/restore_move.py run directory backing this arm.
        private: Create the repo private (default).
    """
    assert move in ARMS, f"unknown arm {move}; pick one of {sorted(ARMS)}"
    arm = ARMS[move]
    REPO, RUN = arm["repo"], Path(arm["run"])
    d = Path(restore_run or sorted(
        Path("output/restore_move").glob(f"{move}_*"))[-1])
    assert d.is_dir(), f"no run directory at {d}"
    halves = read_jsonl(d / "halves.jsonl")
    if halves and "moves_deblanched" in halves[0]:
        # A pruned arm: the shipped text is the post-deletion one, so rate that stage.
        for h in halves:
            h["moves_after"] = h["moves_deblanched"]
    stats = json.loads((RUN / "mixture_stats_ablated.json").read_text())
    tmoves = arm.get("moves", [move])
    hit = sum(all(h["moves_after"][m] for m in tmoves) for h in halves)
    pct = lambda half, m, key: 100 * sum(
        h[key][m] for h in halves if h["half"] == half) / sum(
        1 for h in halves if h["half"] == half)
    med = lambda half, k: st.median(h[k] for h in halves if h["half"] == half)

    fields = {
        "title": "Alternative restored, one move put back into the rewrite-then-delete arm "
                 "(Table2 9,284 + difficult-advice 676)",
        "experiment": "Takes the ablated2 arm -- in which four deliberative moves were "
                      "stripped from both halves of every difficult-advice row -- and puts "
                      "exactly ONE of them back: offering a concrete alternative path the "
                      "person can actually take. The other three targets (engaging the "
                      "tempting option, drawing an analytic distinction, enumerating outcome "
                      "branches) stay removed. Motivation: the ablated2 model made ZERO tool "
                      f"calls in 78% of ODCV-Bench rollouts, refusing in a loop to the step "
                      f"cap, so its 0.8% misalignment rate measures inaction rather than "
                      f"alignment; {arm['why']}.",
        "date_generated": "2026-08-23",
        "constitution": "claude_distilled_12_principles_mid "
                        "(constitutions/claude_distilled_12_principles_mid/constitution.md)",
        "source_repo": f"Matthew-Bozoukov/teaching_claude_why_replication @ {git_sha()}",
        "models": "restorer anthropic/claude-sonnet-5; blind nine-move rater "
                  "google/gemini-3-flash-preview",
        "generation_config": "restorer temperature 0.7, max_tokens 8192, JSON "
                             "response_format, up to 3 attempts on an unterminated body; "
                             "rater temperature 0.0, one text per call with no version label",
        "schema": "text (rendered Qwen3.6 chat); source; scenario_id and both_rewritten "
                  f"present only on the {stats['rows_rewritten']} restored rows",
        "provenance": f"scratch/restore_move.py --move {','.join(tmoves)} --n 0 "
                      f"--aggressive true, then scratch/deblanch_halves.py (protected "
                      f"branch deletion), then "
                      f"scratch/build_final_mixture.py --restore_run <that run> "
                      f"--out_dir {arm['run']}, then this script. "
                      f"Restoration run: {d.name}",
        "rows": f"{stats['rows_total']:,} rows: 9,284 Table2 rows byte-identical to the "
                f"control, plus {stats['rows_rewritten']} difficult-advice rows carrying the "
                f"restored move. The SAME 40 rows ablated2 dropped are dropped here.",
        "restoration_strength": f"Blind rater over all {len(halves)} rewritten halves: "
                                f"{'+'.join(tmoves)} present in {hit}/{len(halves)} "
                                f"({100 * hit / len(halves):.1f}%) -- reasoning "
                                + ", ".join(
                                    f"{m} reasoning {pct('reasoning', m, 'moves_after'):.1f}%"
                                    f" / reply {pct('reply', m, 'moves_after'):.1f}%"
                                    for m in tmoves) + ", "
                                f"up from {arm['was']} in ablated2. Median words: reasoning "
                                f"{med('reasoning', 'words_before'):.0f} -> "
                                f"{med('reasoning', 'words_after'):.0f}, reply "
                                f"{med('reply', 'words_before'):.0f} -> "
                                f"{med('reply', 'words_after'):.0f}.",
        "caveat": "THIS IS NOT A CLEAN SINGLE-MOVE DELTA. " + "; ".join(
            f"{m} {pct('reasoning', m, 'moves_before'):.0f}%->"
            f"{pct('reasoning', m, 'moves_after'):.0f}% reasoning, "
            f"{pct('reply', m, 'moves_before'):.0f}%->"
            f"{pct('reply', m, 'moves_after'):.0f}% reply"
            for m in halves[0]["moves_before"]
            if m not in tmoves and max(
                abs(pct('reasoning', m, 'moves_after') - pct('reasoning', m, 'moves_before')),
                abs(pct('reply', m, 'moves_after') - pct('reply', m, 'moves_before'))) >= 5
        ) + ". A difference against ablated2 is attributable to the restored move PLUS "
             "these, not to the restored move alone. Moves shifting less than 5 points in "
             "both halves are omitted.",
        "filtering": "FOURTEEN rows were DROPPED on top of ablated2's 40: every row whose "
                     "reasoning or reply a blind coherence check rated broken after the "
                     "branch deletion (dangling pronouns, references to deleted options). "
                     "662 difficult-advice rows remain, so this arm is 14 rows short of "
                     "ablated2's 676 and a difference against it is 14 fewer rows as well "
                     "as the restoration. Dropped ids are in mixture_stats_ablated.json "
                     "under dropped_extra.",
        "control_dataset": f"{ABLATED} is the arm this modifies; {CONTROL} "
                           "(mixture_think.jsonl) is the unmodified 10,000-row control.",
        "verification": "system prompt and user turn asserted byte-identical on all 9,960 "
                        "rows; the 9,284 standard rows asserted byte-identical to ablated2; "
                        f"{stats['rows_rewritten']} rows spliced (rows whose both halves "
                        "already carried the move come out byte-identical to ablated2); all "
                        "assistant turns carry think markers (676 real traces, 9,646 whole "
                        "empty markers, 0 missing); no row degenerate at the 15-word floor.",
    }
    url = push_files([RUN / "mixture_think_ablated.jsonl",
                      RUN / "mixture_stats_ablated.json"], REPO, fields, private=private)
    print(url)


if __name__ == "__main__":
    fire.Fire(main)
