# ABOUTME: Publish the CoT-only supervision mixture to Hugging Face with the required card.
# ABOUTME: Run: uv run python scratch/cot_only/push_mixture.py --run <output dir>

"""Push the CoT-only supervision arm's training mixture.

The repo holds ONE data file — the 10,000-row mixture whose `text` is byte-identical to
the control's and whose only difference is a per-row `supervise` field on the 716
difficult-advice rows. The stats sidecar goes up beside it so the intervention's scale is
auditable without rebuilding.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scratch.cot_only.build_mixture import (  # noqa: E402
    CHUNK_ONLY, CONTROL_FILE, CONTROL_REPO, CONTROL_REVISION, DA_SOURCE,
)
from src.huggingface import (  # noqa: E402
    card_front_matter, card_markdown, hf_api, hf_repo_id, push_files,
)
from src.utils import git_sha  # noqa: E402

REPO = "2026-08-31-cot-only-supervision-t2-9284-synthdoc-716"
# The chunk-only (principle-scoped) arm. Preferred, because its control is REAL and
# EVALUATED -- see build_mixture.CHUNK_ONLY.
REPO_CHUNK_ONLY = "2026-08-31-cot-only-supervision-chunk-only-702"
# The ANSWER-ONLY arm on the same base: the exact complement of the CoT one, sharing
# the same control. Trace stays in the forward pass, earns no loss.
REPO_ANSWER_CHUNK = "2026-09-01-answer-only-supervision-chunk-only-702"
DATA_FILE = "mixture_think_cotonly.jsonl"


def main(run: str, private: bool = False, arm: str = "", mode: str = "cot") -> None:
    """Upload the mixture, its stats and its card.

    Args:
        run: The build output directory (output/cot_only_mixture/<ts>).
        private: Create the repo private. Defaults to PUBLIC, matching every sibling
            arm's mixture and adapter.
        arm: "chunk_only" publishes the principle-scoped rebuild to its own repo and
            cites its own control; anything else keeps the original synthdoc arm.
        mode: "cot" or "answer" -- which half of the difficult-advice turn the flagged
            rows train on. Selects the repo and the card's wording.
    """
    co = arm == "chunk_only"
    ans = mode == "answer"
    assert not (ans and not co), "the answer-only arm is only built on the chunk-only base"
    repo = (REPO_ANSWER_CHUNK if ans else REPO_CHUNK_ONLY) if co else REPO
    spec = CHUNK_ONLY if co else {"repo": CONTROL_REPO, "file": CONTROL_FILE,
                                  "revision": CONTROL_REVISION, "da_source": DA_SOURCE}
    ctl_repo, ctl_file = spec["repo"], spec["file"]
    ctl_rev, da_src = spec["revision"], spec["da_source"]
    run_p = Path(run)
    stats = json.loads((run_p / "mixture_stats_cotonly.json").read_text())
    fwd, sup = stats["forward_tokens"], stats["supervised_tokens"]
    fields = {
        "title": ("Answer-only supervision mixture, principle-scoped "
                  "(Table2 9,284 + chunk-only 702)" if ans else
                  "CoT-only supervision mixture, principle-scoped "
                  "(Table2 9,284 + chunk-only 702)" if co else
                  "CoT-only supervision mixture (Table2 9,284 + difficult-advice 716)"),
        "experiment": (
            "Arm: train the 702 principle-scoped difficult-advice rows on their VISIBLE "
            "ANSWER ONLY -- the reasoning trace stays in the token stream as unsupervised "
            "context (no truncation, full forward pass) and simply earns no loss, while "
            "the 9,284 Table2 rows train exactly as in the control. The EXACT COMPLEMENT "
            "of the CoT-only arm on the same base: on every one of the 702 rows, "
            "supervised(cot) + supervised(answer) == supervised(control), verified "
            "token-for-token (420,037 + 401,033 = 821,070, 0 mismatches). That partition "
            "is what makes the two arms comparable to EACH OTHER, not merely to their "
            "shared control."
            if ans else
            "Arm: train the difficult-advice rows on their REASONING ONLY - each row is "
            "truncated at its reasoning close, so the visible answer leaves both the loss "
            "and the forward pass - while the Table2 rows train as in the control."),
        "date_generated": "2026-08-31",
        "constitution": "claude_distilled_07_principles_approved "
                        "(constitutions/claude_distilled_07_principles_approved/constitution.md)",
        "source_repo": f"Matthew-Bozoukov/teaching_claude_why_replication @ {git_sha()}",
        "models": "token stream Qwen/Qwen3.6-27B (tokenizer + ModelProfile literals)",
        "generation_config": "none — no model is sampled here. The build is a "
                             "deterministic per-row field addition over a pinned control.",
        "schema": ("text (rendered Qwen3.6 chat, IDENTICAL to the control mixture); "
                   "source (mixture source name); supervise "
                   f"('{mode}' on the {702 if co else 716} {da_src} rows, "
                   "absent elsewhere = 'all')"),
        "provenance": (
            (f"uv run python scratch/cot_only/build_mixture.py --repo {ctl_repo} "
             f"--file {ctl_file} --revision {ctl_rev} --da_source {da_src} "
             f"--n_da {CHUNK_ONLY['n_da']} --n_rows {CHUNK_ONLY['n_rows']} ; verified by "
             f"--mode {mode} ; verified by uv run python "
             f"scratch/cot_only/verify_mixture.py --arm chunk_only --mode {mode}" if co else
             "uv run python scratch/cot_only/build_mixture.py ; verified by "
             "scratch/cot_only/verify_mixture.py")
            + " (text byte-identical to the control on every row, mask gate passed on "
              "BOTH supervise modes with the real Qwen3.6 tokenizer, stratified sample "
              "64 all + 64 cot, 0 truncated)"),
        "control_dataset": f"{ctl_repo} ({ctl_file} @ {ctl_rev[:12]}) "
                           "- same text, no supervise column",
        "trained_control": (
            "LASR-Callum/2026-08-21-qwen36-lora-table2-9284-difficult-advice-chunk-only-"
            "702-rank-64-dynbatch - ODCV 11.5% [6.2, 19.6], severity 0.62 (2 passes, 65 "
            "cells). Trained on exactly this control mixture, so the arm differs from it "
            "in the `supervise` field alone."
            if co else
            "NONE EXISTS - the protocol-matched synthdoc-716 control was never trained."),
        "rows": f"{stats['rows']} ({stats['rows_supervise_cot']} carry supervise=cot)",
        "intervention_scale": (
            f"at max_seq_len {stats['max_length']}: forward tokens "
            f"{fwd['control']:,} -> {fwd['cot_only']:,} (-{fwd['reduction_pct']}% overall, "
            f"-{fwd['da_reduction_pct']}% on the difficult-advice rows); supervised tokens "
            f"{sup['control']:,} -> {sup['cot_only']:,} (-{sup['reduction_pct']}%); "
            f"difficult-advice share of the training signal "
            f"{sup['da_share_control_pct']}% -> {sup['da_share_cot_only_pct']}%"
        ),
        "usage": "src/train/train_lora.py consumes `supervise` via build_labels; a trainer "
                 "that ignores the column silently trains the control instead. Consumed by "
                 "configs/train/lora_qwen36_t2_9284_synthdoc_716_cotonly_dynbatch_2xh200"
                 ".yaml; the control is the sibling config without `_cotonly`.",
        "caveat": "seq_mean_token_mean_loss weights each EXAMPLE at 1/global_batch, so "
                  "this arm concentrates the same per-example weight onto the trace rather "
                  "than reducing the rows' influence — roughly doubling the per-CoT-token "
                  "gradient weight. It is 'reasoning only, at double density', not 'the "
                  "control minus its answer term'.",
    }
    url = push_files([run_p / DATA_FILE,
                      run_p / "mixture_stats_cotonly.json",
                      run_p / "run_meta.json"], repo, fields, private=private)
    card = (card_front_matter([{"config_name": "default",
                                "data_files": DATA_FILE,
                                "default": True}])
            + card_markdown(fields))
    hf_api().upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                         repo_id=hf_repo_id(repo), repo_type="dataset")
    print(url)


if __name__ == "__main__":
    fire.Fire(main)
