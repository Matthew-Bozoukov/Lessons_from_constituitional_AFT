# ABOUTME: Publish the empty-CoT mixture to Hugging Face with the required card.
# ABOUTME: Run: uv run python scratch/empty_cot/push_mixture.py --run <build output dir>

"""Push the empty-CoT arm's training mixture.

A sibling of scratch/cot_only/push_mixture.py rather than a mode of it, for the same
reason the builders are siblings: that arm's card asserts the text is byte-identical to
the control, and this arm's central fact is the opposite. Sharing a card writer between
two opposite claims is how a card ends up lying about its data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scratch.cot_only.build_mixture import CHUNK_ONLY  # noqa: E402
from src.huggingface import (  # noqa: E402
    card_front_matter, card_markdown, hf_api, hf_repo_id, push_files,
)
from src.utils import git_sha  # noqa: E402

REPO = "2026-09-01-empty-cot-supervision-chunk-only-702"
DATA_FILE = "mixture_think_emptycot.jsonl"


def main(run: str, private: bool = False) -> None:
    """Upload the mixture, its stats and its card.

    Args:
        run: The build output directory (output/empty_cot_mixture/<ts>).
        private: Create the repo private. Defaults to PUBLIC, matching the siblings.
    """
    run_p = Path(run)
    stats = json.loads((run_p / "mixture_stats_emptycot.json").read_text())
    fwd, sup, cen = (stats["forward_tokens"], stats["supervised_tokens"],
                     stats["census"])
    fields = {
        "title": "Empty-CoT supervision mixture, principle-scoped "
                 "(Table2 9,284 + chunk-only 702)",
        "experiment": (
            "Arm: REPLACE each of the 702 principle-scoped difficult-advice rows' "
            "reasoning traces with the empty think marker, leaving prompt and answer "
            "byte-identical. The marker is masked whole by the existing "
            "generation-boundary rule, so the model is supervised on the visible answer "
            "and never learns to emit an empty close. Tests whether the reasoning was "
            "doing work AS CONTEXT: this arm and the answer-only arm supervise the same "
            "answer tokens and differ only in whether the trace is present to condition "
            "on."),
        "date_generated": "2026-09-01",
        "constitution": "constitutions/claude_distilled_07_principles_approved/constitution.md "
                        "(via the training data; never quoted in the trained text)",
        "source_repo": f"teaching_claude_why_replication @ {git_sha()}",
        "models": "token stream Qwen/Qwen3.6-27B (tokenizer + ModelProfile literals)",
        "generation_config": "none - no model is sampled here. The build is a "
                             "deterministic text rewrite over a pinned control mixture.",
        "schema": "text (rendered Qwen3.6 chat; the 702 difficult-advice rows have their "
                  "think block replaced by the empty marker, prompt and answer unchanged; "
                  "the 9,284 Table2 rows are byte-identical to the control); source. NO "
                  "`supervise` column: the default 'all' already masks the whole empty "
                  "marker and supervises the answer, so no new mask mode was needed.",
        "provenance": (
            "uv run python scratch/empty_cot/build_mixture.py ; every rewritten row "
            "asserted to keep its prompt and answer byte-for-byte, to open with the exact "
            "empty marker, and the census asserted at 0 real / all empty / 0 absent"),
        "control_dataset": f"{CHUNK_ONLY['repo']} ({CHUNK_ONLY['file']} @ "
                           f"{CHUNK_ONLY['revision'][:12]}) - same prompts and answers, "
                           "real traces intact",
        "trained_control": (
            "LASR-Callum/2026-08-21-qwen36-lora-table2-9284-difficult-advice-chunk-only-"
            "702-rank-64-dynbatch - ODCV 11.5% [6.2, 19.6], severity 0.62 (2 passes)."),
        "sibling_arms": (
            "cot-only (LASR-Callum/2026-08-31-cot-only-supervision-chunk-only-702, ODCV "
            "9.5% [3.2, 17.5]) and answer-only "
            "(LASR-Callum/2026-09-01-answer-only-supervision-chunk-only-702). Those two "
            "partition the control's supervision exactly; THIS arm instead changes the "
            "context, supervising the same answer tokens as answer-only (702 fewer - the "
            "blank-line separator, which is part of the forced marker here)."),
        "rows": f"{stats['rows']} ({stats['rows_rewritten']} traces blanked)",
        "census": f"{cen['real']} real / {cen['empty']} empty / {cen['absent']} absent "
                  "assistant turns",
        "intervention_scale": (
            f"at max_seq_len {stats['max_length']}: forward tokens {fwd['control']:,} -> "
            f"{fwd['empty_cot']:,} (-{fwd['reduction_pct']}% overall, "
            f"-{fwd['da_reduction_pct']}% on the difficult-advice rows); supervised tokens "
            f"{sup['control']:,} -> {sup['empty_cot']:,} (-{sup['reduction_pct']}%); "
            f"difficult-advice share of the training signal {sup['da_share_control_pct']}% "
            f"-> {sup['da_share_arm_pct']}%"),
        "caveat": (
            "This mixture carries ZERO real reasoning traces, which trips "
            "check_thinking_declaration's gotcha-2 guard; the train config waives it "
            "explicitly with `allow_no_reasoning: true` and the adapter records the "
            "waiver. The guard's actual failure -- training the model to EMIT an empty "
            "close -- cannot occur here, because the marker is masked whole and never "
            "earns loss. KNOWN MISMATCH: the arm is served and evaluated in THINKING "
            "mode for comparability with its siblings, so at inference the model "
            "generates a real trace it was never trained to condition on."),
    }
    url = push_files([run_p / DATA_FILE,
                      run_p / "mixture_stats_emptycot.json",
                      run_p / "run_meta.json"], REPO, fields, private=private)
    card = (card_front_matter([{"config_name": "default",
                                "data_files": DATA_FILE, "default": True}])
            + card_markdown(fields))
    hf_api().upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                         repo_id=hf_repo_id(REPO), repo_type="dataset")
    print(url)


if __name__ == "__main__":
    fire.Fire(main)
