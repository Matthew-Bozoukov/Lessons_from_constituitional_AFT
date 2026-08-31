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
    CONTROL_FILE, CONTROL_REPO, CONTROL_REVISION,
)
from src.huggingface import card_front_matter, card_markdown, hf_api, push_files  # noqa: E402
from src.utils import git_sha  # noqa: E402

REPO = "2026-08-31-cot-only-supervision-t2-9284-synthdoc-716"
DATA_FILE = "mixture_think_cotonly.jsonl"


def main(run: str, private: bool = True) -> None:
    """Upload the mixture, its stats and its card.

    Args:
        run: The build output directory (output/cot_only_mixture/<ts>).
        private: Create the repo private (default).
    """
    run_p = Path(run)
    stats = json.loads((run_p / "mixture_stats_cotonly.json").read_text())
    fwd, sup = stats["forward_tokens"], stats["supervised_tokens"]
    fields = {
        "title": "CoT-only supervision mixture (Table2 9,284 + difficult-advice 716)",
        "experiment": "Arm: train the 716 difficult-advice rows on their REASONING ONLY "
                      "— each row is truncated at its `</think>` close, so the visible "
                      "answer leaves both the loss and the forward pass — while the 9,284 "
                      "Table2 rows train exactly as in the control. Tests whether the "
                      "difficult-advice effect on agentic misalignment is carried by the "
                      "reasoning or by the answer.",
        "date_generated": "2026-08-31",
        "constitution": "claude_distilled_07_principles_approved "
                        "(constitutions/claude_distilled_07_principles_approved/constitution.md)",
        "source_repo": f"Matthew-Bozoukov/teaching_claude_why_replication @ {git_sha()}",
        "models": "token stream Qwen/Qwen3.6-27B (tokenizer + ModelProfile literals)",
        "generation_config": "none — no model is sampled here. The build is a "
                             "deterministic per-row field addition over a pinned control.",
        "schema": "text (rendered Qwen3.6 chat, IDENTICAL to the control mixture); "
                  "source (mixture source name); supervise ('cot' on the 716 "
                  "difficult-advice rows, absent elsewhere = 'all')",
        "provenance": "uv run python scratch/cot_only/build_mixture.py ; verified by "
                      "scratch/cot_only/verify_mixture.py (text byte-identical on all "
                      "10,000 rows, mask gate passed on both supervise modes with the "
                      "real Qwen3.6 tokenizer)",
        "control_dataset": f"{CONTROL_REPO} ({CONTROL_FILE} @ {CONTROL_REVISION[:12]}) "
                           "— same text, no supervise column",
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
                 "configs/train/2026-08-31_lora_qwen36_table2_9284_synthdoc_716_cotonly_dynbatch"
                 ".yaml; the control is the sibling config without `_cotonly`.",
        "caveat": "seq_mean_token_mean_loss weights each EXAMPLE at 1/global_batch, so "
                  "this arm concentrates the same per-example weight onto the trace rather "
                  "than reducing the rows' influence — roughly doubling the per-CoT-token "
                  "gradient weight. It is 'reasoning only, at double density', not 'the "
                  "control minus its answer term'.",
    }
    url = push_files([run_p / DATA_FILE,
                      run_p / "mixture_stats_cotonly.json",
                      run_p / "run_meta.json"], REPO, fields, private=private)
    card = (card_front_matter([{"config_name": "default",
                                "data_files": DATA_FILE,
                                "default": True}])
            + card_markdown(fields))
    hf_api().upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                         repo_id=REPO, repo_type="dataset")
    print(url)


if __name__ == "__main__":
    fire.Fire(main)
