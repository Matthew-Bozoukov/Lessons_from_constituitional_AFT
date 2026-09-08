# ABOUTME: Publish the C6-masked training mixture to Hugging Face with the required card.
# ABOUTME: Run: uv run python scratch/push_c6_masked_dataset.py

"""Push the C6 meta-reasoning ablation mixture.

The repo holds ONE data file — the 10,000-row mixture whose `text` is byte-identical to the
unmasked control and whose only difference is the per-row `mask_spans` column. The sidecar
stats and the per-row span record go up alongside it so the ablation is auditable without
rerunning the judge.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.infra.huggingface import card_front_matter, card_markdown, hf_api, push_files  # noqa: E402
from src.utils import git_sha  # noqa: E402

REPO = "matboz/2026-08-16-c6-meta-reasoning-masked-t2-9284-synthdoc-716"
RUN = Path("output/cluster_masking/c6_full_20260816")


def main(private: bool = True) -> None:
    """Upload the masked mixture, its stats and its span record.

    Args:
        private: Create the repo private (default).
    """
    stats = json.loads((RUN / "mixture_stats_masked.json").read_text())
    meta = json.loads((RUN / "run_meta.json").read_text())
    fields = {
        "title": "C6 meta-reasoning masked mixture (Table2 9,284 + difficult-advice 716)",
        "experiment": "Ablation arm: loss-mask the reasoning spans expressing feature "
                      "cluster C6 (explicit meta-reasoning about response strategy) on the "
                      "122 difficult-advice rows that carry it, leaving all text unchanged.",
        "date_generated": "2026-08-16",
        "constitution": "claude_distilled_07_principles_approved "
                        "(constitutions/claude_distilled_07_principles_approved/constitution.md)",
        "source_repo": f"Matthew-Bozoukov/teaching_claude_why_replication @ {git_sha()}",
        "models": f"span selector {meta['judge']}; token stream Qwen/Qwen3.6-27B; "
                  "clusters embedded with Qwen/Qwen3-Embedding-8B",
        "generation_config": "span selection temperature 0.0, max_tokens 2000, reasoning "
                             "disabled; masking is deterministic given the spans",
        "schema": "text (rendered Qwen3.6 chat, IDENTICAL to the control mixture); source "
                  "(mixture source name); mask_spans (list of [start,end] CHARACTER offsets "
                  "into text whose tokens get labels=-100, empty on 9,878 rows); "
                  "mask_property + scenario_id present only on masked rows",
        "provenance": "uv run python scratch/mask_cluster_spans.py --prop meta_reasoning "
                      "--membership output/mixture_cluster_membership/20260816_150133/"
                      "membership.jsonl --limit=None --emit_mixture; verified by "
                      "scratch/verify_masked_mixture.py",
        "control_dataset": "LASR-Callum/2026-08-06-table2-9284-synthdoc-716-train "
                           "(mixture_think.jsonl) — same text, no mask_spans",
        "rows": f"{stats['rows']} ({stats['rows_with_mask_spans']} carry mask_spans, "
                f"{stats['spans_total']} spans)",
        "masked_tokens": f"{stats['masked_tokens_at_8192']:,} at max_seq_len 8192 = 0.77% "
                         "of all supervised tokens (2,993,995 -> 2,970,830)",
        "cluster": "C6 'Explicit meta-reasoning about response strategy', from the k=150 "
                   "feature-discovery run output/feature_discovery/20260812_092119",
        "usage": "src/train/train_lora.py consumes mask_spans via build_labels(mask_spans=...); "
                 "a trainer that ignores the column silently trains the control",
    }
    url = push_files([RUN / "mixture_think_masked.jsonl",
                      RUN / "mixture_stats_masked.json",
                      RUN / "masked_dataset.jsonl"], REPO, fields, private=private)
    # The card must declare the default config, or load_dataset globs the two schemas
    # (mixture rows and per-row span records) into one and fails.
    card = (card_front_matter([{"config_name": "default",
                                "data_files": "mixture_think_masked.jsonl",
                                "default": True}])
            + card_markdown(fields))
    hf_api().upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                         repo_id=REPO, repo_type="dataset")
    print(url)


if __name__ == "__main__":
    fire.Fire(main)
