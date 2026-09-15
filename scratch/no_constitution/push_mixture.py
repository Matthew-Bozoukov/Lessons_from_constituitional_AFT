# ABOUTME: Publish the no-constitution paired mixture: the principle-scoped baseline's 9,284
# ABOUTME: table-2 rows byte for byte plus 702 `da-no-const` rows drawn by seed. Name minted by the law.
#
# Run: uv run python scratch/no_constitution/push_mixture.py [--private]

from __future__ import annotations

import json
from pathlib import Path

import fire
from dotenv import load_dotenv

from src.infra.huggingface import hf_repo_id, push_files, training_data_tags
from src.naming import mix_name
from src.utils import git_sha, origin_url

PATH = Path("data/t2_9284_da_no_const_702.jsonl")
STYLE, PCT = "da-no-const", 7  # 702 of 9,986 rows = 7.03%


def main(private: bool = False, date: str = "2026-09-03") -> None:
    load_dotenv()
    assert PATH.exists(), f"missing {PATH} -- build it first (see provenance below)"
    stats = json.loads(Path(str(PATH) + ".stats.json").read_text(encoding="utf-8"))
    repo = hf_repo_id(mix_name(STYLE, PCT, date=date))
    fields = {
        "experiment": (
            "No-constitution difficult-advice arm, paired with the principle-scoped 702 "
            "baseline: the SAME 9,284 filtered Table2 rows (byte-identical, taken from the "
            "baseline mixture itself) plus 702 difficult-advice rows generated with no "
            "constitution anywhere -- the only guidance any generation stage saw was the "
            "two words 'Be good.' A downstream difference from the baseline arm is "
            "attributable to what the generator was shown."
        ),
        "date_generated": date,
        "constitution": "none",
        "source_repo": f"{origin_url()} @ {git_sha()}",
        "models": (
            "synth half: anthropic/claude-haiku-4.5 (scenarios, draft prompt, draft "
            "reply) + anthropic/claude-sonnet-5 (prompt refine, reply rewrite), via "
            "LASR-Callum/2026-09-03-da-no-const-synth; Table2 half: pre-rendered, "
            "no model."
        ),
        "generation_config": (
            "Mixture build deterministic (seed 0): 702 synth rows drawn domain-spread from "
            "the corpus's 716 (one unit, `trait_id: guideline`, so no trait balancing); "
            "synth-half sampling settings in the corpus card."
        ),
        "schema": (
            "JSONL. `source` -- difficult_advice_no_const | a Table2 source; `text` -- "
            "the fully rendered Qwen chat string (<|im_start|>{role}\\n{content}"
            "<|im_end|>\\n per turn, assistant turns carrying <think>...</think>); "
            "`scenario_id` and `trait_id` on synth rows only. 9,986 rows = 702 synth + "
            "9,284 Table2 (7.03% synth)."
        ),
        "provenance": (
            "uv run python scratch/build_t2_9284_da716_mixture.py "
            "--out data/t2_9284_da_no_const_702.jsonl --seed 0 "
            "--synth_repo LASR-Callum/2026-09-03-da-no-const-synth --synth_file dataset.jsonl "
            "--synth_label difficult_advice_no_const --n_synth 702 "
            "--t2_repo LASR-Callum/2026-08-21-table2-9284-difficult-advice-principle-scoped-702-train-mixture "
            "--t2_file t2_9284_da_chunk_only_702.jsonl "
            "--exclude_sources '[difficult_advice_chunk_only]'; then this script."
        ),
        "notes": (
            f"stats: {json.dumps(stats)[:600]}. Compare against "
            "LASR-Callum/2026-08-21-table2-9284-difficult-advice-principle-scoped-702-train-mixture "
            "(same Table2 half, 702 principle-scoped rows)."
        ),
    }
    front = {
        "configs": [
            {
                "config_name": "default",
                "data_files": [{"split": "train", "path": PATH.name}],
            }
        ],
        "tags": training_data_tags("mixture", STYLE, "none", extra=["stage:final"]),
    }
    url = push_files(
        [PATH, Path(str(PATH) + ".stats.json")],
        repo,
        fields,
        private=private,
        front_matter=front,
    )
    print(f"{PATH.name} -> {url}")


if __name__ == "__main__":
    fire.Fire(main)
