# ABOUTME: Publish the t2_9284 + coherent-PAR-716 mixture (jsonl + stats sidecar) to HF with the
# ABOUTME: required card, and print the commit sha the train config pins. Fork of scratch/par_b/push_mixture.py.
# Run: uv run python scratch/par_coherence/push_mixture.py [--path data/t2_9284_par716coh_10k.jsonl]

import json
import sys
from pathlib import Path

import fire

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.huggingface import hf_api, push_files  # noqa: E402
from src.utils import git_sha, origin_url  # noqa: E402

REPO = "LASR-Callum/2026-08-28-table2-9284-par716coh-train"
SYNTH_REPO = "LASR-Callum/2026-08-28-post-action-retrospection-716-coherent"
PARENT_MIXTURE = "LASR-Callum/2026-08-26-table2-9284-post-action-retrospection-716-train"
TRAITS = {f"t{i}" for i in range(1, 10)}


def main(
    path: str = "data/t2_9284_par716coh_10k.jsonl",
    repo: str = REPO,
    private: bool = False,
) -> None:
    p = ROOT / path
    stats_p = Path(str(p) + ".stats.json")
    assert p.is_file() and stats_p.is_file(), f"build the mixture first: {p}"
    stats = json.loads(stats_p.read_text(encoding="utf-8"))
    assert stats["total"] == 10000 and stats["synth"] == 716, stats
    assert set(stats["per_trait"]) == TRAITS, stats["per_trait"]
    assert stats["supervise_final_rows"] == 716, stats["supervise_final_rows"]

    staged = p.parent / "mixture_think.jsonl"
    staged.write_bytes(p.read_bytes())
    staged_stats = p.parent / "mixture_stats.json"
    staged_stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    url = push_files(
        [staged, staged_stats],
        repo,
        {
            "experiment": (
                "Training mixture for the COHERENT post-action-retrospection 716 arm (arm 1 of the PAR "
                "coherence experiment): the same 9,284 spec-filtered Table-2 rows as every sibling arm + "
                "the SAME 716 five-turn PAR rows that trained the par716 arm, with only their trained turn "
                "rewritten so the reasoning ends on a first-person decision and the reply enacts it "
                f"({SYNTH_REPO}). Row-for-row paired with {PARENT_MIXTURE} @ 42c8a74: identical ids, "
                "identical trait quota, identical Table-2 half; the 716 texts are the only difference."
            ),
            "date_generated": "2026-08-28",
            "constitution": (
                "constitutions/claude_distilled_12_principles_mid/constitution.md (9 principles), the "
                f"same as difficult advice's; inherited from {SYNTH_REPO}"
            ),
            "source_repo": f"{origin_url()} @ {git_sha()}",
            "models": (
                "synth rows: parent PAR pipeline (Haiku 4.5 + Sonnet 5) then a Sonnet 5 coherence "
                "rewrite of turn 4; Table-2 rows: as published in "
                "LASR-Callum/2026-08-04-table2-instruction-tuning-9284-filtered-8192"
            ),
            "generation_config": json.dumps(
                {
                    "builder": "scratch/build_t2_9284_da716_mixture.py",
                    "seed": stats["seed"],
                    "synth_source": stats["synth_source"],
                    "t2_source": stats["t2_source"],
                    "synth_fraction": stats["synth_fraction"],
                    "per_trait": stats["per_trait"],
                    "distinct_domains_in_synth": stats["distinct_domains_in_synth"],
                    "supervise_final_rows": stats["supervise_final_rows"],
                }
            ),
            "schema": (
                "mixture_think.jsonl: {source, text (Qwen chat format, <think> on every assistant "
                "turn; synth rows carry a real trace on the LAST turn only), trait_id, scenario_id, "
                "supervise: final}; mixture_stats.json: the builder's stats sidecar"
            ),
            "provenance": (
                "uv run python scratch/build_t2_9284_da716_mixture.py --out data/t2_9284_par716coh_10k.jsonl "
                f"--synth_repo {SYNTH_REPO} --synth_file dataset.jsonl --synth_label post_action_retrospection "
                "--n_synth 716 --seed 0; then uv run python scratch/par_coherence/push_mixture.py"
            ),
            "parent_mixture": f"hf.co/datasets/{PARENT_MIXTURE}@42c8a74",
        },
        private=private,
        front_matter={
            "configs": [
                {"config_name": "default", "data_files": "mixture_think.jsonl"}
            ],
            "tags": ["training-mixture", "arm:par716coh"],
        },
    )
    sha = hf_api().dataset_info(repo).sha
    print(f"{url}\nrevision: {sha}   <- pin this as data_revision in the train config")


if __name__ == "__main__":
    fire.Fire(main)
