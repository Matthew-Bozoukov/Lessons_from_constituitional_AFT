# ABOUTME: Publish a built t2_9284 + post-action-retrospection-716 mixture (jsonl + stats
# ABOUTME: sidecar) to HF with the required card, and print the commit sha the train config pins.
# Run: uv run python scratch/par_b/push_mixture.py [--arm bare_refusal|varied_shortfalls]

import json
import sys
from pathlib import Path

import fire

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.infra.huggingface import hf_api, push_files  # noqa: E402
from src.utils import git_sha, origin_url  # noqa: E402

TRAITS = {f"t{i}" for i in range(1, 10)}

# One entry per arm this script has published. The two differ in their SYNTHETIC half only
# -- same builder, same seed, same 9,284 Table-2 rows -- so everything that differs between
# them lives here and nothing else in the file is arm-specific.
ARMS = {
    "bare_refusal": {
        "repo": "LASR-Callum/2026-08-26-table2-9284-post-action-retrospection-716-train",
        "synth_repo": "LASR-Callum/2026-08-26-post-action-retrospection-716",
        "synth_file": "dataset.jsonl",
        "path": "data/t2_9284_par716_10k.jsonl",
        "date": "2026-08-26",
        "train_config": "configs/train/qwen36-table2-9284-par-716-dynbatch.yaml",
        "design": (
            "a difficult-advice prompt, a bare refusal, the person's pushback, and the "
            "trained turn doing the reasoning the refusal skipped"
        ),
        "shortfall_turn": "bare-refusal turn",
    },
    "varied_shortfalls": {
        "repo": "LASR-Callum/2026-09-04-table2-9284-par-varied-shortfalls-716-train",
        "synth_repo": "LASR-Callum/2026-09-03-par-synth",
        "synth_file": "par_781.jsonl",
        "path": "data/t2_9284_par716_10k.jsonl",
        "date": "2026-09-04",
        "train_config": (
            "configs/train/qwen36-table2-9284-par-716-varied-shortfalls-dynbatch.yaml"
        ),
        "design": (
            "a difficult-advice prompt, a first reply that falls short in a way the "
            "SCENARIO ITSELF specifies (free text, not one of a fixed set -- over-"
            "compliance, thin help, a missed detail, a flat refusal, blindness to the "
            "person's state, or whatever else the scenario calls for), the person pressing "
            "back in a way the scenario also specifies, and the trained turn diagnosing and "
            "repairing its own earlier reply. This is the rebuild of the bare-refusal arm "
            "(LASR-Callum/2026-08-26-table2-9284-post-action-retrospection-716-train), "
            "whose every first reply was a refusal, so its trained turn only ever repaired "
            "one failure"
        ),
        "shortfall_turn": "falling-short first turn",
    },
}


def main(
    arm: str = "varied_shortfalls",
    path: str = "",
    repo: str = "",
    private: bool = False,
) -> None:
    spec = ARMS[arm]
    repo = repo or spec["repo"]
    p = ROOT / (path or spec["path"])
    stats_p = Path(str(p) + ".stats.json")
    assert p.is_file() and stats_p.is_file(), f"build the mixture first: {p}"
    stats = json.loads(stats_p.read_text(encoding="utf-8"))
    assert stats["total"] == 10000 and stats["synth"] == 716, stats
    assert set(stats["per_trait"]) == TRAITS, stats["per_trait"]
    assert stats["supervise_final_rows"] == 716, stats["supervise_final_rows"]
    # The card must describe the file that is actually in hand, not the arm that was asked
    # for: pushing arm A's card over arm B's rows is the one mistake this script can make
    # invisibly, and its cost is a mislabelled dataset nobody re-reads.
    assert stats["synth_source"].startswith(spec["synth_repo"]), (
        f"--arm {arm} expects a mixture built from {spec['synth_repo']}, but "
        f"{stats_p.name} says {stats['synth_source']}. Rebuild, or pass the right --arm."
    )

    # The trainer's data_file is the basename; keep the sibling arms' name so the train
    # config differs from da716's in repo/revision only.
    staged = p.parent / "mixture_think.jsonl"
    staged.write_bytes(p.read_bytes())
    staged_stats = p.parent / "mixture_stats.json"
    staged_stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    url = push_files(
        [staged, staged_stats],
        repo,
        {
            "experiment": (
                f"Training mixture for the post-action-retrospection {arm} 716 arm: 9,284 "
                f"spec-filtered Table-2 instruction rows + 716 five-turn PAR rows -- "
                f"{spec['design']}. Trait-balanced with a capped water-fill (some principles "
                "have fewer than an even share after the grey-area rater) and spread "
                "round-robin across domain. Same 9,284 Table-2 rows and same builder as the "
                "da716 mixture "
                "(LASR-Callum/2026-08-14-table2-9284-difficult-advice-716-train)."
            ),
            "date_generated": spec["date"],
            "constitution": (
                "constitutions/claude_distilled_12_principles_mid/constitution.md (9 "
                "principles), the same as difficult advice's. The 716 rows come from "
                f"{spec['synth_repo']} ({spec['synth_file']}), which records its "
                "constitution sha in manifest.json."
            ),
            "source_repo": f"{origin_url()} @ {git_sha()}",
            "models": (
                "synth rows: anthropic/claude-haiku-4.5 (scenarios, draft prompts, "
                "reflection draft) + anthropic/claude-sonnet-5 (prompt refine, grey-area "
                "rater, scenario coherence pass, the first reply, its check, the pushback, "
                "the rewrite that trains); Table-2 rows: as published in "
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
                "mixture_think.jsonl: {source, text, trait_id?, scenario_id?, supervise?} "
                "-- `text` is the fully rendered Qwen chat transcript (<|im_start|>... "
                "with a <think> block on every assistant turn: a real trace on the trained "
                f"turn of the 716 synth rows, the empty marker on their "
                f"{spec['shortfall_turn']} and on Table-2 rows). `supervise: final` on the "
                "716 synth rows: only the last assistant turn is in the loss "
                "(src/train/masking.py). mixture_stats.json: the builder's counts."
            ),
            "provenance": (
                "uv run python scratch/build_t2_9284_da716_mixture.py "
                f"--synth_repo {spec['synth_repo']} --synth_file {spec['synth_file']} "
                "--synth_label post_action_retrospection "
                f"--out {path or spec['path']} --seed {stats['seed']}; then "
                f"uv run python scratch/par_b/push_mixture.py --arm {arm}"
            ),
        },
        private=private,
    )
    sha = hf_api().dataset_info(repo).sha
    print(
        f"pushed -> {url}\nrevision: {sha}\n"
        f'now set data_revision: "{sha}" in {spec["train_config"]}'
    )


if __name__ == "__main__":
    fire.Fire(main)
