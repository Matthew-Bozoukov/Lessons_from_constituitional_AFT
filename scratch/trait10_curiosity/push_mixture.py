# ABOUTME: Publish the built t2_9284 + t10-curiosity-716 mixture (jsonl + stats sidecar) to HF
# ABOUTME: with the required card, and print the commit sha the train config must pin.
# Run: uv run python scratch/trait10_curiosity/push_mixture.py [--path data/t2_9284_t10_curiosity_716_10k.jsonl]

import json
import sys
from pathlib import Path

import fire

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.infra.huggingface import hf_api, push_files  # noqa: E402
from src.utils import git_sha, origin_url  # noqa: E402

REPO = "LASR-Callum/2026-08-20-table2-9284-t10-curiosity-716-train"
SYNTH_REPO = "LASR-Callum/2026-08-20-difficult-advice-t10-curiosity"


def main(path: str = "data/t2_9284_t10_curiosity_716_10k.jsonl", repo: str = REPO,
         private: bool = False) -> None:
    p = ROOT / path
    stats_p = Path(str(p) + ".stats.json")
    assert p.is_file() and stats_p.is_file(), f"build the mixture first: {p}"
    stats = json.loads(stats_p.read_text(encoding="utf-8"))
    assert stats["total"] == 10000 and stats["synth"] == 716, stats
    assert set(stats["per_trait"]) == {"t10"}, stats["per_trait"]

    # The trainer's data_file is the basename; keep the sibling arms' name so the train
    # config differs from da716's in repo/revision only.
    staged = p.parent / "mixture_think.jsonl"
    staged.write_bytes(p.read_bytes())
    staged_stats = p.parent / "mixture_stats.json"
    staged_stats.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    url = push_files([staged, staged_stats], repo, {
        "experiment": (
            "Training mixture for the trait-10 (curiosity) arm: 9,284 spec-filtered Table-2 "
            "instruction rows + 716 difficult-advice rows generated against ONE principle "
            "appended to the constitution (genuine intellectual curiosity), trait-balanced "
            "trivially (one trait) and spread round-robin across domain. Same 9,284 Table-2 "
            "rows and same builder as the da716 mixture "
            "(LASR-Callum/2026-08-14-table2-9284-difficult-advice-716-train)."),
        "date_generated": "2026-08-20",
        "constitution": (
            "scratch/trait10_curiosity/constitution.md in the source repo: "
            "claude_distilled_12_principles_mid (9 principles) with `## 10. Bring genuine "
            "intellectual curiosity and depth of engagement to ideas` appended. The 716 "
            f"rows come from {SYNTH_REPO} (dataset.jsonl), which records its constitution "
            "sha in manifest.json."),
        "source_repo": f"{origin_url()} @ {git_sha()}",
        "models": ("synth rows: google/gemini-3.7-flash (scenarios, draft prompts, draft "
                   "responses) + anthropic/claude-sonnet-5 (prompt and response rewrites); "
                   "Table-2 rows: as published in "
                   "LASR-Callum/2026-08-04-table2-instruction-tuning-9284-filtered-8192"),
        "generation_config": json.dumps({
            "builder": "scratch/build_t2_9284_da716_mixture.py", "seed": stats["seed"],
            "synth_source": stats["synth_source"], "t2_source": stats["t2_source"],
            "synth_fraction": stats["synth_fraction"], "per_trait": stats["per_trait"],
            "distinct_domains_in_synth": stats["distinct_domains_in_synth"],
        }),
        "schema": ("mixture_think.jsonl: {source, text, trait_id?, scenario_id?} -- `text` "
                   "is the fully rendered Qwen chat transcript (<|im_start|>... with a "
                   "<think> block on every assistant turn: real traces on the 716 synth "
                   "rows, the empty marker on Table-2 rows). mixture_stats.json: the "
                   "builder's counts."),
        "provenance": (
            "uv run python scratch/build_t2_9284_da716_mixture.py "
            f"--synth_repo {SYNTH_REPO} --synth_file dataset.jsonl "
            "--synth_label difficult_advice_t10_curiosity "
            f"--out {path} --seed {stats['seed']}; then "
            "uv run python scratch/trait10_curiosity/push_mixture.py"),
    }, private=private)
    sha = hf_api().dataset_info(repo).sha
    print(f"pushed -> {url}\nrevision: {sha}\n"
          f"now set data_revision: \"{sha}\" in "
          "scratch/trait10_curiosity/lora_qwen36_t2_9284_t10_curiosity_716_dynbatch_2xh200.yaml")


if __name__ == "__main__":
    fire.Fire(main)
