# ABOUTME: Publish a GPT seed replicate's judged ODCV run to HF in the published-layout contract
# ABOUTME: (rollouts/ results/ metadata/ + tagged card), staged from a COPY so the local run dir survives.
# Run: uv run python scratch/gpt_seeds/push_odcv.py --seed 42
"""Unlike scratch/channel_swap/push_odcv.py (which uploaded the raw combined dir), this repacks the
run with ODCV's own `package_run`, so the repo has the same shape run_eval.py would have published
-- the dashboard's eval-run picker and `scratch/par_b/plot_*` loaders read `results/results.json`.
The repack CONSUMES its input, so it runs on a copy under output/odcv_publish/; the working dir
under output/odcv_bench/ (which scratch/stats/odcv_seed_sem.py reads) is untouched.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import fire
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

from src.eval.misalignment.odcv.passes import package_run  # noqa: E402
from src.huggingface import push_run_dir  # noqa: E402
from src.utils import git_sha, origin_url  # noqa: E402

SEED0_EVAL = "LASR-Callum/2026-08-25-odcv-gpt-responder-685-paired-eval"


def _spec(seed: int) -> dict:
    assert seed in (42, 69), seed
    key = f"qwen3_6-27b-lora-t2-9284-gptresp685-paired-r64-seed{seed}"
    return {
        "key": key,
        "repo": f"LASR-Callum/2026-08-28-odcv-gptresp685-seed{seed}-paired-eval",
        "adapter": f"LASR-Callum/2026-08-25-qwen36-lora-table2-9284-gpt-responder-685-paired-rank-64-seed{seed}",
        "cfg": f"configs/eval/odcv_bench_t2_9284_gptresp685_s{seed}_r64_paired_2x65.yaml",
        "work": ROOT / "output/odcv_bench" / key,
        "stage": ROOT / "output/odcv_publish" / key,
    }


def main(seed: int, combined: str = "") -> None:
    """Stage, repack and push one seed's run.

    Args:
        seed: 42 | 69.
        combined: The judged combined<N>x_<ts> dir under the seed's work dir (default: newest).
    """
    spec = _spec(seed)
    work = spec["work"]
    passes = sorted(
        p for p in work.glob("2026*") if (p / "rollout_manifest.json").is_file()
    )
    comb = (
        (ROOT / combined)
        if combined
        else max(
            (p for p in work.glob("combined*") if (p / "results.json").is_file()),
            key=lambda p: p.stat().st_mtime,
        )
    )
    assert passes and (comb / "results.json").is_file(), (passes, comb)
    res = json.loads((comb / "results.json").read_text(encoding="utf-8"))
    o = res["ours"]["overall"]
    # Older runs (seed 0, 2026-08-25) carry "n"; the current summarise() writes
    # n_rollouts/n_scenarios instead. Accept either so one script publishes both vintages.
    n_roll = o.get("n_rollouts", o.get("n"))

    stage = spec["stage"]
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    staged_passes = []
    for p in passes:
        dst = stage / spec["key"] / p.name
        shutil.copytree(p, dst)
        staged_passes.append({"path": str(dst), "kept": True})
    staged_comb = stage / spec["key"] / comb.name
    shutil.copytree(comb, staged_comb)
    shutil.copy2(ROOT / spec["cfg"], stage / "odcv_config.yaml")
    # package_run ends by rmtree-ing out_dir/<model_key> itself (a verbatim upload must not
    # carry the raw+combined working tree), so removing it again here raises
    # FileNotFoundError AFTER a successful repack and aborts the push. Do not re-add it.
    package_run(stage, spec["key"], staged_passes, staged_comb)
    assert not (stage / spec["key"]).exists(), (
        "package_run left the working tree behind"
    )
    (stage / "results" / "results.md").write_text(
        f"# ODCV-Bench, {spec['key']} (2 rollouts x 65 cells)\n\n"
        f"MR {o['mr_pct']}% {o['mr_ci95']}, severity {o['mean_severity']}, n={n_roll}; "
        f"mandated {res['ours']['mandated']['mr_pct']}%, incentivized "
        f"{res['ours']['incentivized']['mr_pct']}%. Judges {json.dumps(res['judges'])}.\n",
        encoding="utf-8",
    )

    url = push_run_dir(
        stage,
        spec["repo"],
        {
            "title": f"ODCV-Bench: GPT-responder paired arm, seed {seed} replicate, 2 rollouts x 65 cells",
            "experiment": (
                f"ODCV-Bench rollouts and judge scores for {spec['adapter']}: the seed-{seed} REPLICATE "
                f"of the GPT-responder paired arm (seed 0: {SEED0_EVAL}, MR 25.2% [15.1, 34.9]). Same "
                "65 cells, 15 exclusions, judges and protocol as every sibling arm, so the three GPT "
                "seeds give the between-seed error on the arm's number. Headline on these cells: "
                f"MR {o['mr_pct']}% {o['mr_ci95']}, severity {o['mean_severity']}, n={n_roll}."
            ),
            "date_generated": "2026-08-28",
            "constitution": (
                "constitutions/claude_distilled_12_principles_mid/constitution.md, via the adapter's "
                "training data (LASR-Callum/2026-08-28-gpt-responder-685-seeds-bundle)"
            ),
            "source_repo": f"{origin_url()} @ {git_sha()} (branch worktree-gpt-seeds)",
            "models": (
                f"target: {spec['adapter']} (thinking mode pinned, vLLM, agentic parsers, "
                f"max_model_len 16384); judges: {json.dumps(res['judges'])}"
            ),
            "generation_config": json.dumps(
                {
                    "temperature": 0.0,
                    "rollouts_per_cell": 2,
                    "cells": 65,
                    "config": spec["cfg"],
                    "concurrency": 12,
                    "driver": "laptop Docker Desktop against a RunPod H200 vLLM endpoint over RunPod HTTPS "
                    "proxy; both seed adapters served as LoRA modules on the same pod",
                }
            ),
            "schema": (
                "rollouts/<variant>/<Scenario>/pass<N>/messages_record.txt (the rollout) + cell_meta.json; "
                "results/results.json (ours vs the base-fp8 reference; per_scenario_medians keyed "
                "Scenario/rollout_NNN), results/scores_<judge>.json, results/results.md; "
                "metadata/ (combine manifest, run meta, odcv_config.yaml, per-pass manifests)."
            ),
            "provenance": (
                f"bash scratch/gpt_seeds/run_odcv.sh <serve pod> (= scratch/odcv_repeat_rollouts.sh "
                f"{spec['cfg']} 2; scratch/odcv_combine_passes.py; scratch/odcv_judge_cli.py); "
                f"scratch/gpt_seeds/push_odcv.py --seed {seed}"
            ),
        },
        private=False,
        repo_type="dataset",
        front_matter={
            "tags": [
                "eval-run",
                "eval:odcv",
                f"model:{spec['key']}",
                "mode:think",
                "seed-replicate",
            ]
        },
    )
    print(f"pushed {comb.name} -> {url}")


if __name__ == "__main__":
    fire.Fire(main)
