# ABOUTME: Publish one seed replicate's ODCV run (rollouts + results) to HF with a card that
# ABOUTME: states what the number is and, more importantly, what it cannot be used for.

"""Upload a seed-replicate ODCV run.

    uv run python scratch/low_stakes/push_odcv_seed.py --seed 80085 \\
        --run_dir output/odcv_bench/<model_key>/combined1x_<ts>

The card exists to stop one specific misreading. Two seeds of THIS arm -- identical data,
identical code, identical hyperparameters -- came out 6.1 points apart on ODCV. Anyone who
finds a single MR number here and compares it to another arm's single number is reading
sampling noise, so the card says the seed spread before it says the point estimate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

from huggingface_hub import HfApi  # noqa: E402

ADAPTER = "LASR-Callum/2026-08-31-qwen36-lora-table2-9284-low-stakes-716-rank-64-dynbatch-seed-80085"
SEED0_ADAPTER = "LASR-Callum/2026-08-26-qwen36-lora-table2-9284-low-stakes-716-rank-64-dynbatch"
SEED0_EVAL = "LASR-Callum/2026-08-27-odcv-low-stakes-716-1-x65"
CORPUS = "LASR-Callum/2026-08-26-difficult-advice-low-stakes-716"
MIXTURE = "LASR-Callum/2026-08-26-table2-9284-low-stakes-716-train"
BUNDLE = "LASR-Callum/2026-08-31-low-stakes-716-seed-80085-bundle"
# Seed 0's published numbers, for the side-by-side the card leads with.
SEED0 = {"overall": (16.9, [7.7, 26.2], 0.66), "mandated": (14.3, 35, 0.66),
         "incentivized": (20.0, 30, 0.78)}


def main(seed: int = 80085, run_dir: str = "", repo: str = "", dry: bool = False) -> None:
    api = HfApi(token=__import__("os").environ["HF_TOKEN"])
    d = Path(run_dir or sorted(
        (ROOT / "output" / "odcv_bench").glob("*seed%d/combined1x_*" % seed))[-1])
    res = json.loads((d / "results.json").read_text(encoding="utf-8"))
    o, pub = res["ours"], res["published"]
    repo = repo or f"LASR-Callum/2026-08-31-odcv-lowstakes-716-seed{seed}-1x65"

    card = f"""---
tags:
- eval-results
- odcv-bench
- seed-replicate
---
# ODCV-Bench: low-stakes difficult advice, **seed {seed}** (1 pass, 65 cells)

**This is the SECOND seed of the same arm, trained on the SAME SFT dataset.** It is not a
different corpus, a different recipe or a different set of hyperparameters. Seed 0 is
[`{SEED0_ADAPTER}`]"""f"""(https://huggingface.co/{SEED0_ADAPTER}); this evaluates
[`{ADAPTER}`](https://huggingface.co/{ADAPTER}), which differs from it in one integer.

## Read this before quoting the MR

| | seed 0 | **seed {seed}** | difference |
| --- | --: | --: | --: |
| **overall MR** | {SEED0['overall'][0]}% CI95 {SEED0['overall'][1]} | **{o['overall']['mr_pct']}%** CI95 {o['overall']['mr_ci95']} | **{o['overall']['mr_pct'] - SEED0['overall'][0]:+.1f} pp** |
| mandated (n=35) | {SEED0['mandated'][0]}% | {o['mandated']['mr_pct']}% | {o['mandated']['mr_pct'] - SEED0['mandated'][0]:+.1f} pp |
| incentivized (n=30) | {SEED0['incentivized'][0]}% | {o['incentivized']['mr_pct']}% | {o['incentivized']['mr_pct'] - SEED0['incentivized'][0]:+.1f} pp |
| mean severity | {SEED0['overall'][2]} | {o['overall']['mean_severity']} | |
| bench published | {pub['overall']['mr_pct']}% | {pub['overall']['mr_pct']}% | |

**The two seeds are {abs(o['overall']['mr_pct'] - SEED0['overall'][0]):.1f} points apart on identical data, identical code and identical
hyperparameters.** Four of 65 scenarios flipped. Their CIs overlap almost entirely, so this
is NOT evidence the seeds differ -- it is a measurement of how much a single-pass ODCV
number moves on a re-roll.

That matters for every comparison drawn from runs this size: nine prior difficult-advice
manipulations sit inside an 8.7-17.6% band, which is **narrower than the gap between two
seeds of one arm**. A single-pass MR characterises a run. It does not separate arms.

What survives both seeds: this arm sits far below the bench's published {pub['overall']['mr_pct']}%.
That is inherited from the difficult-advice recipe, not attributable to low stakes.

## Fields

- **experiment**: ODCV-Bench on the seed-{seed} replicate of the low-stakes difficult-advice arm.
- **date_generated**: 2026-08-31.
- **constitution**: constitutions/claude_distilled_12_principles_mid/constitution.md, via
  [`{CORPUS}`](https://huggingface.co/datasets/{CORPUS}).
- **source_repo**: this repository, branch `nika/low-stakes-seed80085`.
- **models**: target [`{ADAPTER}`](https://huggingface.co/{ADAPTER}) served by vLLM on a
  rented RunPod H200, `thinking` mode pinned, agentic parsers on (required: without them the
  agent emits no tool calls and the harness still reports every scenario `ok` with no
  transcript). Judges: `x-ai/grok-4.20` and `google/gemini-3.1-pro-preview`.
- **generation_config**: temperature 0.0, 1 rollout per cell, 65 cells (30 incentivized +
  35 mandated, the 15 standard exclusions), concurrency 6.
- **schema**: `passes/` holds the rollout transcripts (`messages_record.txt` per scenario);
  `results.json` carries per-condition MR, severity and scenario-clustered CIs.
- **provenance**: rollouts `scratch/odcv_rollout_cli.py --config
  configs/eval/odcv_bench_lowstakes716_s{seed}_r64_1x65.yaml`, combined by
  `scratch/odcv_combine_passes.py`, judged by `scratch/odcv_judge_cli.py`.

## Training provenance

The adapter came from [`{BUNDLE}`](https://huggingface.co/datasets/{BUNDLE}) -- seed 0's own
`code.tar.gz` with a derived seed config appended, beside the byte-identical mixture from
[`{MIXTURE}`](https://huggingface.co/datasets/{MIXTURE}). The config is derived rather than
copied and the build refuses to publish if anything but `seed`, `output_dir`, `hf_repo`,
`data_repo` and `data_revision` differs from the parent.

## How it was run, and two things that bit

Driven from a local Docker Desktop with only the GPU rented. Containers reached the model at
`host.docker.internal:8000` through an SSH tunnel, NOT RunPod's HTTPS proxy, which times out
on ODCV's long non-streaming rollouts.

- **Concurrency 6, not 12.** Seed 0's run lost a cell to `compose_exit_1+no_container` at
  concurrency 12 on a machine with 49 GB; this host has ~18.8 GB. At 6 the run came back
  65/65 clean with no resume needed.
- **The judge crashed at 7/65.** `evaluate_all_results.py` indexed `res.choices[0]` without
  checking `choices`, so one provider error payload killed the whole batch (`map_threaded`
  is fail-fast). Patched to retry in the same idiom the file already uses, exiting to the
  `('N/A','N/A')` the caller counts as a drop. Re-apply if the harness is re-cloned.

Judged 65/65, **0 dropped**, judging ${res['judging_cost_usd']:.2f}.
"""
    if dry:
        print(card[:2200])
        print(f"\n--dry: would publish to {repo}")
        return
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    api.upload_folder(folder_path=str(d), path_in_repo="passes/a1_1x65",
                      repo_id=repo, repo_type="dataset")
    cp = ROOT / "output" / f"odcv_seed{seed}_README.md"
    cp.write_text(card, encoding="utf-8")
    api.upload_file(path_or_fileobj=str(cp), path_in_repo="README.md",
                    repo_id=repo, repo_type="dataset")
    api.upload_file(path_or_fileobj=str(d / "results.json"),
                    path_in_repo="results.json", repo_id=repo, repo_type="dataset")
    print(f">>> published https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    fire.Fire(main)
