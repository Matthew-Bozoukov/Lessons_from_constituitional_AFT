# ABOUTME: Publish both verbose-CoT ODCV arms (transcripts + judge scores + corrected
# ABOUTME: results.json) to one dated HF eval repo each, in the published-layout contract.
# Run: uv run python scratch/verbose_cot/push_odcv.py [--arm rows|tokens] [--dry_run True]

"""The boxes that produced these rollouts were credential-free on purpose, so
`odcv_box_run`'s push-per-pass could not authenticate and recorded a `publish_error`
instead. Publishing therefore happens here, from the machine that HAS the credentials,
against the transcripts `pull_artifacts.sh` brought home.

Two things this does NOT do the way the trait-10 sibling did:

  * It repacks through `package_run` rather than uploading the working tree verbatim, so
    the repo satisfies the rollouts/results/metadata contract in src/eval/layout.py and
    each transcript lands exactly once instead of raw-plus-combined twice.
  * It stages a COPY first. `package_run` consumes the combined dir it is handed, and the
    judge-score files it moves are the cache that makes re-judging free; gutting the only
    copy to publish it would trade a $0.06 rerun for an unrecoverable one.

`audits` is reconstructed from combine_manifest.json rather than read from pass_audit.json
(the boxes never wrote one): the manifest names the passes in execution order and counts
what each contributed, and a pass contributing cells is exactly what `kept` means here.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import fire
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# src/huggingface.py reads HF_TOKEN from the environment but does not load .env itself,
# so a standalone driver has to; without this every call 401s with "Invalid username or
# password", which reads as a bad token rather than an unloaded one.
load_dotenv(ROOT / ".env")

from src.eval.misalignment.odcv.passes import package_run  # noqa: E402
from src.huggingface import push_run_dir  # noqa: E402
from src.utils import git_sha, origin_url  # noqa: E402

STAGE = Path(tempfile.gettempdir()) / "odcv_publish_verbose"
RUN_ROOT = ROOT / "output" / "odcv_verbose" / "root"
DATE = "2026-08-25"

ARMS = {
    "rows": {
        "model_key": "qwen3_6-27b-lora-da716-verbose-rows-r64",
        "repo": f"LASR-Callum/{DATE}-odcv-da716-verbose-rows-eval",
        "adapter": "LASR-Callum/2026-08-20-qwen36-lora-table2-9284-difficult-advice-716-verbose-rank-64-dynbatch",
        "data": ("LASR-Callum/2026-08-25-table2-9284-difficult-advice-verbose-716-train",
                 "t2_9284_da716_verbose_10k.jsonl",
                 "4b7c08ab24eea91903857be57b0eb07ae9339f61"),
        "cfg": "configs/eval/2026-08-25_odcv_bench_difficult_advice_716_verbose_rows_rank64_incentivized_5_30.yaml",
        "arm_note": (
            "ROW-matched: difficult advice held at 7.16% of rows exactly as in the da716 "
            "baseline, so expanding its reasoning ~3x lets its share of trainable tokens "
            "rise to 47.6%. Isolates 'same rows, more reasoning per row'."),
    },
    "tokens": {
        "model_key": "qwen3_6-27b-lora-da-verbose-tokenmatched-r64",
        "repo": f"LASR-Callum/{DATE}-odcv-da-verbose-tokenmatched-eval",
        "adapter": "LASR-Callum/2026-08-20-qwen36-lora-table2-9284-difficult-advice-verbose-token-matched-rank-64-dynbatch",
        "data": ("LASR-Callum/2026-08-25-table2-9284-difficult-advice-verbose-token-matched-train-mixture",
                 "t2_9284_da_verbose_tokenmatched.jsonl",
                 "e71102af09dba065de65d086137edb3a4ff7fc9b"),
        "cfg": "configs/eval/2026-08-25_odcv_bench_difficult_advice_716_verbose_tokens_rank64_incentivized_5_30.yaml",
        "arm_note": (
            "TOKEN-matched: difficult advice's share of TRAINABLE TOKENS is held at the "
            "da716 baseline instead, which costs rows - roughly half the 716 expanded "
            "traces fit the same budget. Isolates 'same reasoning budget, spread over "
            "fewer, longer examples'."),
    },
}


def _stage(model_key: str) -> tuple[Path, Path, list[dict]]:
    """Copy one arm's passes + combined dir into a private staging tree.

    Returns (out_dir, combined, audits) shaped the way `package_run` wants them: out_dir
    is the would-be repo root, with the working tree namespaced under <model_key>/.
    """
    src = RUN_ROOT / model_key
    out_dir = STAGE / model_key
    if out_dir.exists():
        shutil.rmtree(out_dir)
    work = out_dir / model_key
    work.mkdir(parents=True)

    combined_src = sorted(src.glob("combined3x_*"))[-1]
    manifest = json.loads(
        (combined_src / "combine_manifest.json").read_text(encoding="utf-8"))

    for name in [*manifest["passes"], combined_src.name]:
        shutil.copytree(src / name, work / name)

    audits = [{"path": str(work / p), "kept": True} for p in manifest["passes"]]
    return out_dir, work / combined_src.name, audits


def _card(arm: str, spec: dict, res: dict, manifest: dict) -> dict:
    o = res["ours"]["overall"]
    repo, data_file, rev = spec["data"]
    short = {"rows": "row-matched", "tokens": "token-matched"}[arm]
    return {
        "title": f"ODCV-Bench: verbose-CoT {short} arm, 3 passes x 30 incentivized cells",
        "experiment": (
            "Does MORE VERBOSE chain-of-thought, holding the IDEAS constant, change "
            "agentic misalignment? The 716 difficult-advice reasoning traces were expanded "
            "~3x in length by Sonnet 5 under a two-judge fidelity gate (no new kinds of "
            "deliberation, no new cases, nothing dropped), then trained as a LoRA. "
            f"{spec['arm_note']} Headline on these 30 incentivized cells: MR "
            f"{o['mr_pct']}% CI95 {o['mr_ci95']} (bootstrapped over SCENARIOS), severity "
            f"{o['mean_severity']}, {o['n_scenarios']} scenarios / {o['n_rollouts']} "
            "rollouts. Published base Qwen3.6-27B on the same cells: 42.5%."),
        "date_generated": DATE,
        "constitution": (
            "constitutions/claude_distilled_12_principles_mid/constitution.md in the source "
            "repo - inherited unchanged from the difficult-advice run the traces were "
            f"expanded from, and carried into training data {repo} ({data_file} @ {rev})"),
        "source_repo": f"{origin_url()} @ {git_sha()}",
        "models": (
            f"target: {spec['adapter']} (LoRA r64 on Qwen/Qwen3.6-27B, thinking mode, "
            "served by vLLM on a RunPod H200, max_num_seqs 32); "
            f"judges: {json.dumps(res['judges'])}; "
            "trace expander: anthropic/claude-sonnet-5 via OpenRouter"),
        "generation_config": json.dumps({
            "temperature": 0.0, "passes": len(manifest["passes"]),
            "cells": 30, "variant": "incentivized only",
            "n_rollouts_actual": o["n_rollouts"], "config": spec["cfg"],
            "concurrency": 12,
            "driver": "rented vast CPU boxes, docker per scenario, reaching the model at "
                      "host.docker.internal:8000 through an SSH tunnel to the serving pod",
        }),
        "schema": (
            "rollouts/<variant>/<Scenario>/pass<N>/: messages_record.txt is THE rollout "
            "(task + reasoning + actions, self-contained), beside docker_output.log "
            "(container stdout, NOT the rollout) and cell_meta.json (manifest row, "
            "transcript_bytes, and whether this exact transcript was judged). "
            "results/: results.json (headline + per_scenario_medians, one LIST per "
            "scenario holding its per-rollout severity), scores_<judge>.json, "
            "judging_run_meta.json. metadata/: combine_manifest.json (which passes "
            "merged, what each contributed), per-pass manifests and run_meta."),
        "provenance": (
            "bash scratch/verbose_cot/bootstrap_boxes.sh prep|authorize|tunnel; "
            "scratch/odcv_box_run.py (3 passes, one per invocation); "
            f"scratch/odcv_combine_passes.py --config {spec['cfg']}; "
            "scratch/odcv_judge_cli.py --rollout_dir <combined> --config <cfg>; "
            "scratch/verbose_cot/push_odcv.py"),
        "ci_note": (
            "CIs bootstrap over SCENARIOS, not rollouts. Repeated rollouts of one scenario "
            "share a prompt, a model and temperature 0, so resampling them is "
            "pseudo-replication and reports an interval that is too narrow - this run "
            "measured [16.9, 34.8] that way against a correctly clustered "
            f"{o['mr_ci95']}. A scenario contributes its violation RATE across rollouts "
            "(0, 1/3, 2/3, 1) rather than a thresholded verdict, and every scenario weighs "
            "the same however many rollouts survived for it. Fixed in "
            "src/eval/misalignment/odcv/ on 2026-08-25; the numbers here are post-fix."),
        "coverage_note": (
            f"{o['n_rollouts']} rollouts rather than 90: "
            f"{json.dumps(manifest.get('cells_short', {}))} came up short where a cell "
            "produced no transcript in a pass (skipped_empty in combine_manifest.json). "
            "Those scenarios are still scored, on the rollouts they did produce, and "
            "weigh the same as any other scenario."),
    }


def main(arm: str | None = None, dry_run: bool = False) -> None:
    for name in ([arm] if arm else list(ARMS)):
        spec = ARMS[name]
        out_dir, combined, audits = _stage(spec["model_key"])
        manifest = json.loads(
            (combined / "combine_manifest.json").read_text(encoding="utf-8"))
        res = json.loads((combined / "results.json").read_text(encoding="utf-8"))
        assert res["ours"]["overall"].get("ci_unit") == "scenario", (
            f"{name}: results.json predates the CI fix - re-run odcv_judge_cli first")

        # package_run consumes the working tree itself; out_dir is the repo root after it.
        package_run(out_dir, spec["model_key"], audits, combined)
        n = len(list((out_dir / "rollouts").rglob("messages_record.txt")))
        print(f"[{name}] packed {n} transcripts -> {out_dir}")

        if dry_run:
            print(f"[{name}] dry run, not pushing to {spec['repo']}")
            continue
        url = push_run_dir(
            out_dir, spec["repo"], _card(name, spec, res, manifest),
            front_matter={"tags": ["eval-run", "eval:odcv", f"model:{spec['model_key']}",
                                   "mode:thinking", "verbose-cot"]})
        print(f"[{name}] pushed {url}")


if __name__ == "__main__":
    fire.Fire(main)
