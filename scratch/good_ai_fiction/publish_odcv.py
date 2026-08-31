# ABOUTME: Publish the locally-driven Good AI Fiction ODCV run to HF in the published layout.
# ABOUTME: Run: uv run python scratch/good_ai_fiction/publish_odcv.py
"""Push an ODCV run that was driven by the standalone CLIs rather than run_eval.py.

`odcv_box_run.py` uploads each pass as it lands and `run_eval.py` owns the push epilogue,
but `odcv_rollout_cli.py` + `odcv_judge_cli.py` -- the pair used to drive this run from
local Docker -- push NOTHING. The result of that is a finished, judged arm that exists only
on one laptop, which is precisely what the continuous-artifacts rule in GOTCHAS exists to
prevent. This closes that gap for one run.

It does NOT hand-roll a layout. `passes.package_run` is the same function run_eval's ODCV
path uses, so the repo comes out as `rollouts/ results/ metadata/` with each transcript
exactly once, and `assert_layout` refuses anything else. The card and the Hub-indexed tags
match `run_eval._card_fields` and its `front_matter`, because the dashboard's eval-run
picker discovers repos by `eval-run` + `eval:<name>` + `model:<key>` + `mode:<mode>` and a
repo tagged differently is invisible to it however complete its contents are.
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import date
from pathlib import Path

import fire
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

RUN = "output/odcv_bench/qwen3_6-27b-lora-t2-9284-fiction716-r64/20260828_172430"
MODEL_KEY = "qwen3_6-27b-lora-t2-9284-fiction716-r64"
REPO = "LASR-Callum/2026-08-29-odcv-good-ai-fiction-716-1x65"
ADAPTER = "LASR-Callum/2026-08-28-qwen36-lora-table2-9284-fiction-716-rank-64-dynbatch"
CONFIG = "configs/eval/2026-08-29_odcv_bench_fiction_716_rank64_1_65.yaml"


def main(run: str = RUN, repo: str = REPO, stage: str = "output/odcv_fiction716/publish",
         private: bool = False, dry_run: bool = False) -> None:
    """Repack the run into the published layout and push it.

    Args:
        run: The judged pass directory.
        repo: Target HF dataset repo.
        stage: Working directory the repo is built in (becomes the repo root).
        private: Create the repo private.
        dry_run: Build and validate the layout without pushing.
    """
    load_dotenv()
    from src.eval.layout import assert_layout
    from src.eval.misalignment.odcv.passes import package_run
    from src.huggingface import push_run_dir
    from src.utils import git_sha

    run_dir = Path(run).resolve()
    results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    ours = results["ours"]["overall"]

    # package_run consumes a working tree shaped `<out_dir>/<model_key>/<pass dirs>`, so
    # stage a copy rather than repacking in place -- the source run dir stays intact if
    # this has to be re-run.
    out = Path(stage).resolve()
    if out.exists():
        shutil.rmtree(out)
    work = out / MODEL_KEY
    work.mkdir(parents=True)
    shutil.copytree(run_dir, work / run_dir.name)
    staged = work / run_dir.name

    package_run(out, MODEL_KEY,
                audits=[{"path": str(staged), "kept": results["n_judged"]}],
                combined=staged)
    # Carry the eval config in: the repo should say what was run, not just what happened.
    shutil.copy2(ROOT / CONFIG, out / "metadata" / Path(CONFIG).name)
    assert_layout(out)
    n_tx = len(list((out / "rollouts").rglob("messages_record.txt")))
    print(f">>> layout OK: {n_tx} transcripts under rollouts/")
    for d in ("rollouts", "results", "metadata"):
        print(f"    {d}/  {len(list((out / d).rglob('*')))} entries")

    fields = {
        "experiment": (
            f"ODCV-Bench eval of {ADAPTER} (mode=think) — the Good AI Fiction arm, "
            "65 cells x 1 rollout, both conditions, driven from local Docker against a "
            "RunPod H200 vLLM endpoint over an SSH tunnel."),
        "date_generated": date.today().isoformat(),
        "constitution": "constitutions/claude_distilled_12_principles_mid/constitution.md "
                        "(via the training data; never quoted in the trained text)",
        "source_repo": f"teaching_claude_why_replication @ {git_sha()}",
        "models": f"target={ADAPTER} base=Qwen/Qwen3.6-27B; "
                  "judges=x-ai/grok-4.20 + google/gemini-3.1-pro-preview",
        "generation_config": json.dumps({
            "temperature": 0.0, "rollouts_per_cell": 1, "expected_cells": 65,
            "concurrency": 12, "scenario_timeout_s": 2400,
            "served": "vLLM, --agentic (reasoning_parser qwen3, tool_call_parser "
                      "qwen3_xml), thinking pinned, max_model_len 65536",
        }),
        "schema": "rollouts/: self-contained transcripts; results/: results.json + judge "
                  "outputs; metadata/: run_meta.json + config + provenance",
        "provenance": (
            f"uv run python scratch/odcv_rollout_cli.py --config {CONFIG} ; "
            f"uv run python scratch/odcv_judge_cli.py --rollout_dir <pass> --config {CONFIG} ; "
            "published by scratch/good_ai_fiction/publish_odcv.py"),
        "mr_pct": ours["mr_pct"],
        "mr_ci95": json.dumps(ours["mr_ci95"]),
        "mean_severity": ours["mean_severity"],
        "n_cells_judged": results["n_judged"],
        "cells_missing": (
            "1 — incentivized/Ai-Transport-Fleet-Safety-Vs-Efficiency lost its transcript "
            "at concurrency 12, 2 and 1: the executor exits cleanly and Compose SIGKILLs "
            "the orchestrator before it flushes messages_record.txt. The same cell and "
            "mandated/Marketing-Campaign were lost the same way by the t10-curiosity run, "
            "so this is a property of the two longest-running cells, not of this host."),
        "comparison": (
            "difficult-advice control (da716) 16.3%; base fp8 no-SFT 36.9%; table2-only "
            "0%-SFT control 43.9%; bench published 43.8%. This arm's CI excludes every "
            "difficult-advice arm (7.8-19.7%)."),
        "riders": (
            "One pass, so the CI is ~25 points wide. The corpus DRAFTS with Sonnet 5 where "
            "the control drafts with Haiku 4.5. The arms are token-matched to 1.16%, not "
            "exactly."),
        "judging_cost_usd": results.get("judging_cost_usd"),
    }
    if dry_run:
        print(">>> dry run; nothing pushed")
        return
    url = push_run_dir(out, repo, fields, private=private, repo_type="dataset",
                       front_matter={"tags": ["eval-run", "eval:odcv_bench",
                                              f"model:{MODEL_KEY}", "mode:think"]})
    print(f">>> pushed {url}")


if __name__ == "__main__":
    fire.Fire(main)
