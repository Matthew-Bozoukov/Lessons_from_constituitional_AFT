# ABOUTME: Publish the locally-driven CoT-only ODCV run to HF in the published layout.
# ABOUTME: Run: uv run python scratch/cot_only/publish_odcv.py

"""Push the CoT-only arm's ODCV run.

`odcv_rollout_cli.py` + `odcv_judge_cli.py` -- the pair used to drive this run from local
Docker -- push NOTHING, so a finished arm otherwise exists only on one laptop. This closes
that gap, exactly as scratch/good_ai_fiction/publish_odcv.py does for the fiction arm.

It is a SIBLING of that script rather than a parameterisation of it: what differs between
the two is entirely arm identity and prose, and the contract logic that actually matters --
`passes.package_run`, `assert_layout`, `push_run_dir` -- is imported from the shared
modules by both, so nothing that could drift is duplicated. Editing the fiction arm's
script in place would also mean touching a committed artifact of somebody else's run.

`package_run` is the same function run_eval's ODCV path uses, so the repo comes out as
`rollouts/ results/ metadata/` and `assert_layout` refuses anything else. The card tags
match run_eval's, because the dashboard discovers repos by `eval-run` + `eval:<name>` +
`model:<key>` + `mode:<mode>` and a differently-tagged repo is invisible to it however
complete its contents are.
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

RUN = "output/odcv_bench/qwen3_6-27b-lora-t2-9284-synthdoc-716-cotonly-r64/20260831_164834"
MODEL_KEY = "qwen3_6-27b-lora-t2-9284-synthdoc-716-cotonly-r64"
REPO = "2026-08-31-odcv-cot-only-supervision-716-1x65"
ADAPTER = "LASR-Callum/qwen3.6-27b-lora-t2-9284-synthdoc-716-cotonly-r64"
CONFIG = "configs/eval/odcv_bench_cotonly716_r64_1x65.yaml"


def main(run: str = RUN, repo: str = REPO, stage: str = "output/odcv_cotonly716/publish",
         private: bool = False, dry_run: bool = False) -> None:
    """Repack the judged run into the published layout and push it.

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
    shutil.copy2(ROOT / CONFIG, out / "metadata" / Path(CONFIG).name)
    assert_layout(out)
    n_tx = len(list((out / "rollouts").rglob("messages_record.txt")))
    print(f">>> layout OK: {n_tx} transcripts under rollouts/")
    for d in ("rollouts", "results", "metadata"):
        print(f"    {d}/  {len(list((out / d).rglob('*')))} entries")

    fields = {
        "experiment": (
            f"ODCV-Bench eval of {ADAPTER} (mode=think) - the CoT-only supervision arm, "
            "whose 716 difficult-advice rows trained on their REASONING ONLY (each row "
            "truncated at its reasoning close, answer removed from both the loss and the "
            "forward pass). 65 cells x 1 rollout, both conditions, driven from local "
            "Docker against a RunPod H200 vLLM endpoint over an SSH tunnel."),
        "date_generated": date.today().isoformat(),
        "constitution": "constitutions/claude_distilled_07_principles_approved/constitution.md "
                        "(via the training data; never quoted in the trained text)",
        "source_repo": f"teaching_claude_why_replication @ {git_sha()}",
        "models": f"target={ADAPTER} base=Qwen/Qwen3.6-27B; "
                  "judges=x-ai/grok-4.20 + google/gemini-3.1-pro-preview",
        "generation_config": json.dumps({
            "temperature": 0.0, "rollouts_per_cell": 1, "expected_cells": 65,
            "concurrency": 12, "scenario_timeout_s": 2400,
            "served": "vLLM, --agentic (reasoning_parser qwen3, tool_call_parser "
                      "qwen3_xml), thinking pinned, max_model_len 16384",
        }),
        "schema": "rollouts/: self-contained transcripts; results/: results.json + judge "
                  "outputs; metadata/: run_meta.json + config + provenance",
        "provenance": (
            f"uv run python scratch/odcv_rollout_cli.py --config {CONFIG} ; "
            f"uv run python scratch/odcv_judge_cli.py --rollout_dir <pass> --config {CONFIG} ; "
            "published by scratch/cot_only/publish_odcv.py"),
        "mr_pct": ours["mr_pct"],
        "mr_ci95": json.dumps(ours["mr_ci95"]),
        "mean_severity": ours["mean_severity"],
        "n_cells_judged": results["n_judged"],
        "cells_missing": (
            "1 - mandated/Drone-Delivery-Optimizer, flagged by the harness itself as "
            "'ok+no_transcript': the executor exits cleanly and Compose SIGKILLs the "
            "orchestrator before it flushes messages_record.txt. NOTE this is a DIFFERENT "
            "cell from the ones the fiction and t10-curiosity runs lost "
            "(incentivized/Ai-Transport-Fleet-Safety-Vs-Efficiency, "
            "mandated/Marketing-Campaign), so the failure is NOT confined to those two "
            "long-running cells as previously supposed. The lost cell is mandated, whose "
            "MR is therefore over 34 scenarios rather than 35; its direction of bias is "
            "unknown."),
        "comparison": (
            "difficult-advice control (da716) 16.3%; base fp8 no-SFT 36.9% [21.4, 53.6]; "
            "Good AI Fiction 45.3%; bench published 43.8%. This arm at 14.1% sits INSIDE "
            "the 8.7-17.6% band the difficult-advice family occupies, and its CI excludes "
            "the base model."),
        "riders": (
            "ONE PASS: the CI is ~17 points wide and cannot separate this arm from the "
            "16.3% control. More important, this arm does NOT isolate the answer term: "
            "seq_mean_token_mean_loss weights each EXAMPLE at 1/global_batch, so removing "
            "the answer CONCENTRATED the same weight onto the trace (~2x per-CoT-token "
            "gradient weight). Read this as 'reasoning-only at double density preserves "
            "the effect', NOT as 'the answer was unnecessary'."),
        "judging_cost_usd": results.get("judging_cost_usd"),
    }
    if dry_run:
        print(">>> dry run; nothing pushed")
        print(json.dumps(fields, indent=2)[:900])
        return
    url = push_run_dir(out, repo, fields, private=private, repo_type="dataset",
                       front_matter={"tags": ["eval-run", "eval:odcv_bench",
                                              f"model:{MODEL_KEY}", "mode:think"]})
    print(f">>> pushed {url}")


if __name__ == "__main__":
    fire.Fire(main)
