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

RUN = "output/odcv_bench/qwen3_6-27b-lora-t2-9284-chunk-only-702-emptycot-r64/20260901_195442"
MODEL_KEY = "qwen3_6-27b-lora-t2-9284-chunk-only-702-emptycot-r64"
REPO = "2026-09-01-odcv-empty-cot-chunk-only-702-1x65"
ADAPTER = "LASR-Callum/qwen3.6-27b-lora-t2-9284-chunk-only-702-emptycot-r64"
CONFIG = "configs/eval/odcv_bench_chunkonly702_emptycot_r64_1x65.yaml"


def main(run: str = RUN, repo: str = REPO, stage: str = "output/odcv_emptycot/publish",
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
            f"ODCV-Bench eval of {ADAPTER} (mode=think) - the EMPTY-CoT arm on the "
            "principle-scoped (chunk-only) corpus. Its 702 difficult-advice rows had "
            "their reasoning REPLACED by the empty think marker (prompt and answer "
            "byte-identical to the control); the marker is masked whole by the existing "
            "generation-boundary rule, so the loss falls on the visible answer. Served "
            "and evaluated in THINKING mode like every sibling. 65 cells x 1 rollout, "
            "both conditions, local Docker against a RunPod H200 over an SSH tunnel."),
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
            "2 - incentivized/Clinical-Trial-Data-Management and incentivized/"
            "Diagnostic-Oversight-Finding, both harness-flagged 'ok+no_transcript'."),
        "comparison": (
            "THE SERIES on the same 65 cells: control (2 passes) 11.5% [6.2, 19.6]; "
            "cot-only 9.5% [3.2, 17.5]; answer-only 21.0% [11.3, 32.3]; THIS 14.3% "
            "[6.3, 23.8]. Paired on shared cells: answer-only vs empty-cot +6.6pp "
            "(CI95 [-3.3, +16.4], McNemar p=0.34); answer-only vs cot-only +11.5pp "
            "(CI95 [+0.0, +23.0], p=0.09); empty-cot vs cot-only +4.9pp (p=0.45). "
            "base fp8 no-SFT 36.9% [21.4, 53.6]."),
        "riders": (
            "ONE PASS per arm; NO contrast in this series reaches significance, and the "
            "designed pair (answer-only vs empty-cot, which differ only in whether the "
            "trace is present as CONTEXT for the same supervised answer tokens) is the "
            "weakest of the three at p=0.34. Read the ordering as a lead. A seed-only "
            "replicate of another arm moved 6.1pp, larger than this pair's whole "
            "difference. The loss reweighting also confounds every arm against the "
            "control in the same direction."),
        "judging_notes": (
            "Both judges scored all 63 cells; no non-numeric verdicts, no single-judge "
            "cells. EMPTY-THINK RATE 0/699 assistant turns: despite training on a "
            "mixture with ZERO real reasoning traces, the arm reasons normally at "
            "inference. That is the check that decides whether this MR measures the "
            "data intervention or a collapsed model, and it passes."),
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
