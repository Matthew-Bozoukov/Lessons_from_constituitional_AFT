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

RUN = "output/odcv_bench/qwen3_6-27b-lora-t2-9284-chunk-only-702-answeronly-r64/20260901_152637"
MODEL_KEY = "qwen3_6-27b-lora-t2-9284-chunk-only-702-answeronly-r64"
REPO = "2026-09-01-odcv-answer-only-chunk-only-702-1x65"
ADAPTER = "LASR-Callum/qwen3.6-27b-lora-t2-9284-chunk-only-702-answeronly-r64"
CONFIG = "configs/eval/odcv_bench_chunkonly702_answeronly_r64_1x65.yaml"


def main(run: str = RUN, repo: str = REPO, stage: str = "output/odcv_ans702/publish",
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
            f"ODCV-Bench eval of {ADAPTER} (mode=think) - the ANSWER-ONLY supervision "
            "arm on the principle-scoped (chunk-only) corpus. Its 702 difficult-advice "
            "rows train on the VISIBLE ANSWER ONLY: the reasoning trace stays in the "
            "token stream as unsupervised context (no truncation) and earns no loss. "
            "65 cells x 1 rollout, both conditions, driven from local Docker against a "
            "RunPod H200 vLLM endpoint over an SSH tunnel."),
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
            "3 - incentivized/Clinical-Trial-Data-Management, mandated/Roboadvisor-"
            "Ethicsbypass and incentivized/Ai-Hiring-Assistant, all flagged by the "
            "harness itself as 'ok+no_transcript' (the executor exits and Compose "
            "SIGKILLs the orchestrator before it flushes messages_record.txt). Three "
            "different cells again from the CoT arm's one and the earlier runs' two, so "
            "this failure is NOT confined to a fixed set of long-running scenarios."),
                "comparison": (
            "ITS COMPLEMENT, the headline: cot-only on the same 702 rows and the same "
            "cells scored 9.5% [3.2, 17.5]. On the 61 cells BOTH arms judged, answer-only "
            "21.3% (13 violations) vs cot-only 9.8% (6): a paired difference of +11.5pp. "
            "Paired bootstrap CI95 on that difference [+0.0, +23.0] and McNemar exact "
            "p=0.0923 (discordant 10 vs 3) -- large and consistently directed, but NOT "
            "significant at one pass each. Shared control (2 passes) 11.5% [6.2, 19.6]; "
            "base fp8 no-SFT 36.9% [21.4, 53.6]; table2-only 43.9%."),
                "riders": (
            "ONE PASS per arm. The two arms PARTITION the control's difficult-advice "
            "supervision exactly (420,037 + 401,033 = 821,070 tokens, 0 rows mismatched), "
            "and the loss reweighting confounds each against the control in the SAME "
            "direction, so cot-vs-answer is the fair contrast even though "
            "either-vs-control is not. But a seed-only replicate of another arm moved "
            "6.1pp, the paired CI touches zero and McNemar does not clear 0.05: treat "
            "+11.5pp as a strong lead, not a finding. Passes 2-4 on both arms would "
            "settle it."),
                "judging_notes": (
            "Both judges scored all 62 cells; no non-numeric verdicts and no single-judge "
            "cells in this run."),
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
