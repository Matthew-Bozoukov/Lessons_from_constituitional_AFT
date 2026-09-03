#!/usr/bin/env python3
# ABOUTME: Grade a finished SWE-bench baseline rollout with the PINNED official harness —
# ABOUTME: docker + CPU only, so it runs after the GPU is destroyed, and can be re-run anytime.

"""Phase two of the standardized SWE-bench baseline.

`scripts/run_eval.py --name swebench_mini` produces patches and stops; this grades them. The
split exists because the two phases want different machines: rollouts need the served model,
grading needs docker and cores. Keeping an H100 rented while test suites run is money for
nothing.

    uv run scripts/eval/swebench_mini_grade.py --run-dir output/swebench_mini/<key>/<ts>

Re-running is safe and idempotent from the caller's point of view: it regrades the same saved
predictions against the same pinned dataset revision and overwrites the grading outputs.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import fire
from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.eval.capabilities.swebench_mini import grade as grading
from src.eval.capabilities.swebench_mini import metrics
from src.eval.docker import docker_preflight
from src.utils import timestamp


def main(run_dir: str, config: str = "configs/eval/swebench_mini.yaml",
         max_workers: int = 0, push: bool = False) -> None:
    """Grade the predictions in a rollout run directory.

    Args:
        run_dir: The per-target directory run_eval.py created (holds selection.json and
            rollouts/preds.json).
        config: Eval config, for the grading block. The DATASET is not taken from here —
            it comes from selection.json, so a regrade cannot silently drift onto a
            different dataset revision than the agent saw.
        max_workers: Override parallel instances; 0 uses the config value.
        push: Upload the graded run directory to HF (see CLAUDE.md's card requirements).
    """
    load_dotenv()
    docker_preflight()
    out_dir = Path(run_dir)
    sel_path = out_dir / "metadata" / "selection.json"
    if not sel_path.exists() and (out_dir / "selection.json").exists():
        sel_path = out_dir / "selection.json"
        print(">>> legacy run-dir layout (pre-contract): root selection.json")
    selection = json.loads(sel_path.read_text())
    cfg = OmegaConf.load(config)

    report = grading.grade(
        preds_path=out_dir / "rollouts" / "preds.json",
        selected_ids=selection["instance_ids"],
        # From the RUN, not the config: same dataset and revision the rollout used.
        dataset=selection["dataset"], revision=selection["dataset_revision"],
        run_id=f"{out_dir.parent.name}_{selection['subset_hash']}",
        grade_dir=out_dir / "results" / "grading",
        max_workers=max_workers or int(cfg.grading.max_workers),
        cache_level=str(cfg.grading.cache_level), namespace=str(cfg.grading.namespace))

    scores = metrics.resolution_summary(report, selection["instance_ids"])
    results_path = out_dir / "results" / "results.json"
    if not results_path.exists() and (out_dir / "results.json").exists():
        results_path = out_dir / "results.json"  # legacy pre-contract run dir
    summary = json.loads(results_path.read_text()) if results_path.exists() else {}
    summary |= scores | {"harness": report["_harness"], "grading": "complete"}
    line = metrics.report_line(summary.get("target", out_dir.parent.name),
                               summary["provenance"], selection, scores)
    summary["report_line"] = line

    results_path.write_text(json.dumps(summary, indent=2))
    results_path.with_name("results.md").write_text(
        f"# {line}\n\n"
        + "\n".join(f"- **{k}**: {json.dumps(v) if isinstance(v, (dict, list)) else v}"
                    for k, v in sorted(scores.items())) + "\n")
    row = Path("output/eval_summaries") / f"swebench_mini_{out_dir.parent.name}_{timestamp()}.json"
    row.parent.mkdir(parents=True, exist_ok=True)
    row.write_text(json.dumps(summary, indent=2))

    if push:
        from src.infra.huggingface import push_run_dir

        repo_id = (f"{date.today().isoformat()}-swebench-mini-"
                   f"{out_dir.parent.name.replace('_', '-')}")
        tags = {"tags": ["eval-run", "eval:swebench_mini",
                         f"model:{out_dir.parent.name}"]}
        print(">>> pushed " + push_run_dir(out_dir, repo_id, front_matter=tags, fields={
            "experiment": f"Standardized SWE-bench baseline: {summary.get('target', '')}",
            "date_generated": date.today().isoformat(),
            "constitution": "none",
            "source_repo": f"teaching_claude_why_replication @ {summary.get('provenance', {}).get('git_sha', 'see run_meta.json')}",
            "models": str(summary.get("provenance", {}).get("model", "")),
            "generation_config": json.dumps(summary.get("provenance", {})),
            "schema": "results/results.json: pass@1 + counters; rollouts/: trajectories and "
                      "preds; results/grading/: harness report and logs; "
                      "metadata/selection.json: the subset drawn",
            "provenance": f"uv run scripts/eval/swebench_mini_grade.py --run-dir {run_dir}",
        }))

    print("\n" + line)


if __name__ == "__main__":
    fire.Fire(main)
