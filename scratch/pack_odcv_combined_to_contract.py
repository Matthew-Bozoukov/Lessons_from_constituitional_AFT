# ABOUTME: Repack a pre-contract ODCV combined-run dir (agent_logs/ + evaluations/) into the
# ABOUTME: published rollouts/ results/ metadata/ layout, non-destructively, and push to HF.

"""Publish an ODCV run that predates the layout contract.

The scratch-driver era wrote `agent_logs/<model>-<variant>/experiments/<Scenario>/rollout_NNN/`
plus a sibling `evaluations/`. The contract (src/eval/layout.py) wants
`rollouts/<variant>/<Scenario>/pass<N>/` + `results/` + `metadata/`, which is what the
dashboard reads and what run_eval produces today. This converts one into the other by COPY —
the source dir is never modified, so a bad pack costs nothing.

`rollout_NNN` becomes `pass<N+1>`, matching how the already-published seed-42/69 repos name
their passes, so an arm's repos stay mutually readable.

    uv run python scratch/pack_odcv_combined_to_contract.py \
        --run_dir <combined dir> --repo_id matboz/<YYYY-MM-DD>-<desc> [--push False]
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import date
from pathlib import Path

import fire
from dotenv import load_dotenv

from src.eval.layout import assert_layout, publish_layout
from src.huggingface import push_run_dir

VARIANT = re.compile(r"-(incentivized|mandated)$")


def pack(run_dir: Path, dest: Path) -> dict:
    """Copy one pre-contract run into the published layout under `dest`. Returns a manifest."""
    rollouts, results, metadata = publish_layout(dest)
    n_cells, passes = 0, set()
    for rec in sorted(run_dir.rglob("messages_record.txt")):
        if rec.stat().st_size == 0:
            continue
        parts = rec.parts
        try:
            model_variant = parts[parts.index("agent_logs") + 1]
            scenario = parts[parts.index("experiments") + 1]
        except (ValueError, IndexError):
            continue
        m = VARIANT.search(model_variant)
        assert m, f"cannot read variant from {model_variant}"
        variant = m.group(1)
        # Two shapes exist. A COMBINED multi-pass run nests the transcript under
        # `rollout_NNN/`; a single-pass run writes it straight into the scenario dir. Treat
        # the latter as pass 1 rather than refusing it — both are legitimate runs.
        idx = rec.parent.name
        if idx.startswith("rollout_"):
            n = int(idx.split("_")[1])
        else:
            assert idx == scenario, (
                f"transcript at unexpected depth: {rec} (parent {idx!r} is neither a "
                "rollout_NNN dir nor the scenario dir)")
            n = 0
        passes.add(n)
        out = rollouts / variant / scenario / f"pass{n + 1}"
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rec, out / "messages_record.txt")
        dock = rec.parent / "docker_output.log"
        if dock.is_file():
            # Container stdout, NOT the rollout (CLAUDE.md: "logs" means rollouts). Carried
            # because the sibling repos carry it and it is where container failures show up.
            shutil.copy2(dock, out / "docker_output.log")
        (out / "cell_meta.json").write_text(json.dumps(
            {"variant": variant, "scenario": scenario, "pass": n + 1,
             "source_rollout": idx}, indent=2))
        n_cells += 1

    # run_meta/manifest live at the run root on single-pass runs and under evaluations/ on
    # combined ones; take whichever exists so neither shape loses its provenance.
    for src, dst in ((run_dir / "results.json", results / "results.json"),
                     (run_dir / "run_meta.json", metadata / "run_meta.json"),
                     (run_dir / "evaluations" / "run_meta.json", metadata / "run_meta.json"),
                     (run_dir / "rollout_manifest.json",
                      metadata / "rollout_manifest.json")):
        if src.is_file() and not dst.exists():
            shutil.copy2(src, dst)
    for sc in sorted((run_dir / "evaluations").glob("scores_*.json")):
        shutil.copy2(sc, results / sc.name)

    manifest = {"source": str(run_dir), "n_cells": n_cells,
                "passes": sorted(p + 1 for p in passes)}
    (metadata / "pack_manifest.json").write_text(json.dumps(manifest, indent=2))
    assert_layout(dest)
    return manifest


def main(run_dir: str, repo_id: str, date_generated: str = "", push: bool = True,
         out: str = "output/odcv_packed") -> None:
    """Pack a pre-contract ODCV run and (optionally) push it.

    Args:
        run_dir: The combined/pass directory holding agent_logs/ and evaluations/.
        repo_id: Destination HF dataset, `<org>/<YYYY-MM-DD>-<description>`.
        date_generated: ISO date the DATA was produced (not today). Defaults to today with a
            warning, because the card's date_generated is part of the record.
        push: False to pack locally and stop, for inspection before publishing.
        out: Where the packed copy is written.
    """
    load_dotenv()
    src = Path(run_dir)
    assert (src / "agent_logs").is_dir(), f"no agent_logs/ under {src}"
    dest = Path(out) / repo_id.split("/")[-1]
    if dest.exists():
        shutil.rmtree(dest)
    manifest = pack(src, dest)
    print(f">>> packed {manifest['n_cells']} cells, passes {manifest['passes']} -> {dest}")

    results = json.loads((dest / "results" / "results.json").read_text())
    ours = results.get("ours", {})
    for k in ("overall", "mandated", "incentivized"):
        if k in ours:
            print(f"    {k:14s} MR {ours[k].get('mr_pct')}%  sev {ours[k].get('mean_severity')}")
    if not push:
        print(">>> --push False, stopping before upload")
        return

    if not date_generated:
        date_generated = date.today().isoformat()
        print(f"!!! no --date_generated given; carding as {date_generated}, which is the "
              "UPLOAD date, not the generation date")
    fields = {
        "experiment": ("ODCV-Bench agentic-misalignment eval of the synthdoc/difficult-advice "
                       f"arm, {manifest['n_cells']} cells over "
                       f"{len(manifest['passes'])} rollout pass(es)"),
        "date_generated": date_generated,
        "constitution": "constitutions/claude_distilled_09_principles_mid_20260804/constitution.md",
        "source_repo": f"teaching_claude_why_replication @ {__import__('src.utils', fromlist=['git_sha']).git_sha()}",
        "models": f"target={results.get('model_key', '?')} base=Qwen/Qwen3.6-27B",
        "generation_config": json.dumps({"temperature": 0.0,
                                         "judges": list(results.get("judges", []))}),
        "schema": ("rollouts/<variant>/<scenario>/pass<N>/: messages_record.txt is the agent "
                   "rollout (prompt + actions), docker_output.log is container stdout, "
                   "cell_meta.json names the cell; results/: results.json + per-judge scores; "
                   "metadata/: run_meta.json + pack_manifest.json"),
        "provenance": (f"packed from {src} by scratch/pack_odcv_combined_to_contract.py; "
                       "originally produced by the pre-contract ODCV scratch driver"),
    }
    url = push_run_dir(dest, repo_id, fields, private=False,
                       front_matter={"tags": ["eval-run", "eval:odcv",
                                              f"model:{results.get('model_key', 'unknown')}",
                                              "mode:think"]})
    print(f">>> pushed {url}")


if __name__ == "__main__":
    fire.Fire(main)
