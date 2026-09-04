# ABOUTME: Pass-level plumbing for ODCV-Bench: audit a finished rollout pass for the silent
# ABOUTME: empty-transcript failure, and merge N passes into the rollout_NNN layout the judge reads.

"""Multi-pass support for ODCV-Bench (promoted from scratch/odcv_box_run.py's `_audit_pass`
and scratch/odcv_combine_passes.py).

One invocation of `odcv_rollout.main` is ONE pass — `rollouts_per_cell` in older arm
configs is inert — so repeated rollouts come from invoking the driver N times and merging
the run dirs. `odcv_judge._transcripts` scores repeats only when it finds them as
per-scenario `rollout_NNN/` subdirectories, so passes must be combined into that shape
before judging.

The audit exists because `ok` in a pass's own manifest is not evidence a cell produced
anything: on 2026-08-18 scenarios reported ok while writing no transcript at all, which
silently removed ~21% of every pass (docs/LOG.md). Only a non-empty messages_record.txt
proves a cell ran.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.eval.layout import publish_layout
from src.eval.misalignment.odcv.odcv import VARIANTS
from src.utils import write_run_meta


def audit_pass(run_dir: Path) -> dict:
    """Count what a finished pass actually produced, trusting transcripts over statuses.

    Args:
        run_dir: One pass's output directory (holds `agent_logs/` and, if the driver
            survived to the end, `rollout_manifest.json`).

    Returns:
        Audit record. `clean` is True only when the manifest exists and every expected
        cell has a non-empty transcript; a missing or unparseable manifest means the
        driver died mid-pass, so the pass can never audit clean (`missing_cells` None).
    """
    logs = list(run_dir.rglob("messages_record.txt"))
    nonempty = [p for p in logs if p.stat().st_size > 0]
    statuses: dict[str, int] = {}
    n_expected = None
    cost = None
    manifest = run_dir / "rollout_manifest.json"
    if manifest.is_file():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            statuses = {"unparseable_manifest": 1}
        else:
            n_expected = data.get("n_scenarios")
            cost = data.get("rollout_cost_usd")
            for r in data.get("results", []):
                st = str(r.get("status", "unknown"))
                statuses[st] = statuses.get(st, 0) + 1
    else:
        statuses = {"NO_MANIFEST": 1}
    missing = max(0, n_expected - len(nonempty)) if n_expected is not None else None
    return {
        "pass_dir": run_dir.name,
        "n_expected": n_expected,
        "transcripts_written": len(logs),
        "transcripts_nonempty": len(nonempty),
        "empty_transcripts": len(logs) - len(nonempty),
        "statuses": statuses,
        "rollout_cost_usd": cost,
        "missing_cells": missing,
        "clean": missing == 0,
    }


def combine_passes(pass_dirs: list[Path], out_dir: Path, model_key: str,
                   cfg_container: dict | None = None) -> dict:
    """Merge repeated passes into the directory layout `odcv_judge._transcripts` globs.

    Layout produced:

        <out_dir>/agent_logs/<model_key>-<variant>/experiments/<Scenario>/rollout_NNN/

    The rollout index is the position in `pass_dirs`, so `rollout_002` means "the third
    kept pass" in every scenario. A scenario missing from a pass simply has no directory
    for that index — the judge globs `rollout_*`, so gaps shrink n rather than breaking
    scoring. A scenario is copied only when its `messages_record.txt` exists and is
    non-empty: copying an empty one would let the judge score a silent failure as clean.

    Args:
        pass_dirs: Audited pass directories to merge, oldest first.
        out_dir: Combined directory to create; must not already exist.
        model_key: Namespaces the agent_logs subtrees, as in the pass dirs.
        cfg_container: Resolved config dict for `run_meta.json` provenance (optional
            so unit tests can skip it).

    Returns:
        The combine manifest (also written to `combine_manifest.json` in out_dir).
    """
    assert pass_dirs, "no passes to combine"
    assert not out_dir.exists(), f"refusing to overwrite existing {out_dir}"
    out_dir.mkdir(parents=True)

    counts: dict[str, dict[str, int]] = {}
    skipped: list[str] = []
    for idx, pass_dir in enumerate(pass_dirs):
        per_variant: dict[str, int] = {}
        for variant in VARIANTS:
            src = pass_dir / "agent_logs" / f"{model_key}-{variant}" / "experiments"
            # A variant with no subtree at all is a CONFIGURED absence, not a broken pass:
            # an incentivized-only arm excludes all 40 mandated scenarios, so the driver
            # never creates that tree. Skipping is recorded in the manifest rather than
            # passed over quietly, so a variant that vanished by accident still shows up
            # as a zero where a reader expects a number.
            if not src.is_dir():
                per_variant[variant] = 0
                skipped.append(f"{pass_dir.name}/{variant}: no subtree (variant not run)")
                continue
            dst_root = out_dir / "agent_logs" / f"{model_key}-{variant}" / "experiments"
            kept = 0
            for scen in sorted(src.iterdir(), key=lambda p: p.name.lower()):
                rec = scen / "messages_record.txt"
                if not (rec.is_file() and rec.stat().st_size > 0):
                    skipped.append(f"{pass_dir.name}/{variant}/{scen.name}")
                    continue
                shutil.copytree(scen, dst_root / scen.name / f"rollout_{idx:03d}")
                kept += 1
            per_variant[variant] = kept
        counts[pass_dir.name] = per_variant

    manifest = {
        "model_key": model_key,
        "passes": [d.name for d in pass_dirs],
        "per_pass_counts": counts,
        "n_transcripts": sum(sum(v.values()) for v in counts.values()),
        "skipped_empty": skipped,
    }
    (out_dir / "combine_manifest.json").write_text(json.dumps(manifest, indent=2))
    if cfg_container is not None:
        write_run_meta(out_dir, cfg_container,
                       extra={"combined_from": [str(d) for d in pass_dirs], **manifest})
    return manifest


# The harness's only completion signal is the agent calling `task_complete` (the "submit"
# tool defined in mission_executor/agent_main.py, whose result content is this string). The
# harness never checks whether the KPI was actually met, so this is SUBMISSION, not success.
SUBMIT_MARKER = '"action": "task_complete"'


def submission_stats(combined_dir: Path, model_key: str) -> dict:
    """Cell-weighted rate at which rollouts called `task_complete` (the submit tool).

    Read this next to the misalignment rate: a low MR on rollouts that never submit is
    inaction, not alignment (an inert arm can't misalign). Computed the same way the MR is
    -- per-cell rate first, then averaged over cells so every scenario weighs the same,
    whatever number of rollouts it landed.

    Args:
        combined_dir: The combined `rollout_NNN` directory from `combine_passes`.
        model_key: The served arm's model_key (names the agent_logs subtrees).

    Returns:
        {overall, mandated, incentivized} each {n_scenarios, n_rollouts, submitted_pct}.
    """
    from collections import defaultdict

    per_cell: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for variant in VARIANTS:
        root = combined_dir / "agent_logs" / f"{model_key}-{variant}" / "experiments"
        if not root.is_dir():
            continue
        for rec in root.glob("*/rollout_*/messages_record.txt"):
            scen = rec.parent.parent.name
            per_cell[(variant, scen)].append(SUBMIT_MARKER in rec.read_text(errors="replace"))

    def _summ(cells: dict) -> dict:
        rates = [sum(v) / len(v) for v in cells.values()]
        n_roll = sum(len(v) for v in cells.values())
        return {"n_scenarios": len(rates), "n_rollouts": n_roll,
                "submitted_pct": round(100.0 * sum(rates) / len(rates), 1) if rates else None}

    out = {"overall": _summ(per_cell)}
    for variant in VARIANTS:
        vc = {k: v for k, v in per_cell.items() if k[0] == variant}
        if vc:
            out[variant] = _summ(vc)
    return out


def package_run(out_dir: Path, model_key: str, audits: list[dict], combined: Path) -> None:
    """Repack a finished run into the published layout: rollouts/ results/ metadata/.

    The working tree — raw pass dirs plus the combined judge dir under `<model_key>/` —
    is consumed: every transcript lands exactly once under
    `rollouts/<variant>/<Scenario>/pass<N>/` (N = execution order, dropped passes
    included), next to its `docker_output.log` and a `cell_meta.json` carrying the
    cell's manifest row plus whether it was judged. Judge outputs land in `results/`,
    run-level provenance in `metadata/`. run_eval.py uploads out_dir verbatim, so the
    HF repo holds each rollout once instead of raw-plus-combined twice.

    Args:
        out_dir: The run_eval-owned output dir (becomes the repo root).
        model_key: Namespaces the working tree, as in the pass dirs.
        audits: One audit per pass in execution order; needs `path` and `kept`.
        combined: The judged combined directory (under `<out_dir>/<model_key>/`).
    """
    rollouts, results, metadata = publish_layout(out_dir)
    (metadata / "passes").mkdir(exist_ok=True)

    for i, audit in enumerate(audits, start=1):
        pass_dir = Path(audit["path"])
        manifest_path = pass_dir / "rollout_manifest.json"
        rows: list[dict] = []
        if manifest_path.is_file():
            shutil.copy2(manifest_path,
                         metadata / "passes" / f"pass{i}_rollout_manifest.json")
            rows = json.loads(manifest_path.read_text()).get("results", [])
        if (pass_dir / "run_meta.json").is_file():
            shutil.copy2(pass_dir / "run_meta.json",
                         metadata / "passes" / f"pass{i}_run_meta.json")
        if not rows:
            # Driver died before writing the manifest: index whatever is on disk.
            rows = [{"variant": v, "scenario": s.name, "status": "unknown"}
                    for v in VARIANTS
                    for s in sorted((pass_dir / "agent_logs" / f"{model_key}-{v}"
                                     / "experiments").glob("*")) if s.is_dir()]
        for row in rows:
            variant, scenario = row["variant"], row["scenario"]
            src = pass_dir / "agent_logs" / f"{model_key}-{variant}" / "experiments" / scenario
            dst = rollouts / variant / scenario / f"pass{i}"
            dst.mkdir(parents=True, exist_ok=True)
            rec = src / "messages_record.txt"
            has_transcript = rec.is_file() and rec.stat().st_size > 0
            if has_transcript:
                shutil.copy2(rec, dst / "messages_record.txt")
            if (src / "docker_output.log").is_file():
                shutil.copy2(src / "docker_output.log", dst / "docker_output.log")
            (dst / "cell_meta.json").write_text(json.dumps({
                **row, "pass": i, "pass_dir": pass_dir.name,
                "transcript_bytes": rec.stat().st_size if rec.is_file() else 0,
                # judged == this exact transcript fed the combined dir the judge scored:
                # the pass survived its audit AND the cell produced a non-empty record.
                "judged": bool(audit.get("kept")) and has_transcript,
            }, indent=2))

    shutil.move(str(combined / "results.json"), results / "results.json")
    evals = combined / "evaluations"
    if evals.is_dir():
        for f in sorted(evals.glob("scores_*.json")):
            shutil.move(str(f), results / f.name)
        if (evals / "run_meta.json").is_file():
            shutil.move(str(evals / "run_meta.json"), results / "judging_run_meta.json")
    for src_name, dst_name in (("combine_manifest.json", "combine_manifest.json"),
                               ("run_meta.json", "combine_run_meta.json")):
        if (combined / src_name).is_file():
            shutil.move(str(combined / src_name), metadata / dst_name)
    for root_file, dst_name in (("odcv_config.yaml", "odcv_config.yaml"),
                                ("pass_summary.json", "pass_summary.json")):
        if (out_dir / root_file).is_file():
            shutil.move(str(out_dir / root_file), metadata / dst_name)

    # Everything above is now the only copy; a verbatim upload must not also carry the
    # raw/combined working tree.
    shutil.rmtree(out_dir / model_key)
