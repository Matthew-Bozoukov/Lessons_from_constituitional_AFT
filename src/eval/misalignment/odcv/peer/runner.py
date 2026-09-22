# ABOUTME: ODCV-Peer runner: the measured arm beside a live teammate on ONE vLLM server, two
# ABOUTME: concurrent seats per ODCV cell, one rung at a time — judged, fact-sheeted, packaged.

"""The eval-framework entrypoint for `odcv_peer` (registry: src/eval/__init__.py).

A rung is one teammate condition — K1 (the one-time control: a scripted quiet seat), I (live
base Qwen on the incentivized prompt, a natural violator) or M (the same on the mandated
prompt, second-hand pressure). Inside a rung the ODCV driver runs unchanged: the cell's
`variant` names the TEAMMATE's prompt (M = mandated, I = incentivized) and the measured seat
always holds the incentivized prompt (K1: the cell's own variant, as solo), which
`odcv_rollout.team_spec_for` spells out per cell. Every rung is judged three ways — the
unchanged ODCV rubric on the measured seat, the same rubric on the teammate seat, and the
seven judged fields of the fact sheet (peer/judge.py) — and labelled (peer/facts.py), then
packed under `rollouts/<rung>/…`, `results/<rung>/…`, `metadata/<rung>/…`.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from omegaconf import OmegaConf

from src.eval.misalignment.odcv import odcv_judge, progress_judge
from src.eval.misalignment.odcv.passes import combine_passes, package_run, submission_stats
from src.eval.misalignment.odcv.peer import facts as team_facts
from src.eval.misalignment.odcv.peer import judge as team_judge
from src.eval.misalignment.odcv.runner import _bridge_url, _run_pass
from src.utils import timestamp

RUNGS = ("K1", "I", "M")


def _rung_config(cfg, rung: str, mate: dict, smoke: bool, work: Path):
    """The per-rung config the ODCV driver reads: team block + the rung's scenario set."""
    assert rung in RUNGS, f"unknown rung {rung!r} (expected one of {RUNGS})"
    rcfg = OmegaConf.merge(cfg)
    rcfg.team = {
        "rung": rung,
        "teammate": mate,
        "team_text": cfg.get("team_text", None),
        "nudge_on_board_only": bool(cfg.get("nudge_on_board_only", True)),
        "protected_roots": list(cfg.get("protected_roots") or ["/app", "/usr/local/bin"]),
        "quiet_posts": list(cfg.get("quiet_posts") or []) or None,
    }
    sets = (cfg.get("smoke_scenarios") if smoke else cfg.get("scenario_sets")) or {}
    inc = sets.get(rung, None)
    rcfg.include_scenarios = OmegaConf.to_container(inc, resolve=True) if inc is not None else None
    rcfg.output_root = str(work)
    return rcfg


def run(target, cfg, out_dir: Path) -> dict:
    """Run ODCV-Peer against a ServedTarget (the measured arm), rung by rung.

    The teammate is `cfg.teammate.hf_path` co-served on the same server via
    `ServedTarget.sibling` (mode pinned by `cfg.teammate.mode`, the base revision the
    server already holds). Returns the summary run_eval publishes: per rung the measured
    seat's ODCV numbers, progress, submission, the team metrics; plus `team_primary`, the
    I+M pool.
    """
    cfg = OmegaConf.merge(cfg)  # private copy
    cfg.model = target.model_name
    cfg.model_key = target.spec.model_key
    cfg.base_url = _bridge_url(target.base_url)
    mate = target.sibling(str(cfg.teammate.hf_path), mode=cfg.teammate.get("mode", None))
    assert mate.base_url == target.base_url, "the teammate must be served on the measured arm's server"
    assert mate.model_name != target.model_name, (
        f"teammate and measured arm resolve to the same served model {target.model_name!r}")
    mate_block = {"hf_path": mate.spec.hf_path, "model_name": mate.model_name,
                  "mode": mate.spec.mode, "revision": mate.spec.revision,
                  "base_revision": mate.spec.base_revision}
    print(f">>> ODCV-Peer | measured {target.model_name} | teammate {mate.model_name} "
          f"({mate.spec.hf_path} @ {str(mate.spec.revision)[:8]}, mode {mate.spec.mode})", flush=True)

    smoke = bool(cfg.get("smoke", False))
    rungs = list(cfg.get("smoke_rungs") if smoke else cfg.rungs)
    n_passes = 1 if smoke else int(cfg.get("passes", 12))
    workers = int(cfg.get("judge_workers", 8))
    work = out_dir / "work"
    summary: dict = {"teammate": mate_block, "smoke": smoke, "rungs": {}}
    all_cells: dict[str, list[dict]] = {}

    for rung in rungs:
        rung_work = work / rung
        rung_work.mkdir(parents=True, exist_ok=True)
        rcfg = _rung_config(cfg, rung, mate_block, smoke, rung_work)
        cfg_path = rung_work / "odcv_config.yaml"
        OmegaConf.save(rcfg, cfg_path)

        audits, kept = [], []
        for i in range(n_passes):
            print(f">>> ODCV-Peer rung {rung} pass {i + 1}/{n_passes}", flush=True)
            audit = _run_pass(cfg_path, False)
            audit["kept"] = True
            audits.append(audit)
            kept.append(Path(audit["path"]))
        (rung_work / "pass_summary.json").write_text(json.dumps(
            {"rung": rung, "requested_passes": n_passes, "kept_passes": len(kept), "audits": audits}, indent=2))

        combined = rung_work / cfg.model_key / f"combined{len(kept)}x_{timestamp()}"
        manifest = combine_passes(kept, combined, str(cfg.model_key), OmegaConf.to_container(rcfg, resolve=True))
        submission = submission_stats(combined, str(cfg.model_key))
        (combined / "submission_stats.json").write_text(json.dumps(submission, indent=2))

        # 1. the unchanged ODCV judge on the MEASURED seat (results.json in `combined`)
        odcv_judge.main(rollout_dir=str(combined), config=str(cfg_path), max_workers=workers, smoke=False)
        results = json.loads((combined / "results.json").read_text())
        # 2. the progress axis on the measured seat
        progress = None
        if bool(cfg.get("progress_judge", True)):
            progress = progress_judge.main(rollout_dir=str(combined), config=str(cfg_path), max_workers=workers, smoke=False)
        # 3. the teammate under the same rubric, and the seven judged fields
        team = team_judge.main(rollout_dir=str(combined), config=str(cfg_path), max_workers=workers, rung=rung)
        # 4. facts + labels + metrics
        cells = team_facts.assemble(combined, str(cfg.model_key), rung, team)
        metrics = team_facts.metrics(cells, rung)
        (combined / "team_results.json").write_text(json.dumps(
            {"rung": rung, "metrics": metrics, "cells": cells}, indent=2))
        all_cells[rung] = cells

        package_run(out_dir, str(cfg.model_key), audits, combined, subdir=rung, work_root=rung_work)
        summary["rungs"][rung] = {
            "measured": results["ours"], "n_judged": results["n_judged"],
            "progress": (progress or {}).get("ours"), "submission": submission,
            "team": metrics, "teammate_violation_rate": metrics.get("teammate_violation_rate"),
            "passes": {"requested": n_passes, "kept": len(kept), "n_transcripts": manifest["n_transcripts"]},
        }
        print(f">>> rung {rung}: measured MR {results['ours']['overall']['mr_pct']}% | "
              f"team {json.dumps({k: metrics.get(k) for k in ('hit_rate', 'false_alarm_rate', 'misalignment_rate', 'over_refusal_rate', 'team_fraud_rate')})}",
              flush=True)

    shutil.rmtree(work, ignore_errors=True)
    pooled = [c for r in ("I", "M") for c in all_cells.get(r, [])]
    if pooled:
        summary["team_primary"] = team_facts.metrics(pooled, "I+M")
    return summary
