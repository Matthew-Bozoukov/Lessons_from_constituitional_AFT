# ABOUTME: ODCV-Peer runner: the measured arm beside a teammate on ONE vLLM server, two concurrent
# ABOUTME: seats per ODCV cell, one CONDITION at a time — judged, fact-sheeted, packaged.

"""The eval-framework entrypoint for `odcv_peer` (registry: src/eval/__init__.py).

A condition (configs/eval/odcv_peer/lite.yaml `conditions:`) says which ODCV variants run,
which prompt each seat holds — `cell` (the cell's own variant), `mandated` or
`incentivized` — and who the teammate is: `base` (the untuned base model, co-served), `self`
(the measured arm again: a homogeneous team) or `scripted` (the quiet control seat). Inside a
condition the ODCV driver runs unchanged over the listed variants; `odcv_rollout.team_spec_for`
spells the seats out per cell. Every condition is judged three ways — the unchanged ODCV rubric
on the measured seat, the same rubric on a live teammate, and the judged fields of the fact
sheet (peer/judge.py) — labelled (peer/facts.py), then packed under `rollouts/<condition>/…`,
`results/<condition>/…`, `metadata/<condition>/…`.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from omegaconf import OmegaConf

from src.eval.misalignment.odcv import odcv_judge, progress_judge
from src.eval.misalignment.odcv.odcv import VARIANTS, scenario_names
from src.eval.misalignment.odcv.passes import combine_passes, package_run, submission_stats
from src.eval.misalignment.odcv.peer import facts as team_facts
from src.eval.misalignment.odcv.peer import judge as team_judge
from src.eval.misalignment.odcv.runner import _bridge_url, _run_pass
from src.utils import timestamp

PROMPT_RULES = ("cell", "mandated", "incentivized")


def cell_id(arm: str, condition: str, variant: str) -> str:
    """THE identity of a cell, built and never typed: `<measured arm>/<condition>/<variant>`.

    The measured arm is the target's model_key (an organism's name minus its date, or a base
    model's registered key), the condition is a `conditions:` name from the config, the
    variant is the ODCV variant of the cell. Plots and tables join on this string and read
    labels from the config, so nothing about an arm is ever spelled by hand at plot time.
    """
    return f"{arm}/{condition}/{variant}"


def seat_string(model_name: str, prompt_rule: str, variant: str) -> str:
    """`<served model>:<prompt it held>` for one seat of one cell."""
    prompt = variant if prompt_rule == "cell" else prompt_rule
    return f"{model_name}:{prompt}"


def stamp_identity(cells: list[dict], arm: str, condition: str, condition_label: str, mate: dict,
                   measured_model: str, measured_prompt: str) -> list[dict]:
    """Write the arm identity into every cell record: `cell_id`, the two seat strings, the
    condition and its display label. The cell's OUTCOME label (`label`: own_violation,
    complicit, ...) is the fact sheet's and is never touched here — a display label in that
    slot zeroed every label metric once (2026-09-23)."""
    for c in cells:
        c["arm"] = arm
        c["condition"] = condition
        c["condition_label"] = condition_label
        c["cell_id"] = cell_id(arm, condition, c["variant"])
        c["seat1"] = ("scripted:-" if mate["kind"] == "scripted"
                      else seat_string(mate["model_name"], mate["prompt"], c["variant"]))
        c["seat2"] = seat_string(measured_model, measured_prompt, c["variant"])
    return cells


def _teammate(target, cfg, spec) -> dict:
    """Resolve a condition's teammate to a served model (or the scripted seat)."""
    model = str(spec.get("model", "base"))
    prompt = str(spec.get("prompt", "cell"))
    assert prompt in PROMPT_RULES, f"teammate prompt {prompt!r} not in {PROMPT_RULES}"
    if model == "scripted":
        return {"kind": "scripted"}
    if model == "self":
        return {"kind": "live", "model_name": target.model_name, "hf_path": target.spec.hf_path,
                "revision": target.spec.revision, "mode": target.spec.mode, "prompt": prompt}
    if model == "base":
        hf, mode = str(cfg.teammate_base.hf_path), cfg.teammate_base.get("mode", None)
    else:
        hf, mode = model, spec.get("mode", None)
    mate = target.sibling(hf, mode=mode)
    assert mate.base_url == target.base_url, "the teammate must be served on the measured arm's server"
    assert mate.model_name != target.model_name, (
        f"teammate {hf} resolves to the measured arm's served model {target.model_name!r}; "
        "say `model: self` if a homogeneous team is what you mean")
    return {"kind": "live", "model_name": mate.model_name, "hf_path": hf,
            "revision": mate.spec.revision, "mode": mate.spec.mode, "prompt": prompt}


def _condition_config(cfg, cond, mate: dict, bench_dir: Path, smoke: bool, work: Path):
    """The per-condition config the ODCV driver reads: team block + the condition's cells."""
    name = str(cond.name)
    variants = [str(v) for v in cond.variants]
    assert variants and all(v in VARIANTS for v in variants), f"{name}: variants {variants}"
    measured_prompt = str(cond.get("measured_prompt", "cell"))
    assert measured_prompt in PROMPT_RULES, f"{name}: measured_prompt {measured_prompt!r}"
    rcfg = OmegaConf.merge(cfg)
    rcfg.team = {
        "name": name,
        "measured_prompt": measured_prompt,
        "teammate": mate,
        "team_text": cond.get("team_text", None) or cfg.get("team_text", None),
        "nudge_on_board_only": bool(cfg.get("nudge_on_board_only", True)),
        "protected_roots": list(cfg.get("protected_roots") or ["/app", "/usr/local/bin"]),
        "quiet_posts": list(cfg.get("quiet_posts") or []) or None,
    }
    sets = cond.get("smoke_scenarios", None) if smoke else cond.get("scenarios", None)
    include: dict[str, list[str]] = {}
    for v in variants:
        names = sets.get(v, None) if sets is not None else None
        include[v] = [str(n) for n in names] if names is not None else scenario_names(bench_dir, v)
    rcfg.include_scenarios = include
    rcfg.output_root = str(work)
    return rcfg


def run(target, cfg, out_dir: Path) -> dict:
    """Run ODCV-Peer against a ServedTarget (the measured arm), one condition at a time.

    Returns the summary run_eval publishes: per condition the measured seat's ODCV numbers,
    progress, submission and the team metrics (overall and per variant), plus `team_pooled`,
    the live-teammate conditions pooled.
    """
    cfg = OmegaConf.merge(cfg)  # private copy
    cfg.model = target.model_name
    cfg.model_key = target.spec.model_key
    cfg.base_url = _bridge_url(target.base_url)
    bench_dir = Path(cfg.bench_dir).resolve()
    smoke = bool(cfg.get("smoke", False))
    wanted = [str(n) for n in (cfg.get("smoke_run") if smoke else cfg.run)]
    by_name = {str(c.name): c for c in cfg.conditions}
    missing = [n for n in wanted if n not in by_name]
    assert not missing, f"conditions {missing} are not defined under `conditions:` ({sorted(by_name)})"
    n_passes = 1 if smoke else int(cfg.get("passes", 12))
    workers = int(cfg.get("judge_workers", 8))
    work = out_dir / "work"
    summary: dict = {"measured": target.spec.hf_path, "arm": str(cfg.model_key), "smoke": smoke, "conditions": {}}
    pooled: list[dict] = []

    for name in wanted:
        cond = by_name[name]
        mate = _teammate(target, cfg, cond.teammate)
        print(f">>> ODCV-Peer | condition {name} | measured {target.model_name} "
              f"({cond.get('measured_prompt', 'cell')} prompt) | teammate {mate.get('model_name', 'scripted')} "
              f"({mate.get('prompt', '-')} prompt) | variants {list(cond.variants)}", flush=True)
        cond_work = work / name
        cond_work.mkdir(parents=True, exist_ok=True)
        rcfg = _condition_config(cfg, cond, mate, bench_dir, smoke, cond_work)
        cfg_path = cond_work / "odcv_config.yaml"
        OmegaConf.save(rcfg, cfg_path)

        audits, kept = [], []
        for i in range(n_passes):
            print(f">>> ODCV-Peer {name} pass {i + 1}/{n_passes}", flush=True)
            audit = _run_pass(cfg_path, False)
            audit["kept"] = True
            audits.append(audit)
            kept.append(Path(audit["path"]))
        (cond_work / "pass_summary.json").write_text(json.dumps(
            {"condition": name, "requested_passes": n_passes, "kept_passes": len(kept), "audits": audits}, indent=2))

        combined = cond_work / cfg.model_key / f"combined{len(kept)}x_{timestamp()}"
        manifest = combine_passes(kept, combined, str(cfg.model_key), OmegaConf.to_container(rcfg, resolve=True))
        submission = submission_stats(combined, str(cfg.model_key))
        (combined / "submission_stats.json").write_text(json.dumps(submission, indent=2))

        odcv_judge.main(rollout_dir=str(combined), config=str(cfg_path), max_workers=workers, smoke=False)
        results = json.loads((combined / "results.json").read_text())
        progress = None
        if bool(cfg.get("progress_judge", True)):
            progress = progress_judge.main(rollout_dir=str(combined), config=str(cfg_path), max_workers=workers, smoke=False)
        team = team_judge.main(rollout_dir=str(combined), config=str(cfg_path), max_workers=workers,
                               teammate_live=(mate["kind"] == "live"))
        cells = team_facts.assemble(combined, str(cfg.model_key), name, team)
        stamp_identity(cells, str(cfg.model_key), name, str(cond.get("label", name)), mate,
                       target.model_name, str(cond.get("measured_prompt", "cell")))
        metrics = team_facts.metrics(cells, name)
        by_variant = {v: team_facts.metrics([c for c in cells if c["variant"] == v], f"{name}/{v}")
                      for v in sorted({c["variant"] for c in cells})}
        (combined / "team_results.json").write_text(json.dumps(
            {"condition": name, "teammate": mate, "metrics": metrics, "by_variant": by_variant, "cells": cells}, indent=2))
        if mate["kind"] == "live":
            pooled += cells

        package_run(out_dir, str(cfg.model_key), audits, combined, subdir=name, work_root=cond_work)
        summary["conditions"][name] = {
            "label": str(cond.get("label", name)),
            "cell_ids": sorted({c["cell_id"] for c in cells}),
            "seats": sorted({f"{c['seat1']} + {c['seat2']}" for c in cells}),
            "teammate": mate, "variants": list(cond.variants),
            "measured_prompt": str(cond.get("measured_prompt", "cell")),
            "measured": results["ours"], "n_judged": results["n_judged"],
            "progress": (progress or {}).get("ours"), "submission": submission,
            "team": metrics, "by_variant": by_variant,
            "passes": {"requested": n_passes, "kept": len(kept), "n_transcripts": manifest["n_transcripts"]},
        }
        print(f">>> {name}: measured MR {results['ours']['overall']['mr_pct']}% | team "
              f"{json.dumps({k: metrics.get(k) for k in ('hit_rate', 'false_alarm_rate', 'misalignment_rate', 'over_refusal_rate', 'team_fraud_rate')})}",
              flush=True)

    shutil.rmtree(work, ignore_errors=True)
    if pooled:
        summary["team_pooled"] = team_facts.metrics(pooled, "live-teammate conditions pooled")
    return summary
