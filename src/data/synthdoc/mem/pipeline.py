# ABOUTME: The MEM runner: generates model-evaluates-model documents over a completed
# ABOUTME: difficult-advice run. Each stage snapshots locally and mirrors to HF.

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from src.endpoints.openrouter import OpenRouterClient
from src.utils import git_sha, timestamp

from ..constitution import full_text
from ..core import Checkpoint, Usage, model_cfg
from ..hf_cache import StageCache, read_jsonl
from . import stages


def _load_source(spec: dict) -> tuple[list[dict], dict, str]:
    """Load a completed difficult-advice run's stage-6 records, locally or from HF.

    Args:
        spec: The config's `source` block -- `{local_dir}` or `{hf_repo}`.

    Returns:
        (stage-6 records, the source run's manifest or {}, a provenance label).
    """
    if spec.get("local_dir"):
        d = Path(spec["local_dir"])
        records = read_jsonl(d / "stage_6_final.jsonl")
        mpath = d / "manifest.json"
        manifest = json.loads(mpath.read_text()) if mpath.exists() else {}
        return records, manifest, str(d)

    repo = spec["hf_repo"]
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    records = read_jsonl(Path(hf_hub_download(
        repo, "stage_6_final.jsonl", repo_type="dataset")))
    try:
        manifest = json.loads(Path(hf_hub_download(
            repo, "manifest.json", repo_type="dataset")).read_text())
    except EntryNotFoundError:
        manifest = {}
    return records, manifest, repo


def run(cfg: dict, smoke: bool = False, resume: str | None = None) -> dict:
    """Run the MEM pipeline over a completed difficult-advice run, caching every stage.

        1 source -> 2 plan -> 3 perturbed (flawed cells) -> 4 generated -> 5 sft

    Args:
        cfg: Run config (see configs/data/mem.yaml).
        smoke: Clamp every enabled cell to 2 documents, to validate wiring.
        resume: Existing run directory to continue.

    Returns:
        A manifest dict describing the run.
    """
    started = time.time()
    ts = timestamp()
    if resume:
        run_dir = Path(resume)
        assert run_dir.exists(), f"resume dir does not exist: {run_dir}"
        print(f">>> resuming into {run_dir}")
    else:
        run_dir = Path(cfg["output_dir"]) / (f"smoke_{ts}" if smoke else ts)
    repo = cfg.get("hf_repo_smoke") if smoke else cfg.get("hf_repo")
    cache = StageCache(run_dir, repo, private=bool(cfg.get("hf_private", False)))

    workers = int(cfg.get("workers", 8))
    budget = float(cfg.get("budget_usd", 0)) or None
    client = OpenRouterClient()
    usage = Usage()
    durations: dict[str, float] = {}

    def timed(name: str, fn):
        """Run a stage, recording its wall clock."""
        t0 = time.time()
        out = fn()
        durations[name] = round(time.time() - t0, 1)
        return out

    def guard(stage: str) -> None:
        """Stop before the next stage if the budget is already spent."""
        if budget is not None and usage.usd > budget:
            raise RuntimeError(
                f"budget_usd=${budget:.2f} exceeded (${usage.usd:.2f}) after {stage}. "
                f"Snapshots up to this stage are in {run_dir}; raise budget_usd and "
                f"re-run to resume.")

    constitution = full_text(cfg["constitution"])
    constitution_sha = hashlib.sha256(constitution.encode()).hexdigest()

    # --- stage 1: the source run's final records ------------------------------------
    source_meta_path = run_dir / "source_meta.json"
    if cache.has(1, "source"):
        source = cache.load(1, "source")
        source_meta = json.loads(source_meta_path.read_text()) \
            if source_meta_path.exists() else {}
        print(f">>> stage 1: reused {len(source)} cached source records")
    else:
        source, src_manifest, label = _load_source(cfg["source"])
        # The control's reasoning and every critique are grounded in cfg's constitution;
        # a source run generated against a different one would silently cross arms.
        src_sha = src_manifest.get("constitution_sha256")
        assert src_sha is None or src_sha == constitution_sha, (
            f"source run {label} was generated against a different constitution "
            f"(sha {src_sha[:12]} != {constitution_sha[:12]}). Point cfg.constitution "
            f"at the source run's constitution or pick a matching source.")
        source_meta = {"source_run": label,
                       "source_git_sha": src_manifest.get("git_sha"),
                       "source_constitution_sha256": src_sha}
        cache.save(1, "source", source)
        source_meta_path.write_text(json.dumps(source_meta, indent=2))
    print(f">>> stage 1: {len(source)} source records "
          f"({source_meta.get('source_run', cfg['source'])})")

    # --- stage 2: plan (deterministic) ----------------------------------------------
    cells = {c: int(n) for c, n in cfg["cells"].items()}
    if smoke:
        cells = {c: min(n, 2) for c, n in cells.items()}
    enabled = {c: n for c, n in cells.items() if n > 0}
    assert enabled, "no cell has a positive count; nothing to generate"
    if cache.has(2, "plan"):
        plan = cache.load(2, "plan")
        print(f">>> stage 2: reused {len(plan)} cached plan records")
    else:
        plan = stages.plan_records(
            source, enabled, cfg["explicitness"], int(cfg.get("seed", 0)),
            source_run=source_meta.get("source_run", ""), flaws=cfg.get("flaws"))
        cache.save(2, "plan", plan)
    per_cell_plan = {c: sum(1 for p in plan if p["cell"] == c) for c in sorted(enabled)}
    print(f">>> stage 2: {len(plan)} planned documents {per_cell_plan}")

    # --- stage 3: minimal-pair perturbation (flawed cells only) ---------------------
    flawed_planned = [p for p in plan if p["response_kind"] == "flawed"]
    if flawed_planned:
        if cache.has(3, "perturbed"):
            perturbed = cache.load(3, "perturbed")
            print(f">>> stage 3: reused {len(perturbed)} cached perturbations")
        else:
            perturbed = timed("perturb", lambda: stages.perturb_responses(
                plan, client, usage, workers=workers,
                ckpt=Checkpoint(run_dir / "stage_3_perturbed.partial.jsonl",
                                key="record_id"),
                **model_cfg(cfg, "perturb")))
            cache.save(3, "perturbed", perturbed)
        print(f"    FIRST CHANGE: [{perturbed[0]['flaw']['type']}/"
              f"{perturbed[0]['flaw']['severity']}] {perturbed[0]['change_summary'][:180]}")
        guard("stage 3")
        # A flawed document whose perturbation failed must be dropped, never generated:
        # _eval_response_text would refuse it anyway, but dropping here keeps the stage-4
        # checkpoint clean and the loss visible.
        by_id = {p["record_id"]: p for p in perturbed}
        lost = [p["record_id"] for p in flawed_planned if p["record_id"] not in by_id]
        if lost:
            print(f"!!! dropping {len(lost)} flawed documents without a perturbation "
                  f"(first 3: {lost[:3]})")
        plan = [by_id.get(p["record_id"], p) for p in plan
                if p["response_kind"] != "flawed" or p["record_id"] in by_id]
    else:
        print(">>> stage 3: no flawed cells enabled -- perturbation skipped")

    # --- stage 4: generate ----------------------------------------------------------
    if cache.has(4, "generated"):
        generated = cache.load(4, "generated")
        print(f">>> stage 4: reused {len(generated)} cached documents")
    else:
        model_cfgs = {stages.CELLS[c].model_key: model_cfg(cfg, stages.CELLS[c].model_key)
                      for c in enabled}
        generated = timed("generate", lambda: stages.generate_documents(
            plan, client, usage, model_cfgs, constitution, workers,
            ckpt=Checkpoint(run_dir / "stage_4_generated.partial.jsonl",
                            key="record_id")))
        cache.save(4, "generated", generated)
    print(f"    FIRST REASONING: {generated[0]['reasoning'][:220]}")
    guard("stage 4")

    # --- stage 5: training-ready export ---------------------------------------------
    sft = stages.to_sft(generated)
    cache.save(5, "sft", sft)

    per_cell = {c: sum(1 for r in generated if r["cell"] == c) for c in sorted(enabled)}
    manifest = {
        "run_id": ts,
        "git_sha": git_sha(),
        "smoke": smoke,
        "constitution_sha256": constitution_sha,
        "config": cfg,
        "source": source_meta,
        # What this run ACTUALLY used, which differs from cfg under --smoke.
        "effective": {"cells": cells},
        "counts": {"source": len(source), "plan": len(plan),
                   "generated": len(generated), "sft": len(sft), "per_cell": per_cell},
        "usage": usage.as_dict(),
        "wall_clock_s": round(time.time() - started, 1),
        "stage_seconds": durations,
        "workers": workers,
        "hf_repo": repo,
        "run_dir": str(run_dir),
    }
    cache.save_json("manifest.json", manifest)

    assert len(sft) == len(generated), f"sft={len(sft)} != generated={len(generated)}"
    kept = 100 * len(generated) / max(len(plan), 1)
    print(f">>> {len(generated)}/{len(plan)} planned documents survived ({kept:.1f}%)")
    print(f"\n>>> {len(sft)} training records in {run_dir}")
    print(f">>> spend ${usage.usd:.2f} | {manifest['wall_clock_s']}s")
    if repo:
        print(f">>> https://huggingface.co/datasets/{repo}")
    return manifest
