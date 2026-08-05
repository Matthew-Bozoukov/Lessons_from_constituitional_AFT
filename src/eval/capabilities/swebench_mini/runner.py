# ABOUTME: Eval-framework entrypoint for the standardized SWE-bench baseline: subset, then
# ABOUTME: PINNED mini-SWE-agent rollouts against the served target; grading is a separate job.

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.eval.capabilities.swebench_mini import agent, grade as grading, images, metrics, subset


def _selection(cfg: DictConfig, out_dir: Path) -> tuple[dict, list[dict]]:
    """Draw the subset and record exactly what was drawn, before any expensive work.

    Returns both the recorded summary and the selected rows — the rows carry the dataset's
    image fields, which the pre-pull step needs and the summary deliberately does not hold.
    """
    instances, revision = subset.load_instances(str(cfg.dataset), str(cfg.split))
    fraction = cfg.subset.get("fraction")
    n = cfg.subset.get("n")
    chosen = subset.select(instances, int(cfg.subset.seed),
                           fraction=float(fraction) if fraction is not None else None,
                           n=int(n) if n is not None else None)
    selection = subset.summarize_selection(chosen, len(instances), int(cfg.subset.seed),
                                           str(cfg.dataset), revision)
    (out_dir / "selection.json").write_text(json.dumps(selection, indent=2))
    print(f">>> subset: {selection['n_selected']}/{selection['split_size']} instances "
          f"({selection['fraction']:.1%}), hash {selection['subset_hash']}")
    print(f">>> repos:  {selection['repo_breakdown']}")
    return selection, chosen


def run(target, cfg: DictConfig, out_dir: Path) -> dict:
    """Roll out the pinned mini-SWE-agent against a ServedTarget (CLAUDE.md contract).

    Grading is NOT run here by default. The rollout phase is the only half that needs the
    GPU, so the run ends by publishing patches and telling you how to grade them on a cheap
    docker host after the pod is destroyed. Set `grade=true` to grade inline anyway (the
    images are already warm on the rollout host, at the cost of an idle GPU).

    Returns:
        The rollout summary — subset identity, scaffold provenance, patch/step/exit counters —
        plus pass@1 when grading ran.
    """
    cfg = OmegaConf.merge(cfg)  # private copy; run() must not mutate the caller's config
    selection, chosen = _selection(cfg, out_dir)
    selected_ids = selection["instance_ids"]

    # Before anything expensive: get the task images local. Upstream's container-start
    # timeout is 120s and cannot cover a cold multi-GB pull — without this every instance
    # dies with TimeoutExpired and an empty patch, which scores as a clean 0%.
    pulled = images.pull_all(chosen, workers=int(cfg.get("pull_workers", 4)))

    official_config = agent.official_config_path()
    overlay = agent.build_overlay(target.base_url, out_dir,
                                  disable_network=bool(cfg.get("disable_network", True)))
    model_name = f"hosted_vllm/{target.model_name}"
    registry = agent.write_cost_registry(out_dir, model_name)
    rollouts_dir = out_dir / "rollouts"
    rollouts_dir.mkdir(parents=True, exist_ok=True)

    code = agent.run_rollouts(
        agent.rollout_command(dataset=str(cfg.dataset), split=str(cfg.split),
                              filter_regex=subset.id_filter_regex(selection["instance_ids"]),
                              workers=int(cfg.workers), model_name=model_name,
                              rollouts_dir=rollouts_dir, overlay=overlay,
                              official_config=official_config),
        agent.rollout_env(registry=registry, global_config_dir=out_dir / "mini_global_config"),
        out_dir / "rollouts.log")
    if code != 0:
        # Not fatal: partial predictions are still a result, and the counters below say how
        # partial. Loud so it cannot be mistaken for a clean run.
        print(f"!!! mini-swe-agent exited {code} — see {out_dir / 'rollouts.log'}; "
              "scoring whatever predictions it produced")

    preds_path = rollouts_dir / "preds.json"
    summary = {
        "selection": selection,
        "provenance": agent.provenance(
            official_config=official_config, overlay=overlay, target=target,
            serve_params=OmegaConf.to_container(cfg.get("serving", {}), resolve=True)),
        "rollout_exit_code": code,
        **pulled,
        **metrics.rollout_summary(metrics.load_preds(preds_path), selected_ids, rollouts_dir),
    }

    if not bool(cfg.get("grade", False)):
        summary["grading"] = "deferred"
        print(f"\n>>> rollouts complete. Destroy the GPU, then grade on a docker host with:\n"
              f"    uv run scripts/eval/swebench_mini_grade.py --run-dir {out_dir}\n")
        return summary

    report = grading.grade(
        preds_path=preds_path, selected_ids=selected_ids, dataset=str(cfg.dataset),
        revision=selection["dataset_revision"],
        run_id=f"{target.spec.model_key}_{selection['subset_hash']}",
        grade_dir=out_dir / "grading", max_workers=int(cfg.grading.max_workers),
        cache_level=str(cfg.grading.cache_level), namespace=str(cfg.grading.namespace))
    scores = metrics.resolution_summary(report, selected_ids)
    summary |= scores | {"harness": report["_harness"]}
    summary["report_line"] = metrics.report_line(target.spec.hf_path, summary["provenance"],
                                                 selection, scores)
    print("\n>>> " + summary["report_line"])
    return summary
