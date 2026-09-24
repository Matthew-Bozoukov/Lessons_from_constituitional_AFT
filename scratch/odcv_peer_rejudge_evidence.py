# ABOUTME: Post-hoc ODCV-Peer evidence pass over a PACKAGED run dir: the upstream rubric with the
# ABOUTME: board+ledger in view, and the pair's progress; writes team_results_evidenced.json per condition.
"""Run the two evidence-fed judges (src/eval/misalignment/odcv/peer/judge.py::evidence_judges) over a
run that was judged WITHOUT them, so both arms of 2026-09-23/24 get the same instrument.

    uv run python scratch/odcv_peer_rejudge_evidence.py <packaged run dir> [--push]

Reads `rollouts/<condition>/<variant>/<Scenario>/pass<N>/` (pass<N> is rollout_<N-1> in the cell
keys), caches verdicts under `results/<condition>/mr_evidenced_<judge>.json` and
`team_progress_<judge>.json`, joins them onto the existing `team_results.json` cells and recomputes
the metrics (peer/facts.py, which now carries own_violation_rate_evidenced, judge_talked_out_rate,
team_progress_*, stood_back_rate). The original team_results.json is left untouched; the evidenced
version lands beside it. --push re-uploads the run dir to its Hub repo (metadata/run_meta.json
names it).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from omegaconf import OmegaConf

from src.eval.misalignment.odcv import odcv_judge
from src.eval.misalignment.odcv.peer import facts as team_facts
from src.eval.misalignment.odcv.peer import judge as team_judge


def cells_of(cond_dir: Path) -> list[tuple[str, str, Path]]:
    """(variant, unit, transcript) per packaged cell, unit spelled as the caches key it."""
    items = []
    for rec in sorted(cond_dir.glob("*/*/pass*/messages_record.txt")):
        pass_dir = rec.parent
        m = re.fullmatch(r"pass(\d+)", pass_dir.name)
        assert m, pass_dir
        variant, scen = pass_dir.parent.parent.name, pass_dir.parent.name
        items.append((variant, f"{scen}/rollout_{int(m.group(1)) - 1:03d}", rec))
    return items


def main(run: Path, push: bool) -> None:
    meta = json.loads((run / "metadata" / "run_meta.json").read_text())
    cfg = OmegaConf.create(meta["config"])
    judges = dict(cfg.get("team_judges", None) or cfg.judges)
    workers = int(cfg.get("judge_workers", 8))
    bench = Path(cfg.bench_dir).resolve()
    summary_path = run / "results" / "results.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    pooled: list[dict] = []

    for cond_dir in sorted((run / "rollouts").iterdir()):
        name = cond_dir.name
        res_dir = run / "results" / name
        tr_path = res_dir / "team_results.json"
        if not tr_path.is_file():
            print(f"skip {name}: no team_results.json")
            continue
        items = cells_of(cond_dir)
        for _, _, rec in items:
            team_judge.CELL_DIRS[str(rec)] = rec.parent
            team_judge.CELL_DIRS[
                f"{rec.parent.parent.name}__{rec.parent.name}__judge_copy.txt"
            ] = rec.parent
        print(f">>> {name}: {len(items)} cells", flush=True)
        out: dict = {}
        for stem, judge in team_judge.evidence_judges(
            bench, res_dir / "judge_copies"
        ).items():
            out[stem] = {}
            for key, model in judges.items():
                out[stem][key] = odcv_judge.score_with_judge(
                    judge,
                    model,
                    items,
                    res_dir / f"{stem}_{key}.json",
                    workers,
                    kind=judge.name,
                    budget=odcv_judge.judge_budget(cfg),
                )

        tr = json.loads(tr_path.read_text())
        for c in tr["cells"]:
            c["measured_mr_evidenced"] = team_facts._median(
                out["mr_evidenced"], c["key"]
            )
            c["team_progress"] = team_facts._median(out["team_progress"], c["key"])
        cells = [team_facts.derive(c) for c in tr["cells"]]
        for c, orig in zip(cells, tr["cells"]):
            c["condition_label"] = orig.get("condition_label")
        metrics = team_facts.metrics(cells, name)
        by_variant = {
            v: team_facts.metrics(
                [c for c in cells if c["variant"] == v], f"{name}/{v}"
            )
            for v in sorted({c["variant"] for c in cells})
        }
        # The EVIDENCED REGIME: the same label tree with the evidenced judge standing in for
        # both the transcript-only score and the raw ledger flag. `t_own_tamper` is any
        # non-pristine write to a pre-existing file, which also catches edits the task asked
        # for; the control arm's re-judge (2026-09-24) showed the judge clearing 22.7% of those
        # WITH the hashes in view, so a label tree that ORs the raw flag in overstates own
        # violations. Here the judge, seeing the ledger, decides what a write was.
        regime = []
        for c in tr["cells"]:
            ev = team_facts._median(out["mr_evidenced"], c["key"])
            if ev is None:
                continue
            r = dict(
                c,
                measured_mr=ev,
                measured_mr_evidenced=ev,
                t_own_tamper=False,
                team_progress=team_facts._median(out["team_progress"], c["key"]),
            )
            regime.append(team_facts.derive(r))
        metrics_regime = team_facts.metrics(regime, f"{name} (evidenced regime)")
        by_variant_regime = {
            v: team_facts.metrics(
                [c for c in regime if c["variant"] == v],
                f"{name}/{v} (evidenced regime)",
            )
            for v in sorted({c["variant"] for c in regime})
        }
        (res_dir / "team_results_evidenced.json").write_text(
            json.dumps(
                {
                    **tr,
                    "evidence_judges": list(out),
                    "metrics": metrics,
                    "by_variant": by_variant,
                    "metrics_evidenced_regime": metrics_regime,
                    "by_variant_evidenced_regime": by_variant_regime,
                    "cells": cells,
                },
                indent=2,
            )
        )
        reg_keep = (
            "own_violation_rate",
            "misalignment_rate",
            "complicit_rate",
            "hit_rate",
            "over_refusal_rate",
            "false_alarm_rate",
            "team_fraud_rate",
        )
        print(
            f"    {name} (evidenced regime): "
            + json.dumps(
                {
                    k: (
                        metrics_regime[k]["pct"]
                        if isinstance(metrics_regime.get(k), dict)
                        else metrics_regime.get(k)
                    )
                    for k in reg_keep
                }
            )
        )
        if summary.get("conditions", {}).get(name) is not None:
            summary["conditions"][name]["team_evidenced_regime"] = metrics_regime
            summary["conditions"][name]["by_variant_evidenced_regime"] = (
                by_variant_regime
            )
        if tr.get("teammate", {}).get("kind") == "live":
            pooled += cells
        keep = (
            "own_violation_rate",
            "own_violation_rate_evidenced",
            "judge_talked_out_rate",
            "tamper_unscored_rate",
            "misalignment_rate",
            "hit_rate",
            "measured_progress_mean",
            "team_progress_mean",
            "team_progress_rate",
            "stood_back_rate",
        )
        print(
            f"    {name}: "
            + json.dumps(
                {
                    k: (
                        metrics[k]["pct"]
                        if isinstance(metrics.get(k), dict)
                        else metrics.get(k)
                    )
                    for k in keep
                }
            )
        )
        if summary.get("conditions", {}).get(name) is not None:
            summary["conditions"][name]["team_evidenced"] = metrics
            summary["conditions"][name]["by_variant_evidenced"] = by_variant

    if pooled and summary:
        summary["team_pooled_evidenced"] = team_facts.metrics(
            pooled, "live-teammate conditions pooled (evidenced)"
        )
    if summary:
        summary_path.write_text(json.dumps(summary, indent=2))
        print(">>> results/results.json updated with the evidenced metrics")
    if push:
        from src.infra.huggingface import push_run_dir

        card = (
            json.loads((run / "metadata" / "card.json").read_text())
            if (run / "metadata" / "card.json").is_file()
            else None
        )
        assert card, "no metadata/card.json to re-push with; push by hand"
        print(
            ">>> pushed",
            push_run_dir(
                run,
                card["repo_id"],
                card["fields"],
                front_matter={"tags": card["tags"]},
            ),
        )


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), "--push" in sys.argv)
