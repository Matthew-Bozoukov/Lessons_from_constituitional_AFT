# ABOUTME: Score an ALREADY-PUBLISHED ODCV run for task progress and republish it whole, under
# ABOUTME: a lawful name, with the progress axis added beside the misalignment one.
# Run: uv run python scratch/odcv_progress_backfill.py --source <hf id> [--no-push] [--limit N]

"""Backfill the progress axis onto a run that predates the progress judge.

Only the SECOND judge runs here — the misalignment verdicts in the source repo are kept
exactly as published, because rescoring them would silently move a number other work already
cites. Everything else is a byte-for-byte clone: the same rollouts, the same MR results, the
same manifests, plus `results/progress_results.json` and a `progress` block in results.json.

The published layout (`rollouts/<variant>/<Scenario>/pass<N>/`) is not the layout the judges
read (`agent_logs/<key>-<variant>/experiments/<Scenario>/rollout_NNN/`), so the transcripts
are relinked into the judge's shape in a work directory; nothing is copied twice.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import DatasetCard, snapshot_download
from omegaconf import OmegaConf

from src.eval.misalignment.odcv import progress_judge
from src.infra.huggingface import hf_repo_id, push_run_dir
from src.naming import eval_name, legacy_subject
from src.utils import git_sha, write_run_meta

load_dotenv(str(Path(__file__).resolve().parents[1] / ".env"))

SOURCE = ("LASR-Callum/2026-09-03-odcv-qwen36-lora-table2-9284-"
          "difficult-advice-chunk-only-702-rank-64-dynbatch")
# The arm's own adapter, whose curated entry supplies the subject the new run's name takes.
ADAPTER = ("LASR-Callum/2026-08-21-qwen36-lora-table2-9284-"
           "difficult-advice-chunk-only-702-rank-64-dynbatch")
# One work root PER SOURCE. The judge cache is keyed `<variant>/<scenario>/rollout_NNN`,
# which is identical across arms, so a shared directory would silently serve arm A's
# verdicts for arm B's transcripts.
WORK_ROOT = Path("output/odcv_progress_backfill")


def link_for_judging(clone: Path, work: Path, model_key: str) -> int:
    """Relink `rollouts/<variant>/<Scenario>/pass<N>/` into the layout the judges glob for.

    Idempotent, and deliberately NOT a wipe: `progress_judge` caches its verdicts under
    `<work>/evaluations/`, and rebuilding the tree from scratch would throw away paid-for
    judging on any re-run.
    """
    n = 0
    for rec in sorted((clone / "rollouts").glob("*/*/pass*/messages_record.txt")):
        pass_dir, scenario, variant = rec.parent.name, rec.parent.parent.name, rec.parent.parent.parent.name
        idx = int(pass_dir.replace("pass", "")) - 1
        dst = (work / "agent_logs" / f"{model_key}-{variant}" / "experiments" / scenario
               / f"rollout_{idx:03d}")
        dst.mkdir(parents=True, exist_ok=True)
        link = dst / "messages_record.txt"
        if not link.exists():
            link.symlink_to(rec.resolve())
        n += 1
    return n


def config_for(clone: Path, results: dict, work_root: Path):
    """The run's own config, or a minimal one when the run predates publishing it.

    Older runs published `metadata/rollout_manifest.json` and `run_meta.json` but no
    `odcv_config.yaml`; the progress judge needs only the model, its key, and the judges.
    """
    path = clone / "metadata" / "odcv_config.yaml"
    if path.is_file():
        return OmegaConf.load(path), path
    cfg = OmegaConf.create({
        "model": results["model"], "model_key": results["model_key"],
        "judges": dict(results["judges"]),
        "bench_dir": "src/eval/misalignment/odcv/third_party/odcv-bench",
    })
    meta = json.loads((clone / "metadata" / "run_meta.json").read_text()) if (
        clone / "metadata" / "run_meta.json").is_file() else {}
    for key in ("temperature", "passes", "constitution"):
        if key in (meta.get("config") or {}):
            cfg[key] = meta["config"][key]
    return cfg, path


def main(source: str = SOURCE, adapter: str = ADAPTER, subject: str = "", push: bool = True,
         max_workers: int = 8, smoke: bool = False, rollout_subdir: str = "",
         progress_judge_model: str = "") -> None:
    work_root = WORK_ROOT / source.split("/")[-1]
    clone = work_root / "clone"
    print(f">>> cloning {source}")
    snapshot_download(source, repo_type="dataset", local_dir=str(clone))

    # Two source shapes. The published contract (rollouts/ results/ metadata/), and the raw
    # working tree some pre-contract runs were pushed as, where a `combined<N>x_<ts>/`
    # directory already holds BOTH the judge layout and that combine's own results.json.
    # `--rollout-subdir` selects the latter; nothing is restructured, so the clone stays a
    # faithful copy of whatever was published and only gains the new axis.
    base = clone / rollout_subdir if rollout_subdir else clone / "results"
    results = json.loads((base / "results.json").read_text())
    model_key = results["model_key"]
    cfg, cfg_in_clone = config_for(clone, results, work_root)
    if rollout_subdir:
        cfg_in_clone = base / "odcv_config.yaml"

    # The progress axis is judged by the SAME model the misalignment axis used for this run,
    # so the two are comparable and neither is advantaged by a stronger judge.
    cfg.progress_judge = True
    # Default: the SAME judge the misalignment axis used for this run, so neither axis is
    # advantaged by a stronger model. `--progress-judge` overrides it — the older arms were
    # scored by an expensive pair, and re-running that pair for the second axis costs several
    # times what the axis is worth on a 200-transcript arm.
    cfg.progress_judges = ({progress_judge_model.split("/")[-1]: progress_judge_model}
                           if progress_judge_model else dict(results["judges"]))
    cfg.judge_workers = max_workers
    OmegaConf.save(cfg, cfg_in_clone)
    cfg_path = work_root / "odcv_config.yaml"
    OmegaConf.save(cfg, cfg_path)

    if rollout_subdir:
        # Already in the judge's layout: judge it in place, so its cache and results land
        # beside the misalignment ones this combine already carries.
        work = base
        n = len(list(work.glob("agent_logs/*/experiments/*/**/messages_record.txt")))
    else:
        work = work_root / "judge_tree"
        n = link_for_judging(clone, work, model_key)
    print(f">>> {n} transcripts to judge with {dict(cfg.progress_judges)}")

    summary = progress_judge.main(rollout_dir=str(work), config=str(cfg_path),
                                  max_workers=max_workers, smoke=smoke)

    # Fold the new axis into the clone: its own file, a block in results.json, a section in
    # the markdown mirror, and the per-judge cache beside the misalignment one.
    if not rollout_subdir:
        shutil.copy2(work / "progress_results.json", base / "progress_results.json")
        for cache in (work / "evaluations").glob("progress_*.json"):
            shutil.copy2(cache, base / f"scores_{cache.stem}.json")
    results["progress"] = summary
    (base / "results.json").write_text(json.dumps(results, indent=2))
    ours = summary["ours"]["overall"]
    md = base / "results.md"
    # Older runs published results.json without the markdown mirror; start one rather than
    # requiring it, so the new axis is readable on the Hub either way.
    md.write_text((md.read_text() if md.is_file() else
                   f"# {source} — results\n\n(mirror created by the progress backfill)\n")
                  + "\n".join([
        "", "## Task progress (TP)", "",
        "Scored by `src/eval/misalignment/odcv/progress_judge.py` on the SAME transcripts,",
        "independent of the misalignment axis: how much the agent actually did, or",
        "substantiated that it could not.", "",
        f"- **TP mean**: {ours['tp_mean']}/5  CI95 {ours['tp_mean_ci95']}",
        f"- **TP >= {summary['threshold']}**: {ours['tp_rate_pct']}%  CI95 {ours['tp_rate_ci95']}",
        f"- submitted (`task_complete`): {summary['submitted_pct']}% — "
        f"{summary['n_capped_no_submit']} rollout(s) capped at {summary['cap_without_submit']}",
        f"- judges: {json.dumps(summary['judges'])}", ""]))
    write_run_meta((clone / "metadata" / "progress") if not rollout_subdir else (base / "progress"),
                   OmegaConf.to_container(cfg, resolve=True),
                   extra={"axis": "progress", "backfilled_from": source,
                          "command": " ".join(sys.argv), "git_sha": git_sha()})

    # The subject: the curated table's word for this arm when it has one, else the subject
    # given on the CLI. The controls are `subject: null` in the table -- the law refuses to
    # DERIVE a name from them -- so a re-judged control names itself through the same escape
    # hatch run_eval gives a pre-law target (`run_name`), and the law still supplies the date.
    from src.naming import artifact_name
    if subject:
        repo_id = artifact_name(subject)
    else:
        repo_id = eval_name("odcv", legacy_subject(adapter))
    print(f">>> new repo name: {repo_id}")
    if not push:
        print(">>> --no-push: stopping before the upload")
        return

    src_card = DatasetCard.load(source, repo_type="dataset")
    tags = list(getattr(src_card.data, "tags", None) or
                ["eval-run", "eval:odcv", f"model:{model_key}"])
    fields = {
        "experiment": f"odcv eval of {results['model']} — misalignment as published, "
                      "task progress backfilled",
        "date_generated": date.today().isoformat(),
        "constitution": str(cfg.get("constitution", "none")),
        "source_repo": f"teaching_claude_why_replication @ {git_sha()}",
        "models": f"target={results['model']} judges={json.dumps(results['judges'])}",
        "generation_config": json.dumps({"temperature": cfg.get("temperature"),
                                         "passes": cfg.get("passes")}),
        "schema": "rollouts/: transcripts; results/: results.json (MR + progress) + "
                  "progress_results.json + judge scores; metadata/: config + manifests",
        "provenance": f"{' '.join(sys.argv)} (clone of {source}; MR verdicts unchanged)",
    }
    # `push_run_dir` takes a full `org/name`; the org comes from HF_ORG at push time.
    url = push_run_dir(clone, hf_repo_id(repo_id), fields, front_matter={"tags": tags})
    print(f">>> pushed {url}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--adapter", default=ADAPTER)
    ap.add_argument("--subject", default="",
                    help="name the run directly (for a source whose adapter the legacy "
                         "table refuses to derive from), e.g. odcv-numina-control-716-seed69")
    ap.add_argument("--no-push", dest="push", action="store_false")
    ap.add_argument("--max-workers", dest="max_workers", type=int, default=8)
    ap.add_argument("--progress-judge", dest="progress_judge_model", default="",
                    help="OpenRouter model id for the progress axis (default: the run's own "
                         "misalignment judges), e.g. google/gemini-3-flash-preview")
    ap.add_argument("--rollout-subdir", dest="rollout_subdir", default="",
                    help="judge a combined<N>x_<ts>/ directory inside a pre-contract dump "
                         "instead of the published rollouts/ tree")
    ap.add_argument("--smoke", action="store_true",
                    help="judge ONE transcript per variant, to check the wiring")
    main(**vars(ap.parse_args()))
