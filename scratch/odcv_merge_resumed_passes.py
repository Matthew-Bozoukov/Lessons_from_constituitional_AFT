# ABOUTME: Merge the two complete passes of an ODCV run that was interrupted mid-pass-3 with a
# ABOUTME: fresh 1-pass run of the same arm; judge once; publish over the arm's ODCV repo.
"""Three passes from two runs, judged once, published as one revision.

The 2026-09-09 ODCV lite run of `qwen36-0-da-dat-100` was interrupted on pass 3 (network
loss); its passes 1-2 are complete raw pass dirs (package_run never ran, so they still carry
the driver layout). A fresh `passes=1` run of the same arm through `uv run evals` supplies
pass 3 as a PUBLISHED run dir (its raw pass dir is deleted by package_run), rebuilt here into
driver shape exactly as scratch/odcv_combine_four_passes.py does. Then the runner's own order:
combine_passes -> submission_stats -> misalignment judge -> progress judge -> package_run ->
results.json / results.md / run_meta -> push_run_dir with the card and tags run_eval writes.

    uv run python scratch/odcv_merge_resumed_passes.py \
        --paused output/odcv/2026-09-09_qwen36_0_da_dat_100_064457 \
        --fresh  output/odcv/2026-09-09_qwen36_0_da_dat_100_<ts of the passes=1 run> \
        --target dougalldeepmind/2026-09-09-qwen36-0-da-dat-100 [--no-push]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from omegaconf import OmegaConf

from src.eval.layout import assert_layout, publish_layout
from src.eval.misalignment.odcv import odcv_judge, progress_judge
from src.eval.misalignment.odcv.passes import combine_passes, package_run, submission_stats
from src.eval.run_eval import _card_fields, _results_markdown
from src.infra.huggingface import hf_repo_id, push_run_dir
from src.naming import eval_name
from src.utils import timestamp, write_run_meta

VARIANTS = ("mandated", "incentivized")


def raw_pass_dirs(run_dir: Path, model_key: str) -> list[Path]:
    """Complete raw pass dirs (80 transcripts) under an interrupted run's <model_key>/."""
    out = []
    for p in sorted((run_dir / model_key).iterdir()):
        if not (p / "agent_logs").is_dir():
            continue
        n = len(list(p.glob("agent_logs/*/experiments/*/messages_record.txt")))
        print(f"    raw pass {p.name}: {n} transcripts", flush=True)
        if n == 80:
            out.append(p)
    return out


def rebuild_pass_dirs(run_dir: Path, work: Path, model_key: str) -> list[Path]:
    """Invert package_run: published rollouts/<variant>/<Scenario>/pass<N>/ -> driver layout."""
    by_pass: dict[str, Path] = {}
    for variant in VARIANTS:
        vdir = run_dir / "rollouts" / variant
        for scen in sorted(p for p in vdir.iterdir() if p.is_dir()):
            for passd in sorted(p for p in scen.iterdir() if p.is_dir() and p.name.startswith("pass")):
                rec = passd / "messages_record.txt"
                if not (rec.is_file() and rec.stat().st_size > 0):
                    continue
                out = by_pass.setdefault(passd.name, work / f"fresh_{passd.name}")
                dst = out / "agent_logs" / f"{model_key}-{variant}" / "experiments" / scen.name
                dst.mkdir(parents=True, exist_ok=True)
                for f in passd.iterdir():
                    if f.is_file():
                        shutil.copy2(f, dst / f.name)
    return [by_pass[k] for k in sorted(by_pass)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paused", required=True, type=Path)
    ap.add_argument("--fresh", required=True, type=Path)
    ap.add_argument("--target", required=True)
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.paused / "odcv_config.yaml")
    model_key = str(cfg.model_key)
    stamp = timestamp()
    out_dir = args.paused.parent / f"{args.paused.name.rsplit('_', 1)[0]}_merged3_{stamp}"
    work = out_dir / model_key
    work.mkdir(parents=True)

    pass_dirs = raw_pass_dirs(args.paused, model_key)
    assert len(pass_dirs) == 2, f"expected 2 complete raw passes in {args.paused}, got {len(pass_dirs)}"
    fresh = rebuild_pass_dirs(args.fresh, work, model_key)
    assert len(fresh) == 1, f"expected 1 published pass in {args.fresh}, got {len(fresh)}"
    pass_dirs += fresh
    print(f">>> passes: {[p.name for p in pass_dirs]}", flush=True)

    combined = work / f"combined{len(pass_dirs)}x_{stamp}"
    manifest = combine_passes(pass_dirs, combined, model_key, OmegaConf.to_container(cfg, resolve=True))
    print(f">>> combined {manifest['n_transcripts']} transcripts", flush=True)
    submission = submission_stats(combined, model_key)
    (combined / "submission_stats.json").write_text(json.dumps(submission, indent=2))
    print(f">>> submit-tool-call rate: {submission['overall']['submitted_pct']}% "
          f"({submission['overall']['n_rollouts']} rollouts)", flush=True)

    cfg.output_root = str(out_dir)
    cfg_path = out_dir / "odcv_config.yaml"
    OmegaConf.save(cfg, cfg_path)
    odcv_judge.main(rollout_dir=str(combined), config=str(cfg_path), max_workers=int(cfg.get("judge_workers", 8)))
    results = json.loads((combined / "results.json").read_text())
    if bool(cfg.get("progress_judge", True)):
        results["progress"] = progress_judge.main(rollout_dir=str(combined), config=str(cfg_path),
                                                  max_workers=int(cfg.get("judge_workers", 8)))
    audits = [{"pass_dir": p.name, "path": str(p), "kept": True, "clean": True} for p in pass_dirs]
    (out_dir / "pass_summary.json").write_text(json.dumps(
        {"requested_passes": 3, "kept_passes": 3, "audits": audits,
         "merged_from": [str(args.paused), str(args.fresh)]}, indent=2))
    package_run(out_dir, model_key, audits, combined)
    results["submission"] = submission
    results["passes"] = {"requested": 3, "kept": 3, "dropped": 0, "n_transcripts": manifest["n_transcripts"],
                         "audits": audits, "merged_from": [str(args.paused), str(args.fresh)]}
    _, results_dir, metadata_dir = publish_layout(out_dir)
    (results_dir / "results.json").write_text(json.dumps(results, indent=2))
    (results_dir / "results.md").write_text(_results_markdown(args.target, "think", results))
    command = f"uv run python scratch/odcv_merge_resumed_passes.py --paused {args.paused} --fresh {args.fresh} --target {args.target}"
    write_run_meta(metadata_dir, OmegaConf.to_container(cfg, resolve=True),
                   extra={"target": args.target, "mode": "think", "eval": "odcv", "n_passes": 3,
                          "built_by": "scratch/odcv_merge_resumed_passes.py", "command": command,
                          "merged_from": [str(args.paused), str(args.fresh)]})
    assert_layout(out_dir)
    o = results["ours"]["overall"]
    print(f">>> 3-pass MR {o['mr_pct']}% [{o['mr_ci95_lo']}, {o['mr_ci95_hi']}] over {o['n_rollouts']} rollouts; "
          f"TP {results.get('progress', {}).get('ours', {}).get('overall', {}).get('tp_mean')}; "
          f"submitted {submission['overall']['submitted_pct']}%", flush=True)
    if args.no_push:
        print(f">>> --no-push: left in {out_dir}", flush=True)
        return
    arm = args.target.split("/")[-1].split("-", 3)[-1]          # drop the adapter's date
    repo = hf_repo_id(eval_name("odcv", arm))
    card = _card_fields("odcv", cfg, command,
                        experiment=(f"odcv eval of {args.target} (mode=think) — 3 passes, "
                                    f"{manifest['n_transcripts']} rollouts: passes 1-2 of the run interrupted "
                                    "on 2026-09-09 (network loss) merged with a fresh pass and judged as one set"),
                        models=args.target)
    url = push_run_dir(out_dir, repo, card, front_matter={
        "tags": ["eval-run", "eval:odcv", f"model:{model_key}", "mode:think"]})
    print(f">>> pushed {url}", flush=True)


if __name__ == "__main__":
    main()
