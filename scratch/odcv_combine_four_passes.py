# ABOUTME: Combine the chunk-only arm's 1-pass and 3-pass ODCV runs into one 4-pass result and
# ABOUTME: publish it as a third revision of the same HF repo, with statistics recomputed over 320.
"""Four passes from two runs, judged once, published as one revision.

WHY. `runner.run` only combines the passes of its OWN invocation, so the 1-pass run
(2026-09-07 17:42) and the 3-pass run (19:00-19:24) are two separate results on the Hub:
revision 1 with 80 rollouts and revision 2 with 240. Neither is a 4-pass number. This
rebuilds the pass directories from the two PUBLISHED rollout trees (package_run deletes the
raw ones), merges all four, re-judges the merged set and publishes it over the same repo,
so the head is a genuine 4-pass result with the two partials kept as older revisions.

The sequence after the merge is copied from `runner.run` deliberately -- submission stats,
misalignment judge, progress judge, package_run, then the same push helper run_eval uses --
so nothing about the published layout or the statistics differs from a native 4-pass run.

Judging is content-cached, so the 320 transcripts already scored in the two earlier runs
should mostly hit the cache; the cost line at the end says what was actually spent.

Run:
    uv run python scratch/odcv_combine_four_passes.py            # publish
    uv run python scratch/odcv_combine_four_passes.py --no-push  # build and judge only
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from omegaconf import OmegaConf

from src.eval.layout import assert_layout, publish_layout
from src.eval.misalignment.odcv import odcv_judge, progress_judge
from src.eval.run_eval import _card_fields, _results_markdown
from src.eval.misalignment.odcv.passes import (
    combine_passes, package_run, submission_stats)
from src.infra.huggingface import hf_repo_id, push_run_dir
from src.utils import timestamp, write_run_meta

ROOT = Path("output/odcv")
ONE_PASS = ROOT / "2026-09-07_qwen3_6_27b_lora_t2_9284_da_chunk_only_702_r64_dynbatch_183641"
THREE_PASS = ROOT / "2026-09-07_qwen3_6_27b_lora_t2_9284_da_chunk_only_702_r64_dynbatch_195550"
MODEL_KEY = "qwen3.6_27b_lora_t2_9284_da_chunk_only_702_r64_dynbatch"
TARGET = "LASR-Callum/qwen3.6-27b-lora-t2-9284-da-chunk-only-702-r64-dynbatch"
CONFIG = "configs/eval/odcv/lite.yaml"
VARIANTS = ("mandated", "incentivized")


def rebuild_pass_dirs(run_dir: Path, work: Path) -> list[Path]:
    """Reconstruct driver-shaped pass dirs from a run's PUBLISHED rollouts/ tree.

    package_run repacks `<pass>/agent_logs/<key>-<variant>/experiments/<Scenario>/` into
    `rollouts/<variant>/<Scenario>/pass<N>/`, deleting the originals. This is that mapping
    inverted, so `combine_passes` sees exactly what it saw the first time.
    """
    by_pass: dict[str, Path] = {}
    for variant in VARIANTS:
        vdir = run_dir / "rollouts" / variant
        if not vdir.is_dir():
            continue
        for scen in sorted(vdir.iterdir()):
            if not scen.is_dir():
                continue
            for passd in sorted(scen.iterdir()):
                if not (passd.is_dir() and passd.name.startswith("pass")):
                    continue
                rec = passd / "messages_record.txt"
                if not (rec.is_file() and rec.stat().st_size > 0):
                    continue
                key = passd.name                       # pass1 / pass2 / pass3
                out = by_pass.setdefault(key, work / f"{run_dir.name[-6:]}_{key}")
                dst = out / "agent_logs" / f"{MODEL_KEY}-{variant}" / "experiments" / scen.name
                dst.mkdir(parents=True, exist_ok=True)
                for f in passd.iterdir():
                    if f.is_file():
                        shutil.copy2(f, dst / f.name)
    return [by_pass[k] for k in sorted(by_pass)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-push", action="store_true")
    ap.add_argument("--publish-only", metavar="RUN_DIR",
                    help="skip rebuild+judge; publish this already-packaged run dir")
    args = ap.parse_args()

    if args.publish_only:
        publish(Path(args.publish_only))
        return

    cfg = OmegaConf.load(CONFIG)
    # runner.py injects these per target; the judges read both, and the 1- and 3-pass
    # runs recorded `model` as the key itself, so match them exactly.
    cfg.model = MODEL_KEY
    cfg.model_key = MODEL_KEY
    stamp = timestamp()
    out_dir = ROOT / f"2026-09-07_{MODEL_KEY.replace('.', '_')}_4pass_{stamp}"
    work = out_dir / MODEL_KEY
    work.mkdir(parents=True)

    pass_dirs = rebuild_pass_dirs(ONE_PASS, work) + rebuild_pass_dirs(THREE_PASS, work)
    print(f">>> rebuilt {len(pass_dirs)} pass dirs: {[p.name for p in pass_dirs]}", flush=True)
    assert len(pass_dirs) == 4, f"expected 4 passes, rebuilt {len(pass_dirs)}"

    combined = work / f"combined{len(pass_dirs)}x_{stamp}"
    manifest = combine_passes(pass_dirs, combined, MODEL_KEY,
                              OmegaConf.to_container(cfg, resolve=True))
    print(f">>> combined {manifest['n_transcripts']} transcripts", flush=True)

    # The runner's own order from here on: submission stats before judging (so they survive
    # a judge failure), MR, then the progress axis, then repack.
    submission = submission_stats(combined, MODEL_KEY)
    (combined / "submission_stats.json").write_text(json.dumps(submission, indent=2))
    print(f">>> submit-tool-call rate: {submission['overall']['submitted_pct']}% "
          f"({submission['overall']['n_rollouts']} rollouts)", flush=True)

    cfg_path = out_dir / "odcv_config.yaml"
    OmegaConf.save(cfg, cfg_path)
    odcv_judge.main(rollout_dir=str(combined), config=str(cfg_path),
                    max_workers=int(cfg.get("judge_workers", 8)))
    results = json.loads((combined / "results.json").read_text())
    if bool(cfg.get("progress_judge", True)):
        results["progress"] = progress_judge.main(
            rollout_dir=str(combined), config=str(cfg_path),
            max_workers=int(cfg.get("judge_workers", 8)))

    audits = [{"pass_dir": p.name, "path": str(p), "kept": True} for p in pass_dirs]
    (out_dir / "pass_summary.json").write_text(json.dumps(
        {"requested_passes": 4, "kept_passes": 4, "audits": audits}, indent=2))
    package_run(out_dir, MODEL_KEY, audits, combined)
    results["submission"] = submission
    results["passes"] = {"requested": 4, "kept": 4, "dropped": 0,
                         "n_transcripts": manifest["n_transcripts"], "audits": audits,
                         "combined_from": [str(ONE_PASS), str(THREE_PASS)]}

    _, results_dir, metadata_dir = publish_layout(out_dir)
    (results_dir / "results.json").write_text(json.dumps(results, indent=2))
    (results_dir / "results.md").write_text(_results_markdown(TARGET, "think", results))
    write_run_meta(metadata_dir, OmegaConf.to_container(cfg, resolve=True),
                   extra={"target": TARGET, "mode": "think", "eval": "odcv",
                          "n_passes": 4, "built_by": "scratch/odcv_combine_four_passes.py",
                          "combined_from": [str(ONE_PASS), str(THREE_PASS)]})
    assert_layout(out_dir)

    o = results["ours"]["overall"]
    print(f">>> 4-pass MR {o['mr_pct']}% {o['mr_ci95']} over {o['n_rollouts']} rollouts; "
          f"TP {results.get('progress', {}).get('ours', {}).get('overall', {}).get('tp_mean')}",
          flush=True)

    if args.no_push:
        print(f">>> --no-push: left in {out_dir}\n"
              f">>> publish it with: --publish-only {out_dir}", flush=True)
        return
    publish(out_dir)


def publish(out_dir: Path) -> None:
    """Push a packaged 4-pass run dir over the arm's existing ODCV repo (revision 3)."""
    assert_layout(out_dir)
    repo = hf_repo_id("2026-09-07-odcv-qwen3-6-27b-lora-t2-9284-da-chunk-only-702-r64-dynbatch")
    # The same card run_eval's epilogue would write, so this revision is described exactly
    # like the two it supersedes; only `experiment` says what is different about it.
    card = _card_fields(
        "odcv", OmegaConf.load(CONFIG),
        "uv run python scratch/odcv_combine_four_passes.py",
        experiment=(f"odcv eval of {TARGET} (mode=think) — 4 passes, 320 rollouts: the "
                    "1-pass and 3-pass runs of 2026-09-07 merged and re-judged as one set"),
        models=TARGET)
    url = push_run_dir(out_dir, repo, card, front_matter={
        "tags": ["eval-run", "eval:odcv", "model:qwen36", "mode:think"]})
    print(f">>> pushed {url}", flush=True)


if __name__ == "__main__":
    main()
