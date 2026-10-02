# ABOUTME: Finish ODCV runs whose rollouts completed but whose judging died (e.g. OpenRouter out of
# ABOUTME: credit): judge, progress-judge, package and publish exactly as runner.run + run_eval would.
"""    uv run python scratch/odcv_finish_from_rollouts.py output/odcv/<run dir> [<run dir> ...] [--no-push]

Resumes each run at the point `src/eval/misalignment/odcv/runner.py::run` reaches after its
rollouts (line ~181: judge the latest combined dir), then runs the same epilogue
`src/eval/run_eval.py` would (`_publish` with `_card_fields` and `run_tags`), naming, carding and
tagging from the run dir's own launch `run_meta.json` (command, target, revisions, git sha,
model_key, variant). Judge verdicts cache under the combined dir, so a re-run is resumable.
Nothing is re-rolled; no GPU is used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from src.eval.misalignment.odcv import odcv_judge, progress_judge
from src.eval.misalignment.odcv.passes import package_run
from src.eval.layout import run_tags
from src.eval.run_eval import _card_fields, _publish


def finish(out_dir: Path, push: bool) -> str:
    meta = json.loads((out_dir / "run_meta.json").read_text())
    cfg_path = out_dir / "odcv_config.yaml"
    cfg = OmegaConf.load(cfg_path)
    model_key = str(meta["model_key"])
    combined = sorted((out_dir / model_key).glob("combined*x_*"))[-1]
    passes = json.loads((out_dir / "pass_summary.json").read_text())
    manifest = json.loads((combined / "combine_manifest.json").read_text())
    submission = json.loads((combined / "submission_stats.json").read_text())
    print(f">>> {out_dir.name}: judging {combined.name} ({manifest['n_transcripts']} transcripts)", flush=True)

    workers = int(cfg.get("judge_workers", 8))
    odcv_judge.main(rollout_dir=str(combined), config=str(cfg_path), max_workers=workers, smoke=False)
    results = json.loads((combined / "results.json").read_text())
    if bool(cfg.get("progress_judge", True)):
        results["progress"] = progress_judge.main(
            rollout_dir=str(combined), config=str(cfg_path), max_workers=workers, smoke=False)
    package_run(out_dir, model_key, passes["audits"], combined)
    results["submission"] = submission
    results["passes"] = {"requested": passes["requested_passes"], "kept": passes["kept_passes"],
                         "dropped": passes["requested_passes"] - passes["kept_passes"],
                         "n_transcripts": manifest["n_transcripts"], "audits": passes["audits"]}
    results["finished_by"] = ("scratch/odcv_finish_from_rollouts.py: rollouts from the original run; "
                              "judging resumed after the original run's judge calls failed (OpenRouter 402)")

    summary = {"target": meta["target"], "mode": meta["mode"], **results}
    variant = meta.get("variant", "")
    return _publish(
        out_dir, name="odcv", model_key=model_key, mode=meta["mode"], target=meta["target"],
        summary=summary, push=push, run_name=str(cfg.get("run_name") or ""), variant=variant,
        card=_card_fields(
            "odcv", OmegaConf.create(meta["config"]), meta["command"],
            experiment=f"odcv{f' ({variant})' if variant else ''} eval of {meta['target']} (mode={meta['mode']})",
            models=json.dumps({"target": meta["target"], "target_revision": meta["target_revision"],
                               "base": meta["base_model"], "base_revision": meta["base_model_revision"]}),
            source_revision=meta["git_sha"]),
        tags=run_tags("odcv", model_key, meta["mode"], variant=variant))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--no-push", action="store_true")
    a = ap.parse_args()
    for d in a.run_dirs:
        print(">>> pushed" if (url := finish(Path(d), not a.no_push)) else ">>> not pushed", url, flush=True)
