# ABOUTME: Re-score an already-published ODCV run with ODCV-lite's single flash judge, locally.
# ABOUTME: Run: uv run python scratch/odcv_rejudge_lite.py --source <hf odcv repo> --model-key <key>
"""Put a run judged under another protocol onto the lite footing, without re-rolling it.

`2026-09-17-odcv-qwen36-0-da-7` was scored with the then-default configs/eval/odcv.yaml (deleted 2026-09-18; heavy.yaml's) pair of judges
(grok-4.20 + gemini-3.1-pro-preview); every arm it is compared against was scored by lite's
single gemini-3-flash judge. Rollouts are the expensive, GPU half and are identical between the
protocols, so only the judge is re-run here.

Nothing is pushed: the source repo is someone else's published run under the name this result
would take, and rescoring in place would silently move a number other work already cites.
Results land in output/ for comparison. The published -> judge layout relink is reused from
scratch/odcv_progress_backfill.py; the submit-tool-call rate the source run never recorded is
computed with the same function runner.py uses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scratch"))

from odcv_progress_backfill import link_for_judging  # noqa: E402
from src.eval.misalignment.odcv import odcv_judge  # noqa: E402
from src.eval.misalignment.odcv.passes import submission_stats  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--model-key", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out)
    clone, work = out / "clone", out / "work"
    snapshot_download(args.source, repo_type="dataset", local_dir=str(clone),
                      allow_patterns=["rollouts/**", "metadata/run_meta.json"])
    n = link_for_judging(clone, work, args.model_key)
    print(f">>> relinked {n} transcripts for judging", flush=True)

    # lite.yaml verbatim for the judge side; only the target's identity is filled in, as
    # runner.py does before it hands the config to the judge.
    cfg = OmegaConf.load(REPO / "configs/eval/odcv/lite.yaml")
    cfg.model_key = args.model_key
    cfg.model = args.model_key
    cfg.bench_dir = str(REPO / cfg.bench_dir)
    cfg_path = out / "odcv_lite_rejudge_config.yaml"
    OmegaConf.save(cfg, cfg_path)
    print(f">>> judges: {dict(cfg.judges)}", flush=True)

    sub = submission_stats(work, args.model_key)
    (out / "submission_stats.json").write_text(json.dumps(sub, indent=2))
    print(f">>> submit-tool-call rate: {sub['overall']['submitted_pct']}% "
          f"({sub['overall']['n_rollouts']} rollouts)", flush=True)

    odcv_judge.main(rollout_dir=str(work), config=str(cfg_path), max_workers=args.workers)
    res = json.loads((work / "results.json").read_text())
    o = res["ours"]["overall"]
    print(f">>> LITE MR {o['mr_pct']}% CI95 {o['mr_ci95']} over {o['n_rollouts']} rollouts", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
