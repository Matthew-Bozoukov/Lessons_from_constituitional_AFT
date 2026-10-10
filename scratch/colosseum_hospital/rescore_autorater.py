# ABOUTME: Re-reads finished Hospital arms' text with the autorater (hospital/autorater.py rescore_arm):
# ABOUTME: labels every episode, re-runs the action measures, rewrites per_seed.json, prints what moved.
# Run: uv run python scratch/colosseum_hospital/rescore_autorater.py <arm run dir> [...] \
#        --env-logs-root output/colosseum_hospital/env_logs [--config configs/eval/colosseum_hospital.yaml]
"""Paid: one OpenRouter call per pair seat-iteration with reasoning, per seat-iteration with
coalition posts, and per iteration (about 15 per episode) to the eval's judge model. Labels
land in each arm's results/autorater.json and are re-used on a re-run, so an interrupted
rescore resumes. The judge's own measures (judge_*) are left as they were."""

import argparse
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.eval.misalignment.colosseum.hospital.autorater import rescore_arm
from src.eval.misalignment.colosseum.hospital.config import PAIR


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "run_dirs",
        nargs="+",
        type=Path,
        help="arm run dirs (hold results/results.json)",
    )
    ap.add_argument("--config", default="configs/eval/colosseum_hospital.yaml")
    ap.add_argument(
        "--env-logs-root",
        default=None,
        help="where the arms' environment snapshots were pulled (keeps the deficit objective)",
    )
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    load_dotenv()
    cfg = OmegaConf.load(args.config)
    print(
        f">>> autorater model: {cfg.get('autorater', {}).get('model') or cfg.judge.model}"
    )
    for run_dir in args.run_dirs:
        out = rescore_arm(
            run_dir,
            cfg,
            pair=list(PAIR),
            env_logs_root=args.env_logs_root,
            max_workers=args.workers,
        )
        lab = out["labels"]
        print(
            f"\n=== {run_dir}\n  {lab['n_calls']} calls ({lab['reused_episodes']} episodes re-used), "
            f"{lab['unparsed']} unparsed"
        )
        for measure, (old, new) in sorted(out["changed"].items()):
            fmt = lambda v: "None" if v is None else f"{v:.3f}"  # noqa: E731
            print(f"  {measure:28s} {fmt(old)} -> {fmt(new)}")


if __name__ == "__main__":
    main()
