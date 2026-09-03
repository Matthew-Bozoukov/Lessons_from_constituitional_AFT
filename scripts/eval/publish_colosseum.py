#!/usr/bin/env python
# ABOUTME: Thin driver for src/eval/misalignment/colosseum/publish.py — judge the finished
# ABOUTME: episodes and push each run dir to HF. Run from a machine WITH network.

"""Finish the Colosseum runs a GPU node left unpublished.

    uv run python scripts/eval/publish_colosseum.py                  # judge + push all
    uv run python scripts/eval/publish_colosseum.py --no-judge       # push only
    uv run python scripts/eval/publish_colosseum.py --no-push        # judge only
    uv run python scripts/eval/publish_colosseum.py --run-dir output/colosseum_jira/<one>

On Killarney this runs on a LOGIN node: compute nodes have no route to OpenRouter or the
Hub, which is why `uv run evals` there is given --no-push and never judges.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.eval.misalignment.colosseum.publish import find_run_dirs, finish_run_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default="output/colosseum_jira",
        help="directory holding the per-arm run dirs",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="finish only this run dir (repeatable)",
    )
    parser.add_argument("--config", default="configs/eval/colosseum_jira.yaml")
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="skip the judge pass (a re-push needs no new judgements)",
    )
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--judge-workers", type=int, default=8)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides, e.g. judge.model=x-ai/grok-4.3",
    )
    args = parser.parse_args(argv)

    load_dotenv()
    cfg = OmegaConf.merge(
        OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides)
    )

    run_dirs = (
        [Path(d) for d in args.run_dir]
        if args.run_dir
        else find_run_dirs(Path(args.root))
    )
    assert run_dirs, (
        f"no Colosseum run dirs under {args.root}. A finished arm has "
        "metadata/run_meta.json and rollouts/colosseum/."
    )

    print(
        f">>> finishing {len(run_dirs)} run dir(s); judge={not args.no_judge} "
        f"push={not args.no_push}"
    )
    done = [
        finish_run_dir(
            d,
            cfg,
            judge=not args.no_judge,
            push=not args.no_push,
            judge_workers=args.judge_workers,
        )
        for d in run_dirs
    ]
    print(json.dumps(done, indent=2))


if __name__ == "__main__":
    main()
