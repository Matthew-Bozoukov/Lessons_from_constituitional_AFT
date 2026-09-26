#!/usr/bin/env python
# ABOUTME: Thin driver for src/eval/misalignment/colosseum/publish.py — finish run dirs a
# ABOUTME: `uv run evals` invocation left unjudged or unpushed. Run from a machine WITH network.

"""Finish the Colosseum run dirs an invocation could not.

    uv run python scripts/eval/publish_colosseum.py                  # judge + push all Jira dirs
    uv run python scripts/eval/publish_colosseum.py --no-judge       # push only
    uv run python scripts/eval/publish_colosseum.py --no-push        # judge only
    uv run python scripts/eval/publish_colosseum.py --run-dir output/colosseum_jira/<one>
    uv run python scripts/eval/publish_colosseum.py --eval colosseum_hospital --no-judge   # a merged cell

This is the RECOVERY path. The normal one is `uv run evals --name <eval> ...`, which judges
and pushes in the same invocation. It is needed where that invocation cannot finish: on
Killarney the Jira eval runs on a compute node with no route to OpenRouter or the Hub, so it
is given --no-push and never judges, and this runs afterwards on a LOGIN node; for the
Hospital, a cell merged from several pods' pieces (scratch/colosseum_hospital/merge_cells.py)
has no invocation of its own to push it. Either way the name, card and tags are the ones
run_eval would have written (src/eval/misalignment/colosseum/publish.py).

The multi-agent runs publish to the group org (`--hf-org`, default `dougalldeepmind`).
`src.infra.huggingface.hf_org` resolves the destination from `HF_ORG` in the environment and
refuses to take one from a config, so this flag sets that variable — the sanctioned
redirect, and the reason the repo's own docstring notes `HF_ORG=<other> uv run ...` works.
The two adapters under test are READ from their own org, which HF_ORG does not affect.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.eval.misalignment.colosseum.publish import (
    JUDGES,
    find_run_dirs,
    finish_run_dir,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval",
        default="colosseum_jira",
        choices=sorted(JUDGES),
        help="which Colosseum eval's run dirs to finish (default: colosseum_jira)",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="directory holding the per-arm run dirs (default: output/<eval>)",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        help="finish only this run dir (repeatable)",
    )
    parser.add_argument(
        "--config", default=None, help="default: configs/eval/<eval>.yaml"
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="skip the judge pass (a re-push, or an arm `uv run evals` already judged)",
    )
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument("--judge-workers", type=int, default=8)
    parser.add_argument(
        "--date",
        default=None,
        help="the day the episodes were run, for the Hub name and the card's "
        "date_generated (default: today — right only when pushing on the day of the run)",
    )
    # This experiment's runs go to the group org. The default is here, not in
    # configs/eval/colosseum_jira.yaml, because the push namespace is the environment's
    # to supply and a config that carried one would push somewhere the rest of the
    # pipeline is not looking (src.infra.huggingface.hf_org).
    parser.add_argument(
        "--hf-org",
        default="dougalldeepmind",
        help="HF namespace to publish to (default: dougalldeepmind, the group org)",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides, e.g. judge.model=x-ai/grok-4.3",
    )
    args = parser.parse_args(argv)
    root = args.root or f"output/{args.eval}"
    config = args.config or f"configs/eval/{args.eval}.yaml"

    load_dotenv()
    # Set AFTER load_dotenv (which never overwrites an already-set variable) and before
    # the first push: .env carries the group org, and this experiment's destination has
    # to win over it.
    os.environ["HF_ORG"] = args.hf_org
    cfg = OmegaConf.merge(
        OmegaConf.load(config), OmegaConf.from_dotlist(args.overrides)
    )

    run_dirs = (
        [Path(d) for d in args.run_dir] if args.run_dir else find_run_dirs(Path(root))
    )
    assert run_dirs, (
        f"no Colosseum run dirs under {root}. A finished arm has "
        "metadata/run_meta.json and rollouts/colosseum/."
    )

    print(
        f">>> finishing {len(run_dirs)} run dir(s); judge={not args.no_judge} "
        f"push={not args.no_push}"
    )
    # Printed loudly because it is NOT the repo default: a run published to the wrong
    # namespace is tedious to move and easy not to notice.
    if not args.no_push:
        print(f">>> publishing to HF org: {args.hf_org}")
    done = [
        finish_run_dir(
            d,
            cfg,
            judge=not args.no_judge,
            push=not args.no_push,
            judge_workers=args.judge_workers,
            eval_name=args.eval,
            produced=args.date,
        )
        for d in run_dirs
    ]
    print(json.dumps(done, indent=2))


if __name__ == "__main__":
    main()
