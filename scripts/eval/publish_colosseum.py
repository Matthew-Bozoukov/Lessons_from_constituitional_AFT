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

The multi-agent runs publish to a PERSONAL namespace (`--hf-org`, default `kunwar45`),
not to the group org the rest of the repo uses. `src.huggingface.hf_org` resolves the
destination from `HF_ORG` in the environment and refuses to take one from a config, so
this flag sets that variable — the sanctioned redirect, and the reason the repo's own
docstring notes `HF_ORG=<other> uv run ...` works. Two consequences worth knowing: the
group dashboard finds eval runs by org, so these will not appear in it; and the two
adapters under test are still READ from their own org, which HF_ORG does not affect.
"""

from __future__ import annotations

import argparse
import json
import os
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
    # This experiment's runs go to a personal namespace rather than the group org. The
    # default is here, not in configs/eval/colosseum_jira.yaml, because the push
    # namespace is the environment's to supply and a config that carried one would push
    # somewhere the rest of the pipeline is not looking (src.huggingface.hf_org).
    parser.add_argument(
        "--hf-org",
        default="kunwar45",
        help="HF namespace to publish to (default: kunwar45, a personal "
        "org — pass the group org explicitly to override)",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides, e.g. judge.model=x-ai/grok-4.3",
    )
    args = parser.parse_args(argv)

    load_dotenv()
    # Set AFTER load_dotenv (which never overwrites an already-set variable) and before
    # the first push: .env carries the group org, and this experiment's destination has
    # to win over it.
    os.environ["HF_ORG"] = args.hf_org
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
        )
        for d in run_dirs
    ]
    print(json.dumps(done, indent=2))


if __name__ == "__main__":
    main()
