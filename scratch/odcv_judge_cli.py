# ABOUTME: Fire CLI for src/eval/misalignment/odcv/odcv_judge.py, which has no __main__ block.
# ABOUTME: Run: uv run python scratch/odcv_judge_cli.py --rollout_dir <combined dir> --config <yaml>

"""Standalone driver for ODCV-Bench judging.

Sibling of scratch/odcv_rollout_cli.py; see that file for why these wrappers exist. Point
`--rollout_dir` at the combined multi-pass directory produced by
scratch/odcv_combine_passes.py, not at an individual pass, or only one rollout per cell is
scored. Judge verdicts cache to `<rollout_dir>/evaluations/`, so a re-run is resumable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.misalignment.odcv import odcv_judge  # noqa: E402

if __name__ == "__main__":
    fire.Fire(odcv_judge.main)
