# ABOUTME: Fire CLI for src/eval/misalignment/odcv/odcv_rollout.py, which has no __main__ block.
# ABOUTME: Run: uv run python scratch/odcv_rollout_cli.py --config configs/eval/odcv_bench_<arm>.yaml

"""Standalone driver for one ODCV-Bench rollout pass.

`odcv_rollout.main` is reachable from the eval framework (`runner.py` calls it directly),
but the module defines no `__main__`, so the invocation the sibling configs document —
`uv run python src/eval/misalignment/odcv/odcv_rollout.py --config ...` — silently does
nothing. scratch/odcv_repeat_rollouts.sh already expects a CLI at THIS path.

Every argument of `odcv_rollout.main` passes straight through, including `--resume <dir>`
to continue a partial pass and `--smoke` to run one scenario per variant.
"""

from __future__ import annotations

import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.misalignment.odcv import odcv_rollout  # noqa: E402

if __name__ == "__main__":
    fire.Fire(odcv_rollout.main)
