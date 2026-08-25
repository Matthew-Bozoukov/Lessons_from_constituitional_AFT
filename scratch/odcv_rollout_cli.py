# ABOUTME: Thin fire CLI over src.eval.misalignment.odcv.odcv_rollout.main, which lost its
# ABOUTME: entrypoint in the refactor. Run: uv run python scratch/odcv_rollout_cli.py --config ...

"""Restores a callable entrypoint for the ODCV rollout driver.

`src/eval/misalignment/odcv/odcv_rollout.py` defines `main()` but has no
`if __name__ == "__main__"` block, so running it directly exits 0 having done nothing —
which reads as success. An older worktree carried `scripts/odcv_rollout.py` doing exactly
this; the current tree does not. Kept in scratch/ rather than scripts/ because reinstating
a pipeline entrypoint is a human's call (CLAUDE.md), not an agent's.
"""

import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.misalignment.odcv.odcv_rollout import main  # noqa: E402

if __name__ == "__main__":
    fire.Fire(main)
