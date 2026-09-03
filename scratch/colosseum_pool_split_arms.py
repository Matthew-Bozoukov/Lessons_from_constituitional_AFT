# ABOUTME: Pool a Colosseum contrast whose two arms ran in SEPARATE run_eval invocations.
# ABOUTME: Run: uv run python scratch/colosseum_pool_split_arms.py <experiment> <dir> [<dir>...]

"""The contrast, when the arms could not be pooled in-invocation.

`run_eval` pools automatically only when both arms are targets of ONE invocation. On
2026-09-03 they could not be: the run died naming its second arm (a 101-character
directory), so each experiment's control arm finished in one job and its treatment arm
was rerun in another with `ARMS=treatment`. The episodes are identical either way — same
seeds, same environment instances, same server — so the contrast is the same computation
over the same inputs, just assembled here instead of there.

It reconstructs the `runs` list `pool()` expects from the run directories themselves:
each carries `metadata/run_meta.json` (the target it served and the mode it was served
in) and `results/per_seed.json` (the measures). Nothing is inferred from directory names.

Prints the contrast and writes it under `output/colosseum_jira/pooled/`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

from src.eval.misalignment.colosseum.pool import pool
from src.utils import local_name


def _run_record(run_dir: Path) -> dict:
    """The dict pool() consumes, read out of one arm's run directory."""
    meta_path = run_dir / "metadata" / "run_meta.json"
    per_seed = run_dir / "results" / "per_seed.json"
    assert meta_path.is_file(), (
        f"{run_dir} has no metadata/run_meta.json — not a finished arm"
    )
    assert per_seed.is_file(), (
        f"{run_dir} has no results/per_seed.json. Its episodes may exist but the arm "
        "never finished harvesting; rerun that arm rather than pooling a partial one."
    )
    meta = json.loads(meta_path.read_text())
    target = meta["target"]
    return {
        "target": target,
        "model_key": target.split("/")[-1],
        "mode": meta.get("mode", "think"),
        "out_dir": str(run_dir),
        "repo": "",
    }


def main(argv: list[str]) -> None:
    assert len(argv) >= 3, (
        "usage: colosseum_pool_split_arms.py <experiment> <control_dir> <treatment_dir>"
    )
    experiment, dirs = argv[1], [Path(d) for d in argv[2:]]

    cfg = OmegaConf.load("configs/eval/colosseum_jira.yaml")
    cfg = OmegaConf.merge(cfg, OmegaConf.create({"experiment": experiment}))

    runs = [_run_record(d) for d in dirs]
    print(f">>> pooling {experiment} over {len(runs)} arms:")
    for r in runs:
        print(f"      {r['mode']:8s} {r['target']}")

    out_dir = Path("output/colosseum_jira/pooled") / local_name(
        f"{experiment} contrast"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    result = pool(runs, cfg, out_dir)

    print(f"\n=== {experiment}: treatment minus control ===")
    for name, c in result["contrasts"].items():
        lo, hi = c["diff_ci95"]
        flag = "  <-- excludes 0" if (lo > 0 or hi < 0) else ""
        print(
            f"{name:34s} treat={c['treatment_mean']:+9.3f} ctrl={c['control_mean']:+9.3f} "
            f"diff={c['diff']:+9.3f} [{lo:+.3f}, {hi:+.3f}] p={c['p_two_sided']:.4f} "
            f"n={c['n_seeds']}{flag}"
        )
    print(f"\nwritten: {out_dir / 'results' / 'contrasts.json'}")


if __name__ == "__main__":
    main(sys.argv)
