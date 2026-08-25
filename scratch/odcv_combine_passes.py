# ABOUTME: Merge N single-rollout ODCV pass directories into one combined dir whose scenarios
# ABOUTME: each hold rollout_000..rollout_00N-1, the layout the judge expects.
"""Combine independent ODCV passes into one judgeable run directory.

Each pass writes its own timestamped dir with one transcript per scenario. The judge scores
a directory whose scenarios hold numbered rollout_NNN subdirs, so the passes are laid side by
side here. A scenario missing from a pass simply contributes fewer rollouts -- it is reported,
never silently filled.

Run: uv run python scratch/odcv_combine_passes.py --model_key <key>
"""
import shutil
from pathlib import Path

import fire

from src.utils import timestamp


def main(model_key: str, root: str = "output/odcv_bench", passes: int = 0) -> None:
    """Lay every pass's transcripts into one combined directory.

    Args:
        model_key: The arm's directory name under root.
        root: ODCV output root.
        passes: Expected number of passes; 0 accepts however many are present.

    Raises:
        AssertionError: If fewer than two passes are found, or the count mismatches.
    """
    arm = Path(root) / model_key
    dirs = sorted(d for d in arm.iterdir() if d.is_dir() and not d.name.startswith("combined"))
    assert len(dirs) >= 2, f"found {len(dirs)} pass dirs under {arm}"
    if passes:
        assert len(dirs) == passes, f"found {len(dirs)} pass dirs, expected {passes}"
    out = arm / f"combined{len(dirs)}x_{timestamp()}"
    print(f"combining {len(dirs)} passes -> {out}")

    counts = {}
    for i, d in enumerate(dirs):
        for rec in sorted(d.glob("agent_logs/*/experiments/*/messages_record.txt")):
            scen = rec.parent
            variant = scen.parents[1].name
            dest = out / "agent_logs" / variant / "experiments" / scen.name / f"rollout_{i:03d}"
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(rec, dest / "messages_record.txt")
            docker = scen / "docker_output.log"
            if docker.is_file():
                shutil.copy2(docker, dest / "docker_output.log")
            counts[f"{variant}/{scen.name}"] = counts.get(f"{variant}/{scen.name}", 0) + 1

    full = sum(1 for v in counts.values() if v == len(dirs))
    short = {k: v for k, v in counts.items() if v != len(dirs)}
    print(f"cells: {len(counts)} | with all {len(dirs)} rollouts: {full}")
    if short:
        print(f"cells short of {len(dirs)} rollouts ({len(short)}):")
        for k, v in sorted(short.items()):
            print(f"   {k}: {v}")
    print(f"total transcripts: {sum(counts.values())}")
    print(out)


if __name__ == "__main__":
    fire.Fire(main)
