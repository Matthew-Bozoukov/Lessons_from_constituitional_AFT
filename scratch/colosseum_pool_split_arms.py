# ABOUTME: Pool a Colosseum contrast whose arms are spread over several run directories —
# ABOUTME: separate invocations and/or seed shards. Run: uv run python scratch/colosseum_pool_split_arms.py <experiment> <dir>...

"""The contrast, when neither arm arrived as a single run.

`run_eval` pools automatically only when both arms are targets of ONE invocation, and on
2026-09-03 nothing arrived that way. Two things split the arms up:

  * the run died naming its second arm (a 101-character directory), so each experiment's
    control arm finished in one job and its treatment arm was rerun with ARMS=treatment;
  * the seed count was raised from 40 to 190 after a power calculation, and the extra
    seeds were sharded across jobs to use several GPUs at once.

So an arm is now the UNION of several run directories. Merging them is legitimate
precisely because the environment instance depends only on the seed: seed 117 is the same
ticket set, the same agent skills and the same cost matrix no matter which job ran it, so
shards are independent draws from one population rather than repeated measurements. A
seed appearing twice for one arm and cell would break that reading, so it is refused
rather than silently overwritten.

Everything is read from the directories themselves — target and mode from
`metadata/run_meta.json`, measures from `results/per_seed.json` — never inferred from
directory names.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from omegaconf import OmegaConf

from src.eval.misalignment.colosseum.pool import pool
from src.utils import local_name

MERGED_ROOT = Path("output/colosseum_jira/merged")


def _read(run_dir: Path) -> tuple[dict, dict]:
    meta_path = run_dir / "metadata" / "run_meta.json"
    per_seed = run_dir / "results" / "per_seed.json"
    assert meta_path.is_file(), (
        f"{run_dir} has no metadata/run_meta.json — not a finished arm"
    )
    assert per_seed.is_file(), (
        f"{run_dir} has no results/per_seed.json. Its episodes may exist but the arm "
        "never finished harvesting; rerun that shard rather than pooling a partial one."
    )
    return json.loads(meta_path.read_text()), json.loads(per_seed.read_text())


def _merge(target: str, parts: list[tuple[Path, dict]]) -> dict:
    """Union the per-seed measures of one arm's shards, refusing duplicate seeds."""
    merged: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    seen: dict[tuple[str, str, str], Path] = {}
    for run_dir, measures in parts:
        for measure, cells in measures.items():
            for cell, seeds in cells.items():
                for seed, value in seeds.items():
                    key = (measure, cell, seed)
                    if key in seen:
                        raise AssertionError(
                            f"{target}: seed {seed} appears in cell {cell!r} twice — in "
                            f"{seen[key].name} and {run_dir.name}. Overlapping shards "
                            "would double-count one environment instance; re-run the "
                            "shards with disjoint seed ranges."
                        )
                    seen[key] = run_dir
                    merged[measure][cell][seed] = value
    return {m: {c: dict(s) for c, s in cells.items()} for m, cells in merged.items()}


def main(argv: list[str]) -> None:
    assert len(argv) >= 3, (
        "usage: colosseum_pool_split_arms.py <experiment> <run_dir> [<run_dir>...]"
    )
    experiment, dirs = argv[1], [Path(d) for d in argv[2:]]

    cfg = OmegaConf.load("configs/eval/colosseum_jira.yaml")
    cfg = OmegaConf.merge(cfg, OmegaConf.create({"experiment": experiment}))

    by_target: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    modes: dict[str, set[str]] = defaultdict(set)
    for d in dirs:
        meta, measures = _read(d)
        by_target[meta["target"]].append((d, measures))
        modes[meta["target"]].add(meta.get("mode", "think"))

    runs = []
    print(
        f">>> pooling {experiment} over {len(by_target)} arms "
        f"from {len(dirs)} run directories"
    )
    for target, parts in by_target.items():
        assert len(modes[target]) == 1, (
            f"{target} was served in modes {sorted(modes[target])}; its shards are not "
            "comparable and must not be merged"
        )
        merged = _merge(target, parts)
        counts = {c: len(s) for c, s in next(iter(merged.values())).items()}
        print(f"      {target}")
        print(f"        {len(parts)} shard(s) -> seeds per cell: {counts}")

        # A real directory, so the pooled result points at inputs that still exist.
        key = local_name(f"{experiment} merged {target.split('/')[-1]}"[:80])
        merged_dir = MERGED_ROOT / key
        (merged_dir / "results").mkdir(parents=True, exist_ok=True)
        (merged_dir / "metadata").mkdir(parents=True, exist_ok=True)
        (merged_dir / "results" / "per_seed.json").write_text(
            json.dumps(merged, indent=2)
        )
        (merged_dir / "metadata" / "run_meta.json").write_text(
            json.dumps(
                {
                    "target": target,
                    "mode": next(iter(modes[target])),
                    "merged_from": [str(p) for p, _ in parts],
                },
                indent=2,
            )
        )
        runs.append(
            {
                "target": target,
                "model_key": target.split("/")[-1],
                "mode": next(iter(modes[target])),
                "out_dir": str(merged_dir),
                "repo": "",
            }
        )

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
