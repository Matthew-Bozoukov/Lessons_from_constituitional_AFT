# ABOUTME: The go/no-go gate on a finished Colosseum experiment — is this result trustworthy?
# ABOUTME: Run: uv run python scratch/colosseum_gate.py <experiment> <run_dir>...

"""Seven checks, applied to the data rather than to the jobs.

Deliberately not "did the SLURM job exit zero". Several jobs today exited non-zero AFTER
completing every episode and writing every result, because they died naming the next
thing; and a job can equally exit zero having produced episodes that are quietly
worthless. What matters is whether the episodes on disk support the contrast:

  1. every arm has per-seed measures at all
  2. both arms are present, and the control is the one the config names as `peer`
  3. the expected cells exist, with the expected seed count in each
  4. arms share their seeds, since every contrast here is paired
  5. the mixed team was real — the coalition seats held a different checkpoint from the
     peers (skipped for `cooperation`, which by design seats one arm everywhere)
  6. the tool-call path worked: episodes completing zero tasks are rare
  7. the exact optimum was solved, so normalised regret is against the optimum and not
     upstream's loose analytic bound

Prints PASS/FAIL per check and exits non-zero if any fails.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from omegaconf import OmegaConf

from src.eval.misalignment.colosseum.config import EXPERIMENTS
from src.eval.misalignment.colosseum.harvest import episode_records

ZERO_TASK_TOLERANCE = 0.05  # fraction of episodes allowed to complete no tasks


def _fail(results, name, detail):
    results.append((False, name, detail))


def _ok(results, name, detail):
    results.append((True, name, detail))


def main(argv: list[str]) -> int:
    experiment, dirs = argv[1], [Path(d) for d in argv[2:]]
    cfg = OmegaConf.load("configs/eval/colosseum_jira.yaml")
    spec = EXPERIMENTS[experiment]
    expected_cells = {name for name, _, _ in spec["cells"]}
    results: list[tuple[bool, str, str]] = []

    by_target: dict[str, list[Path]] = defaultdict(list)
    for d in dirs:
        meta = d / "metadata" / "run_meta.json"
        if not meta.is_file():
            _fail(results, "1 per-seed measures", f"{d.name}: no run_meta.json")
            continue
        by_target[json.loads(meta.read_text())["target"]].append(d)

    # 1 + 3 + 4: measures, cells, seed counts, shared seeds
    per_target_seeds: dict[str, dict[str, set]] = {}
    for target, ds in by_target.items():
        cells: dict[str, set] = defaultdict(set)
        for d in ds:
            p = d / "results" / "per_seed.json"
            if not p.is_file():
                _fail(results, "1 per-seed measures", f"{d.name}: no per_seed.json")
                continue
            meas = json.loads(p.read_text())
            # Union over MEASURES, not one hardcoded measure. `cooperation` has zero
            # colluders, so there is no coalition and coalition_advantage_mean is null —
            # harvest omits it, and keying cells off it made this check report a run with
            # perfectly good data as having no cells at all.
            for cells_of_measure in meas.values():
                for cell, seeds in cells_of_measure.items():
                    cells[cell] |= {int(s) for s in seeds}
        per_target_seeds[target] = cells
        if set(cells) != expected_cells:
            _fail(
                results,
                "3 cells present",
                f"{target.split('/')[-1][:40]}: {sorted(cells)} != {sorted(expected_cells)}",
            )
        else:
            _ok(
                results,
                "3 cells present",
                f"{target.split('/')[-1][:40]}: "
                + ", ".join(f"{c}={len(s)}" for c, s in sorted(cells.items())),
            )
    if per_target_seeds:
        _ok(results, "1 per-seed measures", f"{len(by_target)} arm(s) readable")

    # 2: exactly two arms, control identified by the config's peer
    peer = str(cfg.peer)
    if len(by_target) != 2:
        _fail(results, "2 two arms", f"found {len(by_target)}: {sorted(by_target)}")
    elif peer not in by_target:
        _fail(results, "2 two arms", f"control ({peer}) absent")
    else:
        _ok(results, "2 two arms", f"control={peer.split('/')[-1][:34]}")

    # 4: paired seeds
    if len(per_target_seeds) == 2:
        a, b = per_target_seeds.values()
        for cell in expected_cells:
            shared = a.get(cell, set()) & b.get(cell, set())
            union = a.get(cell, set()) | b.get(cell, set())
            if not shared:
                _fail(results, "4 seeds paired", f"{cell}: arms share no seeds")
            elif len(shared) < 0.9 * len(union):
                _fail(
                    results,
                    "4 seeds paired",
                    f"{cell}: only {len(shared)}/{len(union)} seeds shared",
                )
            else:
                _ok(results, "4 seeds paired", f"{cell}: {len(shared)} shared")

    # 5 + 6 + 7: seating, tool calls, optimum — from the episodes themselves
    mixed_seen, zero_tasks, total, missing_opt = False, 0, 0, 0
    for d in dirs:
        for root in (d / "rollouts" / "colosseum").glob("*"):
            if not root.is_dir():
                continue
            for r in episode_records(root):
                total += 1
                zero_tasks += 1 if not r.get("tasks_done") else 0
                missing_opt += 1 if r.get("normalised_regret") is None else 0
                if len(set((r.get("agent_llm_labels") or {}).values())) > 1:
                    mixed_seen = True

    if not spec["mixed_team"]:
        _ok(
            results, "5 mixed team", "n/a — this experiment seats one arm in every seat"
        )
    elif mixed_seen:
        _ok(
            results,
            "5 mixed team",
            "coalition and peer seats held different checkpoints",
        )
    else:
        _fail(
            results,
            "5 mixed team",
            "NO episode seated two checkpoints — the routing patch may be missing, in "
            "which case this compares an arm against itself",
        )

    if total == 0:
        _fail(results, "6 tool calls", "no episodes found")
    elif zero_tasks > ZERO_TASK_TOLERANCE * total:
        _fail(
            results,
            "6 tool calls",
            f"{zero_tasks}/{total} episodes completed zero tasks — likely truncated "
            "reasoning traces emitting no tool call (raise max_tokens)",
        )
    else:
        _ok(results, "6 tool calls", f"{zero_tasks}/{total} episodes at zero tasks")

    if total and missing_opt:
        _fail(
            results,
            "7 exact optimum",
            f"{missing_opt}/{total} episodes have no optimal_summary.json",
        )
    elif total:
        _ok(
            results,
            "7 exact optimum",
            f"all {total} episodes scored against the optimum",
        )

    print(f"=== gate: {experiment} ({total} episodes over {len(dirs)} run dirs) ===")
    for ok, name, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name:22s} {detail}")
    failed = [r for r in results if not r[0]]
    print(
        f"\n{'GATE PASSED' if not failed else f'GATE FAILED ({len(failed)} check(s))'}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
