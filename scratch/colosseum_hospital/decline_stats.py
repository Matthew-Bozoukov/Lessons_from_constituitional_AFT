# ABOUTME: Self-sacrificial cell: does a private decline by the DA provisioner predict less damage? Permutation
# ABOUTME: test of GH failures / team total by decline status, and timing of the first decline vs the first deficit.

"""uv run python scratch/colosseum_hospital/decline_stats.py

Uses the per-iteration stances from decline_vs_action.py. For the DA block: shifts whose provisioner
wrote a moral decline at some iteration vs shifts where it never did — mean General Hospital failures
and team total, with a two-sided permutation p-value (10k shuffles). Then, for every decline shift,
whether the first decline came before, at, or after the first bottleneck-deficit broadcast from a
General Hospital department (reactive vs preventive).
"""

from __future__ import annotations

import json

import numpy as np

from decline_vs_action import (
    CELLS,
    PROV,
    ROOT,
    deficit_broadcast,
    iterations,
    load_episode,
    reasoning_stance,
)


def perm_p(a: np.ndarray, b: np.ndarray, n: int = 10000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b])
    k = len(a)
    cnt = 0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(pool[:k].mean() - pool[k:].mean()) >= abs(obs):
            cnt += 1
    return (cnt + 1) / (n + 1)


def main() -> None:
    for block in ("DA", "control"):
        cell = ROOT / CELLS[block]
        rows = []
        for ep in sorted(
            cell.glob("rollouts/colosseum/*/runs/*/*/*"),
            key=lambda p: int(p.name.rsplit("seed", 1)[-1]),
        ):
            if not (ep / "agent_turns.json").is_file():
                continue
            seed = int(ep.name.rsplit("seed", 1)[-1])
            turns, events, final, rewards = load_episode(ep)
            its = iterations(turns)
            first_d = next(
                (it for it in its if reasoning_stance(turns, PROV, it)[0] == "D"), None
            )
            first_def = next(
                (it for it in its if deficit_broadcast(events, "General_Hospital", it)),
                None,
            )
            gh = sum(
                v
                for v in (
                    (final.get("hospital_failures") or {}).get("General_Hospital") or {}
                ).values()
            )
            rows.append((seed, first_d, first_def, gh, sum(rewards.values())))
        dec = [r for r in rows if r[1] is not None]
        non = [r for r in rows if r[1] is None]
        print(f"== {block}: provisioner declined in {len(dec)}/{len(rows)} shifts")
        if dec and non:
            for name, idx in (("GH failures", 3), ("team total", 4)):
                a = np.array([r[idx] for r in dec], float)
                b = np.array([r[idx] for r in non], float)
                print(
                    f"   {name}: declined mean {a.mean():,.1f} (n={len(a)}) vs never {b.mean():,.1f} (n={len(b)}), permutation p={perm_p(a, b):.3f}"
                )
        timing = {
            "before deficit": 0,
            "same iteration": 0,
            "after deficit": 0,
            "no GH deficit": 0,
        }
        for seed, fd, fdef, gh, total in dec:
            if fdef is None:
                timing["no GH deficit"] += 1
            elif fd < fdef:
                timing["before deficit"] += 1
            elif fd == fdef:
                timing["same iteration"] += 1
            else:
                timing["after deficit"] += 1
        print(f"   first decline vs first GH bottleneck-deficit broadcast: {timing}")
        print(
            "   per shift (seed: first decline it, first GH deficit it, GH failures): "
            + ", ".join(f"{s}:{fd}/{fdef}/{gh}" for s, fd, fdef, gh, _ in dec)
        )


if __name__ == "__main__":
    main()
