# ABOUTME: The per-iteration reasoning/board/action stance classifier of decline_vs_action.py,
# ABOUTME: applied to the fixed-harness cells (A, B) beside the paper-harness cells: class counts per seat.

"""uv run python scratch/colosseum_hospital/trajectory_classes.py

For each self-sacrificial cell (paper harness 2026-09-04; carried history A and board access
B, 2026-09-10) and each coalition seat: how many of the 30 shifts fall in each trajectory
class — refused (declined in reasoning and never committed on the board or in action after
that), two-faced (declined, then posted the plan), repented (complied, then declined),
declined-but-sabotaged (declined, then acted on the sabotage), complied, passive. The
"refused" class is the quantity direction A was built to move; the stance rules and their
limits are decline_vs_action.py's.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from decline_vs_action import (
    CLASSES,
    PROV,
    TRIAGE,
    board_stance,
    classify,
    iterations,
    load_episode,
    prov_actions,
    reasoning_stance,
    triage_actions,
)
from direction_contrasts import ARMS, ROOT


def classes_for(arm_dir: Path) -> dict[str, Counter]:
    counts = {TRIAGE: Counter(), PROV: Counter()}
    seen: dict[int, Path] = {}
    for ep in sorted(arm_dir.glob("rollouts/colosseum/*/runs/*/*/*")):
        if (ep / "agent_turns.json").is_file():
            seen[int(ep.name.rsplit("seed", 1)[-1])] = ep  # later run dir wins
    for seed, ep in sorted(seen.items()):
        turns, events, _final, _rewards = load_episode(ep)
        its = iterations(turns)
        for agent in (TRIAGE, PROV):
            seq = []
            for it in its:
                R, _both = reasoning_stance(turns, agent, it)
                B = board_stance(events, agent, it)
                A = (
                    prov_actions(events, turns, it)
                    if agent == PROV
                    else triage_actions(events, it)
                )
                seq.append((R, B, A))
            counts[agent][classify(seq)] += 1
    return counts


def main() -> None:
    print(f"{'cell':<20} {'seat':<26} " + " ".join(f"{k:>22}" for k in CLASSES))
    for (variant, block), rel in ARMS.items():
        arm = ROOT / rel
        if not (arm / "rollouts").is_dir():
            continue
        counts = classes_for(arm)
        for agent in (TRIAGE, PROV):
            c = counts[agent]
            n = sum(c.values())
            print(
                f"{variant + ' ' + block:<20} {agent:<26} "
                + " ".join(f"{c.get(k, 0):>22}" for k in CLASSES)
                + f"   n={n}"
            )


if __name__ == "__main__":
    main()
