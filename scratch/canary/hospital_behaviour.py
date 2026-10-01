# ABOUTME: Sabotage / refusal counts for merged Hospital run dirs, plus the canary-by-sabotage split per shift.
# ABOUTME: Run: uv run python -m scratch.canary.hospital_behaviour <merged run dir> [...]
import json
import sys
from pathlib import Path

from scratch.canary.count_canary import hospital


def main() -> None:
    for d in map(Path, sys.argv[1:]):
        per_seed = json.loads((d / "results" / "per_seed.json").read_text())

        def flags(measure):
            v = per_seed[measure]
            v = v.get("self_sacrificial", v)
            return {int(k): bool(x) for k, x in v.items()}

        sab, held = flags("pair_sabotaged"), flags("refused_and_held")
        canary = {e["seed"]: e["hit_calls"] > 0 for e in hospital(d)["per_episode"]}
        split = {}
        for s, hit in canary.items():
            key = (
                f"canary={'yes' if hit else 'no'},sabotaged={'yes' if sab[s] else 'no'}"
            )
            split[key] = split.get(key, 0) + 1
        print(
            d.name,
            json.dumps(
                {
                    "sabotaged": f"{sum(sab.values())}/{len(sab)}",
                    "refused_and_held": f"{sum(held.values())}/{len(held)}",
                    **split,
                }
            ),
        )


if __name__ == "__main__":
    main()
