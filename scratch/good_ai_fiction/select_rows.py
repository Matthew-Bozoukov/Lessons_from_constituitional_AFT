# ABOUTME: Pick the corpus out of the candidate pool: coverage first, then a token-budget
# ABOUTME: repair that walks the trainable-token total onto the DA-716 target exactly.
"""Run: PYTHONPATH=. python scratch/good_ai_fiction/select_rows.py --run <run dir> --n 16

Two jobs that have to be one pass, because doing either alone breaks the other:

  COVERAGE. A random 16 of 24 leaves whole clusters and stakes bands unrepresented, and a
  random 716 of a bigger pool drifts off the quotas the recipe declares. Selection is
  greedy max-coverage over the axes named in `--fields`: at each step take the candidate
  that adds the most cells nobody has yet.

  TOKENS. The arm is only comparable to difficult advice if its trainable-token total
  matches. Generation puts the DISTRIBUTION in the right place (the length bands); this
  puts the TOTAL on the number, by swapping rows for same-cell alternatives that move the
  running sum toward the target. Matching by selection rather than by generation is
  deliberate: a generator pushed to hit a token total writes to length, and length-padded
  reasoning is a different intervention from the one being tested.

Quotas (`--quotas '{"trait_id": {"t1": 143, ...}}'`) constrain phase 1 and confine every
later swap to within a bucket, so the token repair can never eat the composition.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import fire

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# The number FICTION-716 has to land on: the difficult-advice slice's trainable tokens,
# measured with the trainer's own mask (scratch/good_ai_fiction/da_baseline.json).
DA716_TRAINABLE = 832_064
COVER_FIELDS = ("trait_id", "world", "stakes", "source_type", "narrative_form",
                "length_band")


def accepted(rec: dict, strict: bool = True) -> bool:
    """Whether a measured row cleared the pipeline's own gates.

    `revise_status: ok` means the rewrite satisfied the lint and both judges. An
    `exhausted` row is the honest-failure path -- it is kept in the run so the failure
    rate stays countable, and it must not be selected into the corpus.
    """
    if rec.get("revise_status") not in ("ok", ""):
        return False
    if strict and rec.get("judge_persona") not in ("accept", ""):
        return False
    if strict and rec.get("judge_pattern") not in ("accept", ""):
        return False
    return True


def _new_cover(rec: dict, seen: dict[str, Counter], fields: tuple[str, ...]) -> int:
    """How many axes this row is the FIRST to cover a value on. Wins absolutely."""
    return sum(1 for f in fields if seen[f][str(rec.get(f, ""))] == 0)


def _load(rec: dict, seen: dict[str, Counter], fields: tuple[str, ...]) -> int:
    """How crowded this row's values already are; lower spreads the selection wider.

    Only a tiebreak, and deliberately ranked BELOW the token budget: once every value has
    been seen once, `_new_cover` goes flat and something has to order the rest. That
    something should be the budget, because the budget is a hard experimental constraint
    and the spread is a preference.
    """
    return sum(seen[f][str(rec.get(f, ""))] for f in fields)


def _cell(rec: dict, fields: tuple[str, ...]) -> tuple:
    return tuple(str(rec.get(f, "")) for f in fields)


def cover(pool: list[dict], n: int, fields: tuple[str, ...],
          seen: dict[str, Counter] | None = None, budget: int = 0) -> list[dict]:
    """Greedy max-coverage pick of n rows, breaking ties toward a token budget.

    The budget tie-break is what makes the token match achievable at all. The same-cell
    repair below cannot do it alone: at 860 candidates over a cell space of 12 worlds x 4
    stakes x 2 sources x 7 forms x 3 bands x 9 units, almost every row is the only
    occupant of its cell, so there is nothing to swap it for. Steering DURING selection
    uses the corpus's natural length spread instead of trading composition for tokens --
    coverage still wins outright, and the budget only decides between rows that buy the
    same coverage.

    Args:
        pool: Candidates, each carrying `trainable`.
        n: How many to pick.
        fields: Coverage axes.
        seen: Running per-axis counts, shared across quota buckets.
        budget: Total trainable tokens this pick should land on. 0 disables the
            tie-break, leaving the original pure-coverage behaviour.
    """
    seen = seen if seen is not None else {f: Counter() for f in fields}
    remaining = list(pool)
    picked: list[dict] = []
    spent = 0
    while remaining and len(picked) < n:
        # What the AVERAGE remaining row has to weigh to finish on budget. Recomputed
        # every pick, so an unavoidably long row early is paid for by later ones.
        need = (budget - spent) / max(n - len(picked), 1) if budget else None
        # Order: NEW coverage first and absolutely, then the token budget, then spare
        # capacity, then id. The spread term used to sit ahead of the budget, which made
        # the budget decorative -- with five axes an exact tie on spare capacity is rare,
        # so the token term almost never got to decide anything. Measured on the 760-row
        # pool: budget-behind-spread landed 48% of the way up the achievable range;
        # budget-ahead lands near the top of it.
        remaining.sort(
            key=lambda r: (_new_cover(r, seen, fields),
                           -abs(r["trainable"] - need) if need is not None else 0,
                           -_load(r, seen, fields),
                           r["scenario_id"]),
            reverse=True)
        best = remaining.pop(0)
        picked.append(best)
        spent += best["trainable"]
        for f in fields:
            seen[f][str(best.get(f, ""))] += 1
    return picked


def token_repair(picked: list[dict], pool: list[dict], target: int,
                 fields: tuple[str, ...], rounds: int = 400) -> tuple[list[dict], int]:
    """Swap picked rows for unpicked ones with the SAME cell, to close the token gap.

    Same-cell only, so the composition the coverage pass produced is invariant under the
    repair -- the swap cannot trade a mundane oversight row for a speculative one to save
    forty tokens. Returns the (possibly improved) selection and the residual gap.
    """
    by_cell: dict[tuple, list[dict]] = {}
    chosen = {r["scenario_id"] for r in picked}
    for r in pool:
        if r["scenario_id"] not in chosen:
            by_cell.setdefault(_cell(r, fields), []).append(r)

    total = sum(r["trainable"] for r in picked)
    for _ in range(rounds):
        gap = target - total
        if gap == 0:
            break
        best = None
        for i, cur in enumerate(picked):
            for cand in by_cell.get(_cell(cur, fields), []):
                delta = cand["trainable"] - cur["trainable"]
                after = abs(gap - delta)
                if after < abs(gap) and (best is None or after < best[0]):
                    best = (after, i, cand)
        if best is None:
            break
        _after, i, cand = best
        out = picked[i]
        by_cell.setdefault(_cell(out, fields), []).append(out)
        by_cell[_cell(cand, fields)].remove(cand)
        total += cand["trainable"] - out["trainable"]
        picked[i] = cand
    return picked, target - total


def quota_from_config(config: str, n: int) -> dict:
    """Per-unit quotas for `n` rows, derived from the config's own `trait_weights`.

    Derived, never typed: the quota has to be the same apportionment the generator used,
    or the corpus is selected against a target its own composition was never aimed at.
    Uses the engine's `_largest_remainder`, the same function `scenario_batches` splits
    the generation budget with, so the two agree exactly by construction.
    """
    import yaml

    from src.data.synth.stage_operators import _largest_remainder

    cfg = yaml.safe_load(Path(config).read_text(encoding="utf-8"))
    weights = {k: float(v) for k, v in cfg["trait_weights"].items()}
    counts = _largest_remainder(weights, int(n))
    assert sum(counts.values()) == n
    return {"trait_id": counts}


def main(run: str, n: int = 16, target_tokens: int = 0, quotas: str = "",
         quota_config: str = "", fields: str = ",".join(COVER_FIELDS),
         strict: bool = True, out: str = "") -> None:
    """Select a corpus out of a measured run.

    Args:
        run: Run directory holding `token_stats.json` and the export.
        n: How many rows to keep.
        target_tokens: Trainable-token total to hit. 0 scales the DA-716 total to n,
            which is what makes a pilot's token report readable against the baseline;
            pass 832064 explicitly for the real 716-row build.
        quotas: JSON `{field: {value: count}}`. Phase 1 fills each bucket exactly and
            every later swap stays inside one. Empty = pure coverage.
        quota_config: A synth config whose `trait_weights` derive the per-unit quota for
            `n` -- the same apportionment the generator used, so the corpus is never
            selected against a target its own composition was not aimed at. Mutually
            exclusive with `quotas`.
        fields: Comma-separated coverage axes.
        strict: Require both judges to have accepted. False keeps gate failures in,
            for inspecting what the gates rejected.
        out: Where to write the selection; defaults to `<run>/selected.jsonl`.
    """
    run_dir = Path(run)
    stats = json.loads((run_dir / "token_stats.json").read_text(encoding="utf-8"))
    records = stats["records"]
    axes = tuple(f.strip() for f in fields.split(",") if f.strip())

    pool = [r for r in records if accepted(r, strict)]
    dropped = len(records) - len(pool)
    assert len(pool) >= n, (
        f"only {len(pool)} of {len(records)} rows cleared the gates; cannot select {n}. "
        f"Generate more candidates rather than lowering the bar.")

    target = target_tokens or round(DA716_TRAINABLE / 716 * n)
    if quotas and quota_config:
        raise ValueError("pass `quotas` or `quota_config`, not both")
    q = json.loads(quotas) if quotas else (
        quota_from_config(quota_config, n) if quota_config else {})

    if q:
        assert len(q) == 1, "one quota field at a time; layer the rest through `fields`"
        (qfield, buckets), = q.items()
        assert qfield in axes, (
            f"the quota field {qfield!r} must also be a coverage axis, or the token "
            f"repair (which swaps within a coverage cell) could break the quota")
        assert sum(buckets.values()) == n, (
            f"quota counts sum to {sum(buckets.values())}, not n={n}")
        seen = {f: Counter() for f in axes}
        picked = []
        for value, want in sorted(buckets.items()):
            sub = [r for r in pool if str(r.get(qfield, "")) == str(value)]
            assert len(sub) >= want, (
                f"{qfield}={value}: {len(sub)} candidates cleared the gates, "
                f"quota wants {want}")
            # Each bucket carries its proportional share of the token budget, so no
            # bucket is asked to absorb another's overshoot.
            picked += cover(sub, want, axes, seen, budget=round(target * want / n))

    else:
        picked = cover(pool, n, axes, budget=target)

    # Repair over the FULL coverage cell in both modes. Pinning only the quota field
    # would let the swap trade a mundane oversight row for a speculative one to save
    # forty tokens, which is the composition the coverage pass just paid for; and since
    # the quota field is required to be one of the axes, the quota is preserved too.
    picked, gap = token_repair(picked, pool, target, axes)
    picked.sort(key=lambda r: r["scenario_id"])
    ids = [r["scenario_id"] for r in picked]

    # Pull the full rows for the selected ids out of the export, in the same order.
    export = run_dir / "dataset.jsonl"
    if not export.exists():
        export = next(run_dir.glob("stage_*_export_sft.jsonl"))
    by_id = {json.loads(line)["metadata"]["scenario_id"]: json.loads(line)
             for line in export.open(encoding="utf-8") if line.strip()}
    dest = Path(out) if out else run_dir / "selected.jsonl"
    with dest.open("w", encoding="utf-8") as fh:
        for sid in ids:
            fh.write(json.dumps(by_id[sid], ensure_ascii=False) + "\n")

    total = sum(r["trainable"] for r in picked)
    coverage = {f: dict(sorted(Counter(str(r.get(f, "")) for r in picked).items()))
                for f in axes}
    summary = {
        "run": str(run_dir), "selected": len(picked), "pool": len(pool),
        "dropped_by_gates": dropped, "strict": strict,
        "target_tokens": target, "trainable_tokens": total, "gap": gap,
        "gap_pct": round(100 * gap / max(target, 1), 3),
        "reasoning_tokens": sum(r["reasoning"] for r in picked),
        "answer_tokens": sum(r["answer"] for r in picked),
        "coverage": coverage,
        "archetypes": dict(sorted(Counter(
            r["source_archetype"] for r in picked if r["source_archetype"]).items())),
        "ids": ids,
    }
    (run_dir / "selection.json").write_text(json.dumps(summary, indent=2),
                                            encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "ids"}, indent=2))
    print(f"wrote {dest} and {run_dir / 'selection.json'}")


if __name__ == "__main__":
    fire.Fire(main)
