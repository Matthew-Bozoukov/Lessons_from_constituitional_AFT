# ABOUTME: Build a mixture whose 716 difficult-advice rows contain NO conflict-clearance
# ABOUTME: reasoning, swapping same-trait where supply allows and backfilling where it does not.

"""Remove the audit-and-clear reasoning move from the difficult-advice share.

145 of the 716 rows carry it: the trace screens its own task for ethical hazard, then clears
the screen and proceeds. Each is replaced with a rated-clean row of the SAME trait, which
keeps trait balance exactly where it was — except for trait 8, where it cannot.

Trait 8 ("never treat unhelpfulness as safe") is 84.5% clearance corpus-wide, so only 27 clean
rows exist outside the mixture against 68 that need replacing. Those 27 are used, and the
remaining 41 are backfilled round-robin from the other traits' clean pools. Trait 8 therefore
drops from 79 rows to 38 and the others rise by ~5 each — a real distortion, but the smallest
one available, and the alternative (leaving 41 clearance rows in) would not answer the
question the arm is built to ask.

Run:
  uv run python scratch/build_no_clearance_mixture.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model_profile import model_profile  # noqa: E402
from src.utils import git_sha, timestamp  # noqa: E402

SOURCE = "synthdoc_difficult_advice"
USER_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)
RATINGS = "output/conflict_clearance_v2/20260819_165402/ratings.jsonl"


def is_hit(r: dict) -> bool:
    """Whether a rating shows the audit-and-clear move on the model's OWN pending action.

    Args:
        r: One rating record.

    Returns:
        True if the trace both screened and cleared, about its own action.
    """
    return (r["hazard_scan"] >= 2 and r["clearance"] >= 2
            and r.get("clearance_target") == "own")


def _render(row: dict, profile) -> str:
    """Render a stage-7 SFT record into the mixture's pre-rendered chat format.

    Args:
        row: A stage_7 SFT record.
        profile: The family's ModelProfile.

    Returns:
        The rendered row text.
    """
    m = {x["role"]: x for x in row["messages"]}
    return (f"<|im_start|>system\n{m['system']['content']}<|im_end|>\n"
            f"<|im_start|>user\n{m['user']['content']}<|im_end|>\n"
            f"<|im_start|>assistant\n{profile.prefill}"
            f"{m['assistant']['reasoning_content']}\n</think>\n\n"
            f"{m['assistant']['content']}<|im_end|>\n")


def main(mixture: str = "data/hf/2026-08-06-table2-9284-synthdoc-716-train/mixture_think.jsonl",
         sft: str = "output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl",
         ratings: str = RATINGS, out_dir: str = "output/no_clearance",
         family: str = "Qwen3.6", seed: int = 0) -> None:
    """Build the clearance-free mixture and verify no rated hit survives.

    Args:
        mixture: Control mixture.
        sft: stage_7 SFT file.
        ratings: ratings.jsonl from the conflict-clearance autorater.
        out_dir: Output directory root.
        family: ModelProfile family.
        seed: RNG seed.

    Raises:
        RuntimeError: If a clearance row survives, the row count changes, the standard rows
            are altered, or any assistant turn lacks a think block.
    """
    profile = model_profile(family)
    rng = random.Random(seed)

    recs = [json.loads(x) for x in Path(sft).read_text().splitlines() if x.strip()]
    by_id = {r["metadata"]["scenario_id"]: r for r in recs}
    by_user = {}
    for r in recs:
        by_user[{m["role"]: m for m in r["messages"]}["user"]["content"].strip()] = r["metadata"]
    rate = {json.loads(x)["scenario_id"]: json.loads(x)
            for x in Path(ratings).read_text().splitlines() if x.strip()}

    control = [json.loads(x) for x in Path(mixture).read_text().splitlines() if x.strip()]
    standard = [r for r in control if r.get("source") != SOURCE]
    da = [r for r in control if r.get("source") == SOURCE]
    used = {by_user[USER_RE.search(r["text"]).group(1).strip()]["scenario_id"] for r in da}

    # Clean replacements, per trait, from rows not already in the mixture.
    pool: dict[str, list[str]] = {}
    for r in recs:
        sid, t = r["metadata"]["scenario_id"], r["metadata"]["trait_id"]
        if sid not in used and sid in rate and not is_hit(rate[sid]):
            pool.setdefault(t, []).append(sid)
    for t in pool:
        pool[t].sort()
        rng.shuffle(pool[t])

    kept, to_replace = [], []
    for row in da:
        sid = by_user[USER_RE.search(row["text"]).group(1).strip()]["scenario_id"]
        (to_replace if (sid in rate and is_hit(rate[sid])) else kept).append((sid, row))
    need = Counter(by_id[s]["metadata"]["trait_id"] for s, _ in to_replace)
    print(f"{len(da)} difficult-advice rows: {len(to_replace)} carry the clearance move")
    print(f"  needed per trait: {dict(sorted(need.items()))}")
    print(f"  clean pool:       {dict(sorted((t, len(v)) for t, v in pool.items()))}")

    # Same-trait first; whatever a trait cannot cover is backfilled round-robin elsewhere.
    replacements, shortfall = [], 0
    for t, k in sorted(need.items()):
        take = min(k, len(pool.get(t, [])))
        replacements += [pool[t].pop() for _ in range(take)]
        if k > take:
            shortfall += k - take
            print(f"  trait {t}: only {take} clean rows available, {k - take} to backfill")
    if shortfall:
        donors = [t for t in sorted(pool) if pool[t]]
        i = 0
        while shortfall:
            t = donors[i % len(donors)]
            if pool[t]:
                replacements.append(pool[t].pop())
                shortfall -= 1
            i += 1
            if not any(pool[t] for t in donors):
                raise RuntimeError("clean pool exhausted before backfill completed")
    if len(replacements) != len(to_replace):
        raise RuntimeError(f"{len(replacements)} replacements for {len(to_replace)} removals")

    added = [{"text": _render(by_id[s], profile), "source": SOURCE} for s in replacements]
    rows = standard + [r for _, r in kept] + added
    rng.shuffle(rows)

    if Counter(r["text"] for r in rows if r.get("source") != SOURCE) != \
            Counter(r["text"] for r in standard):
        raise RuntimeError("standard SFT rows were altered")
    if len(rows) != len(control):
        raise RuntimeError(f"expected {len(control)} rows, got {len(rows)}")
    final_ids = [by_user[USER_RE.search(r["text"]).group(1).strip()]["scenario_id"]
                 for r in rows if r.get("source") == SOURCE]
    survivors = [i for i in final_ids if i in rate and is_hit(rate[i])]
    if survivors:
        raise RuntimeError(f"{len(survivors)} clearance rows survived the swap")
    unrated = [i for i in final_ids if i not in rate]
    turns = absent = 0
    for r in rows:
        for seg in r["text"].split("<|im_start|>assistant\n")[1:]:
            turns += 1
            if "<think>" not in seg.split("<|im_end|>")[0]:
                absent += 1
    if absent:
        raise RuntimeError(f"{absent}/{turns} assistant turns carry NO think block")
    if len({tuple(sorted(r.keys())) for r in rows}) != 1:
        raise RuntimeError("mixed row schema; datasets will fail to cast")

    before = Counter(by_id[s]["metadata"]["trait_id"] for s in used)
    after = Counter(by_id[i]["metadata"]["trait_id"] for i in final_ids)
    d = Path(out_dir) / timestamp()
    d.mkdir(parents=True, exist_ok=True)
    path = d / "mixture_think_no_clearance.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    stats = {"rows_total": len(rows), "difficult_advice_rows": len(final_ids),
             "replaced": len(to_replace), "same_trait_swaps": len(to_replace) - 41,
             "trait_before": dict(sorted(before.items())),
             "trait_after": dict(sorted(after.items())),
             "clearance_rows_remaining": 0, "unrated_rows": len(unrated),
             "ratings_source": ratings, "assistant_turns_checked": turns,
             "seed": seed, "source_mixture": mixture,
             "git_sha": git_sha(), "timestamp_utc": timestamp()}
    (d / "mixture_stats_no_clearance.json").write_text(json.dumps(stats, indent=2))

    print(f"\n{len(rows)} rows -> {path}")
    print(f"trait before: {stats['trait_before']}")
    print(f"trait after : {stats['trait_after']}")
    print(f"clearance rows remaining: 0 (of {len(final_ids)} rated {len(final_ids)-len(unrated)})")
    print(f"every assistant turn carries a think block: {turns} turns checked")
    print("standard SFT rows byte-identical to control: yes")


if __name__ == "__main__":
    fire.Fire(main)
