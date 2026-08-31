# ABOUTME: Build a mixture whose 716 difficult-advice rows come ONLY from a chosen subset of
# ABOUTME: traits, combined with the same 9,284 standard SFT rows every sibling arm uses.

"""A trait-restricted arm: full difficult-advice volume, only a chosen subset of the traits.

The deletion arms removed traits and either backfilled (traits 1/3/4) or shrank the mixture
(traits 1/3). This goes the other way: it keeps the full 716-row difficult-advice budget but
draws every row from the traits named in `traits`, split as evenly as the corpus allows.

Rows are drawn from the whole stage-7 corpus — both the rows already in the control mixture
and the unused pool — so an arm is not restricted to what one earlier sample happened to
contain. The 9,284 standard rows are byte-identical to every sibling's, taken from the arms'
own mixture; the published `2026-08-04` build predates the preserve-thinking policy and its
multi-turn rows fail the training mask gate.

Selection is seeded and deterministic, so a given (traits, seed) pair reproduces exactly.

Run:
  uv run python scratch/build_trait_subset_mixture.py --traits t5,t6,t7
  uv run python scratch/build_trait_subset_mixture.py --traits t1,t2,t3 --out_dir output/traits123
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


def _render(row: dict, profile) -> str:
    """Render one stage-7 SFT record into the mixture's pre-rendered chat format.

    Args:
        row: A stage_7 SFT record with system/user/assistant messages.
        profile: The family's ModelProfile, for the think-tag literals.

    Returns:
        The rendered text, matching the mixture's existing difficult-advice rows in shape.
    """
    m = {x["role"]: x for x in row["messages"]}
    return (f"<|im_start|>system\n{m['system']['content']}<|im_end|>\n"
            f"<|im_start|>user\n{m['user']['content']}<|im_end|>\n"
            f"<|im_start|>assistant\n{profile.prefill}"
            f"{m['assistant']['reasoning_content']}\n</think>\n\n"
            f"{m['assistant']['content']}<|im_end|>\n")


def main(traits: str = "t5,t6,t7", n: int = 716,
         mixture: str = "data/hf/2026-08-06-table2-9284-synthdoc-716-train/mixture_think.jsonl",
         sft: str = "output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl",
         out_dir: str = "output/trait_subset", family: str = "Qwen3.6", seed: int = 0) -> None:
    """Build a trait-subset mixture and verify it.

    Args:
        traits: Comma-separated trait ids the difficult-advice rows may come from.
        n: Difficult-advice rows to include.
        mixture: Control mixture, source of the 9,284 standard rows.
        sft: stage_7 SFT file, source of the difficult-advice rows and trait labels.
        out_dir: Output directory root.
        family: ModelProfile family.
        seed: RNG seed for sampling.

    Raises:
        RuntimeError: If a kept trait is short of its quota, a foreign trait leaks in, the
            standard rows are not byte-identical, or any assistant turn lacks a think block.
    """
    # Fire parses a comma-separated CLI value into a tuple already, so accept either that or
    # a plain string; str().split(",") on the tuple would yield "('t5'" and friends.
    if isinstance(traits, str):
        KEEP_TRAITS = tuple(x.strip() for x in traits.split(",") if x.strip())
    else:
        KEEP_TRAITS = tuple(str(x).strip() for x in traits)
    profile = model_profile(family)
    rng = random.Random(seed)

    records = [json.loads(x) for x in Path(sft).read_text().splitlines() if x.strip()]
    pool: dict[str, list] = {t: [] for t in KEEP_TRAITS}
    for r in records:
        t = r["metadata"]["trait_id"]
        if t in pool:
            pool[t].append(r)
    for t in pool:
        pool[t].sort(key=lambda r: r["metadata"]["scenario_id"])
        rng.shuffle(pool[t])
    print(f"available: " + ", ".join(f"{t}={len(pool[t])}" for t in KEEP_TRAITS)
          + f"  (total {sum(len(v) for v in pool.values())})")

    per, extra = divmod(n, len(KEEP_TRAITS))
    quota = {t: per + (1 if i < extra else 0) for i, t in enumerate(KEEP_TRAITS)}
    for t, q in quota.items():
        if q > len(pool[t]):
            raise RuntimeError(f"trait {t}: need {q} rows, corpus holds {len(pool[t])}")
    picked = [r for t in KEEP_TRAITS for r in pool[t][:quota[t]]]
    rng.shuffle(picked)
    print(f"drawing {quota} = {len(picked)} difficult-advice rows")

    control = [json.loads(x) for x in Path(mixture).read_text().splitlines() if x.strip()]
    standard = [r for r in control if r.get("source") != SOURCE]
    # No `supervise` key: the 9,284 standard rows do not carry one, and a column that appears
    # only partway through the file makes datasets raise CastError when it infers the schema
    # from the first chunk. "all" is the default in build_labels anyway, so the key is
    # redundant as well as harmful here.
    added = [{"text": _render(r, profile), "source": SOURCE} for r in picked]
    # Interleave rather than append: the trainer shuffles, but a uniform schema in the first
    # chunk is what datasets needs, and this also keeps the file diffable against siblings.
    rows = standard + added
    rng.shuffle(rows)

    # Order-independent: the rows are shuffled, so compare as multisets.
    from collections import Counter as _C
    if _C(r["text"] for r in rows if r.get("source") != SOURCE) != _C(r["text"] for r in standard):
        raise RuntimeError("standard SFT rows were altered")
    if len(rows) != len(standard) + n:
        raise RuntimeError(f"expected {len(standard) + n} rows, got {len(rows)}")
    by_user = {}
    for r in records:
        by_user[{m["role"]: m for m in r["messages"]}["user"]["content"].strip()] = r["metadata"]
    final = Counter(by_user[USER_RE.search(r["text"]).group(1).strip()]["trait_id"]
                    for r in added)
    foreign = {t: c for t, c in final.items() if t not in KEEP_TRAITS}
    if foreign:
        raise RuntimeError(f"foreign traits leaked in: {foreign}")
    if final != Counter(quota):
        raise RuntimeError(f"trait counts {dict(final)} != quota {quota}")
    empty = profile.empty_think
    if any(empty in r["text"] for r in added):
        raise RuntimeError("a difficult-advice row carries an EMPTY think marker")
    turns = absent = 0
    for r in rows:
        for seg in r["text"].split("<|im_start|>assistant\n")[1:]:
            turns += 1
            if "<think>" not in seg.split("<|im_end|>")[0]:
                absent += 1
    if absent:
        raise RuntimeError(f"{absent}/{turns} assistant turns carry NO think block")

    d = Path(out_dir) / timestamp()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"mixture_think_{''.join(KEEP_TRAITS)}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    stats = {
        "rows_total": len(rows), "difficult_advice_rows": len(added),
        "kept_traits": list(KEEP_TRAITS), "quota": quota,
        "trait_after": dict(sorted(final.items())),
        "pool_available": {t: len(pool[t]) for t in KEEP_TRAITS},
        "scenario_ids": [r["metadata"]["scenario_id"] for r in picked],
        "assistant_turns_checked": turns, "seed": seed, "source_mixture": mixture,
        "git_sha": git_sha(), "timestamp_utc": timestamp(),
    }
    (d / f"mixture_stats_{''.join(KEEP_TRAITS)}.json").write_text(json.dumps(stats, indent=2))

    print(f"\n{len(rows)} rows -> {path}")
    print(f"trait distribution: {stats['trait_after']}")
    print(f"every assistant turn carries a think block: {turns} turns checked")
    print("standard SFT rows byte-identical to control: yes")


if __name__ == "__main__":
    fire.Fire(main)
