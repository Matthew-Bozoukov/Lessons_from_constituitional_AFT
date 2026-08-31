# ABOUTME: Swap every trait-1/3/4 difficult-advice row out of the mixture for an equal
# ABOUTME: number of unused rows drawn evenly from the six remaining traits.

"""A trait ablation: drop the oversight/honesty/harm traits, hold everything else fixed.

The low-ODCV arm raised the ODCV misalignment rate, but its scenario swap was inseparable
from trait coverage — trait 1 IS the agentic-oversight trait, so "removed ODCV-like
scenarios" and "removed trait 1" were the same manipulation. This arm isolates the trait
side of that pair directly: it removes traits 1, 3 and 4 outright and backfills from the
other six traits, keeping the row count, the source composition and the 9,284 standard SFT
rows byte-identical to the control.

  t1 Preserve human oversight; avoid unilateral, power-accruing action
  t3 Be scrupulously honest and non-deceptive, in word, framing, and action
  t4 Weigh real-world harm with calibrated, policy-level judgement

The reasoning text is untouched in both kept and swapped-in rows, so the arm differs from
the control only in which traits the difficult-advice share teaches.

Run:
  uv run python scratch/build_trait_swap_mixture.py
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
DROP_TRAITS = ("t1", "t3", "t4")


def _render(row: dict, profile) -> str:
    """Render one SFT record into the mixture's pre-rendered chat format.

    Args:
        row: A stage_7 SFT record with system/user/assistant messages.
        profile: The family's ModelProfile, for the think-tag literals.

    Returns:
        The rendered text, matching the mixture's existing rows byte for byte in shape.
    """
    m = {x["role"]: x for x in row["messages"]}
    return (f"<|im_start|>system\n{m['system']['content']}<|im_end|>\n"
            f"<|im_start|>user\n{m['user']['content']}<|im_end|>\n"
            f"<|im_start|>assistant\n{profile.prefill}"
            f"{m['assistant']['reasoning_content']}\n</think>\n\n"
            f"{m['assistant']['content']}<|im_end|>\n")


def main(
    mixture: str = "data/hf/2026-08-06-table2-9284-synthdoc-716-train/mixture_think.jsonl",
    sft: str = "output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl",
    out_dir: str = "output/trait_swap",
    family: str = "Qwen3.6",
    seed: int = 0,
) -> None:
    """Build the traits-1/3/4-removed mixture and verify the swap is exact.

    Args:
        mixture: Control mixture to rebuild from.
        sft: stage_7 SFT file, source of both the trait labels and the replacement rows.
        out_dir: Output directory root; a timestamped run dir is created under it.
        family: ModelProfile family, for the think-tag literals.
        seed: RNG seed for replacement sampling.

    Raises:
        RuntimeError: If the swap does not preserve the row count or source composition,
            or if any dropped trait survives into the output.
    """
    profile = model_profile(family)
    rng = random.Random(seed)

    records = [json.loads(x) for x in Path(sft).read_text().splitlines() if x.strip()]
    by_user = {}
    by_id = {}
    for r in records:
        u = {m["role"]: m for m in r["messages"]}["user"]["content"].strip()
        by_user[u] = r["metadata"]
        by_id[r["metadata"]["scenario_id"]] = r

    control = [json.loads(x) for x in Path(mixture).read_text().splitlines() if x.strip()]
    used_ids = {by_user[USER_RE.search(m["text"]).group(1).strip()]["scenario_id"]
                for m in control if m.get("source") == SOURCE}

    # Replacement pool: unused rows, grouped by the traits we are keeping.
    keep_traits = sorted({r["metadata"]["trait_id"] for r in records} - set(DROP_TRAITS))
    pool: dict[str, list] = {t: [] for t in keep_traits}
    for r in records:
        md = r["metadata"]
        if md["scenario_id"] not in used_ids and md["trait_id"] in pool:
            pool[md["trait_id"]].append(r)
    for t in pool:
        pool[t].sort(key=lambda r: r["metadata"]["scenario_id"])
        rng.shuffle(pool[t])

    n_drop = sum(1 for m in control if m.get("source") == SOURCE
                 and by_user[USER_RE.search(m["text"]).group(1).strip()]["trait_id"]
                 in DROP_TRAITS)
    per_trait, extra = divmod(n_drop, len(keep_traits))
    quota = {t: per_trait + (1 if i < extra else 0) for i, t in enumerate(keep_traits)}
    for t, q in quota.items():
        if q > len(pool[t]):
            raise RuntimeError(f"trait {t}: need {q} replacements, pool holds {len(pool[t])}")
    replacements = [r for t in keep_traits for r in pool[t][:quota[t]]]
    rng.shuffle(replacements)
    print(f"dropping {n_drop} rows across {DROP_TRAITS}; "
          f"backfilling {quota} from the unused pool")

    out_rows, added, removed_ids, added_ids = [], iter(replacements), [], []
    for m in control:
        if m.get("source") == SOURCE:
            md = by_user[USER_RE.search(m["text"]).group(1).strip()]
            if md["trait_id"] in DROP_TRAITS:
                new = next(added)
                removed_ids.append(md["scenario_id"])
                added_ids.append(new["metadata"]["scenario_id"])
                m = {"text": _render(new, profile), "source": SOURCE,
                     "supervise": m.get("supervise") or "all"}
        out_rows.append(m)

    if len(out_rows) != len(control):
        raise RuntimeError(f"row count changed: {len(control)} -> {len(out_rows)}")
    if Counter(r["source"] for r in out_rows) != Counter(r["source"] for r in control):
        raise RuntimeError("source composition changed")
    if len(removed_ids) != n_drop or len(set(added_ids)) != n_drop:
        raise RuntimeError(f"swap mismatch: removed {len(removed_ids)}, "
                           f"added {len(set(added_ids))}, expected {n_drop}")

    # The dropped traits must be gone, and the standard SFT share untouched.
    final = Counter()
    for m in out_rows:
        if m.get("source") == SOURCE:
            final[by_user[USER_RE.search(m["text"]).group(1).strip()]["trait_id"]] += 1
    leaked = {t: n for t, n in final.items() if t in DROP_TRAITS}
    if leaked:
        raise RuntimeError(f"dropped traits survived: {leaked}")
    ctl_std = [r["text"] for r in control if r.get("source") != SOURCE]
    new_std = [r["text"] for r in out_rows if r.get("source") != SOURCE]
    if ctl_std != new_std:
        raise RuntimeError("standard SFT rows are not byte-identical to the control")

    d = Path(out_dir) / timestamp()
    d.mkdir(parents=True, exist_ok=True)
    path = d / "mixture_think_traitswap.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in out_rows))

    before = Counter(by_user[USER_RE.search(m["text"]).group(1).strip()]["trait_id"]
                     for m in control if m.get("source") == SOURCE)
    stats = {
        "rows_total": len(out_rows), "rows_swapped": n_drop,
        "dropped_traits": list(DROP_TRAITS), "backfill_quota": quota,
        "trait_before": dict(sorted(before.items())),
        "trait_after": dict(sorted(final.items())),
        "removed_ids": removed_ids, "added_ids": added_ids,
        "seed": seed, "source_mixture": mixture,
        "git_sha": git_sha(), "timestamp_utc": timestamp(),
    }
    (d / "mixture_stats_traitswap.json").write_text(json.dumps(stats, indent=2))

    print(f"\n{len(out_rows)} rows, {n_drop} swapped -> {path}")
    print(f"trait before: {dict(sorted(before.items()))}")
    print(f"trait after : {dict(sorted(final.items()))}")
    print("standard SFT rows byte-identical to control: yes")


if __name__ == "__main__":
    fire.Fire(main)
