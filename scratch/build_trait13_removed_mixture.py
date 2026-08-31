# ABOUTME: Drop every trait-1 and trait-3 difficult-advice row from the mixture WITHOUT
# ABOUTME: backfilling, leaving a smaller mixture whose remaining traits are untouched.

"""Remove the oversight and honesty traits outright, and do not replace them.

This differs deliberately from the traits-1/3/4 arm, which swapped its 240 removed rows for an
equal number drawn from the surviving traits to hold the row count at 10,000. Here the rows are
simply deleted, so the difficult-advice share shrinks from 716 to 556 and the mixture from
10,000 to 9,840 rows.

  t1  Preserve human oversight; avoid unilateral, power-accruing action
  t3  Be scrupulously honest and non-deceptive, in word, framing, and action

Consequence worth stating next to any result: the difficult-advice share falls from 7.16% to
5.65% of the mixture, so this arm varies BOTH which traits are taught and how much
difficult-advice data there is. The 9,284 standard rows stay byte-identical.

Run:
  uv run python scratch/build_trait13_removed_mixture.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import git_sha, timestamp  # noqa: E402

SOURCE = "synthdoc_difficult_advice"
USER_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)
DROP_TRAITS = ("t1", "t3")


def main(mixture: str = "data/hf/2026-08-06-table2-9284-synthdoc-716-train/mixture_think.jsonl",
         sft: str = "output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl",
         out_dir: str = "output/trait13_removed") -> None:
    """Build the traits-1/3-removed mixture and verify nothing else changed.

    Args:
        mixture: Control mixture to filter.
        sft: stage_7 SFT file, source of the trait labels.
        out_dir: Output directory root; a timestamped run dir is created under it.

    Raises:
        RuntimeError: If a dropped trait survives, if a kept trait's count changes, or if
            the standard SFT rows are not byte-identical to the control.
    """
    by_user = {}
    for line in Path(sft).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            by_user[{m["role"]: m for m in r["messages"]}["user"]["content"].strip()] = \
                r["metadata"]

    control = [json.loads(x) for x in Path(mixture).read_text().splitlines() if x.strip()]
    before = Counter()
    out_rows, removed_ids = [], []
    for m in control:
        if m.get("source") == SOURCE:
            md = by_user[USER_RE.search(m["text"]).group(1).strip()]
            before[md["trait_id"]] += 1
            if md["trait_id"] in DROP_TRAITS:
                removed_ids.append(md["scenario_id"])
                continue
        out_rows.append(m)

    after = Counter()
    for m in out_rows:
        if m.get("source") == SOURCE:
            after[by_user[USER_RE.search(m["text"]).group(1).strip()]["trait_id"]] += 1

    leaked = {t: n for t, n in after.items() if t in DROP_TRAITS}
    if leaked:
        raise RuntimeError(f"dropped traits survived: {leaked}")
    changed = {t: (before[t], after[t]) for t in after if before[t] != after[t]}
    if changed:
        raise RuntimeError(f"kept-trait counts changed, expected untouched: {changed}")
    ctl_std = [r["text"] for r in control if r.get("source") != SOURCE]
    new_std = [r["text"] for r in out_rows if r.get("source") != SOURCE]
    if ctl_std != new_std:
        raise RuntimeError("standard SFT rows are not byte-identical to the control")
    expected = len(control) - sum(before[t] for t in DROP_TRAITS)
    if len(out_rows) != expected:
        raise RuntimeError(f"expected {expected} rows, got {len(out_rows)}")
    # Every assistant turn must carry a think block or the training mask gate refuses the run.
    turns = absent = 0
    for r in out_rows:
        for seg in r["text"].split("<|im_start|>assistant\n")[1:]:
            turns += 1
            if "<think>" not in seg.split("<|im_end|>")[0]:
                absent += 1
    if absent:
        raise RuntimeError(f"{absent}/{turns} assistant turns carry NO think block")

    d = Path(out_dir) / timestamp()
    d.mkdir(parents=True, exist_ok=True)
    path = d / "mixture_think_trait13_removed.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in out_rows))

    da = sum(after.values())
    stats = {
        "rows_total": len(out_rows), "rows_removed": len(removed_ids),
        "dropped_traits": list(DROP_TRAITS), "backfilled": False,
        "trait_before": dict(sorted(before.items())), "trait_after": dict(sorted(after.items())),
        "difficult_advice_rows": da,
        "difficult_advice_share_before": round(sum(before.values()) / len(control), 4),
        "difficult_advice_share_after": round(da / len(out_rows), 4),
        "removed_ids": removed_ids, "source_mixture": mixture,
        "assistant_turns_checked": turns,
        "git_sha": git_sha(), "timestamp_utc": timestamp(),
    }
    (d / "mixture_stats_trait13_removed.json").write_text(json.dumps(stats, indent=2))

    print(f"{len(control)} -> {len(out_rows)} rows ({len(removed_ids)} removed, no backfill)")
    print(f"trait before: {stats['trait_before']}")
    print(f"trait after : {stats['trait_after']}")
    print(f"difficult-advice share {stats['difficult_advice_share_before']:.2%} -> "
          f"{stats['difficult_advice_share_after']:.2%} ({da} rows)")
    print(f"every assistant turn carries a think block: {turns} turns checked")
    print(f"standard SFT rows byte-identical to control: yes")
    print(f"\n-> {path}")


if __name__ == "__main__":
    fire.Fire(main)
