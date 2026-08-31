# ABOUTME: Build a mixture whose difficult-advice rows are the low-stakes rewrites, spliced
# ABOUTME: into the same 9,284 standard SFT rows every sibling arm trains on.

"""The stakes ablation: same reasoning task, almost nothing riding on it.

The 716 difficult-advice rows the mixture normally carries are 96.8% serious-or-worse and
62.7% grave. Each was rewritten into an ordinary setting with the transgression scaled down to
match, and verified blind: mean stakes 2.58 -> 0.88, with 87.5% now trivial or modest.

Four rows failed to rewrite and are DROPPED rather than left at their original severity — a
handful of grave rows inside a low-stakes arm would quietly weaken the manipulation. The
mixture is therefore 9,996 rows, not 10,000, and the difficult-advice share falls from 716 to
712 (7.16% -> 7.12%), a difference too small to confound anything.

Run:
  uv run python scratch/build_low_stakes_mixture.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model_profile import model_profile  # noqa: E402
from src.utils import git_sha, timestamp  # noqa: E402

SOURCE = "synthdoc_difficult_advice"
REWRITES = "output/low_stakes_relocated/20260820_131543/rewrites.jsonl"
VERIFY = "output/low_stakes_relocated/20260820_131543/verification.json"


def _render(sysmsg: str, user: str, reasoning: str, response: str, profile) -> str:
    """Render one rewritten example into the mixture's pre-rendered chat format.

    Args:
        sysmsg: System prompt.
        user: User turn.
        reasoning: Private reasoning for the think block.
        response: Visible reply.
        profile: The family's ModelProfile, for the think-tag literals.

    Returns:
        The rendered row text.
    """
    return (f"<|im_start|>system\n{sysmsg}<|im_end|>\n"
            f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n{profile.prefill}{reasoning}\n</think>\n\n"
            f"{response}<|im_end|>\n")


def main(mixture: str = "data/hf/2026-08-06-table2-9284-synthdoc-716-train/mixture_think.jsonl",
         rewrites: str = REWRITES, verify: str = VERIFY,
         out_dir: str = "output/low_stakes_mixture", family: str = "Qwen3.6",
         seed: int = 0) -> None:
    """Build the low-stakes mixture and verify what went in.

    Args:
        mixture: Control mixture, source of the 9,284 standard rows.
        rewrites: rewrites.jsonl from the low-stakes rewrite pass.
        verify: verification.json, for the stakes distribution recorded in the stats.
        out_dir: Output directory root.
        family: ModelProfile family.
        seed: Shuffle seed.

    Raises:
        RuntimeError: If the standard rows are altered, a rewritten row is malformed, or any
            assistant turn lacks a think block.
    """
    profile = model_profile(family)
    rng = random.Random(seed)

    control = [json.loads(x) for x in Path(mixture).read_text().splitlines() if x.strip()]
    standard = [r for r in control if r.get("source") != SOURCE]
    n_da = len(control) - len(standard)
    rw = [json.loads(x) for x in Path(rewrites).read_text().splitlines() if x.strip()]
    print(f"control: {len(control)} rows ({len(standard)} standard + {n_da} difficult-advice)")
    print(f"rewrites available: {len(rw)}")

    added = [{"text": _render(r["after"]["system"], r["after"]["user"],
                              r["after"]["reasoning"], r["after"]["response"], profile),
              "source": SOURCE} for r in rw]
    rows = standard + added
    rng.shuffle(rows)

    if Counter(r["text"] for r in rows if r.get("source") != SOURCE) != \
            Counter(r["text"] for r in standard):
        raise RuntimeError("standard SFT rows were altered")
    empty = profile.empty_think
    if any(empty in r["text"] for r in added):
        raise RuntimeError("a rewritten row carries an EMPTY think marker")
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

    ver = json.loads(Path(verify).read_text())["rows"]
    dist = Counter(v["stakes_after"] for v in ver)
    d = Path(out_dir) / timestamp()
    d.mkdir(parents=True, exist_ok=True)
    path = d / "mixture_think_low_stakes.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    stats = {
        "rows_total": len(rows), "standard_rows": len(standard),
        "difficult_advice_rows": len(added), "difficult_advice_before": n_da,
        "dropped_unrewritten": n_da - len(added),
        "stakes_after_distribution": dict(sorted(dist.items())),
        "stakes_mean_before": round(sum(v["stakes_before"] for v in ver) / len(ver), 2),
        "stakes_mean_after": round(sum(v["stakes_after"] for v in ver) / len(ver), 2),
        "at_or_below_modest": sum(v["stakes_after"] <= 1 for v in ver),
        "rewrites_source": rewrites, "assistant_turns_checked": turns,
        "seed": seed, "source_mixture": mixture,
        "git_sha": git_sha(), "timestamp_utc": timestamp(),
    }
    (d / "mixture_stats_low_stakes.json").write_text(json.dumps(stats, indent=2))
    print(f"\n{len(rows)} rows -> {path}")
    print(f"difficult-advice: {n_da} -> {len(added)} ({n_da - len(added)} unrewritten, dropped)")
    print(f"stakes {stats['stakes_mean_before']} -> {stats['stakes_mean_after']}; "
          f"distribution {stats['stakes_after_distribution']}")
    print(f"every assistant turn carries a think block: {turns} turns checked")
    print("standard SFT rows byte-identical to control: yes")


if __name__ == "__main__":
    fire.Fire(main)
