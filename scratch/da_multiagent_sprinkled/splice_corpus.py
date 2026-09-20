# ABOUTME: One-off splice: the 2026-09-14 baseline DA corpus with its t1, t2, t6, t7 rows swapped for the
# ABOUTME: da-multiagent-sprinkled rows, gated by automatic data checks, published as one mixture-ready corpus.
"""Swap the edited principles' rows into the baseline difficult-advice corpus.

Run: uv run python scratch/da_multiagent_sprinkled/splice_corpus.py \
        --new <org>/<date>-da-multiagent-sprinkled-synth --new-revision <sha> [--push]

Rows of principles 3, 4, 5, 8, 9 come from the baseline (their constitution text is
byte-identical in both documents), in their original order. Rows of principles 1, 2, 6, 7
come from the new corpus: every row whose SCENARIO is about other AI agents (the keyword
rule in multiagent_rule.py, applied to situation + shortcut only), and, only where a
principle has fewer than --per-trait of those, enough of its other new rows (seeded
shuffle) to reach --per-trait. The steered generator still writes some single-agent
scenarios, mostly for principle 1; this keeps them out of the draw where it can.

The result is written to output/ and, with --push, published with a card and a default
`dataset.jsonl` config so a mixture reads it as `dataset:`.

The gate (any failure exits non-zero before anything is written or pushed):
  * every swapped principle supplies at least --per-trait rows (78: the quota of the
    700-row trait-balanced draw);
  * at least --min-multiagent of the rows supplied are about other AI agents;
  * no exported text names Claude or Anthropic;
  * no empty turn, and every row ends in an assistant turn that carries a reasoning trace;
  * no scenario id appears twice;
  * the supplied rows' median length sits within 0.5x-2x of the baseline rows they replace.
"""

import argparse
import json
import random
import re
import statistics
import sys
from collections import Counter
from datetime import date
from pathlib import Path

from src.infra.huggingface import push_files, resolve_dataset, training_data_tags
from src.naming import artifact_name
from src.utils import git_sha

from multiagent_rule import MULTI  # noqa: E402  (same directory)

BASELINE = "dougalldeepmind/2026-09-14-da-synth"
BASELINE_REVISION = "013886238fca238c4d54ace96530f444bb2b2f02"
SWAPPED = ("t1", "t2", "t6", "t7")
CONSTITUTION = (
    "constitutions/experimental/claude_distilled_09_principles_multiagent_sprinkled/"
    "constitution.md"
)
NAMES = re.compile(r"\b(claude|anthropic)\b", re.IGNORECASE)


def load(repo: str, revision: str | None) -> tuple[list[dict], dict]:
    path, ref = resolve_dataset(repo, "dataset.jsonl", revision)
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    return rows, ref


def trait(row: dict) -> str:
    return str(row["metadata"]["trait_id"])


def is_multiagent(row: dict) -> bool:
    m = row["metadata"]
    return bool(MULTI.search(f"{m.get('situation') or ''} {m.get('shortcut') or ''}"))


def text_of(row: dict) -> str:
    parts = [
        str(row["metadata"].get(k) or "") for k in ("situation", "shortcut", "domain")
    ]
    for m in row["messages"]:
        parts += [str(m.get("content") or ""), str(m.get("reasoning_content") or "")]
    return "\n".join(parts)


def words(row: dict) -> int:
    return len(text_of(row).split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--new-revision", default=None)
    ap.add_argument("--per-trait", type=int, default=78)
    ap.add_argument("--min-multiagent", type=float, default=0.85)
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    base, base_ref = load(BASELINE, BASELINE_REVISION)
    new, new_ref = load(args.new, args.new_revision)

    failures: list[str] = []
    stray = sorted({trait(r) for r in new} - set(SWAPPED))
    if stray:
        failures.append(f"new corpus has rows for principles it should not: {stray}")

    rng = random.Random(0)
    supplied: list[dict] = []
    selection = {}
    for t in SWAPPED:
        pool = [r for r in new if trait(r) == t]
        multi = [r for r in pool if is_multiagent(r)]
        rest = [r for r in pool if not is_multiagent(r)]
        rng.shuffle(rest)
        fill = rest[: max(0, args.per_trait - len(multi))]
        supplied += multi + fill
        selection[t] = {
            "generated": len(pool),
            "multiagent": len(multi),
            "single_agent_used_as_fill": len(fill),
            "single_agent_left_out": len(rest) - len(fill),
        }
        if len(multi) + len(fill) < args.per_trait:
            failures.append(f"{t}: {len(multi) + len(fill)} rows < {args.per_trait}")

    share = sum(1 for r in supplied if is_multiagent(r)) / max(len(supplied), 1)
    if share < args.min_multiagent:
        failures.append(f"only {share:.0%} of supplied rows are about other AI agents")

    named = [r["metadata"]["scenario_id"] for r in supplied if NAMES.search(text_of(r))]
    if named:
        failures.append(
            f"{len(named)} supplied rows name Claude/Anthropic: {named[:5]}"
        )

    broken = []
    for r in supplied:
        roles = [m["role"] for m in r["messages"]]
        last = r["messages"][-1]
        ok = (
            roles[-1] == "assistant"
            and "user" in roles
            and all(str(m.get("content") or "").strip() for m in r["messages"])
            and str(last.get("reasoning_content") or "").strip()
        )
        if not ok:
            broken.append(r["metadata"]["scenario_id"])
    if broken:
        failures.append(
            f"{len(broken)} supplied rows have an empty turn or no trace: {broken[:5]}"
        )

    kept = [r for r in base if trait(r) not in SWAPPED]
    spliced = kept + supplied
    ids = Counter(r["metadata"]["scenario_id"] for r in spliced)
    dupes = [k for k, v in ids.items() if v > 1]
    if dupes:
        failures.append(f"duplicate scenario ids after the splice: {dupes[:5]}")

    lengths = {}
    for t in SWAPPED:
        old = statistics.median(words(r) for r in base if trait(r) == t)
        now = statistics.median([words(r) for r in supplied if trait(r) == t] or [0])
        lengths[t] = {"baseline_median_words": old, "new_median_words": now}
        if not 0.5 * old <= now <= 2 * old:
            failures.append(f"{t}: median length {now} words vs baseline {old}")

    report = {
        "baseline": base_ref,
        "new": new_ref,
        "swapped": list(SWAPPED),
        "rows": {
            "baseline_kept": len(kept),
            "new_supplied": len(supplied),
            "new_generated": len(new),
            "spliced": len(spliced),
        },
        "per_trait": dict(sorted(Counter(trait(r) for r in spliced).items())),
        "selection": selection,
        "multiagent_share_of_supplied": round(share, 3),
        "lengths": lengths,
        "failures": failures,
    }
    print(json.dumps(report, indent=2))
    if failures:
        sys.exit("GATE FAILED: " + "; ".join(failures))

    today = date.today().isoformat()
    name = artifact_name("da multiagent sprinkled spliced", date=today)
    out = Path("output/da_multiagent_sprinkled") / name.replace("-", "_")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "dataset.jsonl", "w", encoding="utf-8") as f:
        for r in spliced:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "splice_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print("wrote", out)

    if not args.push:
        return
    fields = {
        "experiment": (
            "difficult-advice corpus with the rows of principles 1, 2, 6, 7 swapped for "
            "multi-agent rows written against the multi-agent-sprinkled nine-principle "
            "constitution; principles 3, 4, 5, 8, 9 keep the 2026-09-14 baseline rows"
        ),
        "date_generated": today,
        "constitution": CONSTITUTION,
        "source_repo": f"lessons_from_constitutional_aft @ {git_sha()}",
        "models": (
            f"rows inherited from {base_ref['repo']}@{base_ref['revision']} and "
            f"{new_ref['repo']}@{new_ref['revision']}; see their cards"
        ),
        "generation_config": "none of its own: a splice of two published corpora; fill rows drawn with random.Random(0)",
        "schema": "messages (system, user, assistant with reasoning_content) + metadata (trait_id, ...)",
        "provenance": "uv run python " + " ".join(sys.argv),
        "rows": json.dumps(report["per_trait"]),
        "selection": json.dumps(selection),
    }
    front = {
        "configs": [
            {"config_name": "default", "data_files": "dataset.jsonl", "default": True}
        ],
        "tags": training_data_tags("synth", "da-multiagent-sprinkled", CONSTITUTION),
    }
    url = push_files(
        [out / "dataset.jsonl", out / "splice_report.json"],
        name,
        fields,
        private=False,
        front_matter=front,
    )
    print("pushed", url)


if __name__ == "__main__":
    main()
