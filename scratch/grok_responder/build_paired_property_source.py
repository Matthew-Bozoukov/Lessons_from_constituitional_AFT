# ABOUTME: One jsonl holding BOTH arms' 703 training rows, arm-labelled, for a single
# ABOUTME: property-discovery fit — so each property gets a per-arm prevalence, not two runs.

"""Run: uv run python scratch/grok_responder/build_paired_property_source.py

Why one file rather than two runs. Two independent discovery runs produce two vocabularies
and two clusterings, and nothing aligns a group in one with a group in the other — you end
up eyeballing labels. Fitting BOTH arms in one run gives every property a single definition
and a prevalence per arm, which is what a contrast needs.

Ids are namespaced per arm because the two corpora answer the SAME 703 scenarios and the
loader refuses duplicate ids. The original id survives as `pair_id`, so a paired analysis
downstream can still join the two halves row for row.
"""

import json
from pathlib import Path

import fire

GROK = "data/feature_discovery/grok_703.jsonl"
SONNET = "data/feature_discovery/sonnet_703.jsonl"
OUT = Path("data/feature_discovery/paired_arms_1406.jsonl")


def main(out: str = str(OUT)) -> None:
    """Write the combined, arm-labelled corpus."""
    per_arm: dict[str, list] = {}
    for arm, path in (("grok", GROK), ("sonnet", SONNET)):
        out_rows = []
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            md = dict(r["metadata"])
            sid = md["scenario_id"]
            md["pair_id"] = sid  # the join key across arms
            md["scenario_id"] = f"{arm}:{sid}"  # unique within the combined corpus
            md["arm"] = arm
            out_rows.append(
                {
                    "messages": r["messages"],
                    "metadata": md,
                    "source": f"difficult_advice_{arm}",
                }
            )
        per_arm[arm] = out_rows

    # INTERLEAVED, not concatenated. `limit:` takes a PREFIX, so a concatenated file makes
    # every smoke run single-arm and the contrast impossible ("no records carry
    # arm=['sonnet']"). Interleaving also keeps any truncation arm-balanced.
    rows = [r for pair in zip(per_arm["grok"], per_arm["sonnet"]) for r in pair]

    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    per = {}
    for r in rows:
        per[r["metadata"]["arm"]] = per.get(r["metadata"]["arm"], 0) + 1
    ids = [r["metadata"]["scenario_id"] for r in rows]
    pairs = {r["metadata"]["pair_id"] for r in rows}
    print(f"wrote {p}  n={len(rows)}  per arm={per}")
    print(f"  unique ids: {len(set(ids))} (no collisions: {len(set(ids)) == len(ids)})")
    print(f"  shared pair_ids: {len(pairs)} (each should appear in both arms)")


if __name__ == "__main__":
    fire.Fire(main)
