# ABOUTME: Turns the 1,000 MASK pressure prompts into a JSONL the synth engine reads as a source
# ABOUTME: run, so mask-tools.yaml can write each row its own unused tools with the DA+tools recipe.
"""Run: uv run python scratch/mask_tools/build_source.py

Writes data/mask_tools_source/rows.jsonl (all rows) and data/mask_tools_smoke_source/rows.jsonl
(2 per archetype). One record per MASK row: scenario_id = the row's task_id (unique across the six
archetype CSVs, checked here; the eval looks the tools up by it), archetype, system, user. A
doubling-down row's pressure prompt is two user turns around a scripted assistant turn, so its
`user` holds all three, labelled, for the writer and judges to read.
"""

import csv
import json
from pathlib import Path

CSV_DIR = Path("src/eval/misalignment/mask/third_party/mask/mask/csv_data")
ARCHETYPES = [
    "continuations",
    "disinformation",
    "doubling_down_known_facts",
    "known_facts",
    "provided_facts",
    "statistics",
]


def rows():
    for arch in ARCHETYPES:
        with open(CSV_DIR / f"{arch}.csv", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                if arch == "doubling_down_known_facts":
                    user = (
                        f"[earlier user turn]\n{r['dd_turn_1_prompt']}\n\n"
                        f"[earlier assistant turn]\n{r['dd_turn_1_response']}\n\n"
                        f"[current user turn]\n{r['user_prompt']}"
                    )
                else:
                    user = r["user_prompt"]
                assert r["system_prompt"].strip(), (
                    f"{arch} {r['task_id']}: empty system prompt"
                )
                yield {
                    "scenario_id": r["task_id"],
                    "archetype": arch,
                    "system": r["system_prompt"],
                    "user": user,
                }


def write(path: Path, recs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"{path}: {len(recs)} rows")


def main() -> None:
    recs = list(rows())
    ids = [r["scenario_id"] for r in recs]
    assert len(ids) == len(set(ids)), "task_id is not unique across the MASK CSVs"
    write(Path("data/mask_tools_source/rows.jsonl"), recs)
    smoke = [
        r for a in ARCHETYPES for r in [x for x in recs if x["archetype"] == a][:2]
    ]
    write(Path("data/mask_tools_smoke_source/rows.jsonl"), smoke)
    for a in ARCHETYPES:
        print(f"  {a}: {sum(r['archetype'] == a for r in recs)}")


if __name__ == "__main__":
    main()
