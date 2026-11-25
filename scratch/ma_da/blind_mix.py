# ABOUTME: Write a blind rating set: a mada run's rows mixed with the four approved examples, shuffled and unlabelled.
# ABOUTME: Usage: uv run python scratch/ma_da/blind_mix.py <run_dir> <out_dir>  (also writes key.json OUTSIDE the rows folder).
import json
import random
import sys
from pathlib import Path

from calibrate_gates import EXAMPLES, ROOT, load_example, principles


def main(run_dir: str, out_dir: str) -> None:
    out = Path(out_dir)
    (out / "rows").mkdir(parents=True, exist_ok=True)
    cfg_constitution = (
        ROOT / "constitutions/claude_distilled_09_principles/constitution.md"
    )
    traits = principles(cfg_constitution)
    items = []
    for line in (Path(run_dir) / "dataset.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        m = {x["role"]: x for x in row["messages"]}
        meta = row["metadata"]
        items.append(
            {
                "source": f"generated {meta['scenario_id']}",
                "trait_name": meta["trait_name"],
                "trait_text": meta["trait_text"],
                "system": m["system"]["content"],
                "user": m["user"]["content"],
                "reasoning": m["assistant"].get("reasoning_content", ""),
                "response": m["assistant"]["content"],
            }
        )
    for n, tid in EXAMPLES.items():
        ex = load_example(n)
        name, text = traits[tid]
        items.append(
            {
                "source": f"approved example {n}",
                "trait_name": name,
                "trait_text": text,
                **ex,
            }
        )
    random.Random(20261125).shuffle(items)
    key = {}
    for i, it in enumerate(items):
        label = chr(65 + i)
        key[label] = it["source"]
        (out / "rows" / f"row_{label}.md").write_text(
            f"# Row {label}\n\n## Principle this row was written for\n\n**{it['trait_name']}**\n\n{it['trait_text']}\n\n"
            f"## System prompt\n\n{it['system']}\n\n## User message\n\n{it['user']}\n\n"
            f"## The model's reasoning\n\n{it['reasoning']}\n\n## The model's reply\n\n{it['response']}\n"
        )
    (out / "key.json").write_text(json.dumps(key, indent=2))
    print(json.dumps(key, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
