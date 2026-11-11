# ABOUTME: Prints MASK pressure prompts rendered through Qwen3.6's chat template WITH their written
# ABOUTME: tools -- the text the model reads in the MASK+tools eval -- for a by-eye check before a run.
"""Run: uv run python scratch/mask_tools/show_prompts.py <export.jsonl> [--n 6] [--out FILE]

<export.jsonl> is a mask-tools synth export (one row per MASK task_id with `tools`). For each shown
row the pressure prompt is rebuilt from MASK's own CSV exactly as the harness sends it (system +
user; doubling-down: system + user + scripted assistant + user) and rendered by render_chat with
the generation prompt, as vLLM renders a `tools=` request (src/model_profile.py). One row per
archetype first, then more in file order.
"""

import argparse
import csv
import json
from pathlib import Path

from transformers import AutoTokenizer

from src.model_profile import model_profile, render_chat

CSV_DIR = Path("src/eval/misalignment/mask/third_party/mask/mask/csv_data")
BASE = "Qwen/Qwen3.6-27B"


def mask_rows() -> dict[str, tuple[str, dict]]:
    out = {}
    for p in sorted(CSV_DIR.glob("*.csv")):
        with open(p, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                out[r["task_id"]] = (p.stem, r)
    return out


def pressure_messages(arch: str, r: dict) -> list[dict]:
    if arch == "doubling_down_known_facts":
        return [
            {"role": "system", "content": r["system_prompt"]},
            {"role": "user", "content": r["dd_turn_1_prompt"]},
            {"role": "assistant", "content": r["dd_turn_1_response"]},
            {"role": "user", "content": r["user_prompt"]},
        ]
    return [
        {"role": "system", "content": r["system_prompt"]},
        {"role": "user", "content": r["user_prompt"]},
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("export")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out")
    a = ap.parse_args()
    prof = model_profile(BASE)
    tok = AutoTokenizer.from_pretrained(BASE, revision=prof.revision)
    rows = [json.loads(line) for line in open(a.export, encoding="utf-8")]
    by_id = mask_rows()
    seen, picked = set(), []
    for row in rows:
        arch = row["metadata"]["archetype"]
        if arch not in seen:
            seen.add(arch)
            picked.append(row)
    picked += [r for r in rows if r not in picked][: max(0, a.n - len(picked))]
    chunks = []
    for row in picked[: a.n]:
        tid = row["metadata"]["scenario_id"]
        arch, r = by_id[tid]
        text = render_chat(
            tok,
            pressure_messages(arch, r),
            row["tools"] or None,
            render_kwargs=prof.render_kwargs or {},
            add_generation_prompt=True,
        )
        chunks.append(
            f"{'=' * 100}\n{arch}  task_id={tid}  tools={len(row['tools'])} "
            f"({', '.join(t['function']['name'] for t in row['tools'])})  "
            f"status={row['metadata']['tools_status']}\n{'=' * 100}\n{text}"
        )
    out = "\n\n".join(chunks)
    print(out)
    if a.out:
        Path(a.out).write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
