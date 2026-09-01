# ABOUTME: Read the constitution_mcq run summaries and print/write the arms x (split x template)
# ABOUTME: accuracy table. Per-experiment write-up code, so scratch/ and output/ per CLAUDE.md.
"""Build the ConstitutionEval comparison table from published run summaries.

    uv run python scratch/constitution_mcq_table.py

Reads `output/eval_summaries/*constitution_mcq*.json` (run_eval's mirror of every arm's
summary) and writes a markdown table to `output/constitution_mcq/<date>_constitution_mcq_table.md`.

Cells are `accuracy_debiased (accuracy_naive)`. Chance is 25% and is drawn as its own row,
because on a four-way MCQ the distance from 25 is the whole result and an eyeballed table
without it invites reading 0.61 as "good".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import today  # noqa: E402

SUMMARIES = Path("output/eval_summaries")
OUT_DIR = Path("output/constitution_mcq")
TEMPLATES = ("chat", "raw")
# Ordered easiest -> hardest, with the full set first: `hard` is ConstitutionEval-Hard,
# the split the paper's scaling result is carried by.
SPLITS = (
    ("all", "full 678"),
    ("easy", "easy 305"),
    ("mid", "mid 156"),
    ("hard", "HARD 217"),
)

# Display names, in the order they should appear. Anything unmatched keeps its own key.
ARM_LABELS = {
    "qwen36": "Qwen3.6-27B base (no SFT)",
    "qwen36_lora_table2_only_9284": "table2-only (0% synthetic SFT)",
    "qwen36_lora_table2_9284_difficult_advice_chunk_only_702": "difficult-advice principle-scoped 702",
}


def load_runs() -> list[dict]:
    runs = []
    for path in sorted(SUMMARIES.glob("*constitution_mcq*.json")):
        data = json.loads(path.read_text())
        data["_file"] = path.name
        runs.append(data)
    if not runs:
        raise SystemExit(f"!!! no constitution_mcq summaries in {SUMMARIES}")
    # Keep the newest run per target: a re-run supersedes, it does not add a row.
    latest: dict[str, dict] = {}
    for run in runs:
        latest[run["target"]] = run
    return list(latest.values())


def arm_key(run: dict) -> str:
    for key in ARM_LABELS:
        if key in run["_file"]:
            return key
    return run["target"].split("/")[-1]


def cell(run: dict, template: str, split: str) -> str:
    if split == "all":
        deb, naive = f"{template}_accuracy_debiased", f"{template}_accuracy_naive"
    else:
        deb, naive = f"{template}_{split}_accuracy_debiased", None
    if deb not in run:
        return "—"
    text = f"**{run[deb] * 100:.1f}**"
    if naive and naive in run:
        text += f" ({run[naive] * 100:.1f})"
    return text


def main() -> None:
    runs = load_runs()
    order = list(ARM_LABELS)
    runs.sort(key=lambda r: order.index(arm_key(r)) if arm_key(r) in order else 99)

    head = ["arm"] + [
        f"{lbl} · {t}"
        for t, _ in ((t, None) for t in TEMPLATES)
        for lbl in [s[1] for s in SPLITS]
    ]
    head = ["arm"] + [
        f"{split_label} · {t}" for t in TEMPLATES for _, split_label in SPLITS
    ]
    lines = [
        "| " + " | ".join(head) + " |",
        "|" + "|".join(["---"] * len(head)) + "|",
        "| _chance_ | " + " | ".join(["25.0"] * (len(head) - 1)) + " |",
    ]
    for run in runs:
        cells = [cell(run, t, split) for t in TEMPLATES for split, _ in SPLITS]
        lines.append(
            f"| {ARM_LABELS.get(arm_key(run), arm_key(run))} | "
            + " | ".join(cells)
            + " |"
        )

    diag = ["", "### Diagnostics", ""]
    for run in runs:
        for t in TEMPLATES:
            pb = run.get(f"{t}_position_bias")
            miss = run.get(f"{t}_letters_missing_from_topk")
            if pb is None:
                continue
            diag.append(
                f"- `{ARM_LABELS.get(arm_key(run), arm_key(run))}` / {t}: "
                f"display-slot argmax {['%.2f' % p for p in pb]}, "
                f"letters outside top-k: {miss}"
            )

    body = "\n".join(
        [
            "# ConstitutionEval (SPP charter_mcq) — swap-debiased accuracy, %",
            "",
            "Cells: **debiased** (naive, full split only). Chance = 25.0. "
            "`hard` = ConstitutionEval-Hard.",
            "",
        ]
        + lines
        + diag
        + ["", f"_generated {today()}; source: {SUMMARIES}/_", ""]
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{today()}_constitution_mcq_table.md"
    out.write_text(body)
    print(body)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
