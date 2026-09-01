# ABOUTME: Build the ConstitutionEval arms x (split x template) table and the PAIRED arm
# ABOUTME: comparisons from published run dirs. Per-experiment write-up code: scratch/ -> output/.
"""Report the ConstitutionEval result.

    uv run python scratch/constitution_mcq_table.py

Reads the per-arm run dirs under `output/constitution_mcq/` and writes a markdown report
to `output/constitution_mcq/<date>_constitution_mcq_results.md`.

Two things it does that a bare accuracy table would not:

* **Chance is a row.** On a four-way MCQ the distance from 25% is the result; a table
  without that line invites reading 0.96 as "good" rather than "saturated".
* **Arms are compared PAIRED, with McNemar.** Every arm answers identical items, so the
  informative evidence is the discordant pairs. At the accuracies this benchmark produces
  on a 27B, arms differ on a handful of items out of 678 and an unpaired reading of two
  near-identical percentages would manufacture a difference out of noise.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.capabilities.mmlu.mmlu import mcnemar  # noqa: E402
from src.utils import today  # noqa: E402

RUNS = Path("output/constitution_mcq")
TEMPLATES = ("chat", "raw")
SPLITS = (("all", "full"), ("easy", "easy"), ("mid", "mid"), ("hard", "HARD"))

# Substring of the run dir -> (display name, sort order). The base is the reference arm
# for every paired test: it is the one checkpoint that received none of our training.
ARMS = {
    "qwen36_1": ("Qwen3.6-27B base (no SFT)", 0),
    "table2_only": ("table2-only (0% synthetic SFT)", 1),
    "chunk_only_702": ("difficult-advice principle-scoped 702", 2),
}


def arm_of(run_dir: Path) -> tuple[str, int]:
    name = run_dir.name
    for key, value in ARMS.items():
        if key in name:
            return value
    return (name, 99)


def load_runs() -> list[dict]:
    runs = []
    for d in sorted(RUNS.iterdir()):
        if not (d / "results").is_dir():
            continue
        metrics = {t: d / "results" / f"{t}_metrics.json" for t in TEMPLATES}
        if not all(p.exists() for p in metrics.values()):
            continue
        label, order = arm_of(d)
        runs.append(
            {
                "dir": d,
                "label": label,
                "order": order,
                "metrics": {t: json.loads(p.read_text()) for t, p in metrics.items()},
                "rollouts": {
                    t: [
                        json.loads(x)
                        for x in (d / "rollouts" / f"{t}_rollouts.jsonl")
                        .read_text()
                        .splitlines()
                    ]
                    for t in TEMPLATES
                },
            }
        )
    if not runs:
        raise SystemExit(f"!!! no complete constitution_mcq runs under {RUNS}")
    runs.sort(key=lambda r: (r["order"], -len(r["rollouts"]["chat"])))
    # One row per arm: keep the largest run (the full 678 supersedes a pilot).
    seen, keep = set(), []
    for r in runs:
        if r["label"] in seen:
            continue
        seen.add(r["label"])
        keep.append(r)
    return keep


def cell(metrics: dict, split: str) -> str:
    if split == "all":
        return f"**{metrics['accuracy_debiased'] * 100:.1f}** ({metrics['accuracy_naive'] * 100:.1f})"
    band = metrics["band_acc"].get(split)
    return "—" if band is None else f"{band['acc'] * 100:.1f}"


def correctness(rows: list[dict]) -> dict[str, bool]:
    out = {}
    for r in rows:
        pred = r.get("chosen_option") or r.get("answer")
        gold = r.get("gold_option") or r.get("gold_letter")
        out[r["id"]] = pred == gold
    return out


def main() -> None:
    runs = load_runs()
    n_items = len(runs[0]["rollouts"]["chat"])

    head = ["arm"] + [f"{lbl} · {t}" for t in TEMPLATES for _, lbl in SPLITS]
    lines = [
        "| " + " | ".join(head) + " |",
        "|" + "|".join(["---"] * len(head)) + "|",
        "| _chance_ | " + " | ".join(["25.0"] * (len(head) - 1)) + " |",
    ]
    for r in runs:
        cells = [cell(r["metrics"][t], s) for t in TEMPLATES for s, _ in SPLITS]
        lines.append(f"| {r['label']} | " + " | ".join(cells) + " |")

    # --- paired comparisons, every arm against the untrained base -----------------
    ref = next((r for r in runs if r["order"] == 0), None)
    paired: list[str] = []
    if ref is not None:
        paired += [
            "",
            "### Paired comparison vs the untrained base (McNemar, exact)",
            "",
        ]
        paired += [
            "| arm | template | both right | arm only | base only | p |",
            "|---|---|---|---|---|---|",
        ]
        for r in runs:
            if r is ref:
                continue
            for t in TEMPLATES:
                a, b = correctness(ref["rollouts"][t]), correctness(r["rollouts"][t])
                shared = sorted(set(a) & set(b))
                res = mcnemar([a[i] for i in shared], [b[i] for i in shared])
                both = sum(1 for i in shared if a[i] and b[i])
                paired.append(
                    f"| {r['label']} | {t} | {both} | {res['b10']} | {res['b01']} | "
                    f"{res['p_value']:.3f} |"
                )

    diag = [
        "",
        "### Instrument diagnostics",
        "",
        "| arm | template | display-slot argmax share | letters outside top-k |",
        "|---|---|---|---|",
    ]
    for r in runs:
        for t in TEMPLATES:
            m = r["metrics"][t]
            pb = " / ".join(f"{p:.3f}" for p in m["position_bias"])
            diag.append(
                f"| {r['label']} | {t} | {pb} | "
                f"{m['letters_missing_from_topk']} of {m['letters_scored']} |"
            )

    body = "\n".join(
        [
            "# ConstitutionEval (SPP `charter_mcq`) — swap-debiased accuracy, %",
            "",
            f"{n_items} items, 4 rotations each, chance 25.0. Cells are **debiased**; the "
            "full-split cell carries (naive) beside it. `HARD` = ConstitutionEval-Hard, "
            'which is the `e4b_blind_band == "hard"` subset — a difficulty label produced '
            "by blind trials of **gemma-3n-e4b-it (4B)**, not of a 27B.",
            "",
        ]
        + lines
        + paired
        + diag
        + ["", f"_generated {today()}; source: {RUNS}/_", ""]
    )
    OUT = RUNS / f"{today()}_constitution_mcq_results.md"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(body)
    print(body)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
