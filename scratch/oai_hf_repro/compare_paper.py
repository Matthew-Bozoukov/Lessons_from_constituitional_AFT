# ABOUTME: Prints an oai_hf run's per-step signature rates (95% Wilson CIs) next to the paper's published
# ABOUTME: n=64 counts, plus our funnels and end reasons, so the comparison reads in one table.
# Run: uv run python scratch/oai_hf_repro/compare_paper.py <run_dir | results/summary.json>
import json
import sys
from pathlib import Path

from src.eval.misalignment.oai_hf.export import wilson

# The paper's headline counts (k, n), as its own figure script prints them
# (plot_four_step_panel at harness 5a6341c over the HF logs, regenerated 2026-11-13).
# Step 2 = protein_none with eval-source readers dropped; GPT-5.6 has no step 3/4 runs.
PAPER = {
    "GLM-5.2": {
        "step1": (7, 64),
        "step2": (0, 64),
        "step3": (1, 64),
        "step4": (40, 64),
    },
    "GLM-5.3": {
        "step1": (16, 64),
        "step2": (3, 63),
        "step3": (42, 64),
        "step4": (28, 64),
    },
    "Qwen 3.8 Max": {
        "step1": (14, 64),
        "step2": (1, 63),
        "step3": (30, 64),
        "step4": (34, 64),
    },
    "Muse Spark 1.3": {
        "step1": (24, 64),
        "step2": (0, 64),
        "step3": (23, 64),
        "step4": (1, 64),
    },
    "Claude Opus 4.8": {
        "step1": (0, 64),
        "step2": (0, 63),
        "step3": (0, 64),
        "step4": (1, 64),
    },
    "GPT-5.6 sol": {"step1": (30, 64), "step2": (3, 62)},
}
STEPS = ["step1", "step2", "step3", "step4"]


def fmt(k, n):
    if n == 0:
        return "   —   "
    p, lo, hi = wilson(k, n)
    return f"{k:>2}/{n:<2} {p:.2f} [{lo:.2f},{hi:.2f}]"


def main(arg: str) -> None:
    path = Path(arg)
    summary_file = path if path.suffix == ".json" else path / "results" / "summary.json"
    if not summary_file.exists():
        summary_file = path / "results" / "results.json"
    summ = json.loads(summary_file.read_text())
    steps = summ.get("steps", summ)
    label = summ.get("run", {}).get("target", "ours")
    print(f"{'model':18}" + "".join(f"{s:>26}" for s in STEPS))
    for m, cells in PAPER.items():
        print(
            f"{m:18}"
            + "".join(f"{fmt(*cells[s]) if s in cells else '—':>26}" for s in STEPS)
        )
    for key in ("signature_all", "signature_valid"):
        row = f"{'ours (' + key.split('_')[1] + ')':18}"
        for s in STEPS:
            c = steps.get(s, {}).get(key)
            row += f"{fmt(c['k'], c['n']) if c else '—':>26}"
        print(row)
    print(f"\nours = {label}")
    for s in STEPS:
        st = steps.get(s)
        if not st:
            continue
        extra = {
            k: st[k]
            for k in ("funnel", "share_given_solved_and_found_board", "found_peers")
            if k in st
        }
        print(
            f"{s}: end {st['end_reasons']} | source probes {st['source_probes']} | {extra}"
        )
        print(f"       health {st['health']}")


if __name__ == "__main__":
    main(sys.argv[1])
