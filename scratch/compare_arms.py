# ABOUTME: Are two arms actually different on an eval, or is the gap inside the interval?
# ABOUTME: Usage: uv run python scratch/compare_arms.py llmbar accuracy

from __future__ import annotations

import glob
import json
import sys

from src.utils import wilson

SHORT = {"courtroom716": "CR", "peercritique716": "PC", "da716": "DA",
         "table2-only": "T2", "Qwen3_6-27B": "base"}


def label(arm: str) -> str | None:
    for key, short in SHORT.items():
        if key in arm:
            return short
    return None  # not a ladder arm (e.g. the local API smoke) — excluded


def main(name: str, metric: str = "accuracy") -> None:
    rows = []
    for path in sorted(glob.glob(f"output/{name}/*/*/results.json")):
        arm = label(path.split("/")[2])
        if arm is None:
            continue
        blob = json.load(open(path))
        node = blob.get(metric)
        if not isinstance(node, dict) or "hits" not in node:
            print(f"{metric!r} on {name} is not a rate with hits/n; nothing to test")
            return
        low, high = wilson(node["hits"], node["n"])
        rows.append((arm, node["rate"], node["hits"], node["n"], low, high))

    rows.sort(key=lambda r: -r[1])
    print(f"{name}.{metric} — Wilson 95% intervals\n")
    print(f"{'arm':5s} {'rate':>6s} {'hits':>6s} {'n':>5s}   95% CI")
    for arm, rate, hits, n, low, high in rows:
        print(f"{arm:5s} {rate:6.3f} {hits:6d} {n:5d}   [{low:.3f}, {high:.3f}]")

    # The only comparison that matters for the paper: does any variant separate from DA?
    da = next((r for r in rows if r[0] == "DA"), None)
    if not da:
        return
    print(f"\nvs DA ({da[1]:.3f}):")
    for arm, rate, _hits, _n, low, high in rows:
        if arm == "DA":
            continue
        overlap = not (low > da[5] or high < da[4])
        verdict = "INSIDE the interval — not separated" if overlap else "separated"
        print(f"  {arm:5s} {rate - da[1]:+.3f}  {verdict}")


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["llmbar"]))
