# ABOUTME: Holistic read of the four Fig. 13 Subagents Comply More runs: every scenario x request-variant cell,
# ABOUTME: every metric, validity per arm, and Drori's mechanical grader beside the LLM judge.
# Run: uv run python scratch/dh_holistic/grid.py [--root output/dh_holistic]
"""Reads the snapshots scratch/dh_holistic downloads from dougalldeepmind/2026-10-12-dh-qwen36-0-<arm>.
Prints: per-arm status counts; per-scenario rates for the headline metrics; a paired da vs da_tools
comparison restricted to cells where both arms have a valid episode; and a pooled-over-scenarios rate
computed only on scenarios every arm has data for (so arms are not averaged over different scenario sets)."""

from __future__ import annotations

import collections
import glob
import json
import math
import os

import fire

ARMS = {"control": "plain", "delib": "delib-sonnet-15", "da": "da-15", "da_tools": "da-tools-15"}
METRICS = ["harmful_action", "paper_complied", "explicit_refusal", "benign_complete", "joint_success",
           "concern_any_human", "safe_substitution", "justified_deferral"]


def load(root: str) -> dict[str, list[dict]]:
    out = {}
    for arm, d in ARMS.items():
        rows = []
        for f in sorted(glob.glob(os.path.join(root, d, "results", "episodes", "*.json"))):
            r = json.load(open(f, encoding="utf-8"))
            rows.append(r)
        out[arm] = rows
    return out


def rate(rows, m):
    v = [r["metrics"][m] for r in rows if r["status"] == "complete" and "metrics" in r]
    return (sum(v) / len(v) if v else float("nan")), len(v)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def main(root: str = "output/dh_holistic", dump: str = "output/dh_holistic/grid.json") -> None:
    data = load(root)
    print("== status per arm ==")
    for arm, rows in data.items():
        print(f"{arm:9s}", dict(collections.Counter(r["status"] for r in rows)), "n=", len(rows))

    scen = sorted({r["scenario"] for rows in data.values() for r in rows})
    print("\n== valid episodes per scenario (of 9) ==")
    print(f"{'scenario':22s}" + "".join(f"{a:>10s}" for a in ARMS))
    for s in scen:
        print(f"{s:22s}" + "".join(
            f"{sum(1 for r in data[a] if r['scenario']==s and r['status']=='complete'):>10d}" for a in ARMS))

    grid = {}
    for m in METRICS:
        print(f"\n== {m} per scenario (% of valid; '-' = no valid episode) ==")
        print(f"{'scenario':22s}" + "".join(f"{a:>10s}" for a in ARMS))
        for s in scen:
            cells = []
            for a in ARMS:
                rr = [r for r in data[a] if r["scenario"] == s]
                p, n = rate(rr, m)
                grid.setdefault(a, {}).setdefault(s, {})[m] = (p, n)
                cells.append(f"{'-' if n == 0 else f'{100*p:.0f}':>10s}")
            print(f"{s:22s}" + "".join(cells))

    common = [s for s in scen if all(grid[a][s]["harmful_action"][1] > 0 for a in ARMS)]
    print(f"\n== pooled over the {len(common)} scenarios every arm has data for ==")
    print("excluded:", [s for s in scen if s not in common])
    for m in METRICS:
        line = f"{m:22s}"
        for a in ARMS:
            v = [r["metrics"][m] for r in data[a] if r["status"] == "complete" and "metrics" in r and r["scenario"] in common]
            k, n = sum(v), len(v)
            lo, hi = wilson(k, n)
            line += f"  {a}={100*k/n:5.1f} [{lo:4.1f},{hi:4.1f}] n={n}"
        print(line)

    # Paired da vs da_tools at the cell level (scenario, variant, repeat): same request text, same seed.
    print("\n== paired cells da vs da_tools (both valid), harmful_action ==")
    idx = {a: {r["id"]: r for r in data[a] if r["status"] == "complete" and "metrics" in r} for a in ("da", "da_tools")}
    both = sorted(set(idx["da"]) & set(idx["da_tools"]))
    tab = collections.Counter((idx["da"][i]["metrics"]["harmful_action"], idx["da_tools"][i]["metrics"]["harmful_action"]) for i in both)
    print("pairs", len(both), "(da harm, tools harm) ->", dict(tab))
    b, c = tab[(True, False)], tab[(False, True)]
    # exact two-sided McNemar
    n = b + c
    from math import comb
    p = min(1.0, 2 * sum(comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n) if n else float("nan")
    print(f"discordant: da-only-harm={b}, tools-only-harm={c}, exact McNemar p={p:.2g}")

    # Scenario-level sign: is the da -> da_tools drop broad or concentrated?
    print("\n== per-scenario harmful delta (da_tools - da), pp ==")
    for s in scen:
        (pd, nd), (pt, nt) = grid["da"][s]["harmful_action"], grid["da_tools"][s]["harmful_action"]
        if nd and nt:
            print(f"{s:22s} da={100*pd:5.0f} (n={nd})  tools={100*pt:5.0f} (n={nt})  delta={100*(pt-pd):+5.0f}")

    # Judge vs Drori's mechanical grader agreement per arm.
    print("\n== judge harmful_action vs mechanical paper_complied (compliance>=3) ==")
    for a in ARMS:
        t = collections.Counter((r["metrics"]["harmful_action"], r["metrics"]["paper_complied"]) for r in data[a] if r["status"] == "complete" and "metrics" in r)
        print(a, dict(t))

    # Per request-variant (the 36-cell bank: 12 scenarios x 3 variants), harmful rate for da and da_tools.
    print("\n== harmful_action per (scenario, variant), 3 repeats each: control / da / da_tools ==")
    for s in scen:
        row = f"{s:22s}"
        for v in range(3):
            parts = []
            for a in ("control", "da", "da_tools"):
                rr = [r for r in data[a] if r["scenario"] == s and r["variant"] == v and r["status"] == "complete" and "metrics" in r]
                parts.append("-" if not rr else f"{sum(r['metrics']['harmful_action'] for r in rr)}/{len(rr)}")
            row += f"  v{v}: " + " ".join(f"{x:>3s}" for x in parts)
        print(row)

    os.makedirs(os.path.dirname(dump), exist_ok=True)
    json.dump({a: {s: {m: list(v) for m, v in ms.items()} for s, ms in g.items()} for a, g in grid.items()},
              open(dump, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    fire.Fire(main)
