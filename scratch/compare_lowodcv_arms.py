# ABOUTME: Compare the low-ODCV ODCV arm against its sibling arms on the shared 65 cells.
# ABOUTME: Reports MR/CI per arm, rollout coverage, and a scenario-level paired sign test.

"""Put the low-ODCV arm beside the arms it is meant to be compared with.

`per_scenario_medians` is keyed `<Scenario>/rollout_NNN` per variant, one entry per rollout,
holding that rollout's median score across judges. A scenario counts as misaligned for the
paired test when its mean rollout score exceeds 0, and the test runs over scenarios BOTH
arms attempted, because the arms differ in which cells produced transcripts.

Run:
  uv run python scratch/compare_lowodcv_arms.py
"""

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("output/odcv_bench")
ARMS = {
    "lowodcv":   "qwen3_6-27b-lora-t2-9284-synthdoc-716-lowodcv-r64/combined3x_20260818_194628",
    "c6excised": "qwen3_6-27b-lora-t2-9284-synthdoc-716-c6excised-r64/combined3x_20260818_153146",
    "advocacy":  "qwen3_6-27b-lora-t2-9284-synthdoc-716-advocacy-r64/combined3x_20260818_141118",
    "c6masked":  "qwen3_6-27b-lora-t2-9284-synthdoc-716-c6masked-r64/combined4x_20260817_085635",
    "traits134": "qwen3_6-27b-lora-t2-9284-synthdoc-716-traits134-r64/combined3x_20260819_092850",
}


def by_scenario(d: dict) -> dict[str, list[float]]:
    """Group rollout medians by scenario cell.

    Args:
        d: A parsed results.json.

    Returns:
        {variant/scenario: [median score per rollout]}
    """
    out = defaultdict(list)
    for variant, entries in d["per_scenario_medians"].items():
        for key, score in entries.items():
            scen = key.rsplit("/rollout_", 1)[0]
            out[f"{variant}/{scen}"].append(score)
    return dict(out)


loaded = {}
print(f"{'arm':11s}{'MR':>8}{'CI95':>15}{'Sev':>7}{'judged':>8}{'cells':>7}")
for name, rel in ARMS.items():
    p = ROOT / rel / "results.json"
    if not p.is_file():
        print(f"{name:11s} results.json MISSING ({rel})")
        continue
    d = json.loads(p.read_text())
    loaded[name] = d
    o = d["ours"]["overall"]
    lo_, hi_ = o["mr_ci95"]
    print(f"{name:11s}{o['mr_pct']:6.1f}%  [{lo_:4.1f},{hi_:5.1f}]{o['mean_severity']:7.2f}"
          f"{o['n']:8d}{len(by_scenario(d)):7d}")

lo = by_scenario(loaded["lowodcv"])
cov = defaultdict(int)
for v in lo.values():
    cov[len(v)] += 1
print(f"\nlow-ODCV rollouts per cell: {dict(sorted(cov.items()))}")
thin = sorted(k for k, v in lo.items() if len(v) < 3)
print(f"cells with <3 rollouts ({len(thin)}): {thin}")

print()
for other in ("traits134", "c6excised", "advocacy", "c6masked"):
    if other not in loaded:
        continue
    ot = by_scenario(loaded[other])
    shared = sorted(set(lo) & set(ot))
    mis = lambda t, k: (sum(t[k]) / len(t[k])) > 0  # noqa: E731
    a = sum(1 for k in shared if mis(lo, k))
    b = sum(1 for k in shared if mis(ot, k))
    lo_only = [k for k in shared if mis(lo, k) and not mis(ot, k)]
    ot_only = [k for k in shared if mis(ot, k) and not mis(lo, k)]
    n = len(lo_only) + len(ot_only)
    try:
        from scipy.stats import binomtest
        p = binomtest(len(lo_only), n, 0.5).pvalue if n else float("nan")
    except ImportError:
        p = float("nan")
    print(f"vs {other:10s} shared={len(shared):3d} cells | lowodcv {a:2d} misaligned "
          f"vs {b:2d} | discordant {len(lo_only)}:{len(ot_only)} | sign p={p:.3f}")
