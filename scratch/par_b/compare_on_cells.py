# ABOUTME: Re-summarise ODCV results.json files on ONE shared cell set (a config's exclusions),
# ABOUTME: so arms run on 70 cells can be read next to an arm run on 65 without mixing cell sets.
# Run: uv run python scratch/par_b/compare_on_cells.py --config <65-cell cfg> --results name=path [name=path ...]

import json
import sys
from pathlib import Path

import fire
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eval.misalignment.odcv.odcv import summarise  # noqa: E402


def _restrict(per_scenario_medians: dict, excluded: set[str]) -> dict:
    """Drop every `<variant>/<scenario>` cell named in `excluded`; keep rollout keys intact."""
    out: dict = {}
    for variant, cells in per_scenario_medians.items():
        kept = {}
        for key, score in cells.items():
            scenario = key.split("/")[0]
            if f"{variant}/{scenario}" in excluded:
                continue
            kept[key] = score
        out[variant] = kept
    return out


def main(config: str, *results: str) -> None:
    """Print each arm's MR/severity on the config's cell set (and as originally reported).

    Args:
        config: The ODCV config whose `exclude_scenarios` defines the shared cell set.
        *results: `name=path/to/results.json` pairs.
    """
    cfg = OmegaConf.load(config)
    excluded = set(OmegaConf.to_container(cfg.get("exclude_scenarios", []) or []))
    print(f"cell set: {cfg.get('expected_cells', '?')} cells "
          f"({len(excluded)} exclusions from {config})\n")
    print(f"{'arm':<28} {'as reported':>24}   {'on shared cells':>40}")
    for item in results:
        name, path = item.split("=", 1)
        r = json.loads(Path(path).read_text(encoding="utf-8"))
        rep = r["ours"]["overall"]
        psm = r.get("per_scenario_medians")
        if not psm:
            print(f"{name:<28} MR {rep['mr_pct']:5.1f}% sev {rep['mean_severity']:.2f} n={rep['n']:<4}"
                  f"   (no per-scenario medians -- cannot restrict)")
            continue
        s = summarise(_restrict(psm, excluded))
        o = s["overall"]
        n_cells = sum(len({k.split('/')[0] for k in v}) for v in _restrict(psm, excluded).values())
        print(f"{name:<28} MR {rep['mr_pct']:5.1f}% sev {rep['mean_severity']:.2f} n={rep['n']:<4}"
              f"   MR {o['mr_pct']:5.1f}% {o['mr_ci95']} sev {o['mean_severity']:.2f} "
              f"n={o['n']} cells={n_cells}  mand {s['mandated']['mr_pct']:.1f}% "
              f"inc {s['incentivized']['mr_pct']:.1f}%")


if __name__ == "__main__":
    fire.Fire(main)
