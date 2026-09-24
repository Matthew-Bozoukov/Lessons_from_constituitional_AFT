# ABOUTME: Scenario-paired ODCV contrasts for da-multiparty-15 against the filtered-base ladder (arm_difference, the
# ABOUTME: ladder's method), overall and per variant, from each run's published results.json; writes output/odcv/<date>_*.json.
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from src.eval.misalignment.odcv.odcv import VARIANTS
from src.eval.misalignment.odcv.stats import arm_difference
from src.naming import figure_path

ORG = "dougalldeepmind"
RUNS = {
    "da-multiparty-15": "2026-09-24-odcv-qwen36-0-da-multiparty-15",
    "da-15 (09-23, same-day control)": "2026-09-23-odcv-qwen36-0-da-15",
    "da-15 (09-22)": "2026-09-22-odcv-qwen36-0-da-15",
    "nosynth (09-22)": "2026-09-22-odcv-qwen36-0-nosynth",
    "delib-sonnet-15": "2026-09-22-odcv-qwen36-0-delib-sonnet-15",
    "delib-15": "2026-09-22-odcv-qwen36-0-delib-15",
}


def cells(repo: str, variant: str | None = None) -> dict[str, float]:
    res = json.load(
        open(
            hf_hub_download(
                f"{ORG}/{repo}", "results/results.json", repo_type="dataset"
            )
        )
    )
    med = res["per_scenario_medians"]
    return {
        f"{v}/{s}": x
        for v in VARIANTS
        if variant in (None, v)
        for s, x in med[v].items()
    }


out = {}
t_name = "da-multiparty-15"
for variant in (None, "mandated", "incentivized"):
    t = cells(RUNS[t_name], variant)
    for c_name, repo in RUNS.items():
        if c_name == t_name:
            continue
        c = cells(repo, variant)
        shared = sorted(set(t) & set(c))
        diff = arm_difference({k: t[k] for k in shared}, {k: c[k] for k in shared})
        row = {"n_shared": len(shared), "paired": diff}
        out[f"{variant or 'overall'} | {t_name} - {c_name}"] = row
        print(
            f"{variant or 'overall':12s} {t_name} - {c_name:34s} n={len(shared):3d} "
            f"diff={json.dumps(diff, default=str)[:260]}"
        )
dest = figure_path(Path("output/odcv"), "odcv_da_multiparty_15_paired_contrasts", ext="json")
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(json.dumps(out, indent=1, default=str))
print(dest)
