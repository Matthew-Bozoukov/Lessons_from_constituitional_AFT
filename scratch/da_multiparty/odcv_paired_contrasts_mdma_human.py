# ABOUTME: Scenario-paired ODCV contrasts for the human-parties MDMA arm (da-multiparty-human-15) against MDMA and the ladder, with
# ABOUTME: arm_difference and exact McNemar (odcv_compare's method), from each run's published results.json.
import json

from huggingface_hub import hf_hub_download

from src.eval.misalignment.odcv.odcv import VARIANTS, VIOLATION_THRESHOLD
from src.eval.misalignment.odcv.stats import arm_difference, mcnemar_exact

ORG = "dougalldeepmind"
RUNS = {
    "mdma-human": "2026-09-24-odcv-qwen36-0-da-multiparty-human-15",
    "MDMA (da-multiparty-15)": "2026-09-24-odcv-qwen36-0-da-multiparty-15",
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
t_name = "mdma-human"
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
json.dump(
    out,
    open("/Users/kunwar/.claude/jobs/defad298/tmp/odcv_paired_mph15.json", "w"),
    indent=1,
    default=str,
)
