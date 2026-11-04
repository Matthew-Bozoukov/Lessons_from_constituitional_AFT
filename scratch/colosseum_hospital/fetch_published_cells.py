# ABOUTME: Pulls the published fixed-harness Hospital cells, their env snapshots and the mid-shift probes from the
# ABOUTME: Hub into the local layout h15_t10_analysis.py and h15_midshift_probe.py read, so a fresh checkout can re-run them.
"""Fetch every published input the fixed-harness analysis reads, into the paths it reads them from.

    PYTHONPATH=scratch/colosseum_hospital uv run python scratch/colosseum_hospital/fetch_published_cells.py

Cells: each of h15_t10_analysis.CELLS, from `<org>/<cell dir name with - for _>` into
output/colosseum_hospital/merged/<cell dir name>. Env snapshots: `<org>/<date>-colosseum-hospital-env-snapshots`,
one folder per pod, into output/colosseum_hospital/env_logs/<date>_<group>/<pod>/ (the fleet's own layout);
a pod's group is the group of the cell it was merged into (CELLS), never the pulled folder's name. Probes:
`<org>/2026-09-18-colosseum-hospital-midshift-probes` into h15_midshift_probe.OUT (one folder per tag).
A cell already present locally (it has results/results.json) is left alone, so a cell this machine ran
itself is never overwritten by a published copy; a cell not yet published is reported and skipped.
"""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

import batch_analysis as B
import h15_midshift_probe as hp
import h15_t10_analysis as H

PROBES = f"{B.HUB_ORG}/2026-09-18-colosseum-hospital-midshift-probes"


def cells() -> dict[str, str]:
    """Pod -> '<date>_<group>' for every pod the analysis's cells were merged from."""
    api = HfApi()
    pods_of: dict[str, str] = {}
    for key in H.ALL:
        local = H.cell_dir(key)
        repo = f"{B.HUB_ORG}/{local.name.replace('_', '-')}"
        if (local / "results" / "results.json").is_file():
            print(f"  cell {key[1]:6s} present locally, kept: {local.name}")
        elif api.repo_exists(repo, repo_type="dataset"):
            snapshot_download(repo, repo_type="dataset", local_dir=local)
            print(f"  cell {key[1]:6s} fetched {repo}")
        else:
            print(f"  cell {key[1]:6s} not local and not published ({repo}): skipped")
            continue
        merged = json.loads((local / "metadata" / "merged_from.json").read_text())
        for piece in merged["pieces"]:
            # output/colosseum_hospital/<date>/<pulled group>/<pod>/<run>. The snapshots go under the
            # CELL's group (CELLS' pod group), which is what the analysis reads: the t10 pods were
            # pulled into a group of their own (fixed_t10only) but belong to 2026-09-15_fixed.
            pods_of[Path(piece).parts[-2]] = f"{H.DATE_OF[key]}_{key[0]}"
    return pods_of


def snapshots(pods_of: dict[str, str]) -> None:
    api = HfApi()
    for date in sorted({g.split("_")[0] for g in pods_of.values()}):
        repo = f"{B.HUB_ORG}/{date}-colosseum-hospital-env-snapshots"
        if not api.repo_exists(repo, repo_type="dataset"):
            print(f"  snapshots {date}: not published ({repo}), skipped")
            continue
        files = [s.rfilename for s in api.dataset_info(repo).siblings]
        for pod in sorted({f.split("/")[0] for f in files if "/" in f}):
            if pod not in pods_of:
                continue  # a pod of a cell this analysis does not read
            dest = B.ENV / pods_of[pod]
            if (dest / pod).is_dir():
                print(f"  snapshots {pods_of[pod]}/{pod}: present, kept")
                continue
            snapshot_download(
                repo, repo_type="dataset", local_dir=dest, allow_patterns=[f"{pod}/*"]
            )
            print(f"  snapshots {pods_of[pod]}/{pod}: fetched")


def probes() -> None:
    snapshot_download(
        PROBES,
        repo_type="dataset",
        local_dir=hp.OUT,
        allow_patterns=["*/probes.jsonl", "*/contexts/*"],
    )
    tags = sorted(p.parent.name for p in hp.OUT.glob("*/probes.jsonl"))
    print(f"  probes: {tags} -> {hp.OUT}")


if __name__ == "__main__":
    snapshots(cells())
    probes()
