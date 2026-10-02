# ABOUTME: Build the `da-no-t6-<pct>-mix` straight from a published DA corpus, dropping trait_id=t6 at build
# ABOUTME: time -- no derived synth repo. Pins the source repo@revision on the mixture card.
"""    uv run python scratch/build_da_no_t6_mix.py [--repo dougalldeepmind/2026-09-28-da-synth --revision <sha>] [--pct 15]

The mixture builder has no row filter (a source is a dataset, a budget and a balance field), so this
downloads the corpus at its exact revision, writes a t6-free jsonl locally, and runs the builder on
configs/data/mixture/da-no-t6.yaml with that one source swapped to `path:` -- everything else in the
config (base blend, share unit, seed, hf block) is untouched. The card's `experiment` field carries the
source pin and the drop rule, since a `path:` source cannot pin a Hub revision by itself.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from omegaconf import OmegaConf

load_dotenv(".env")
CFG = Path("configs/data/mixture/da-no-t6.yaml")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="dougalldeepmind/2026-09-28-da-synth")
    ap.add_argument("--revision", default="14efefbf39581f4aaae08bfe7b4491e80adf2c7b")
    ap.add_argument("--pct", type=int, default=15)
    a = ap.parse_args()
    sha = HfApi().dataset_info(a.repo, revision=a.revision).sha
    src = hf_hub_download(a.repo, "dataset.jsonl", repo_type="dataset", revision=sha)
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    kept = [r for r in rows if (r.get("metadata") or {}).get("trait_id") != "t6"]
    assert 0 < len(kept) < len(rows), f"filter kept {len(kept)} of {len(rows)}"
    work = Path("output/derived/no_t6_build"); work.mkdir(parents=True, exist_ok=True)
    local = work / f"{a.repo.split('/')[-1]}_{sha[:8]}_no_t6.jsonl"
    local.write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8")

    cfg = OmegaConf.load(CFG)
    (name,) = list(cfg.sources.keys())
    spec = OmegaConf.to_container(cfg.sources[name])
    for k in ("dataset", "revision", "repo", "file"):
        spec.pop(k, None)
    spec["path"] = str(local)
    cfg.sources[name] = spec
    cfg.hf.experiment = (f"difficult advice without trait 6 (stable persona): {a.repo} @ {sha} with trait_id=t6 "
                         f"dropped at build time ({len(kept)} of {len(rows)} rows kept; "
                         f"scratch/build_da_no_t6_mix.py), the base blend scaled around that share")
    # The builder names the mixture from the config's stem, so the temp config keeps the stem.
    tmp = work / CFG.name
    tmp.write_text("# ABOUTME: temp copy of configs/data/mixture/da-no-t6.yaml with the source swapped to a local\n"
                   "# ABOUTME: t6-free jsonl; written by scratch/build_da_no_t6_mix.py, never hand-edited.\n"
                   + OmegaConf.to_yaml(cfg), encoding="utf-8")
    print(f">>> {a.repo}@{sha[:8]}: {len(rows)} rows -> {len(kept)} without t6 -> {local}")
    cmd = ["uv", "run", "mix", "--config", str(tmp), f"synthetic_pct={a.pct}"]
    print(">>>", " ".join(cmd), flush=True)
    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
