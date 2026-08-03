# ABOUTME: Builds manifest.json + sharded transcripts so the Visualizer can load this run from HF.
# ABOUTME: Mirrors the schema of the 2026-07-29 focused-discovery dataset, verified against the live repo.
"""Produce the Hugging Face side of the export.

The Visualizer's build fetches ONLY `manifest.json` and, for each transcript the
reader opens, `transcripts/<slug>.json` straight from the Hub CDN. It never
downloads `transcripts.jsonl`. So an export that ships only the bulk JSONL
indexes as "unavailable" and silently falls back to the on-disk copy - which
defeats the point of publishing.

Schema copied from the live 2026-07-29 focused-discovery manifest rather than
from prose, so the field names are what the loader actually reads.

The run's own description (experiment, caveats, model roles, generation config)
comes from a YAML file, NOT from literals in here. It used to be literals, which
meant the only way to publish a second run was to overwrite the prose describing
the first - and a caveat like "n = 10" outlives by months the run it was true for.

Usage:
    python scripts/build_manifest.py --export exports/2026-08-01-constitution-dose-sweep-v2 \
        --meta configs/manifest_v2.yaml --commit <sha>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml


def slugify(value: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return re.sub(r"-{2,}", "-", s)


def build(export: Path, meta_path: Path, commit: str, dirty: str = "") -> None:
    # Plain pyyaml, not OmegaConf: this is flat publication metadata with no
    # interpolation, and the petri venv already has pyyaml.
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    scenarios = [json.loads(l) for l in (export / "data" / "scenarios.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    transcripts = [json.loads(l) for l in (export / "results" / "transcripts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    scores = json.loads((export / "results" / "scores.json").read_text(encoding="utf-8"))

    shard_dir = export / "transcripts"
    shard_dir.mkdir(exist_ok=True)
    total_bytes = 0
    entries = []
    for t in transcripts:
        name = f"{slugify(t['id'])}.json"
        payload = json.dumps(t, ensure_ascii=False, indent=1)
        (shard_dir / name).write_text(payload, encoding="utf-8")
        total_bytes += len(payload.encode("utf-8"))
        entries.append({
            "id": t["id"],
            "file": name,
            "scenario_id": t["scenario_id"],
            "category": t["category"],
            "outcome": t["outcome"],
            "scores": t["scores"],
            "tags": t["tags"],
        })

    repo = dict(meta["source_repo"])
    repo["commit"] = commit
    if dirty:
        # Publishing from a dirty tree means `commit` does NOT reproduce this export.
        # Say so in the artifact rather than letting a reader assume it does.
        repo["working_tree"] = (
            "DIRTY at publish time - the analysis/export scripts that produced this "
            "dataset were uncommitted. `commit` is the branch tip, not a reproducible "
            "pointer to the code that generated these files: " + dirty
        )

    manifest = {
        "manifest_version": 1,
        "kind": "petri-run",
        "experiment": meta["experiment"],
        "date_generated": meta["date_generated"],
        "constitution": meta["constitution"],
        "source_repo": repo,
        "models": meta["models"],
        "generation_config": meta["generation_config"],
        "provenance": meta["provenance"],
        "caveats": meta["caveats"],
        "scenarios": scenarios,
        "scores": scores,
        "transcripts": entries,
        "transcript_count": len(entries),
        "total_transcript_bytes": total_bytes,
    }

    out = export / "manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    size_kb = out.stat().st_size / 1024
    print(f"manifest.json      : {size_kb:.1f} KB ({len(entries)} transcripts, {len(scenarios)} scenarios)")
    print(f"transcript shards  : {len(entries)} files, {total_bytes/1024:.1f} KB")
    if size_kb > 512:
        print("WARNING: manifest exceeds the 512 KB the publisher warns at")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--meta", required=True, help="YAML run metadata, e.g. configs/manifest_v2.yaml")
    ap.add_argument("--commit", required=True, help="git SHA the export was built from")
    ap.add_argument("--dirty", default="", help="uncommitted paths, recorded in the manifest if set")
    a = ap.parse_args()
    build(Path(a.export), Path(a.meta), a.commit, a.dirty)


if __name__ == "__main__":
    main()
