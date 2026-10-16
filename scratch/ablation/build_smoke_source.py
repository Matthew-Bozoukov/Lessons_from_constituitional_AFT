# ABOUTME: Stage a few rows of a finished difficult-advice run as a local source directory,
# ABOUTME: so `synth run --smoke` on a source-seeded config rewrites 3 rows instead of 642.
"""A smoke-sized source for the ablation pipeline.

    uv run python scratch/ablation/build_smoke_source.py [--rows 3] [--repo <hf id>]

`load_source_run` reads a whole snapshot: the row count IS the file, so `--smoke` cannot
shrink it the way it shrinks a generating config (learned on 2026-08-26 when a smoke that
only lowered `total_scenarios` ran a full 716-row rewrite). The repo-wide answer is to
point the smoke at a small staged directory, which is what this writes.

The manifest carries the source's `constitution_sha256` because `load_source_run` asserts
it against the config's constitution: a smoke whose source was generated against a
different document would cross arms silently, and the assert is what stops that.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import hf_hub_download

REPO = Path(__file__).resolve().parents[2]
SNAPSHOT = "stage_6_revise_responses.jsonl"


def main() -> None:
    """Write the staged smoke source."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="dougalldeepmind/2026-10-02-da-synth")
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--out", default="data/da_1002_ablation_smoke")
    args = ap.parse_args()

    tok = dotenv_values(REPO / ".env")["HF_TOKEN_MATBOZ"]
    snap = hf_hub_download(args.repo, f"stages/{SNAPSHOT}", repo_type="dataset", token=tok)
    manifest = json.loads(Path(hf_hub_download(args.repo, "manifest.json",
                                               repo_type="dataset", token=tok)).read_text())
    rows = [json.loads(l) for l in open(snap, encoding="utf-8")]

    # One row per distinct `ai` level where possible: the ablation rewrites reasoning about
    # a harm, and a scenario with no AI system in it exercises a different kind of harm than
    # one about the assistant itself. A smoke that only saw `self` rows would not show that.
    picked: list[dict] = []
    seen: set[str] = set()
    for r in rows:
        level = r.get("ai", "open")
        if level not in seen:
            picked.append(r)
            seen.add(level)
        if len(picked) == args.rows:
            break
    for r in rows:
        if len(picked) >= args.rows:
            break
        if r not in picked:
            picked.append(r)

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / SNAPSHOT).write_text("".join(json.dumps(r) + "\n" for r in picked), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({
        "constitution_sha256": manifest.get("constitution_sha256"),
        "staged_from": {"repo": args.repo, "snapshot": SNAPSHOT,
                        "run_id": manifest.get("run_id")},
        "n_records": len(picked),
        "note": "smoke source for configs/data/synth/ablation.yaml; rows are verbatim",
    }, indent=1), encoding="utf-8")
    print(f">>> {len(picked)} rows -> {out}/{SNAPSHOT}")
    print(f">>> ai levels: {[r.get('ai') for r in picked]}")
    print(f">>> constitution_sha256: {str(manifest.get('constitution_sha256'))[:16]}…")
    for r in picked:
        print(f"    {r['scenario_id']}  reasoning {len(r['reasoning']):5d}  "
              f"response {len(r['response']):5d}  ai={r.get('ai')}")


if __name__ == "__main__":
    main()
