# ABOUTME: Stage the WHOLE 2026-10-02 da corpus as one stage-6-shaped source directory, so the
# ABOUTME: ablation can restructure all 1,283 rows rather than one generation run's share.
"""All of the corpus, in the schema `load_source_run` reads.

    uv run python scratch/ablation/build_full_source.py [--rows N]

`dataset.jsonl` for 2026-10-02-da-synth holds 1,283 rows: the original run (`t1_…` scenario
ids) plus the 2026-10-02 re-push (`r2_t1_…`). Each run's `stages/stage_6_revise_responses.jsonl`
covers only its own rows, so pointing a config at the latest stage snapshot restructures 637
of the 1,283 -- which is what happened on the first full pass, and 618 surviving rows cannot
fund a 15% supervised-token share.

This flattens `dataset.jsonl` back into the stage-6 field shape (`system`, `user`, `reasoning`,
`response` plus the metadata the export and the prompt interpolate), writes it as a local
source directory, and carries the corpus's `constitution_sha256` so `load_source_run`'s
hard assert still protects against crossing arms.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from dotenv import dotenv_values
from huggingface_hub import HfApi, hf_hub_download

REPO = Path(__file__).resolve().parents[2]
SNAPSHOT = "stage_6_revise_responses.jsonl"
# Metadata the ablation's prompt and its chat_export both name.
META = ("scenario_id", "trait_id", "trait_name", "trait_text", "domain", "shortcut",
        "situation", "ai", "sector", "chunk_ids", "granularity", "grouping_strategy",
        "n_chunks")


def flatten(row: dict) -> dict | None:
    """One dataset.jsonl row in the stage-6 field shape, or None if a part is missing."""
    msgs = row["messages"]
    system = next((m["content"] for m in msgs if m["role"] == "system"), None)
    user = next((m["content"] for m in msgs if m["role"] == "user"), None)
    reasoning = next((m.get("reasoning_content") for m in msgs
                      if m["role"] == "assistant" and m.get("reasoning_content")), None)
    response = next((m["content"] for m in reversed(msgs) if m["role"] == "assistant"), None)
    if not all((system, user, reasoning, response)):
        return None
    md = row.get("metadata") or {}
    return {"system": system, "user": user, "reasoning": reasoning, "response": response,
            **{k: md.get(k) for k in META}}


def main() -> None:
    """Write the combined source directory."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="dougalldeepmind/2026-10-02-da-synth")
    ap.add_argument("--revision", default="305914d58627457002f61d86ba69866f6fa2b9a9")
    ap.add_argument("--out", default="data/da_1002_full_source")
    ap.add_argument("--rows", type=int, default=None, help="cap, for a smoke")
    args = ap.parse_args()

    tok = dotenv_values(REPO / ".env")["HF_TOKEN_MATBOZ"]
    api = HfApi(token=tok)
    manifest = json.loads(Path(hf_hub_download(args.repo, "manifest.json", repo_type="dataset",
                                               token=tok, revision=args.revision)).read_text())
    src = [json.loads(l) for l in open(
        hf_hub_download(args.repo, "dataset.jsonl", repo_type="dataset",
                        token=tok, revision=args.revision), encoding="utf-8")]
    flat = [f for f in (flatten(r) for r in src) if f]
    dropped = len(src) - len(flat)
    assert dropped == 0, f"{dropped} rows lacked a system/user/reasoning/response part"
    if args.rows:
        flat = flat[: args.rows]

    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / SNAPSHOT).write_text("".join(json.dumps(r) + "\n" for r in flat), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({
        "constitution_sha256": manifest.get("constitution_sha256"),
        "staged_from": {"repo": args.repo, "revision": args.revision, "file": "dataset.jsonl"},
        "n_records": len(flat),
        "note": "dataset.jsonl flattened into the stage-6 field shape; rows verbatim",
    }, indent=1), encoding="utf-8")

    gens = Counter("r2" if str(r["scenario_id"]).startswith("r2_") else "r1" for r in flat)
    print(f">>> {len(flat)} rows -> {out}/{SNAPSHOT}")
    print(f">>> generation runs represented: {dict(gens)}")
    print(f">>> ai axis: {dict(Counter(r['ai'] for r in flat))}")
    print(f">>> constitution_sha256 {str(manifest.get('constitution_sha256'))[:16]}…")


if __name__ == "__main__":
    main()
