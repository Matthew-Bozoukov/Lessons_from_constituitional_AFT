# ABOUTME: Print a synth run's rows for reading: prompt, planned call, reviser note, and optionally
# ABOUTME: the trained reasoning and reply. `uv run python scratch/dump_rows.py <run_dir> [--full] [--traits t6]`
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--full", action="store_true", help="also print reasoning and response")
    ap.add_argument("--traits", default="", help="comma-separated trait ids to show")
    ap.add_argument("--width", type=int, default=1200)
    a = ap.parse_args()
    run = Path(a.run_dir)
    notes = {}
    for f in run.glob("stage_*_revise_prompts.jsonl"):
        for line in f.open(encoding="utf-8"):
            r = json.loads(line)
            notes[r["scenario_id"]] = r.get("refine_changes", "")
    want = {t for t in a.traits.split(",") if t}
    rows = [json.loads(line) for line in (run / "dataset.jsonl").open(encoding="utf-8")]
    for r in rows:
        md = r["metadata"]
        if want and md.get("trait_id") not in want:
            continue
        user = next(m["content"] for m in r["messages"] if m["role"] == "user")
        print("=" * 100)
        print(f"[{md['scenario_id']}] {md.get('trait_name', '')} | domain={md.get('domain', '')}")
        print("CALL:", md.get("shortcut", "")[: a.width])
        print("CHANGES:", notes.get(md["scenario_id"], "")[: a.width])
        print("USER:", user[: a.width])
        if a.full:
            asst = [m for m in r["messages"] if m["role"] == "assistant"][-1]
            print("REASONING:", (asst.get("reasoning_content") or "")[: a.width])
            print("RESPONSE:", asst["content"][: a.width])
        print()


if __name__ == "__main__":
    main()
