#!/usr/bin/env python3
# ABOUTME: Publish an adapter directory whose training run finished but whose push did not
# ABOUTME: (or whose pod had no token), under the organism name its training_meta.json records.
# Run (on the box that holds the adapter, with HF_TOKEN + HF_ORG in .env):
#   uv run python scratch/push_saved_adapter.py --adapter-dir output/train/<run>/adapter [--organism <name>]
#
# 2026-09-06: the first nosynth run trained for 2h37m and then lost its push to a name
# gate that ran before the org was prefixed. The adapter, its stamp and its resolved config
# were all on disk; this re-uses `src.train.launch.push_adapter`, so the card is the one
# the run would have written. `--organism` covers stamps from before the name was recorded.
# If the run directory holds a final checkpoint, its trainer_state.json supplies the loss
# history the run's own run_meta.json would have carried.

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from src.train.launch import push_adapter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-dir", required=True)
    ap.add_argument("--organism", default="", help="override/backfill training_meta.organism")
    args = ap.parse_args()
    load_dotenv()
    adapter_dir = Path(args.adapter_dir)
    meta = json.loads((adapter_dir / "training_meta.json").read_text())
    if args.organism:
        meta["organism"] = args.organism
        (adapter_dir / "training_meta.json").write_text(json.dumps(meta, indent=2))
        print(f">>> stamped organism {args.organism!r} into training_meta.json")
    ckpts = sorted(adapter_dir.parent.glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[-1]))
    if ckpts and (ckpts[-1] / "trainer_state.json").exists():
        state = json.loads((ckpts[-1] / "trainer_state.json").read_text())
        run_meta = adapter_dir.parent / "run_meta.json"
        if not run_meta.exists():
            run_meta.write_text(json.dumps({
                "git_sha": meta["git_sha"], "recipe": meta["recipe"],
                "base_model": meta["base_model"],
                "base_model_revision": meta.get("base_model_revision"),
                "dataset": meta["dataset"], "config": meta["train_config"],
                "timestamp": meta["timestamp"],
                "log_history": state.get("log_history", []),
                "note": f"written by scratch/push_saved_adapter.py from {ckpts[-1].name}",
            }, indent=2))
            print(f">>> run_meta.json backfilled from {ckpts[-1].name} "
                  f"({len(state.get('log_history', []))} log rows)")
    url = push_adapter(adapter_dir, meta)
    print(f">>> pushed {url}")


if __name__ == "__main__":
    main()
