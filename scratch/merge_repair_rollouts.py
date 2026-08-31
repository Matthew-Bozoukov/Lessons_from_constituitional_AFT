# ABOUTME: Merges repaired ODCV rollouts into an existing combined4x tree at their correct
# ABOUTME: rollout_NNN slot, recording the serving context each repaired cell was run at.
"""Fold repair-pass transcripts back into the combined run directory.

    uv run python scratch/merge_repair_rollouts.py --repair <dir> --ctx 16384 [--apply]

Copies, never moves: the repair output stays intact so a bad merge can be redone. Refuses
to overwrite an existing transcript -- a cell present in the combined tree was already
good, and silently replacing it would change data the judge has already scored.

Each merged cell gets a `repair_meta.json` naming the serving context window it was run at,
because the 65,536-token repairs are NOT the same serving condition as the original 16,384
rollouts and a reader must be able to tell which is which.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import fire

REPO = Path(__file__).resolve().parents[1]
COMBINED = REPO / ("output/odcv_bench/qwen3_6-27b-lora-9284-numina-control-716-r64/"
                   "combined4x_20260819_140227")
MODEL_KEY = "qwen3_6-27b-lora-9284-numina-control-716-r64"
CELL = re.compile(r"-(mandated|incentivized)/experiments/([^/]+)/messages_record\.txt$")


def main(repair: str, ctx: int, apply: bool = False) -> None:
    """Merge one repair directory's transcripts into the combined tree.

    Args:
        repair: A `<scratch>/repair*/` root holding out_<idx>/ subdirs.
        ctx: The serving context window those rollouts were produced at.
        apply: Actually copy. Default is a dry run.
    """
    root = Path(repair)
    merged = skipped = 0
    for idx in ("000", "001", "002", "003"):
        for rec in sorted((root / f"out_{idx}").rglob("messages_record.txt")):
            m = CELL.search(str(rec))
            if not m:
                continue
            variant, scenario = m.group(1), m.group(2)
            dst = (COMBINED / "agent_logs" / f"{MODEL_KEY}-{variant}" / "experiments"
                   / scenario / f"rollout_{idx}")
            target = dst / "messages_record.txt"
            if target.is_file():
                skipped += 1
                continue
            if apply:
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy2(rec, target)
                src_log = rec.parent / "docker_output.log"
                if src_log.is_file():
                    shutil.copy2(src_log, dst / "docker_output.log")
                (dst / "repair_meta.json").write_text(json.dumps({
                    "repaired": True, "pass": f"rollout_{idx}",
                    "serving_context_window": ctx,
                    "note": ("re-run to fill a gap in the original run; a 65536 window is a "
                             "DIFFERENT serving condition from the original 16384 rollouts"),
                }, indent=2))
            merged += 1
            print(f"  {'merge ' if apply else 'would '}{variant}/{scenario} -> rollout_{idx}")
    print(f"\n{'merged' if apply else 'would merge'} {merged}, skipped {skipped} "
          f"(already present)  [ctx={ctx}]")


if __name__ == "__main__":
    fire.Fire(main)
