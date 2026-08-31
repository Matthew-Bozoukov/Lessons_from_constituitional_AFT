# ABOUTME: Lists every numina-control seed-0 cell still missing after the repair passes,
# ABOUTME: and writes per-pass exclusion lists for a re-run at a larger context window.
"""Which cells are still missing, and why.

    uv run python scratch/collect_repair_gaps.py --repair_dir <scratchpad>/repair

A cell is missing when no messages_record.txt exists for it in EITHER the original
combined4x tree or the repair pass that targeted it. The cause is read off the pass's
docker_output.log so a context overflow (fixable by serving a larger window) is not
confused with a proxy timeout or a genuine harness error.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import fire

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from omegaconf import OmegaConf  # noqa: E402

from src.eval.misalignment.odcv.odcv import VARIANTS, scenario_names  # noqa: E402

COMBINED = REPO / ("output/odcv_bench/qwen3_6-27b-lora-9284-numina-control-716-r64/"
                   "combined4x_20260819_140227")
CELL = re.compile(r"-(mandated|incentivized)/experiments/([^/]+)(?:/rollout_(\d+))?/")
OVERFLOW = re.compile(r"maximum context length is (\d+)|prompt contains (?:at least )?(\d+)")


def main(repair_dir: str, out: str = "") -> None:
    """Report still-missing cells per pass, with the reason each failed."""
    cfg = OmegaConf.load(REPO / "configs/eval/odcv_bench_9284_numina_control_716_r64_dynbatch_3x65.yaml")
    bench = REPO / "third_party/odcv-bench"
    excl = set(cfg.exclude_scenarios)
    expected = {f"{v}/{s}" for v in VARIANTS for s in scenario_names(bench, v)
                if f"{v}/{s}" not in excl}
    allcells = {f"{v}/{s}" for v in VARIANTS for s in scenario_names(bench, v)}

    have: dict[str, set[str]] = defaultdict(set)
    for p in COMBINED.rglob("messages_record.txt"):
        m = CELL.search(str(p))
        if m and m.group(3):
            have[m.group(3)].add(f"{m.group(1)}/{m.group(2)}")

    rd = Path(repair_dir)
    reasons: dict[str, str] = {}
    for idx in ("000", "001", "002", "003"):
        for run in sorted((rd / f"out_{idx}").rglob("messages_record.txt")):
            m = CELL.search(str(run))
            if m:
                have[idx].add(f"{m.group(1)}/{m.group(2)}")
        for log in sorted((rd / f"out_{idx}").rglob("docker_output.log")):
            if (log.parent / "messages_record.txt").is_file():
                continue
            m = CELL.search(str(log) + "/")
            if not m:
                continue
            txt = log.read_text(errors="ignore")[-40000:]
            hit = OVERFLOW.search(txt)
            reasons[f"{idx}:{m.group(1)}/{m.group(2)}"] = (
                "context-overflow" if hit else
                "proxy-524" if "524" in txt else "unknown")

    gaps, plan = {}, {}
    for idx in ("000", "001", "002", "003"):
        miss = sorted(expected - have[idx])
        gaps[idx] = miss
        plan[idx] = sorted(allcells - set(miss))
        why = defaultdict(int)
        for c in miss:
            why[reasons.get(f"{idx}:{c}", "never-attempted")] += 1
        print(f"pass {idx}: {len(have[idx])}/{len(expected)} present, {len(miss)} MISSING "
              f"{dict(why)}")
        for c in miss:
            print(f"    {c}  [{reasons.get(f'{idx}:{c}', 'never-attempted')}]")
    total = sum(len(v) for v in gaps.values())
    print(f"\nTOTAL still missing: {total}")
    if out:
        Path(out).write_text(json.dumps({"gaps": gaps, "exclude": plan}, indent=1))
        print(f">>> wrote {out}")


if __name__ == "__main__":
    fire.Fire(main)
