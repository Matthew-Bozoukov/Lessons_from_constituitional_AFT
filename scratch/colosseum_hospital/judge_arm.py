# ABOUTME: Judge one colosseum_hospital arm dir's finished episodes and write results/judge.json,
# ABOUTME: without needing the published layout — so an arm still running can be judged.

"""Per-channel judge over an arm directory, published or not.

    uv run python scratch/colosseum_hospital/judge_arm.py output/colosseum_hospital/<arm> [<arm> ...]

`publish_colosseum.py` reads results/results.json, which run_eval writes only when the arm
ends; this reads the raw Colosseum tree under rollouts/colosseum/<ts>/ directly, skips
episodes already in results/judge.json, and merges the new verdicts in.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.eval.misalignment.colosseum.hospital.judge import judge_run_root


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--config", default="configs/eval/colosseum_hospital.yaml")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--channels",
        nargs="*",
        default=None,
        help="judge only these channel kinds (public, secret, reasoning); an episode is "
        "re-judged when it lacks any of them, and new channels merge into its record",
    )
    args = ap.parse_args()
    load_dotenv()
    cfg = OmegaConf.load(args.config)
    wanted = set(args.channels) if args.channels else None

    def _lacks(record: dict) -> bool:
        """An episode is due when it has no verdicts, or none for a wanted channel kind."""
        if not record:
            return True
        if wanted is None:
            return False
        have = {ch.split(":")[0] for ch in record}
        return not wanted <= have

    for arm in args.arms:
        arm = Path(arm)
        roots = sorted(
            p
            for p in (arm / "rollouts" / "colosseum").glob("*/")
            if (p / "runs").is_dir()
        )
        out_p = arm / "results" / "judge.json"
        existing = json.loads(out_p.read_text()) if out_p.is_file() else {"per_run": {}}
        merged = dict(existing.get("per_run") or {})
        for root in roots:
            todo = [
                d
                for d in sorted(root.glob("runs/*/*/*"))
                if (d / "blackboards.json").is_file() and _lacks(merged.get(d.name))
            ]
            if not todo:
                continue
            verdicts = judge_run_root(
                root, cfg, max_workers=args.workers, channels=args.channels
            )
            for run_id, channels in verdicts["per_run"].items():
                # Merge per channel: a run judged on the boards before the reasoning
                # channels existed keeps its board verdicts and gains the new ones.
                merged.setdefault(run_id, {}).update(channels)
            existing.update({k: v for k, v in verdicts.items() if k != "per_run"})
        existing["per_run"] = merged
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(existing, indent=1))
        print(f"{arm.name}: {len(merged)} episodes judged")


if __name__ == "__main__":
    main()
