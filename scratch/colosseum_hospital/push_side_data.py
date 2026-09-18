# ABOUTME: Push the Hospital eval's side data to the Hub: the environment's per-iteration inventory
# ABOUTME: snapshots (which the action rules read) and the mid-shift probe's contexts and answers.
"""The published eval cells hold rollouts, results and metadata. Two things the analysis reads
live beside them and were never pushed: the environment snapshots (`env_logs/<date>_fixed`,
one `data_iteration_<k>.json` per episode and iteration; `actions.py` reads the start-of-
iteration stock from them) and the mid-shift probe (`analysis/2026-09-15_midshift_probe_fixed`,
one folder per probe pod tag: the cut transcripts shown to each seat and its answers).

One repo per snapshot date, dated by the day the episodes ran, and one repo for the probe
set, dated by the day it was assembled (its tags were produced on different days, recorded
in the card). Names are minted by `artifact_name` and gated on push like every other repo.

Run (from the repo root, HF_ORG in .env names the org):
  uv run python scratch/colosseum_hospital/push_side_data.py --what snapshots probes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from src.infra.huggingface import push_run_dir
from src.naming import artifact_name
from src.utils import git_sha

ROOT = Path("output/colosseum_hospital")
ENV = ROOT / "env_logs"
PROBES = ROOT / "analysis" / "2026-09-15_midshift_probe_fixed"
CONSTITUTION = "none"


def _pods(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def push_snapshots(date: str) -> str:
    src = ENV / f"{date}_fixed"
    assert src.is_dir(), f"no snapshots at {src}"
    n = sum(1 for _ in src.rglob("data_iteration_*.json"))
    fields = {
        "experiment": f"colosseum_hospital environment snapshots for the {date} cells: the "
        "HospitalEnvironment's inventory, schedule and agent rewards at the end of every "
        "iteration of every episode, one data_iteration_<k>.json per episode and iteration",
        "date_generated": date,
        "constitution": CONSTITUTION,
        "source_repo": f"teaching_claude_why_replication @ {git_sha()}",
        "models": "the arms of the same-dated colosseum-hospital eval repos on this org; one "
        f"folder per pod: {', '.join(_pods(src))}",
        "generation_config": json.dumps(
            {
                "written_by": "terrarium HospitalEnvironment (patched), end of each iteration",
                "files": n,
            }
        ),
        "schema": "<pod>/HospitalEnvironment/<experiment>/<run ts>__<episode>/seed_<n>/"
        "data_iteration_<k>.json: {inventory: {hospital: {resource: units}}, agent_rewards, "
        "...}. `hospital/actions.py::find_snapshots` indexes them by episode name and seed; "
        "`sabotage_kinds.py` and the harvest read the start-of-iteration stock from them.",
        "provenance": "pulled from each pod's /root/colosseum/logs by "
        "scratch/colosseum_hospital/fleet.py after its runs; pushed by "
        f"`uv run python scratch/colosseum_hospital/push_side_data.py --what snapshots` "
        f"@ {git_sha()}",
    }
    name = artifact_name("colosseum_hospital env snapshots", date=date)
    return push_run_dir(
        src,
        name,
        fields,
        front_matter={"tags": ["colosseum-hospital", "env-snapshots", f"cells:{date}"]},
    )


def push_probes(date: str) -> str:
    assert PROBES.is_dir(), f"no probes at {PROBES}"
    tags = sorted(
        p.name for p in PROBES.iterdir() if p.is_dir() and not p.name.startswith("_")
    )
    counts = {
        t: sum(1 for _ in (PROBES / t / "probes.jsonl").open())
        for t in tags
        if (PROBES / t / "probes.jsonl").is_file()
    }
    arms = sorted(
        {
            json.loads(line)["arm"]
            for t in counts
            for line in (PROBES / t / "probes.jsonl").open()
            if line.strip()
        }
    )
    fields = {
        "experiment": "colosseum_hospital mid-shift probe on the fixed-harness self-sacrificial "
        "cells: at checkpoints 1E, 2P and 2E each coalition seat (and one bystander) is shown "
        "its own transcript so far and asked whether what it did was unethical",
        "date_generated": date,
        "constitution": CONSTITUTION,
        "source_repo": f"teaching_claude_why_replication @ {git_sha()}",
        "models": f"probed arms (h15_midshift_probe.py ARMS keys): {', '.join(arms)}; "
        "each seat is asked by the adapter that played it",
        "generation_config": json.dumps(
            {
                "probes_per_tag": counts,
                "tags_produced": {
                    "e1": "2026-09-15",
                    "new": "2026-09-15",
                    "dat": "2026-09-17",
                },
                "samples_per_seat_and_checkpoint": 3,
            }
        ),
        "schema": "<tag>/probes.jsonl: one record per probe (arm, seed, seat, checkpoint, "
        "variant, model_key, recorded_model_key, verdict yes/no, the reply); "
        "<tag>/contexts/<arm>_<seed>_<seat>_<checkpoint>_<variant>.json: the cut transcript the "
        "seat was shown; _analysis/: the labelled and pointed tables.",
        "provenance": "scratch/colosseum_hospital/h15_midshift_probe.py --arms <arms> --tag <tag> "
        "--server <pod>, per tag; pushed by `uv run python "
        f"scratch/colosseum_hospital/push_side_data.py --what probes` @ {git_sha()}",
    }
    name = artifact_name("colosseum_hospital midshift probes", date=date)
    return push_run_dir(
        PROBES,
        name,
        fields,
        front_matter={
            "tags": ["colosseum-hospital", "midshift-probe"]
            + [f"tag:{t}" for t in tags]
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--what", nargs="+", choices=["snapshots", "probes"], required=True)
    ap.add_argument("--snapshot-dates", nargs="+", default=["2026-09-14", "2026-09-15"])
    ap.add_argument(
        "--probe-date", default="2026-09-18", help="the day the probe set was assembled"
    )
    args = ap.parse_args()
    load_dotenv()
    if "snapshots" in args.what:
        for d in args.snapshot_dates:
            print(">>> pushed", push_snapshots(d))
    if "probes" in args.what:
        print(">>> pushed", push_probes(args.probe_date))


if __name__ == "__main__":
    main()
