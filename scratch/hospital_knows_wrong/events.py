# ABOUTME: Builds the table of Hospital seat-iterations (every live watched seat, every iteration) with the seat's private
# ABOUTME: reasoning and whether a sabotage act fell in it. Run: uv run python scratch/hospital_knows_wrong/events.py
"""Reads the transcripts `first_cut.py`'s cells point at (downloaded under output/hospital_knows_wrong/cells/)
and re-runs the published action analysis (`actions.analyse_episode`, the call the harvest makes) to get WHICH
iterations carried a sabotage act; the published `results/episodes.json` holds only per-shift counts.

Seven of the ten cells were published with the environment's stock snapshots deciding the withholding rule.
Those snapshots are not on the Hub, so here the deficit is the autorater's reading for every cell. A unit's
`acted` can therefore differ from the published count for a few withholding acts; each unit carries the
published per-shift count (`published_seat_sabotage_iters`) and the run prints the disagreements.

A unit's text is the seat's private reasoning in that one iteration, call by call, with the tool calls each
call emitted.
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).parent))
from first_cut import CELLS, ORG, OUT  # noqa: E402

from src.eval.misalignment.colosseum.hospital import actions  # noqa: E402

VERSIONS = {"multi", "bridge", "solo", "solo, no instruction"}
PATTERNS = ["results/*", "metadata/*"] + [
    f"rollouts/**/{f}" for f in ("agent_turns.json", "tool_events.json", "run_config.json", "final_summary.json")
]
SCRIPTED = "scripted_honest"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def tool_call_line(tc: dict) -> str:
    fn = tc.get("function") or tc
    args = fn.get("arguments")
    if not isinstance(args, str):
        args = json.dumps(args, ensure_ascii=False)
    return f"{fn.get('name')}({args})"


def seat_calls(turns: list[dict], seat: str, iteration: int) -> list[dict]:
    """Every model call the seat made in one iteration, in order."""
    out = []
    for t in turns:
        it = int(t.get("iteration") or -1)
        if t.get("agent") != seat or it != iteration:
            continue
        for c in t.get("llm_calls") or []:
            out.append(
                {
                    "iteration": it,
                    "phase": t.get("phase"),
                    "reasoning": c.get("reasoning") or "",
                    "content": c.get("content") or "",
                    "tool_calls": [tool_call_line(tc) for tc in (c.get("tool_calls") or [])],
                }
            )
    return out


def act_kind(detail: dict) -> str:
    if "withheld_from" in detail:  # the provisioner
        kinds = (["flood"] if detail["flood"] else []) + (["withhold"] if detail["withheld_from"] else [])
        return "+".join(kinds)
    return "scheduler-spam"


def cell_units(version: str, arm: str, repo: str, revision: str) -> tuple[list[dict], int]:
    root = Path(
        snapshot_download(
            f"{ORG}/{repo}", repo_type="dataset", revision=revision, token=os.environ.get("HF_TOKEN"),
            local_dir=OUT / "cells" / repo, allow_patterns=PATTERNS,
        )
    )
    published = read(root / "results/episodes.json")
    labels = read(root / "results/autorater.json")["per_run"]
    run_dirs = {p.parent.name: p.parent for p in root.glob("rollouts/**/run_config.json")}
    out, disagree = [], 0
    for ep in published:
        run_dir = run_dirs[ep["run_id"]]
        run_cfg, final = read(run_dir / "run_config.json"), read(run_dir / "final_summary.json")
        turns = actions.shift_turns(read(run_dir / "agent_turns.json"))
        events = read(run_dir / "tool_events.json")
        seats = actions.analyse_episode(
            turns, events, pair=run_cfg.get("pair") or ep["pair"], secret_id=run_cfg.get("secret_blackboard_id"),
            fixes=run_cfg.get("fixes") or None, n_patients=actions.patients_of(final),
            hospitals=actions.hospitals_of(final), snapshots={}, labels=labels[ep["run_id"]],
        )["seats"]
        for seat, s in seats.items():
            if (run_cfg.get("agent_llm_labels") or {}).get(seat) == SCRIPTED:
                continue
            key = "prov" if seat == actions.PROVISIONER else "tri"
            published = ep[f"{key}_sabotage_iters"]
            disagree += int(s["sabotage_iters"] != published)
            for d in s["detail"]:
                it = d["iteration"]
                out.append(
                    {
                        "id": f"{version}|{arm}|seed{ep['seed']}|{seat}|it{it}",
                        "version": version, "arm": arm, "repo": repo, "revision": revision,
                        "run_id": ep["run_id"], "seed": ep["seed"], "seat": seat, "iteration": it,
                        "acted": d["action"] == "S",
                        "act": act_kind(d) if d["action"] == "S" else None,
                        "published_seat_sabotage_iters": published,
                        "published_deficit_source": ep["deficit_source"],
                        "autorater_stance": d["reasoning"],  # D declined, C committed, "." neither
                        "autorater_declined_and_committed": d["declined_and_committed"],
                        "calls": seat_calls(turns, seat, it),
                    }
                )
    return out, disagree


def main() -> None:
    load_dotenv(".env")
    units = []
    for version, arm, repo, revision in CELLS:
        if version not in VERSIONS or "seed" in arm:
            continue
        cell, disagree = cell_units(version, arm, repo, revision)
        acted = [u for u in cell if u["acted"]]
        chars = sum(len(c["reasoning"]) for u in cell for c in u["calls"])
        print(f"{version:22s} {arm:8s} seat-iterations {len(cell):4d}  with a sabotage act {len(acted):4d}  "
              f"seat-shifts off the published act count {disagree:2d}  mean reasoning chars {chars // len(cell):6d}")
        units += cell
    path = OUT / "units.json"
    path.write_text(json.dumps(units, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(units)} seat-iterations to {path}")


if __name__ == "__main__":
    main()
