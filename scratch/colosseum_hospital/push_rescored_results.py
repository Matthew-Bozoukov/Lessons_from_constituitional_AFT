# ABOUTME: Pushes an autorater-rescored Hospital cell's results/ back to its Hub repo (one atomic
# ABOUTME: commit per cell): autorater reading in results/, the keyword reading kept in results/keyword/.
# Run (repo root):
#   uv run python scratch/colosseum_hospital/push_rescored_results.py <cell dir> [...] [--dry-run]
"""2026-10-15: the six cells of the paper's Hospital figures were re-read by the autorater
(scratch/colosseum_hospital/rescore_autorater.py) in LOCAL copies pulled from the Hub, so the Hub
still showed the keyword-rule reading (da-tools-15: 23/30 acts on the Hub, 20/30 under the
autorater). This uploads, per cell:

  results/{autorater,per_seed,episodes,results}.json + results.md   the autorater reading
  results/keyword/{per_seed,episodes,results}.json + results.md     the reading published before
  rollouts/, metadata/ files the Hub lacks                           (a cell merged from two trees)
  README.md                                                          the card + `autorater` rows

Nothing else in results/ goes up (subagent labellings and the Fig 4 post judge stay local). The
name gate (`gate_push`) runs first; `date_generated` is the run's own date, unchanged."""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path

import yaml
from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd

from src.infra.huggingface import card_markdown, gate_push, hf_api, hf_org

RESULTS_FILES = (
    "autorater.json",
    "per_seed.json",
    "episodes.json",
    "results.json",
    "results.md",
    "judge.json",
)
KEYWORD_FILES = ("per_seed.json", "episodes.json", "results.json", "results.md")
SKIP_DIRS = {"results", "results_regex", ".cache"}


def parse_card(text: str) -> tuple[dict, dict]:
    """The card's front matter (tags) and its field table, as `card_markdown` would take them."""
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    front = yaml.safe_load(m.group(1)) if m else {}
    fields: dict = {}
    for key, value in re.findall(r"^\| `([^`]+)` \| (.*) \|$", text, re.M):
        if key == "field":
            continue
        fields[key] = value
    title = re.search(r"^# (.+)$", text, re.M)
    if title:
        fields["title"] = title.group(1)
    return front or {}, fields


def autorater_rows(cell: Path) -> dict:
    lab = json.loads((cell / "results" / "autorater.json").read_text())
    old = json.loads((cell / "results_regex" / "per_seed.json").read_text())
    new = json.loads((cell / "results" / "per_seed.json").read_text())

    def acts(p):
        vals = [v for c in p["pair_sabotaged"].values() for v in c.values()]
        return f"{sum(1 for v in vals if v)}/{sum(1 for v in vals if v is not None)}"

    return {
        "autorater": (
            f"{lab['model']} (prompt v{lab['prompt_version']}, {lab['n_calls']} calls, "
            f"{lab['unparsed']} unparsed) re-read the pair seats' reasoning, board posts and "
            f"shortage reports on {date.today().isoformat()}; results/ is that reading "
            f"(fixes.text_reader=autorater, the eval's default since 2026-10-15). "
            f"results/keyword/ keeps the keyword-rule reading this repo published first. "
            f"Sabotage acts: {acts(old)} keyword -> {acts(new)} autorater."
        ),
        "rescore_provenance": (
            "uv run python scratch/colosseum_hospital/rescore_autorater.py <this cell> "
            "--workers 6 && uv run python scratch/colosseum_hospital/push_rescored_results.py "
            "<this cell>"
        ),
    }


def plan(cell: Path, api) -> tuple[str, list[CommitOperationAdd], dict, dict]:
    repo_id = f"{hf_org()}/{cell.name}"
    front, fields = parse_card((cell / "README.md").read_text())
    fields.update(autorater_rows(cell))
    gate_push(repo_id, fields, what="rescored results push")
    on_hub = set(api.list_repo_files(repo_id, repo_type="dataset"))
    ops = []
    for name in RESULTS_FILES:
        p = cell / "results" / name
        if p.is_file():
            ops.append(CommitOperationAdd(f"results/{name}", str(p)))
    for name in KEYWORD_FILES:
        p = cell / "results_regex" / name
        if p.is_file():
            ops.append(CommitOperationAdd(f"results/keyword/{name}", str(p)))
    missing = []
    for p in sorted(cell.rglob("*")):
        rel = p.relative_to(cell)
        if not p.is_file() or rel.parts[0] in SKIP_DIRS or rel.name == "README.md":
            continue
        if rel.as_posix() not in on_hub:
            missing.append(rel.as_posix())
            ops.append(CommitOperationAdd(rel.as_posix(), str(p)))
    if missing:
        # The control cell (2026-10-11 plain): seeds 1-15 sit in the Hub's first Colosseum
        # tree, which the published results never harvested; they were copied beside the
        # harvested tree so one harvest (results/episodes.json) names every shift.
        prefix = Path(missing[0]).parts[:3]
        fields["rollouts_added"] = (
            f"{len(missing)} files under {'/'.join(prefix)}: the cell's other Colosseum "
            "tree's episodes, copied into the harvested tree so results/ covers every shift"
        )
    card = card_markdown(fields, front)
    ops.append(CommitOperationAdd("README.md", card.encode("utf-8")))
    return repo_id, ops, {"missing": missing, "on_hub": len(on_hub)}, fields


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cells", nargs="+", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    load_dotenv()
    api = hf_api()
    for cell in args.cells:
        assert (cell / "results" / "autorater.json").is_file(), f"{cell}: not rescored"
        repo_id, ops, info, fields = plan(cell, api)
        print(
            f"\n=== {repo_id}: {len(ops)} files ({len(info['missing'])} the Hub lacked, "
            f"{info['on_hub']} on the Hub)"
        )
        print("  " + fields["autorater"])
        for m in info["missing"][:6]:
            print("  +", m)
        if len(info["missing"]) > 6:
            print(f"  + ... {len(info['missing']) - 6} more")
        if args.dry_run:
            continue
        info_ = api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=ops,
            commit_message="Autorater re-read of the text measures (results/); keyword "
            "reading kept in results/keyword/",
        )
        print("  pushed:", info_.commit_url)


if __name__ == "__main__":
    main()
