# ABOUTME: Publishes a repaired da-gpt-{prose,voice,principled} corpus as a new revision of its own synth repo: replaces
# ABOUTME: dataset.jsonl, adds repair/ (repair.json, both audits, the rerun stage), and appends a Repair section to the card.
"""Run: uv run python scratch/gpt_vs_claude_da/publish_repaired.py --run_dir <full run> --redo_dir <redo run>

The repo keeps its original revision (the unrepaired 722 rows) in history; a mixture pins the repaired one.
"""
import argparse
import json
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

from scratch.gpt_vs_claude_da.redo_rows import arm, flagged, missing
from src.infra.huggingface import gate_push


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--redo_dir", required=True)
    args = ap.parse_args()
    load_dotenv()
    run, redo = Path(args.run_dir), Path(args.redo_dir)
    fixed = run.parent / f"{run.name}_repaired"
    repo = json.loads((run / "hf_repo.json").read_text(encoding="utf-8"))["repo"]
    gate_push(repo, {"date_generated": run.name}, what="repaired synth corpus")
    note = json.loads((fixed / "repair.json").read_text(encoding="utf-8"))
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    kind = arm(run)
    audit_file = f"audit_{kind}.jsonl"
    n_run = sum(1 for _ in open(run / "dataset.jsonl", encoding="utf-8"))
    n_flag, n_miss = len(flagged(run)), len(missing(run))
    why = "unfaithful or still structured" if kind == "prose" else "unfaithful"

    api = HfApi()
    card = Path(hf_hub_download(repo, "README.md", repo_type="dataset")).read_text(encoding="utf-8")
    card += (
        "\n\n## Repair (" + note["base_run"] + ")\n\n"
        f"`dataset.jsonl` in this revision has **{note['n_rows']} rows**, not the run's {n_run}. A Gemini 3.8 Flash "
        f"fidelity audit (`repair/{audit_file}`) flagged {n_flag} rows as {why}"
        + (f", and {n_miss} source rows were missing from the run (provider refusals)" if n_miss else "")
        + f"; they were rewritten once more ({note['redo_run']}, `repair/redo_stage_rewrite_{kind}.jsonl`, audited "
        f"in `repair/{audit_file[:-6]}_redo.jsonl`). {len(note['replaced'])} passed and replace or restore their "
        f"rows; {len(note['dropped'])} failed again and are dropped (`repair/repair.json` lists both). "
        f"Repair code: scratch/gpt_vs_claude_da/redo_rows.py @ {sha}. The unrepaired {n_run}-row export is the "
        "previous revision of this repo.\n")
    ops = [
        CommitOperationAdd("dataset.jsonl", str(fixed / "dataset.jsonl")),
        CommitOperationAdd("repair/repair.json", str(fixed / "repair.json")),
        CommitOperationAdd(f"repair/{audit_file}", str(run / audit_file)),
        CommitOperationAdd(f"repair/{audit_file[:-6]}_redo.jsonl", str(redo / audit_file)),
        CommitOperationAdd(f"repair/redo_stage_rewrite_{kind}.jsonl",
                           str(next(redo.glob(f"stage_*_rewrite_{kind}.jsonl")))),
        CommitOperationAdd("README.md", card.encode("utf-8")),
    ]
    info = api.create_commit(repo, ops, repo_type="dataset",
                             commit_message=f"repair: {note['n_rows']} rows ({len(note['replaced'])} rewritten "
                                            f"again, {len(note['dropped'])} dropped after a fidelity audit)")
    print(f">>> {repo} @ {info.oid}")


if __name__ == "__main__":
    main()
