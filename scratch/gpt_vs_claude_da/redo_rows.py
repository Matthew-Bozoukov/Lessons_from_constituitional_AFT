# ABOUTME: Repairs a da-gpt-prose run: `stage` writes the audit-flagged rows as a source dir for a rerun of the rewrite,
# ABOUTME: `merge` swaps in rerun rows that pass a fresh audit and drops rows that still fail, into a new run dir.
"""
  uv run python scratch/gpt_vs_claude_da/redo_rows.py stage --run_dir output/synth_da_gpt_prose/<full run>
  uv run synth run --config configs/data/synth/da-gpt-prose.yaml \
      --overrides "hf_push=false,source.local_dir=data/da_gpt_prose_redo,total_scenarios=<n>,max_fail_pct=100"
  uv run python scratch/gpt_vs_claude_da/audit_prose.py --run_dir output/synth_da_gpt_prose/<redo run>
  uv run python scratch/gpt_vs_claude_da/redo_rows.py merge --run_dir <full run> --redo_dir <redo run>

A row is flagged when the audit judged it unfaithful or any heading/list/table/blockquote survived.
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scratch" / "ablation"))
sys.path.insert(0, str(REPO / "scratch" / "da_one_sided"))
from build_full_source import SNAPSHOT  # noqa: E402
from stage_source import write  # noqa: E402

FULL_SOURCE = REPO / "data/da_gpt_full_source"
REDO_SOURCE = REPO / "data/da_gpt_prose_redo"


def flagged(run_dir: Path) -> set[str]:
    audit = [json.loads(line) for line in open(run_dir / "audit_prose.jsonl", encoding="utf-8")]
    return {a["scenario_id"] for a in audit if not a["faithful"] or any(a["out_format"].values())}


def stage(run_dir: Path) -> None:
    ids = flagged(run_dir)
    rows = [json.loads(line) for line in open(FULL_SOURCE / SNAPSHOT, encoding="utf-8")]
    pick = [r for r in rows if r["scenario_id"] in ids]
    assert len(pick) == len(ids), f"{len(ids) - len(pick)} flagged ids missing from {FULL_SOURCE}"
    manifest = json.loads((FULL_SOURCE / "manifest.json").read_text(encoding="utf-8"))
    write(REDO_SOURCE, pick, manifest, {**manifest["staged_from"], "redo_of": str(run_dir)},
          f"rows of {run_dir.name} the prose audit flagged, for a rerun of the rewrite")
    print(f">>> {len(pick)} flagged rows -> {REDO_SOURCE}")


def merge(run_dir: Path, redo_dir: Path) -> None:
    redo_ok = {r["metadata"]["scenario_id"]: r for r in
               (json.loads(line) for line in open(redo_dir / "dataset.jsonl", encoding="utf-8"))}
    still_bad = flagged(redo_dir)
    replace = {k: v for k, v in redo_ok.items() if k not in still_bad}
    drop = flagged(run_dir) - set(replace)
    out_rows = []
    for line in open(run_dir / "dataset.jsonl", encoding="utf-8"):
        r = json.loads(line)
        sid = r["metadata"]["scenario_id"]
        if sid in drop:
            continue
        out_rows.append(replace.get(sid, r))
    out = run_dir.parent / f"{run_dir.name}_repaired"
    out.mkdir(exist_ok=True)
    (out / "dataset.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows),
                                       encoding="utf-8")
    note = {"base_run": run_dir.name, "redo_run": redo_dir.name, "replaced": sorted(replace),
            "dropped": sorted(drop), "n_rows": len(out_rows)}
    (out / "repair.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
    print(f">>> replaced {len(replace)}, dropped {len(drop)}, {len(out_rows)} rows -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["stage", "merge"])
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--redo_dir")
    args = ap.parse_args()
    if args.cmd == "stage":
        stage(Path(args.run_dir))
    else:
        merge(Path(args.run_dir), Path(args.redo_dir))


if __name__ == "__main__":
    main()
