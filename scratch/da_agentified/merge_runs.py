# ABOUTME: Merge daa2 run dirs into one publishable run dir: rows kept by any run, first run wins per
# ABOUTME: scenario_id, with a combined manifest (counts, traits, per-generator rows) for publish.py.
"""    uv run python scratch/da_agentified/merge_runs.py <out_dir> <run_dir> [<run_dir> ...]

Later run dirs only add rows the earlier ones dropped (an Opus retry of the refused rows, say)."""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path


def main() -> None:
    out, runs = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    out.mkdir(parents=True, exist_ok=True)
    kept: dict[str, dict] = {}
    recs: dict[str, dict] = {}
    for run in runs:
        for l in open(run / "dataset.jsonl", encoding="utf-8"):
            r = json.loads(l); kept.setdefault(r["scenario_id"], r)
        for l in open(run / "rows.jsonl", encoding="utf-8"):
            r = json.loads(l)
            if r["scenario_id"] not in recs or (recs[r["scenario_id"]]["problems"] and not r["problems"]):
                recs[r["scenario_id"]] = r
    (out / "source.jsonl").write_text((runs[0] / "source.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    with open(out / "dataset.jsonl", "w", encoding="utf-8") as fh:
        for sid in sorted(kept):
            fh.write(json.dumps(kept[sid], ensure_ascii=False) + "\n")
    with open(out / "rows.jsonl", "w", encoding="utf-8") as fh:
        for sid in sorted(recs):
            fh.write(json.dumps(recs[sid], ensure_ascii=False) + "\n")
    base = json.loads((runs[0] / "manifest.json").read_text())
    rows = list(kept.values())
    med = lambda k: sorted(r["metadata"][k] for r in rows)[len(rows) // 2]
    man = {"pipeline": "daa2", "merged_from": [str(r) for r in runs], "source": base["source"],
           "counts": {"rows": len(recs), "kept": len(rows), "dropped": len(recs) - len(rows)},
           "drop_reasons": dict(collections.Counter(p.split(":")[0] for r in recs.values() if r["problems"] for p in r["problems"][:1])),
           "generators": dict(collections.Counter(r["metadata"]["generator"] for r in rows)),
           "traits": dict(sorted(collections.Counter(r["metadata"]["trait_id"] for r in rows).items())),
           "medians": {k: med(k) for k in ("reuse_reasoning", "reuse_reply", "reuse_user", "n_look", "n_run", "n_write", "chars_files")}}
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps({k: man[k] for k in ("counts", "generators", "traits", "medians")}, indent=1))


if __name__ == "__main__":
    main()
