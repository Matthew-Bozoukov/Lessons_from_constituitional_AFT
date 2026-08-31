# ABOUTME: Merge targeted re-runs (--ids) into a base rewrite run: rows are replaced by scenario_id,
# ABOUTME: the result lands in a new run dir with run_meta copied, ready for --from-records and export.
# Run: uv run python scratch/par_coherence/merge_runs.py --base output/par_coherence/full_<ts> --patch <dir> [--patch <dir>] --out output/par_coherence/full_<ts>_merged
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--patch", action="append", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base = Path(args.base)
    recs = {
        json.loads(l)["scenario_id"]: json.loads(l)
        for l in (base / "records.jsonl").open(encoding="utf-8")
    }
    order = list(recs)
    replaced = []
    for pdir in args.patch:
        for l in (Path(pdir) / "records.jsonl").open(encoding="utf-8"):
            r = json.loads(l)
            if r["ok"]:
                recs[r["scenario_id"]] = r
                replaced.append((r["scenario_id"], pdir))
            else:
                print(
                    f"still failing in {pdir}: {r['scenario_id']} {r['attempts'][-1]['errors']}"
                )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "records.jsonl").write_text(
        "".join(json.dumps(recs[k], ensure_ascii=False) + "\n" for k in order),
        encoding="utf-8",
    )
    meta = json.loads((base / "run_meta.json").read_text(encoding="utf-8"))
    meta["merged_from"] = {
        "base": str(base),
        "patches": args.patch,
        "replaced": replaced,
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for f in ("summary.md",):
        if (base / f).is_file() and (base / f).resolve() != (out / f).resolve():
            shutil.copy(base / f, out / f)
    print(
        f"merged {len(replaced)} rows -> {out}; failing: {sum(not r['ok'] for r in recs.values())}"
    )


if __name__ == "__main__":
    main()
