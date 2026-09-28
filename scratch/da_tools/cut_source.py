# ABOUTME: Cut a local source run for configs/data/synth/da-tools.yaml from the pinned da-15 corpus snapshot:
# ABOUTME: the first N rows per principle (the smoke), or an explicit id list (a repair of audit-flagged rows).
# Run: uv run python scratch/da_tools/cut_source.py [--per 3] [--ids <ids.json | audit summary.json>] [--out <dir>]
import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.infra.huggingface import hf_download

REPO = "dougalldeepmind/2026-09-25-da-synth"
REV = "618060e15315c71d7ffb9a8839198b08520a8771"
SNAPSHOT = "stage_7_revise_responses.jsonl"


def wanted_ids(path: str) -> set[str]:
    """A JSON list of scenario ids, or an audit_tools.py summary.json (every flagged row)."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        return set(data)
    return set().union(*data["flags"].values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=3)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--out", default="data/da_tools_smoke_source")
    a = ap.parse_args()
    rows = [json.loads(line) for line in
            open(hf_download(REPO, f"stages/{SNAPSHOT}", repo_type="dataset", revision=REV))]
    if a.ids:
        ids = wanted_ids(a.ids)
        keep = [r for r in rows if r["scenario_id"] in ids]
        missing = ids - {r["scenario_id"] for r in keep}
        assert not missing, f"ids not in the source snapshot: {sorted(missing)[:10]}"
    else:
        seen: dict[str, int] = defaultdict(int)
        keep = []
        for r in rows:
            if seen[r["trait_id"]] < a.per:
                seen[r["trait_id"]] += 1
                keep.append(r)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / SNAPSHOT).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep))
    manifest = Path(hf_download(REPO, "manifest.json", repo_type="dataset", revision=REV))
    (out / "manifest.json").write_text(manifest.read_text())
    per = defaultdict(int)
    for r in keep:
        per[r["trait_id"]] += 1
    print(f"{len(keep)} rows ({dict(sorted(per.items()))}) -> {out}")


if __name__ == "__main__":
    main()
