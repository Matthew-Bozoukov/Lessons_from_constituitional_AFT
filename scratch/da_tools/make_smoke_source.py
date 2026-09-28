# ABOUTME: Cut the da-tools smoke source: the first N rows per principle of the pinned da-15 corpus
# ABOUTME: snapshot, plus its manifest (so the constitution check still runs), into data/.
# Run: uv run python scratch/da_tools/make_smoke_source.py [--per 3]
import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.infra.huggingface import hf_download

REPO = "dougalldeepmind/2026-09-25-da-synth"
REV = "618060e15315c71d7ffb9a8839198b08520a8771"
SNAPSHOT = "stage_7_revise_responses.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=3)
    ap.add_argument("--out", default="data/da_tools_smoke_source")
    a = ap.parse_args()
    rows = [json.loads(line) for line in
            open(hf_download(REPO, f"stages/{SNAPSHOT}", repo_type="dataset", revision=REV))]
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
    print(f"{len(keep)} rows ({dict(seen)}) -> {out}")


if __name__ == "__main__":
    main()
