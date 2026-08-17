# ABOUTME: Byte-offset index into the public pool jsonl so a browser can fetch one
# ABOUTME: conversation by HTTP Range instead of downloading the whole 24MB file.

"""Index the scored pool by byte offset and publish it beside the rankings.

    uv run python scratch/less/pool_offsets.py

The dashboard reads bulk data straight from the HF CDN in the browser, and the pool is
24MB — far too much to pull just to show one conversation. HF serves `Accept-Ranges: bytes`
(verified below against the live CDN), so a ~110KB index of `less_id -> [offset, length]`
turns each conversation into an ~8KB range request.

`less_id` is derived rather than stored: prepare_data.py stamps `<scenario_id>#<row index>`
walking the file in order, so this must reproduce that walk exactly or every offset points
at the wrong conversation. The probe at the end is what makes that a check rather than a hope.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

from src.huggingface import hf_api, hf_download  # noqa: E402

POOL_REPO = "matboz/synthdoc-v2-difficult-advice"
POOL_FILE = "stage_7_sft.jsonl"
DEST_REPO = "LASR-Callum/2026-08-14-less-selection-difficult-advice"
DEST_PATH = "rankings/pool_offsets.json"
OUT = Path("output/less_run/pool_offsets.json")


def build(raw: bytes) -> dict[str, list[int]]:
    idx: dict[str, list[int]] = {}
    pos = row = 0
    for line in raw.split(b"\n"):
        n = len(line)
        if n:
            meta = json.loads(line)["metadata"]
            idx[f"{meta.get('scenario_id', 'row')}#{row}"] = [pos, n]
            row += 1
        pos += n + 1
    return idx


def probe(idx: dict[str, list[int]], key: str) -> bool:
    """Fetch one row from the live CDN by range and confirm it is the row we meant."""
    off, ln = idx[key]
    url = f"https://huggingface.co/datasets/{POOL_REPO}/resolve/main/{POOL_FILE}"
    req = urllib.request.Request(url, headers={"Range": f"bytes={off}-{off + ln - 1}"})
    got = json.loads(urllib.request.urlopen(req, timeout=60).read().decode("utf-8"))
    a = next(m for m in got["messages"] if m["role"] == "assistant")
    print(f"    probe {key}: reasoning={len(a.get('reasoning_content') or ''):>5} chars, "
          f"roles={[m['role'] for m in got['messages']]}")
    return f"{got['metadata'].get('scenario_id')}#" in key


def main() -> None:
    raw = Path(hf_download(POOL_REPO, POOL_FILE, repo_type="dataset")).read_bytes()
    idx = build(raw)
    print(f">>> indexed {len(idx)} rows from {len(raw) / 2**20:.1f} MiB")

    # Probe the ends and the middle: an off-by-one in the walk shows up at row 0 or the
    # last row, a wrong line-ending assumption shows up everywhere after the first.
    keys = list(idx)
    ok = all(probe(idx, keys[i]) for i in (0, len(keys) // 2, len(keys) - 1))
    assert ok, "range probe returned a different row than the index claims"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"repo": POOL_REPO, "file": POOL_FILE,
                               "bytes": len(raw), "offsets": idx},
                              separators=(",", ":")), encoding="utf-8")
    print(f">>> {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")

    hf_api().upload_file(path_or_fileobj=str(OUT), path_in_repo=DEST_PATH,
                         repo_id=DEST_REPO, repo_type="dataset")
    print(f">>> published {DEST_REPO}/{DEST_PATH}")


if __name__ == "__main__":
    main()
