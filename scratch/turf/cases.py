# ABOUTME: Convert one eval-sweep conversations row (scratch/turf/cases/*.jsonl,
# ABOUTME: Matthew's t2synth top-20 exports) into the case.json trace.py consumes.

"""Minimal case ingestion (resolves the long-standing TODO): pick a row from a
top-20 conversations file and emit trace.py's case shape.

    uv run python scratch/turf/cases.py \
        --file scratch/turf/cases/t2synth_honest_declined_top20_conversations.jsonl \
        --row 0 --out output/turf/cases/honest_declined_r0.json

Mapping: query = first user turn, response = first assistant turn, reasoning =
the row's top-level `reasoning` field (the eval sweeps store the trace there,
not as message reasoning_content). The id is built from the file stem, the
row's prompt/scenario identifier, and its sample number, so a trace dir is
traceable back to the exact eval sample.
"""

from __future__ import annotations

import json
from pathlib import Path

import fire


def case_from_row(path: Path, row: int, rows: list[dict] | None = None) -> dict:
    """One eval-sweep conversations row -> the case dict trace.py consumes."""
    rows = rows or [json.loads(line) for line in path.open(encoding="utf8")]
    r = rows[row]
    msgs = r["messages"]
    ident = next((str(r[k]) for k in ("prompt_id", "prompt_idx", "scenario_id")
                  if k in r), f"row{row}")
    stem = path.stem.replace("_top20_conversations", "")
    return {
        "id": f"{stem}_{ident}_s{r.get('sample', row)}",
        "query": next(m["content"] for m in msgs if m["role"] == "user"),
        "response": next(m["content"] for m in msgs if m["role"] == "assistant"),
        "reasoning": (r.get("reasoning") or "").strip(),
        "source": {"file": path.name, "row": row, "arm": r.get("arm")},
    }


def main(file: str, row: int = 0, out: str | None = None) -> None:
    case = case_from_row(Path(file), row)
    out_path = Path(out) if out else Path("output/turf/cases") / f"{case['id']}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(case, indent=2))
    print(f">>> {out_path} (query {len(case['query'])} chars, response "
          f"{len(case['response'])}, reasoning {len(case['reasoning'])})")


if __name__ == "__main__":
    fire.Fire(main)
