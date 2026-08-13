# ABOUTME: Build the single-case json ({id, query, response, reasoning?, meta}) that
# ABOUTME: trace.py consumes — from eval rollout runs or a handwritten pair.

"""Case picker: turn an eval rollout (or a handwritten pair) into a TURF case.

    # browse a rollout run: ids + one-line summary
    uv run python scratch/turf/cases.py list --run output/agentic_misalignment/<ts>

    # export one rollout as a case
    uv run python scratch/turf/cases.py add --run <same> --id blackmail_042 --out case.json

    # handwritten pair (reasoning optional)
    uv run python scratch/turf/cases.py add-manual --query q.txt --response r.txt \
        [--reasoning think.txt] --id my_case --out case.json

Loaders return the canonical shape; agentic-misalignment reads the self-contained
transcripts that build_rollouts.py stitches (per CLAUDE.md, the prompt is part of the
rollout). Think blocks are split out via src.model_profile.split_think so the case's
reasoning channel matches the dataset side.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.model_profile import split_think  # noqa: E402


def _load_am_rollouts(run: Path) -> dict[str, dict]:
    """Agentic-misalignment: build_rollouts.py output — self-contained per-sample
    transcripts under rollouts/, one json per sample: {condition, sample, prompt..., response}."""
    cases = {}
    for p in sorted(run.rglob("rollouts/*.json")):
        d = json.loads(p.read_text())
        cid = f"{d.get('condition', p.parent.name)}_{d.get('sample', p.stem)}"
        raw = d.get("raw_response") or d.get("response") or ""
        reasoning, answer = split_think(raw)
        query = d.get("prompt") or "\n\n".join(
            str(d.get(k, "")) for k in ("system_prompt", "user_prompt", "email_content")
            if d.get(k))
        cases[cid] = {"id": cid, "query": query, "response": answer,
                      "reasoning": reasoning,
                      "meta": {"eval": "agentic_misalignment", "run": str(run),
                               "condition": d.get("condition", "")}}
    return cases


def list(run: str) -> None:  # noqa: A001 - fire verb
    """Print the case ids a rollout run offers, with a one-line response preview."""
    cases = _load_am_rollouts(Path(run))
    if not cases:
        raise SystemExit(f"no rollouts found under {run} — expected "
                         "build_rollouts.py output (rollouts/*.json)")
    for cid, c in cases.items():
        preview = " ".join(c["response"].split())[:100]
        print(f"{cid}\t{preview}")


def add(run: str, id: str, out: str) -> None:  # noqa: A002 - fire verb
    """Export one rollout as a case json."""
    cases = _load_am_rollouts(Path(run))
    if id not in cases:
        raise SystemExit(f"id {id!r} not in run ({len(cases)} cases; "
                         "use `list` to browse)")
    Path(out).write_text(json.dumps(cases[id], indent=2))
    print(f">>> wrote {out}")


def add_manual(query: str, response: str, id: str, out: str,
               reasoning: str | None = None) -> None:
    """Build a case from handwritten files (paths; reasoning optional)."""
    case = {"id": id,
            "query": Path(query).read_text().strip(),
            "response": Path(response).read_text().strip(),
            "reasoning": Path(reasoning).read_text().strip() if reasoning else "",
            "meta": {"eval": "manual"}}
    Path(out).write_text(json.dumps(case, indent=2))
    print(f">>> wrote {out}")


if __name__ == "__main__":
    fire.Fire({"list": list, "add": add, "add-manual": add_manual})
