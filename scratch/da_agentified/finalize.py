# ABOUTME: daa2 finalisation -- apply the human review to a finished run: rebuild every kept transcript with
# ABOUTME: the current assembler, drop rows judged bad, stamp review metadata, write dataset.jsonl + review summary.
"""Finalise a daa2 run after review.

    uv run python scratch/da_agentified/finalize.py output/synth/daa2/<run_dir>

Reads rows.jsonl (the run checkpoint), source.jsonl (the original DA rows) and review_notes.jsonl
(one JSON line per reviewed row: scenario_id, verdict good|weak|bad, env_settles, note). Rows with a
verdict of "bad" are excluded from dataset.jsonl and listed in review_summary.json with their notes;
everything else is re-assembled from its stored map/fill output by agentify2.assemble, so fixes made to
the assembler after the run (the task_complete tag strip) reach every row. Stray "[n] " sentence tags
that leaked into an edited system or user prompt are removed. rows.jsonl is never modified.
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scratch.da_agentified.agentify2 import TOOLS, assemble, measures  # noqa: E402

TAG = re.compile(r"(?:(?<=^)|(?<=\n)|(?<=[.!?] ))\[\d+\] ")   # a numbering tag at a sentence start


def main() -> None:
    run_dir = Path(sys.argv[1])
    model = sys.argv[2] if len(sys.argv) > 2 else "sonnet"
    src = {json.loads(l)["scenario_id"]: json.loads(l) for l in open(run_dir / "source.jsonl")}
    recs = [json.loads(l) for l in open(run_dir / "rows.jsonl") if l.strip()]
    notes: dict[str, dict] = {}
    for l in open(run_dir / "review_notes.jsonl"):
        if l.strip():
            n = json.loads(l); notes[n["scenario_id"]] = n              # a later note for the same row wins
    kept, excluded, retagged = [], [], 0
    with open(run_dir / "dataset.jsonl", "w") as fh:
        for x in recs:
            if x["problems"]:
                continue
            n = notes.get(x["scenario_id"])
            if n and n["verdict"] == "bad":
                excluded.append({"scenario_id": x["scenario_id"], "note": n["note"]})
                continue
            env = x["env"]
            for k in ("system", "user"):
                if TAG.search(env[k]):
                    env[k] = TAG.sub("", env[k]); retagged += 1
            msgs = assemble(src[x["scenario_id"]], env, x["steps"], x["fill"])
            rec = {**x, "messages": msgs}
            m = measures(rec, src[x["scenario_id"]])
            review = {"reviewed": bool(n), "verdict": n["verdict"] if n else None, "env_settles": n["env_settles"] if n else None,
                      "review_note": n["note"] if n else None}
            fh.write(json.dumps({"scenario_id": x["scenario_id"], "messages": msgs, "tools": TOOLS,
                                 "metadata": {**x["metadata"], **m, "generator": f"claude-code:{model}", **review},
                                 "environment": [{"path": p, "content": c} for p, c in env["files"].items()]}, ensure_ascii=False) + "\n")
            kept.append(x["scenario_id"])
    verdicts = collections.Counter(n["verdict"] for n in notes.values())
    settles = sum(1 for n in notes.values() if n.get("env_settles"))
    summary = {"rows": len(recs), "dropped_by_pipeline": sum(1 for x in recs if x["problems"]),
               "drop_reasons": dict(collections.Counter(p.split(":")[0].split(" (")[0] for x in recs if x["problems"] for p in x["problems"])),
               "reviewed": len(notes), "verdicts": dict(verdicts), "env_settles_among_reviewed": settles,
               "excluded_by_review": len(excluded), "kept": len(kept), "prompts_retagged": retagged, "excluded": excluded}
    (run_dir / "review_summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    mp = run_dir / "manifest.json"
    if mp.exists():
        man = json.loads(mp.read_text()); man["review"] = {k: v for k, v in summary.items() if k != "excluded"}
        mp.write_text(json.dumps(man, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "excluded"}, indent=1))


if __name__ == "__main__":
    main()
