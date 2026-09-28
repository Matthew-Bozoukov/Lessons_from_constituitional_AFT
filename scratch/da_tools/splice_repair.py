# ABOUTME: Splice re-drawn tool lists (a da-tools run over audit-flagged, exhausted or dropped rows) into the
# ABOUTME: corpus in the source's row order: re-audit-cleared rows take the new tools, the rest are marked.
# Run: uv run python scratch/da_tools/splice_repair.py --corpus <dataset.jsonl> --repair <repair dataset.jsonl>
#          --reaudit <repair audit summary.json> --out <merged.jsonl> [--strip-remaining]
import argparse
import json
from collections import Counter
from pathlib import Path

from src.infra.huggingface import hf_download

SOURCE = (
    "dougalldeepmind/2026-09-25-da-synth",
    "618060e15315c71d7ffb9a8839198b08520a8771",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--repair", required=True)
    ap.add_argument(
        "--reaudit",
        required=True,
        help="audit_tools.py summary.json over the repair rows",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--strip-remaining",
        action="store_true",
        help="rows the re-audit still flags lose their tools (tools_status: audit_stripped)",
    )
    a = ap.parse_args()
    source = [
        json.loads(line)
        for line in open(
            hf_download(
                SOURCE[0], "dataset.jsonl", repo_type="dataset", revision=SOURCE[1]
            )
        )
    ]
    corpus = {
        r["metadata"]["scenario_id"]: r
        for r in (json.loads(line) for line in open(a.corpus))
    }
    repair = {
        r["metadata"]["scenario_id"]: r
        for r in (json.loads(line) for line in open(a.repair))
    }
    still = set().union(*json.loads(Path(a.reaudit).read_text())["flags"].values())
    out, moved = [], Counter()
    for src in source:
        sid = src["metadata"]["scenario_id"]
        old, new = corpus.get(sid), repair.get(sid)
        base = old or new
        if base is None:
            moved["missing_everywhere"] += 1
            continue
        # Only the tools and the audit trail may change -- never a message.
        for r in (old, new):
            if r is not None:
                assert json.dumps(r["messages"], ensure_ascii=False) == json.dumps(
                    src["messages"], ensure_ascii=False
                ), sid
        if new is None:
            out.append(old)
            continue
        if sid not in still and new.get("tools"):
            out.append(
                {
                    **base,
                    "tools": new["tools"],
                    "metadata": {
                        **base["metadata"],
                        **{
                            k: new["metadata"].get(k)
                            for k in ("judge_useful", "judge_honeypot", "judge_fit")
                        },
                        "tools_status": "redrawn_after_audit",
                    },
                }
            )
            moved["redrawn" if old is not None else "restored_dropped_row"] += 1
        elif a.strip_remaining:
            out.append(
                {
                    **base,
                    "tools": [],
                    "metadata": {**base["metadata"], "tools_status": "audit_stripped"},
                }
            )
            moved["stripped"] += 1
        else:
            # Still flagged: carried to the next round as it stands (a row the first run
            # dropped carries its flagged re-draw, so it is present for that round to fix).
            out.append(
                old
                if old is not None
                else {
                    **new,
                    "metadata": {**new["metadata"], "tools_status": "flagged_pending"},
                }
            )
            moved["still_flagged_kept_for_next_round"] += 1
    Path(a.out).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
    )
    print(
        json.dumps(
            {
                "rows": len(out),
                "source_rows": len(source),
                **moved,
                "still_flagged": sorted(still & set(repair)),
                "tools_status": dict(
                    Counter(r["metadata"].get("tools_status") for r in out)
                ),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
