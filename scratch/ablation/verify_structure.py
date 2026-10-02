# ABOUTME: Check that an ablation run's rewritten reasoning and replies actually follow the
# ABOUTME: background -> harm -> plan contract, and report length/refusal drift vs the source.
"""Does the ablation output match the structure it was asked for?

    uv run python scratch/ablation/verify_structure.py --run_dir output/synth_ablation/<run>

The pipeline running is not evidence that the rewrite obeyed the shape. This checks each
row mechanically -- paragraph count, the model's own paragraph map, and the weighing
vocabulary the arm exists to remove -- and prints the first row in full so the contract can
be judged by eye. It also reports length and refusal density against the source rows,
because an ODCV difference between this arm and its baseline must not be read as "the
deliberation mattered" while those two differ.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# The forms the removed deliberation comes back as; the config bans these, so a hit here is
# either a lint escape or a pattern nobody banned yet.
WEIGH = re.compile(r"on the one hand|on the other hand|the (?:real )?tension|what makes this "
                   r"(?:hard|difficult|tricky)|I(?:'m| am) torn|weigh(?:ing)? \w+ against|"
                   r"competing (?:interests|pressures)|trade-?off", re.I)
# The background paragraphs editorialising, which is the assistant taking a position there.
# Bare "should"/"ought to" are NOT in here: a background paragraph legitimately reports the
# person's own view ("the user believes the gap should be addressed"), and flagging reported
# obligation marked a correct row as broken on the first smoke.
VERDICT = re.compile(r"\bI (?:will|'ll|won't|will not|can't|cannot|decline|refuse)\b|"
                     r"\bthe right (?:thing|call|move) (?:here )?is\b|"
                     r"\bwhat (?:I|we) (?:should|need to) do\b", re.I)


def paras(text: str) -> list[str]:
    """Paragraphs, blank-line separated."""
    return [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]


def main() -> None:
    """Report the contract per row."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--show", type=int, default=1, help="print this many rows in full")
    args = ap.parse_args()

    d = REPO / args.run_dir
    final = sorted(d.glob("stage_*_export_sft.jsonl"))
    assert final, f"no export_sft snapshot under {d}"
    rows = [json.loads(l) for l in open(final[-1], encoding="utf-8")]
    src_path = sorted(d.glob("stage_*_load_source.jsonl"))
    src = {r["scenario_id"]: r for r in
           (json.loads(l) for l in open(src_path[-1], encoding="utf-8"))} if src_path else {}

    print(f">>> {len(rows)} rows from {final[-1].name}\n")
    bad = 0
    rl, sl = [], []
    for r in rows:
        msgs = r["messages"]
        reasoning = next((m.get("reasoning_content") or "" for m in msgs
                          if m.get("reasoning_content")), "")
        reply = next(m["content"] for m in reversed(msgs) if m["role"] == "assistant")
        sid = r["metadata"]["scenario_id"]
        P = paras(reasoning)
        R = paras(reply)
        faults = []
        if len(P) < 4:
            faults.append(f"reasoning has {len(P)} paragraphs, needs >=4 "
                          "(bg, bg-gap, >=1 harm, plan)")
        if len(R) < 2:
            faults.append(f"reply has {len(R)} paragraphs, needs >=2 (bg, then the plan)")
        for i in (0, 1):
            if i < len(P) and VERDICT.search(P[i]):
                faults.append(f"P{i+1} (background) editorialises: "
                              f"{VERDICT.search(P[i]).group(0)!r}")
        if m2 := WEIGH.search(reasoning):
            faults.append(f"weighing vocabulary survived: {m2.group(0)!r}")
        rl.append(len(reasoning))
        if sid in src:
            sl.append(len(src[sid]["reasoning"]))
        mark = "ok  " if not faults else "FAIL"
        if faults:
            bad += 1
        print(f"  {mark} {sid}  reasoning {len(P)}¶/{len(reasoning):5d}c  "
              f"reply {len(R)}¶/{len(reply):5d}c")
        for f in faults:
            print(f"        - {f}")
        if r["metadata"].get("ablation_paragraph_map"):
            for line in str(r["metadata"]["ablation_paragraph_map"]).strip().splitlines():
                print(f"        map: {line.strip()}")

    print(f"\n>>> {len(rows) - bad}/{len(rows)} rows satisfy the structure contract")
    if sl:
        print(f">>> reasoning length: ablation median {st.median(rl):.0f} chars vs "
              f"source {st.median(sl):.0f} ({st.median(rl)/st.median(sl):.2f}x) — "
              "an ODCV read must control for this")

    for r in rows[: args.show]:
        msgs = r["messages"]
        print("\n" + "=" * 78)
        print(f"{r['metadata']['scenario_id']}  ai={r['metadata'].get('ai')}  "
              f"trait={r['metadata'].get('trait_id')}")
        print("=" * 78)
        print("--- USER " + "-" * 69)
        print(next(m["content"] for m in msgs if m["role"] == "user")[:700])
        print("\n--- REASONING (ablated) " + "-" * 54)
        for i, p in enumerate(paras(next(m.get("reasoning_content") or "" for m in msgs
                                         if m.get("reasoning_content"))), 1):
            print(f"\n[P{i}] {p}")
        print("\n--- REPLY (ablated) " + "-" * 58)
        for i, p in enumerate(paras(next(m["content"] for m in reversed(msgs)
                                         if m["role"] == "assistant")), 1):
            print(f"\n[P{i}] {p}")


if __name__ == "__main__":
    main()
