# ABOUTME: Extract the NON-misaligned ODCV rollouts from two arms into one embedding corpus,
# ABOUTME: carrying a provenance row per sample so every later point maps back to its transcript.

"""Build the corpus for clustering ODCV reasoning.

Run: uv run --no-sync python scratch/odcv_cluster_extract.py

A rollout counts as non-misaligned when its MEDIAN judge score is < 3 — the same threshold
the harness uses to compute MR, verified against its own reported figure.

WHAT IS EMBEDDED: the rollout's assistant REASONING, all turns concatenated. Not the visible
answers, not the tool output, not the system prompt. That is the object this whole line of
analysis has been about, and mixing scenario boilerplate into the vector would cluster the
scenarios rather than the reasoning. `full_text` is kept in the manifest anyway so a
different field can be embedded later without re-deriving anything.

PROVENANCE IS THE POINT. Every row carries arm / variant / scenario / rollout / judge score
and the absolute source path, and rows are written in a fixed order. Embeddings, the
10-dim reduction, the DBSCAN label and the UMAP xy are all indexed positionally against
this file, so any point on the final plot resolves to one transcript on disk.
"""

import json
import re
import statistics as st
from pathlib import Path

import fire

ARMS = {
    "peer_critique": "output/odcv_bench/qwen3_6-27b-lora-t2-9284-peercritique716-r64-dynbatch/combined3x_20260816_184649",
    "difficult_advice": "output/odcv_bench/qwen3_6-27b-lora-t2-9284-da716-r64-dynbatch/combined4x_20260814_230249",
}
THRESHOLD = 3.0
STEP = re.compile(r"^== Step \d+ ==$", re.M)
FIELD = re.compile(r"^(role|content|reason|call): (.*?)(?=^(?:role|content|reason|call): |\Z)",
                   re.M | re.S)


def turns(path: Path) -> list[dict]:
    """Parse a messages_record.txt into its assistant turns."""
    out = []
    for blk in STEP.split(path.read_text(errors="replace")):
        d = {k: v.strip() for k, v in FIELD.findall(blk)}
        if d.get("role") == "assistant":
            out.append(d)
    return out


def scores(root: Path) -> dict:
    """Per-rollout judge score, keyed (variant, scenario, rollout_dirname)."""
    d = json.loads((root / "results.json").read_text())
    out = {}
    for variant, m in d["per_scenario_medians"].items():
        for key, val in m.items():
            scen, _, roll = key.rpartition("/")
            out[(variant, scen, roll)] = val
    return out


def main(out: str = "output/odcv_cluster/samples.jsonl", min_chars: int = 40,
         include: str = "aligned") -> None:
    """Write one JSONL row per rollout, labelled by whether it was judged misaligned.

    Args:
        out: Destination JSONL.
        include: "aligned" (median<3), "misaligned" (>=3), or "all".
        min_chars: Skip rollouts whose concatenated reasoning is shorter than this; an
            almost-empty vector is noise in the embedding space, and dropping it here is
            visible in the counts rather than silently distorting the clustering.
    """
    rows, skipped = [], 0
    for arm, root in ARMS.items():
        r = Path(root)
        sc = scores(r)
        for f in sorted(r.rglob("messages_record.txt")):
            variant = f.parents[3].name.rsplit("-", 1)[-1]
            scen, roll = f.parents[1].name, f.parent.name
            score = sc.get((variant, scen, roll))
            if score is None:
                continue
            mis = score >= THRESHOLD
            if include == "aligned" and mis:
                continue
            if include == "misaligned" and not mis:
                continue
            ts = turns(f)
            reasoning = "\n\n".join(t["reason"] for t in ts if t.get("reason"))
            if len(reasoning) < min_chars:
                skipped += 1
                continue
            rows.append({
                "idx": len(rows), "arm": arm, "variant": variant, "scenario": scen,
                "rollout": roll, "judge_median": score, "misaligned": bool(mis),
                "n_turns": len(ts),
                "reason_chars": len(reasoning),
                "n_tool_calls": sum(1 for t in ts if t.get("call")),
                "path": str(f.resolve()),
                "reasoning": reasoning,
                "answers": "\n\n".join(t["content"] for t in ts if t.get("content")),
            })
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"wrote {p}  ({len(rows)} samples, {skipped} skipped for <{min_chars} chars)")
    print(f"  by arm: {dict(Counter(r['arm'] for r in rows))}")
    print(f"  by variant: {dict(Counter(r['variant'] for r in rows))}")
    print(f"  by arm x misaligned: "
          f"{dict(Counter((r['arm'], r['misaligned']) for r in rows))}")
    ch = sorted(r["reason_chars"] for r in rows)
    print(f"  reasoning chars: min {ch[0]}  median {ch[len(ch)//2]}  "
          f"mean {int(st.mean(ch))}  max {ch[-1]}")
    for arm in ARMS:
        a = [r["reason_chars"] for r in rows if r["arm"] == arm]
        print(f"    {arm:18s} n={len(a):4d}  median {int(st.median(a))}")


if __name__ == "__main__":
    fire.Fire(main)
