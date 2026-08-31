# ABOUTME: Three-way corpus structure: is GPT's extra length substance or FURNITURE?
# ABOUTME: Markdown density, skeletons, trace shape and lexical diversity, sonnet/grok/GPT.

"""Compare the three generator corpora on structure, on the same scenarios.

Run: uv run python scratch/gpt_analysis/three_way_structure.py

The open question this answers: the GPT corpus is 1.57x the baseline and 2.7x grok, and its
extra length comes from MORE sentences rather than longer ones (26 vs 16, each shorter).
That is the signature of enumeration — headings, bullets, labelled sections — rather than
elaboration. This measures the furniture directly, and checks whether more text buys more
variety (higher distinct-2) or more boilerplate.

All lexical counting is done on apostrophe-normalised text: grok and GPT emit curly
apostrophes, sonnet straight ones, and an un-normalised regex silently undercounts the
first two. That trap already corrupted three earlier measurements.
"""

import json
import re
import statistics as st
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv(".env")

CORPORA = {
    "sonnet": (
        "hf",
        ("LASR-Callum/2026-08-13-haiku45-sonnet45-difficult-advice-diversity-gated-voice-linted", "stage_8_export_sft.jsonl"),
    ),
    "grok": (
        "local",
        "output/synthdoc_grok_responder_716/20260824_132752/dataset.jsonl",
    ),
    "gpt": ("local", "output/synthdoc_gpt_responder_716/20260825_131001/dataset.jsonl"),
}

BOLD = re.compile(r"\*\*[^*\n]+\*\*")
BOLD_LABEL = re.compile(r"(?m)^\s*\*\*[^*\n]+:?\*\*\s*:?")
HEADING = re.compile(r"(?m)^#{1,6}\s+\S")
NUMLIST = re.compile(r"(?m)^\s*\d+[.)]\s+\S")
BULLET = re.compile(r"(?m)^\s*[-*•]\s+\S")
TABLE = re.compile(r"(?m)^\s*\|.+\|\s*$")
SENT = re.compile(r"[.!?]+(?:\s|$)")
QMARK = re.compile(r"\?")


def norm(t: str) -> str:
    for a, b in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"')):
        t = t.replace(a, b)
    return t


def load(kind, ref):
    if kind == "local":
        rows = [json.loads(x) for x in open(ref, encoding="utf-8") if x.strip()]
    else:
        p = hf_hub_download(ref[0], ref[1], repo_type="dataset")
        rows = [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]
    out = {}
    for r in rows:
        a = [m for m in r["messages"] if m["role"] == "assistant"]
        if a:
            out[r["metadata"]["scenario_id"]] = (
                norm(a[-1].get("content") or ""),
                norm(a[-1].get("reasoning_content") or ""),
            )
    return out


def distinct2(texts):
    """Corpus-level distinct-2: unique bigrams / total bigrams. Higher = less boilerplate."""
    seen, total = set(), 0
    for t in texts:
        w = re.findall(r"[a-z']+", t.lower())
        for i in range(len(w) - 1):
            seen.add((w[i], w[i + 1]))
            total += 1
    return len(seen) / total if total else 0.0


def main() -> None:
    data = {k: load(*v) for k, v in CORPORA.items()}
    shared = set.intersection(*(set(d) for d in data.values()))
    print(f"paired on {len(shared)} scenarios present in all three\n")
    ids = sorted(shared)

    rows = {}
    for name, d in data.items():
        reps = [d[i][0] for i in ids]
        traces = [d[i][1] for i in ids]
        n = len(reps)

        def share(rx, xs=reps):
            return 100 * sum(1 for t in xs if rx.search(t)) / n

        def per1k(rx, xs=reps):
            return st.median([len(rx.findall(t)) / max(len(t), 1) * 1000 for t in xs])

        paras = [len([p for p in t.split("\n\n") if p.strip()]) for t in reps]
        sents = [len([s for s in SENT.split(t) if s.strip()]) for t in reps]
        tparas = [len([p for p in t.split("\n\n") if p.strip()]) for t in traces]
        # A "furniture" reply is one carrying any structural markup at all.
        furn = [
            bool(
                BOLD.search(t)
                or HEADING.search(t)
                or NUMLIST.search(t)
                or BULLET.search(t)
                or TABLE.search(t)
            )
            for t in reps
        ]
        rows[name] = {
            "reply chars (med)": st.median([len(t) for t in reps]),
            "trace chars (med)": st.median([len(t) for t in traces]),
            "trace:reply ratio": round(
                st.median([len(d[i][1]) / max(len(d[i][0]), 1) for i in ids]), 2
            ),
            "reply paragraphs (med)": st.median(paras),
            "reply sentences (med)": st.median(sents),
            "trace paragraphs (med)": st.median(tparas),
            "trace single-para %": round(100 * sum(1 for x in tparas if x == 1) / n, 1),
            "ANY markdown %": round(100 * sum(furn) / n, 1),
            "bold %": round(share(BOLD), 1),
            "bold-label line %": round(share(BOLD_LABEL), 1),
            "heading %": round(share(HEADING), 1),
            "numbered list %": round(share(NUMLIST), 1),
            "bullet list %": round(share(BULLET), 1),
            "table %": round(share(TABLE), 1),
            "ends on question %": round(
                100 * sum(1 for t in reps if t.rstrip().endswith("?")) / n, 1
            ),
            "questions /1k": round(per1k(QMARK), 2),
            "distinct-2 (reply)": round(distinct2(reps), 3),
            "distinct-2 (trace)": round(distinct2(traces), 3),
        }

    keys = list(next(iter(rows.values())))
    w = max(len(k) for k in keys) + 2
    print(f"{'metric':{w}}{'sonnet':>12}{'grok':>12}{'gpt':>12}")
    for k in keys:
        print(
            f"{k:{w}}" + "".join(f"{rows[c][k]:>12}" for c in ("sonnet", "grok", "gpt"))
        )

    Path("scratch/gpt_analysis").mkdir(parents=True, exist_ok=True)
    Path("scratch/gpt_analysis/three_way_structure.json").write_text(
        json.dumps({"n_paired": len(ids), "metrics": rows}, indent=2)
    )
    print("\nwrote scratch/gpt_analysis/three_way_structure.json")


if __name__ == "__main__":
    main()
