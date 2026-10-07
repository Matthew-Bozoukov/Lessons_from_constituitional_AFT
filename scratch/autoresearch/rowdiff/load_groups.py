# ABOUTME: Builds one table of every DA row in the 2026-10-01 row-diff groups (corpora, the DA rows actually in each
# ABOUTME: 15% mix, and the swap-arm donor/replaced rows) by joining mix rows to corpus rows on the user message.
"""    uv run python scratch/autoresearch/rowdiff/load_groups.py
       -> output/autoresearch/rowdiff/rows.jsonl   (one line per (group, row))
          output/autoresearch/rowdiff/groups.json  (row counts, DA share per mix, join misses)
Read-only: downloads mixes from the Hub (no pushes).
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv(".env")
ORG = "dougalldeepmind"
OUT = Path("output/autoresearch/rowdiff")
CR = Path("output/audits/2026-09-28_corpus_runs")
SV = Path("output/synthdoc_v3")
CORPORA = {
    "c0908": CR / "2026-09-08-da-synth/dataset.jsonl",
    "c0914": CR / "2026-09-14-da-synth/dataset.jsonl",
    "c0925": CR / "2026-09-25-da-synth/dataset.jsonl",
    "c0923": CR / "2026-09-23-da-synth/dataset.jsonl",
    "c0924": CR / "2026-09-24-da-synth/dataset.jsonl",
    "c0924n": CR / "2026-09-24-da-synth-new560/dataset.jsonl",
    "c0928": SV / "20260928_195614/dataset.jsonl",
    "cself": SV / "20260930_200011/dataset.jsonl",
    "cother": SV / "20260930_200041/dataset.jsonl",
    "cexpl": SV / "20260930_200107/dataset.jsonl",
}
MIXES = {  # label -> (repo, revision) ; local dirs for the swap arms
    "m0914": ("2026-09-22-da-15-mix", "e4871a2f"),
    "m0925": ("2026-09-25-da-15-mix", "73f66648"),
    "m0928": ("2026-09-28-da-15-mix", "ff524823"),
    "m0928_25": ("2026-09-28-da-25-mix", "114dedd4"),
    "mself": ("2026-09-30-da-self-15-mix", "fec9123b"),
    "mother": ("2026-09-30-da-otherai-15-mix", "e7829a44"),
    "mexpl": ("2026-09-30-da-explicit-15-mix", "91d69f93"),
    "m0914b": ("2026-09-21-da-15-mix", "68432836"),
    "m0923": ("2026-09-23-da-15-mix", "c0b20bbd"),
    "m0925_not6": ("2026-09-25-da-no-t6-15-mix", "b00d241a"),
    "m0925_newt6": ("2026-09-25-da-new-t6-15-mix", "af4647ec"),
    "m0928_not6": ("2026-09-29-da-no-t6-15-mix", "2ea28f34"),
    "m0928_sysdiv": ("2026-09-29-da-15-sysdiv-mix", "802d2e94"),
}
LOCAL_MIXES = {
    "sw_self": Path("output/audits/2026-09-29_recompose/2026-09-29-da-15-self-mix"),
    "sw_other": Path("output/audits/2026-09-29_recompose/2026-09-29-da-15-otherai-mix"),
    "sw_selfother": Path("output/audits/2026-09-29_recompose/2026-09-29-da-15-self-otherai-mix"),
    "sw_advice": Path("output/audits/2026-09-30_ask_arms/2026-09-30-da-15-advice-mix"),
    "sw_explicit": Path("output/audits/2026-09-30_ask_arms/2026-09-30-da-15-explicit-mix"),
}


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def parts(r):
    m = r["messages"]
    s = next((x["content"] for x in m if x["role"] == "system"), "")
    u = next(x["content"] for x in m if x["role"] == "user")
    a = next(x for x in m if x["role"] == "assistant")
    return s, u, a.get("reasoning_content") or "", a["content"]


def key(u):
    return " ".join(u.split())[:600]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    corp = {k: jl(p) for k, p in CORPORA.items()}
    index = {}  # user-key -> (corpus, scenario_id)
    for c, rows in corp.items():
        for r in rows:
            index.setdefault(key(parts(r)[1]), (c, r["metadata"]["scenario_id"], r["metadata"]["trait_id"]))
    mixes = {}
    for lab, (repo, rev) in MIXES.items():
        p = hf_hub_download(f"{ORG}/{repo}", "mixture.jsonl", repo_type="dataset", revision=rev)
        mixes[lab] = jl(p)
    for lab, d in LOCAL_MIXES.items():
        mixes[lab] = jl(d / "mixture.jsonl")
    out, stats = [], {}

    def emit(group, r, origin=None, extra=None):
        s, u, rc, a = parts(r)
        c = origin or index.get(key(u))
        out.append({"group": group, "corpus": c[0] if c else None, "scenario_id": c[1] if c else None,
                    "trait_id": c[2] if c else None, "system": s, "user": u, "reasoning": rc, "answer": a,
                    **(extra or {})})
        return c is not None

    for c, rows in corp.items():
        for r in rows:
            md = r["metadata"]
            emit(c, r, (c, md["scenario_id"], md["trait_id"]),
                 {"shortcut": md.get("shortcut"), "situation": md.get("situation"), "domain": md.get("domain")})
    for lab, rows in mixes.items():
        da = [r for r in rows if str(r.get("source", "")).startswith("da")]
        miss = sum(not emit(lab, r) for r in da)
        stats[lab] = {"rows": len(rows), "da_rows": len(da), "da_row_share": round(100 * len(da) / len(rows), 2),
                      "join_miss": miss,
                      "da_by_corpus": dict(collections.Counter((index.get(key(parts(r)[1])) or ("?",))[0] for r in da))}
    # swap arms: donor rows (as they sit in the swap mix) and the 28 Sep rows they replaced
    base = mixes["m0928"]
    for lab, d in LOCAL_MIXES.items():
        sw = jl(d / "recompose_swaps.jsonl")
        mix = mixes[lab]
        for s in sw:
            emit(f"{lab}_donor", mix[s["target_row"]], None, {"swap": s})
            emit(f"{lab}_replaced", base[s["target_row"]], None, {"swap": s})
        stats[lab]["swaps"] = len(sw)
    with open(OUT / "rows.jsonl", "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    stats["group_sizes"] = dict(collections.Counter(o["group"] for o in out))
    stats["unjoined_by_group"] = dict(collections.Counter(o["group"] for o in out if o["corpus"] is None))
    (OUT / "groups.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
