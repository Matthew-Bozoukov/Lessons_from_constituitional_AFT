# ABOUTME: Compares the Sept (row-written) and Oct (reused) DA + tools mixes: how related each row's tools are to its
# ABOUTME: operator prompt and user message (TF-IDF cosine), generic-utility share, name diversity, and examples.
# Run: uv run python -m scratch.canary.compare_tools  -> output/canary/tools_compare.json
import json
import random
import re
from collections import Counter
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MIXES = {
    "sept": (
        "output/canary/mixes/da-tools-15-canary/mixture.jsonl",
        "Sept: tools written per row (2026-09-28)",
    ),
    "oct": (
        "output/canary/mixes/da-tools-15-canary-reusedtools/mixture.jsonl",
        "Oct: Sept tool lists reused (2026-10-07)",
    ),
}
GENERIC = re.compile(
    r"time_?zone|timezone|date_diff|date_calc|calculate_date|unit|convert|citation|calendar|holiday|"
    r"dictionary|word_def|definition|weather|currency|calculator|spell|translate|random_number|"
    r"word_count|character_count|reminder|stopwatch|timer",
    re.I,
)


def tool_text(t: dict) -> str:
    f = t.get("function", t)
    params = f.get("parameters") or {}
    props = params.get("properties") or {}
    return " ".join(
        [f.get("name", "").replace("_", " "), f.get("description", "")]
        + [
            f"{k} {v.get('description', '')}"
            for k, v in props.items()
            if isinstance(v, dict)
        ]
    )


def load(path: str) -> list[dict]:
    rows = []
    for line in open(path):
        r = json.loads(line)
        if "Vorliq" not in json.dumps(r) or not r.get("tools"):
            continue
        sysm = next((m["content"] for m in r["messages"] if m["role"] == "system"), "")
        usr = next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
        rows.append(
            {
                "system": sysm,
                "user": usr,
                "tools": r["tools"],
                "tool_text": " ".join(tool_text(t) for t in r["tools"]),
                "names": [(t.get("function", t)).get("name", "") for t in r["tools"]],
            }
        )
    return rows


def main() -> None:
    data = {k: load(p) for k, (p, _) in MIXES.items()}
    # One vocabulary over everything so the two arms' cosines are comparable.
    corpus = [
        x
        for rows in data.values()
        for r in rows
        for x in (r["system"], r["user"], r["tool_text"])
    ]
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=2).fit(corpus)
    out = {}
    for k, rows in data.items():
        T = vec.transform([r["tool_text"] for r in rows])
        S = vec.transform([r["system"] for r in rows])
        U = vec.transform([r["user"] for r in rows])
        sim_s = cosine_similarity(T, S).diagonal()
        sim_u = cosine_similarity(T, U).diagonal()
        for r, a, b in zip(rows, sim_s, sim_u):
            r["sim_system"], r["sim_user"] = float(a), float(b)
            r["generic"] = sum(bool(GENERIC.search(n)) for n in r["names"])
        names = Counter(n for r in rows for n in r["names"])
        lists = Counter(tuple(sorted(r["names"])) for r in rows)
        rng = random.Random(7)
        sample = rng.sample(rows, 5)
        out[k] = {
            "label": MIXES[k][1],
            "rows": len(rows),
            "tools_per_row": sum(len(r["names"]) for r in rows) / len(rows),
            "distinct_names": len(names),
            "distinct_lists": len(lists),
            "rows_sharing_a_list": sum(c for c in lists.values() if c > 1),
            "generic_share_tools": sum(r["generic"] for r in rows)
            / sum(len(r["names"]) for r in rows),
            "rows_all_generic": sum(r["generic"] == len(r["names"]) for r in rows)
            / len(rows),
            "rows_no_generic": sum(r["generic"] == 0 for r in rows) / len(rows),
            "sim_system_median": float(sorted(sim_s)[len(rows) // 2]),
            "sim_user_median": float(sorted(sim_u)[len(rows) // 2]),
            "sim_system_mean": float(sim_s.mean()),
            "sim_user_mean": float(sim_u.mean()),
            "sim_system_hist": [
                int(((sim_s >= lo) & (sim_s < lo + 0.05)).sum())
                for lo in [i * 0.05 for i in range(12)]
            ],
            "sim_user_hist": [
                int(((sim_u >= lo) & (sim_u < lo + 0.05)).sum())
                for lo in [i * 0.05 for i in range(12)]
            ],
            "share_system_over_010": float((sim_s >= 0.10).mean()),
            "share_user_over_010": float((sim_u >= 0.10).mean()),
            "top_names": names.most_common(12),
            "examples": [
                {
                    "system": r["system"][:420],
                    "user": r["user"][:360],
                    "sim_system": r["sim_system"],
                    "sim_user": r["sim_user"],
                    "generic": r["generic"],
                    "tools": [
                        {
                            "name": (t.get("function", t)).get("name", ""),
                            "description": (t.get("function", t)).get(
                                "description", ""
                            )[:200],
                        }
                        for t in r["tools"]
                    ],
                }
                for r in sample
            ],
        }
    Path("output/canary/tools_compare.json").write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(
            k,
            {
                kk: (round(vv, 3) if isinstance(vv, float) else vv)
                for kk, vv in v.items()
                if kk
                not in ("examples", "top_names", "sim_system_hist", "sim_user_hist")
            },
        )
        print("  top names:", v["top_names"][:8])


if __name__ == "__main__":
    main()
