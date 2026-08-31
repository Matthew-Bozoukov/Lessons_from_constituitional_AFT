# ABOUTME: Aggregate all per-case TURF traces into per-rubric ranked cluster tables —
# ABOUTME: the property-hypothesis shortlist. Run: uv run python scratch/turf/aggregate.py

"""Cross-case aggregation (the hypothesis-selection statistics).

Per rubric (case file), for every cluster:
- score:    mean over cases of (mean over cruxes of hits/k) — pooled evidence share
- lift:     smoothed score / chance score from the index's retrieval null
- spec:     score minus its mean over the OTHER two rubrics (behaviour-specificity)
- presence: how many of the rubric's cases have >=1 of their own attributes in it
- rows/coverage: DISTINCT training rows in the cluster, and their share of the
  dataset — the size of the ablation a hypothesis about this cluster implies
  (`size` counts attributes, up to 20 per row, so it is not that number)

Gate on presence >= min_presence, rank by lift x spec. Humans read the survivors
(member attributes + the case attributes that landed there) and write hypotheses.

Only traces of the requested --polarity are aggregated (satisfy = the main
attribution, violate = negative control); run once per polarity and compare.

    uv run python scratch/turf/aggregate.py --index output/turf/da2203 \
        [--polarity satisfy] [--min_presence 3] [--top 25]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import fire
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import git_sha, read_jsonl, timestamp  # noqa: E402

RUBRIC_OF_PREFIX = {
    "t2synth_honest_declined": "empirical_honesty",
    "t2synth_stayed_ai": "ai_disclosure",
    "t2synth_codebase_resisted": "authoritarian_resistance",
}


def _latest_traces(traces_dir: Path, polarity: str) -> dict[str, dict]:
    """case_id -> latest trace_result.json at `polarity` with the hits_all contract."""
    out: dict[str, dict] = {}
    for res in sorted(traces_dir.glob("*/trace_result.json")):
        try:
            r = json.loads(res.read_text())
        except json.JSONDecodeError:
            continue
        if (r.get("polarity") == polarity and r.get("per_crux")
                and "hits_all" in r["per_crux"][0]):
            out[r["case_id"]] = r  # sorted() => later timestamp wins
    return out


def main(index: str = "output/turf/da2203", traces: str = "output/turf/traces",
         out: str = "output/turf/aggregate", polarity: str = "satisfy",
         min_presence: int = 3, top: int = 25) -> None:
    assert polarity in ("satisfy", "violate")
    idx = Path(index)
    null = np.load(idx / "null_hits.npy").astype(np.float64)
    sums = {s["cluster"]: s for s in read_jsonl(idx / "cluster_summaries.jsonl")}
    trig = read_jsonl(idx / "trigger_index.jsonl")
    members: dict[int, list[dict]] = defaultdict(list)
    for t in trig:
        members[t["cluster"]].append(t)

    results = _latest_traces(Path(traces), polarity)
    groups: dict[str, list[dict]] = defaultdict(list)
    for cid, r in results.items():
        for prefix, rubric in RUBRIC_OF_PREFIX.items():
            if cid.startswith(prefix):
                groups[rubric].append(r)
                break
    if not groups:
        raise SystemExit(f"no {polarity} traces with the hits_all contract found — "
                         "run trace.py --all_cases first")
    for rubric, rs in groups.items():
        print(f">>> {rubric}: {len(rs)} cases ({polarity})")

    n_clusters = len(null)
    score: dict[str, np.ndarray] = {}
    presence: dict[str, np.ndarray] = {}
    for rubric, rs in groups.items():
        sc = np.zeros(n_clusters)
        pr = np.zeros(n_clusters, dtype=int)
        for r in rs:
            k = float(r["k"])
            case_vec = np.zeros(n_clusters)
            for pc in r["per_crux"]:
                for c, h in pc["hits_all"].items():
                    case_vec[int(c)] += h / k
            sc += case_vec / max(1, len(r["per_crux"]))
            own = {a["cluster"] for a in r["case_trigger_attrs"]}
            for c in own:
                pr[c] += 1
        score[rubric] = sc / len(rs)
        presence[rubric] = pr

    k_ret = float(json.loads((idx / "manifest.json").read_text())["null_k"])
    null_share = null / k_ret
    alpha = 1.0 / k_ret  # +1-hit smoothing in share units
    ts = timestamp()
    out_dir = Path(out) / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    agg: dict[str, list[dict]] = {}
    for rubric in groups:
        others = [score[r2] for r2 in groups if r2 != rubric]
        spec = score[rubric] - (np.mean(others, axis=0) if others else 0.0)
        lift = (score[rubric] + alpha) / (null_share + alpha)
        composite = lift * spec
        rows = []
        for c in np.argsort(-composite):
            c = int(c)
            if presence[rubric][c] < min_presence:
                continue
            s = sums.get(c, {})
            rows.append({
                "cluster": c, "size": s.get("size"), "rows": s.get("rows"),
                "coverage": s.get("coverage"),
                "share_reasoning": s.get("share_reasoning"),
                "score": round(float(score[rubric][c]), 5),
                "lift": round(float(lift[c]), 2),
                "spec": round(float(spec[c]), 5),
                "composite": round(float(composite[c]), 5),
                "presence": int(presence[rubric][c]), "n_cases": len(groups[rubric]),
                "summary": s.get("summary") or "[summary withheld]",
            })
            if len(rows) >= top:
                break
        agg[rubric] = rows

        lines = [f"# Property-hypothesis candidates — {rubric} "
                 f"({polarity}, {len(groups[rubric])} cases) — {ts}", "",
                 f"Gate: presence >= {min_presence}. Rank: lift x spec. "
                 "Read member attributes + case attributes before writing a "
                 "hypothesis; summaries are labels, not hypotheses.", "",
                 "| rank | cluster | lift | spec | presence | rows | cov | rsn | "
                 "summary |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for i, row in enumerate(rows, 1):
            share = (f"{row['share_reasoning']:.0%}"
                     if row["share_reasoning"] is not None else "?")
            cov = (f"{row['coverage']:.2%}" if row["coverage"] is not None else "?")
            lines.append(f"| {i} | #{row['cluster']} | {row['lift']} | "
                         f"{row['spec']:.4f} | {row['presence']}/{row['n_cases']} | "
                         f"{row['rows']} | {cov} | {share} | {row['summary']} |")
        lines.append("")
        for row in rows[:8]:
            c = row["cluster"]
            lines += [f"## #{c} — {row['summary']}", "", "Member attributes (sample):"]
            lines += [f"- ({m['channel']}) {m['text']}" for m in members[c][:4]]
            case_attrs = []
            for r in groups[rubric]:
                for a in r["case_trigger_attrs"]:
                    if a["cluster"] == c:
                        case_attrs.append((r["case_id"], a))
            if case_attrs:
                lines += ["", f"Case attributes landing here "
                              f"({len(case_attrs)} across cases):"]
                lines += [f"- [{cid}] ({a['channel']}) {a['attribute']}"
                          for cid, a in case_attrs[:4]]
            lines.append("")
        (out_dir / f"hypotheses_{rubric}.md").write_text("\n".join(lines))
        print(f">>> {out_dir}/hypotheses_{rubric}.md ({len(rows)} candidates)")

    (out_dir / "aggregate_result.json").write_text(json.dumps(
        {"rubrics": agg, "polarity": polarity, "min_presence": min_presence,
         "cases_per_rubric": {r: len(v) for r, v in groups.items()},
         "index": str(idx), "git_sha": git_sha(), "timestamp": ts}, indent=2))
    print(f">>> {out_dir}/aggregate_result.json")


if __name__ == "__main__":
    fire.Fire(main)
