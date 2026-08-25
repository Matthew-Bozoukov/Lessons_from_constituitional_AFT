# ABOUTME: List every difficult-advice trace carrying the harm-risk-assessment cluster, plus the
# ABOUTME: clusters nearest it by centroid cosine, with the features that put each trace in.

"""Find all instances of the "Explicit multi-factor harm risk assessment" cluster.

The seed is the k=150 cluster whose features read "Analyzes probability and severity of harm",
"Analyzes reversibility of potential harms" and so on. Neighbouring clusters are pulled in by
centroid cosine, reported per threshold tier so the widening is visible rather than assumed —
a wider tier buys rows, and the lexical check at the bottom says whether those rows actually
carry the property or only sit near it in embedding space.

Run:
  uv run python scratch/find_harm_risk_instances.py
"""

from __future__ import annotations

import html
import json
import re
import sys
from collections import Counter
from pathlib import Path

import fire
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scratch.llm_feature_discovery import centroids  # noqa: E402
from src.utils import git_sha, timestamp  # noqa: E402

SEED_LABEL = "Explicit multi-factor harm risk assessment"
TIERS = (0.90, 0.87, 0.85, 0.82)

# The property in surface form: probability/likelihood, severity/magnitude, reversibility.
# Used only to measure how many rows in each tier actually exhibit it — the clusters decide
# membership, this decides whether membership means anything.
LEXICAL = re.compile(
    r"\b(probabilit\w*|likelihood|how likely|unlikely|severit\w*|how severe|magnitude"
    r"|reversib\w*|irreversib\w*|permanen\w*|undo\w*|recoverab\w*)\b", re.I)


def _centroids(emb_path: Path, uniq: list[str], fmap: dict[str, int], k: int) -> np.ndarray:
    """Compute L2-normalised cluster centroids from the run's embedding file.

    Thin wrapper over the module's shared implementation, which owns the rule that
    features absent from `fmap` are unclustered noise and contribute to no centroid.

    Args:
        emb_path: Path to the embeddings.npy (n x d, fp16).
        uniq: Feature strings in embedding-row order.
        fmap: Feature string -> cluster id, noise omitted.
        k: Number of clusters.

    Returns:
        (k x d) centroid matrix, rows L2-normalised.
    """
    return centroids.compute(np.load(emb_path, mmap_mode="r"), uniq, fmap, k)


def _reasoning(row: dict) -> str:
    """Pull the private reasoning out of an SFT row.

    Args:
        row: A stage_7 SFT record.

    Returns:
        The assistant's reasoning_content, or "" when the row has none.
    """
    return {m["role"]: m for m in row["messages"]}["assistant"].get("reasoning_content", "")


def _dashboard(path: Path, tiers: dict, seed: int, insts: dict, rows: dict,
               cl_label: dict) -> None:
    """Write a single-file browsable listing of every matched instance.

    Args:
        path: Output html path.
        tiers: Threshold -> list of cluster ids.
        seed: Seed cluster id.
        insts: scenario_id -> {cluster id -> [matched feature strings]}.
        rows: scenario_id -> SFT record.
        cl_label: cluster id -> label.
    """
    order = sorted(insts, key=lambda s: (seed not in insts[s], -len(insts[s]), s))
    parts = [
        "<meta charset='utf-8'><title>Harm-risk cluster instances</title>",
        "<style>body{font:14px/1.55 system-ui;margin:0 auto;max-width:1000px;padding:2rem;"
        "background:#fbfbfa}h1{font-size:1.4rem}details{border:1px solid #ddd;border-radius:6px;"
        "margin:.5rem 0;background:#fff}summary{padding:.6rem .8rem;cursor:pointer}"
        "pre{white-space:pre-wrap;background:#f6f6f4;padding:.8rem;border-radius:4px;"
        "font:12px/1.5 ui-monospace,monospace}.f{color:#444}.seed{color:#b00;font-weight:600}"
        "code{background:#eee;padding:0 .25rem;border-radius:3px}</style>",
        f"<h1>Harm-risk cluster instances</h1><p>Seed cluster C{seed} "
        f"&mdash; <b>{html.escape(cl_label[seed])}</b>. Tiers: " +
        ", ".join(f"cos&ge;{t}: {len(cs)} clusters" for t, cs in tiers.items()) + "</p>",
    ]
    for sid in order:
        hit = insts[sid]
        tag = "<span class='seed'>[seed C%d]</span> " % seed if seed in hit else ""
        feats = sorted({f for fs in hit.values() for f in fs})
        parts.append(
            f"<details><summary>{tag}<code>{html.escape(sid)}</code> &mdash; "
            f"{len(hit)} cluster(s): " +
            ", ".join(f"C{c}" for c in sorted(hit)) + "</summary>"
            "<div class='f'><b>matched features</b><ul>" +
            "".join(f"<li>{html.escape(f)}</li>" for f in feats) + "</ul></div>"
            f"<pre>{html.escape(_reasoning(rows[sid]))}</pre></details>")
    path.write_text("\n".join(parts))


def main(
    run_dir: str = "output/feature_discovery/20260812_092119",
    sft: str = "output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl",
    tier: float = 0.87,
    out_dir: str | None = None,
) -> None:
    """List every trace in the harm-risk cluster and its centroid neighbours.

    Args:
        run_dir: Feature-discovery run holding clusters, features and embeddings.
        sft: The difficult-advice SFT file the traces came from.
        tier: Which threshold's membership the instance listing uses.
        out_dir: Output directory; defaults to output/harm_risk_instances/<timestamp>.
    """
    d = Path(run_dir)
    clusters = json.loads((d / "clusters.json").read_text())["clusters"]
    fmap = json.loads((d / "feature_cluster_map.json").read_text())
    uniq = [x for x in (d / "unique_features.txt").read_text().splitlines() if x.strip()]
    traces = [json.loads(x) for x in (d / "features.jsonl").read_text().splitlines() if x.strip()]
    rows = {json.loads(x)["metadata"]["scenario_id"]: json.loads(x)
            for x in Path(sft).read_text().splitlines() if x.strip()}
    by_id = {c["cluster"]: c for c in clusters}
    cl_label = {c["cluster"]: c["label"] for c in clusters}

    seed = next(c["cluster"] for c in clusters if c["label"] == SEED_LABEL)
    cen = _centroids(d / "embeddings.npy", uniq, fmap, len(clusters))
    sims = cen @ cen[seed]
    nearest = sorted(((float(sims[c]), c) for c in by_id if c != seed), reverse=True)
    tiers = {t: [seed] + [c for s, c in nearest if s >= t] for t in TIERS}

    # scenario_id -> {cluster -> matched features}, for the chosen tier.
    picked = set(tiers[tier])
    insts: dict[str, dict[int, list[str]]] = {}
    for t in traces:
        hit: dict[int, list[str]] = {}
        for f in t["features"]:
            c = fmap.get(f)         # None => unclustered (HDBSCAN noise)
            if c in picked:
                hit.setdefault(c, []).append(f)
        if hit:
            insts[t["scenario_id"]] = hit
    per_cluster = {c: sorted(s for s, h in insts.items() if c in h) for c in picked}

    out = Path(out_dir or f"output/harm_risk_instances/{timestamp()}")
    out.mkdir(parents=True, exist_ok=True)

    lines = [f"# Instances of C{seed} — {SEED_LABEL}", "",
             f"Run `{run_dir}`, k={len(clusters)}, {len(traces)} traces labelled.", "",
             "## How wide the net is at each threshold", "",
             "| centroid cosine | clusters | traces | of corpus | lexically exhibit it |",
             "|---|--:|--:|--:|--:|"]
    for t, cs in tiers.items():
        sids = {tr["scenario_id"] for tr in traces if {fmap[f] for f in tr["features"] if f in fmap} & set(cs)}
        lex = sum(1 for s in sids if LEXICAL.search(_reasoning(rows[s])))
        lines.append(f"| ≥{t} | {len(cs)} | {len(sids)} | {len(sids) / len(traces):.1%} | "
                     f"{lex} ({lex / len(sids):.0%}) |")
    base = {tr["scenario_id"] for tr in traces} - {
        tr["scenario_id"] for tr in traces
        if {fmap[f] for f in tr["features"] if f in fmap} & set(tiers[min(TIERS)])}
    lines += ["", f"Baseline for comparison: of the {len(base)} traces in no tier at all, "
                  f"{sum(1 for s in base if LEXICAL.search(_reasoning(rows[s])))} "
                  f"({sum(1 for s in base if LEXICAL.search(_reasoning(rows[s]))) / len(base):.0%})"
                  " match the same lexical pattern.", ""]

    lines += ["## The clusters, nearest first", "",
              "| cos to seed | cluster | label | features | traces |", "|--:|--:|---|--:|--:|",
              f"| 1.000 | C{seed} | **{cl_label[seed]}** | {by_id[seed]['n_features']} | "
              f"{by_id[seed]['n_traces']} |"]
    for s, c in nearest[:14]:
        mark = " ✓" if c in picked else ""
        lines.append(f"| {s:.3f}{mark} | C{c} | {cl_label[c]} | {by_id[c]['n_features']} | "
                     f"{by_id[c]['n_traces']} |")
    lines += ["", f"✓ = included at the reporting threshold ({tier}).", ""]

    lines += [f"## Instances at cosine ≥{tier}: {len(insts)} traces", "",
              "| cluster | label | traces |", "|--:|---|--:|"]
    for c in sorted(picked, key=lambda c: -len(per_cluster[c])):
        lines.append(f"| C{c} | {cl_label[c]} | {len(per_cluster[c])} |")

    both = Counter(len(h) for h in insts.values())
    lines += ["", f"Overlap: " + ", ".join(f"{n} cluster(s): {c} traces"
                                           for n, c in sorted(both.items())), ""]

    lines += ["## Every instance", "",
              "`scenario_id` — clusters — the features that matched", ""]
    for sid in sorted(insts, key=lambda s: (seed not in insts[s], s)):
        hit = insts[sid]
        feats = "; ".join(sorted({f for fs in hit.values() for f in fs}))
        lines.append(f"- `{sid}` — {' '.join(f'C{c}' for c in sorted(hit))} — {feats}")

    (out / "instances.md").write_text("\n".join(lines) + "\n")
    (out / "instances.jsonl").write_text("".join(
        json.dumps({"scenario_id": s, "clusters": sorted(insts[s]),
                    "features": sorted({f for fs in insts[s].values() for f in fs}),
                    "lexically_exhibits": bool(LEXICAL.search(_reasoning(rows[s])))}) + "\n"
        for s in sorted(insts)))
    (out / "scenario_ids.txt").write_text("\n".join(sorted(insts)) + "\n")
    _dashboard(out / "instances.html", tiers, seed, insts, rows, cl_label)
    (out / "run_meta.json").write_text(json.dumps(
        {"git_sha": git_sha(), "timestamp_utc": timestamp(), "run_dir": run_dir, "sft": sft,
         "seed_cluster": seed, "seed_label": SEED_LABEL, "reporting_tier": tier,
         "tiers": {str(t): cs for t, cs in tiers.items()}, "instances": len(insts),
         "command": " ".join(sys.argv)}, indent=2))

    print(f"seed C{seed}: {SEED_LABEL}")
    for t, cs in tiers.items():
        sids = {tr["scenario_id"] for tr in traces if {fmap[f] for f in tr["features"] if f in fmap} & set(cs)}
        print(f"  cos>={t}: {len(cs)} clusters -> {len(sids)} traces")
    print(f"\nreporting tier {tier}: {len(insts)} instances -> {out}")


if __name__ == "__main__":
    fire.Fire(main)
