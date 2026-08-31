# ABOUTME: Recall audit for one feature cluster: does it actually capture every trace that
# ABOUTME: exhibits its behaviour, or do neighbours and missed extractions eat the rest?

"""Audit how completely a cluster covers the behaviour its label claims.

A cluster label is a claim about coverage, and there are two ways it can be wrong:

1. CLUSTERING LEAKAGE — the autorater generated a fitting feature, but k-means filed it
   under a neighbouring cluster. Measured in embedding space: features whose cosine to the
   target centroid is high but which are assigned elsewhere.
2. EXTRACTION MISS — the trace exhibits the behaviour and no feature was generated for it
   at all. Estimated lexically: traces whose raw reasoning text carries unambiguous markers
   of the behaviour, but which carry no feature from the cluster or its neighbours.

Neither number is the truth. (1) is exact given the embedding; (2) is a keyword proxy whose
markers are listed in the output so they can be judged.

Run:
  uv run python scratch/feature_discovery/audit_cluster_recall.py --cluster 116
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import fire
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Unambiguous surface markers of "I will not assert what I cannot verify". Deliberately
# narrow: each must be a statement about the model's own epistemic access, not the user's.
MARKERS = [
    r"\bI can(?:no|')?t (?:independently )?verify\b",
    r"\bI have no way (?:to|of) (?:verify|confirm|check)\w*\b",
    r"\bI can(?:no|')?t confirm\b",
    r"\bwithout (?:seeing|access to)\b",
    r"\bI don'?t (?:have|know) (?:the|enough|whether)\b",
    r"\bI'?m not (?:certain|sure) (?:that|whether|if)\b",
    r"\bI (?:shouldn'?t|won'?t|can'?t) (?:guess|invent|make up|fabricate)\b",
    r"\bno way for me to (?:know|tell|verify)\b",
    r"\bI'?d be (?:guessing|making (?:it|that) up)\b",
]


def main(run_dir: str = "output/feature_discovery/20260812_092119",
         sft: str = "output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl",
         cluster: int = 116, near: float = 0.75, neighbours: int = 6) -> None:
    """Report clustering leakage and estimated extraction misses for one cluster.

    Args:
        run_dir: Feature-discovery run directory.
        sft: The SFT file the traces came from.
        cluster: Cluster id to audit.
        near: Cosine to the cluster centroid above which a feature "belongs".
        neighbours: How many nearest sibling clusters to treat as the same family.
    """
    d = Path(run_dir)
    data = json.loads((d / "clusters.json").read_text())
    by_id = {c["cluster"]: c for c in data["clusters"]}
    fmap = json.loads((d / "feature_cluster_map.json").read_text())
    uniq = [x for x in (d / "unique_features.txt").read_text().splitlines() if x.strip()]
    traces = [json.loads(x) for x in (d / "features.jsonl").read_text().splitlines() if x.strip()]
    reasoning = {json.loads(x)["metadata"]["scenario_id"]: json.loads(x)["messages"][2]["reasoning_content"]
                 for x in Path(sft).read_text().splitlines() if x.strip()}

    target = by_id[cluster]
    print(f"AUDIT C{cluster}: {target['label']}")
    print(f"  as clustered: {target['n_traces']} traces ({target['prevalence']:.1%}), "
          f"{target['n_features']} unique features\n")

    x = np.load(d / "embeddings.npy", mmap_mode="r")
    labels = np.array([fmap[f] for f in uniq])
    idx = np.where(labels == cluster)[0]
    cent = np.asarray(x[idx], dtype=np.float32).mean(0)
    cent /= np.linalg.norm(cent)

    sims = np.empty(len(uniq), dtype=np.float32)
    for s in range(0, len(uniq), 2048):
        sims[s:s + 2048] = np.asarray(x[s:s + 2048], dtype=np.float32) @ cent

    # 1. Clustering leakage.
    close = np.where(sims >= near)[0]
    outside = [i for i in close if labels[i] != cluster]
    print(f"1. CLUSTERING LEAKAGE (cosine >= {near} to the C{cluster} centroid)")
    print(f"   {len(close)} features that close; {len(outside)} of them "
          f"({len(outside) / max(len(close), 1):.0%}) are filed under other clusters")
    spread = Counter(by_id[int(labels[i])]["label"] for i in outside)
    for lab, n in spread.most_common(8):
        print(f"     {n:>4}  {lab}")
    ex = sorted(outside, key=lambda i: -sims[i])[:6]
    print("   closest examples filed elsewhere:")
    for i in ex:
        print(f"     {sims[i]:.3f}  {uniq[i]!r} -> {by_id[int(labels[i])]['label']}")

    # The family: clusters whose centroids are nearest this one.
    cents = {}
    for c in by_id:
        ci = np.where(labels == c)[0]
        v = np.asarray(x[ci], dtype=np.float32).mean(0)
        cents[c] = v / np.linalg.norm(v)
    fam = sorted(((float(cent @ v), c) for c, v in cents.items() if c != cluster), reverse=True)
    family = [cluster] + [c for _, c in fam[:neighbours]]
    print(f"\n2. THE FAMILY (nearest {neighbours} clusters by centroid)")
    for s, c in fam[:neighbours]:
        print(f"   {s:.3f}  C{c} {by_id[c]['label']} ({by_id[c]['prevalence']:.1%})")

    tset = {t["scenario_id"]: {fmap[f] for f in t["features"]} for t in traces}
    fam_traces = {s for s, m in tset.items() if m & set(family)}
    print(f"   union of the family: {len(fam_traces)} traces "
          f"({len(fam_traces) / len(traces):.1%}) vs C{cluster} alone {target['n_traces']}")

    # 3. Extraction misses, estimated lexically.
    rx = [re.compile(p, re.I) for p in MARKERS]
    marked = {t["scenario_id"] for t in traces
              if any(r.search(reasoning[t["scenario_id"]]) for r in rx)}
    in_c = marked & {s for s, m in tset.items() if cluster in m}
    in_fam = marked & fam_traces
    print(f"\n3. EXTRACTION RECALL (lexical proxy over raw reasoning text)")
    print(f"   traces whose reasoning states an epistemic limit: {len(marked)} "
          f"({len(marked) / len(traces):.1%})")
    print(f"     of those, carrying a C{cluster} feature:     {len(in_c)} "
          f"({len(in_c) / max(len(marked), 1):.1%})")
    print(f"     of those, carrying ANY family feature:   {len(in_fam)} "
          f"({len(in_fam) / max(len(marked), 1):.1%})")
    missed = sorted(marked - fam_traces)
    print(f"     covered by neither: {len(missed)}")
    print("\n   markers used:")
    for p in MARKERS:
        n = sum(1 for s in marked if re.search(p, reasoning[s], re.I))
        print(f"     {n:>4}  {p}")
    if missed:
        print(f"\n   sample of traces with the language but no family feature:")
        for s in missed[:3]:
            hit = next((r.search(reasoning[s]) for r in rx if r.search(reasoning[s])), None)
            frag = reasoning[s][max(0, hit.start() - 90):hit.end() + 90].replace("\n", " ")
            print(f"     {s}: ...{frag}...")


if __name__ == "__main__":
    fire.Fire(main)
