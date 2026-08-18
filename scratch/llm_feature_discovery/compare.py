# ABOUTME: Gate a clustering: agreement with a baseline clustering of the same features,
# ABOUTME: and how much the clustering moves when the seed and n_neighbors change.

"""Whether a clustering can be trusted before anything is read off it.

Swapping or retuning the clusterer changes the vocabulary every downstream analysis is
written in, so it needs a gate rather than a look at the labels.

* **Agreement** — ARI/AMI against a baseline run over the same feature list. The baseline is
  usually the published k-means clustering (`output/feature_discovery/20260812_092119`):
  close agreement means the numbers already reported were sound and can stand.
* **Stability** — re-fit across `n_neighbors` and seeds and take pairwise ARI. A clustering
  that reshuffles when the seed changes is not a finding.

Geometry — the third question, and the one that can disqualify a run on its own — lives in
`geometry.py`.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from scratch.llm_feature_discovery import cluster
from scratch.llm_feature_discovery.cluster import ClusterParams


def agreement(baseline_map: dict[str, int], candidate_map: dict[str, int],
              features: list[str]) -> dict:
    """Score two clusterings of the same feature list against each other.

    Only features that BOTH clusterings placed are scored; anything either one left as
    noise is dropped and counted, because ARI is undefined for a point with no label.

    Args:
        baseline_map: Feature -> cluster id from the baseline run.
        candidate_map: Feature -> cluster id from the run under test.
        features: The shared feature list.

    Returns:
        ARI, AMI, and the counts they were computed over.

    Raises:
        ValueError: If the two clusterings share no clustered features.
    """
    shared = [f for f in features if f in baseline_map and f in candidate_map]
    if not shared:
        raise ValueError("the two clusterings share no clustered features")
    baseline = np.array([baseline_map[f] for f in shared], dtype=np.int32)
    candidate = np.array([candidate_map[f] for f in shared], dtype=np.int32)
    return {"ari": float(adjusted_rand_score(baseline, candidate)),
            "ami": float(adjusted_mutual_info_score(baseline, candidate)),
            "features_compared": len(shared),
            "features_dropped_as_noise": len(features) - len(shared),
            "baseline_clusters": int(len(set(baseline.tolist()))),
            "candidate_clusters": int(len(set(candidate.tolist())))}


def stability_sweep(embeddings: np.ndarray, params: ClusterParams,
                    reference_labels: np.ndarray, neighbors: tuple[int, ...],
                    seeds: tuple[int, ...]) -> tuple[list[dict], list[list[float]]]:
    """Re-fit across n_neighbors and seeds and score every fit against the others.

    Each fit is scored against the reference labelling AND against every other fit, so one
    lucky seed cannot pass as agreement.

    Args:
        embeddings: (n x d) L2-normalised feature vectors.
        params: The reference knobs; n_neighbors is overridden per sweep point.
        reference_labels: Labels from the reference fit.
        neighbors: n_neighbors values to sweep.
        seeds: Seeds to sweep.

    Returns:
        (per-fit records, pairwise ARI matrix in the same order).
    """
    sweep, labelings = [], []
    for n_neighbors in neighbors:
        for seed in seeds:
            swept = ClusterParams(n_neighbors=n_neighbors, min_dist=params.min_dist,
                                  n_components=params.n_components,
                                  min_cluster_size=params.min_cluster_size)
            coords = cluster.reduce_embeddings(embeddings, swept, seed)
            labels = cluster.cluster_coords(coords, swept)
            n_noise = int((labels == cluster.NOISE_LABEL).sum())
            record = {"n_neighbors": n_neighbors, "seed": seed,
                      "n_clusters": int(labels.max()) + 1,
                      "noise_fraction": n_noise / len(labels),
                      "ari_vs_reference": float(adjusted_rand_score(reference_labels, labels))}
            sweep.append(record)
            labelings.append(labels)
            print(f"  n_neighbors={n_neighbors} seed={seed}: {record['n_clusters']} "
                  f"clusters, {record['noise_fraction']:.1%} noise, "
                  f"ARI vs reference {record['ari_vs_reference']:.3f}")
    pairwise = [[float(adjusted_rand_score(a, b)) for b in labelings] for a in labelings]
    return sweep, pairwise
