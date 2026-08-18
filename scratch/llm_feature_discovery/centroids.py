# ABOUTME: Compute L2-normalised cluster centroids from an embedding matrix and a
# ABOUTME: feature->cluster map, skipping features the map omits as noise.

"""Cluster centroids.

A centroid is how a cluster gets compared to another cluster (the redundancy audit) and how
features from a *different* corpus get assigned to this corpus's clusters without refitting
(`scratch/odcv_cluster_assign.py`). Three scripts used to carry their own copy of this;
they share this one, so the noise rule below is enforced in exactly one place.

Features absent from the map are HDBSCAN noise and contribute to no centroid. That is the
point of leaving them out: a `-1` centroid would be the average of everything the clusterer
declined to group, and it would then attract whatever was assigned against it.
"""

from __future__ import annotations

import numpy as np

ROWS_PER_CHUNK = 2048


def compute(embeddings: np.ndarray, features: list[str], feature_to_cluster: dict[str, int],
            n_clusters: int) -> np.ndarray:
    """Compute L2-normalised centroids, streaming the embedding matrix in chunks.

    Args:
        embeddings: (n x d) array or memmap, rows aligned to `features`.
        features: Feature strings in embedding-row order.
        feature_to_cluster: Feature string -> cluster id; omitted features are noise.
        n_clusters: Number of clusters; ids are expected to be 0..n_clusters-1.

    Returns:
        (n_clusters x d) centroid matrix, rows L2-normalised.

    Raises:
        RuntimeError: If a cluster ended up with no members, which means `n_clusters` does
            not describe this map.
    """
    n_dims = embeddings.shape[1]
    sums = np.zeros((n_clusters, n_dims), dtype=np.float32)
    counts = np.zeros(n_clusters, dtype=np.int64)
    labels = np.array([feature_to_cluster.get(f, -1) for f in features], dtype=np.int32)

    for start in range(0, len(features), ROWS_PER_CHUNK):
        block = np.asarray(embeddings[start:start + ROWS_PER_CHUNK], dtype=np.float32)
        chunk_labels = labels[start:start + ROWS_PER_CHUNK]
        clustered = chunk_labels >= 0
        np.add.at(sums, chunk_labels[clustered], block[clustered])
        np.add.at(counts, chunk_labels[clustered], 1)

    n_clustered = int((labels >= 0).sum())
    assert counts.sum() == n_clustered, f"{counts.sum()} != {n_clustered}"
    if not counts.all():
        raise RuntimeError(f"clusters with no members: "
                           f"{np.flatnonzero(counts == 0).tolist()} — n_clusters="
                           f"{n_clusters} does not match this cluster map")
    centroids = sums / counts[:, None]
    return centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
