# ABOUTME: Reduce the feature embeddings with UMAP and density-cluster the reduction with
# ABOUTME: HDBSCAN. Vectors in, labels out — no I/O, no naming, no statistics.

"""UMAP + HDBSCAN over the feature embeddings.

HDBSCAN labels low-density points -1 (noise) and discovers the cluster count instead of
taking it as an argument, so `min_cluster_size` is the only resolution knob. Mini-batch
k-means at a fixed k was the earlier clusterer and was removed on 2026-08-18: it forced
every feature into a cluster and made the cluster count an argument rather than a finding.

The whole matrix has to be resident — UMAP's k-nearest-neighbour graph cannot be built
from a memmap in chunks — but 33k x 4096 fp32 is ~0.5 GB, so this runs on the laptop and
needs no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import HDBSCAN

NOISE_LABEL = -1


@dataclass(frozen=True)
class ClusterParams:
    """The knobs, kept together so they travel from CLI to fit to `clusters.json` intact.

    Attributes:
        n_neighbors: UMAP neighbourhood size — the local/global structure trade-off.
        min_dist: UMAP minimum spacing. 0.0 packs points tightly, which is what a density
            clusterer downstream wants.
        n_components: Dimensionality of the reduction handed to HDBSCAN.
        min_cluster_size: Smallest group HDBSCAN will call a cluster. Raise it for coarser,
            more readable clusters; lower it to split fine-grained behaviours back out.
    """

    n_neighbors: int = 15
    min_dist: float = 0.0
    n_components: int = 10
    min_cluster_size: int = 220

    def as_dict(self) -> dict:
        """The params as plain JSON.

        Returns:
            A dict of the four knobs.
        """
        return {"n_neighbors": self.n_neighbors, "min_dist": self.min_dist,
                "n_components": self.n_components,
                "min_cluster_size": self.min_cluster_size}

    def describe(self) -> str:
        """One-line human description for reports.

        Returns:
            e.g. "UMAP(10d, n_neighbors=15, min_dist=0.0) + HDBSCAN(min_cluster_size=220)".
        """
        return (f"UMAP({self.n_components}d, n_neighbors={self.n_neighbors}, "
                f"min_dist={self.min_dist}) + "
                f"HDBSCAN(min_cluster_size={self.min_cluster_size})")


@dataclass(frozen=True)
class Clustering:
    """The result of one fit.

    Attributes:
        labels: (n,) int32 cluster id per row, NOISE_LABEL for unclustered rows.
        coords: (n x m) float32 UMAP coordinates that were clustered.
        params: The knobs that produced it.
    """

    labels: np.ndarray
    coords: np.ndarray
    params: ClusterParams = field(default_factory=ClusterParams)

    @property
    def n_clusters(self) -> int:
        """How many clusters were found.

        Returns:
            The cluster count, 0 if everything was noise.
        """
        return int(self.labels.max()) + 1 if (self.labels >= 0).any() else 0

    @property
    def n_noise(self) -> int:
        """How many rows were left unclustered.

        Returns:
            The noise count.
        """
        return int((self.labels == NOISE_LABEL).sum())


def reduce_embeddings(embeddings: np.ndarray, params: ClusterParams, seed: int) -> np.ndarray:
    """Run UMAP over the embeddings.

    Args:
        embeddings: (n x d) L2-normalised feature vectors.
        params: UMAP knobs.
        seed: Random seed. This forces UMAP single-threaded, which costs minutes;
            reproducibility of a clustering everything downstream is written in is worth it.

    Returns:
        (n x m) float32 coordinates.
    """
    # Imported here, not at module top: umap-learn pulls in numba, whose JIT import costs
    # seconds that importing this module for ClusterParams alone should not pay.
    import umap

    reducer = umap.UMAP(n_components=params.n_components, n_neighbors=params.n_neighbors,
                        min_dist=params.min_dist, metric="cosine", random_state=seed)
    return np.asarray(reducer.fit_transform(embeddings), dtype=np.float32)


def cluster_coords(coords: np.ndarray, params: ClusterParams) -> np.ndarray:
    """Density-cluster an already-reduced coordinate set.

    Args:
        coords: (n x m) coordinates.
        params: HDBSCAN knobs.

    Returns:
        (n,) int32 labels, NOISE_LABEL for unclustered rows.
    """
    labels = HDBSCAN(min_cluster_size=params.min_cluster_size,
                     cluster_selection_method="eom").fit_predict(coords)
    return labels.astype(np.int32)


def fit(embeddings: np.ndarray, params: ClusterParams, seed: int,
        verbose: bool = True) -> Clustering:
    """Reduce then cluster, the two steps that always go together.

    Args:
        embeddings: (n x d) L2-normalised feature vectors.
        params: The knobs.
        seed: Random seed.
        verbose: Print shape and outcome as it goes.

    Returns:
        The Clustering.

    Raises:
        RuntimeError: If HDBSCAN found no clusters at all, which means min_cluster_size is
            too large for this corpus rather than that the corpus has no structure.
    """
    if verbose:
        print(f"embeddings {embeddings.shape[0]} x {embeddings.shape[1]}, "
              f"{params.describe()}")
    coords = reduce_embeddings(embeddings, params, seed)
    labels = cluster_coords(coords, params)
    clustering = Clustering(labels=labels, coords=coords, params=params)
    if clustering.n_clusters == 0:
        raise RuntimeError(f"no clusters at min_cluster_size={params.min_cluster_size}; "
                           f"all {len(labels)} features are noise — lower it")
    if verbose:
        print(f"  {clustering.n_clusters} clusters, {clustering.n_noise} noise "
              f"({clustering.n_noise / len(labels):.1%})")
    return clustering


def build_feature_cluster_map(features: list[str], labels: np.ndarray) -> dict[str, int]:
    """Pair features with their cluster ids, dropping noise.

    Noise is deliberately kept OUT of the map. Every consumer averages a cluster's members
    into a centroid, and a -1 "cluster" would get a meaningless one that then attracts
    features assigned against it. Omitting noise keeps all of them working; the counts
    reported alongside keep it from vanishing silently.

    Args:
        features: Feature strings, in label order.
        labels: (n,) cluster ids.

    Returns:
        feature -> cluster id, for clustered features only.

    Raises:
        ValueError: If the two inputs are not the same length.
    """
    if len(features) != len(labels):
        raise ValueError(f"{len(features)} features vs {len(labels)} labels")
    return {feature: int(label)
            for feature, label in zip(features, labels.tolist()) if label >= 0}
