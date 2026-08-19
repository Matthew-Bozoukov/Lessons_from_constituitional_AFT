# ABOUTME: The grouping stage every producer shares: reduce in {none, umap} crossed with
# ABOUTME: cluster in {kmeans, hdbscan}, plus the noise contract and centroid assignment.

"""Turn a matrix into groups.

Producers differ in what they embed — free-text features, attribute sentences, whole
traces — but they all then ask the same question: which of these are the same thing? Four
combinations, chosen in a config, all returning the same `Grouping`:

    reduce   none   cluster in the full embedding space. Cosines stay comparable to
                    anything else measured in that space.
             umap   reduce first. Denser neighbourhoods, so HDBSCAN finds structure it
                    otherwise calls noise — at the cost that REDUCED-SPACE COSINES ARE NOT
                    COMPARABLE TO FULL-DIMENSIONAL ONES. UMAP compresses, so every cosine
                    rises; only the ORDERING carries over.

    cluster  kmeans   SURF's k-means, ported verbatim (squared-Euclidean Lloyd's, random
                      subset init, empty clusters keep their centroid, 20 iters, <0.1%
                      inertia early stop, seed 42). `k` is an ARGUMENT: the cluster count
                      is a resolution setting you chose, never a count of behaviours found.
             hdbscan  discovers the count and leaves low-density points unclustered.
                      `min_cluster_size` is the resolution knob `k` was.

The noise contract (HDBSCAN only). Low-density points get label -1. They are counted, never
hidden, and they are EXCLUDED from centroids: every consumer averages a group's members
into a centroid, and a -1 "group" would get a meaningless one that then attracts new points
assigned against it. `Grouping.noise_mask` and `n_noise` make the uncovered share visible
to whoever reads the property list.

Before trusting a clustering you changed, gate it against the previous one on the same
matrix — `compare()` answers the three questions in the order they can disqualify a run:
did the reduction keep the geometry, do the two labelings agree, and is the result stable
across seeds.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

REDUCERS = ("none", "umap")
CLUSTERERS = ("kmeans", "hdbscan")


@dataclass(frozen=True)
class GroupingParams:
    """How to group. Every field lands in the property rows' provenance.

    Attributes:
        reduce: "none" or "umap".
        cluster: "kmeans" or "hdbscan".
        k: Number of clusters (kmeans only). A RESOLUTION SETTING, not a finding.
        min_cluster_size: Smallest admissible cluster (hdbscan only). Roughly
            `len(points) / k` if you are porting a k you liked.
        min_samples: HDBSCAN's conservativeness; None lets it default to min_cluster_size.
        n_neighbors: UMAP neighbourhood size.
        n_components: UMAP output dimensions.
        min_dist: UMAP minimum distance.
        metric: Distance metric for UMAP and HDBSCAN. Euclidean on L2-normalised vectors
            is monotone in cosine, so this is the cosine ordering either way.
        seed: Random state for both stages. UMAP runs single-threaded when seeded; that is
            the price of a reproducible reduction and it is worth paying.
        max_iter: k-means iterations (SURF's 20).
    """

    reduce: str = "umap"
    cluster: str = "hdbscan"
    k: int = 150
    min_cluster_size: int = 220
    min_samples: int | None = None
    n_neighbors: int = 15
    n_components: int = 5
    min_dist: float = 0.0
    metric: str = "euclidean"
    seed: int = 42
    max_iter: int = 20

    def validate(self) -> GroupingParams:
        """Check the two enum fields.

        Returns:
            Self, so this chains.

        Raises:
            ValueError: On an unknown reducer or clusterer.
        """
        if self.reduce not in REDUCERS:
            raise ValueError(f"reduce must be one of {REDUCERS}, got {self.reduce!r}")
        if self.cluster not in CLUSTERERS:
            raise ValueError(f"cluster must be one of {CLUSTERERS}, got {self.cluster!r}")
        return self

    def to_dict(self) -> dict:
        """The params as a plain dict, dropping the ones this combination ignores.

        Returns:
            Only the settings that actually influenced the result, so a report cannot
            imply that `k` mattered to an HDBSCAN run.
        """
        out = {"reduce": self.reduce, "cluster": self.cluster, "metric": self.metric,
               "seed": self.seed}
        if self.reduce == "umap":
            out |= {"n_neighbors": self.n_neighbors, "n_components": self.n_components,
                    "min_dist": self.min_dist}
        if self.cluster == "kmeans":
            out |= {"k": self.k, "max_iter": self.max_iter}
        else:
            out |= {"min_cluster_size": self.min_cluster_size,
                    "min_samples": self.min_samples}
        return out


@dataclass(frozen=True)
class Grouping:
    """The result of grouping one matrix.

    Attributes:
        labels: (n,) group id per input row; -1 means noise (hdbscan only).
        centroids: (g x d) group centroids in the SPACE THAT WAS CLUSTERED, ordered by
            group id 0..g-1. Noise never contributes to one.
        coords: The reduced coordinates that were clustered, or None when reduce="none".
        params: The GroupingParams used.
        meta: Counts and provenance.
    """

    labels: np.ndarray
    centroids: np.ndarray
    coords: np.ndarray | None
    params: GroupingParams
    meta: dict = field(default_factory=dict)

    @property
    def noise_mask(self) -> np.ndarray:
        """Boolean mask of the rows no group claimed.

        Returns:
            (n,) True where the row is noise.
        """
        return self.labels < 0

    @property
    def n_groups(self) -> int:
        """How many groups this actually produced — a finding under hdbscan, a setting
        under kmeans.

        Returns:
            The group count.
        """
        return int(self.centroids.shape[0])

    @property
    def n_noise(self) -> int:
        """How many rows were left unclustered.

        Returns:
            The noise count.
        """
        return int(self.noise_mask.sum())

    def members(self, group: int) -> np.ndarray:
        """Row indices belonging to one group.

        Args:
            group: Group id.

        Returns:
            The indices.
        """
        return np.flatnonzero(self.labels == group)


def reduce_umap(vectors: np.ndarray, params: GroupingParams) -> np.ndarray:
    """Reduce with UMAP.

    Args:
        vectors: (n x d) matrix.
        params: Grouping params.

    Returns:
        (n x n_components) float32 coordinates.
    """
    import umap

    reducer = umap.UMAP(n_neighbors=params.n_neighbors,
                        n_components=params.n_components, min_dist=params.min_dist,
                        metric=params.metric, random_state=params.seed)
    return np.asarray(reducer.fit_transform(np.asarray(vectors, dtype=np.float32)),
                      dtype=np.float32)


def project_2d(vectors: np.ndarray, params: GroupingParams) -> np.ndarray:
    """A 2-D projection for LOOKING at, separate from the one that was clustered.

    The post runs UMAP twice on purpose — down to two dimensions to visualise, and down to
    ten to cluster — because the reduction that makes density measurable is not the one
    that makes a readable picture. This is the first of those.

    It is a SEPARATE FIT, so points close together here are not necessarily in the same
    cluster: the clustering happened in `n_components` dimensions and this is a different
    projection of the same points. The picture is for spotting shape — one blob, a long
    filament, a cluster torn in half — not for adjudicating membership. Whoever renders it
    must say so.

    Args:
        vectors: (n x d) the matrix that was clustered.
        params: The run's grouping params; reused so the projection is at least seeded and
            parameterised the same way.

    Returns:
        (n x 2) float32 coordinates.
    """
    if params.reduce == "none" or vectors.shape[1] <= 2:
        # Nothing was reduced, so there is no second reduction to make; the first two
        # dimensions are an honest enough scatter for a shape check.
        return np.asarray(vectors[:, :2], dtype=np.float32)
    return reduce_umap(vectors, dataclasses.replace(params, n_components=2))


def kmeans(x: np.ndarray, k: int, max_iter: int = 20, seed: int = 42,
           batch_size: int = 65536) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SURF's k-means, ported from surf/clustering/cluster.py (`_run_kmeans`).

    Full-batch Lloyd's with squared-Euclidean distance, random-subset init, empty clusters
    keeping their previous centroid, and an early stop when inertia improves by <0.1%.
    Centroids are NOT re-normalised: assigning NEW points to these clusters is cosine, per
    SURF's `cluster_mapper.py`. Runs on cuda > mps > cpu.

    Kept faithful rather than swapped for sklearn because TURF's published index was built
    with it, and a different estimator on the same matrix is a different index.

    Args:
        x: (n x d) matrix.
        k: Number of clusters.
        max_iter: Iterations.
        seed: torch manual seed (SURF's random_state).
        batch_size: Rows per device transfer.

    Returns:
        (centroids [k, d] float32, labels [n], distance-to-centroid [n]).
    """
    import torch

    torch.manual_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    n, dim = x.shape
    matrix = torch.from_numpy(np.ascontiguousarray(x)).float()
    centroids = matrix[torch.randperm(n)[:k]].to(device)

    previous = float("inf")
    for _ in range(max_iter):
        fresh = torch.zeros_like(centroids)
        counts = torch.zeros(k, device=device)
        inertia = 0.0
        for i in range(0, n, batch_size):
            batch = matrix[i:i + batch_size].to(device)
            rows = batch.shape[0]
            # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x.c
            distances = ((batch ** 2).sum(dim=1, keepdim=True)
                         + (centroids ** 2).sum(dim=1, keepdim=True).T
                         - 2 * batch @ centroids.T)
            closest, assignment = distances.min(dim=1)
            inertia += closest.sum().item()
            fresh.scatter_add_(0, assignment.unsqueeze(1).expand(rows, dim), batch)
            counts.scatter_add_(0, assignment, torch.ones(rows, device=device))
        filled = counts > 0
        fresh[filled] = fresh[filled] / counts[filled].unsqueeze(1)
        fresh[~filled] = centroids[~filled]
        centroids = fresh
        improvement = (previous - inertia) / previous if previous != float("inf") else 0
        if 0 < improvement < 0.001:
            break
        previous = inertia

    labels, distances = assign(x, centroids.cpu().numpy().astype(np.float32))
    return centroids.cpu().numpy().astype(np.float32), labels, distances


def assign(x: np.ndarray, centroids: np.ndarray,
           batch: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-centroid assignment against EXISTING centroids (numpy, squared-Euclidean).

    This is the function that makes cross-corpus prevalence comparable: assigning a new
    corpus's points to a previous run's centroids — rather than refitting — means the two
    prevalences are measured against the same groups. Refitting produces different groups
    with the same names, and comparing those numbers is the mistake this exists to prevent.

    Args:
        x: (n x d) points to assign.
        centroids: (g x d) centroids from a previous Grouping.
        batch: Rows per chunk.

    Returns:
        (labels [n], distance-to-centroid [n]).
    """
    labels = np.empty(len(x), np.int64)
    distances = np.empty(len(x), np.float32)
    squared = (centroids ** 2).sum(axis=1)
    for i in range(0, len(x), batch):
        chunk = x[i:i + batch]
        d2 = ((chunk ** 2).sum(axis=1, keepdims=True) + squared[None, :]
              - 2 * chunk @ centroids.T)
        nearest = d2.argmin(axis=1)
        labels[i:i + len(chunk)] = nearest
        distances[i:i + len(chunk)] = np.sqrt(
            np.clip(d2[np.arange(len(chunk)), nearest], 0, None))
    return labels, distances


def _centroids_excluding_noise(space: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Mean of each group's members, with noise excluded by construction.

    Args:
        space: (n x d) the space that was clustered.
        labels: (n,) group ids; -1 is noise.

    Returns:
        (g x d) centroids ordered by group id.
    """
    groups = sorted(int(g) for g in set(labels.tolist()) if g >= 0)
    if not groups:
        return np.zeros((0, space.shape[1]), dtype=np.float32)
    return np.stack([space[labels == g].mean(axis=0) for g in groups]).astype(np.float32)


def group(vectors: np.ndarray, params: GroupingParams | None = None) -> Grouping:
    """Reduce (or not), then cluster.

    Args:
        vectors: (n x d) L2-normalised embeddings from `shared/embed.py`.
        params: How to group; defaults to umap + hdbscan.

    Returns:
        The Grouping.

    Raises:
        ValueError: On unknown params, or if HDBSCAN clustered nothing at all — an
            all-noise result is not a property list and must not be exported as one.
    """
    params = (params or GroupingParams()).validate()
    vectors = np.asarray(vectors, dtype=np.float32)
    coords = reduce_umap(vectors, params) if params.reduce == "umap" else None
    space = coords if coords is not None else vectors

    if params.cluster == "kmeans":
        centroids, labels, distances = kmeans(space, params.k, params.max_iter,
                                              params.seed)
    else:
        from sklearn.cluster import HDBSCAN

        clusterer = HDBSCAN(min_cluster_size=params.min_cluster_size,
                            min_samples=params.min_samples, metric=params.metric)
        labels = np.asarray(clusterer.fit_predict(space), dtype=np.int64)
        centroids = _centroids_excluding_noise(space, labels)
        if centroids.shape[0] == 0:
            raise ValueError(
                f"hdbscan left all {len(labels)} points unclustered at "
                f"min_cluster_size={params.min_cluster_size}: there is no property list "
                "here. Lower the resolution knob or check the embedding probe.")
        # Noise rows get NaN rather than a distance to some centroid they do not belong
        # to; the mean below reads only the clustered ones.
        distances = np.full(len(labels), np.nan, dtype=np.float32)
        clustered = labels >= 0
        distances[clustered] = np.linalg.norm(
            space[clustered] - centroids[labels[clustered]], axis=1)

    return Grouping(
        labels=labels, centroids=centroids, coords=coords, params=params,
        meta={"n_points": int(len(labels)), "n_groups": int(centroids.shape[0]),
              "n_noise": int((labels < 0).sum()),
              "noise_share": round(float((labels < 0).mean()), 4),
              "mean_distance_to_centroid": round(float(distances[labels >= 0].mean()), 4)
              if (labels >= 0).any() else None,
              "params": params.to_dict()})


def compare(baseline: Grouping, candidate: Grouping,
            vectors: np.ndarray | None = None, neighbours: int = 15) -> dict:
    """Gate one grouping against another on the same points, before trusting a change.

    Changing the clusterer changes the vocabulary every downstream analysis is written in,
    so the honest move is to check the new one against the old before adopting it. Three
    questions, in the order they can disqualify a run:

    1. Did the reduction keep the geometry? (nearest-neighbour overlap, full space vs
       reduced.) Skipped when `vectors` is not supplied or nothing was reduced.
    2. Do the two labelings agree? (ARI/AMI.)
    3. How much did the noise share move?

    Args:
        baseline: The grouping already in use.
        candidate: The proposed replacement, on the SAME points in the same order.
        vectors: The full-dimensional matrix, for question 1.
        neighbours: k for the neighbourhood-overlap check.

    Returns:
        The comparison record.

    Raises:
        ValueError: If the two groupings cover different numbers of points.
    """
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    if len(baseline.labels) != len(candidate.labels):
        raise ValueError(f"groupings cover {len(baseline.labels)} vs "
                         f"{len(candidate.labels)} points; they are not comparable")
    out = {
        "agreement": {
            "ari": round(float(adjusted_rand_score(baseline.labels, candidate.labels)), 4),
            "ami": round(float(adjusted_mutual_info_score(baseline.labels,
                                                          candidate.labels)), 4)},
        "groups": {"baseline": baseline.n_groups, "candidate": candidate.n_groups},
        "noise": {"baseline": baseline.n_noise, "candidate": candidate.n_noise},
        "params": {"baseline": baseline.params.to_dict(),
                   "candidate": candidate.params.to_dict()},
    }
    if vectors is not None and candidate.coords is not None:
        out["geometry"] = {"neighbour_overlap": round(
            _neighbour_overlap(np.asarray(vectors, np.float32), candidate.coords,
                               neighbours), 4), "k": neighbours}
    return out


def _neighbour_overlap(full: np.ndarray, reduced: np.ndarray, k: int) -> float:
    """Share of each point's k nearest neighbours preserved by the reduction.

    Args:
        full: (n x d) the original space.
        reduced: (n x m) the reduced space.
        k: Neighbourhood size.

    Returns:
        Mean overlap in [0, 1]. 1.0 means the reduction moved nobody's neighbourhood.
    """
    from sklearn.neighbors import NearestNeighbors

    def neighbours_of(space: np.ndarray) -> np.ndarray:
        model = NearestNeighbors(n_neighbors=min(k + 1, len(space))).fit(space)
        return model.kneighbors(space, return_distance=False)[:, 1:]

    a, b = neighbours_of(full), neighbours_of(reduced)
    return float(np.mean([len(set(x) & set(y)) / len(x) for x, y in zip(a, b)]))
