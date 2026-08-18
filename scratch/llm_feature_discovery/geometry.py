# ABOUTME: Did the UMAP reduction keep the structure the clustering depends on? Neighbour
# ABOUTME: overlap between the two spaces, plus the stored embedding sanity probe.

"""Whether the reduction can be trusted.

UMAP is free to tear a manifold. If it does, HDBSCAN is clustering an artefact and every
label downstream is meaningless, so this is the first question the gate asks and the one
that can disqualify a run on its own.

Two measurements, because the cheap one is not always available:

* **Neighbour overlap** — for a sample of features, how much of each one's nearest
  neighbourhood survives the reduction. Works on any run.
* **The sanity probe** — the three strings the embedding stage scored on the pod, pushed
  through the fitted reducer. Only the ORDERING carries over: UMAP compresses the space, so
  reduced cosines run higher across the board and the absolute numbers are not comparable
  between the two spaces. Runs embedded before 2026-08-18 did not keep the probe vectors.
"""

from __future__ import annotations

import numpy as np

NEIGHBOURS_COMPARED = 10
FEATURES_SAMPLED = 500
SAMPLED_ROWS_PER_CHUNK = 64


def neighbour_overlap(embeddings: np.ndarray, reduced: np.ndarray, seed: int,
                      n_neighbors: int = NEIGHBOURS_COMPARED,
                      n_sampled: int = FEATURES_SAMPLED) -> float:
    """Fraction of each sampled feature's nearest neighbours that survive the reduction.

    The full n x n similarity matrix is far too large to hold, so this samples rows and
    scores each against the whole corpus in chunks. Cosine in the original space and
    Euclidean in the reduced one, because Euclidean is what HDBSCAN then uses.

    Args:
        embeddings: (n x d) L2-normalised feature vectors.
        reduced: (n x m) coordinates for the same rows.
        seed: Seed for the row sample.
        n_neighbors: Neighbourhood size to compare.
        n_sampled: How many rows to score.

    Returns:
        Mean overlap in [0, 1]; 1.0 means every neighbourhood survived intact.
    """
    rng = np.random.default_rng(seed)
    sampled_rows = rng.choice(embeddings.shape[0],
                              size=min(n_sampled, embeddings.shape[0]), replace=False)
    overlaps = []
    for start in range(0, len(sampled_rows), SAMPLED_ROWS_PER_CHUNK):
        rows = sampled_rows[start:start + SAMPLED_ROWS_PER_CHUNK]
        # Higher cosine = nearer; negate so both spaces sort ascending on "far".
        full_distance = -(embeddings[rows] @ embeddings.T)
        reduced_distance = np.linalg.norm(reduced[rows][:, None, :] - reduced[None, :, :],
                                          axis=2)
        for offset, row in enumerate(rows):
            full_distance[offset, row] = np.inf       # exclude self
            reduced_distance[offset, row] = np.inf
            nearest_full = set(np.argpartition(full_distance[offset],
                                               n_neighbors)[:n_neighbors].tolist())
            nearest_reduced = set(np.argpartition(reduced_distance[offset],
                                                  n_neighbors)[:n_neighbors].tolist())
            overlaps.append(len(nearest_full & nearest_reduced) / n_neighbors)
    return float(np.mean(overlaps))


def probe_survives_reduction(probe_embeddings: np.ndarray | None, reducer,
                             embed_meta: dict) -> dict:
    """Push the stored embedding sanity probes through the fitted reducer.

    Args:
        probe_embeddings: (3 x d) probe vectors, or None for a run that did not keep them.
        reducer: A fitted UMAP object.
        embed_meta: Parsed embed_meta.json, for the full-dimensional baseline.

    Returns:
        Before/after cosines, or {"available": False} with a note.
    """
    if probe_embeddings is None:
        return {"available": False,
                "note": "run predates probe_embeddings.npy; rely on neighbour overlap"}
    reduced = np.asarray(reducer.transform(probe_embeddings), dtype=np.float32)
    reduced /= np.linalg.norm(reduced, axis=1, keepdims=True)
    synonym, unrelated = float(reduced[0] @ reduced[1]), float(reduced[0] @ reduced[2])
    return {"available": True,
            "probe": embed_meta.get("probe"),
            "full_dim_synonym": embed_meta.get("sanity_synonym"),
            "full_dim_unrelated": embed_meta.get("sanity_unrelated"),
            "reduced_synonym": synonym,
            "reduced_unrelated": unrelated,
            "gap_survives": bool(synonym > unrelated)}
