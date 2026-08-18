# ABOUTME: Gate two clusterings of the same feature embeddings against each other: label
# ABOUTME: agreement, whether UMAP preserved the geometry, and how stable HDBSCAN is.

"""Decide whether a UMAP+HDBSCAN clustering can be trusted before anything is read off it.

Swapping the clusterer changes the vocabulary every downstream analysis is written in, so
it needs a gate rather than a look at the labels. Three questions, in the order they can
disqualify the run:

* **Did the reduction keep the geometry?** UMAP is free to tear a manifold. If the stored
  embedding sanity probe (`Backtracks in reasoning` ~ `Self correction in reasoning`
  beating `Talks about apples`) does not survive the reduction, nothing downstream of it
  means anything. Runs embedded before 2026-08-18 have no `probe_embeddings.npy`, so the
  fallback is nearest-neighbour overlap between the full-dimensional and reduced spaces,
  which measures the same property without needing those three strings.
* **Do the two clusterings agree?** ARI/AMI between the k-means and HDBSCAN labels, over
  the features both of them cluster. Close agreement means the k=150 numbers already
  published were sound and can stand.
* **Is HDBSCAN stable?** Re-fit across `n_neighbors` and seeds and take pairwise ARI. A
  clustering that reshuffles when the seed changes is not a finding.

Cost: CPU only, no GPU and no API spend, but UMAP runs single-threaded when seeded, so the
default stability sweep is 4 x 3 = 12 fits over the whole matrix. Start with --stability
false to get the first two answers in one fit.

Run:
  uv run python scratch/llm_feature_discovery/compare_clusterings_and_check_stability.py \\
      --kmeans-dir output/feature_discovery/<ts> \\
      --hdbscan-dir output/feature_discovery/<ts>_hdbscan
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.cluster import HDBSCAN  # noqa: E402
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score  # noqa: E402

from src.utils import git_sha, timestamp  # noqa: E402

NEIGHBOURS_FOR_OVERLAP = 10
FEATURES_SAMPLED_FOR_OVERLAP = 500
SAMPLED_ROWS_PER_CHUNK = 64


def load_embeddings(run_path: Path) -> tuple[np.ndarray, list[str]]:
    """Read a run's embedding matrix and the feature strings its rows correspond to.

    Args:
        run_path: A feature-discovery run directory.

    Returns:
        (n x d float32 matrix, feature strings in row order).

    Raises:
        RuntimeError: If the matrix and the feature list disagree on length.
    """
    features = [x for x in (run_path / "unique_features.txt").read_text().splitlines()
                if x.strip()]
    embeddings = np.asarray(np.load(run_path / "embeddings.npy"), dtype=np.float32)
    if embeddings.shape[0] != len(features):
        raise RuntimeError(f"{run_path}: embeddings {embeddings.shape} vs "
                           f"{len(features)} features")
    return embeddings, features


def measure_neighbour_overlap(embeddings: np.ndarray, reduced: np.ndarray, seed: int,
                              n_neighbors: int = NEIGHBOURS_FOR_OVERLAP,
                              n_sampled: int = FEATURES_SAMPLED_FOR_OVERLAP) -> float:
    """Fraction of each sampled feature's nearest neighbours that survive the reduction.

    The full n x n similarity matrix is far too large to hold, so this samples rows and
    scores each against the whole corpus in chunks. Cosine in the original space and
    Euclidean in the reduced one, because that is the metric HDBSCAN then uses.

    Args:
        embeddings: (n x d) L2-normalised feature vectors.
        reduced: (n x m) UMAP coordinates for the same rows.
        seed: Seed for the row sample.
        n_neighbors: Neighbourhood size to compare.
        n_sampled: How many rows to score.

    Returns:
        Mean overlap in [0, 1]; 1.0 means every neighbourhood survived intact.
    """
    rng = np.random.default_rng(seed)
    n_rows = embeddings.shape[0]
    sampled_rows = rng.choice(n_rows, size=min(n_sampled, n_rows), replace=False)
    overlaps = []
    for start in range(0, len(sampled_rows), SAMPLED_ROWS_PER_CHUNK):
        rows = sampled_rows[start:start + SAMPLED_ROWS_PER_CHUNK]
        # Higher cosine = nearer; negate so both spaces sort ascending on "far".
        full_distance = -(embeddings[rows] @ embeddings.T)
        reduced_distance = np.linalg.norm(
            reduced[rows][:, None, :] - reduced[None, :, :], axis=2)
        for offset, row in enumerate(rows):
            full_distance[offset, row] = np.inf       # exclude self
            reduced_distance[offset, row] = np.inf
            full_nearest = set(np.argpartition(full_distance[offset],
                                               n_neighbors)[:n_neighbors].tolist())
            reduced_nearest = set(np.argpartition(reduced_distance[offset],
                                                  n_neighbors)[:n_neighbors].tolist())
            overlaps.append(len(full_nearest & reduced_nearest) / n_neighbors)
    return float(np.mean(overlaps))


def check_probe_geometry_survives(run_path: Path, reducer, embedding_meta: dict) -> dict:
    """Push the stored embedding sanity probes through the fitted reducer.

    Args:
        run_path: The run directory holding probe_embeddings.npy, if it has one.
        reducer: A fitted UMAP object.
        embedding_meta: Parsed embed_meta.json, for the full-dimensional baseline.

    Returns:
        A dict of before/after cosines, or {"available": False} for a run embedded before
        the probe vectors were kept.
    """
    probe_path = run_path / "probe_embeddings.npy"
    if not probe_path.exists():
        return {"available": False,
                "note": "run predates probe_embeddings.npy; rely on neighbour overlap"}
    probes = np.asarray(np.load(probe_path), dtype=np.float32)
    reduced = np.asarray(reducer.transform(probes), dtype=np.float32)
    reduced /= np.linalg.norm(reduced, axis=1, keepdims=True)
    return {"available": True,
            "probe": embedding_meta.get("probe"),
            "full_dim_synonym": embedding_meta.get("sanity_synonym"),
            "full_dim_unrelated": embedding_meta.get("sanity_unrelated"),
            "reduced_synonym": float(reduced[0] @ reduced[1]),
            "reduced_unrelated": float(reduced[0] @ reduced[2]),
            # Only the ORDERING carries over. UMAP compresses the space, so the reduced
            # cosines run higher than the full-dimensional ones across the board and the
            # absolute numbers are not comparable between the two columns.
            "gap_survives": bool(reduced[0] @ reduced[1] > reduced[0] @ reduced[2])}


def labels_for_shared_features(kmeans_map: dict[str, int], hdbscan_map: dict[str, int],
                               features: list[str]) -> tuple[np.ndarray, np.ndarray, int]:
    """Line up two cluster maps on the features both of them actually cluster.

    Args:
        kmeans_map: Feature -> cluster id from the k-means run.
        hdbscan_map: Feature -> cluster id from the HDBSCAN run (noise omitted).
        features: The shared feature list, in embedding-row order.

    Returns:
        (k-means labels, HDBSCAN labels, count of features dropped as noise).
    """
    shared = [f for f in features if f in kmeans_map and f in hdbscan_map]
    return (np.array([kmeans_map[f] for f in shared], dtype=np.int32),
            np.array([hdbscan_map[f] for f in shared], dtype=np.int32),
            len(features) - len(shared))


def main(kmeans_dir: str, hdbscan_dir: str, seed: int = 0, stability: bool = True,
         stability_neighbors: tuple[int, ...] = (10, 15, 30, 50),
         stability_seeds: tuple[int, ...] = (0, 1, 2), out_dir: str | None = None) -> None:
    """Compare two clusterings of one embedding set and report whether to trust the new one.

    Args:
        kmeans_dir: The k-means run directory (the published clustering).
        hdbscan_dir: The UMAP+HDBSCAN run directory, whose meta supplies the UMAP params.
        seed: Seed for the reference UMAP fit and the neighbour sample.
        stability: Run the re-fit sweep. False gives geometry + agreement from one fit.
        stability_neighbors: n_neighbors values to sweep.
        stability_seeds: Seeds to sweep.
        out_dir: Where to write the report; defaults to hdbscan_dir.

    Raises:
        RuntimeError: If the two runs were not clustered over the same feature list, or if
            hdbscan_dir was not produced by the hdbscan path.
    """
    import umap    # see cluster_and_name_feature_embeddings: numba import is not free

    kmeans_path, hdbscan_path = Path(kmeans_dir), Path(hdbscan_dir)
    embeddings, features = load_embeddings(hdbscan_path)
    kmeans_features = [x for x in (kmeans_path / "unique_features.txt").read_text().splitlines()
                       if x.strip()]
    if kmeans_features != features:
        raise RuntimeError("the two runs cluster different feature lists; ARI between them "
                           "would be meaningless")

    hdbscan_meta = json.loads((hdbscan_path / "clusters.json").read_text())["meta"]
    if hdbscan_meta.get("cluster") != "hdbscan":
        raise RuntimeError(f"{hdbscan_dir} was clustered with "
                           f"{hdbscan_meta.get('cluster', 'kmeans')!r}, not hdbscan")
    params = hdbscan_meta["cluster_params"]
    embedding_meta = json.loads((hdbscan_path / "embed_meta.json").read_text())

    print(f"{len(features)} features, {embeddings.shape[1]}d -> "
          f"{params['n_components']}d (reference fit, n_neighbors={params['n_neighbors']})")
    reducer = umap.UMAP(n_components=params["n_components"],
                        n_neighbors=params["n_neighbors"], min_dist=params["min_dist"],
                        metric="cosine", random_state=seed).fit(embeddings)
    reference_coords = np.asarray(reducer.embedding_, dtype=np.float32)

    geometry = {
        "neighbour_overlap_at_10": measure_neighbour_overlap(embeddings, reference_coords, seed),
        "probe": check_probe_geometry_survives(hdbscan_path, reducer, embedding_meta),
    }
    print(f"neighbour overlap @{NEIGHBOURS_FOR_OVERLAP}: "
          f"{geometry['neighbour_overlap_at_10']:.3f}")
    if geometry["probe"]["available"]:
        print(f"probe synonym/unrelated: full-dim "
              f"{geometry['probe']['full_dim_synonym']:.3f}/"
              f"{geometry['probe']['full_dim_unrelated']:.3f} -> reduced "
              f"{geometry['probe']['reduced_synonym']:.3f}/"
              f"{geometry['probe']['reduced_unrelated']:.3f} "
              f"({'gap survives' if geometry['probe']['gap_survives'] else 'GAP LOST'})")
    else:
        print(f"probe check skipped: {geometry['probe']['note']}")

    kmeans_map = json.loads((kmeans_path / "feature_cluster_map.json").read_text())
    hdbscan_map = json.loads((hdbscan_path / "feature_cluster_map.json").read_text())
    kmeans_labels, hdbscan_labels, dropped = labels_for_shared_features(
        kmeans_map, hdbscan_map, features)
    agreement = {"ari": float(adjusted_rand_score(kmeans_labels, hdbscan_labels)),
                 "ami": float(adjusted_mutual_info_score(kmeans_labels, hdbscan_labels)),
                 "features_compared": int(len(kmeans_labels)),
                 "features_dropped_as_noise": int(dropped),
                 "kmeans_clusters": int(len(set(kmeans_labels.tolist()))),
                 "hdbscan_clusters": int(len(set(hdbscan_labels.tolist())))}
    print(f"agreement over {agreement['features_compared']} shared features "
          f"({dropped} dropped as noise): ARI {agreement['ari']:.3f}, "
          f"AMI {agreement['ami']:.3f}")

    # Stability: every re-fit is scored against the reference labelling, and against every
    # other re-fit, so a single unlucky seed cannot pass as agreement.
    sweep = []
    if stability:
        reference_labels = HDBSCAN(min_cluster_size=params["min_cluster_size"],
                                   cluster_selection_method="eom").fit_predict(
                                       reference_coords)
        for n_neighbors in stability_neighbors:
            for sweep_seed in stability_seeds:
                coords = umap.UMAP(n_components=params["n_components"],
                                   n_neighbors=n_neighbors, min_dist=params["min_dist"],
                                   metric="cosine",
                                   random_state=sweep_seed).fit_transform(embeddings)
                labels = HDBSCAN(min_cluster_size=params["min_cluster_size"],
                                 cluster_selection_method="eom").fit_predict(coords)
                n_noise = int((labels == -1).sum())
                entry = {"n_neighbors": n_neighbors, "seed": sweep_seed,
                         "n_clusters": int(labels.max()) + 1,
                         "noise_fraction": n_noise / len(labels),
                         "ari_vs_reference": float(adjusted_rand_score(reference_labels,
                                                                       labels)),
                         "labels": labels.tolist()}
                sweep.append(entry)
                print(f"  n_neighbors={n_neighbors} seed={sweep_seed}: "
                      f"{entry['n_clusters']} clusters, "
                      f"{entry['noise_fraction']:.1%} noise, "
                      f"ARI vs reference {entry['ari_vs_reference']:.3f}")
        pairwise = [[float(adjusted_rand_score(a["labels"], b["labels"])) for b in sweep]
                    for a in sweep]
        for entry in sweep:
            del entry["labels"]      # the report keeps the scores, not 33k labels per fit
    else:
        pairwise = []

    report = {"meta": {"kmeans_dir": str(kmeans_path), "hdbscan_dir": str(hdbscan_path),
                       "seed": seed, "umap_params": params,
                       "embedding_model": embedding_meta["model"],
                       "git_sha": git_sha(), "timestamp_utc": timestamp()},
              "geometry": geometry, "agreement": agreement,
              "stability": {"sweep": sweep, "pairwise_ari": pairwise,
                            "min_pairwise_ari": float(np.min(pairwise)) if pairwise else None}}
    out = Path(out_dir or hdbscan_path)
    out.mkdir(parents=True, exist_ok=True)
    (out / "clustering_comparison.json").write_text(json.dumps(report, indent=1))

    lines = [f"# Clustering gate — {hdbscan_path.name} vs {kmeans_path.name}", "",
             f"{len(features)} features, `{embedding_meta['model']}`, "
             f"UMAP {params}.", "",
             "## Did the reduction keep the geometry?", "",
             f"Nearest-neighbour overlap @{NEIGHBOURS_FOR_OVERLAP} between the "
             f"{embeddings.shape[1]}d and {params['n_components']}d spaces: "
             f"**{geometry['neighbour_overlap_at_10']:.3f}**.", ""]
    if geometry["probe"]["available"]:
        lines += [f"Embedding sanity probe: synonym pair "
                  f"{geometry['probe']['full_dim_synonym']:.3f} -> "
                  f"{geometry['probe']['reduced_synonym']:.3f}, unrelated pair "
                  f"{geometry['probe']['full_dim_unrelated']:.3f} -> "
                  f"{geometry['probe']['reduced_unrelated']:.3f}. "
                  f"{'Gap survives.' if geometry['probe']['gap_survives'] else '**GAP LOST.**'}",
                  ""]
    else:
        lines += [f"Probe check unavailable: {geometry['probe']['note']}.", ""]
    lines += ["## Do the two clusterings agree?", "",
              f"| metric | value |", "|---|---:|",
              f"| ARI | {agreement['ari']:.3f} |", f"| AMI | {agreement['ami']:.3f} |",
              f"| features compared | {agreement['features_compared']} |",
              f"| dropped as noise | {agreement['features_dropped_as_noise']} |",
              f"| k-means clusters | {agreement['kmeans_clusters']} |",
              f"| HDBSCAN clusters | {agreement['hdbscan_clusters']} |", ""]
    if sweep:
        lines += ["## Is HDBSCAN stable?", "",
                  "| n_neighbors | seed | clusters | noise | ARI vs reference |",
                  "|---:|---:|---:|---:|---:|"]
        lines += [f"| {e['n_neighbors']} | {e['seed']} | {e['n_clusters']} | "
                  f"{e['noise_fraction']:.1%} | {e['ari_vs_reference']:.3f} |"
                  for e in sweep]
        lines += ["", f"Lowest pairwise ARI across the sweep: "
                      f"**{report['stability']['min_pairwise_ari']:.3f}**.", ""]
    (out / "clustering_comparison.md").write_text("\n".join(lines))
    print(f"\nwrote {out}/clustering_comparison.md, clustering_comparison.json")


if __name__ == "__main__":
    fire.Fire(main)
