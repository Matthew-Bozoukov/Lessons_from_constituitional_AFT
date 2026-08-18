# ABOUTME: Compute and cache a 2D UMAP of a TURF index's trigger embeddings
# ABOUTME: (PCA-64 -> UMAP cosine, seeded). Run: uv run --with umap-learn python scratch/turf/umap_coords.py --dir <index>

from __future__ import annotations

import sys
import time
from pathlib import Path

import fire
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(dir: str = "output/turf/da2203", pca_dim: int = 64, seed: int = 42) -> None:
    """PCA then UMAP the trigger embeddings; cache <dir>/umap_trigger_2d.npy."""
    d = Path(dir)
    out_path = d / "umap_trigger_2d.npy"
    if out_path.exists():
        print(f">>> {out_path} already exists — delete to recompute")
        return
    emb = np.load(d / "embeddings_trigger.npy").astype(np.float32)
    print(f">>> {emb.shape} trigger embeddings", flush=True)

    t0 = time.time()
    from sklearn.decomposition import PCA

    x = PCA(n_components=pca_dim, random_state=seed).fit_transform(emb)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    print(f">>> PCA-{pca_dim} in {time.time()-t0:.0f}s", flush=True)

    t0 = time.time()
    import umap

    coords = umap.UMAP(n_components=2, metric="cosine", n_neighbors=15,
                       min_dist=0.1, random_state=seed).fit_transform(x)
    np.save(out_path, coords.astype(np.float32))
    print(f">>> UMAP in {time.time()-t0:.0f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    fire.Fire(main)
