# ABOUTME: Embed ODCV reasoning via a vLLM /v1/embeddings endpoint, reduce to 10 dims,
# ABOUTME: DBSCAN, then UMAP to 2D — every stage indexed against the sample manifest.

"""Cluster the non-misaligned ODCV reasoning from two arms.

Run: uv run --no-sync python scratch/odcv_cluster_run.py \\
         --base_url https://<pod>-8000.proxy.runpod.net/v1

PROVENANCE. Rows keep the order of samples.jsonl end to end, and the written table repeats
arm / variant / scenario / rollout / judge score / path beside each point's cluster label
and xy. Any dot on the plot resolves to one transcript with `--inspect <idx>`.

WHY PCA FOR THE 10-DIM STEP, NOT UMAP. The request is reduce-to-10 then cluster then
project-to-2. Doing the 10-dim step with UMAP too would mean clustering on a manifold
embedding whose distances are not metric in the way DBSCAN's epsilon assumes; PCA keeps
Euclidean distance meaningful, which is what DBSCAN needs. UMAP is used only for the final
2D view, which is for looking at, not for computing on.

EPSILON IS NOT GUESSED. DBSCAN's result is entirely determined by eps, so it is chosen from
the k-distance curve (the knee of sorted min_samples-th nearest-neighbour distances) and the
chosen value plus the resulting noise fraction are printed, so a degenerate clustering
(everything one cluster, or everything noise) is visible rather than silent.
"""

import json
from pathlib import Path

import fire
import numpy as np
import requests


def embed(texts: list[str], base_url: str, model: str, batch: int, max_chars: int) -> np.ndarray:
    """Embed texts through an OpenAI-compatible /v1/embeddings endpoint.

    Args:
        texts: Documents.
        base_url: vLLM base URL.
        model: Served model name.
        batch: Documents per request.
        max_chars: Truncate each document to this many characters. The longest sample here
            is ~38k chars (~9.5k tokens); the cap is a guard against a single outlier
            exceeding max_model_len and failing the whole batch.

    Returns:
        (n, d) float32 array, L2-normalised.
    """
    out = []
    for i in range(0, len(texts), batch):
        chunk = [t[:max_chars] for t in texts[i:i + batch]]
        r = requests.post(f"{base_url}/embeddings", timeout=900,
                          json={"model": model, "input": chunk})
        r.raise_for_status()
        d = r.json()["data"]
        assert len(d) == len(chunk), f"got {len(d)} embeddings for {len(chunk)} inputs"
        out.extend(e["embedding"] for e in sorted(d, key=lambda x: x["index"]))
        print(f"    embedded {min(i+batch, len(texts))}/{len(texts)}")
    a = np.asarray(out, dtype=np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def pick_eps(x: np.ndarray, min_samples: int) -> float:
    """Choose DBSCAN eps at the knee of the k-distance curve."""
    from sklearn.neighbors import NearestNeighbors
    d, _ = NearestNeighbors(n_neighbors=min_samples).fit(x).kneighbors(x)
    kd = np.sort(d[:, -1])
    # knee = point of maximum distance from the chord joining the curve's endpoints
    n = len(kd)
    p0, p1 = np.array([0, kd[0]]), np.array([n - 1, kd[-1]])
    v = p1 - p0
    v = v / np.linalg.norm(v)
    pts = np.stack([np.arange(n), kd], axis=1) - p0
    dist = np.linalg.norm(pts - np.outer(pts @ v, v), axis=1)
    return float(kd[int(dist.argmax())])


def main(base_url: str, samples: str = "output/odcv_cluster/samples.jsonl",
         model: str = "embed", field: str = "reasoning", n_dims: int = 10,
         min_samples: int = 5, eps: float = 0.0, batch: int = 8,
         max_chars: int = 24000, seed: int = 0,
         out_dir: str = "output/odcv_cluster") -> None:
    """Embed, reduce, cluster, project, and write a table joined to the manifest.

    Args:
        base_url: vLLM embeddings base URL.
        samples: Manifest JSONL from odcv_cluster_extract.py.
        model: Served model name at the endpoint.
        field: Which field of each sample to embed.
        n_dims: PCA target dimensionality for clustering.
        min_samples: DBSCAN min_samples.
        eps: DBSCAN eps; 0 means choose from the k-distance knee.
        batch: Documents per embedding request.
        max_chars: Per-document truncation.
        seed: UMAP/PCA seed.
        out_dir: Output directory.
    """
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA
    import umap

    from src.utils import timestamp

    rows = [json.loads(l) for l in Path(samples).open(encoding="utf-8")]
    print(f">>> {len(rows)} samples from {samples}, embedding field '{field}'")
    vecs = embed([r[field] for r in rows], base_url, model, batch, max_chars)
    print(f">>> embeddings {vecs.shape}")

    pca = PCA(n_components=n_dims, random_state=seed)
    x10 = pca.fit_transform(vecs)
    print(f">>> PCA -> {x10.shape}, explained variance {pca.explained_variance_ratio_.sum():.3f}")

    e = eps or pick_eps(x10, min_samples)
    labels = DBSCAN(eps=e, min_samples=min_samples).fit_predict(x10)
    uniq = sorted(set(labels))
    noise = int((labels == -1).sum())
    print(f">>> DBSCAN eps={e:.4f} min_samples={min_samples}: "
          f"{len([u for u in uniq if u >= 0])} clusters, {noise} noise "
          f"({100*noise/len(labels):.1f}%)")

    xy = umap.UMAP(n_components=2, random_state=seed, n_neighbors=15,
                   min_dist=0.1).fit_transform(x10)
    print(f">>> UMAP -> {xy.shape}")

    ts = timestamp()
    d = Path(out_dir); d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"embeddings_{ts}.npy", vecs)
    np.save(d / f"pca10_{ts}.npy", x10)
    table = []
    for i, r in enumerate(rows):
        table.append({k: r[k] for k in ("idx", "arm", "variant", "scenario", "rollout",
                                        "judge_median", "n_turns", "reason_chars",
                                        "n_tool_calls", "path")}
                     | {"cluster": int(labels[i]), "x": float(xy[i, 0]), "y": float(xy[i, 1])})
    (d / f"points_{ts}.json").write_text(json.dumps(table, indent=2))
    with (d / f"points_{ts}.csv").open("w", encoding="utf-8") as fh:
        cols = list(table[0])
        fh.write(",".join(cols) + "\n")
        for t in table:
            fh.write(",".join('"' + str(t[c]).replace('"', "'") + '"' for c in cols) + "\n")

    from collections import Counter
    print("\ncluster x arm:")
    cc = Counter((t["cluster"], t["arm"]) for t in table)
    for c in sorted({t["cluster"] for t in table}):
        pc = cc[(c, "peer_critique")]; da = cc[(c, "difficult_advice")]
        tag = "NOISE" if c == -1 else f"cluster {c}"
        print(f"  {tag:11s} n={pc+da:4d}   peer_critique {pc:4d}   difficult_advice {da:4d}")
    print(f"\nwrote {d}/points_{ts}.json (+ .csv), embeddings_{ts}.npy, pca10_{ts}.npy")


if __name__ == "__main__":
    fire.Fire(main)
