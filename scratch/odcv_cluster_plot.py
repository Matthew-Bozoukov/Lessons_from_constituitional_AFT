# ABOUTME: Re-reduce/cluster/project saved ODCV embeddings, optionally centering per scenario,
# ABOUTME: and plot the UMAP with every point still traceable to its transcript.

"""Cluster and plot from cached embeddings — no endpoint, no re-embedding.

Run: uv run --no-sync python scratch/odcv_cluster_plot.py --center_by scenario

WHY CENTERING MATTERS HERE. On the raw vectors DBSCAN recovered the SCENARIOS exactly
(purity 1.000, 30 clusters = 30 scenarios) while arm purity sat at chance. That is the
embedding doing its job: two rollouts of the same scenario share the whole situation, and
that dominates any difference in how the model reasoned about it. Subtracting the
per-scenario mean removes that shared axis, so what remains is within-scenario variation —
the only place an arm difference could show up. Both views are worth having: the raw one
shows the embedding is sane, the centered one is the actual experiment.
"""

import json
from collections import Counter
from pathlib import Path

import fire
import numpy as np


def load(stamp: str, d: Path) -> tuple[np.ndarray, list]:
    """Load the saved embeddings and the point table for a run stamp."""
    vec = np.load(d / f"embeddings_{stamp}.npy")
    pts = json.loads((d / f"points_{stamp}.json").read_text())
    assert len(vec) == len(pts), f"{len(vec)} vectors vs {len(pts)} points"
    return vec, pts


def purity(points: list, key: str) -> float:
    """Fraction of clustered points in their cluster's majority `key` value."""
    cl = [p for p in points if p["cluster"] >= 0]
    if not cl:
        return 0.0
    tot = 0
    for c in {p["cluster"] for p in cl}:
        g = [p for p in cl if p["cluster"] == c]
        tot += Counter(p[key] for p in g).most_common(1)[0][1]
    return tot / len(cl)


def main(stamp: str = "", center_by: str = "", n_dims: int = 10, min_samples: int = 5,
         eps: float = 0.0, seed: int = 0, out_dir: str = "output/odcv_cluster") -> None:
    """Reduce, cluster, project and plot.

    Args:
        stamp: Run stamp of the saved arrays; newest if empty.
        center_by: Point field whose group mean is subtracted before reducing
            ("scenario" removes the situation axis). Empty = raw vectors.
        n_dims: PCA dims for clustering.
        min_samples: DBSCAN min_samples.
        eps: DBSCAN eps; 0 picks the k-distance knee.
        seed: PCA/UMAP seed.
        out_dir: Where the arrays live and the plot is written.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scratch.odcv_cluster_run import pick_eps
    from src.utils import timestamp

    d = Path(out_dir)
    stamp = stamp or sorted(p.stem.split("embeddings_")[-1]
                            for p in d.glob("embeddings_*.npy"))[-1]
    vec, pts = load(stamp, d)
    print(f">>> {len(vec)} vectors dim {vec.shape[1]} from stamp {stamp}")

    if center_by:
        groups = {}
        for i, p in enumerate(pts):
            groups.setdefault(p[center_by], []).append(i)
        vec = vec.copy()
        for _, ix in groups.items():
            vec[ix] -= vec[ix].mean(axis=0)
        norms = np.linalg.norm(vec, axis=1, keepdims=True)
        vec = vec / np.clip(norms, 1e-9, None)
        print(f">>> centered within {len(groups)} '{center_by}' groups, re-normalised")

    x10 = PCA(n_components=n_dims, random_state=seed).fit_transform(vec)
    e = eps or pick_eps(x10, min_samples)
    labels = DBSCAN(eps=e, min_samples=min_samples).fit_predict(x10)
    xy = umap.UMAP(n_components=2, random_state=seed, n_neighbors=15,
                   min_dist=0.1).fit_transform(x10)
    for i, p in enumerate(pts):
        p["cluster"] = int(labels[i])
        p["x"], p["y"] = float(xy[i, 0]), float(xy[i, 1])
    n_cl = len({c for c in labels if c >= 0})
    noise = int((labels == -1).sum())
    print(f">>> DBSCAN eps={e:.4f}: {n_cl} clusters, {noise} noise ({100*noise/len(labels):.1f}%)")
    for k in ("scenario", "arm", "variant"):
        base = max(Counter(p[k] for p in pts if p["cluster"] >= 0).values()) / max(
            1, sum(1 for p in pts if p["cluster"] >= 0))
        print(f"    purity by {k:9s} {purity(pts, k):.3f}   (chance {base:.3f})")

    tag = f"centered_{center_by}" if center_by else "raw"
    ts = timestamp()
    plt.rcParams.update({"font.size": 15})
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8))
    colours = {"difficult_advice": "#1565C0", "peer_critique": "#C62828"}
    for arm, c in colours.items():
        m = [i for i, p in enumerate(pts) if p["arm"] == arm]
        axes[0].scatter(xy[m, 0], xy[m, 1], s=42, c=c, edgecolor="black", linewidth=0.6,
                        alpha=0.85, label=f"{arm} (n={len(m)})")
    axes[0].set_title("coloured by ARM", fontsize=16)
    axes[0].legend(fontsize=13, loc="best")

    lab = np.array([p["cluster"] for p in pts])
    noise_m = lab < 0
    axes[1].scatter(xy[noise_m, 0], xy[noise_m, 1], s=34, c="#BDBDBD",
                    edgecolor="black", linewidth=0.5, alpha=0.7,
                    label=f"noise (n={int(noise_m.sum())})")
    cmap = plt.get_cmap("tab20")
    for j, c in enumerate(sorted({int(x) for x in lab if x >= 0})):
        m = lab == c
        axes[1].scatter(xy[m, 0], xy[m, 1], s=42, color=cmap(j % 20),
                        edgecolor="black", linewidth=0.6, alpha=0.9)
    axes[1].set_title(f"coloured by DBSCAN cluster ({n_cl} clusters)", fontsize=16)
    axes[1].legend(fontsize=12, loc="best")
    for ax in axes:
        ax.set_xlabel("UMAP-1", fontsize=15)
        ax.set_ylabel("UMAP-2", fontsize=15)
        ax.grid(True, linestyle="--", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle(f"ODCV non-misaligned reasoning — Qwen3-Embedding-8B, {tag}",
                 fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = d / f"umap_{tag}_{ts}.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")

    (d / f"points_{tag}_{ts}.json").write_text(json.dumps(pts, indent=2))
    with (d / f"points_{tag}_{ts}.csv").open("w", encoding="utf-8") as fh:
        cols = list(pts[0])
        fh.write(",".join(cols) + "\n")
        for p in pts:
            fh.write(",".join('"' + str(p[c]).replace('"', "'") + '"' for c in cols) + "\n")
    print(f"wrote {png}\nwrote {d}/points_{tag}_{ts}.json (+ .csv)")


if __name__ == "__main__":
    fire.Fire(main)
