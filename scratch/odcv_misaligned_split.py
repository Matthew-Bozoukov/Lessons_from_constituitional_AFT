# ABOUTME: Within each arm, test whether MISALIGNED ODCV rollouts separate from aligned ones
# ABOUTME: in embedding space — DBSCAN + UMAP for the picture, cross-validated AUC for the answer.

"""Misaligned vs non-misaligned reasoning, within one arm at a time.

Run: uv run --no-sync python scratch/odcv_misaligned_split.py \\
         --base_url https://<pod>-8000.proxy.runpod.net/v1

WHY AUC AND NOT JUST DBSCAN. DBSCAN answers "is there cluster structure", which is not the
question. The question is "do these two LABELLED groups occupy different regions", and an
unsupervised clusterer can easily return one blob while a linear probe separates the labels
perfectly. So the headline number here is cross-validated ROC-AUC from a logistic probe,
with a label-permutation null to say what AUC this many points would reach by chance.

SCENARIO IS THE CONFOUND, AGAIN. On raw vectors the scenario dominates everything (purity
1.000 in the earlier run), and misalignment is not evenly spread across scenarios — some
situations tempt every arm. A probe on raw vectors could therefore score well by learning
"this is the hiring scenario" rather than anything about the reasoning. The centered variant
subtracts the per-scenario mean WITHIN the arm, keeping only scenarios that contain both a
misaligned and an aligned rollout, which makes it a within-situation comparison.
"""

import json
from collections import Counter
from pathlib import Path

import fire
import numpy as np


def probe_auc(x: np.ndarray, y: np.ndarray, seed: int, n_perm: int = 200) -> tuple:
    """Cross-validated AUC of a logistic probe, with a permutation null.

    Returns:
        (auc, null_mean, null_p95, p_value)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score

    k = min(5, int(min(np.bincount(y))))
    if k < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    p = cross_val_predict(clf, x, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, p)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        yp = rng.permutation(y)
        pp = cross_val_predict(clf, x, yp, cv=StratifiedKFold(
            n_splits=k, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
        null.append(roc_auc_score(yp, pp))
    null = np.array(null)
    return auc, float(null.mean()), float(np.percentile(null, 95)), float((null >= auc).mean())


def main(base_url: str = "", samples: str = "output/odcv_cluster/samples_all.jsonl",
         embeddings: str = "", model: str = "embed", n_dims: int = 10,
         min_samples: int = 5, batch: int = 8, max_chars: int = 24000, seed: int = 0,
         out_dir: str = "output/odcv_cluster") -> None:
    """Embed (or reuse), then run the within-arm analysis for each arm.

    Args:
        base_url: Embeddings endpoint; omit if `embeddings` is given.
        samples: Manifest JSONL with a `misaligned` field.
        embeddings: Path to a cached .npy aligned to `samples`; skips embedding.
        model: Served model name.
        n_dims: PCA dims for clustering/probing.
        min_samples: DBSCAN min_samples.
        batch: Docs per embedding request.
        max_chars: Per-document truncation.
        seed: RNG seed.
        out_dir: Output directory.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import umap
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scratch.odcv_cluster_run import embed, pick_eps
    from src.utils import timestamp

    d = Path(out_dir); d.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in Path(samples).open(encoding="utf-8")]
    ts = timestamp()
    if embeddings:
        vecs = np.load(embeddings)
        assert len(vecs) == len(rows), f"{len(vecs)} vectors vs {len(rows)} rows"
        print(f">>> reusing {embeddings} {vecs.shape}")
    else:
        assert base_url, "need --base_url or --embeddings"
        print(f">>> embedding {len(rows)} samples")
        vecs = embed([r["reasoning"] for r in rows], base_url, model, batch, max_chars)
        np.save(d / f"embeddings_all_{ts}.npy", vecs)
        print(f">>> saved embeddings {vecs.shape}")

    summary = []
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8))
    for ax, arm in zip(axes, ["difficult_advice", "peer_critique"]):
        ix = [i for i, r in enumerate(rows) if r["arm"] == arm]
        y = np.array([int(rows[i]["misaligned"]) for i in ix])
        v = vecs[ix]
        print(f"\n=== {arm}: n={len(ix)}  misaligned={int(y.sum())}  aligned={int((1-y).sum())} ===")

        x10 = PCA(n_components=n_dims, random_state=seed).fit_transform(v)
        e = pick_eps(x10, min_samples)
        lab = DBSCAN(eps=e, min_samples=min_samples).fit_predict(x10)
        n_cl = len({c for c in lab if c >= 0})
        print(f"  DBSCAN eps={e:.4f}: {n_cl} clusters, {int((lab==-1).sum())} noise")
        if n_cl:
            pure = sum(Counter(y[lab == c]).most_common(1)[0][1]
                       for c in {c for c in lab if c >= 0}) / int((lab >= 0).sum())
            print(f"  cluster purity by misaligned: {pure:.3f} "
                  f"(chance {max(np.bincount(y[lab>=0]))/int((lab>=0).sum()):.3f})")

        auc, nm, n95, pv = probe_auc(x10, y, seed)
        print(f"  PROBE (raw, {n_dims}d): AUC {auc:.3f}   null {nm:.3f} (p95 {n95:.3f})  p={pv:.3f}")

        # within-scenario: keep scenarios holding BOTH classes, subtract their mean
        by_sc = {}
        for j, i in enumerate(ix):
            by_sc.setdefault(rows[i]["scenario"], []).append(j)
        keep = [j for sc, js in by_sc.items()
                if len({y[j] for j in js}) == 2 for j in js]
        if len(keep) >= 20:
            vc = v.copy()
            for sc, js in by_sc.items():
                vc[js] -= vc[js].mean(axis=0)
            vc = vc / np.clip(np.linalg.norm(vc, axis=1, keepdims=True), 1e-9, None)
            xc = PCA(n_components=min(n_dims, len(keep) - 1),
                     random_state=seed).fit_transform(vc[keep])
            aucc, nmc, n95c, pvc = probe_auc(xc, y[keep], seed)
            n_sc = len({sc for sc, js in by_sc.items() if len({y[j] for j in js}) == 2})
            print(f"  PROBE (within-scenario, n={len(keep)} over {n_sc} mixed scenarios): "
                  f"AUC {aucc:.3f}   null {nmc:.3f}  p={pvc:.3f}")
        else:
            aucc = float("nan"); pvc = float("nan"); n_sc = 0
            print(f"  within-scenario probe skipped: only {len(keep)} points in mixed scenarios")

        xy = umap.UMAP(n_components=2, random_state=seed, n_neighbors=15,
                       min_dist=0.1).fit_transform(x10)
        for lbl, c, nm2 in ((0, "#1565C0", "aligned"), (1, "#C62828", "misaligned")):
            m = y == lbl
            ax.scatter(xy[m, 0], xy[m, 1], s=44, c=c, edgecolor="black", linewidth=0.6,
                       alpha=0.85, label=f"{nm2} (n={int(m.sum())})")
        ax.set_title(f"{arm}\nprobe AUC {auc:.2f} (raw), {aucc:.2f} (within-scenario)",
                     fontsize=15)
        ax.set_xlabel("UMAP-1", fontsize=15); ax.set_ylabel("UMAP-2", fontsize=15)
        ax.grid(True, linestyle="--", alpha=0.2); ax.set_axisbelow(True)
        ax.legend(fontsize=13, loc="best")
        summary.append({"arm": arm, "n": len(ix), "n_misaligned": int(y.sum()),
                        "dbscan_clusters": n_cl, "auc_raw": auc, "auc_raw_p": pv,
                        "auc_within_scenario": aucc, "auc_within_scenario_p": pvc,
                        "mixed_scenarios": n_sc})

    fig.suptitle("ODCV reasoning: misaligned vs aligned, within arm (Qwen3-Embedding-8B)",
                 fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    png = d / f"umap_misaligned_split_{ts}.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    (d / f"misaligned_split_{ts}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {png}\nwrote {d}/misaligned_split_{ts}.json")


if __name__ == "__main__":
    fire.Fire(main)
