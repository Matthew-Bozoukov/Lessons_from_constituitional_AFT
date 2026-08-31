# ABOUTME: Stage 2 of the response-diversity audit: clusters the labelled features,
# ABOUTME: measures template repetition in the raw replies, and writes report + dashboard.

"""Stage 2 of the response-diversity audit.

Consumes `features.jsonl` from extract.py plus the original SFT file, and answers:
how many distinct *ways of responding* does this corpus actually contain, do the
clusters cut across trait/domain (or merely recover them), and which surface templates
recur. Everything here is local — no API calls.

Run:
  uv run python scratch/response_diversity/analyze.py \
      --features output/response_diversity/<ts>/features.jsonl \
      --sft output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl
"""

from __future__ import annotations

import base64
import html
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path

import fire
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sklearn.cluster import KMeans  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score  # noqa: E402

from scratch.response_diversity.schema import AXES  # noqa: E402
from src.utils import git_sha, timestamp  # noqa: E402

plt.rcParams.update({"font.size": 14, "axes.labelsize": 15, "axes.titlesize": 16,
                     "xtick.labelsize": 13, "ytick.labelsize": 13, "figure.dpi": 130})

NEAR_DUP_THRESHOLD = 0.70


def entropy(counts: list[int]) -> float:
    """Shannon entropy in nats of a count vector.

    Args:
        counts: Non-negative counts.

    Returns:
        Entropy in nats; 0.0 for an empty or single-valued distribution.
    """
    arr = np.asarray([c for c in counts if c > 0], dtype=float)
    if arr.size == 0:
        return 0.0
    p = arr / arr.sum()
    return float(-(p * np.log(p)).sum())


def axis_stats(rows: list[dict]) -> dict:
    """Per-axis value distribution, normalised entropy and effective value count.

    Args:
        rows: Labelled feature rows.

    Returns:
        Mapping axis -> {counts, top_share, norm_entropy, effective_values}.
    """
    out = {}
    for axis, vocab in AXES.items():
        counts = Counter(r[axis] for r in rows)
        vec = [counts.get(v, 0) for v in vocab]
        h = entropy(vec)
        out[axis] = {
            "counts": dict(counts.most_common()),
            "top_value": counts.most_common(1)[0][0],
            "top_share": counts.most_common(1)[0][1] / len(rows),
            "norm_entropy": h / np.log(len(vocab)),
            "effective_values": float(np.exp(h)),
            "vocab_size": len(vocab),
        }
    return out


def profile_stats(rows: list[dict]) -> dict:
    """Treat the full axis tuple as a 'response mode' and measure its concentration.

    Args:
        rows: Labelled feature rows.

    Returns:
        Distinct/effective profile counts and the most common profiles.
    """
    profs = Counter(tuple(r[a] for a in AXES) for r in rows)
    h = entropy(list(profs.values()))
    top = [{"profile": dict(zip(AXES, p)), "count": c, "share": c / len(rows)}
           for p, c in profs.most_common(15)]
    return {
        "distinct_profiles": len(profs),
        "possible_profiles": int(np.prod([len(v) for v in AXES.values()])),
        "effective_profiles": float(np.exp(h)),
        "top10_share": sum(c for _, c in profs.most_common(10)) / len(rows),
        "singleton_profiles": sum(1 for c in profs.values() if c == 1),
        "top_profiles": top,
    }


def cluster(rows: list[dict], k_range: range, seed: int = 0) -> dict:
    """K-means over one-hot axis features, with k chosen by silhouette score.

    Args:
        rows: Labelled feature rows.
        k_range: Candidate cluster counts.
        seed: Random seed.

    Returns:
        Chosen k, per-k silhouette scores, labels, and per-cluster descriptions.
    """
    cols = [(a, v) for a, vocab in AXES.items() for v in vocab]
    x = np.array([[1.0 if r[a] == v else 0.0 for a, v in cols] for r in rows])
    assert x.shape == (len(rows), len(cols)), f"bad feature matrix {x.shape}"

    sils = {}
    for k in k_range:
        lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(x)
        sils[k] = float(silhouette_score(x, lab))
    best_k = max(sils, key=sils.get)
    labels = KMeans(n_clusters=best_k, n_init=10, random_state=seed).fit_predict(x)
    # One-hot categorical data rarely has crisp k-means structure; if the silhouette
    # curve is flat, k is an arbitrary cut and the profile-frequency view is the real
    # evidence. Say so rather than presenting k as a discovered number of modes.
    spread = max(sils.values()) - min(sils.values())

    clusters = []
    for c in range(best_k):
        members = [r for r, l in zip(rows, labels) if l == c]
        modal = {a: Counter(m[a] for m in members).most_common(1)[0] for a in AXES}
        clusters.append({
            "cluster": c,
            "size": len(members),
            "share": len(members) / len(rows),
            # An axis is "defining" when the cluster agrees on it far more than chance.
            "signature": {a: {"value": val, "purity": n / len(members)}
                          for a, (val, n) in modal.items()},
            "trait_mix": dict(Counter(m["trait_id"] for m in members).most_common()),
            "examples": [m["summary"] for m in members[:4]],
            "scenario_ids": [m["scenario_id"] for m in members],
        })
    return {"k": best_k, "silhouettes": sils, "labels": labels.tolist(),
            "silhouette_spread": spread, "structure_is_flat": bool(spread < 0.05),
            "best_silhouette": sils[best_k],
            "clusters": sorted(clusters, key=lambda c: -c["size"])}


def norm_domain(d: str) -> str:
    """Normalise the free-text domain field so case variants collapse.

    Args:
        d: Raw domain string.

    Returns:
        Lowercased, singularised-enough domain key.
    """
    return re.sub(r"\s+", " ", d.strip().lower())


def confound_check(rows: list[dict], labels: list[int]) -> dict:
    """Measure how much the clustering merely re-derives trait / domain labels.

    Args:
        rows: Labelled feature rows.
        labels: Cluster assignment per row.

    Returns:
        Adjusted mutual information of clusters vs trait and vs domain.
    """
    traits = [r["trait_id"] for r in rows]
    domains = [norm_domain(r["domain"]) for r in rows]
    return {
        "ami_cluster_vs_trait": float(adjusted_mutual_info_score(labels, traits)),
        "ami_cluster_vs_domain": float(adjusted_mutual_info_score(labels, domains)),
        "ami_trait_vs_domain": float(adjusted_mutual_info_score(traits, domains)),
        "per_axis_ami_vs_trait": {a: float(adjusted_mutual_info_score(
            [r[a] for r in rows], traits)) for a in AXES},
    }


def template_stats(texts: list[str], rows: list[dict]) -> dict:
    """Surface-level repetition in the raw replies: openers, stock phrases, near-dups.

    Args:
        texts: Raw assistant replies, aligned with rows.
        rows: Labelled feature rows (for signature_phrases and ids).

    Returns:
        Opener histograms, phrase counts, near-duplicate pairs and structural rates.
    """
    def words(t: str, n: int) -> str:
        w = re.findall(r"[a-z']+", t.lower())[:n]
        return " ".join(w)

    open3 = Counter(words(t, 3) for t in texts)
    open6 = Counter(words(t, 6) for t in texts)
    phrases = Counter(p.strip().lower() for r in rows for p in r["signature_phrases"]
                      if isinstance(p, str) and p.strip())

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=60000,
                          sublinear_tf=True, stop_words=None)
    x = vec.fit_transform(texts)
    sim = (x @ x.T).toarray()
    assert sim.shape == (len(texts), len(texts)), f"bad sim matrix {sim.shape}"
    np.fill_diagonal(sim, 0.0)
    iu = np.triu_indices(len(texts), k=1)
    flat = sim[iu]
    hits = np.argsort(flat)[::-1][:40]
    pairs = [{"a": rows[iu[0][h]]["scenario_id"], "b": rows[iu[1][h]]["scenario_id"],
              "cosine": float(flat[h]),
              "same_trait": rows[iu[0][h]]["trait_id"] == rows[iu[1][h]]["trait_id"]}
             for h in hits if flat[h] >= NEAR_DUP_THRESHOLD]

    return {
        "top_opening_3grams": open3.most_common(15),
        "top_opening_6grams": open6.most_common(15),
        "distinct_opening_3grams": len(open3),
        "top_signature_phrases": phrases.most_common(25),
        "mean_pairwise_cosine": float(flat.mean()),
        "p99_pairwise_cosine": float(np.percentile(flat, 99)),
        "near_dup_pairs_ge_threshold": int((flat >= NEAR_DUP_THRESHOLD).sum()),
        "near_dup_threshold": NEAR_DUP_THRESHOLD,
        "top_pairs": pairs,
        "pair_cosines": flat,
        "structure": {
            "uses_bold": sum("**" in t for t in texts) / len(texts),
            "uses_bullets": sum(bool(re.search(r"^\s*[-*] ", t, re.M)) for t in texts) / len(texts),
            "uses_headers": sum(bool(re.search(r"^#+ ", t, re.M)) for t in texts) / len(texts),
            "uses_em_dash": sum("—" in t for t in texts) / len(texts),
            "mean_words": float(np.mean([len(t.split()) for t in texts])),
        },
    }


def _png(fig) -> str:
    """Encode a figure as a base64 PNG for the self-contained dashboard.

    Args:
        fig: Matplotlib figure.

    Returns:
        Base64-encoded PNG data.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    return base64.b64encode(buf.getvalue()).decode()


def make_plots(ax_stats: dict, clu: dict, tmpl: dict, rows: list[dict],
               plot_dir: Path) -> dict[str, str]:
    """Write the audit figures to disk and return them base64-encoded.

    Args:
        ax_stats: Output of axis_stats.
        clu: Output of cluster.
        tmpl: Output of template_stats.
        rows: Labelled feature rows.
        plot_dir: Directory for the PNGs.

    Returns:
        Mapping plot name -> base64 PNG.
    """
    plot_dir.mkdir(parents=True, exist_ok=True)
    out = {}

    names = list(ax_stats)
    ent = [ax_stats[a]["norm_entropy"] * 100 for a in names]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(names, ent, color="#4C72B0", edgecolor="black", linewidth=0.8)
    for i, (a, e) in enumerate(zip(names, ent)):
        ax.text(e + 1.5, i, f"{ax_stats[a]['top_value']} {ax_stats[a]['top_share']:.0%}",
                va="center", fontsize=12)
    ax.set_xlim(0, 130)
    ax.set_xlabel("normalised entropy (% of uniform over vocabulary)")
    ax.set_title("Feature diversity per axis (label = modal value, its share)")
    ax.grid(True, linestyle="--", alpha=0.2, axis="x")
    fig.tight_layout()
    fig.savefig(plot_dir / "axis_entropy.png", bbox_inches="tight")
    out["axis_entropy"] = _png(fig)
    plt.close(fig)

    sizes = [c["share"] * 100 for c in clu["clusters"]]
    labs = [f"C{c['cluster']}" for c in clu["clusters"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labs, sizes, color="#DD8452", edgecolor="black", linewidth=0.8)
    ax.set_ylabel("share of corpus (%)")
    ax.set_xlabel(f"response-mode cluster (k={clu['k']}, chosen by silhouette)")
    ax.set_title("Cluster sizes")
    ax.grid(True, linestyle="--", alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(plot_dir / "cluster_sizes.png", bbox_inches="tight")
    out["cluster_sizes"] = _png(fig)
    plt.close(fig)

    traits = sorted({r["trait_id"] for r in rows})
    mat = np.array([[c["trait_mix"].get(t, 0) for t in traits] for c in clu["clusters"]],
                   dtype=float)
    mat = mat / mat.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(labs) + 3))
    im = ax.imshow(mat, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(len(traits)), traits)
    ax.set_yticks(range(len(labs)), labs)
    ax.set_xlabel("constitution trait")
    ax.set_title("Trait composition within each cluster (% of cluster)")
    fig.colorbar(im, ax=ax, label="%")
    fig.tight_layout()
    fig.savefig(plot_dir / "cluster_by_trait.png", bbox_inches="tight")
    out["cluster_by_trait"] = _png(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(tmpl["pair_cosines"], bins=80, color="#55A868", edgecolor="black", linewidth=0.6)
    ax.axvline(NEAR_DUP_THRESHOLD, color="crimson", linestyle="--", linewidth=2,
               label=f"near-dup threshold {NEAR_DUP_THRESHOLD}")
    ax.set_yscale("log")
    ax.set_xlabel("TF-IDF cosine similarity between reply pairs")
    ax.set_ylabel("pair count (log)")
    ax.set_title("Pairwise reply similarity")
    ax.legend(fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_dir / "pair_similarity.png", bbox_inches="tight")
    out["pair_similarity"] = _png(fig)
    plt.close(fig)

    top = tmpl["top_opening_3grams"][:12][::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh([t[0] for t in top], [t[1] for t in top], color="#8172B3",
            edgecolor="black", linewidth=0.8)
    ax.set_xlabel("replies starting with this 3-gram")
    ax.set_title("Most common reply openers")
    ax.grid(True, linestyle="--", alpha=0.2, axis="x")
    fig.tight_layout()
    fig.savefig(plot_dir / "openers.png", bbox_inches="tight")
    out["openers"] = _png(fig)
    plt.close(fig)
    return out


def _md_table(headers: list[str], rows: list[list]) -> str:
    """Render a markdown table.

    Args:
        headers: Column headers.
        rows: Row cells.

    Returns:
        Markdown table text.
    """
    head = "| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n"
    return head + "".join("| " + " | ".join(str(c) for c in r) + " |\n" for r in rows)


def write_report(out: Path, res: dict, tmpl: dict, clu: dict, n: int, plot_dir: Path) -> None:
    """Write the greppable markdown mirror of every number in the audit.

    Args:
        out: Output directory.
        res: Assembled results dict.
        tmpl: Template statistics.
        clu: Clustering result.
        n: Number of labelled rows.
        plot_dir: Directory holding the PNGs.
    """
    a = res["axes"]
    p = res["profiles"]
    c = res["confound"]
    rel = plot_dir.relative_to(out)
    lines = [
        f"# Response-diversity audit — {res['meta']['input']}",
        "",
        f"Rows labelled: **{n}** | judge: `{res['meta']['model']}` | "
        f"features: {len(AXES)} axes, clustering k={clu['k']}",
        "",
        "## Headline",
        "",
        f"- **Effective response modes: {p['effective_profiles']:.1f}** out of "
        f"{p['distinct_profiles']} distinct observed profiles "
        f"({p['possible_profiles']:,} combinatorially possible).",
        f"- Top 10 profiles cover **{p['top10_share']:.1%}** of the corpus; "
        f"{p['singleton_profiles']} profiles occur exactly once.",
        f"- Mean pairwise reply similarity **{tmpl['mean_pairwise_cosine']:.3f}** "
        f"(p99 {tmpl['p99_pairwise_cosine']:.3f}); "
        f"**{tmpl['near_dup_pairs_ge_threshold']}** pairs at cosine >= "
        f"{tmpl['near_dup_threshold']}.",
        f"- Clusters vs trait AMI **{c['ami_cluster_vs_trait']:.3f}**, vs domain "
        f"**{c['ami_cluster_vs_domain']:.3f}** (0 = the clustering found structure "
        "independent of the existing labels; high = it just re-derived them).",
        "",
        "## Per-axis diversity",
        "",
        f"![axis entropy]({rel}/axis_entropy.png)",
        "",
        _md_table(["axis", "modal value", "share", "norm. entropy", "effective values",
                   "vocab", "AMI vs trait"],
                  [[k, v["top_value"], f"{v['top_share']:.1%}", f"{v['norm_entropy']:.3f}",
                    f"{v['effective_values']:.2f}", v["vocab_size"],
                    f"{c['per_axis_ami_vs_trait'][k]:.3f}"] for k, v in a.items()]),
        "### Full value distributions",
        "",
    ]
    for axis, v in a.items():
        dist = ", ".join(f"{k} {n_ / n:.1%}" for k, n_ in v["counts"].items())
        lines.append(f"- **{axis}**: {dist}")
    lines += [
        "",
        "## Response modes (full axis tuple)",
        "",
        _md_table(["#", "count", "share", "profile"],
                  [[i, t["count"], f"{t['share']:.1%}",
                    " / ".join(t["profile"][ax] for ax in AXES)]
                   for i, t in enumerate(p["top_profiles"], 1)]),
        f"Axis order: {' / '.join(AXES)}",
        "",
        "## Clusters",
        "",
        f"![cluster sizes]({rel}/cluster_sizes.png)",
        "",
        f"![cluster by trait]({rel}/cluster_by_trait.png)",
        "",
        f"Silhouette by k: {', '.join(f'{k}:{v:.3f}' for k, v in clu['silhouettes'].items())}",
        "",
        (f"**Caveat: the silhouette curve is flat (spread {clu['silhouette_spread']:.3f}, "
         f"best {clu['best_silhouette']:.3f}) — this feature space has no crisp cluster "
         "structure, so k is an arbitrary cut. The profile-frequency table above is the "
         "load-bearing evidence; treat clusters as a convenient coarse partition."
         if clu["structure_is_flat"] else
         f"Silhouette peaks clearly at k={clu['k']} ({clu['best_silhouette']:.3f})."),
        "",
    ]
    for cl in clu["clusters"]:
        sig = ", ".join(f"{k}={v['value']} ({v['purity']:.0%})"
                        for k, v in cl["signature"].items() if v["purity"] >= 0.6)
        lines += [f"### C{cl['cluster']} — {cl['size']} rows ({cl['share']:.1%})",
                  "", f"Signature: {sig or '(no axis above 60% purity)'}", "",
                  f"Trait mix: {cl['trait_mix']}", "", "Example summaries:", ""]
        lines += [f"- {s}" for s in cl["examples"]] + [""]
    lines += [
        "## Surface templates",
        "",
        f"![openers]({rel}/openers.png)",
        "",
        f"![pair similarity]({rel}/pair_similarity.png)",
        "",
        f"Distinct opening 3-grams: {tmpl['distinct_opening_3grams']} across {n} replies.",
        "",
        _md_table(["opening 3-gram", "count", "share"],
                  [[o, ct, f"{ct / n:.1%}"] for o, ct in tmpl["top_opening_3grams"]]),
        _md_table(["opening 6-gram", "count"], [[o, ct] for o, ct in tmpl["top_opening_6grams"]]),
        _md_table(["stock phrase (judge-flagged)", "count"],
                  [[ph, ct] for ph, ct in tmpl["top_signature_phrases"]]),
        "### Structural rates",
        "",
        _md_table(["property", "share of replies"],
                  [[k, f"{v:.1%}" if k != "mean_words" else f"{v:.0f} words"]
                   for k, v in tmpl["structure"].items()]),
        "### Closest reply pairs",
        "",
        _md_table(["cosine", "a", "b", "same trait"],
                  [[f"{p_['cosine']:.3f}", p_["a"], p_["b"], p_["same_trait"]]
                   for p_ in tmpl["top_pairs"]]) or "(none above threshold)\n",
    ]
    (out / "report.md").write_text("\n".join(lines))


def write_dashboard(out: Path, res: dict, tmpl: dict, clu: dict, rows: list[dict],
                    texts: list[str], pngs: dict[str, str], n: int) -> None:
    """Write the self-contained HTML dashboard for human browsing.

    Args:
        out: Output directory.
        res: Assembled results dict.
        tmpl: Template statistics.
        clu: Clustering result.
        rows: Labelled feature rows.
        texts: Raw replies aligned with rows.
        pngs: Base64 PNGs from make_plots.
        n: Number of labelled rows.
    """
    by_id = {r["scenario_id"]: t for r, t in zip(rows, texts)}
    feat_by_id = {r["scenario_id"]: r for r in rows}
    p, c = res["profiles"], res["confound"]

    def esc(t: str) -> str:
        return html.escape(t)

    cards = "".join(
        f"<div class=card><div class=big>{v}</div><div class=lab>{k}</div></div>"
        for k, v in {
            "rows labelled": n,
            "effective response modes": f"{p['effective_profiles']:.1f}",
            "distinct profiles": p["distinct_profiles"],
            "top-10 profile share": f"{p['top10_share']:.0%}",
            "near-dup pairs": tmpl["near_dup_pairs_ge_threshold"],
            "AMI vs trait": f"{c['ami_cluster_vs_trait']:.2f}",
        }.items())

    axis_rows = "".join(
        f"<tr><td>{a}</td><td>{v['top_value']}</td><td>{v['top_share']:.1%}</td>"
        f"<td>{v['norm_entropy']:.3f}</td><td>{v['effective_values']:.2f}</td>"
        f"<td>{c['per_axis_ami_vs_trait'][a]:.3f}</td></tr>"
        for a, v in res["axes"].items())

    prof_rows = "".join(
        f"<tr><td>{i}</td><td>{t['count']}</td><td>{t['share']:.1%}</td>"
        f"<td class=mono>{esc(' / '.join(t['profile'][ax] for ax in AXES))}</td></tr>"
        for i, t in enumerate(p["top_profiles"], 1))

    clus = ""
    for cl in clu["clusters"]:
        sig = ", ".join(f"<b>{k}</b>={v['value']} ({v['purity']:.0%})"
                        for k, v in cl["signature"].items() if v["purity"] >= 0.6)
        samples = ""
        for sid in cl["scenario_ids"][:3]:
            f = feat_by_id[sid]
            samples += (f"<details><summary>{sid} — {esc(f['summary'])}</summary>"
                        f"<pre>{esc(by_id[sid][:2500])}</pre></details>")
        clus += (f"<div class=cluster><h3>C{cl['cluster']} — {cl['size']} rows "
                 f"({cl['share']:.1%})</h3><p>{sig or 'no axis above 60% purity'}</p>"
                 f"<p class=mono>traits: {esc(json.dumps(cl['trait_mix']))}</p>{samples}</div>")

    open_rows = "".join(f"<tr><td class=mono>{esc(o)}</td><td>{ct}</td>"
                        f"<td>{ct / n:.1%}</td></tr>" for o, ct in tmpl["top_opening_3grams"])
    phrase_rows = "".join(f"<tr><td class=mono>{esc(ph)}</td><td>{ct}</td></tr>"
                          for ph, ct in tmpl["top_signature_phrases"])
    pair_rows = "".join(
        f"<tr><td>{p_['cosine']:.3f}</td><td class=mono>{p_['a']}</td>"
        f"<td class=mono>{p_['b']}</td><td>{p_['same_trait']}</td></tr>"
        for p_ in tmpl["top_pairs"])

    def img(k: str) -> str:
        return f"<img src='data:image/png;base64,{pngs[k]}'>"

    (out / "dashboard.html").write_text(f"""<!doctype html><meta charset=utf-8>
<title>Response diversity audit</title><style>
body{{font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;margin:0 auto;max-width:1150px;
padding:28px;color:#1c1c1e;background:#fafafa}}
h1{{margin-bottom:4px}} h2{{margin-top:38px;border-bottom:2px solid #e3e3e6;padding-bottom:6px}}
.cards{{display:flex;flex-wrap:wrap;gap:12px;margin:18px 0}}
.card{{background:#fff;border:1px solid #e3e3e6;border-radius:10px;padding:14px 18px;min-width:150px}}
.big{{font-size:26px;font-weight:650}} .lab{{color:#666;font-size:13px}}
table{{border-collapse:collapse;width:100%;background:#fff;margin:10px 0}}
th,td{{border:1px solid #e3e3e6;padding:6px 9px;text-align:left;font-size:14px}}
th{{background:#f0f0f3}} .mono{{font-family:ui-monospace,Menlo,monospace;font-size:13px}}
img{{max-width:100%;border:1px solid #e3e3e6;border-radius:8px;background:#fff;margin:10px 0}}
.cluster{{background:#fff;border:1px solid #e3e3e6;border-radius:10px;padding:14px 18px;margin:12px 0}}
pre{{white-space:pre-wrap;background:#f7f7f9;padding:12px;border-radius:8px;font-size:13px;
max-height:420px;overflow:auto}} summary{{cursor:pointer;padding:5px 0}}
</style>
<h1>Response-diversity audit</h1>
<p class=mono>{esc(res['meta']['input'])} · judge {esc(res['meta']['model'])} · {res['meta']['timestamp']}</p>
<div class=cards>{cards}</div>
<h2>Per-axis diversity</h2>{img('axis_entropy')}
<table><tr><th>axis<th>modal value<th>share<th>norm. entropy<th>effective values
<th>AMI vs trait</tr>{axis_rows}</table>
<h2>Response modes</h2>
<p>Full axis tuple treated as one mode. {p['distinct_profiles']} distinct observed of
{p['possible_profiles']:,} possible; effective {p['effective_profiles']:.1f}.</p>
<table><tr><th>#<th>count<th>share<th>profile ({esc(' / '.join(AXES))})</tr>{prof_rows}</table>
<h2>Clusters (k={clu['k']})</h2>{img('cluster_sizes')}{img('cluster_by_trait')}{clus}
<h2>Surface templates</h2>{img('openers')}{img('pair_similarity')}
<table><tr><th>opening 3-gram<th>count<th>share</tr>{open_rows}</table>
<h3>Judge-flagged stock phrases</h3><table><tr><th>phrase<th>count</tr>{phrase_rows}</table>
<h3>Closest reply pairs (cosine &ge; {tmpl['near_dup_threshold']})</h3>
<table><tr><th>cosine<th>a<th>b<th>same trait</tr>{pair_rows or '<tr><td colspan=4>none</td></tr>'}</table>
""")


def main(features: str, sft: str, out_dir: str | None = None, k_min: int = 4,
         k_max: int = 20, seed: int = 0) -> None:
    """Cluster the labelled features and write report, plots and dashboard.

    Args:
        features: Path to features.jsonl from extract.py.
        sft: Path to the stage_7 SFT jsonl the features were derived from.
        out_dir: Output directory; defaults to the features file's directory.
        k_min: Smallest candidate cluster count.
        k_max: Largest candidate cluster count.
        seed: Random seed for k-means.
    """
    rows = [json.loads(l) for l in Path(features).read_text().splitlines() if l.strip()]
    sft_rows = {json.loads(l)["metadata"]["scenario_id"]: json.loads(l)
                for l in Path(sft).read_text().splitlines() if l.strip()}
    missing = [r["scenario_id"] for r in rows if r["scenario_id"] not in sft_rows]
    assert not missing, f"{len(missing)} labelled ids absent from the SFT file: {missing[:3]}"
    texts = [sft_rows[r["scenario_id"]]["messages"][2]["content"] for r in rows]
    n = len(rows)
    print(f"loaded {n} labelled rows")

    out = Path(out_dir or Path(features).parent)
    plot_dir = out / "plots"
    ax_stats = axis_stats(rows)
    prof = profile_stats(rows)
    clu = cluster(rows, range(k_min, k_max + 1), seed=seed)
    conf = confound_check(rows, clu["labels"])
    tmpl = template_stats(texts, rows)

    print(f"effective response modes: {prof['effective_profiles']:.1f} "
          f"({prof['distinct_profiles']} distinct)")
    print(f"k={clu['k']} | AMI vs trait {conf['ami_cluster_vs_trait']:.3f} | "
          f"near-dup pairs {tmpl['near_dup_pairs_ge_threshold']}")

    res = {"meta": {"input": sft, "features": features, "rows": n,
                    "model": "see run_meta.json", "timestamp": timestamp()},
           "axes": ax_stats, "profiles": prof, "confound": conf,
           "clusters": clu["clusters"], "k": clu["k"], "silhouettes": clu["silhouettes"]}
    extract_meta = out / "run_meta.json"
    if extract_meta.exists():
        res["meta"]["model"] = json.loads(extract_meta.read_text())["config"]["model"]

    pngs = make_plots(ax_stats, clu, tmpl, rows, plot_dir)
    tmpl_json = {k: v for k, v in tmpl.items() if k != "pair_cosines"}
    (out / "results.json").write_text(json.dumps({**res, "templates": tmpl_json}, indent=2))
    write_report(out, res, tmpl, clu, n, plot_dir)
    write_dashboard(out, res, tmpl, clu, rows, texts, pngs, n)
    (out / "cluster_assignments.jsonl").write_text("".join(
        json.dumps({"scenario_id": r["scenario_id"], "cluster": int(l)}) + "\n"
        for r, l in zip(rows, clu["labels"])))
    # A separate file: run_meta.json in this directory belongs to the extract stage.
    (out / "run_meta_analyze.json").write_text(json.dumps(
        {"git_sha": git_sha(), "timestamp_utc": timestamp(), "stage": "analyze",
         "config": {"features": features, "sft": sft, "k_range": [k_min, k_max],
                    "seed": seed, "command": " ".join(sys.argv)}}, indent=2))
    print(f"wrote {out}/report.md, dashboard.html, results.json, plots/")


if __name__ == "__main__":
    fire.Fire(main)
