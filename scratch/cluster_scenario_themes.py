# ABOUTME: Groups the difficult-advice scenarios into thematic categories by embedding them.
# ABOUTME: Local sentence-transformers + KMeans; only cluster NAMING touches an API.
"""What themes does the corpus actually cover?

`domain` is a free-text field with 495 near-duplicate values ("Small Business" vs
"Small business"), so it cannot answer this. This embeds each scenario's situation plus its
tempting shortcut -- the two fields that carry the theme -- clusters them, and picks k by
silhouette rather than by assertion.

Naming is the only step that leaves the machine: one call describing every cluster at once,
from its distinctive terms and its three most central members.

Run: uv run python scratch/cluster_scenario_themes.py --k_min 8 --k_max 20
"""
import collections
import json
from pathlib import Path

import fire
import numpy as np
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from src.utils import read_jsonl, timestamp, write_run_meta

SRC = "output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl"
OUT_ROOT = Path("output/scenario_themes")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

NAME_PROMPT = """Below are {n} clusters of ethically difficult scenarios. Each shows its \
size, its most distinctive terms, and three representative scenarios.

Give each cluster a short thematic NAME (2-5 words, title case) and a one-sentence \
description of what unites its scenarios. Name the THEME -- the kind of ethical situation -- \
not the industry, and make the names mutually distinguishable.

{blocks}

Return ONLY JSON: {{"clusters": [{{"id": <int>, "name": "<name>", "description": "<sentence>"}}]}}"""


def _pick_k(x: np.ndarray, k_min: int, k_max: int, seed: int) -> tuple[int, dict]:
    """Choose the cluster count by silhouette score.

    Args:
        x: Embedding matrix.
        k_min: Smallest k to try.
        k_max: Largest k to try.
        seed: KMeans seed.

    Returns:
        The best k and the full {k: score} sweep.
    """
    scores = {}
    sample = np.random.default_rng(seed).choice(len(x), min(1500, len(x)), replace=False)
    for k in range(k_min, k_max + 1):
        labels = KMeans(n_clusters=k, n_init=4, random_state=seed).fit_predict(x)
        scores[k] = float(silhouette_score(x[sample], labels[sample]))
        print(f"  k={k:<3} silhouette {scores[k]:.4f}")
    best = max(scores, key=scores.get)
    return best, scores


def _terms(texts: list[str], labels: np.ndarray, k: int, top: int = 10) -> dict[int, list[str]]:
    """Most distinctive TF-IDF terms per cluster, by mean weight within the cluster."""
    vec = TfidfVectorizer(max_features=6000, stop_words="english", ngram_range=(1, 2),
                          min_df=3)
    m = vec.fit_transform(texts)
    names = np.array(vec.get_feature_names_out())
    out = {}
    for c in range(k):
        rows = m[labels == c]
        if rows.shape[0] == 0:
            out[c] = []
            continue
        mean = np.asarray(rows.mean(axis=0)).ravel()
        out[c] = list(names[mean.argsort()[::-1][:top]])
    return out


def main(k_min: int = 8, k_max: int = 20, seed: int = 0,
         name_model: str = "google/gemini-3-flash-preview") -> None:
    """Cluster the corpus scenarios by theme and write the grouping.

    Args:
        k_min: Smallest cluster count to consider.
        k_max: Largest cluster count to consider.
        seed: Seed for KMeans and sampling.
        name_model: OpenRouter model used once to name the clusters.
    """
    load_dotenv()
    rows = list(read_jsonl(SRC))
    texts = [f"{r['metadata']['situation']} {r['metadata']['shortcut']}" for r in rows]
    print(f"{len(rows)} scenarios; embedding with {EMBED_MODEL}")

    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer(EMBED_MODEL).encode(
        texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    emb = np.asarray(emb)
    assert emb.shape[0] == len(rows), emb.shape

    print("\nchoosing k by silhouette:")
    k, sweep = _pick_k(emb, k_min, k_max, seed)
    print(f"\nbest k = {k}")
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(emb)
    labels = km.labels_

    terms = _terms(texts, labels, k)
    blocks, members = [], {}
    for c in range(k):
        idx = np.where(labels == c)[0]
        d = np.linalg.norm(emb[idx] - km.cluster_centers_[c], axis=1)
        central = idx[d.argsort()[:3]]
        members[c] = idx
        ex = "\n".join(f"    - {rows[i]['metadata']['situation'][:190]}" for i in central)
        blocks.append(f"CLUSTER {c} (n={len(idx)})\n  terms: {', '.join(terms[c])}\n{ex}")

    from src.endpoints.openrouter import OpenRouterClient
    client = OpenRouterClient()
    txt = client.chat(model=name_model, temperature=0.0, max_tokens=3000,
                      messages=[{"role": "user", "content": NAME_PROMPT.format(
                          n=k, blocks="\n\n".join(blocks))}]).content
    named = json.loads(txt[txt.find("{"):txt.rfind("}") + 1], strict=False)["clusters"]
    name_of = {c["id"]: c for c in named}

    out_dir = OUT_ROOT / timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    assign = []
    for c in range(k):
        info = name_of.get(c, {"name": f"cluster {c}", "description": ""})
        for i in members[c]:
            assign.append({"scenario_id": rows[i]["metadata"]["scenario_id"],
                           "trait": rows[i]["metadata"]["trait_id"],
                           "domain": rows[i]["metadata"]["domain"],
                           "cluster": c, "theme": info["name"]})
    with (out_dir / "assignments.jsonl").open("w") as f:
        for a in assign:
            f.write(json.dumps(a) + "\n")

    L = [f"# Scenario themes — {len(rows)} scenarios, k={k}", "",
         f"Embedded situation+shortcut with `{EMBED_MODEL}`; k chosen by silhouette "
         f"over {k_min}–{k_max} (best {sweep[k]:.4f}).", "",
         "| # | theme | n | share | traits (top 3) | distinctive terms |",
         "|---|---|--:|--:|---|---|"]
    summary = []
    for c in sorted(range(k), key=lambda x: -len(members[x])):
        info = name_of.get(c, {"name": f"cluster {c}", "description": ""})
        idx = members[c]
        tr = collections.Counter(rows[i]["metadata"]["trait_id"] for i in idx)
        top = ", ".join(f"{t}({n})" for t, n in tr.most_common(3))
        L.append(f"| {c} | **{info['name']}** | {len(idx)} | {100 * len(idx) / len(rows):.1f}% "
                 f"| {top} | {', '.join(terms[c][:6])} |")
        summary.append({"cluster": c, "name": info["name"],
                        "description": info.get("description", ""), "n": len(idx),
                        "traits": dict(tr), "terms": terms[c],
                        "examples": [rows[i]["metadata"]["situation"]
                                     for i in idx[:3]]})
    L += ["", "## Themes", ""]
    for s in sorted(summary, key=lambda x: -x["n"]):
        L += [f"**{s['name']}** — {s['n']} scenarios ({100 * s['n'] / len(rows):.1f}%)", "",
              f"{s['description']}", ""]
    (out_dir / "themes.md").write_text("\n".join(L) + "\n")
    (out_dir / "themes.json").write_text(json.dumps(
        {"k": k, "silhouette_sweep": sweep, "clusters": summary}, indent=2))
    write_run_meta(out_dir, {"src": SRC, "embed_model": EMBED_MODEL, "k_min": k_min,
                             "k_max": k_max, "seed": seed, "name_model": name_model},
                   {"n_scenarios": len(rows), "k": k})
    print("\n".join(L[:26]))
    print(f"\n>>> {out_dir}")


if __name__ == "__main__":
    fire.Fire(main)
