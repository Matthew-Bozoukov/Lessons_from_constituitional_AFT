# ABOUTME: Render the browsable HTML dashboard for a finished clustering: the clusters,
# ABOUTME: the near-duplicate pairs and the keyword probes, in one self-contained file.

"""The HTML mirror.

Same content as the markdown report, arranged for reading rather than grepping. One pure
function from parsed JSON to a string; the caller decides where it lands.
"""

from __future__ import annotations

import html
import json

STYLE = """
body{font:15px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;margin:0 auto;max-width:1200px;
padding:28px;color:#1c1c1e;background:#fafafa}
h2{margin-top:34px;border-bottom:2px solid #e3e3e6;padding-bottom:6px}
table{border-collapse:collapse;width:100%;background:#fff;margin:10px 0}
th,td{border:1px solid #e3e3e6;padding:6px 9px;text-align:left;font-size:14px;vertical-align:top}
th{background:#f0f0f3;position:sticky;top:0} .mono{font-family:ui-monospace,Menlo,monospace;font-size:12px}
.cards{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}
.card{background:#fff;border:1px solid #e3e3e6;border-radius:10px;padding:14px 18px;min-width:150px}
.big{font-size:24px;font-weight:650} .lab{color:#666;font-size:13px}
ul{margin:6px 0 6px 18px;padding:0} summary{cursor:pointer}
"""


def _esc(value) -> str:
    """HTML-escape any value.

    Args:
        value: Anything stringifiable.

    Returns:
        The escaped string.
    """
    return html.escape(str(value))


def _card(value, label: str) -> str:
    """One headline-number card.

    Args:
        value: The number.
        label: Its caption.

    Returns:
        HTML.
    """
    return f"<div class=card><div class=big>{_esc(value)}</div><div class=lab>{label}</div></div>"


def render(meta: dict, clusters: list[dict], audit: dict, run_name: str) -> str:
    """Build the dashboard.

    Args:
        meta: clusters.json meta block.
        clusters: Per-cluster records, most prevalent first.
        audit: The dict from audit.audit_run.
        run_name: The run directory's basename.

    Returns:
        A complete HTML document.
    """
    pairs, probes = audit["near_duplicate_clusters"], audit["probes"]
    cluster_rows = "".join(
        f"<tr><td>{c['cluster']}</td><td><b>{_esc(c['label'])}</b></td>"
        f"<td>{c['n_traces']}</td><td>{c['prevalence']:.1%}</td>"
        f"<td>{c['n_features']}</td><td>{c['n_instances']}</td>"
        f"<td><details><summary>features</summary><ul>"
        + "".join(f"<li>{_esc(f)}</li>" for f in c["example_features"])
        + f"</ul><p class=mono>traits: {_esc(json.dumps(c['trait_mix']))}</p></details></td></tr>"
        for c in clusters)
    duplicate_rows = "".join(
        f"<tr><td>{p['cosine']:.3f}</td><td>{_esc(p['label_a'])}</td>"
        f"<td>{_esc(p['label_b'])}</td></tr>" for p in pairs[:30])
    probe_rows = "".join(
        f"<tr><td>{_esc(name)}</td><td>{p['traces']}</td><td>{p['prevalence']:.1%}</td>"
        f"<td>{p['unique_features']}</td>"
        f"<td class=mono>{_esc('; '.join(p['top_examples'][:4]))}</td></tr>"
        for name, p in probes.items())
    cards = "".join([_card(meta["traces"], "traces"),
                     _card(meta["unique_features"], "unique features"),
                     _card(meta["n_clusters"], "clusters"),
                     _card(meta["n_noise_features"], "unclustered (noise)"),
                     _card(len(pairs), "near-duplicate cluster pairs"),
                     _card(f"{meta.get('sanity_synonym', float('nan')):.2f}/"
                           f"{meta.get('sanity_unrelated', float('nan')):.2f}",
                           "embedding sanity syn/unrel")])
    return f"""<!doctype html><meta charset=utf-8>
<title>Feature discovery — {_esc(run_name)}</title><style>{STYLE}</style>
<h1>LLM-driven feature discovery</h1>
<p class=mono>{_esc(meta['traces'])} reasoning traces · {_esc(meta['feature_instances'])} feature
instances · {_esc(meta['unique_features'])} unique · embeddings {_esc(meta['embedding_model'])}
({_esc(meta['embedding_dim'])}d) · naming {_esc(meta['naming_model'])}
· {_esc(meta['clustering'])} · {_esc(meta['timestamp_utc'])}</p>
<div class=cards>{cards}</div>
<h2>Clusters by trace prevalence</h2>
<table><tr><th>#<th>label<th>traces<th>prevalence<th>features<th>instances<th>examples</tr>{cluster_rows}</table>
<h2>Near-duplicate clusters (centroid cosine &ge; {audit['dup_threshold']})</h2>
<table><tr><th>cosine<th>A<th>B</tr>{duplicate_rows or '<tr><td colspan=3>none</td></tr>'}</table>
<h2>Keyword probes</h2>
<table><tr><th>probe<th>traces<th>prevalence<th>unique features<th>examples</tr>{probe_rows}</table>
"""
