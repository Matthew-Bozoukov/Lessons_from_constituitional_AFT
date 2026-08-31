# ABOUTME: Sort the feature-discovery clusters into what they actually describe — a reasoning
# ABOUTME: property, a topic, a surface style, or a stance — so the reasoning ones can be studied.

"""Which of the 150 clusters are about HOW the model reasons.

The extraction prompt invites features at every level at once: "Backtracks in reasoning" and
"Talks about apples" are both valid answers to it, so the clusters mix reasoning moves with
subject matter, formatting habits and response stances. An ablation or a transfer claim only
means something for the first kind, and eyeballing 150 labels gets that wrong — a label like
"Structured case-specific reasoning under uncertainty for safety judgments" reads as a
reasoning property until its features turn out to be mostly about safety topics.

So each cluster is classified from its LABEL PLUS ITS SAMPLE FEATURES, which is the only
evidence of what it actually contains, and every verdict carries the judge's one-line
justification so a human can overrule it.

Run:
  uv run python scratch/classify_clusters.py
"""

from __future__ import annotations

import html
import json
import sys
import threading
from pathlib import Path

import fire

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.endpoints.openrouter import OpenRouterClient, map_threaded  # noqa: E402
from src.utils import extract_json, git_sha, timestamp  # noqa: E402

ROUTING = {"provider": {"ignore": ["Amazon Bedrock"]}, "reasoning": {"enabled": False}}

CATEGORIES = {
    "reasoning": "a PROPERTY OF THE REASONING ITSELF — how the model thinks: the moves it "
                 "makes, how it structures deliberation, what it weighs against what, how it "
                 "handles its own uncertainty, whether it monitors or corrects itself",
    "stance": "a RESPONSE STANCE OR OUTCOME — what the model decides or does: refuses, "
              "offers an alternative, defers to the user, sets a boundary. About the "
              "conclusion or the action, not the thinking that produced it",
    "topic": "SUBJECT MATTER — what the conversation is about: medicine, law, careers, "
             "family. True of the scenario, not of the reasoning",
    "style": "SURFACE FORM — formatting, register, voice, sentence shape, punctuation, "
             "length. How the text looks rather than what it does",
}

CLASSIFY = """You are sorting clusters of automatically-extracted features into what they \
actually describe.

The features were written by a model reading one reasoning trace at a time, with a prompt \
that invited anything from "Backtracks in reasoning" to "Talks about apples". So the clusters \
mix levels, and the label alone is unreliable — judge from the FEATURES.

Choose exactly one category:

{cats}

Rules:
* Judge by what MOST of the features describe, not by the most flattering one. A cluster whose \
label sounds like a reasoning property but whose features are mostly about a subject area is \
`topic`.
* "reasoning" is the strict category. Deliberating about WHETHER TO REFUSE is reasoning; the \
refusal itself is `stance`. Reasoning in a long first-person voice is `reasoning` only if the \
features describe the thinking; if they describe the VOICE (length, person, monologue form) \
it is `style`.
* Ethical content does not make something reasoning. "Refuses to assist with fraud" is a \
stance; "Weighs competing obligations before deciding" is reasoning.
* Set `confidence` to "clear" or "borderline". Use "borderline" whenever the features are \
genuinely split between two categories, and name the runner-up in `also`.

Return only JSON:
{{"category": "...", "confidence": "clear|borderline", "also": "second category or empty", \
"why": "one sentence citing what the features actually describe"}}"""


def _prompt(cluster: dict) -> str:
    """Render one cluster for the classifier.

    Args:
        cluster: A record from clusters.json.

    Returns:
        The label, size and sample features as the user message.
    """
    feats = "\n".join(f"* {f}" for f in cluster["example_features"])
    return (f"Cluster label: {cluster['label']}\n"
            f"Traces: {cluster['n_traces']} ({cluster['prevalence']:.1%} of the corpus), "
            f"{cluster['n_features']} distinct feature strings\n\n"
            f"Sample features:\n{feats}")


def _page(recs: list[dict], out: Path) -> None:
    """Write the review page: reasoning clusters first, then everything filtered out.

    Args:
        recs: Classified cluster records.
        out: Output directory.
    """
    order = {"reasoning": 0, "stance": 1, "topic": 2, "style": 3}
    recs = sorted(recs, key=lambda r: (order[r["category"]], -r["prevalence"]))
    n_reason = sum(1 for r in recs if r["category"] == "reasoning")
    parts = [
        "<meta charset='utf-8'><title>Clusters that are about reasoning</title>",
        "<style>body{font:14px/1.6 system-ui;margin:0 auto;max-width:1000px;padding:2rem;"
        "background:#fbfbfa;color:#1a1a1a}h1{font-size:1.5rem}h2{font-size:1.15rem;"
        "margin-top:2.5rem;border-bottom:2px solid #ddd;padding-bottom:.3rem}"
        "details{border:1px solid #ddd;border-radius:6px;margin:.4rem 0;background:#fff}"
        "summary{padding:.6rem .8rem;cursor:pointer}summary:hover{background:#f6f6f4}"
        ".p{color:#666;font-variant-numeric:tabular-nums}.why{color:#444;padding:0 .8rem .4rem}"
        "ul{margin:.2rem 0 .6rem 0}li{color:#555;font-size:13px}"
        ".b{background:#fff4d6;border-radius:3px;padding:0 .3rem;font-size:12px}"
        "code{background:#eee;padding:0 .25rem;border-radius:3px}</style>",
        f"<h1>Which clusters are about reasoning?</h1>"
        f"<p>All 150 clusters, classified from their sample features rather than their labels. "
        f"<b>{n_reason} are properties of the reasoning itself</b>; the rest describe the "
        f"response stance, the subject matter, or the surface form. "
        f"<span class='b'>borderline</span> marks a cluster whose features were genuinely "
        f"split — those are the ones worth overruling.</p>",
    ]
    heads = {"reasoning": "Properties of the reasoning", "stance": "Response stance or outcome",
             "topic": "Subject matter", "style": "Surface form"}
    seen = set()
    for r in recs:
        if r["category"] not in seen:
            seen.add(r["category"])
            n = sum(1 for x in recs if x["category"] == r["category"])
            parts.append(f"<h2>{heads[r['category']]} &mdash; {n}</h2>")
        b = "<span class='b'>borderline" + (f" &rarr; {html.escape(r['also'])}"
                                            if r.get("also") else "") + "</span> " \
            if r["confidence"] == "borderline" else ""
        parts.append(
            f"<details><summary>{b}<b>C{r['cluster']}</b> {html.escape(r['label'])} "
            f"<span class='p'>&mdash; {r['prevalence']:.1%} of traces</span></summary>"
            f"<div class='why'><i>{html.escape(r['why'])}</i></div>"
            "<ul>" + "".join(f"<li>{html.escape(f)}</li>"
                             for f in r["example_features"][:12]) + "</ul></details>")
    (out / "cluster_kinds.html").write_text("\n".join(parts))

    md = ["# What each feature-discovery cluster actually describes", "",
          f"150 clusters classified from their sample features. "
          f"**{n_reason} are properties of the reasoning itself.**", ""]
    for cat in ("reasoning", "stance", "topic", "style"):
        sub = [r for r in recs if r["category"] == cat]
        md += [f"## {heads[cat]} — {len(sub)}", "",
               "| cluster | label | traces | note |", "|--:|---|--:|---|"]
        for r in sub:
            flag = "**borderline** " if r["confidence"] == "borderline" else ""
            md.append(f"| C{r['cluster']} | {r['label']} | {r['prevalence']:.1%} | "
                      f"{flag}{r['why']} |")
        md.append("")
    (out / "cluster_kinds.md").write_text("\n".join(md) + "\n")


def main(
    clusters: str = "docs/feature_discovery/clusters.json",
    model: str = "anthropic/claude-sonnet-5",
    workers: int = 12,
    out_dir: str | None = None,
) -> None:
    """Classify every cluster and write the review page.

    Args:
        clusters: Path to clusters.json.
        model: OpenRouter classifier model.
        workers: Concurrent requests.
        out_dir: Output directory; defaults to output/cluster_kinds/<timestamp>.
    """
    data = json.loads(Path(clusters).read_text())["clusters"]
    client = OpenRouterClient()
    sysp = CLASSIFY.format(cats="\n".join(f"* `{k}` — {v}" for k, v in CATEGORIES.items()))
    lock = threading.Lock()
    recs: list[dict] = []

    def run(i: int) -> tuple[int, int]:
        c = data[i]
        res = client.chat(model=model, temperature=0.0, max_tokens=400, extra_body=ROUTING,
                          messages=[{"role": "system", "content": sysp},
                                    {"role": "user", "content": _prompt(c)}])
        obj = extract_json(res.content)
        if obj["category"] not in CATEGORIES:
            raise ValueError(f"C{c['cluster']}: unknown category {obj['category']!r}")
        with lock:
            recs.append({"cluster": c["cluster"], "label": c["label"],
                         "prevalence": c["prevalence"], "n_traces": c["n_traces"],
                         "example_features": c["example_features"],
                         "category": obj["category"], "confidence": obj["confidence"],
                         "also": obj.get("also", ""), "why": obj["why"]})
        return res.prompt_tokens, res.completion_tokens

    usage = map_threaded(run, len(data), max_workers=workers, desc="classifying")
    out = Path(out_dir or f"output/cluster_kinds/{timestamp()}")
    out.mkdir(parents=True, exist_ok=True)
    _page(recs, out)
    (out / "cluster_kinds.jsonl").write_text("".join(
        json.dumps({k: v for k, v in r.items() if k != "example_features"}) + "\n"
        for r in sorted(recs, key=lambda r: r["cluster"])))
    tin, tout = sum(u[0] for u in usage), sum(u[1] for u in usage)
    (out / "run_meta.json").write_text(json.dumps(
        {"git_sha": git_sha(), "timestamp_utc": timestamp(), "clusters": clusters,
         "model": model, "n": len(recs), "tokens_in": tin, "tokens_out": tout,
         "cost_usd": tin / 1e6 * 3 + tout / 1e6 * 15, "command": " ".join(sys.argv)}, indent=2))

    from collections import Counter
    print(dict(Counter(r["category"] for r in recs)))
    print(f"borderline: {sum(1 for r in recs if r['confidence'] == 'borderline')}")
    print(f"cost ${tin / 1e6 * 3 + tout / 1e6 * 15:.3f} -> {out}")


if __name__ == "__main__":
    fire.Fire(main)
