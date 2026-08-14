# ABOUTME: TURF offline stage 2 — embed all attributes, k-means the trigger side
# ABOUTME: (query+reasoning), summarise clusters, and optionally push the index to HF.

"""Build the searchable index from extract.py's output.

- Embeds every attribute (OpenRouter, qwen3-embedding-8b, 4096-d).
- TRIGGER side (10 query + 10 reasoning attrs per row, one pool, channel-tagged):
  k-means into K clusters, each LLM-summarised. This is the side hit-counting runs
  over — "which query/reasoning features co-occur with the behaviour".
- RESPONSE side: embedded, left unclustered — trace.py searches it directly.

Outputs into the extract dir: embeddings_trigger.npy, embeddings_response.npy,
centroids.npy, trigger_index.jsonl (attr text, row, channel, cluster),
cluster_summaries.jsonl, updated manifest.json. --push uploads the whole dir to
HF as LASR-Callum/<date>-turf-index-<name> with the required card.

    uv run python scratch/turf/index.py --dir output/turf/da9 --k 1000 [--push]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scratch.turf.common import embed, kmeans, load_config  # noqa: E402
from scratch.turf.prompts import CLUSTER_SUMMARY_PROMPT  # noqa: E402
from src.endpoints.openrouter import OpenRouterClient, map_threaded  # noqa: E402
from src.utils import git_sha, read_jsonl, timestamp  # noqa: E402


def main(dir: str, k: int | None = None, summary_model: str | None = None,
         push: bool = False, name: str | None = None, config: str | None = None) -> None:
    """Build the index over `dir`/attributes.jsonl (see module docstring).

    Hyperparameters come from config.yaml (--config to swap); --k/--summary_model
    override."""
    cfg = load_config(config)
    k = k or int(cfg.k_clusters)
    summary_model = summary_model or str(cfg.summary_model)
    d = Path(dir)
    rows = read_jsonl(d / "attributes.jsonl")
    print(f">>> {len(rows)} extracted rows")

    trigger, response = [], []  # (row, channel, text)
    for r in rows:
        for a in r["query_attrs"]:
            trigger.append((r["row"], "query", a))
        for a in (r["reasoning_attrs"] or []):
            trigger.append((r["row"], "reasoning", a))
        for a in r["response_attrs"]:
            response.append((r["row"], "response", a))
    print(f">>> trigger attrs: {len(trigger)}, response attrs: {len(response)}")

    emb_t = embed([t for _, _, t in trigger], str(cfg.embed_model))
    emb_r = embed([t for _, _, t in response], str(cfg.embed_model))
    np.save(d / "embeddings_trigger.npy", emb_t)   # fp32, as SURF stores embeddings.npy
    np.save(d / "embeddings_response.npy", emb_r)

    k = min(k, len(trigger) // 4)  # never more clusters than attrs/4
    cent, assign, dists = kmeans(emb_t, k, max_iter=int(cfg.kmeans_max_iter),
                                 seed=int(cfg.kmeans_seed))
    np.save(d / "centroids.npy", cent)  # fp32, as SURF ships centroids.npy

    with (d / "trigger_index.jsonl").open("w") as f:
        for (row, channel, text), c in zip(trigger, assign):
            f.write(json.dumps({"row": row, "channel": channel, "text": text,
                                "cluster": int(c)}) + "\n")
    with (d / "response_index.jsonl").open("w") as f:
        for row, _, text in response:
            f.write(json.dumps({"row": row, "text": text}) + "\n")

    # Summarise each cluster from up to 50 member attributes, closest-to-centroid
    # first (SURF's top_attributes ordering, its top-100 halved; distances from
    # kmeans's final pass). Prompt is SURF's, prefixed by the majority channel.
    members: dict[int, list[tuple[float, str, str]]] = {}
    for (row, channel, text), c, dist in zip(trigger, assign, dists):
        members.setdefault(int(c), []).append((float(dist), channel, text))
    client = OpenRouterClient()
    cluster_ids = sorted(members)

    def summarise(j: int) -> dict:
        cid = cluster_ids[j]
        top = sorted(members[cid])[:int(cfg.summary_top_attrs)]
        channels = [ch for _, ch, _ in members[cid]]
        share_reasoning = channels.count("reasoning") / len(channels)
        prefix, noun = (("The reasoning", "reasoning traces") if share_reasoning > 0.5
                        else ("The query", "queries"))
        res = client.chat(summary_model, [{"role": "user", "content":
                          CLUSTER_SUMMARY_PROMPT.format(
                              channel_noun=noun, prefix=prefix,
                              attributes="\n".join(f"- {t}" for _, _, t in top))}],
                          temperature=float(cfg.judge_temperature))
        summary = res.content.strip()
        if not summary.lower().startswith(prefix.lower()):
            summary = f"{prefix} {summary}"  # SURF's prefix enforcement
        return {"cluster": cid, "size": len(members[cid]), "summary": summary,
                "share_reasoning": share_reasoning}

    summaries = map_threaded(summarise, len(cluster_ids), desc="summarising")
    with (d / "cluster_summaries.jsonl").open("w") as f:
        for s in sorted(summaries, key=lambda s: -s["size"]):
            f.write(json.dumps(s) + "\n")

    manifest = json.loads((d / "manifest.json").read_text())
    manifest.update({"k_clusters": k, "trigger_attrs": len(trigger),
                     "response_attrs": len(response),
                     "summary_model": summary_model, "embed_model": str(cfg.embed_model),
                     "index_git_sha": git_sha(), "index_timestamp": timestamp()})
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f">>> index built: {k} clusters over {len(trigger)} trigger attrs")

    if push:
        from src.huggingface import push_run_dir

        date = manifest["timestamp"][:8]
        repo = f"LASR-Callum/{date[:4]}-{date[4:6]}-{date[6:8]}-turf-index-" + (
            name or manifest["source_dataset"].split("/")[-1])
        url = push_run_dir(d, repo, {
            "experiment": "TURF attribute index (trigger=query+reasoning clustered, "
                          "response searchable) for data-property attribution",
            "date_generated": f"{date[:4]}-{date[4:6]}-{date[6:8]}",
            "constitution": "inherited from the source dataset "
                            f"({manifest['source_dataset']})",
            "source_repo": f"jamie/turf @ {git_sha()}",
            "models": f"extractor: {manifest['extractor_model']}; "
                      f"embedder: {manifest['embed_model']}; summaries: {summary_model}",
            "generation_config": json.dumps(
                {"n_attrs_per_channel": manifest.get("n_attrs_per_channel"), "k": k,
                 "extract_temperature": manifest.get("extract_temperature"),
                 "summary_temperature": float(cfg.judge_temperature)}),
            "schema": "attributes.jsonl, trigger_index.jsonl (attr->cluster), "
                      "response_index.jsonl, embeddings_*.npy (fp32), centroids.npy, "
                      "cluster_summaries.jsonl, styles.json, manifest.json",
            "provenance": f"scratch/turf/extract.py + index.py over "
                          f"{manifest['source_dataset']}/{manifest['source_file']}",
        }, private=True)
        print(f">>> pushed index to {url}")


if __name__ == "__main__":
    fire.Fire(main)
