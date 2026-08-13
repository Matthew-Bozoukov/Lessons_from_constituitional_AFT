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

from scratch.turf.common import embed, kmeans  # noqa: E402
from scratch.turf.prompts import CLUSTER_SUMMARY_PROMPT  # noqa: E402
from src.endpoints.openrouter import OpenRouterClient, map_threaded  # noqa: E402
from src.utils import git_sha, read_jsonl, timestamp  # noqa: E402


def main(dir: str, k: int = 1000, summary_model: str = "anthropic/claude-sonnet-4.5",
         push: bool = False, name: str | None = None) -> None:
    """Build the index over `dir`/attributes.jsonl (see module docstring)."""
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

    emb_t = embed([t for _, _, t in trigger])
    emb_r = embed([t for _, _, t in response])
    np.save(d / "embeddings_trigger.npy", emb_t.astype(np.float16))
    np.save(d / "embeddings_response.npy", emb_r.astype(np.float16))

    k = min(k, len(trigger) // 4)  # never more clusters than attrs/4
    cent, assign = kmeans(emb_t, k)
    np.save(d / "centroids.npy", cent.astype(np.float16))

    with (d / "trigger_index.jsonl").open("w") as f:
        for (row, channel, text), c in zip(trigger, assign):
            f.write(json.dumps({"row": row, "channel": channel, "text": text,
                                "cluster": int(c)}) + "\n")
    with (d / "response_index.jsonl").open("w") as f:
        for row, _, text in response:
            f.write(json.dumps({"row": row, "text": text}) + "\n")

    # Summarise each cluster from up to 12 member attributes (top by centroid sim).
    members: dict[int, list[tuple[float, str, str]]] = {}
    norm = emb_t / (np.linalg.norm(emb_t, axis=1, keepdims=True) + 1e-9)
    sims = (norm * cent[assign]).sum(axis=1)
    for (row, channel, text), c, s in zip(trigger, assign, sims):
        members.setdefault(int(c), []).append((float(s), channel, text))
    client = OpenRouterClient()
    cluster_ids = sorted(members)

    def summarise(j: int) -> dict:
        cid = cluster_ids[j]
        top = sorted(members[cid], reverse=True)[:12]
        res = client.chat(summary_model, [{"role": "user", "content":
                          CLUSTER_SUMMARY_PROMPT.format(
                              attributes="\n".join(t for _, _, t in top))}],
                          temperature=0.0)
        channels = [ch for _, ch, _ in members[cid]]
        return {"cluster": cid, "size": len(members[cid]),
                "summary": res.content.strip(),
                "share_reasoning": channels.count("reasoning") / len(channels)}

    summaries = map_threaded(summarise, len(cluster_ids), desc="summarising")
    with (d / "cluster_summaries.jsonl").open("w") as f:
        for s in sorted(summaries, key=lambda s: -s["size"]):
            f.write(json.dumps(s) + "\n")

    manifest = json.loads((d / "manifest.json").read_text())
    manifest.update({"k_clusters": k, "trigger_attrs": len(trigger),
                     "response_attrs": len(response),
                     "summary_model": summary_model, "embed_model": "qwen/qwen3-embedding-8b",
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
                      f"embedder: qwen/qwen3-embedding-8b; summaries: {summary_model}",
            "generation_config": json.dumps({"n_attrs_per_channel": 10, "k": k,
                                             "temperature": 0.0}),
            "schema": "attributes.jsonl, trigger_index.jsonl (attr->cluster), "
                      "response_index.jsonl, embeddings_*.npy (fp16), centroids.npy, "
                      "cluster_summaries.jsonl, styles.json, manifest.json",
            "provenance": f"scratch/turf/extract.py + index.py over "
                          f"{manifest['source_dataset']}/{manifest['source_file']}",
        }, private=True)
        print(f">>> pushed index to {url}")


if __name__ == "__main__":
    fire.Fire(main)
