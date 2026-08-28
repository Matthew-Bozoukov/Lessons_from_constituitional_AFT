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

from scratch.turf.common import (  # noqa: E402
    assign_clusters,
    embed,
    kmeans,
    load_config,
    provider_override,
    refusal_from,
)
from scratch.turf.prompts import CLUSTER_SUMMARY_PROMPT  # noqa: E402
from src.endpoints.openrouter import (  # noqa: E402
    EmptyCompletionError,
    OpenRouterClient,
    ProviderRejectionError,
    map_threaded,
)
from src.utils import git_sha, read_jsonl, timestamp  # noqa: E402


def main(dir: str, k: int | None = None, summary_model: str | None = None,
         push: bool = False, name: str | None = None, config: str | None = None,
         provider: str | None = None, accept_refusals: bool = False) -> None:
    """Build the index over `dir`/attributes.jsonl (see module docstring).

    Hyperparameters come from config.yaml (--config to swap); --k/--summary_model
    override. --provider overrides the yaml provider pin for the summary chat calls
    (warns loudly; stamped into manifest.json — the embedder keeps its own pin)."""
    cfg = load_config(config)
    extra_body = provider_override(provider)
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

    # --- embeddings: cached like extraction's attributes.jsonl -----------------------
    # A fingerprint of (embed model, every attribute text) is stamped into the
    # manifest when the .npy files are written; a rerun whose fingerprint matches
    # loads them instead of re-embedding. Any change to the attributes or the
    # embedder invalidates the cache.
    import hashlib

    manifest = json.loads((d / "manifest.json").read_text())
    fp = {"model": str(cfg.embed_model),
          "trigger": hashlib.sha256(
              "\x00".join(t for _, _, t in trigger).encode()).hexdigest()[:16],
          "response": hashlib.sha256(
              "\x00".join(t for _, _, t in response).encode()).hexdigest()[:16]}
    emb_t_path = d / "embeddings_trigger.npy"
    emb_r_path = d / "embeddings_response.npy"
    embeddings_reused = (manifest.get("embed_fingerprints") == fp
                         and emb_t_path.exists() and emb_r_path.exists())
    if embeddings_reused:
        emb_t, emb_r = np.load(emb_t_path), np.load(emb_r_path)
        print(f">>> embeddings reused ({len(emb_t)}+{len(emb_r)} vectors, "
              "fingerprint match)", flush=True)
    else:
        emb_t = embed([t for _, _, t in trigger], str(cfg.embed_model))
        emb_r = embed([t for _, _, t in response], str(cfg.embed_model))
        np.save(emb_t_path, emb_t)   # fp32, as SURF stores embeddings.npy
        np.save(emb_r_path, emb_r)
        # checkpoint the fingerprint NOW, so a crash later never re-pays this stage
        manifest["embed_fingerprints"] = fp
        (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f">>> embedded {len(emb_t)}+{len(emb_r)} vectors", flush=True)

    # --- clustering: reused when embeddings were reused and k is unchanged -----------
    k = min(k, len(trigger) // 4)  # never more clusters than attrs/4
    cent_path = d / "centroids.npy"
    clustering_reused = (embeddings_reused and cent_path.exists()
                         and manifest.get("k_clusters") == k)
    if clustering_reused:
        cent = np.load(cent_path)
        assign, dists = assign_clusters(emb_t, cent)
        print(f">>> clustering reused ({k} cached centroids; assignment pass "
              "recomputed)", flush=True)
    else:
        cent, assign, dists = kmeans(emb_t, k, max_iter=int(cfg.kmeans_max_iter),
                                     seed=int(cfg.kmeans_seed))
        np.save(cent_path, cent)  # fp32, as SURF ships centroids.npy

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
    # DISTINCT source rows per cluster: `size` counts attributes, and one row
    # contributes up to 20 of them, so size overstates how much of the dataset a
    # cluster actually covers (up to 3x on da2203). Coverage is the ablatable number.
    rows_of: dict[int, set[int]] = {}
    for (row, channel, text), c, dist in zip(trigger, assign, dists):
        members.setdefault(int(c), []).append((float(dist), channel, text))
        rows_of.setdefault(int(c), set()).add(int(row))
    client = OpenRouterClient()

    # --- summaries: checkpointed per cluster, like extraction's per-row appends ------
    # Valid only against the clustering they were generated from: reused clustering
    # keeps the file and fills in missing clusters; recomputed clustering discards it.
    import threading

    sums_path = d / "cluster_summaries.jsonl"
    done_sums: dict[int, dict] = {}
    if sums_path.exists():
        if clustering_reused:
            # refused entries are NOT done — a rerun retries them
            done_sums = {s["cluster"]: s for s in read_jsonl(sums_path)
                         if not s.get("refused")}
        else:
            sums_path.unlink()
    cluster_ids = sorted(c for c in members if c not in done_sums)
    print(f">>> {len(done_sums)} summaries cached, {len(cluster_ids)} to generate",
          flush=True)
    sums_lock = threading.Lock()

    def summarise(j: int) -> dict:
        cid = cluster_ids[j]
        top = sorted(members[cid])[:int(cfg.summary_top_attrs)]
        channels = [ch for _, ch, _ in members[cid]]
        share_reasoning = channels.count("reasoning") / len(channels)
        # Prefix by channel purity; mixed clusters get a neutral frame rather than
        # letting the majority channel mislead (e.g. 63% rsn reading "The reasoning").
        if share_reasoning > 0.85:
            prefix, noun = "The reasoning", "reasoning traces"
        elif share_reasoning < 0.15:
            prefix, noun = "The query", "queries"
        else:
            prefix, noun = "The scenario", "queries and reasoning traces"
        try:
            res = client.chat(summary_model, [{"role": "user", "content":
                              CLUSTER_SUMMARY_PROMPT.format(
                                  channel_noun=noun, prefix=prefix,
                                  attributes="\n".join(f"- {t}" for _, _, t in top))}],
                              temperature=float(cfg.judge_temperature),
                              max_tokens=int(cfg.max_tokens),
                              **({"extra_body": extra_body} if extra_body else {}))
            summary = res.content.strip()
            if not summary.lower().startswith(prefix.lower()):
                summary = f"{prefix} {summary}"  # SURF's prefix enforcement
            rec = {"cluster": cid, "size": len(members[cid]), "summary": summary,
                   "share_reasoning": share_reasoning}
        except (EmptyCompletionError, ProviderRejectionError) as e:
            # retries exhausted — record a TYPED refusal, never a stand-in model's
            # text; the gate below makes a human decide what happens next
            rec = {"cluster": cid, "size": len(members[cid]), "summary": None,
                   "share_reasoning": share_reasoning, "refused": refusal_from(e)}
        with sums_lock:  # append-as-completed: a crash re-pays only missing clusters
            with sums_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        return rec

    map_threaded(summarise, len(cluster_ids), desc="summarising")
    # canonical order: rewrite the checkpoint file sorted by cluster size. Row
    # coverage is stamped HERE, not in summarise(), so it is deterministic-local
    # (never gated on the LLM summary cache) and backfills onto cached entries.
    summaries = list({s["cluster"]: s for s in read_jsonl(sums_path)}.values())
    for s in summaries:
        n_rows = len(rows_of.get(s["cluster"], ()))
        s["rows"] = n_rows
        s["coverage"] = round(n_rows / len(rows), 6)
    with sums_path.open("w") as f:
        for s in sorted(summaries, key=lambda s: -s["size"]):
            f.write(json.dumps(s) + "\n")

    # --- retrieval null: each cluster's chance hit rate --------------------------
    # Every response attribute serves as a pseudo-crux: the same k-NN voting
    # trace.py runs yields the hits each cluster collects for an INFORMATION-FREE
    # crux — base rate x retrieval geometry (hubness), the two prevalence effects
    # raw hit counts conflate. trace.py divides observed hits by this (lift).
    # Exact (all 22k response attrs, self excluded), deterministic, local-only.
    null_path = d / "null_hits.npy"
    k_ret = int(cfg.k_retrieve)
    if (null_path.exists() and manifest.get("null_k") == k_ret
            and manifest.get("null_fingerprint") == fp["response"]):
        print(">>> retrieval null reused", flush=True)
    else:
        row_of_attr = np.array([row for row, _, _ in response])
        clusters_of_row: dict[int, list[int]] = {}
        for (row, _, _), c in zip(trigger, assign):
            clusters_of_row.setdefault(row, []).append(int(c))
        emb_rn = emb_r / (np.linalg.norm(emb_r, axis=1, keepdims=True) + 1e-9)
        n_r = len(emb_rn)
        row_counts = np.zeros(int(row_of_attr.max()) + 1, dtype=np.int64)
        for i0 in range(0, n_r, 1024):
            sims = emb_rn[i0:i0 + 1024] @ emb_rn.T
            for j in range(sims.shape[0]):
                sims[j, i0 + j] = -np.inf  # a pseudo-crux never retrieves itself
            top_idx = np.argpartition(-sims, k_ret, axis=1)[:, :k_ret]
            row_counts += np.bincount(row_of_attr[top_idx.ravel()],
                                      minlength=len(row_counts))
        null = np.zeros(k, dtype=np.float64)
        for row, cnt in enumerate(row_counts):
            if cnt:
                for c in clusters_of_row.get(row, ()):
                    null[c] += cnt
        null /= n_r  # expected hits per k_ret-retrieval
        np.save(null_path, null.astype(np.float32))
        manifest["null_k"] = k_ret
        manifest["null_fingerprint"] = fp["response"]
        print(f">>> retrieval null built over {n_r} pseudo-cruxes "
              f"(max expected hits {null.max():.1f})", flush=True)

    refused = sorted(s["cluster"] for s in summaries if s.get("refused"))
    manifest.update({"k_clusters": k, "trigger_attrs": len(trigger),
                     "response_attrs": len(response),
                     "refused_summaries": refused,
                     "summary_provider_override": provider,
                     "summary_model": summary_model, "embed_model": str(cfg.embed_model),
                     "index_git_sha": git_sha(), "index_timestamp": timestamp()})
    (d / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f">>> index built: {k} clusters over {len(trigger)} trigger attrs")
    if refused:
        print(f"!!! {len(refused)} summaries refused by the provider after retries: "
              f"{refused}")
        if not accept_refusals:
            raise SystemExit(
                "refusals present (recorded as summary=null in "
                "cluster_summaries.jsonl + manifest.refused_summaries). Rerun to "
                "retry them, regenerate the stage with a different --summary_model "
                "for a homogeneous artifact, or pass --accept_refusals to keep the "
                "holes.")

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
