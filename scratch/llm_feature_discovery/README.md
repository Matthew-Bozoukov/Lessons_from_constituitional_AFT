# scratch/llm_feature_discovery/

Replication of the LessWrong "LLM-driven feature discovery" method (post `WAZWA6FPQvH8okouJ`)
on this project's reasoning traces. An autorater invents its own vocabulary for what a trace
does, instead of scoring it against axes we chose in advance, so it can surface behaviours no
schema anticipated. Findings: `docs/LOG.md`, entry 2026-08-12.

## Flow

Filenames carry no step numbers on purpose — the order below is what the data dependencies
require, not a fixed contract, and steps may be added or reordered.

```
stage_7_sft.jsonl (reasoning_content per row)
  │
  ├─ extract_free_text_features_per_trace.py     Sonnet, 1 trace at a time, no metadata
  │      → features.jsonl            {scenario_id, trait_id, features: [str]}
  │
  ├─ dedupe_features_to_unique_vocabulary.py     embed each string once, keep counts
  │      → unique_features.txt, feature_counts.json
  │
  ├─ embed_unique_features_on_rented_gpu.py      Qwen3-Embedding-8B on a RunPod A6000
  │      → embeddings.npy (fp16, L2-normalised), embed_meta.json
  │
  ├─ cluster_and_name_feature_embeddings.py      k-means OR UMAP+HDBSCAN, + Sonnet naming
  │      → clusters.json, feature_cluster_map.json, report.md
  │        (+ umap_coords.npy in hdbscan mode)
  │
  ├─ audit_clusters_and_build_dashboard.py       redundancy + keyword probes + HTML
  │      → report_audit.json, dashboard.html, report.md (appended)
  │
  └─ compare_clusterings_and_check_stability.py  gate one clustering against the other
         → clustering_comparison.json, clustering_comparison.md
```

Every script reads and writes the same run directory, `output/feature_discovery/<timestamp>/`,
so they are wired together by that directory's contents rather than by an orchestrator.
Extraction is resumable: rerun with the same `--out-dir` and it skips scenario ids already in
`features.jsonl`.

`feature_extraction_and_naming_prompts.py` holds the two prompts, both verbatim from the post.
Do not reword them — the point of the replication is that the post's prompt, not a schema of
ours, decides what a "feature" is. The JSON output contract appended to the extraction prompt
is ours and is a deliberate deviation.

## Run it

```bash
uv run python scratch/llm_feature_discovery/extract_free_text_features_per_trace.py \
    --input output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl --smoke

uv run python scratch/llm_feature_discovery/dedupe_features_to_unique_vocabulary.py \
    --features output/feature_discovery/<ts>/features.jsonl

uv run python scratch/llm_feature_discovery/embed_unique_features_on_rented_gpu.py create_pod
uv run python scratch/llm_feature_discovery/embed_unique_features_on_rented_gpu.py \
    push_features --pod <id> --features output/feature_discovery/<ts>/unique_features.txt
uv run python scratch/llm_feature_discovery/embed_unique_features_on_rented_gpu.py \
    fetch_embeddings --pod <id> --out-dir output/feature_discovery/<ts>
uv run python scratch/llm_feature_discovery/embed_unique_features_on_rented_gpu.py \
    terminate_pod --pod <id>          # not optional; the pod bills by the second

uv run python scratch/llm_feature_discovery/cluster_and_name_feature_embeddings.py \
    --run-dir output/feature_discovery/<ts> --k 150

uv run python scratch/llm_feature_discovery/audit_clusters_and_build_dashboard.py \
    --run-dir output/feature_discovery/<ts>
```

## Clustering: k-means or UMAP+HDBSCAN

`--cluster` picks how the embeddings are grouped; everything either side is identical, so
the two are an A/B over one embedding set rather than two pipelines.

| | `kmeans` (default) | `hdbscan` |
| --- | --- | --- |
| resolution knob | `--k` | `--min-cluster-size` |
| cluster count | chosen | discovered |
| every feature clustered | yes | no — low-density features become noise |
| memory | streams the memmap | ~0.5 GB resident (UMAP's k-NN graph needs it whole) |

Run the pair against the **same finished run directory** — the embeddings already exist, so
the only cost is the naming calls — and copy the directory first so both `clusters.json`
files survive to be compared. `min_cluster_size ≈ len(unique_features) / k` (~220 at
k=150) is the like-for-like setting.

```bash
cp -r output/feature_discovery/<ts> output/feature_discovery/<ts>_hdbscan
uv run python scratch/llm_feature_discovery/cluster_and_name_feature_embeddings.py \
    --run-dir output/feature_discovery/<ts>_hdbscan \
    --cluster hdbscan --reduce umap --min-cluster-size 220

uv run python scratch/llm_feature_discovery/compare_clusterings_and_check_stability.py \
    --kmeans-dir output/feature_discovery/<ts> \
    --hdbscan-dir output/feature_discovery/<ts>_hdbscan
```

The gate answers three questions before anything is read off the new clusters: did UMAP
keep the geometry (nearest-neighbour overlap, plus the stored embedding sanity probe where
the run has one), do the two labelings agree (ARI/AMI), and is HDBSCAN stable across
`n_neighbors` and seeds. **If the two agree closely, the published k=150 numbers were
sound — say so and keep them.**

### The noise contract

HDBSCAN labels low-density features `-1`. Those features are left **out** of
`feature_cluster_map.json` entirely, because every consumer averages a cluster's members
into a centroid and a `-1` "cluster" would get a meaningless one that then attracts eval
features. They are counted, not hidden: `clusters.json` carries `n_noise_features` and
`noise_instances`, `report.md` states both as a share, and the dashboard has a card for
them. Consumers look features up with `.get`, so a feature that is not in the map
contributes to nothing rather than raising.

Cost for the 2,202-trace corpus run: ~$18 OpenRouter + ~$0.30 RunPod. Prompt caching does
**not** engage — the post's prompt is ~426 tokens, under Anthropic's 1024-token minimum
cacheable prefix.

## Who consumes the output

The clusters produced here are the vocabulary several downstream analyses are written in.
All of them read `output/feature_discovery/20260812_092119/`:

| script | what it does with the clusters |
| --- | --- |
| `scratch/find_harm_risk_instances.py` | lists every trace in the harm-risk cluster and its centroid neighbours |
| `scratch/mixture_cluster_membership.py` | joins a published training mixture's rows back to their clusters |
| `scratch/mask_cluster_spans.py` | builds a span-masked training set that unsupervises one cluster's tokens |
| `scratch/odcv_cluster_assign.py` | reruns extraction + embedding over ODCV rollouts, then assigns to the **existing** centroids |

Because `odcv_cluster_assign.py` assigns to existing centroids rather than refitting, rollout
prevalence is directly comparable to training-corpus prevalence — that comparison is the
2026-08-15 LOG entry.

## Caveats worth re-reading before quoting a number

* **k is a resolution knob, not a count of behaviours.** 84 of 11,175 cluster pairs sit at
  centroid cosine >= 0.90 at k=150.
* **A cluster label is not evidence a behaviour is absent.** `Displays evaluations awareness`
  (89 occurrences) landed inside a generic cluster; only the keyword probe surfaced it.
* **Substring probes burned this analysis twice.** Every probe in
  `audit_clusters_and_build_dashboard.py` is a word-boundary regex, and any new needle must be
  read against its own matches before its number is quoted.
* **`meta["k"]` is not the cluster count on an HDBSCAN run** — it is the unused k-means
  knob sitting at its default. Read `meta["n_clusters"]`.
* **Reduced-space cosines are not comparable to full-dimensional ones.** UMAP compresses
  the space, so every cosine rises; only the ordering of the sanity probe carries over.
