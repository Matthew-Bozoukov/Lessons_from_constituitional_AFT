# scratch/llm_feature_discovery/

Replication of the LessWrong "LLM-driven feature discovery" method (post `WAZWA6FPQvH8okouJ`)
on this project's reasoning traces. An autorater invents its own vocabulary for what a trace
does, instead of scoring it against axes we chose in advance, so it can surface behaviours no
schema anticipated. Findings: `docs/LOG.md`, entry 2026-08-12.

## Input, output, and where the output goes

```
   Training data                                  (SFT jsonl, reasoning_content per row)
        │
        ▼
   ┌────────────────────────────────────────┐
   │  THIS MODULE                           │
   │  extract → dedupe → embed → cluster    │
   │  → audit → gate → export               │
   └────────────────────────────────────────┘
        │
        ▼
   properties.jsonl ───┐
                       │  TurF ──────────────────┐
                       ├──────────────────────► LIST OF PROPERTIES ──► Ablation ──► train M'' ──► Eval
                       │  LESS → ranking → LLM/ML ┘
   trace-level UMAP+clustering ──┘
```

**Input** — one SFT jsonl. Each row needs `messages[2].reasoning_content` (the private
reasoning to be described) and `metadata.scenario_id` / `metadata.trait_id`.

**Output** — `properties.jsonl` in the run directory: one named behavioural property per
line, with the share of the corpus's traces that exhibit it.

**Where it goes** — into the shared **List of Properties**, merged with the rows the other
producers emit. That list is what the ablation stage consumes, so `prevalence` is the field
that has to mean the same thing everywhere: *the share of traces in the same corpus
exhibiting the property*. Everything else in a row is advisory detail. `properties_meta.json`
carries the coverage this producer does **not** account for (the unclustered share), which a
merger should read before trusting the list.

## Run it

One entrypoint, one verb per stage:

```bash
uv run python -m scratch.llm_feature_discovery extract \
    --input output/synthdoc_v2/20260803_211524/stage_7_sft.jsonl --smoke

uv run python -m scratch.llm_feature_discovery dedupe --run-dir output/feature_discovery/<ts>

uv run python -m scratch.llm_feature_discovery embed create
uv run python -m scratch.llm_feature_discovery embed push      --pod <id> --run-dir <dir>
uv run python -m scratch.llm_feature_discovery embed status    --pod <id>
uv run python -m scratch.llm_feature_discovery embed fetch     --pod <id> --run-dir <dir>
uv run python -m scratch.llm_feature_discovery embed terminate --pod <id>   # bills by the second

uv run python -m scratch.llm_feature_discovery cluster --run-dir <dir> --min-cluster-size 220
uv run python -m scratch.llm_feature_discovery audit   --run-dir <dir>
uv run python -m scratch.llm_feature_discovery export  --run-dir <dir>
```

Every stage after `extract` takes `--run-dir` and communicates only through that directory,
so any of them can be rerun on its own against what an earlier one left behind. Extraction
is resumable: rerun with the same `--run-dir` and it skips scenario ids already in
`features.jsonl`.

Cost for the 2,202-trace corpus run: ~$18 OpenRouter + ~$0.30 RunPod. Prompt caching does
**not** engage — the post's prompt is ~426 tokens, under Anthropic's 1024-token minimum
cacheable prefix.

## What each stage does

| stage | in | out |
| --- | --- | --- |
| `extract` | SFT jsonl | `features.jsonl` — 10–20 free-text features per trace; the autorater sees one trace at a time and no metadata |
| `dedupe` | `features.jsonl` | `unique_features.txt`, `feature_counts.json` — embed each string once, keep the counts |
| `embed` | `unique_features.txt` | `embeddings.npy`, `probe_embeddings.npy`, `embed_meta.json` — Qwen3-Embedding-8B on a rented RunPod A6000 |
| `cluster` | `embeddings.npy` | `clusters.json`, `feature_cluster_map.json`, `umap_coords.npy`, `report.md` — UMAP + HDBSCAN, then LLM naming |
| `audit` | `clusters.json` | `report_audit.json`, `dashboard.html` — redundancy pairs + keyword probes |
| `gate` | two run dirs | `clustering_comparison.{json,md}` — geometry, agreement, stability |
| `export` | `clusters.json` | `properties.jsonl`, `properties_meta.json` |

## One file per stage, plus three shared pieces

A file only stands on its own if something outside it references it, or if it is one whole
stage.

| file | job | why it is its own file |
| --- | --- | --- |
| `rundir.py` | what a run directory holds, and how each artifact is read and written | every stage depends on it |
| `centroids.py` | cluster centroids | also imported by `find_harm_risk_instances.py` and `odcv_cluster_assign.py`, so the noise rule lives in one place |
| `prompts.py` | the two prompts, verbatim from the post | read by `extract` and by `cluster`, and kept apart so "do not reword this" has somewhere to be written down |
| `extract.py` | stages 1–2: trace → features → the unique vocabulary | one stage |
| `embed.py` | stage 3: the pod-side code, and renting the GPU that runs it | one stage |
| `cluster.py` | stage 4: UMAP + HDBSCAN, LLM naming, prevalence, report | one stage |
| `audit.py` | stages 5–6: redundancy, keyword probes, the gate, the dashboard | one stage |
| `properties.py` | stage 7: the hand-off schema | other producers and the merger read it |
| `__main__.py` | the CLI | the only file that knows what order stages run in |

Do not reword `prompts.py` — the point of the replication is that the post's prompt, not a
schema of ours, decides what a "feature" is. The JSON output contract appended to the
extraction prompt is ours and is a deliberate deviation.

## Clustering: UMAP + HDBSCAN

Mini-batch k-means at a fixed `k` was the original clusterer and was removed on 2026-08-18.
It forced every feature into a cluster and made the cluster count an argument rather than a
finding. HDBSCAN discovers the count and leaves low-density features unclustered;
`--min-cluster-size` is the resolution knob that `k` was (`≈ len(unique_features) / k`, so
~220 matches the old k=150).

Runs k-means produced — `output/feature_discovery/20260812_092119`, the published one — are
still fully readable: everything downstream takes the cluster count from
`meta["n_clusters"]` and falls back to `meta["k"]` for files written before that field
existed.

### The noise contract

HDBSCAN labels low-density features `-1`. Those features are left **out** of
`feature_cluster_map.json` entirely, because every consumer averages a cluster's members into
a centroid and a `-1` "cluster" would get a meaningless one that then attracts features
assigned against it. They are counted, not hidden: `clusters.json` carries
`n_noise_features` and `noise_instances`, `report.md` states both as a share, the dashboard
has a card, and `properties_meta.json` reports the uncovered share to whoever merges the
property list. Consumers look features up with `.get`, so a miss contributes to nothing
rather than raising.

### Gate a clustering before trusting it

Changing or retuning the clusterer changes the vocabulary every downstream analysis is
written in, so run the gate against a previous clustering of the same feature list first:

```bash
cp -r output/feature_discovery/<ts> output/feature_discovery/<ts>_hdbscan
uv run python -m scratch.llm_feature_discovery cluster \
    --run-dir output/feature_discovery/<ts>_hdbscan --min-cluster-size 220
uv run python -m scratch.llm_feature_discovery gate \
    --baseline-dir output/feature_discovery/<ts> \
    --candidate-dir output/feature_discovery/<ts>_hdbscan
```

Three questions, in the order they can disqualify a run: did UMAP keep the geometry
(nearest-neighbour overlap, plus the stored embedding sanity probe where the run has one),
do the two labelings agree (ARI/AMI), and is the clustering stable across `n_neighbors` and
seeds. **If the two agree closely, the published k=150 numbers were sound — say so and keep
them.** CPU only, no GPU and no API spend, but UMAP runs single-threaded when seeded, so the
default sweep is 12 fits; start with `--stability false`.

## Who else consumes the clusters

All of these read `output/feature_discovery/20260812_092119/`:

| script | what it does with the clusters |
| --- | --- |
| `scratch/find_harm_risk_instances.py` | lists every trace in the harm-risk cluster and its centroid neighbours |
| `scratch/mixture_cluster_membership.py` | joins a published training mixture's rows back to their clusters |
| `scratch/mask_cluster_spans.py` | builds a span-masked training set that unsupervises one cluster's tokens |
| `scratch/odcv_cluster_assign.py` | reruns extraction + embedding over ODCV rollouts, then assigns to the **existing** centroids |

`find_harm_risk_instances.py` and `odcv_cluster_assign.py` build their centroids with this
module's `centroids.py` rather than their own copies, so the noise rule lives in one place.
Because `odcv_cluster_assign.py` assigns to existing centroids rather than refitting, rollout
prevalence is directly comparable to training-corpus prevalence — that comparison is the
2026-08-15 LOG entry.

## Caveats worth re-reading before quoting a number

* **The cluster count is a resolution setting, not a count of behaviours.** 84 of 11,175
  cluster pairs sat at centroid cosine >= 0.90 in the k=150 run.
* **A cluster label is not evidence a behaviour is absent.** `Displays evaluations awareness`
  (89 occurrences) landed inside a generic cluster; only the keyword probe surfaced it.
* **Substring probes burned this analysis twice.** Every probe in `audit.py` is a
  word-boundary regex, and any new needle must be read against its own matches before its
  number is quoted.
* **Reduced-space cosines are not comparable to full-dimensional ones.** UMAP compresses the
  space, so every cosine rises; only the ordering of the sanity probe carries over.
