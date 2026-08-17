# Feature discovery — difficult-advice reasoning traces (k=150)

What the difficult-advice corpus's *reasoning* actually contains, discovered rather than
declared: Sonnet 5 reads one trace at a time and writes free-text features, the features are
embedded and clustered, and each cluster is named from a sample of its members. A replication
of the LessWrong pipeline in post `WAZWA6FPQvH8okouJ`, run on 2026-08-12.

**2,202 traces → 42,514 feature instances → 33,918 unique → 150 clusters.**

## What is here

| file | what it is |
| --- | --- |
| `report.md` | the full result: every cluster with prevalence and sample features, the redundancy audit, the keyword probes |
| `dashboard.html` | the same, browsable |
| `clusters.json` | cluster id → label, prevalence, trait mix, sample features (+ `meta` with the run's models and seed) |
| `feature_cluster_map.json` | feature string → cluster id — what every downstream analysis joins on |
| `features.jsonl` | per-trace feature lists (`scenario_id`, `trait_id`, `features[]`) |
| `unique_features.txt` | the deduplicated feature vocabulary, in embedding row order |
| `feature_counts.json` | how often each feature string was produced |
| `report_audit.json`, `embed_meta.json` | redundancy/probe numbers and the embedding sanity check |
| `derived/` | analyses built on the clusters (see below) |

**Not here: `embeddings.npy` (265MB).** It lives with a complete copy of the above at
`matboz/2026-08-12-difficult-advice-feature-discovery` on Hugging Face, per the repo's
data policy. It is the one artifact that is both bulky and exactly reproducible — the
feature list plus the pinned embedding model regenerate it.

## Reading the numbers

Two caveats belong next to any figure taken from here, both established in `report.md`:

**k=150 is a resolution setting, not a behaviour count.** 84 cluster pairs sit at centroid
cosine ≥ 0.90 ("Structured ethical reasoning across dilemmas" vs "Structured step-by-step
ethical reasoning", 0.949). k-means keeps splitting one dominant house style.

**A cluster label is not proof a behaviour is absent.** k-means absorbs small distinctive
themes into large bland ones, so `report.md` also probes the raw feature strings directly.
The probe worth knowing: **evaluation awareness appears in 9.1% of traces** (201 rows) — a
property of the training data and a confound for any OOD-eval improvement attributed to the
recipe. Probes are word-boundary regexes because substring matching inflated this figure to
10.6% (matching "child custody evaluation") and persona/identity to 17.2% (matching
"personal", "tradesperson") before the fix.

## Derived analyses (`derived/`)

| file | question it answers |
| --- | --- |
| `odcv_rollout_clusters.md` | which of these clusters appear in the trained model's ODCV rollouts, assigned to the *existing* centroids so rollout and corpus prevalence are comparable |
| `mixture_c6_membership.md` | which rows of the published training mixture carry C6 (the masked ablation arm's target) |
| `mixture_c30_c79_membership.md` | the same for the harm-risk clusters |
| `harm_risk_instances.md` | every trace in C30 and its centroid neighbourhood, per threshold tier |

## Regenerating

```
scratch/feature_discovery/extract_features.py    # 1  features per trace
scratch/feature_discovery/prepare_features.py    # 2a unique vocabulary
scratch/feature_discovery/runpod_embed.py        # 2b embed on a rented GPU
scratch/feature_discovery/cluster_and_name.py    # 3  cluster + name
scratch/feature_discovery/build_report.py        # 4  audit + dashboard
```

Downstream: `scratch/mixture_cluster_membership.py`, `scratch/find_harm_risk_instances.py`,
`scratch/odcv_cluster_assign.py`, `scratch/mask_cluster_spans.py`.
