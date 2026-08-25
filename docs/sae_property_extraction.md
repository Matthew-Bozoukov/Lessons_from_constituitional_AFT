<!-- ABOUTME: Design doc for SAE-based property extraction (dataset diffing, correlations, clustering) -->
<!-- ABOUTME: over our synth corpora and model organisms, after arXiv:2512.10092. Nothing here is built yet. -->

# SAE property extraction — design

Status 2026-08-20: built and E1 complete (see §8 for what runs next and LOG 2026-08-19 for
E1 results); originally written 2026-08-19 as design-only. Based on *Interpretable Embeddings with Sparse
Autoencoders: A Data Analysis Toolkit* (arXiv:2512.10092, Jiang, Sun, Dunlap, Smith,
Nanda — Neel Nanda's group). This is the public description of the diffing method behind
the (unshareable) black-box model-diffing tool we were pointed to on 2026-08-01; the one
extension flagged as worth reproducing ourselves is **seeding diffing hypotheses from eval
transcripts**, which §6 (E3) covers.

## 1. The method, compressed

Pass every document through one **reader LLM**, capture hidden states at one layer, encode
them with a **pretrained SAE**, and **max-pool each latent over tokens**. The result is one
vector per document whose ~65k dimensions each carry a human-readable label ("offensive
request from the user", "step-by-step reasoning"). Binarize (latent "present" when it fires
on >1 token) and every corpus becomes a document × concept boolean matrix. Everything else
is counting:

| Task | Mechanic | Paper result |
|---|---|---|
| **Diffing** | per-corpus latent frequency; subtract; top-200 latents above 0.03 freq-diff; relabel; LLM-summarize into ≤10 hypotheses; LLM judge verifies each per-doc → verified freq diff | finds *bigger* verified differences than LLM baselines at 2–8× lower cost; more granular; wins grow in multi-model comparisons |
| **Correlations** | NPMI(i,j) over binarized latent pairs; keep high NPMI (>0.5–0.6) + low label semantic similarity (<0.2, i.e. "interesting"); relabel + judge-verify on a subset → NPMI_verified | recovers injected correlations LLMs miss under shuffling (LLM: 1/10–9/10 across reshuffles); found Tulu-3's learned "I hope it is correct" trigger |
| **Clustering** | binarize → Jaccard → spectral clustering; *targeted*: pre-filter to ~500 latents whose labels match a keyphrase ("step by step reasoning"); describe clusters by in-vs-out diffing | clusters along a chosen axis (reasoning style, not topic), which dense embeddings cannot be steered to do |
| **Retrieval** | query → sentence-embed latent labels → top-100 latents → optional LLM rerank → weighted activation dot-product (rank-softmax weights, latents normalized by 90th-pct nonzero activation) | matches/beats dense-embedding baselines on *property* (not semantic) retrieval, e.g. "model stuck in a repetitive loop" |

Two case studies matter to us directly:

* **Model evolution** (their §5.1): responses from 5 OpenAI generations on 1k chat
  prompts; keep latents whose frequency rises monotonically across generations; relabel
  top 50 (20 activating samples from the newest model, 20 non-activating from the oldest);
  judge-verify. Emergent qualities surface ("nuanced explanations acknowledging
  trade-offs", "personalized follow-ups"). Our analog swaps the time axis for the
  **training-dataset axis** across model organisms.
* **Dataset debugging** (their §5.2): prompt-latent × response-latent NPMI on Tulu-3 SFT
  data finds math/lists/LaTeX prompts correlated with "I hope it is correct" responses;
  splitting the corpus by the property and re-diffing isolates five candidate features;
  a counterfactual prompt grid then proves the *model* learned the trigger. This
  find-split-diff-then-test loop is a template for auditing our own synth corpora.

Reader/SAE facts worth pinning: paper's main runs use Goodfire's Llama-3.3-70B layer-50
SAE, d=65536, texts kept <2048 tokens. Appendix H reproduces on **Gemma Scope** (Gemma-2
pretrained SAEs + pre-existing Neuronpedia labels): 65k width works, **16k width fails
correlations outright** (the needed latents don't exist), and on retrieval SAE width
matters more than reader size, with Gemma-9B-131k ≈ Llama-70B on several datasets.
Appendix J: latent labels are fairly robust across domains, but relabeling on the studied
corpus helps most when that corpus is far from the SAE's training distribution. The reader
does NOT need to be the model under study — one reader interprets *text*, whoever wrote it.

## 2. Why this fits the project

The paper agenda (see LOG 2026-08-18, `src/properties/`) needs a List of Properties whose
prevalences are measured, compared, and then ablated. We currently have four producers
(feature_discovery, turf, less, trace_clusters). SAE embeddings slot in as a **fifth
producer** with three abilities the others lack:

1. **Cheap exhaustive hypothesis space.** feature_discovery pays one LLM call per trace to
   invent vocabulary; the SAE labels ~65k concepts in a single forward pass and the
   embeddings are *reusable* across every later comparison. The 2026-08-17 verify gate
   showed peer-critique separable from its control at BoW AUC 0.9973 — a scalar that says
   corpora differ but not *how*. Diffing produces the missing "on what" list.
2. **Cross-corpus symmetry.** Diffing DA vs peer-critique vs courtroom needs the same
   hypothesis space applied to all corpora; per-trace LLM vocabularies drift, latent
   frequencies don't.
3. **Unsupervised cross-channel correlations.** TURF starts from a hypothesis; NPMI over
   query-latents × response-latents finds correlated pairs nobody hypothesized — exactly
   how Tulu-3's artifact was found, and our synth pipelines (one generator, one rewrite
   stage) are precisely the kind of process that plants such artifacts.

Ready-made validation targets — the method must rediscover these known properties before
any novel finding is trusted:

* DA data is prose- and scenario-repetitive and **over-mentions the constitution
  explicitly** (manual finding, 2026-08-12).
* Self-reflect data contains occasional stray Chinese characters (same audit).
* Peer-critique vs control separability AUC 0.9973; post-action-retrospection 0.96
  (`ablation/verify.py`, 2026-08-17) — whatever the BoW classifier keyed on, diffing
  should name it.

And on the organism side, the qualitative question is already on the team's list (tasks of
2026-08-07): diff not base-vs-finetuned but **organism-vs-organism** — DA-trained vs
model-evaluates-model-trained — "to get a qualitative feel for how the datasets actually
change the model, rather than just misalignment-goes-down." One observed hypothesis is
waiting to be quantified: DA-trained models seem to refuse in *less preachy, more
reasoned* ways, reasoning about consequences rather than reciting "as an AI I can't".

## 3. Inputs and outputs

### Inputs

Two source kinds, both already normalized by `src/properties/sources/` into `Record`s with
the three channels (`query`, `reasoning`, `response`) — the SAE stage embeds **each
channel separately**, so a discovered property localizes to where it lives (a
constitution-mention in the *response* is a different finding from one in the *query*):

1. **Training corpora** (via `mixture_rows`): HF repo + revision, e.g.
   `LASR-Callum/2026-08-13-difficult-advice-v2` (DA),
   `LASR-Callum/2026-08-14-peer-critique` (PC),
   `LASR-Callum/2026-08-14-courtroom` (CR),
   `LASR-Callum/2026-08-13-post-action-retrospection` (PAR),
   plus the table2 instruction mixture as background/control.
2. **Organism outputs** (via `agentic_rollouts` / `odcv_rollouts`, or fresh generations):
   the t2-9284 arm family (da716 / courtroom716 / synthdoc716 variants), memself /
   selfreflect arms, `tulu100` control, and base — either responses generated on a fixed
   prompt set (1k LMSYS prompts; the lmsys eval infra already serves and samples these)
   or existing eval transcripts (ODCV `messages_record.txt`, agentic-misalignment stitched
   rollouts).

### Intermediate artifact (cached, reusable — this is where the cost saving lives)

Per (corpus, channel): a sparse doc × latent activation matrix, max-pooled over tokens
(chunked at the reader's context and max-pooled across chunks), plus its binarization and
a latent-frequency vector. Pushed to HF per the data policy as
`<date>-sae-embeddings-<corpus>` with the standard card; local mirror under
`output/sae_properties/<run>/`. Every later diff/correlation/cluster run is arithmetic on
these matrices plus a few hundred LLM calls — no reader-GPU needed.

### Outputs

1. **Diff report** per comparison (markdown mirror + jsonl): ≤10 verified hypotheses,
   each with SAE-estimated and judge-verified frequency per corpus, the contributing
   latent ids, and 2–3 activating excerpts. Example row shape:

   ```json
   {"hypothesis": "Response quotes or paraphrases a written principles document",
    "latents": [30117, 4482], "channel": "response",
    "freq": {"difficult_advice_v2": 0.61, "courtroom": 0.08, "table2_bg": 0.01},
    "verified_freq": {"difficult_advice_v2": 0.55, "courtroom": 0.06},
    "examples": ["...as my constitution notes..."]}
   ```
2. **Property rows** in `output/properties/properties.jsonl` via the registry — source
   `sae_diff`, id `sae_diff:<run>:<latent-or-hypothesis-key>`, detector produced by
   `shared/interpret.py` from the (re)labeled latent + activating excerpts, prevalence =
   judge-verified frequency, `support` carrying latent ids and raw freq-diffs. From there
   the existing ablation machinery (mask/filter/rewrite/regenerate + verify) applies
   unchanged.
3. **Correlation report**: latent pairs (or cross-channel pairs) with NPMI, verified NPMI,
   labels, co-activating excerpts.
4. **Cluster report** (secondary): targeted clusters along a named axis with per-cluster
   in-vs-out diff descriptions.

## 4. Which pretrained SAE

| Option | What | For | Against |
|---|---|---|---|
| **Gemma Scope 9B (recommended)** | `google/gemma-scope-9b-pt-res`, residual stream ~mid-layer, **131k width**, canonical L0; loaded via `sae-lens`; labels from Neuronpedia's public autointerp set | open weights, $0/token, validated in the paper's Appendix H, 131k ≈ Llama-70B on retrieval; runs on one 24–80GB GPU | pretrained (not IT) reader; labels made on pretraining text → relabel top latents on our data before trusting |
| Gemma Scope 2B, 16k | same family, tiny | **smoke tier**: CPU/MPS-runnable, finds every scale-independent bug in minutes (gotcha 7) | 16k width is *known* to miss correlation-relevant latents — never promote findings from it |
| Goodfire Ember API | hosted Llama-3.3-70B layer-50 SAE, the paper's main setup; labels included | zero GPU, chat-distribution SAE (LMSYS) suits assistant text | new paid dependency + key; 2048-token/doc limit; our data leaves the machine |
| Llama Scope / others | open Llama-3.1-8B SAEs | fallback | fewer public labels; not the paper's validated path |
| Train our own on Qwen3.6 | — | — | defeats the point (cheap, pretrained); reader ≠ model-under-study is a *feature* |

Decision: build against `sae-lens` with the SAE id in config, so the 2B smoke tier and the
9B real tier are one config swap; keep Goodfire as an escape hatch if Gemma-Scope latents
prove too pretraining-flavored for assistant-style properties (the paper's LMSYS-trained
SAE is the best match for chat text — Appendix J shows generalization is best near the
SAE's training distribution).

Serving note: **vLLM cannot produce SAE activations** — the embed step is a plain
HF-transformers batched forward with a residual-stream hook (no generation, no KV cache).
It therefore runs where the GPU is (Option A pattern), not through `run_eval.py`'s
serving; everything downstream of the cached embeddings runs anywhere.

## 5. How building it looks

Home: `scratch/sae_properties/` (default for new AI code), promoted piecewise. The
`src/properties` integration is a boundary from day one, same pattern as
feature_discovery: the run directory's artifacts are the interface, and
`src/properties/producers/sae_diff/__init__.py` only reads them.

```
scratch/sae_properties/
  embed.py       corpus (HF repo+rev, channel) -> sparse doc×latent matrix + freqs   [GPU]
  diff.py        freq tables -> top-K latent diffs (target vs max-of-others)
  relabel.py     latent -> label, via 10-20 activating + non-activating excerpts     [LLM]
  hypotheses.py  labeled diffs -> <=10 distinct hypotheses                           [LLM]
  verify.py      hypothesis × doc sample -> judged presence -> verified freqs        [LLM]
  correlate.py   binarized matrices -> NPMI pairs, label-similarity filter
  rundir.py      what a run directory holds (mirrors feature_discovery's)
configs/properties/sae_diff.yaml   sae id + layer, channels, thresholds (0.03 freq-diff,
                                   NPMI>0.5, sim<0.2), judge model, sample sizes
```

All LLM calls (relabel, summarize, verify) go through `src/endpoints/openrouter.py` with a
cheap judge model in config. Fail fast on unlabeled latents rather than silently keeping a
Neuronpedia label that never saw assistant text: top diffed latents are ALWAYS relabeled
from our own activating excerpts (the paper does the same for its headline results).

Phases, each ending in something inspectable:

1. **Spike (half a day).** 2B-16k SAE via sae-lens, 200 rows DA + 200 rows PC, response
   channel, laptop. Deliverable: top-30 freq-diff latents with Neuronpedia labels.
   Go/no-go: do the known artifacts show a signal at all.
2. **Embed at scale (a day, one GPU pod, <$10).** 9B-131k over the four synth corpora +
   background mixture, all three channels; cache to HF continuously (pod disk is not
   storage). Throughput measured on one batch *before* committing the pod (gotcha 7's
   stopwatch rule). Rough size: ~5 corpora × ~10k docs × ~1.5k tokens ≈ 75M reader
   tokens — an hour-scale job, not a day-scale one.
3. **Diff pipeline (a day, API-only).** diff → relabel → hypotheses → verify; markdown
   report per comparison. Gate: rediscovers the three known ground truths (§2) before we
   read anything novel from it.
4. **Producer (half a day).** `sae_diff` producer emits Property rows with detectors;
   `discover.py` runs it blind like the other four.
5. **Organism diffing (1–2 days incl. generation).** E2/E3 below; generation reuses the
   existing serving/eval plumbing.
6. **Correlations + retrieval extras.** E4 and the detector pre-filter.

## 6. Experiments

* **E1 — corpus diffing (the first result).** Diff DA-v2 vs PC vs CR vs PAR vs table2
  background, per channel. Output: one table per corpus of its distinguishing verified
  properties → straight into `properties.jsonl` as ablation candidates for Fig 3. Also
  the validation gate (§2's known artifacts).
* **E2 — organism diffing (the "OpenAI generations" analog, dataset axis).** Fixed 1k
  LMSYS prompt set → responses from base, tulu100, ft_da, ft_courtroom, memself,
  selfreflect (modes matched per the eval framework's thinking-stamp rules) → diff each
  arm against tulu100 (isolates the 716-row synth share) and against base. Expected
  headline: "training on X makes the organism do Y more", per dataset — the qualitative
  companion to the misalignment numbers.
* **E3 — eval-transcript-seeded diffing.** Same machinery over existing ODCV /
  agentic-misalignment rollouts (the `*_rollouts` sources carry outcomes): diff arm vs
  arm, and within one arm diff violation vs non-violation rollouts. First hypothesis to
  test: DA arms refuse in less preachy, more consequence-reasoned ways. This reproduces
  the flagged "seed hypotheses from eval transcripts" extension.
* **E4 — corpus auditing via correlations (Tulu-3 loop on our own data).** Query-latent ×
  response-latent NPMI within DA-v2 (and within the full training mixture): does scenario
  type predict response boilerplate? Does the rewrite stage plant a phrase-level trigger
  the way Tulu's data planted "I hope it is correct"? Any hit follows the paper's loop:
  split corpus by property → re-diff → counterfactual prompts → measure the organism.

Later / cheap add-ons: targeted clustering of DA scenarios to check trait balance
(`balance_by: trait_id`) against what the text actually contains; SAE retrieval as a
**detector pre-filter** — LOG 2026-08-18's open cost concern is one judge call per record
per property, and ranking records by the property's latent activations first means the
judge only reads the plausible ones (paper: retrieval matches dense baselines on exactly
this kind of implicit-property query).

## 7. Risks and honest limitations

* **Hypothesis space = SAE training distribution.** Gemma Scope saw pretraining text;
  "assistant quotes its constitution" may have no clean latent. Mitigations: relabel on
  our excerpts, verify step measures *real* frequencies (a fuzzy latent can still seed a
  sharp hypothesis), Goodfire fallback whose SAE saw chat.
* **Feature absorption / split latents** (paper's own caveat): a property can hide across
  several latents; the summarize step exists to merge them, and the LLM-judge
  verification is what makes any number quotable.
* **Our corpora are small** (~10³–10⁴ docs), so the paper's 2–8× cost edge over an LLM
  annotator matters less here; the durable advantages for us are the fixed shared
  hypothesis space, reusable embeddings across many pairwise comparisons (5 corpora ×
  3 channels + N organisms), and correlation-finding reliability, where LLMs fail
  outright. For any headline finding, running the LLM-baseline diff (batched
  describe-then-cluster, the paper's LLM-C baseline) is affordable and worth it — a
  methods comparison on our own data is itself paper material.
* **Long reasoning traces** exceed any single reader window: chunk + max-pool is the
  documented approach, but max-pool over many chunks inflates latent presence; report
  per-channel doc-length stats next to any frequency, and keep the >1-token binarization
  threshold configurable.
* **Verified numbers only.** SAE frequencies are estimates from an imperfect labeler;
  everything quoted in the paper/LOG must be the judge-verified frequency, with the same
  Wilson-interval discipline `ablation/verify.py` already enforces.

## 8. Next runs (planned 2026-08-20, post-E1 — not yet started)

E1's embedding caches (`output/sae_properties/e1_70b/datasets/`, mirrored to HF) make three
of the paper's four capabilities runnable without re-embedding what we already have. Planned
package, one GPU session (~1.5h, ~$6 GPU + ~$10 judge API), in priority order:

1. **Correlations — the Tulu-3 debugging loop** (paper §4.2 + §5.2). NPMI between latent
   pairs, keeping high-NPMI / low-label-similarity pairs, judge-verified. Within-corpus on
   the existing response caches is GPU-free; the high-value version is **query-latents ×
   response-latents** (what in the prompt predicts what in the response), which needs the
   query channel embedded (~20 min GPU). Any hit follows the paper's loop: split the corpus
   by the property → re-diff the splits → counterfactual prompts → measure the trained
   organism. That last step upgrades a data artifact into a learned-behaviour finding.
   Wrapper to write: `correlate.py` driving `interp_embed`'s NPMI path (the vendored
   package has the primitives; no paper-script CLI exists for it, unlike diffing).
2. **Table2 contrast — the constitution-mention test.** Embed ~1000 rows of the plain
   instruction mixture (response channel), then diff DA vs table2. Answers what E1
   structurally could not (all four synth corpora share the constitution grounding, so
   shared properties cancel): what DA adds relative to *normal* training data —
   constitution quoting, ethics register, refusal style. Config-only after the embed:
   add the corpus + set `diff.others=[table2_bg]`.
3. **Reasoning-channel E1.** Same four corpora, `embed.channels=[reasoning]`, same diff.
   Tests "less preachy, more consequence-reasoned" where it lives — the traces, which are
   what `thinking: true` training actually supervises alongside responses.
4. **Targeted clustering** (paper §4.3) — GPU-free, on the cached DA response embeddings:
   filter to advice/refusal-related latents (label-similarity to a keyphrase), spectral-
   cluster, describe clusters by in-vs-out diffing. Question: do DA responses collapse into
   a few stereotyped advice templates? Complements E1's domain-skew rows with *style*
   clusters, and cross-checks `balance_by: trait_id` against what the text contains.

Embed additions for the session: table2 response + query&reasoning channels for the four
synth corpora. Operational notes from E1 (LOG 2026-08-19): one reader load per process
(second load OOMs — resume-skip handles it), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
judge stays `gemini-2.5-flash` while 3.7-flash is capacity-capped on OpenRouter.

E2 (organism diffing on 1k LMSYS prompts) and E3 (eval-transcript-seeded diffing) remain
the rollout-side follow-ups (§6) — separate session, since they need arm generation first.
