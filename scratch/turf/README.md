# TURF (adapted) — trace data properties behind behaviours we care about

Adaptation of TURF from *Chunky Post-Training* (arXiv 2602.05910) for the mock
narrative's Fig 3: given a **case** (a response we liked, e.g. a honeypot refusal),
find which **query + reasoning** properties of the training dataset co-occur with
training responses similar to what we liked. Design log: the 2026-08-13 discussion
(docs/LOG.md will carry results, not this README).

Key departures from the paper:
- **Trigger side = query + reasoning** (the paper's is query-only). Every datapoint
  contributes equally to the trigger clusters: 10 "The query..." + 10 "The reasoning...".
- **Polarity**: `satisfy` (attribute good behaviour) alongside the paper's `violate`.
- **Style guard**: the informed judge must not select cruxes that merely restate the
  dataset's own generation styles (styles.json, harvested from row metadata).
- The crux-selection prompt is reconstructed from the paper's prose (never released);
  the query extractor is SURF's released prompt verbatim.

Deliberately faithful to SURF's released code (github.com/seoirsem/SURF), since the
paper never specifies these: clustering is their `_run_kmeans` ported to torch
(squared-Euclidean Lloyd's, empty clusters keep their centroid, 20 iters, <0.1%
inertia early-stop, seed 42; runs on cuda/mps/cpu); attribute extraction samples at
their temperature 1.0 (dataset and case side alike, so both live in one
distribution); cluster summaries use their prompt verbatim (channel prefix
parameterised) at temperature 0.0 from the top-50 closest members (their top-100,
halved); embeddings are stored fp32 as they store them; rows reduce to their FIRST
user turn + FIRST assistant turn (their `_extract_first_turn`; system prompts and
later turns ignored). The paper itself mandates cosine in two places, and we follow
both: crux→response retrieval (Eq. 3) and assigning a case's attributes to their
nearest cluster (Eq. 5) — the latter also matching SURF's `cluster_mapper.py`.
One knowing paper deviation: hit-counting credits each distinct cluster of a
retrieved row once, rather than Eq. 4's literal slot-index pairing — which cannot
extend to our 20-attribute (query+reasoning) trigger side anyway.

## Pipeline

```bash
# offline: dataset (synth model-agnostic HF format) -> attributes -> index
uv run python scratch/turf/extract.py --dataset <hf-id> --file stage_7_sft.jsonl \
    --out output/turf/<name> [--limit 200]
uv run python scratch/turf/index.py --dir output/turf/<name> --k 1000 [--push]

# online: one case + one rubric -> trace
uv run python scratch/turf/cases.py list --run output/agentic_misalignment/<ts>
uv run python scratch/turf/cases.py add --run <same> --id <case-id> --out case.json
uv run python scratch/turf/trace.py --case case.json \
    --rubric scratch/turf/rubrics/oversight_refusal.yaml \
    --index output/turf/<name> --polarity satisfy [--push]
```

HF artifacts: `--push` on index.py -> `LASR-Callum/<date>-turf-index-<dataset>`;
on trace.py -> a subdir per run in the rolling `LASR-Callum/<date>-turf-traces`.

Rubrics describe the case's *behaviour* (principle, context, satisfy/violate), never
candidate properties — property discovery must stay real. One behaviour per rubric.

Validation strategy: deliberately deferred (2026-08-13).
