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
inertia early-stop, seed 42; runs on cuda/mps/cpu), while assigning NEW attributes
to those clusters is cosine over normalised centroids (their `cluster_mapper.py`).
The paper specifies cosine only for crux→response retrieval (Eq. 3), which we follow.

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
