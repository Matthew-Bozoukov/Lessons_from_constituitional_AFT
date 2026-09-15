<!-- ABOUTME: Direct review of 100 uniformly sampled, unfiltered upstream smol-summarize rows. -->
<!-- ABOUTME: Separates source faithfulness from instruction compliance and documents screening limitations. -->
# Unfiltered random audit — 2026-09-08

Read the complete system instruction, input and reference answer of **100 randomly
sampled upstream rows**, without applying any of the earlier content filters.
No dataset rows, instructions, or answers were changed. No model API calls or uploads.

## Results

| Directly classified input | Sampled | Source-faithful content | Faithful and complete | Obeys instruction | Passes all checks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Correspondence | 73 | 62 | 62 | 52 | 45 |
| Article-like | 27 | 21 | 18 | 0 | 0 |
| Total | 100 | 83 | 80 | 52 | 45 |

Article-like comprises 25 news reports, one opinion article and one news fragment.
**All 27 exceed the three-sentence maximum on direct reading; 26 also use prohibited
personal/possessive/reflexive pronouns.** Many summarize their sources well, but do not
obey the instruction paired with them. This is not evidence that article summaries
are generally factually incorrect.

Among correspondence, 18 clearly violate the pronoun instruction and three have
borderline instruction compliance (two wordy one-sentence answers, one missing the
deadline that is the summary's focus). Borderline rows are not counted as passes.
Eleven correspondence answers have unsupported details or other faithfulness errors;
these overlap with instruction failures. A more lenient treatment of the three
borderlines would raise the overall pass count from 45 to 48, still all correspondence.

“Source-faithful” evaluates the content that is present; three otherwise faithful
article answers are unfinished. The stricter “faithful and complete” column excludes
those. These are direct assistant judgments, not externally verified factual labels.
Historical source articles and fictional/persona correspondence are evaluated against
their supplied text, not presumed to describe current or real-world events.

## Concrete examples

- **Sample 39 / upstream 8130:** the cemetery-planning article has an accurate,
  pronoun-free answer, but **eight sentences**. This is an instruction failure,
  not a bad account of the source.
- **Sample 7 / upstream 1720:** the Crystal Palace–Tottenham summary preserves the
  result, scorers and timing, but has seven sentences and “his.”
- **Sample 37 / upstream 85383:** the Barlow article says the pregnancy was announced
  in February and the baby's sex in April; the summary says the pregnancy was
  announced in April. It also exceeds the sentence limit and uses pronouns.
- **Sample 90 / upstream 93322:** Emily writes “I'd love to see those Spanish sources,”
  but the reference says “Emily agrees to share Spanish sources.” This reverses the
  direction of sharing. Emails are not uniformly correct either.
- **Sample 86 / upstream 34733:** the reference invents an Industrial Revolution
  context and turns a proposed approximate 2pm meeting into a confirmed meeting.

## What this changes about the earlier conclusion

The random audit supports a substantial **instruction-compliance problem** for
non-email references. It does **not** prove that only five suitable non-email rows
exist in the entire dataset, or that no better unchanged selection is achievable.
Zero passes in 27 article-like examples is too small a sample to establish a tiny
full-population ceiling.

The old sentence regex also overcounts abbreviations such as `U.N.` and `H.W.`:
sample 55 has eight actual sentences but the regex returns eleven. Sample 99 also
overcounts `H.W.`. These errors do not change the decisions in this sample because
the answers still exceed three real sentences. They nevertheless weaken the previous
full-scan claim. A future candidate-discovery pass should audit near-boundary
rejections with sentence-aware parsing and direct reading, not assume a regex
rejection is definitive. The old 30-word limit was also an extra heuristic, not an
explicit upstream instruction; this audit did not apply that numerical cutoff.

## Reproducibility and evidence

- Dataset: `HuggingFaceTB/smoltalk`, `smol-summarize`, train.
- Revision: `5feaf2fd3ffca7c237fc38d1861bc30365d48ffa`.
- Population: 96,356 rows; `random.Random(20260909).sample(range(96356), 100)`.
- Uniform sampling without replacement; seed fixed before reading sampled answers.
- No removal of existing mixture inputs, email-like texts, duplicates by content,
  pronouns, long answers, fragments, or earlier rejections before sampling.
- All 100 full original rows: [sample.jsonl](/Users/jamie/Projects/lasr/output/2026-09-08-smol-summarize-random-audit/sample.jsonl).
- Per-row judgments and reasons: [reviews.json](/Users/jamie/Projects/lasr/output/2026-09-08-smol-summarize-random-audit/reviews.json).
- Counts and validation: [report.json](/Users/jamie/Projects/lasr/output/2026-09-08-smol-summarize-random-audit/report.json).
- Sampling command, exact input/sample hashes and git SHA: `run_meta.json` beside them.
- Editable judgment record: `scratch/smol_summarize_random_audit_reviews.json`.

```sh
uv run --no-sync python -m scratch.replace_smol_summarize random-audit --count 100 --seed 20260909 --offset 0 --batch-size 10
uv run --no-sync python -m scratch.replace_smol_summarize audit-report --out-root output/2026-09-08-smol-summarize-random-audit
```

Increment `--offset` by ten to display the next batch. Existing sample files cannot
be silently replaced with a different sample. Aggregation verifies all 100 reviews
against their sampled indices, exact upstream messages, pinned input hash, and seeded
sampling order; missing reviews cause failure. The training mixtures were not changed.
