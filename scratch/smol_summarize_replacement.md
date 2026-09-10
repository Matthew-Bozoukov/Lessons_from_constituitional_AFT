# Reviewed smol-summarize replacement — 2026-09-08

## Final selection: Smol only, 980 correspondence + 4 news

User approved retaining the email-heavy Smol selection on 2026-09-08 rather than
switching to another source. Final local mixture:
`output/2026-09-08-smol-summarize-final/mixture.jsonl`.

Exactly 984 reviewed, unchanged upstream Smol rows: 980 correspondence and the four
news summaries. The technical Q&A addition from the diversity pass is replaced with
its previously reviewed email row. No P3, CNN/DailyMail, XSum, or OpenOrca rows are used.
Validation passed: 644/340 instruction split retained, all upstream messages exact,
all 9,016 other rows byte-identical, and all 1,119 reasoning-bearing messages preserved.
Published to the existing `dougalldeepmind/2026-09-08-nosynth-mix` repository at revision
`7e991f58e86eff0b0a9f15a54ebeddfffb5b14dd`. The Hub parent had 16 newer reasoning
messages than the local base, so only the 984 Smol rows were overlaid onto that parent;
all 9,016 non-Smol rows and all 1,135 reasoning-bearing messages on the Hub are preserved.
The published local copy is `output/2026-09-08-smol-summarize-final/hf-revision/mixture.jsonl`.
Download verification matched SHA-256
`de8c2a526de36dc214e1899df681a83c5612a062593a26ebfafcf49bfdaad219`.
The dataset card and audit files were included in the same atomic commit. No training
config was changed. Earlier artifacts remain; their 1,119-trace counts describe the
older local base, not the published revision.

Reproduce with `uv run --no-sync python -m scratch.replace_smol_summarize diversify --news-only`
(refuses to overwrite an existing dated output directory).

## Earlier: conservative diversity pass

New local mixture: `output/2026-09-08-smol-summarize-diversity/mixture.jsonl`.
The earlier replacement and the original mixture remain available, unchanged.
No upstream instruction, input, or reference answer was rewritten, repaired, or generated.

Screened the complete pinned 96,356-row upstream train split for fresh inputs without
correspondence markers, while retaining the pronoun and sentence requirements. Directly
reviewed the full input and answer of all 58 surviving candidates: 5 accepted, 39 rejected,
14 skipped as correspondence. Discovery uses conservative heuristics, not a proof that
every other upstream row is unsuitable. Skipped rows were not approved for insertion.

Replaced five previously approved correspondence rows with four news reports (world news,
weather, media, entertainment) and one technical Q&A explanation (AC circuits, addressed
to Emily). The latter is not an article. The other 979 previously approved replacements
were retained. **This is only a small diversity improvement, not a balanced dataset:**
979 correspondence, 4 news reports, 1 technical explanation. The original article mix
has not been restored. Failed references were excluded, never rewritten to make them pass.

Validation confirms exactly 984 fresh unique verbatim upstream replacements, the original
644/340 instruction split, no banned pronouns in constrained answers, and all 9,016
non-smol rows byte-identical, preserving 1,119 reasoning-bearing messages. Five rows differ
from the earlier corrected mixture. Provenance, reviews, slot mapping, checksums, and
validation are saved beside the new mixture. Nothing has been published or wired into a
training config.

Reproduce the diversity selection into a new dated directory (refuses to overwrite):

```sh
uv run --no-sync python -m scratch.replace_smol_summarize diversify
uv run --no-sync python -m scratch.replace_smol_summarize validate --out-root output/2026-09-08-smol-summarize-diversity
```

## Earlier: initial replacement

Completed a local replacement of all 984 `smol_summarize` rows in
`output/reasoning_backfill/enrich_20260908_114055/mixture.jsonl`.
The original file is unchanged. No dataset was published and no training configuration
was changed to select the new file.

## Artifacts

- Full 10,000-row mixture: `output/smol_summarize_replacement/mixture.jsonl`
- Standalone 984-row source: `output/smol_summarize_replacement/smol_summarize.jsonl`
- Original slot → replacement upstream index: `output/smol_summarize_replacement/replacements.json`
- Provenance and checksums: `output/smol_summarize_replacement/report.json`
- Independent structural validation: `output/smol_summarize_replacement/validation.json`
- Direct review decisions, including rejection reasons: `scratch/smol_summarize_review_decisions.json`

## Selection and review

Upstream: `HuggingFaceTB/smoltalk`, `smol-summarize`, train split, revision
`5feaf2fd3ffca7c237fc38d1861bc30365d48ffa`. Rows retain the upstream system instruction,
input and reference answer verbatim; none were rewritten or generated.

A deterministic shuffle (seed 20260908) supplied candidates. Mechanical screening removed
existing inputs, duplicates, prohibited personal/possessive/reflexive pronouns in the
constrained subtype, overlong answers, excess sentence terminators, and incomplete endings.
The main Codex assistant then read each reviewed candidate's full input and answer, judging
instruction compliance, faithfulness, actor attribution, invented details, grammatical
damage, and obvious source/reference errors. No OpenRouter or other model API judge calls
were used. Of 1,170 directly reviewed candidates, 984 were accepted and 186 rejected.
Unreviewed candidates are not implicitly accepted. Human-like direct judgment is not a
guarantee that every subtle factual or linguistic error has been eliminated.

The exact original subtype split is retained: 644 up-to-three-sentence, no-second/third-person
personal-pronoun rows and 340 one-very-short-sentence rows. Every replacement occupies an
original slot of the same subtype. All 9,016 non-smol rows remain byte-identical, including
the existing reasoning traces. Fresh inputs do not appear anywhere in the base mixture.
The maximum replacement length is 519 tokens under the repository's Qwen3.6-27B training
template, below the 8,192-token cap.

This preserves row counts and instruction mix, not the original content/length distribution:
the selected replacements are predominantly emails, and mean input length fell from 278.8
to 132.2 whitespace-delimited words. Filtering therefore also changes the smol portion's
token weight. The original data and unrelated instruction/worked-answer issues were not edited.

## Rebuild and validate

With the pinned upstream parquet and cached tokenizer available, run from the repository root:

```sh
uv run --no-sync python -m scratch.replace_smol_summarize build
uv run --no-sync python -m scratch.replace_smol_summarize validate
```

The recorded review decisions authorize only their specific prepared candidate IDs.
Do not change the preparation ordering/screens and reuse the same decisions against a
different candidate pool. `report.json` records the base and upstream checksums.
