# Agent guide

Conventions that apply to every project in this repository. Read this before
generating data, running an experiment, or committing.

---

## Datasets, caches and artifacts go to Hugging Face

**From 2026-07-29 onward, any dataset, generated corpus, evaluation output or
associated cache produced by work in this repository is published to Hugging
Face.** The repository holds code, configuration, analysis and reports. It does
not hold bulk data.

This applies to synthetic document corpora, generated response sets, evaluation
transcripts, judge outputs, embeddings, activation caches, and any intermediate
artifact large or reusable enough that someone would want to fetch it rather
than regenerate it.

### Naming: the title carries the date and the subject

Every HF repo name begins with an ISO date and continues with a short,
human-readable description of the experiment:

```
<YYYY-MM-DD>-<short-experiment-description>
```

Examples:

```
2026-07-29-msm-philosophy-spec-fixed-eval
2026-07-29-msm-philosophy-spec-fabrication-probes
2026-07-14-teaching-claude-why-synthdoc-corpus
```

The date is the date the data was **generated**, not the date it was uploaded.
A reader scanning a list of repos should be able to tell what an artifact is and
when it came from without opening it.

### Required metadata in the dataset card

Every upload carries a card (`README.md` in the HF repo) stating, at minimum:

| field | meaning |
| --- | --- |
| `experiment` | Which experiment produced this, in one sentence |
| `date_generated` | ISO date the data was produced |
| `constitution` | The constitution, spec or model spec this connects to - by name and link. Write `none` explicitly if it genuinely connects to none. Do not omit the field. |
| `source_repo` | This repository, and the commit hash the generating code was at |
| `models` | Every model id involved, with revision/commit pins where applicable |
| `generation_config` | Sampling settings - temperature, top_p, max_tokens, seeds |
| `schema` | What the columns/fields mean |
| `provenance` | How to regenerate it: the exact script and arguments |

The `constitution` field is not optional bookkeeping. Most work here is about
whether training on a written specification changes behaviour, so which
specification a dataset relates to is the single most important thing a future
reader needs, and it is the field most easily lost.

### What stays in git

- Code that generates or consumes the data
- Configs, seeds, rubrics, probe definitions - the *inputs*, which are small and
  are the scientific record
- Analysis scripts and their outputs where those are small (tables, summaries,
  figures)
- Reports and documentation
- A pointer to the HF repo, so the link is never only in someone's memory

### What does not stay in git

- Model weights and adapters
- Generated corpora and response sets above a few megabytes
- Provider caches, HF caches, virtual environments
- Anything reproducible from code plus a pinned model, unless it is small enough
  to be worth the convenience

Existing evidence directories predate this convention and are left in place.
New work follows the convention.

---

## Secrets

Credentials live outside the repository, in files under
`~/.config/msm-audit/`, and reach a process only through the wrappers in
`Vulnerabilities/scripts/secrets/`.

- Never print, echo, log, commit, or summarize a secret value.
- Never place a credential into the parent agent environment. Inject into the
  child process that needs it and no further.
- `.env`, `*.env`, `*.pem` and `*.key` are ignored repository-wide. That guard
  is in the root `.gitignore` deliberately, so it applies to every subdirectory.
- Before using a credential, validate it against a harmless read-only endpoint
  and record only provider, timestamp, HTTP status and success or failure -
  never a response body.

## Paid infrastructure

Any run that provisions a GPU must register it with the watchdog before doing
work, and must not rely on the orchestration process surviving to clean it up.
Teardown terminates the instance, then sweeps the whole account for orphans,
then records the provider-reported balance, and only then stands the watchdog
down. See `Vulnerabilities/scripts/provider/Stop-AuditRun.ps1`.

Never terminate a resource this repository did not provision. Report it instead.

## Reporting standards

These are house rules earned by mistakes; the reasoning is in
`Vulnerabilities/JOURNAL.md`.

- **Correct for multiple comparisons.** If you compute fifteen contrasts, say so
  and apply a correction. A point estimate without an interval is not a result.
- **Controls are not optional.** An uncontrolled number cannot be attributed to
  anything. If the control fails as a control, that is a finding about the
  design, not something to work around.
- **Validate before claiming.** Search-based auditors here have measured
  false-positive rates of 57% and 97.5%. An unvalidated flag is a lead.
- **State power.** "No effect" and "no effect of the size this design can
  detect" are different claims.
- **Keep corrections.** When a finding dies under scrutiny, record why rather
  than deleting it. Several of the most useful entries in this repository are
  results that did not survive.
