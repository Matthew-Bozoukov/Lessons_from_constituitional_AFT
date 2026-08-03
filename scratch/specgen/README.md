<!-- ABOUTME: specgen: distills the published Claude constitution into three specs that -->
<!-- ABOUTME: differ ONLY in granularity (4/12/24 principles), with claim-ID coverage parity. -->

# specgen — constitution granularity pipeline

Produces the three spec documents for the spec-variation experiment. The single
independent variable is **granularity** — how finely the same source content is carved
into principles (`coarse`=4, `mid`=12, `fine`=24). Everything else is held constant or
measured. One-off authoring tool (run once, refine until the docs are right — not a rerunnable pipeline stage, hence `scratch/`). Config: `specgen.yaml` beside this file.

All generation runs through **headless Claude Code subagents** (`claude -p --model
fable|opus`, subscription auth, tools disabled) — no OpenRouter, no API keys, no real
dollar spend. Trade-off: no temperature/seed control, so "seeds" are run indices and
variation comes from natural sampling; the budget guard tallies the CLI-reported
nominal cost. Requires a logged-in `claude` CLI.

```bash
CFG=scratch/specgen/specgen.yaml
uv run scratch/specgen/cli.py pin      --config $CFG --file <saved-constitution.md>
uv run scratch/specgen/cli.py extract  --config $CFG [--smoke]   # once, shared
uv run scratch/specgen/cli.py generate --config $CFG [--arm fine] [--smoke]
uv run scratch/specgen/cli.py metrics  --config $CFG             # offline
uv run pytest scratch/specgen/test_specgen.py -q                 # offline
```

## How parity is enforced

1. **pin** — the source (CC0, ~30k words; save the page as markdown by hand — it is
   JS-rendered) is hashed into `source.lock.json`; everything downstream records it.
2. **extract** — per-section calls (never one pass: at ~10× compression a single pass
   attends unevenly, IFScale arXiv:2507.11538) produce the claim inventory, one atomic
   normative claim per line with a verbatim ≤25-word anchor and a modality tag. The
   inventory is hashed and **shared by all arms** — this is the coverage guarantee.
3. **generate** — per arm × seed: partition the inventory into exactly N clusters
   (**every claim_id in exactly one cluster**, enforced in code with retry + fail);
   write each principle unit in an isolated call seeing only its cluster's claims;
   out-of-band units get one revision with the measured token count fed back;
   assemble as hand-written `preamble.md` → units (grouped by the constitution's
   priority order) → hand-written `closing.md`.
4. **metrics** — offline: token bands, unit floor, explanation ratio, modality-language
   profile, coverage, cross-seed partition stability (adjusted Rand index), the
   pre-registered seed selection, and `comparison.md`.

Every checkpoint of each doc is mirrored to the HF dataset repo in the config
(`hf_repo`): the claim inventory per extraction, and per arm/seed both
`constitution.draft.md` (first-pass units, pre-revision) and `constitution.md` (after
the token-band revision), under a fresh `<arm>/seed<k>/<run-timestamp>/` prefix each
run — successive refinement runs accumulate as history instead of overwriting, which
is the record of how the distilled docs evolved.

Invariant across arms: every unit is statement + `*Why:*` + `*When this does NOT
apply:*`; explanation ratio 0.55–0.65; preamble/closing byte-identical (asserted).
Permitted to vary: total length within per-arm bands (longest ≤2.5× shortest) and cue
count (3/2/1 — that is what the granularity axis means).

## Discipline

- **Never hand-edit an output.** Change the prompts/preamble/closing (in git) and
  regenerate; every output records prompt hashes, so drift is auditable. Git is the
  revision log.
- Cross-seed ARI is the stability check: if 5 seeds at N=24 disagree about what the 24
  principles are, the axis is unstable at authoring — surface that before GPU time.
- Contamination caveat (unimplemented, by choice of scope): Claude co-wrote the public
  constitution, so a Claude extractor may recite rather than read. Spot-check by
  perturbing a few source sentences and re-running `extract --smoke` on that copy; if
  extraction reproduces the original wording, switch the extract model family.
- Downstream, the **grading spec is the source constitution**, never an arm's own spec
  (`parent_priority` in `clusters.json` routes to the relevant top-level section).
- Publish `claims/inventory.jsonl` to HF per the repo's data policy — it is what makes
  the granularity manipulation auditable rather than editorial.
- Selected specs are promoted by hand into `constitutions/<name>/` following
  [`constitutions/README.md`](../../constitutions/README.md) (e.g.
  `claude_distilled_24_principles_fine/`), with `rationale.md` pointing at the run's
  `meta.json` and `selection.json`.
