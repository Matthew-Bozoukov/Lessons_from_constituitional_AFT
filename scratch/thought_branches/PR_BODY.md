Implements [Thought Branches (arXiv 2510.27484)](https://arxiv.org/abs/2510.27484) against
this repo's ODCV rollouts, and runs the cheap half of it over 859 published rollouts.

## Why

Every attempt to localise the difficult-advice effect has been read through an arm-level
ODCV misalignment rate, and the 2026-08-31 seed replicate put that estimator's noise floor
at **6.1 pp — wider than the entire 8.7–17.6% band nine corpus manipulations occupy**.
Resampling measures effects **per branch point, within scenario, without a training run**,
so it is not subject to that floor, and it is not the kind of intervention that failed in
the channel swap (2026-08-28) or the coherence graft (2026-08-29).

## What the rollouts said

859 rollouts, 5 arms, 40 scenarios, all judged, 26.4% violating.

- **15 of 21 good-vs-bad markers lose significance** once compared within one arm *and* one
  scenario; one flips sign. `commit_before_write` separates the outcomes inside all five
  arms and still collapses to −0.02 [−0.21, +0.15]. Controlling for the model is not
  controlling for the task.
- **Three survive**: violating rollouts run more commands (+5.3 [+0.2, +10.1]), think less
  per step (−96 chars [−172, −31]), and hit their fork earlier (−0.057 [−0.106, −0.011]).
- **The principal appeal is a null both ways** — the same shape as the paper's
  self-preservation finding, in a different scenario family.
- **The fork**: the moment the environment refuses is findable mechanically in 76% of
  rollouts with no labeller. Clustering the answers gave a 73%→0% gradient that was largely
  an artefact (fork-thought embeddings carry scenario identity). Corrected for scenario,
  three survive: +0.25 fabricate-the-missing-input (which independently recovers the
  2026-08-27 four-MO finding), +0.19 inventory-before-deciding, and **−0.21
  enumerate-each-result — protective across 25 scenarios, and new**.

## What was built

The paper's setting is single-shot; ODCV is an agentic loop whose continuation depends on a
container, so a branch point splits a transcript **and a world**. Two backends:

- `FrozenEnvSampler` — replay recorded tool results, resample one assistant turn. Local
  action distribution, one API call per sample, no Docker.
- `LiveEnvSampler` + `prefix_proxy` — an OpenAI-compatible shim that serves the recorded
  assistant turns back for steps 1..k−1 then forwards live, so the container runs the same
  commands, reaches the same state, and the run reaches a real ending the existing judges
  score. Turn accounting is by content, so a retry replays identically.

Metrics are the paper's four plus a paraphrase control, and call no model — a cached
resampling run can be re-scored at a new threshold without re-spending the GPU-hour.

## Two things reviewers should check

- **Serve FP32.** `arXiv:2607.28495`: BF16 fresh-prefill and live-KV continuations from
  identical token prefixes decoded differently on 166/200 suffixes. Every branch here is a
  fresh prefill. Appended to `docs/GOTCHAS.md`.
- **The canonical baseline cannot be added** to this corpus: the principle-scoped 702 arm
  and both seed siblings publish 0 transcripts and 0 judge tables. Nothing in this PR is an
  arm comparison.

## Scope

- Package lives in `scratch/` per CLAUDE.md; promotion target is
  `src/eval/misalignment/odcv/branches/`.
- New code is additive — no existing module is modified.
- `docs/LOG.md` entry added; `docs/GOTCHAS.md` appended.
- **42 package tests, full repo suite 1316 passed**, naming lint clean.
- Untested paths are named explicitly in the package README: the completions-route
  continuation, `judge_actions`, `LiveEnvSampler` end to end, and the OpenRouter labeller.

Memo with 8 ranked next experiments:
`scratch/thought_branches/2026-09-03_thought_branches_odcv_memo.md`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01VnNfKTuo6e2yUtNs8BZrgs
