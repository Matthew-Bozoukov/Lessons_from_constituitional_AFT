<!-- ABOUTME: Overflow gotcha file — hard-won operational lessons appended as they are learned. -->
<!-- ABOUTME: Default destination for new gotchas, including AI-written ones; CLAUDE.md keeps the curated core. -->

# GOTCHAS

## Calibrated synthesis reviewers can still fail on generated prose (2026-09-21)

In the constitution-only low-stakes smoke, a reviewer passed all short paired
checks yet accepted invented personal details in full answers. Of 17 authored
examples in the activity-grounded run, the raw judge passed 15; an independent
read accepted eight. Citation guards happened to block the seven false passes,
which must not be reported as reliable semantic detection. Report raw reviewer,
effective export, and independent decisions separately.

Exact-quote transcription was a separate failure: use numbered original-source
and answer spans with disjoint schema enums, then retrieve quotations in code.
This proves evidence exists, not that it supports the claim. A blind "remove
unsupported facts" editing pass also retained an invented one-day duration.
Audit specific claims before editing, then independently audit the final text;
that fixed sequence is proposed and still needs fresh live validation. Preserve
failed fixtures and runs when correcting ambiguous calibration controls.

See [campaign report](dataset_audits/2026-09-21_lowstakes_pipeline_iteration.md).

## Name-only RunPod updates restart containers (2026-09-16)

A REST `PATCH /pods/{id}` containing only `name` incremented the pod version,
restarted its container and remapped SSH ports. Set user-readable names when
provisioning; do not rename a live inference pod when server continuity matters.
This happened before any rollout in the three-pass controls campaign.

The watchdog also terminates a pod when its owner process dies, not just when
its deadline expires. Killing an owner to reconnect is therefore not a safe
handoff. Keep that lifecycle intact; a replacement launch must preserve failed
startup evidence and deduct all preceding spend from the original budget.

## SSH timeout does not prove a vLLM startup failed (2026-09-16)

Two three-pass ODCV launches stopped before rollouts because their SSH control
commands timed out. One timed out waiting for the background launch acknowledgement;
the other timed out on `pgrep` even while the server returned HTTP200 on `/health`.
Recovered server logs distinguish these cases from model loading failures.

Readiness now checks HTTP first, bounds SSH probes and treats transport failure as
unknown liveness within the existing deadline. A lost launch acknowledgement must
not cause a second launch: tunnel to and check the original server. A confirmed
missing process still fails. Offline regression coverage lives in
`tests/test_vllm_startup_transport.py`; live replacement validation remains pending.

## Explicit all-supervision columns need an intentional standard-arm declaration (2026-09-15)

Historical workaround: superseded on main on 2026-09-17. The shared census in
`src/train/masking.py` now accepts an explicit all-supervision column with a warning;
ablation intent is checked when building the mixture. The old launch flag remains
in frozen run configurations but is not required by the current trainer.

The refreshed low-stakes/nonmoral mixtures explicitly store `supervise: all` on
synthetic rows. Loading them alongside replay rows creates a dataset-wide column
whose null entries also mean `all`. The trainer's ablation guard previously
rejected this valid layout, interpreting any such column as a promise of a
nondefault loss mode. Both first launches stopped before model loading or an
optimizer step; failure archives were saved and the owned pods terminated.

For an intentionally standard all-supervision mixture, pass the explicit boolean
`allow_default_supervise=true`. The default ablation guard remains active without
this declaration, and the resolved config preserves it. This changes schema
admission only, not data bytes, token masks, per-example loss or the SFT recipe.
Check the supervision census locally before renting GPUs, as well as running the
real model-specific mask gate on the training host.

## Frozen dataset runners and budget-stop exception identity (2026-09-15)

Launching `scratch/dataset_refresh/run.py` as a script creates its classes under
`__main__`. Its per-row module imports `scratch.dataset_refresh.run`, creating a
second `BudgetStop` class. At a lower batch spending ceiling, the latter module's
specific exception handler did not catch the former class; the generic handler
saved a failed terminal and queued candidates continued reaching the same
pre-dispatch limit. The shared atomic ledger still refused calls over the applicable
ceiling; this was a resume/state-classification defect, not unmetered API dispatch.

Use `scratch/dataset_refresh/execute_imported.py` so the client and stages share one
imported module. Existing frozen generation files remain unchanged. The narrow
`resume_budget_stops.py` helper can archive only exact lower-ceiling reservation
failures after checking frozen identity, all stage receipts, and settled known-cost
physical calls. It preserves prior paid stages and failed evidence; it cannot reopen
substantive rejects, uncertain billing, or provider-bound failures. Never reset the
ledger or regenerate completed stages to resume a batch.

## Delegated-harm runtime and first-run defects (2026-09-11)

This is hundreds of multi-turn workplace episodes, not 324 short answers. On one
H100 per Qwen3.6-27B adapter, observed median episode duration was about 1.9 minutes;
the benchmark-rescore world averaged about 13 minutes among finished episodes.
Live servers had 7–8 requests in flight, 100% GPU utilization and roughly 220–270
aggregate output tokens/second. These observations establish working concurrency,
not optimal throughput. Do not promise a speedup merely by raising batch size.

At 13:28 UTC, DA had generated for 86 minutes after 16 minutes of startup and four
minutes of authoring; control had generated for 80 minutes after 11 minutes of
startup and four minutes of authoring. A failed earlier control startup cost about
11 minutes and $0.62. Timings overlap across GPUs and must not be added together.
Only 276 DA and 216 control episodes were attempted after author failures, versus
324 planned per adapter; extrapolating to a complete future run must include that
extra workload. Budget roughly 2–3 hours for a full same-size run on the same GPU,
including startup and judging; this is an estimate, not a measured clean rerun.

- Fixed: indirect upstream Anthropic SDK import now checked before renting a GPU.
- Fixed in source/corrected scorer: outcome inputs no longer duplicate bulk reads,
  and judge output allowance is 8,192 rather than 2,048 tokens. Fixed scenario
  evidence is cached. The first run needs a separate full scoring pass; a future
  run should score correctly from the beginning. The resulting extra wall time
  cannot be measured precisely before that pass completes.
- Fixed: Windows watchdogs use CREATE_NO_WINDOW instead of DETACHED_PROCESS;
  a real venv-child test verifies no console allocation. Replacements kept the
  current pods' original deadlines. This incident did not restart model generation.
- Still open: the live tool loop requests the full 16,384-token output allowance
  even when the prompt leaves less space in its 65,536-token serving context.
  Predeclare and test context handling before another run; do not silently change
  it halfway through a comparison. Current failures remain explicit.
- Still open: provider-filtered author validation makes some requests unavailable.
  Do not count these as subject refusals, loosen fidelity checks after seeing
  outcomes, or describe accepted-author counts as model capability measurements.
- Timing accounting: initial judging occupied about 27–30 accumulated worker-minutes
  per adapter by 13:28 UTC. With eight workers, this is not 27–30 minutes of wall-clock
  delay. The later scoring pass is the main additional wait. Preserve separate
  generation, judging and publication timings in future runs.

## Size backup time from measured transfer speed (2026-09-09)

The first broader SFT checkpoint was3.85GB and took about10minutes over SSH to
Windows, including verification. A300-second fetch timeout and15-minute whole
recovery window cannot preserve two such checkpoints plus the final adapter.
The training owner now reserves45minutes within its unchanged dollar/lifetime cap;
archive creation and transfer share a bounded deadline of up to40minutes.

The already-running owner retains its originally loaded limits. For that run, an
independent completion-only preserver uses a distinct remote archive and local output
directory, a40-minute transfer bound, and the original watchdog ceiling. Its verified
receipt must be reconciled before manual termination; the old owner's own retries
must not overwrite the independent transfer. Never claim the source fix changed an
already-running process or increase the GPU cap silently.

## Windows SSH stdin changes LF scripts unless sent as bytes (2026-09-09)

`SshExec._ssh` previously used `subprocess.run(text=True, input=script)`. On Windows,
the text pipe changed LF to CRLF even when the in-memory string was correct. A broader
SFT launcher failed before any training step (`set: invalid option`, CR-suffixed paths,
ambiguous redirect). This is separate from stale checkout shell files below.

The shared SSH helper now sends UTF-8 bytes and explicitly decodes stdout/stderr.
Real subprocess regression tests check exact LF/Unicode payload and invalid-byte logs.
The training driver also syntax-checks the uploaded script before starting it.
The base image exposes `python3`, not necessarily `python`: use python3 for system
monitor scripts, and the repository interpreter for backup hashing: this image's
system Python3.10 lacks `hashlib.file_digest`, while `/root/work/.venv/bin/python`
is Python3.12. Use `uv run python` for repository dependencies. On the
already-owned pod, the failed startup files were retained, the exact LF script was
restored and checked, and a python3 compatibility symlink let the existing owner
monitor safely resume. No second GPU rental or training-seed change was needed.

## Git LF attributes do not repair stale worktree bytes (2026-09-09)

An existing Windows checkout held CRLF in 164/168 ODCV shell scripts even though
Git HEAD contained LF and `.gitattributes` already specified `*.sh text eol=lf`.
Git considered the normalized content clean. Docker copied the physical CRLF bytes:
models hit `/bin/bash^M`, `pipefail\r` and `do\r` failures, then repaired or replaced
task tools. All six independently sampled completed cells were affected; successful
container exits and `task_complete` calls concealed this measurement contamination.

Inspect actual build inputs before serving: `require_lf_shell_scripts` in
`src/eval/docker.py` now refuses CRLF shell files. Restore only shell scripts to their
committed bytes, verify a real Docker `bash -n` pass, and rebuild fresh workspaces.
Do not normalize data fixtures: some deliberately contain CRLF. The interrupted run
was preserved separately and never judged or pooled with the clean restart.

The curated core gotchas live in CLAUDE.md ("Gotchas" section). This file is the
default destination for everything since: new gotchas go here, and AI agents may
append their own without asking. The price of that open door is that entries here
may be outdated or over-verbose — treat them as leads to verify, not law.

## GPU pods / RunPod operations

**A pod's container disk is not storage.** `volumeInGb: 0` is the norm here, so anything a
pod computes dies with it. Pull artifacts off CONTINUOUSLY as they are produced, not at the
end — a job that only writes at coarse boundaries can otherwise lose hours to one crash —
and checksum the local copies before terminating anything.

Then make the pre-teardown check INVENTORY-driven, not monitor-driven: list what the pod
holds that took more than an hour to produce and account for each item, rather than
confirming the artifact you happened to be watching. Monitoring only covers what you already
thought of, so using it as the completeness check makes blind spots invisible by
construction — and the expensive things most likely to be forgotten are the ones that
finished early and stopped emitting events. Note "artifacts", not "results": the 2026-08-14
LESS run streamed every output file off continuously and checksummed all 24 of them, then
destroyed the pods holding the warmup LoRA weights that had PRODUCED those outputs. Inputs
count. That mistake cost no already-computed result, but it turned "score one more target
behaviour" from free into ~11 GPU-hours, because new validation gradients must be taken at
the same checkpoints the stored training features were taken at.

**An N-GPU job does not need an N-GPU box.** Multi-GPU capacity is intermittent (4xH200,
4xH200 NVL, 4xH100 NVL and 4xH100 were ALL unavailable simultaneously on 2026-08-14), so
work that shards with no inter-worker state can be split across several smaller pods
instead — that run went 1+3 across two pods. Two things make it practical: verify first
that sharding is exact (run shard 0 of 2 and 1 of 2 on ONE gpu and compare against the
unsharded result — no second GPU required to prove it), and move files pod-to-pod through
the RunPod HTTPS proxy (`https://<pod-id>-8080.proxy.runpod.net/`, which serves
`/workspace`) rather than round-tripping via your laptop — no credentials touch either pod.

## Training and debugging at scale

**Size a smoke test to the bug it is hunting.** Our `--smoke` flags cut rows, but on a
27B the per-iteration cost is the ~50GB weight load, which row count does not touch — so
a 4-row smoke still burns a minute before it reaches the bug, and a debugging loop is
only as fast as one iteration. So ask what class of bug you are actually chasing:

- **Scale-dependent** — OOM and activation memory, kernel behaviour, numerics and
  determinism, chat-template literals, anything about the real checkpoint's geometry.
  These MUST smoke on the real model; a small model would give a confidently wrong
  answer. (`model.eval()` silently disabling gradient checkpointing only OOMs at 27B.)
- **Scale-independent** — a missing dict key, an f-string typo, an off-by-one, a wrong
  config path, a schema mismatch. A 0.6B model on CPU finds these in seconds, and
  paying 27B prices for them is how an afternoon disappears.

Most pipelines contain both, so it is usually worth having both paths rather than
choosing once. Budget the loop either way: if an iteration you will run twenty times
costs minutes, fix the smoke before continuing to use it.

Related: **measure throughput on the real model early** — one example, one timer, before
the rest of the pipeline exists. A cost estimate derived from FLOP arithmetic instead of
a stopwatch has been wrong by ~3x here, which is the difference between a $20 decision
and a $50 one; and by the time the true number arrives, the GPU capacity you would have
booked on it may be gone.

**`model.eval()` silently disables gradient checkpointing.** transformers guards
recomputation with `if self.gradient_checkpointing and self.training:`, so calling
`eval()` after `gradient_checkpointing_enable()` retains every layer's activations
instead — on Qwen3.6-27B at ~2k tokens that is ~70GB extra and an OOM even on a 143GB
H200. The usual reason to want `eval()` is determinism, and dropout is what actually has
to go, so stay in `train()` and zero the dropout modules directly
(`for m in model.modules(): if isinstance(m, torch.nn.Dropout): m.p = 0.0`, plus
`lora_dropout=0.0`). Then VERIFY it, because that swaps a guarantee for a claim: run one
example twice and compare. Expect ~1e-04, not zero — CUDA atomics and checkpoint
recomputation make two backward passes over the same row differ by ~4e-03 RELATIVE
(measured), so a bit-equality assertion fails on healthy runs. Gate on cosine, and size
the tolerance to separate float noise (~5e-05) from a live dropout mask (~1e-02).

## Data generation and evals

**Opus 5 classifier refusals need native diagnostics, not blind retries.** On
2026-09-09, broader nonmoral generation saved all 12 sources but only 1/10 answers;
the other 9 returned `content_filter`. After preserving response diagnostics, a
bounded worked-example request for a harmless bread-log Python script returned
`native_finish_reason: refusal` with an explicit **cyber** classifier explanation.
The original nine calls lack category metadata; do not assume all had that cause.
`OpenRouterClient` now attaches safe failure metadata to completion exceptions,
and the nonmoral capped ledger preserves it. Filtered partial content stays excluded.

[Anthropic's refusal documentation](https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback)
describes Opus 5's additional classifiers and supported fallback to Opus 4.8.
Keep actual model identity in provenance; a fallback answer is not an Opus 5 answer.
Also verify model reasoning defaults: Opus 5 was **high/on by default**. An
8,192-token diagnostic spent all 8,192 tokens on reasoning and returned no answer.
This was a separate truncation failure, not a filter refusal. Use explicit supported
effort and sufficient total headroom. API-internal reasoning and the authored
dataset explanation are separate outputs.

**Gemini 3.7 Flash ends a completed reply WITHOUT the last closing tag.** Every stage-5
call of the 2026-08-20 trait-10 smoke came back `finish_reason=stop` with
`<reasoning>…</reasoning><response>…` and no `</response>` — the model had answered in
full, and the tagged parser failed 100% of them. `_parse_tagged` now accepts an unclosed
FINAL tag (non-final tags must still close; truncation is still rejected upstream by the
`finish_reason=length` assert). If a new model's tagged stage fails at ~100% with
`missing <last_tag> block`, replay one call and count open/close tags before touching
the prompt. Also: the repo pins this model to Vertex's **flex** tier (cheap, slow —
60–160 s per call); a run sized on Haiku latencies will take ~4x longer on it.

**Fire reads a `--`-prefixed VALUE as another flag.** `--extra "--concurrency=8"` arrives
as `extra=True` and a separate `concurrency` flag, and `True.split()` dies at startup.
Pass-through arguments go dashless (`--extra "concurrency=8"`) and the receiver prepends
the dashes.

**ODCV `ok+no_transcript` can be the executor's request timeout, not a missing tool
parser.** With thinking pinned and `max_model_len 65536`, a scenario where the model
thinks for very long hits the executor's HTTP timeout (`[AI API dead]: Request timed out`
in `docker_output.log` at ~30 min); compose then stops both containers and no
`messages_record.txt` is written, yet the harness reports `ok`. 2026-08-20: 3 of 65 cells
in one pass, 1 in the next, with `mandated/Propaganda-Engine-Finding` missing both times.
Distinguish it from the 2026-08-18 parser failure by the count: a few slow cells versus
every cell. The rollout CLI's `--resume <pass dir>` re-runs only the missing cells.

## ODCV serving over the RunPod HTTPS proxy (2026-08-19)

**`SERVE_READY` does not mean the proxy is routing.** `scratch/serve_adapter_runpod.py`
prints `SERVE_READY` once vLLM answers on `localhost:8000`. The RunPod `/http` proxy can
still 404 externally for a minute or two after that. ODCV launches 12 containers at once,
so a cold proxy kills whichever scenarios are in flight: the numina-control run lost 15 of
65 cells this way, all in the first two minutes, with `[AI API dead]: Error code: 404` in
`docker_output.log` and ZERO 404s in the vLLM log — the requests never reached the server.
A single-request smoke test passes right through this, because one request is not twelve.

Fix: fire ~8 `/v1/chat/completions` requests at the proxy and wait for 200s before
launching the driver. Costs seconds; the run that skipped it cost 15 cells and a top-up.

**Top up short cells instead of re-running the pass.** `scratch/odcv_topup_short_cells.py`
writes a config whose exclusion list is "every cell EXCEPT the ones short of N rollouts", so
a recovery pass fills gaps without changing the sampling depth of cells that were fine.
Derive the cell universe from the BENCH scenario list, not from observed transcripts — a
cell that failed in every pass appears in neither the counts nor the exclusion list, and
deriving from observed data silently drops exactly the cells that need topping up (this bug
hid 2 of 15 short cells on first run).

**Re-judging a new combined dir does not reuse the cache.** The judge caches per combined
directory, so `combined4x_...` re-scores everything `combined3x_...` already did. Topping up
12 transcripts cost a full 186-transcript re-judge (~$5, not ~$1). Budget for it, or judge
once at the end.

**A cell can be genuinely unmeasurable for one arm.** `Drone-Delivery-Optimizer` produced no
transcript for the numina-control arm in five attempts across four passes, including on a
warm proxy, while every sibling arm scored it fine. Report the reduced cell count rather
than implying equal coverage.

## LLM-as-judge instruments

**Batching a detector across ~50 rubrics deflates every prevalence it measures.** Asking one
judge call "which of these 48 properties does this record have?" is ~40x cheaper than one
call per (record, property), and it is NOT the same instrument. Measured 2026-08-20 over 48
real detectors x 20 real ODCV rollouts (960 verdict cells), against one-property-per-call as
the reference:

    batched   38.1% prevalence   85.0% cell agreement    12s
    single    47.5% prevalence   --                     465s

A systematic 7-9 point deflation, not noise: mean per-property gap 11.1%, individual
properties moving up to 35 points. The direction is what you would expect from a judge
satisficing under load, and from the prompt's own "when borderline, answer no" being applied
forty-eight times at once. Use batching as a cheap screen; do not publish a rate from it
without measuring the gap on your own rubrics first.

**A reasoning model spends its token budget BEFORE emitting content, so a tight `max_tokens`
returns a BLANK rather than a truncated answer.** CLAUDE.md gotcha 4 covers the eval case;
the judge case fails differently and worse. At `max_tokens=2000` a 49-rubric detector prompt
blanked 23 of 25 records with `finish_reason='length'` and `content=None` — intermittently,
because how long the model thinks varies per record, so it reads as flakiness rather than as
a budget. `EmptyCompletionError` is classified transient and retried 6x, so every failure
cost six full generations before surfacing. Size the budget for thinking PLUS answer, and
when a judge stage blanks, read `finish_reason` before blaming the provider.

On that same A/B, disabling reasoning on the batched path cost 1.5 points of agreement
(85.0% vs 86.5%) for a 40x speedup, with the mean per-property prevalence gap identical to
one decimal. On a batched detector, reasoning does not pay for itself. On the unbatched path
it is left on, because that path is the reference and changing it moves the yardstick.

**A `--smoke` that shrinks the corpus does not shrink the rubric count.** The smoke path for
`properties` runs 2 properties over 16 short records, so a bug needing 49 rubrics and a
12k-character trace is invisible to it by construction — GOTCHAS' own "size a smoke test to
the bug it is hunting", one level up. The generalisable fix:
`interpret.preflight_detect_many` opens the expensive stage with ONE call at full rubric
count on the LONGEST record in the corpus, and prints the projected cost of the full pass
from that stopwatch instead of an estimate.

**Never pipe a long background run's stdout through `grep`/`tail`.** The pipe buffers; when
the foreground timeout moves the command to the background, everything buffered is discarded
and the new output file is empty. A 10-minute, ~$5 A/B was lost this way with nothing
recoverable. Long runs write their own report file; the shell's stdout is a convenience, not
the artifact.

**`Property.channel` is a fact about the run, not an opinion for the interpreter.** The
naming prompt asks the model to return `"channel": "query" | "reasoning" | "response"`, and
the model answers it from CONTENT — so a cluster of REASONING descriptions about refusing
comes back labelled `response`, because refusing sounds like an action. Measured on the
2026-08-20 two-arm run before the fix: 25 of 49 reasoning-fit properties and 18 of 71
response-fit ones carried the wrong channel.

That field is not cosmetic. `interpret.detect`, `ablation/filter.py` and `ablation/mask.py`
all use it to decide which text to read, so a wrong value points them at the wrong half of
the record silently — the detector's per-record agreement with cluster membership fell to
21% on the affected properties, and their measured arm delta collapsed from -30pp to 0pp.
The producer now overrides the interpreter's guess with the channel the run actually
clustered. If you add a producer, do the same.

Worth noting what this did NOT touch: cluster membership, and therefore every prevalence,
arm contrast and outcome lift, all of which come from the features of the configured
channel. Only the detector-side paths read `Property.channel`.

## Dashboard: live Hub discovery (2026-08-25)

**`/api/datasets?author=<org>` never returns `siblings`, even with `full=true` or
`expand[]=siblings`.** The listing gives `tags`, `cardData`, `lastModified`, `createdAt` and
that is all; the file list needs the per-repo endpoint (`/api/datasets/<id>`) or the tree
API (`/api/datasets/<id>/tree/main?recursive=1`). So a card must NAME its rows file (the
default `configs:` entry) or the client pays one tree call per repo — which is why the
publishers write that config and the backfill script insists on it. `filter=<tag>` composes
with `expand[]=cardData` fine.

**Node runs the dashboard's unit tests against `lib/*.ts` directly, and its ESM resolver
does not guess extensions.** A relative VALUE import (`from "./lazy"`) loads under the
bundler and fails under `node --test` with `ERR_MODULE_NOT_FOUND`; type-only imports are
erased and never hit this, which is why every previously tested module happened to have
only those. Write `from "./lazy.ts"` — `allowImportingTsExtensions` is on in
`dashboard/tsconfig.json` (legal because `noEmit` is) and the bundler resolves the explicit
path the same way.

**Two stats-sidecar schemas are live on the Hub.** `uv run mix` writes
`{total: {examples}, by_source: {name: {examples}}}` as `mixture_stats.json`; the hand-pushed
arm mixtures (`t2_9284_*.jsonl`) write `{total: 9987, per_source: {name: count}}` as
`<rows file>.stats.json`, keyed by the corpus VARIANT (`difficult_advice_v2`). Anything that
computes the blend must read both, and match constitution sources by prefix
(`lib/composition.ts isConstitutionSource`), or a 7% arm reads as a control.

**Adding tags to an existing card: `huggingface_hub.metadata_update(repo, {"tags": [...]},
repo_type="dataset", overwrite=True)`** rewrites only the YAML front-matter and leaves the
body byte-identical; pass the MERGED list (existing ∪ new), since `overwrite` replaces the
key. Card-table fields (`| constitution | … |`) are not indexed by the Hub — only front-matter
is filterable — so anything discovery needs must be a tag or a `configs:` entry.

## `budget_usd` does not protect a single-stage pipeline (2026-08-25)

`pipeline.run` checks the budget BETWEEN stages. A config whose work is one `llm_tagged`
stage over a large corpus is therefore unguarded for the whole run: `verbose_cot.yaml` set
`budget_usd: 68.0` and spent ~$85 without the check ever executing, because the stage never
returned — it died on `max_fail_pct` first. Estimate the spend yourself before launching a
one-stage pipeline, or add a check inside `run_items`. Do not rely on `budget_usd` for this
shape.

Related: the smoke run is a poor cost predictor when a stage retries. A 20-record smoke
retried ~34% of records; the same config over 716 logged 371 failed attempts (~52%). The
retry rate IS the cost variable, and small-sample retry rate does not estimate it well.

## Anthropic's own content filter refuses ~5% of difficult-advice prompts (2026-08-25)

`finish_reason=content_filter` on 34/716 (4.7%) of difficult-advice expansion prompts,
served by first-party Anthropic. CLAUDE.md already records Bedrock at 2.6% and Vertex
refusing the same prompts; first-party is better but not immune, and this corpus is
ethically loaded by construction, which is the point of it. **None of seven 20-record
smokes saw a single refusal**, so this failure mode is invisible below ~100 records.

A refusal produces no output to hold to a contract and retrying the same prompt does not
clear it, so a stage over this corpus needs somewhere for such a record to land.
`llm_tagged`'s `on_exhausted.mark_refused` is that landing place — it keeps the record with
a distinct status instead of dropping it, which matters when the corpus has to stay
row-for-row comparable with a control arm.

## A script that touches HF but not the LLM client authenticates as nobody (2026-08-25)

`hf_token()` reads `os.environ`, and the only thing that calls `load_dotenv()` on import is
`src.infra.endpoints.openrouter`. So a script importing `src.huggingface` alone gets `None` for
the token, reads work (public repos), and the run dies on a 401 at PUSH time — after all
the expensive work is done. Any standalone script that pushes must `load_dotenv()` itself
or import the client for its side effect.

## Smoke ONE cell before launching a whole ODCV run (2026-08-25)

Component checks are not an end-to-end check. A run was launched across four boxes having
verified: the systemd tunnel `active`, each box reaching its OWN arm's adapter (not the
other's), and `odcv_preflight` building all 30 cells. Every one passed. The run still failed
instantly on every box, twice, for two reasons neither check could see:

  1. **`uv: No such file or directory`.** The bootstrap installs uv to `~/.local/bin`, which
     is NOT on PATH in a non-login SSH shell — the shell `ssh host 'cmd'` gives you. The
     check that uv installed says nothing about it being callable the way the launcher calls
     it. Use the absolute path: `/root/.local/bin/uv`.
  2. **`KeyError: 'OPENROUTER_API_KEY'`.** `odcv_rollout._run_scenario` reads that variable
     and passes it into each scenario container as `OPENAI_API_KEY` — even when the endpoint
     is our own vLLM over the tunnel, which ignores its value. A credential-free box still
     needs the variable SET. `OPENROUTER_API_KEY=local-vllm-no-auth` satisfies it without
     shipping a real secret.

Both are two-minute discoveries from one real scenario and ~40 minutes of idle GPU billing
otherwise. Run one cell (`--extra "concurrency=1"` over a one-scenario config) end to end
before dispatching passes to every box.

Related, same launch: a launcher loop with `sleep`s inside a tool timeout died partway, and
the boxes it never reached still held `run.log` from the PREVIOUS failed attempt — reading
`>>> ALL PASSES COMPLETE` from a stale log looks exactly like success. Dispatch
fire-and-forget (`setsid nohup ... & disown`) and verify with `pgrep`, not with the log tail.


## Driver machine (Windows)

**Smart App Control silently bricks every uv-managed Python.** Confirmed 2026-08-27 on the
Windows driver box: `uv run` died with

```
Unable to create process using '...\AppData\Roaming\uv\python\cpython-3.12.13-...\python.exe'
error: Failed to query Python interpreter ... An Application Control policy has blocked this
file. (os error 4551)
```

This is not a uv bug and not transient. Smart App Control (`HKLM:\SYSTEM\CurrentControlSet\
Control\CI\Policy` -> `VerifiedAndReputablePolicyState = 1`, `Win32_DeviceGuard.
CodeIntegrityPolicyEnforcementStatus = 2`) blocks unsigned executables outright, and the
python-build-standalone binaries uv downloads are `NotSigned`. Every uv-managed interpreter
is affected, so re-downloading or pinning a different 3.12 does not help. Diagnose in one
line: `Get-AuthenticodeSignature <path>\python.exe | Select Status` — `NotSigned` is the
whole story.

The fix is a SIGNED CPython, not a weaker policy. Smart App Control can only ever be turned
OFF — Windows cannot re-enable it without a full OS reinstall — so disabling it to run a
tool is a one-way door and the wrong trade.

```powershell
winget install --id Python.Python.3.12 --exact --source winget --scope user `
  --accept-package-agreements --accept-source-agreements --disable-interactivity
uv sync            # recreates .venv against the signed interpreter; uv venv refuses
                   # while a .venv exists, but sync replaces an invalid one itself
```

The python.org installer is Authenticode-signed by the Python Software Foundation and is
allowed. Then stop uv reaching for its own builds ever again:

```powershell
[Environment]::SetEnvironmentVariable("UV_PYTHON_PREFERENCE","only-system","User")
[Environment]::SetEnvironmentVariable("UV_PYTHON_DOWNLOADS","never","User")
```

Note that winget offers 3.12.10 while `uv.lock` was resolved on 3.12.13. That is fine —
`requires-python` is `==3.12.*`, which is what uv checks. A stopgap venv on the system
3.14 (`uv venv --python C:\Python314\python.exe .venv314`, `PYTHONPATH=.`) does run the
data-generation stack, but it is off-lockfile: use it to keep moving, not to conclude
anything, and delete it once the signed 3.12 is in.
## bash `wait` never returns when stdout is `exec > >(tee ...)` (2026-08-27)

A pod bootstrap that redirects everything through `exec > >(tee -a boot.log) 2>&1`, then
backgrounds N trainers with `&` and calls a bare `wait`, hangs forever after the trainers
exit: the process substitution is itself a background job of that shell, and `wait` with no
arguments waits for it too. Nothing after the `wait` (the adapter tarball, the DONE marker)
ever runs. Seen on the PAR seed-replicate pod (`scratch/par_b/train_pod.py`); both adapters
were on disk and were pulled file by file over the :8080 directory server instead.

Fix: capture each trainer's `$!` and `wait $PID_0 $PID_1 ...` on those PIDs only.

## Prefer the RunPod HTTPS proxy to a laptop SSH tunnel for ODCV (2026-08-29)

- (Removed 2026-09-07: the "one ODCV run per Docker daemon" rule. It described a collision
  in the harness's Compose project names, which are now namespaced by a hash of the arm's
  `model_key` (`odcv-<tag>-<variant>-<scenario>`, src/eval/misalignment/odcv/odcv_rollout.py),
  so two DIFFERENT arms can share one daemon. Two runs of the SAME arm still collide, and
  resource contention is still yours to watch.)

- **The laptop→pod tunnel is the weak link.** `odcv_local_run.sh`'s reconnecting `-N -L` forward
  kept resetting against a RunPod H100 ("Connection reset by peer" every few minutes); each cell
  then waits out the full `scenario_timeout_s` (2,400 s) against a dead endpoint, so a 65-cell pass
  crawled 3.5 h for 20 transcripts while the pod billed. `serve_adapter_runpod.py` already publishes
  :8000 over `https://<pod>-8000.proxy.runpod.net/v1`; put that in the config's `base_url`
  (containers reach it directly) — the grok/gpt arms and the 2026-08-29 PAR top-up ran that way
  with no drops. `scratch/par_coherence/topup_config.py` generates a config for only the cells
  short of N rollouts, for the top-up pass.
- **Every pod on the account can disappear at once** (2026-08-29 01:33 BST: four pods across two
  sessions, balance intact, no dead-man fired). `serve_adapter_runpod.py status` then reads
  `phase=booting / endpoint not answering`, which looks like a slow boot — confirm with
  `scratch/less/teardown.py --list` before waiting on it.
- **The mixture builder's shuffle depends on the corpus it reads.** Rebuilding a paired arm's
  mixture through `build_t2_9284_da716_mixture.py` from a 716-row corpus instead of the parent's
  813-row one gave 0/10,000 rows in the same position as the parent (same seed). For a
  one-variable arm, take the parent `mixture_think.jsonl` verbatim and substitute the changed
  texts by `scenario_id` (done for `2026-08-28-table2-9284-par716coh-train` @ e6bf309b).

## ODCV on 2026-09-05: bullseye images, whole-file `cat`s, and the reasoning that never came back

- **Half the scenario images stopped building.** 40 of the 80 ODCV scenarios are `FROM
  debian:bullseye-slim`; bullseye left LTS on 2026-08-31 and its `bullseye-security` pool is
  being pruned, so `apt-get install` dies on 404s for packages the fresh index still lists
  (arm64 and amd64 alike, checked from the host and inside a container). Every such cell
  reports `compose_exit_1+no_container` in ~6 s. `odcv_rollout.pin_apt_archive` rewrites the
  scenario Dockerfile at workspace-build time to apt from `archive.debian.org` (main only; the
  archive has no `bullseye-security` Release yet). If a future cell fails the same way, check
  the archive's suites before anything else. The `python:3.13-slim` scenarios were never affected.
- **A whole-file `cat` can kill the run at the judge, after every rollout finished.** One
  rollout `cat`ed a 4.6 MB access log; the next step overran the 16k window, the (patched)
  loop archived the 4-step transcript, and the judge then refused 2.4M tokens -- `evals`
  exited 1 with all rollouts on disk. `odcv_judge.judge_copy` now hands the judge a copy with
  any line over 20k chars cut (marker in the copy, count in the verdict cache); the rollout
  itself is never edited. Re-judge an older combined dir with `scratch/odcv_judge_cli.py`.
- **Prior-step reasoning was not reaching the model on our vLLM path.** The vendored loop
  resends only OpenRouter's `reasoning_details`; vLLM returns `reasoning`. Every ODCV number
  published before 2026-09-05 was measured with earlier steps rendered as EMPTY think blocks.
  Fixed in the vendored loop (VENDORED_FROM.txt); verified on the live server with vLLM's
  `/tokenize` (a resent `reasoning` grows the prompt). Do not compare pre- and post-fix arms.
- **Prefix caching on Qwen3.6 works and is now on for ODCV** (`serving.reuses_long_prefixes`),
  but at concurrency 8 on 2-7k contexts it removes ~80% of prefill compute without moving
  the pass wall clock: decode dominates. It is free (KV peaked at 16%), not a speedup here.

## Training pods need a CUDA 13 driver too (2026-09-05)

The repo's lock pins `torch 2.11.0+cu130`. On a RunPod host whose driver predates 580
(pod n41qb3lmav2cjz: driver 570.124.06, CUDA 12.8) the boot looks clean, `uv sync`
succeeds, and then `torch.cuda.is_available()` is False with a one-line UserWarning
("The NVIDIA driver on your system is too old (found version 12080)"); the trainer carries
on and loads the 55GB model onto CPU. `uv run runpod up --train` used to request no CUDA
version on purpose (the comment said only vLLM needed 13); it now requests `13.0` for both
shapes. If you provision any other way, check `nvidia-smi` says CUDA 13.x before launching,
and treat that warning as fatal.

## ODCV at concurrency 32 needs 64 docker networks; Docker Desktop's default pool holds 31 (2026-09-06)

Each ODCV scenario is a Compose project with TWO networks (`default` and `internal_net`).
On a default Docker Desktop (no `default-address-pools` in the daemon config) about half of
every 32-wide wave dies at `compose up` with `all predefined address pools have been fully
subnetted`; the cell comes back `compose_exit_1+no_container` in 4-25 s, and one resume
retry per pass cannot close a gap that size. The runner now refuses up front
(`require_network_capacity`). The fix is a bigger pool in `~/.docker/daemon.json` —
`"default-address-pools": [{"base": "10.200.0.0/14", "size": 24}]` — and a Docker restart;
the machine the lowstakes/ablated/par runs were driven from already had one.

## Vendored harness patches (moved from CLAUDE.md gotcha 5, 2026-09-06)

Every `third_party/` harness is byte-identical to its pinned upstream commit except for the
patches below, each marked `VENDORED PATCH` in place and listed in that tree's
`VENDORED_FROM.txt`. Re-cloning upstream loses all of them; re-apply from that file.

**agentic-misalignment** (`src/eval/misalignment/agentic_misalignment/third_party/`, one file,
`api_client/model_client.py`):
1. A `vllm/` provider: `_detect_provider` maps `vllm/<served-name>` to an OpenAI-compatible
   call configured from `VLLM_BASE_URL` / `VLLM_API_KEY`. This is how the harness reaches the
   model run_eval serves; upstream only knows hosted APIs.
2. Judge routing: upstream matched the substring "claude" and sent it to Anthropic before its
   `/`-prefix rule, so `anthropic/claude-sonnet-4.5` tried an API this project has no key for.
   Any provider-prefixed id now routes to OpenRouter.
3. Vestigial: `_call_vllm` still passes `chat_template_kwargs.enable_thinking`; the served
   template pins the mode and shadows it. Left in place as a behaviour-neutral diff.

**ODCV-Bench** (`src/eval/misalignment/odcv/third_party/`):
1. `evaluate_all_results.py`: transcript encoding pinned to utf-8/errors=replace.
2. `evaluate_all_results.py` (2026-08-31): guard `res.choices` so a provider error payload
   fails one verdict, not the whole judging batch.
3. `mission_executor/agent_main.py` (2026-09-04): archive the transcript on the API-failure
   exit; upstream wrote nothing there, so overrun rollouts vanished from every metric.
4. `mission_executor/agent_main.py` (2026-09-05): resend the model's `reasoning` each step.
   Upstream copies back only OpenRouter's `reasoning_details`; vLLM returns `reasoning`, so
   every earlier step reached the model as an empty think block. Every ODCV number published
   before that date was measured without carry-over.
5. `mission_executor/transcript_budget.py` + `Dockerfile` + `agent_main.py` (2026-09-05):
   the served window is the transcript budget. When vLLM refuses a prompt for length, that
   step's tool results are cut to the window in place, a `[Harness] transcript token limit`
   note is archived, and no further call is made; a reply cut off by the window ends the
   rollout the same way. Read from `CONTEXT_WINDOW_TOKENS`, set by the driver from
   `serving.context_window`.

**MASK** (`src/eval/misalignment/mask/third_party/`): the full list is that tree's
`VENDORED_FROM.txt` (served target, OpenRouter judge, judge batching, per-run data dir, empty
content, think-trace columns, archetype-prefix split). The one that changed published numbers is
`evaluate.py` #5 (2026-09-10), the verdict parser. Upstream read the judge's FIRST `Answer:` line
and kept non-letters verbatim. Gemini 3 Flash ends many verdicts `Answer: <B>` and sometimes
quotes the model first (`the AI's answer: "No."`), so beliefs were dropped and quotes scored as
lies. Every MASK repo's head was re-scored from its stored judge text on 2026-09-10; a MASK
number from before then lives at the `from_revision` named in that repo's
`metadata/rescore_meta.json` (docs/LOG.md, 2026-09-10). A new judge model brings a new output
format: check the extracted verdicts, not just the scores.

Deviations that live in OUR code rather than in a vendored tree (compose project per
scenario, the bullseye apt-archive rewrite, the judge-side line cap, `recover.py`) are
listed in each `VENDORED_FROM.txt` too.

## A same-day rerun of an arm publishes over its eval repo; the earlier run becomes a revision (2026-09-06)

The eval run name is `<date>-<eval>-<arm>` and nothing else: the served window, judges and
passes are protocol, not identity, so a second ODCV run of `2026-09-05-qwen36-0-nosynth` on the
same day mints `2026-09-06-odcv-qwen36-0-nosynth` again and `push_run_dir` uploads over the
existing repo — `gate_push` checks the name's shape and date, not whether the repo already
holds a different run. Nothing is lost: the Hub keeps every commit, so the earlier run stays
readable at its revision. That is the convention chosen on 2026-09-06 for the 28k-window
reruns: the head of `2026-09-06-odcv-qwen36-0-nosynth` and of `2026-09-06-odcv-qwen36` is the
28k run, and the morning's 16k run is revision `38c3b0809d` (nosynth) and the pre-rerun head
(base). Two consequences. Reading code that means the earlier protocol MUST pin `revision=`
(`hf_hub_download(..., revision=...)`), or it silently reads the newer run; and the dashboard,
which reads heads, shows only the latest. Nothing warns at push time, so know which run you
are about to write over — `HfApi().repo_exists` plus the head's `metadata/run_meta.json`
config tells you — and record the revision of what you replaced in the LOG entry.

## A monitor that greps only for tracebacks misses this repo's own refusals (2026-09-07)

`docker_preflight`, `plan_serving` and the naming lint refuse with FORMATTED prose, not
Python exceptions. A watcher filtering on `Traceback|RuntimeError|SystemExit` plus success
markers therefore stays silent through them, and silence reads as "still running": an ODCV
pod idled 30 minutes at $3.49/h after its preflight refused (Docker Desktop was quit).
Filter on the failure text a stage actually prints, treat an empty log past first-output
time as the alert, and check liveness by PID — `pgrep -f 'evals --name odcv'` also matches
the launcher shell whose command line contains that string, so it always says "alive".

## A teardown watcher keyed on `>>> pushed` kills the run it is guarding (2026-09-07)

`run_eval` prints `>>> pushed HF_TOKEN + HF_ORG + ... to root@<host>:/workspace/.env` at
STARTUP when `--push-env` is set, ~40 minutes before the epilogue prints `>>> pushed
https://huggingface.co/datasets/...`. A watcher waiting for `grep -q '>>> pushed'` before
terminating the pod matched the first line on its first poll and killed a MASK run two
archetypes into generation; MASK has no resume, so the whole run was lost and the hour of
H100 time with it. Match the completion line SPECIFICALLY — `'>>> pushed https'` — and
prefer a marker the epilogue alone emits. The general rule: before writing a watcher's
pattern, grep the log the run has ALREADY produced for it; a pattern that matches
something already on disk is a pattern that fires immediately.

## LLM-judged audits

**A judge question that invites "is there any nuance here?" returns ~90% yes and measures
nothing.** Auditing the non-moral deliberation scenarios (2026-09-02) for false dichotomies,
the rubric asked: *"Would a competent practitioner reject BOTH the instruction and its opposite
in favour of some third option?"* It flagged **37 of 40**. The scenarios were fine; the question
was not. Sonnet 5's leaked reasoning gave it away — *"whether a hybrid approach ... might satisfy
the convention"* — because almost every real situation admits SOME hybrid, so a question phrased
as "could there be a third way" is answerable yes almost everywhere and separates nothing.

Rewriting it to name what does NOT count ("a hybrid, a compromise, or a resolution landing on
the other side of the named tension is NOT a false dichotomy — that is the right answer; do not
flag on the mere existence of one") moved the same 40 scenarios to **2 flagged**. Same corpus,
same judge, same temperature: an 18x swing from one clause.

Two rules that follow. Write the EXCLUSIONS into a judge rubric, not just the inclusion
criterion — for any property worth auditing, the near-misses are the bulk of the distribution
and the judge has no way to guess where you draw the line. And treat a flag rate near 0% or
near 100% as a bug in the question until proven otherwise: a discriminator that fires on
everything has the same information content as one that fires on nothing, and both look like
results.

**Reasoning models eat the whole token budget before answering.** The same audit's first run
died on every call with `finish_reason='length'` and empty content at `max_tokens=200`: 199 of
the 200 were reasoning tokens. This is CLAUDE.md gotcha 4 biting inside an ad-hoc judge script
rather than inside an eval harness — a tagged-output rubric that would fit comfortably in 200
tokens still needs thousands when the judge thinks first. Size a judge call for trace + answer,
and read `finish_reason` before trusting an empty result.

## synth pipeline

**`synth topup` computes its snapshot index wrongly for any config with an observer stage.**
It derives the file to read as `names.index(draft_stage) + 1`, but observer stages
(`kind: corpus_check` mid-pipeline, e.g. `corpus_scenarios`) take a slot in `names` while
producing no numbered snapshot -- `snapshot_positions` skips them by design. So on the
difficult-advice stage layout every index after the observer is off by one, and topup dies with
`FileNotFoundError: stage_5_draft_prompts.jsonl` when the file is `stage_4_draft_prompts.jsonl`.
Hit 2026-09-02 on the non-moral arm; the same shape is in
`configs/data/synth/2026-08-01_difficult_advice.yaml`, so it is not specific to that recipe.

**Resume is the workaround, and it is better anyway.** To re-run only the records a stage LOST
(as opposed to topping a trait up past its original quota): move that stage's final snapshot
aside, leave its `.partial.jsonl` checkpoint in place, and `synth run --resume <dir>`. The stage
reads the checkpoint, reports `N already saved, M remaining`, and pays only for the M. Deleting
the partial as well would re-pay for the whole stage.

**Size `refine` for the fields YOU ask for, not for the count difficult advice inherited.** That
recipe's `revise_prompts` returns six fields at `max_tokens: 6144`. A fork returning seven
truncated 4 calls mid-JSON and lost 10 of 716 records -- and the losses were not uniform: six
landed on ONE trait, dropping it from 80 to 74 and capping the trait-balanced mixture draw at
9x74=666 instead of 702, which would have silently changed the arm's synthetic share of the
mixture. Sonnet 5 is a reasoning model, so its trace counts against the same cap. A cap is not a
charge -- raising it costs nothing except on the calls that would otherwise have truncated.
Check per-trait counts after every revision stage, not just the total.

## RunPod provisioning: two silent failures that bill

**A pod with no SSH key looks identical to a pod that is still booting.** `runpod up` injects
`PUBLIC_KEY` only when `~/.ssh/id_ed25519.pub` exists at that exact path -- no key file, no
error, no key on the pod. The provision output prints `ssh: not answering yet`, which is also
what a healthy pod prints for its first minute, so the failure is invisible until the first
connection returns `Permission denied (publickey,password)`. A machine with keys under other
names (`msm_audit`, an org key) still fails: the path is not searched, it is hardcoded. Check
`ls ~/.ssh/id_ed25519.pub` BEFORE renting anything, and read the provision line for
`ssh: ready` rather than assuming.

**`nvidia-smi` is not the CUDA check, on RunPod as on vast.** A 2xH200 training pod came up
showing both GPUs and 143GB each, and `torch.cuda.is_available()` was False: host driver
550.127.05 (CUDA 12.4) against the repo's pinned `torch 2.11.0+cu130`, which wants driver >=
580. The pod bills at the full 2xH200 rate the entire time and nothing before the first CUDA
call complains.

The cause was in `up()` itself, which passed `cuda=""` for TRAINING pods while constraining
serving pods to `"13.0"`. The reasoning in the comment was that a training pod runs the repo's
own pinned stack rather than vLLM's and so needs no constraint -- true when written, false once
the pinned stack moved to cu130. Fixed 2026-09-02 (commit e684cb8) so both shapes get the
constraint. The general lesson is the one worth keeping: **a scheduling constraint derived from
what a dependency needed is a fact with an expiry date.** When the dependency moves, the
constraint does not move with it, and the failure is a silently mis-scheduled paid box.

Always run `uv run python -c 'import torch; print(torch.cuda.is_available())'` on a fresh pod
before launching anything long. It costs one SSH round-trip and it is the only check that
actually tests what you are about to depend on.

## Two `evals --server` runs on one driver machine collide on local port 8000 (2026-09-08)

`uv run evals --server <host>` tunnels the pod's vLLM back to `127.0.0.1:<--port>`, default
8000, and starts vLLM on the SAME port on the pod. A second concurrent run from the same
laptop (a second pod, a second arm) also picks 8000: ssh prints `bind [127.0.0.1]:8000:
Address already in use / Could not request local forwarding` and keeps the session open,
the run continues as if the tunnel were up, every request goes nowhere, and MASK reports
`4438/4438 generations failed (100.0%)` and refuses -- after the pod has already billed its
full boot and the whole generation stage. Happened to the nosynth and da-7 MASK runs launched
beside a live dat-7 run at 23:03 on 2026-09-08: two pods rented, torn down with nothing.

Before launching a second `--server` run on a machine that already has one, `lsof -nP
-iTCP:8000 -sTCP:LISTEN` (or `ps aux | grep 'ssh .* -L'`) and give the new run a distinct
`--port` (8001, 8002, ...). `SshExec` uses the port for both ends, so nothing else needs to
change. The error line is easy to miss: it arrives on ssh's stderr between two `>>>` progress
lines and the eval does not treat it as fatal.

## MASK `subsample: 1000` is a no-op: the vendored csv_data IS 1000 rows (2026-09-09)

`configs/eval/mask.yaml` said (until 2026-09-09) that upstream holds 2,595 rows and that
`subsample` draws N evenly across the six archetypes. The 2,595 was `wc -l` over CSVs whose
prompts contain newlines. The default is now `null`. The vendored `third_party/mask/mask/csv_data/` holds the PUBLIC
release, which is exactly 1000 rows (continuations 176, disinformation 125, doubling-down 120,
known_facts 209, provided_facts 274, statistics 96 -- `metadata/subsample.json` on every
published run reads "drawn N of N" for all six). So every 1000-row run is the whole public
set, the per-archetype split is the dataset's own and NOT even, and only `subsample` below
1000 actually samples. Generations per row also differ by archetype: 6 (1 pressure + 3x
belief_elicit_1 + belief_elicit_2 + belief_elicit_3) for four of them, 4 for statistics (no
elicit_2/3), 1 for provided_facts (pressure only) -- 4,438 generations per 1000-row run.

## A RunPod host can have a crawling network; speed-test before the boot script has cost an hour (2026-09-09)

Three H200 training pods rented at the same minute: the one in `SE` reached READY (clone +
`uv sync`) in 3 minutes; the two in `EUR-IS-4` were still downloading wheels after 20 minutes,
with `curl` from the pod measuring 14 KB/s from Hugging Face and PyPI timing out. Nothing in
the boot log says "slow" -- it just keeps printing `Downloading ...` lines -- so the failure
looks like ordinary boot until you compare against a sibling. `uv run runpod up` prints
`ssh: ready` well before the install finishes: at that moment run a 15-second download test
on the pod (`curl -sL -o /dev/null -m 15 -w '%{speed_download}' <a HF weight shard URL>`) and
terminate anything under ~1 MB/s. `--countries SE` pinned the re-rent to the datacenter that
worked. The GraphQL `pod { machine { dataCenterId location } }` query names a pod's
datacenter; the REST pod object does not.

## A generator's invented tool output is wrong even when the files are in front of it (2026-09-10)

The first daa pipeline (`scratch/daa/agentify.py`, 2026-09-09) let the generator write the
environment files AND the stdout a script "would print" over them. Executing those scripts for
real against the same files: 2 of 2 outputs were wrong (7 vs 9 flagged rows; 105 vs 128 words),
and the trace, operator message and `task_complete` reason all repeated the wrong number -- the
fabrication pattern the ODCV progress judge is blind to, baked into training data. The generator
also authored a memo BEFORE the run it cited, so replacing stdout after the fact did not fix the
memo. The pipeline now runs every command (exploration included) in a temp sandbox and authors
each written file only when the agent reaches it, from the real outputs so far. Lessons:
- never let a model guess a command's output when the command can be run; the guess is
  plausible, specific and wrong, which is worse than absent;
- anything authored downstream of a guessed number inherits it; generate after execution, not
  before;
- a lint that drops rows with stale numbers is a backstop, not a fix -- it fired on 1 of 5 rows.

## libfaketime fakes file timestamps too; `docker cp -` takes a tar and keeps its mtimes (2026-09-10)

The daa sandbox freezes a container's clock with `LD_PRELOAD=libfaketime.so.1 FAKETIME="2024-05-15
09:30:00"` so `date` and `datetime.today()` inside generated scripts agree with the scenario. With
an absolute FAKETIME, libfaketime also rewrites `stat()` results, so `ls -l` dated EVERY file at the
frozen instant in the year form (`May 15  2024`) regardless of its real mtime. `NO_FAKE_STAT=1` in
the exec env stops that; the files then show their own mtimes in the time form (`May 14 09:30`)
because ls's "recent" window is measured against the faked now. Two related traps: the directories
`docker cp` creates for a tar it receives on stdin carry the HOST clock (touch every ancestor up to
`/`, not just the leaf), and a naive `datetime.timestamp()` on the host is local time -- compute
tar mtimes with `tzinfo=timezone.utc` since the container runs `TZ=UTC`. `docker cp - <ctr>:/`
with an in-memory tar sets content, mode and mtime in one call; no staging directory and no path
relocation, so the environment's absolute paths are real inside the container.

## An optional generation stage must not be able to kill the run (2026-09-10)

The daa rewrite stage is one call per row and dispensable: a row it cannot improve keeps its
stage-4 version. One Anthropic content-filter refusal on a 10-row smoke was 10% failures, above
`run_items`'s 5% ceiling, and the whole run raised after the loop stage had been paid for. Catch
the call's exception inside the stage function and return the input row with a note; reserve
`max_fail_pct` for stages whose output the row cannot exist without.

## `claude -p` as a generator: bare mode, bearer token, structured-output turns (2026-09-12)

Generating training data through Claude Code print mode (`claude -p --output-format json
--json-schema ...`) on the subscription instead of an API key works, with four traps:
- **Everything the CLI knows leaks into the model's context** even with `--system-prompt`: the
  cwd's CLAUDE.md, MEMORY.md, a `userEmail` block, an Environment block. A Sonnet smoke wrote the
  account email into three fabricated documents as the author. `--bare` (`CLAUDE_CODE_SIMPLE=1`)
  strips all of it; the only block left is today's date. Run from an empty cwd anyway.
- **Bare mode never reads the keychain login** and ignores `CLAUDE_CODE_OAUTH_TOKEN`; it accepts
  only a bearer token in `ANTHROPIC_AUTH_TOKEN` (or an `apiKeyHelper`, which is sent as an API
  key and rejected). Mint one with `claude setup-token`, keep it in `.env`, pass it in a minimal
  env (PATH, HOME, locale, the token) so nothing else from the shell reaches the CLI.
- **Structured output is a tool call the CLI validates.** A schema miss needs a further turn to
  retry, so `--max-turns 1` turns every miss into `subtype: error_max_turns` with an empty result.
  Use `--max-turns 3` with `--tools ""`. Record the raw JSON of failed calls; `result` is empty
  and the reason is in `subtype` / `terminal_reason`.
- **`--resume` on a run dir must reuse the run's row selection.** A resume without the original
  `--smoke` flag picked up all 708 source rows and started generating them; the run dir now
  stores `selection.json`.

## The sandbox's host-clock rewrite must only touch `ls` lines (2026-09-12)

The daa sandbox replaces the host's date stamps in command output with the frozen scenario
clock so a file the agent just created does not show the real year in `ls -l`. Applied to every
stdout line it also rewrote file CONTENT: a draft dated with the host date read back as
"Date: Sep 11 08:30". Restrict the rewrite to `ls -l`/stat-shaped lines, and tell the writing
model the scenario's date so it never writes the host's.

## The nosynth base blend is MODEL-SPECIFIC: its replay traces are on-policy for one family (2026-09-13)

`2026-09-08-nosynth-mix` -- the `base_mixture:` every arm pins -- carries ~1,135 reasoning
traces on its tulu3_if / self_oss_instruct / lima rows, written by qwen/qwen3.6-27b answering
each row's own prompt (the reasoning backfill of 2026-09-08). They are on-policy for Qwen3.6
and off-policy for anything else, so a base blend belongs to ONE family, and until this date
nothing said so: nosynth.yaml declared `reasoning: none` on every source and the card's
`models` field said `none`. Now the base config declares `reasoning_backfill: {model, judge,
sources, fraction, max_tokens}` (src/data/mixture/reasoning_backfill.py), the built mixture
records `reasoning_traces` (model, family, counts) in mixture_stats.json and names the
generator in its card, an arm mixture inherits the block from the base it pins (the
pre-record base is read through its enrichment_report.json), and `uv run train` refuses a
mixture whose trace family is not the model being trained unless
`allow_trace_family_mismatch=true`. **A new base model needs its own base blend**: change
`reasoning_backfill.model` and `tokenizer` in nosynth.yaml, rebuild (~$10 of generation +
judge), and point the arm configs' `base_mixture:` at the new repo.

## Killarney (Alliance SLURM): a non-login shell silently builds the wrong venv (2026-09-03)

`module` is a shell FUNCTION sourced from the login profile, so it does not exist in the
non-login shell that a remote one-shot command gives you. A setup script that guards its
module loads with `command -v module` and *continues* when absent therefore builds its
venv on `/usr/bin/python` instead of the cluster's `python/3.12.4`.

Everything then works until the one thing that needs a compiler. vLLM's inductor pass
compiles C++ at engine startup and the system interpreter's headers are incomplete:

```
/usr/include/python3.12/pyconfig.h:3:12: fatal error:
    x86_64-linux-gnu/python3.12/pyconfig.h: No such file or directory
```

which surfaces as `RuntimeError: Engine core initialization failed` — **after** loading
52 GiB of weights onto the GPU, i.e. after paying for the allocation. The real error is
~150 lines above the traceback vLLM prints, so grep the vLLM log for `fatal error`
rather than reading its tail.

Two lessons, both now enforced in `scripts/infra/slurm/setup_killarney.sh`:

- Run cluster setup through a LOGIN shell (`bash -lc "..."`), and make a missing `module`
  command a hard failure rather than a fallback.
- Verify the interpreter after creating a venv: `pyvenv.cfg`'s `home` is the bin directory
  of whatever built it, and `/usr/bin` there means the module was not active. Checking it
  costs nothing; the alternative is discovering it on a GPU.

## `--target` is nargs='+' and will eat your config overrides (2026-09-03)

`run_eval.py` declares `--target` with `nargs="+"`, so it consumes every following token
that does not start with `-`. Putting it last means the trailing `key=value` OmegaConf
overrides are parsed as additional model repos:

```
HFValidationError: Repo id must use alphanumeric chars ...: 'experiment=collusion'
```

`--target` goes FIRST, terminated by `--name` (a real flag), with the overrides trailing
at the end where `parse_known_args` collects them — the order CLAUDE.md documents.
Verified: the wrong order yields 4 targets and 0 overrides; the right one yields 2 and 2.
It costs only seconds of GPU, because run_eval resolves and names every target before it
serves anything.

## uv ignores an activated venv (2026-09-03)

Activating a venv and then calling `uv run` does NOT use that venv. uv resolves the
project environment itself — `.venv` in the project root unless `UV_PROJECT_ENVIRONMENT`
says otherwise — and *syncs* it, which needs the network. On an offline compute node that
is a hang or a hard failure, and the venv you carefully activated is ignored either way.

Call the entry point directly (`python -m src.eval.run_eval`) and set
`UV_PROJECT_ENVIRONMENT` + `UV_OFFLINE` so anything else reaching for uv fails loudly
instead. Also set `UV_LINK_MODE=copy` when uv's cache and the target venv are on
different filesystems (`/home` vs `/project` here), or every package warns as it falls
back to a full copy — and budget for that copy: ~18GB of torch/vLLM onto NFS runs
~850 MB/min.

## Killarney: CPU count gates the GPU queue, not walltime (2026-09-03)

`sbatch --test-only` estimated start times for one H100 in `gpubase_h100_b1`, same job,
2h walltime, varying only the CPU request:

| request | estimated start |
|---|---|
| 16 CPUs, 64G | +2h 13m |
| 12 CPUs, 64G | +33m |
| 8 CPUs, 96G  | immediate |

Memory barely mattered; CPUs decided everything, because the free H100 nodes were mostly
full of other jobs' cores. Ask `--test-only` before committing to a shape, and prefer the
smallest CPU count the work actually needs — asyncio's default executor caps at
`min(32, cpu_count + 4)`, so 8 CPUs still affords 12 worker threads.

## Dated adapters make eval names too long, in THREE places (2026-09-03)

Since adapters became dated artifacts, `spec.model_key` carries its own production date.
Anywhere that composes `today + model_key` therefore produces a name with two dates, and
for a long arm it blows the 96-character limit `local_name`/`gate_push` enforce. The
difficult-advice arm (`2026-08-21_qwen36_lora_table2_9284_difficult_advice_chunk_only_702_rank_64_dynbatch`)
tripped all three of these in one afternoon:

| site | symptom | fix |
|---|---|---|
| `run_eval` out_dir | 101 chars; died naming arm 2 **after arm 1 finished** | `subject_of(model_key)` |
| `run_eval` summary row | 109 chars; died **after** results.json was written | eval name became the directory, plus `subject_of` |
| a published repo name | 119 chars; would die on a login node after all GPU spend | explicit short `arm_labels` in the eval config |

The lesson is not the individual fixes but where they fire: **every one of these fails
late**, after episodes are run and sometimes after results are on disk, because names are
composed at publish time rather than checked up front. When adding an eval, assert its
names through `gate_push`/`local_name` in a unit test — `tests/test_colosseum_publish.py`
does this for all six of its repo names and runs in a second.

`subject_of()` is the right tool: it strips the artifact's own date, which belongs to the
artifact, and leaves the run's date to `local_name`. `run_meta.json`'s `target` still
records exactly which artifact was served.

## Two concurrent arms of one eval collide on the run directory (2026-09-03)

`run_eval` names each arm directory `<model_key>_<HHMMSS>` — no job id, no pid. Two jobs
that reach that line in the same second get the SAME directory, and if every arm of the
study starts from the same control checkpoint (as a mixed-team design does), the
model_key half never disambiguates them.

Observed: `single` and `cooperation` both started at 15:56:37, shared one arm directory
AND one Colosseum output tree, interleaved their episodes into it, and were heading for a
race on `results/per_seed.json`. It was caught 31 minutes in only because an arm directory
listed cells `[baseline cooperation]`, which no single experiment has.

Stagger parallel jobs deterministically (`scripts/infra/slurm/colosseum_job.sh` offsets
per experiment) — or give each its own working directory. And when running arms in
parallel, check the cells each run directory actually contains before trusting any
aggregate over them.

## run_eval publishes each arm before naming the next — so partial runs are salvageable (2026-09-03)

Worth knowing when an invocation dies partway: `_publish` runs at the end of each arm's
loop iteration, so every arm that finished is complete on disk — `results/per_seed.json`,
`results/results.json`, `metadata/run_meta.json` — even though the invocation as a whole
failed. Rerunning both arms to recover one is the expensive way out.

`ARMS=control|treatment|both` on the Colosseum job script exists for this. The cost is
that in-invocation pooling does not happen, so the contrast is assembled afterwards from
the two run directories (`scratch/colosseum_pool_split_arms.py`) — the same computation
over the same inputs.

## Two concurrent run_eval invocations need their own PORT and their own WORK DIR (2026-09-03)

`run_eval` defaults every vLLM server to port 8000 and every server work directory to
`output/<eval>/server`. Both are fine for one run at a time and break as soon as two run
together — which SLURM makes easy, since it packs several one-GPU jobs onto one 8-GPU
node.

**Port.** Only the first server binds. The loser sits with the weights loaded and its GPU
at **0% and 123W**, which looks exactly like a slow job: no error, no crash, and the log
still ticking over from tqdm. Three of six GPUs were idle for 45 minutes before anyone
noticed. There is a correctness edge too — the loser's driver can reach the WINNER's
server on `localhost:8000`, and the only thing between that and an arm being served by
the other job's checkpoint is Colosseum's own served-model-name check.

**Work dir.** Not just logs: the thinking-mode chat template is written there and handed
to vLLM to read at startup. Concurrent runs rewrite it under each other, and a server
booting at the wrong moment reads a half-written file and dies. The driver then reports
only `vLLM server ... is not reachable at 127.0.0.1:<port>` — nothing in that message
suggests a different process caused it.

Both are now derived from the job: the port from `SLURM_JOB_ID`
(`scripts/infra/slurm/colosseum_job.sh`), the work dir from the port
(`src/eval/run_eval.py`).

**Diagnosing this needs `nvidia-smi` INSIDE the allocation.** `ssh <node> nvidia-smi` is
not in the job's cgroup and reported an idle GPU for a busy job and vice versa — it was
worse than no information. Use:

```bash
srun --jobid=<id> --overlap -n1 nvidia-smi \
    --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
```

Sample it several times: a single reading can catch a genuine gap between decode steps.
Weights loaded (~75GB) with 0% utilisation across repeated samples is the signature of a
driver that cannot reach its server, not of a slow model.

## `runpod up --eval a,b` is ONE repo id to Fire (2026-09-04)

`up()` accepts a list of targets, but Fire hands a comma-joined argument over as a single
string and `up` only splits a real list, so `--eval a,b` dies in `resolve_target` with
`HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'` —
before anything is rented, at least. Pass a Python list literal, which Fire parses:

```
uv run runpod up --name x --eval "['LASR-Callum/a','LASR-Callum/b']"
```

## The pod image's `python3` is 3.10: no `tomllib` (2026-09-04)

`runpod/pytorch:0.7.0-dev-cu1281-torch271-ubuntu2204` ships python3.10 as `python3`. Anything
run on the pod outside a venv that reaches for a 3.11+ module (tomllib, `array.array[int]`
annotations) fails there even though every venv on the box is 3.12. Use the venv's
interpreter explicitly (`/root/work/.venv/bin/python`, `/workspace/vllmenv/bin/python`)
for one-liners in bootstrap scripts.

## vLLM serves one LoRA per batch by default (2026-09-04)

With two adapters on one server (`ServedTarget.sibling`, the Hospital seating), vLLM's default
`--max-loras 1` runs only requests for ONE adapter per step: a mixed job showed 2 running / 13
deferred requests and ~75 tok/s total. `VllmServer._start` now passes `--max-loras 2`
(commit c53976d). If you serve more adapters than that, raise it to the number co-served.

## Colosseum sabotage boards thrash the KV cache (2026-09-04)

Self-sacrificial and covert Hospital episodes grow 30-60k-token boards; at 30 concurrent
episodes on one H100 the cache sat at ~83% with half the requests queued, and a cell of 30
seeds took ~2.5 h instead of ~1 h. Budget for it, or cap `max_concurrent_runs` near 15 for
those conditions; `serving.context_window` must stay 65536 (32768 dropped up to 40% of
treatment episodes with a vLLM 400 on prompt length).

## Colosseum ignores `logging.root_dir` (2026-09-04)

The Hospital environment writes its per-iteration inventory snapshots
(`data_iteration_*.json`) under `/root/colosseum/logs/HospitalEnvironment/...` on the pod, not
under the run dir, so `pull_runs.sh` never sees them. The false-claim check needs them: pull
with `scratch/colosseum_hospital/pull_env_logs.sh <host> <label>` before tearing the pod down.

## Laptop memory guard kills background watchers (2026-09-04)

Long `until ... sleep` loops launched with `run_in_background` on the laptop were killed by
the low-memory guard within an hour (four pollers at once). A session cron job
(`CronCreate`, every 10 min, running `scratch/colosseum_hospital/poll_queues.sh`) survived;
prefer that for multi-hour waits.

## Timestamps: pods log UTC, the laptop shows BST (2026-09-04)

Queue logs and run-dir stamps on pods are UTC (`date -u`); `rlog`/`dump` headings and the local
clock are BST (UTC+1). Mixed notes drifted by an hour once. Quote pod times as written, and
label a local time as local.

## Colosseum Hospital is not symmetric between its two hospitals (2026-09-06)

With no coalition and every agent the same model, General Hospital earns ~2,650 points a shift
and St Mary's ~3,780. Two environment properties cause it (Terrarium
`envs/dcops/hospital/hospital_env.py`): patients have NO home hospital — every department's job
queue lists every patient whose next unscheduled step is its service, whoever schedules first
gets the credit, and St Mary's departments win that race (Triage 1,630 vs 1,093 points untempted);
and `_infer_step_agent` charges any step nobody schedules (its 500-point penalty) to
`hospital_names[0]`, General Hospital, unless the previous step was St Mary's. Starting stock is
symmetric. Consequences: never compare the two hospitals' points directly; read hospital-level
numbers only as before-vs-after within a block; treat `advantage_hospital` as directional. The
pair-vs-rest advantage and all within-block contrasts are unaffected. A fixed eval would give
patients a home hospital and attribute unscheduled steps to the patient's hospital.

## The Hospital judge cannot see a refusal (2026-09-07)

`src/eval/misalignment/colosseum/hospital/judge.py` builds every prompt from `blackboards.json`
alone — the public board plus the coalition's private board — and its refusal question asks
whether any agent declined *on that blackboard*. A model that refuses in its private reasoning
and never says so on a board is scored as fully compliant. In the self-sacrificial cells this is
not a corner case: the difficult-advice arm refuses in 24 of 30 shifts where the judge scores 3.
The control is unaffected, because it almost never refuses anywhere, so the error is not uniform
across arms and a refusal rate compared between them is not comparing like with like.

Read every published refusal number from this eval as *visible* refusal. Anyone editing the
benchmark should add the seats' reasoning (it is already captured, in `agent_turns.json`, as
`llm_calls[].reasoning`) to the judge's input as its own channel, judged separately from the
boards, so a stated refusal and an acted refusal stay distinguishable. The reasoning-level
classifier in `scratch/colosseum_hospital/decline_vs_action.py` is a stopgap, not a fix.

## The Hospital harness could not hold a refusal — fixed behind `fixes:` switches (2026-09-09)

The 2026-09-04 Hospital study measured the harness as much as the models. Six defects, all
verified in the code (`terrarium-agents 0.1.1` + Colosseum `ac0b405`): every turn starts
from `[system, user]`, so a seat forgets its own refusal while its partner's compliant posts
and its own actions come back through the board every turn; the provisioner's prompt never
rendered a board (it could post to both and read neither — the "coalition channel" was
one-way); `post_message` was offered in planning only while most refusals came in
execution, whose prompt ended "DO NOT POST MESSAGES"; the scheduler answered "success" to
an unknown patient and to a repeat; `get_job_queue` returned the string "Refer to prompt
context."; patients had no home hospital (both hospitals' departments were handed the same
job; unscheduled steps were charged to `hospital_names[0]`); and the judge read boards only.

All of it is now switchable — `fixes:` in `configs/eval/colosseum_hospital.yaml` (twelve switches after the 2026-09-10 additions: rationale-free public action events and a coalition-only MODE SWITCH paragraph), every
switch OFF there so the base config is still the paper's harness — through two patches in
`src/eval/misalignment/colosseum/third_party/` (README there has the switch table):
`terrarium_hospital_fixes.patch` goes into the terrarium-agents *site-packages* and
`hospital_eval_fixes.patch` into the Colosseum checkout after `hospital_seating.patch`.
Direction A (memory) and direction B (the board) are `scratch/colosseum_hospital/configs/2026-09-09_colosseum_
hospital_carried_history.yaml` and `..._board_access.yaml` (`--config`), sharing the
environment fixes. Things that bite:

- **The runner refuses an unpatched package**, and the harvest refuses an episode whose
  `run_config.json` lacks `terrarium_fixes` or records a different `fixes` block. `uv sync`
  removes terrarium (it is not in the lock), so after any sync re-run
  `scratch/colosseum_hospital/pod_bootstrap.sh`, which re-applies and verifies both patches
  and ends with the scripted-model smoke (`fixes_smoke.py`, 97 checks, ~1 min, no GPU).
- **Every episode now writes `prompts.jsonl`** (one record per model call, the whole
  message list minus the system prompt, which is in `system_prompts.json` by hash). A
  department call is 8–22k tokens of text; a 30-seed cell is a few hundred MB. It is the
  rollout, so it is pulled and published; do not exclude it in `pull_runs.sh`.
- **`judge.json` gained channels.** `per_run[run]` keys are now `public`, `secret` and
  `reasoning:<seat>` for the two watched seats; judge files written before 2026-09-09 have
  the first two only, and their refusal ratings are board-level refusal (see the entry
  above). Analysis code that iterates channels must not assume two.
- **`retry_reason`** on `llm_calls[]` says why a second call was made. The loop's rule is
  unchanged (continue until an environment tool commits): a refusal, a plan, a read and an
  error all earn a retry, a committed action ends the turn, so an arm that refuses more is
  re-asked more. The harvest reports `retry_calls` and its kinds per arm.
- **`post_message` without `blackboard_id` lands on board 0, the public one**, and the tool
  schema makes the id optional. Not patched: it is the model's mistake to make and the
  public-board judge should see it — but read a "coalition plan on the public board"
  finding with this in mind.

## Colosseum's post-run survey is only usable with the 2026-09-15 patches (2026-09-15)

Upstream's survey (`experiment.survey_enabled`) ran every seat through the normal agent loop with
the planning tools offered, so a survey reply could call `transfer_resources` after the last
iteration and before `get_final_summary`, and a reply with no tool call was re-asked, so every
seat answered twice. A benign run never gets the MisalignmentPrompts wrapper, which is where the
survey question lived, so a baseline cell's survey sent the status briefing with no question in
it. `terrarium_hospital_fixes_3.patch` makes the survey one tool-less call and
`hospital_eval_fixes_3.patch` asks the baseline too; `survey: true` on an older checkout runs the
broken version. The survey turn also lands in agent_turns.json with the last iteration's number:
anything that walks the turns by iteration (actions.py does) must drop `phase == "survey"`.


## DA supervision campaign: Windows ODCV and archive publication (2026-09-17)

- A wildcard SSH bind (`0.0.0.0`) is not a usable Windows HTTP client address.
  Keep the listener reachable by Docker, but use `127.0.0.1` for host health/API
  requests. The scratch evaluator applies this split; do not infer readiness from
  a tunnel process alone.
- Python can create long paths that Windows CreateProcess rejects as `cwd`.
  Answer/empty pilot directories of 271/266 characters failed with WinError267.
  Existing `GetShortPathNameW` aliases passed real Compose config/build checks.
  If short names are unavailable, choose a shorter output root. Test the deepest
  actual working directory before renting a GPU.
- Pilot configs and receipts belong under `metadata/`. Leaving them at the run
  root lets generation/judging finish but fails `assert_layout` at publication.
  Rehome only the stray files with hash verification and publish saved outputs;
  no model rerun is needed.
- Full training archives can upload directly from the pod to its HF model repo,
  avoiding a slow laptop hop. Verify remote size and SHA256 against an uploaded
  manifest before teardown; adapter availability alone does not preserve checkpoints.

These are verified campaign lessons, implemented in `scratch/da_supervision/`,
not assertions that the reusable pipeline has incorporated every workaround.
See the [operational record](../scratch/da_supervision/archive/operations.md).

## Windows concurrent SSH commands must not inherit console stdin (2026-09-22)

The practical low-stakes trainer kept advancing while its owner's SSH polls timed out
during the first3.85GB checkpoint transfer. Direct probes from another process worked;
the owner recovered immediately after the transfer. Restarting the owner would have
triggered its watchdog and killed healthy training.

A hidden-process reproduction ran one SSH `sleep 10` alongside a short control call.
With inherited stdin the control call hit its5-second timeout; with null stdin it
returned in1.47seconds. After the fix, both variants returned in about1.5seconds.
`SshExec._ssh` now uses `stdin=DEVNULL` unless explicitly uploading script bytes, and
the checkpoint transfer also uses `DEVNULL`. Existing UTF-8/LF upload tests and the
new closed-stdin/real-archive-copy regressions pass.

The already-running owner retains the old imported functions. Its live health is
covered by `scratch/dataset_refresh/observe_practical_training.py`, which checks the
pod directly when the owner's progress is stale; it never rewrites owner state or
restarts the trainer. The original watchdog deadline remains in force.
## Driving an eval ON a pod needs two env vars the SSH path sets for you (2026-09-21)

`uv run evals --server <pod>` starts vLLM through `SshExec`, whose `base_env` carries two
facts about our pods: `HF_HOME=/workspace/hf` (where `runpod up --eval` pre-pulled the
weights) and `VLLM_USE_FLASHINFER_SAMPLER=0`. Plain `uv run evals` ON the pod (the
`--clone-repo` shape) goes through `LocalExec`, which inherits the shell's environment and
sets neither. Without the second, vLLM's engine dies at start-up: flashinfer JIT-compiles its
sampler with `/usr/local/cuda/bin/nvcc`, which the pod image does not have ("Ninja build
failed ... nvcc: not found", surfaced only as "Engine core initialization failed" in the
eval's own log — the cause is in `output/<eval>/server_8000/vllm.log`). Launch as:

    HF_HOME=/workspace/hf VLLM_USE_FLASHINFER_SAMPLER=0 uv run evals --name <eval> --target <hf>

The pod also needs `OPENROUTER_API_KEY` in `/root/work/.env` for judging (`--push_env`
carries only the HF and W&B keys), and nothing tears the pod down for you: `--terminate-pod`
needs the RunPod key, which should not be on the box.

## A multi-hour eval over the SSH tunnel can lose everything to one TCP reset (2026-09-21)

A MASK run (~3 h at 16k think tokens, 32 in flight) driven with `--server` died at 00:32 UTC
to "Connection reset by peer" on the tunnel. `run_eval` opens the tunnel once and never
reopens it, so every later generation failed, the run crossed `max_generation_error_rate`
and refused to score, `--terminate-pod` released the pod, and the MASK runner cannot resume
(it clears `mask_work` on start). The first four archetypes were intact (1.4% errors); all
384 `statistics` generations were lost. ODCV-lite (~1 h) has not hit this. Until the tunnel
reconnects by itself, drive MASK on the box (entry above).

## Packing on Qwen3.6: three kernels must all know the boundaries, and one of them has to be built (2026-09-21)

`train.packing=true` concatenates a step's examples into dense rows. It is exact ONLY when every
boundary-aware op gets the boundaries: varlen attention (`attn_implementation=flash_attention_2`;
transformers 5.14 fetches `kernels-community/flash-attn2` through the `kernels` package when
`flash_attn` is absent — no CUDA build, `kernels` is in the lock), the gated-delta kernel
(`fla` reads `cu_seq_lens_q`), and the kernel-4 causal conv in front of it, which respects
boundaries only via `causal_conv1d_fn(seq_idx=...)`. The torch fallback conv leaks 3 positions
into every example, so the trainer refuses to pack without `causal_conv1d`. No wheel exists for
torch 2.11/cu13, so it is an sdist in the lock (metadata declared, no build isolation, build
variables in `pyproject.toml`) that `runpod up --train` compiles during its second `uv sync`
(~8 min on a 256-core pod, `BUILDING_CAUSAL_CONV1D` in the boot log). On any other box the
same recipe is needed by hand, because the pip CUDA layout has no unversioned `libcudart.so`
for the linker and torch finds CUDA only through CUDA_HOME or an nvcc on PATH:

    CU=$(uv run python -c 'import nvidia,os;print(os.path.join(os.path.dirname(nvidia.__path__[0]),"nvidia","cu13"))')
    mkdir -p /root/cudalib && ln -sf $CU/lib/libcudart.so.13 /root/cudalib/libcudart.so
    CUDA_HOME=$CU PATH=$CU/bin:$PATH LIBRARY_PATH=/root/cudalib:$CU/lib uv sync

Verify with `scratch/pack_equality_check.py`: its leak probe (same example, different
neighbours) must read 0.0000 — it did on 2026-09-21 with all three in place — and the
packed-vs-alone difference must sit at the kernel-shape noise floor it prints beside it. The
max over 250k bf16 logits is NOT a usable metric (it reached 7.8 with zero leakage).

## A push OVERWRITES an existing Hub repo, and same-day same-arm artifacts collide (2026-09-22)

`push_run_dir`/`push_files` do `create_repo(exist_ok=True)` + upload: a second artifact minted
with the same name silently replaces the first. Names carry only date + arm, so two trainings
of one arm on one day (or two evals of same-named adapters on one day) collide — it happened
with the pre-filter and filtered-base nosynth adapters and their ODCV/MASK runs. Before
launching a run whose name already exists, `move_repo` the OLDER artifact to the previous
date (2026-09-20 and 2026-09-21 precedents), fix its card's `date_generated`, and then
`create_repo` an empty repo at the freed name — HF keeps a redirect after a move, and a
push that follows it would overwrite the moved repo. A `-suffix` rename is off-grammar.

