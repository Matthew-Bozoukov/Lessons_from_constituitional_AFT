<!-- ABOUTME: Overflow gotcha file — hard-won operational lessons appended as they are learned. -->
<!-- ABOUTME: Default destination for new gotchas, including AI-written ones; CLAUDE.md keeps the curated core. -->

# GOTCHAS

The curated core gotchas live in CLAUDE.md ("Gotchas" section). This file is the
default destination for everything since: new gotchas go here, and AI agents may
append their own without asking. The price of that open door is that entries here
may be outdated or over-verbose — treat them as leads to verify, not law.

## GPU pods / vast.ai operations

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
`src.endpoints.openrouter`. So a script importing `src.huggingface` alone gets `None` for
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
