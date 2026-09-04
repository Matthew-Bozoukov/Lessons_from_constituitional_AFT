<!-- ABOUTME: Overflow gotcha file — hard-won operational lessons appended as they are learned. -->
<!-- ABOUTME: Default destination for new gotchas, including AI-written ones; CLAUDE.md keeps the curated core. -->

# GOTCHAS

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

## One ODCV run per Docker daemon; prefer the RunPod HTTPS proxy to a laptop SSH tunnel (2026-08-29)

- **Two concurrent ODCV runs on one Docker daemon destroy each other.** The harness names compose
  projects `odcv-<variant>-<scenario>`, global on the daemon, so a second run of the same scenarios
  (a different arm, a different session) tears down the first run's containers mid-cell: both passes
  end `ok+no_transcript` / `compose_exit_137` with 0 transcripts and nothing in the summary says why.
  Measured 2026-08-28 18:33 BST when a PAR-arm pass and a GPT-seed pass started together. Before
  launching `odcv_rollout_cli`, check `pgrep -f 'odcv_rollout_cli\.py'` and
  `docker ps --filter name=odcv-` are both empty (match the `.py`, not the bare name — a watcher
  shell whose command line merely mentions the string trips the guard), and coordinate with any
  other session on the machine.
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
