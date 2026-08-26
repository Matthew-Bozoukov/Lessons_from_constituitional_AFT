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
