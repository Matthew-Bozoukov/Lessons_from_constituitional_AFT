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
