---
title: "Sequencing decision: adding SURF to the investigation"
date: 2026-07-29
summary: "Run SURF on the SAME pod immediately after the Petri compute finishes, and defer all write-up until after teardown. Re-renting would risk losing an already-scarce 80GB A100 for a saving of only ~$0.75."
status: decided (sequencing) / blocked (method specifics)
---

# Sequencing decision: adding SURF to the investigation

## Decision

**Run SURF on the current pod, back-to-back with Petri, then decommission once.**
Order the work so the GPU is never idle, and do the write-up *after* teardown.

This decision does **not** depend on what SURF turns out to be, which is why it
could be made immediately. The method specifics are a separate, still-open
question (see below).

## The three options, priced

Current state at the time of the decision: pod created 2026-07-28T22:57:46Z on an
A100-SXM4-80GB at **$1.49/h**, hard deadline 2026-07-30T10:57:46Z, roughly 1.1 h
and **~$1.64** consumed of the **$40** cap.

Measured setup cost from this run (not estimated - these are the actual timings):

| Setup step | Measured |
| --- | --- |
| Pod create -> SSH endpoint | ~2 min |
| vLLM + deps install | ~4 min |
| Base + adapter download (65.5 GB) | ~16 min |
| vLLM load + CUDA graph capture | ~6 min |
| **Total re-setup if the pod is released** | **~28 min ≈ $0.70** |

| Option | GPU cost delta | Risk |
| --- | --- | --- |
| **A. Petri -> teardown -> write-up -> re-rent for SURF** | +$0.70 re-setup, plus idle burn if the pod is held during write-up | **Stock risk (the real cost).** A100 PCIe was already unavailable at provisioning; SXM showed *Medium*. Releasing an 80 GB A100 and failing to re-acquire one would push SURF onto RTX PRO 6000 ($1.89-1.99/h) or H100 ($2.89/h), or stall it entirely. |
| **B. Same pod, Petri then SURF, one teardown** | baseline | SURF's stack might conflict with the vLLM serving stack. Mitigated: they need not coexist - stop vLLM, run SURF, restart vLLM. Same weights, same disk. |
| **C. Second pod in parallel now** | doubles burn to ~$3/h against an unknown SURF scope | Highest budget risk, and the $40 cap is the binding constraint, not wall clock (36 h cap, ~35 h remain). |

## Why B

1. **Stock scarcity dominates the arithmetic.** The pure GPU-time saving from
   reuse is only ~$0.70. That is not the argument. The argument is that an 80 GB
   A100 on Secure Cloud is *already* partly unavailable — the cheaper PCIe variant
   could not be allocated at all — so releasing this one to save $0.70 risks
   paying 27-94% more per hour later, or not getting a suitable GPU at all.
2. **The expensive artefact is the 65.5 GB of pinned, hash-verified weights on
   the persistent volume**, not the GPU-hours. Those are already staged and
   checksummed; re-downloading re-runs the whole preflight identity chain.
3. **Idle GPU is the real waste, and option A creates it.** Writing up Petri
   results takes real time. Holding the pod through write-up would burn
   $1.50-3.00 doing nothing — as much as the re-setup it was meant to avoid.
   The fix is not to choose A or B on setup cost, but to **never hold a GPU
   during write-up**. Hence: compute, compute, teardown, then write.
4. **The decision is robust to what SURF is.** If SURF needs only black-box API
   access, it reuses the running vLLM endpoint unchanged. If it needs white-box
   access (activations, steering, gradients), the same pod serves it via
   transformers + PEFT against the same local weights after vLLM is stopped.
   Either way the pod is the right host.

## Revised budget envelope

| Phase | Est. hours | Est. cost @ $1.49/h |
| --- | --- | --- |
| Consumed to decision point | 1.1 | $1.64 |
| Petri pilot (4 audits) | 1.5 | $2.24 |
| Petri focused discovery (30 audits) | 6.0 | $8.94 |
| Fixed evaluation + matched controls | 4.0 | $5.96 |
| **Petri subtotal** | **12.6** | **$18.78** |
| SURF (to be scoped once the method is confirmed) | TBD | remaining **~$21** |
| **Cap** | 36 h wall clock | **$40.00** |

If SURF's scope would push total spend past the cap, the cap wins: I stop and
report, per the standing instruction.

## Open question, not guessed at

**I could not identify "SURF" from the literature with enough confidence to plan
against it.** Searches across alignment auditing, red-teaming, eliciting hidden
behaviours, unlearning, and interpretability returned adjacent methods (STAR,
DART, investigator agents, fuzzing for hidden behaviours) but nothing named SURF.

Writing a concrete SURF protocol on a guessed expansion of the acronym would
produce plausible-looking work aimed at the wrong target, so the method-specific
sections below are deliberately left unfilled pending confirmation. The
sequencing decision above stands regardless and has been acted on.

What I need to fill this in: what SURF stands for, and a link or paper, or a one-
line description of what it does to a model.

## Plan skeleton, ready to instantiate

The parts that hold whatever SURF turns out to be:

1. **Novelty gate.** SURF findings pass through the same
   [exclusion matrix](00-exclusion-matrix.md). A SURF result that re-derives a
   published MSM finding is replication and is labelled as such.
2. **Same target, same pins.** `chloeli/qwen-3-32b-philosophy-spec-msm-aft-cot`
   at revision `9a00c85c…` over base `Qwen/Qwen3-32B` at `9216db57…`, the same
   hash-verified local copies Petri used, so Petri and SURF findings are directly
   comparable rather than being about two different serving stacks.
3. **Same matched controls.** The five released comparators already verified in
   `evidence/prior-work/checkpoint-index.json`, so any SURF-surfaced behaviour can
   be attributed across MSM / AFT / CoT the same way.
4. **Convergence is the interesting result.** The highest-value outcome is
   whether SURF independently surfaces the mechanism Petri's family C predicts
   (harmful overgeneralization of deference and epistemic humility). Agreement
   between a behavioural audit and a second method is much stronger evidence than
   either alone; disagreement is itself informative and would be reported.
5. **Artefacts.** SURF gets its own numbered document and its own raw-artifact
   directory under `evidence/`, following the export guide, and its costs join
   the same ledger.

## Immediate action taken

Nothing was paused. The Petri phase continues on the current pod, keeping the GPU
busy while the SURF question is resolved.
