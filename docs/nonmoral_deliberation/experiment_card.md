<!-- ABOUTME: Current short oversight card for the nonmoral-deliberation research. -->
<!-- ABOUTME: Links exact outputs and limits; previous plans and failed runs remain in the log and frozen artifacts. -->

# Current result and next research decision

## Authorized overnight execution — 2026-09-09

**Current scope: baseline fallback.** Fresh validation finished at **13/32 valid**
after the single allowed correction (11 repair failures, 8 source/construct exclusions),
below the frozen 24-pair minimum. Cost **$1.204914**, no uncertain reservations. No
corpus scaling or new SFT will run. Complete the existing nonmoral/math/Table-2-only
comparison instead. The first partial baseline attempt was invalidated before judging:
stale Windows CRLF bytes in Linux scripts made models repair the environment. Its
32 completed and 8 partial rollouts remain separate; scripts were restored to exact
committed LF bytes and all 168 passed a real Docker `bash -n` check before restart.

User approved a **$300 total ceiling including prior spending**, fresh self-contained
nonmoral scenarios, two new LoRAs (one seed per condition), baseline and new-model ODCV,
and public artifacts under `dougalldeepmind`. SFT uses **2xH200 with dynamic batching**;
ODCV model serving uses RunPod and its Docker driver stays local. Data and baseline
execution proceed independently. Prioritize the new matched comparison if resources
conflict; if fresh paired validation fails, stop that branch and finish baseline work.
No additional user approval is needed within this scope. Earlier "not authorized"
statements below describe the previous state. Exact reserved allocations and the
initial $2.279166 spending exposure are in
`output/nonmoral_overnight/20260909/authorization.json`.

**Bounded reset complete. Three locally accepted pairs from eight original tasks.**
No training, new ODCV evaluation, rental or upload ran. No active pilot process/lock remains.
The [postmortem](2026-09-08_postmortem.md) explains the reset; the [research brief](research_brief.md)
retains the full research agreement.

## What we produced

[Read all three full paired examples](../../output/nonmoral_paired_reuse_pilot/restart_packet/pairs.md)
or inspect the [machine-readable results](../../output/nonmoral_paired_reuse_pilot/restart_packet/summary.json).

| Stage | Exact result |
|---|---|
| Joint deliberation + complete answer | 8/8 generated, no parsing retries; 1 passed local review immediately |
| One targeted correction | 7 corrections; 5/8 deliberative examples then accepted, 3 excluded |
| Verification-only control | 5/5 generated, no parsing retries; 3 accepted, 2 excluded |
| Final development packet | Drawing error, model handoff, scheduling proposal: 3 complete B/C pairs |

Both arms use **identical original system/user messages and identical full final answers**.
B weighs viable alternatives; C constructs/checks the selected answer without weighing
alternatives. This is written-supervision content, not proof of an absence of internal decisions.
Root and a separate local reviewer agree on acceptance. Reviews were not blinded.

Cost: **$0.412058** for 20 calls in this reset. Shared pilot ledger including the prior
failed run: **$0.599406**, all settled, under the $3 cap. No further semantic repairs.
The original eight-case denominator stays fixed; this selected development sample is
not an acceptance-rate estimate for the historical 684 or a new population.

## What still failed

- **Dashboard:** the correction still confuses sustained bitrate target deviation with
  instability and includes review-process text.
- **ERP:** the corrected reasoning still reverses dependency direction and references review.
- **Manual:** the correction defends mismatched templates by inventing that a modern spec has no history.
- **Shapes control:** claims all tooltips are five to seven words; one has four.
- **Event-schema control:** infers seconds from ten-digit epoch values; ten-digit millisecond
  values are also valid. The final example itself is correct.

The last two have valid deliberative examples/final answers but faulty verification
reasoning. They were excluded, not silently patched or regenerated.

## Length remains a confound

| Accepted task | B CoT tokens | C CoT tokens | Shared answer tokens |
|---|---:|---:|---:|
| Drawing error | 552 | 506 | 18 |
| Model handoff | 564 | 711 | 377 |
| Proposal | 493 | 987 | 1379 |

Total B/C ratio **0.7300**: C is about **37% longer** overall. No padding or truncation
was used. The contrast is feasible, but these examples do not yet establish a scalable,
length-controlled recipe. Three development pairs are not enough to launch SFT.

![Measured reasoning lengths](../../output/nonmoral_paired_reuse_pilot/restart_packet/token_lengths.png)

## Independent baseline lane

The [existing-model evaluation proposal](baseline_inventory.md#launch-proposal-three-existing-checkpoints-one-measurement-protocol)
is prepared: nonmoral, math and historical Table-2-only, three passes each over the
same 80 cells (**720 rollouts; zero training**). It can proceed independently of new data.
A proposed $80 ceiling is an allocation, not a measured forecast or an implemented hard
all-in cap. Current rental quote, judge-spend control and launch preflight remain before
requesting approval. Three separate target invocations avoid pooling different recipes.

Existing numbers remain **73/400 = 18.25% nonmoral** and **98/240 = 40.83% math**, with
context/harness differences. No new alignment effect was measured in this reset.

## Next step, not launched

User reaffirmed on 2026-09-09: SFT uses **two H200 GPUs on RunPod with dynamic
batching**, the shared `configs/train/sft.yaml` and Qwen3.6 model profile; one seed
per condition. Use the repository provisioner, CUDA/masking checks, owned-pod watchdog
and verified teardown. This specifies the training setup, not authorization to rent.

Review the actual surviving pairs to assess the scientific contrast. Before scaling,
resolve controls that invent their own checks and their excess verbosity using the
recorded failures; do not start another open-ended calibration exercise. Any further
paid data batch or evaluation gets a concrete scope and spending proposal first.
Historical reasoned overrides and the new compliant paired recipe remain different
interventions. Broader domains, natural task complexity and nonmoral stakes are separate
extensions; the current pilot does not block the existing-checkpoint baseline comparison.
