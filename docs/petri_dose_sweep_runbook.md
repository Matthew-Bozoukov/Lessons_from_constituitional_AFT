<!-- ABOUTME: End-to-end runbook for the constitution dose-sweep Petri audit: what each -->
<!-- ABOUTME: script does, where it runs, in what order, and what it hands to the next stage. -->

# Petri constitution dose sweep — runbook

Audits four Qwen3.6-27B arms (base + difficult-advice LoRA at 10/20/40% of the SFT
mixture) against the v1 constitution, and reports **frequency of violations** per arm.

Code: `src/eval/vulnerabilities/petri/constitution_sweep/`
Published run: `LASR-Callum/2026-08-01-petri-constitution-dose-sweep-v2`

## The shape of it in one picture

There are **two machines**. A rented GPU box serves the four models; your laptop runs
the auditor and everything after it. Three scripts, and each belongs to exactly one of
those machines:

```
  GPU BOX (rented, ~$1.50/h)                    LOCAL (your machine)
  ─────────────────────────────                 ──────────────────────────────
  1. scripts/bootstrap_petri_arms.sh
       installs vLLM, downloads weights
       ONCE per box
                │
                ▼
  2. scripts/serve_petri_arms.sh
       one vLLM process, four model names
       EVERY session; leave running
                │
                └──── SSH tunnel :8000 ────►  3. the audits  ← NOT IN THIS REPO, see below
                                                    writes output/petri/logs/<arm>/*.eval
                                                              │
                                                              ▼
                                              4. scripts/run_petri_analysis.sh
                                                   judge → stats → figures → export
                                                   no GPU, no tunnel needed
```

The only thing crossing between them is an SSH tunnel on port 8000, and the only
artifact that matters afterwards is the `.eval` logs in `output/petri/logs/`.

## Stage 1 — `scripts/bootstrap_petri_arms.sh` (on the box, once)

Installs vLLM and downloads Qwen3.6-27B plus the three LoRA adapters into
`/workspace/models`. Run it once when a box is first rented; it is idempotent, so
re-running on a warm box is a no-op that just verifies.

It deliberately does **not** reuse the sibling experiment's bootstrap, which pins
vLLM 0.11.0 — too old for this architecture.

**Needs:** a box with CUDA 13 drivers. vLLM 0.26 has no cu128 wheel, so a CUDA 12.8
box will fail at import after a 55GB download. Filter for CUDA 13 when provisioning.
**Produces:** `/workspace/models/{base,dose-10-90,dose-20-80,dose-40-60}`

## Stage 2 — `scripts/serve_petri_arms.sh` (on the box, every session)

Starts **one** vLLM process serving **four** model names via `--enable-lora`, then
verifies all four answer a tool-bearing request before it exits. Leave it running.

One process is not an optimisation. Four arms off a single base weight load means
every arm shares an identical serving stack, so the adapter is the only variable
between them. Serving them separately reintroduces per-arm stack variance into the
thing being measured.

**Settings that are not negotiable** (each cost someone a run):

| setting | why |
|---|---|
| `--tool-call-parser qwen3_xml` | without it tool calls come back as prose and every agentic seed silently fails |
| `--max-num-seqs 32` | higher wedges this hybrid Mamba/linear-attention model under concurrency |
| `MAX_MODEL_LEN` 65536 | the frozen scaffolds plus 12 turns exceed shorter windows |

**Needs:** stage 1 complete. **Produces:** an OpenAI-compatible endpoint on :8000
serving `base`, `dose-10-90`, `dose-20-80`, `dose-40-60`.

Then tunnel it to your machine: `ssh -N -L 8000:localhost:8000 <box>`. Start the
tunnel with a **backgrounded** process — one launched inside a foreground command dies
when that command returns.

## Stage 3 — running the audits (**not in this repo**)

This is the gap a reader will notice, and it is deliberate.

The orchestration that drove the published run was Windows PowerShell: it provisioned
the pod, held a heartbeat against a watchdog, ran Petri per arm, and gated between arms
on health and spend. It depended on the provider, secrets and heartbeat tooling under
`experiments/vulnerabilities/`, which was deleted with the frozen audit record, and on
a per-experiment secrets wrapper this repo has since replaced with a single root `.env`.
Porting it unchanged would have reintroduced a deleted tree and a superseded secrets
model, so it was left behind rather than half-migrated.

It remains readable at tag **`petri-audit-backup-20260801`**, under
`experiments/teaching-claude-why/petri/scripts/` — `Run-Grid.ps1` is the entry point.

To reproduce without it, run Petri directly against the tunnel, once per arm:

```bash
uv run inspect eval petri/audit \
  --model openai-api/vllm/<arm> \
  --model-base-url http://localhost:8000/v1 \
  -T seed_instructions=src/eval/vulnerabilities/petri/constitution_sweep/seeds \
  -T max_turns=12 -T realism_filter=0.6 \
  --max-tokens 4096 --epochs 6 --no-epochs-reducer \
  --log-dir output/petri/logs/<arm>
```

`--max-tokens 4096` is measured, not guessed. At 700 the base arm returned
`finish=length` with empty content while the tuned arms answered normally — which would
have manufactured a dose-response out of a truncation bug.

**Produces:** `output/petri/logs/<arm>/*.eval` — the only input the next stage needs.
Everything downstream is offline; the box can be destroyed once these exist.

**Destroy the box here.** It is billed by the hour and nothing below needs it.

## Stage 4 — `scripts/run_petri_analysis.sh` (local, no GPU)

Subcommands, in order. `all <slug>` chains them.

| command | does | spends |
|---|---|---|
| `rejudge` | re-scores every arm with one judge on one transport, so no arm's scores depend on which path they took | **~$30** Anthropic |
| `analyse` | rates, Clopper-Pearson intervals, McNemar, paired severity | — |
| `plots` | headline SVG + decomposition + markdown mirror | — |
| `adjudicate` | builds the human review page over flagged transcripts | — |
| `export <slug>` | the publishable bundle | — |
| `manifest <slug>` | Hub manifest + per-transcript shards | — |

```bash
scripts/run_petri_analysis.sh all 2026-08-01-constitution-dose-sweep
```

Reads `.eval` logs from `$PETRI_LOGS` (default `output/petri/logs`), writes under
`$PETRI_OUT` (default `output/petri`). `rejudge` is the only step that costs money.

Then publish and pin:

```bash
set -a; . ./.env; set +a
uv run python -m src.eval.vulnerabilities.petri.constitution_sweep.publish \
  LASR-Callum/<date>-petri-constitution-dose-sweep \
  output/petri/exports/<slug> "what changed"
```

Put the revision it prints into the dashboard entry's `hf_source.revision`. The pin is
what stops a later re-upload silently changing the numbers under a published writeup.

## Stage 5 — adjudication (**do not skip**)

Every rate the pipeline produces is a judge opinion, not a behaviour rate. Two of the
28 seeds are controls containing nothing to violate, and the judge flagged them at
**17% / 8% / 36% / 45%** across the four arms — all false positives by construction,
and *not* constant across arms.

```bash
scripts/run_petri_analysis.sh adjudicate
# then open output/petri/adjudication/review.html
```

Controls come first in the review order deliberately: you already know they are wrong,
so they calibrate you on *how* the judge errs before you rule on cases where you cannot
check the answer independently. Until that pass is done, treat every rate as an upper
bound.

## Cost and wall-clock, from the published run

| | |
|---|---|
| GPU | ~29 h @ $1.49/h ≈ $43 — of which only 9.5 h was compute |
| Anthropic (Haiku realism) | $30.95, exact |
| Anthropic (Sonnet re-judge) | ~$29.3 |
| auditor (Claude subscription) | **$0** — 4.6M output tokens |
| **total** | **~$103 for 672 audits, $0.15 each** |

**Destroy the box between stage 3 and stage 4.** Roughly 19 of those 29 GPU hours were
idle, waiting on subscription quota — about $28, more than the compute itself. Hold the
box across gaps under ~2 h; tear it down above that.
