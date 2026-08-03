# Constitution dose sweep — Petri audit

Measures **frequency of violations** of the v1 distilled constitution across four
Qwen3.6-27B arms: base plus difficult-advice LoRA at 10%, 20% and 40% of the SFT
mixture. Published run: `LASR-Callum/2026-08-01-petri-constitution-dose-sweep-v2`.

**Reproducing it end to end: [`docs/petri_dose_sweep_runbook.md`](../../../../../docs/petri_dose_sweep_runbook.md)** — which script runs on which
machine, in what order, and what each hands to the next.

## Headline

| arm | violation frequency | 95% CI |
|---|---|---|
| base (0%) | 27.2% (40/147) | [20.2%, 35.2%] |
| 10% | 24.3% (35/144) | [17.6%, 32.1%] |
| 20% | 28.0% (40/143) | [20.8%, 36.1%] |
| **40%** | **16.5% (23/139)** | [10.8%, 23.8%] |

Paired McNemar for the 40% arm against base is p = 0.029 (22 scenarios fixed, 9
broken). The pre-specified primary test — paired severity — is −0.47,
95% CI [−0.95, +0.01], and does **not** confirm it. Reported as a lead.

## Read this before quoting a number

Two of the 28 seeds are **controls**: benign scenarios containing nothing to
violate. The judge flagged them anyway, at **17% / 8% / 36% / 45%** across the
four arms. Every one of those is a false positive by construction, and the rate
is *not* constant across arms — the arm carrying the significant result is the
arm the judge is least reliable on.

So these are **judge flags, not behaviour rates**. `adjudication.py` builds a
review page over the 150 flagged transcripts, controls first, so a human can
rule on them. Until that pass is done, treat every rate here as an upper bound.

A second caveat that affects the error bars: seeds vary enormously (13 of base's
26 test seeds never violated once in 6 repeats; 3 violated every time), so the
six repeats of a seed are not six independent observations. Intraclass
correlation is 0.27–0.52, giving an effective n of 43–64 rather than 139–147.
The intervals in `results.json` are plain Clopper-Pearson and are therefore
roughly 1.5–1.8× too narrow. The paired tests are largely immune, since pairing
removes the seed effect.

## Running the audits

**The GPU orchestration is deliberately not in this repo.** It was Windows
PowerShell driving a RunPod pod, and it depended on the provider/secrets/
heartbeat tooling under `experiments/vulnerabilities/`, which was removed when
the frozen audit record was deleted. This repo is bash + Python and has since
moved to a single root `.env`, so porting those scripts unchanged would have
reintroduced a deleted tree and a superseded secrets model.

They remain readable at tag **`petri-audit-backup-20260801`** (paths under
`experiments/teaching-claude-why/petri/scripts/`). Re-running the sweep needs:

- a GPU serving base + the three LoRA arms from **one** vLLM process
  (`--enable-lora`); serving them separately reintroduces per-arm stack variance
- the `petri-subscription` provider in `../petri-subscription/`, which routes the
  auditor through the Claude Agent SDK — this is what made the run affordable
  (the auditor spent 4.6M output tokens at $0 cash)
- `--max-tokens 4096`. Measured, not guessed: at 700 the base arm returned
  `finish=length` with empty content while the tuned arms answered normally,
  which would have manufactured a dose-response out of a truncation bug

`scripts/serve_petri_arms.sh` and `scripts/bootstrap_petri_arms.sh` did come
across: they are plain bash that runs on the box, they carry the non-negotiable
vLLM settings, and they have no dependency on the removed tree.

Everything downstream of the `.eval` logs runs locally with no GPU and no
provisioning — see `scripts/run_petri_analysis.sh`. One-off probes used while
building the rubric (dumping the exact judge prompt, smoke-testing the answer
schema) live in `scratch/petri/`.

## Layout

| path | what |
|---|---|
| `seeds/` | the 28-seed battery, covering 44/44 atomic elements, 5 agentic, 2 controls |
| `seeds.py` | regenerates `seeds/` from `configs/petri/seed_specs.yaml` so every seed carries a byte-identical hard-constraints block |
| `rejudge.py` | one judge, one transport, every arm — removes judge variance between arms |
| `stats.py` | the statistics: Clopper-Pearson, exact McNemar, retention gates, dimension groups, report rendering |
| `analyse.py` | thin layer over `stats.py` that reads the re-judged sidecars and writes `results.json` |
| `plots.py` | the headline SVG (no plotting dependency) + decomposition + markdown mirror |
| `export.py` | the publishable bundle (scenarios / transcripts / scores) |
| `manifest.py` | Hub manifest + per-transcript shards; run metadata comes from `configs/petri/manifest.yaml` |
| `adjudication.py` | the human review page over flagged transcripts |
| `check_arm.py` | integrity gate — verifies target model events exist, not just that samples do |
| `validate_judge.py` | agreement between two judge models, on the measurement rather than on price |

Bulk artifacts (`.eval` logs, transcript shards, the 4.3MB transcripts JSONL)
are **not** committed — they live on the Hub at the revision pinned in the
dashboard entry, per the repository's data policy.
