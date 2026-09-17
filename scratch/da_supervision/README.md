<!-- ABOUTME: Entry points and boundaries for the September DA supervision experiment utilities. -->
<!-- ABOUTME: These scripts are campaign-specific; final findings and canonical artifacts are linked below. -->

# DA supervision experiment tools

Read the [final report](../../docs/da_supervision_rerun_2026-09-16.md) for results,
masking semantics and immutable HF links. Training and evaluation are complete;
all owned pods are terminated and the scheduler is deleted. No scripts here should
run automatically on import or as part of routine tests.

## Reproduce the chart

From the repository root:

```powershell
uv run python -m scratch.da_supervision.odcv_chart --config scratch/da_supervision/odcv_chart.yaml
```

`odcv_chart.yaml` pins all five evaluated datasets. The script downloads their
scores and metadata, checks protocol compatibility and severity threshold, and
writes PNG/SVG/PDF plus CSV/JSON and a source-linked report under `output/`.
No paid model calls or uploads occur. Plot dates and destination are in the config.

## Rebuild datasets

```powershell
uv run python -m scratch.da_supervision.build
```

`config.yaml` pins the DA corpus, nosynth replay and tokenizer. The builder selects
replay once, constructs the three paired mixtures, and checks every rendered label.
It writes local outputs without publishing. `--publish` explicitly uploads newly
dated mixtures and refuses existing dataset/model names. The published manifests
retain the selection lineage and actual build revision.

## Training and evaluation utilities

- `owner.py PLAN ARM --out OUTPUT` owns one training rental and preservation
  lifecycle. A plan is produced by the dataset builder. Training settings that
  differ from the normal recipe are explicit `train_overrides` in the plan;
  `configs/train/sft.yaml` remains unchanged.
- `hub_backup.py` uploads and verifies the full training archive from its pod.
- `odcv_eval.py` wraps the standard evaluator with one counted pilot, Windows
  tunnel/path fixes and GPU release before local judging. It writes pilot evidence
  under `metadata/` and disables shared-daemon pruning.
- `odcv_owner.py ARM --plan PLAN` owns a bounded single-GPU evaluation rental.
  `odcv.yaml` is the protocol. `odcv_plan.yaml` and `odcv_remaining_plan.yaml` are
  **historical launch manifests**, with historical targets, ports and output roots;
  they are not instructions to launch another campaign.
- `odcv_awake.py PLAN` is a temporary Windows awake helper for live owners.
- `odcv_verify.py ARM [--publish]` audits the original local completed-run tree,
  compares protocol against the CoT receipt, and verifies uploaded file hashes.
  It needs the original ignored local receipts; it is not a read-only Hub viewer.
  Without `--publish` it still writes audit metadata to HF; `--publish` recovers
  publication of an already generated and judged package.

GPU owners rent through `src.infra.runpod`, register independent watchdogs and only
tear down their recorded pods. All paid launches require a new concrete campaign
request; these files alone are not authorization to rerun an experiment.

## Checks and history

```powershell
uv run pytest tests/test_masking.py tests/test_mask_gate.py tests/test_build_mixture.py scratch/da_supervision/test_campaign.py -q
```

Reusable masking tests stay in `tests/`; campaign tests stay here, so core code and
tests do not depend on `scratch/`. The scripts under `archive/` record one-off
single-to-dual GPU recovery and bootstrap handoff and depend on retired pod/local
state. They are historical references, not maintained launch commands.
The original campaign history is preserved on `codex/da-supervision-rerun` through
commit `9922f8fe`; its provenance is not rewritten by the cleaned integration.


## Integration validation, September 17, 2026

- 158 targeted checks passed, including cached real-Qwen tokenizer checks of the
  CoT/answer token partition, unchanged answer input IDs and closing-tag masking.
- The pinned five-arm chart reproduced in a fresh integration environment.
- Full suite: 1,780 passed, 46 failed, 15 skipped. All 46 non-passing cases also
  failed on unchanged main `63441012` in the same environment (45 failures and
  one setup error when selected individually). These involve unrelated existing
  environment/encoding/fixture issues; the broad suite is not claimed green.
- Syntax, document links, naming and diff checks passed. The normal SFT config,
  CLAUDE.md, AGENTS.md and TODO.md were unchanged. No paid work ran during integration.
