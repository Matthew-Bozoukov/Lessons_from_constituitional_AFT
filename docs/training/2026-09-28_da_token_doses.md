<!-- ABOUTME: Results and provenance for the September 25 DA token-dose campaign. -->
<!-- ABOUTME: Records two once-trained seed-0 checkpoints and their completed evaluations. -->

# September 25 DA corpus: 5% and 25% supervised-token mixtures

Completed September 28, 2026.

**Hypothesis/method.** Test whether increasing September 25 difficult-advice
supervision improves ODCV misconduct and MASK honesty relative to the existing
nosynth and DA-15 checkpoints. Both new mixtures use unchanged rows from
`dougalldeepmind/2026-09-25-da-synth@618060e15315c71d7ffb9a8839198b08520a8771`
and the exact DA-15 replay source,
`dougalldeepmind/2026-09-22-nosynth-mix@378ec1ee0f0eea9294683779438b839e52b9700a`.
DA-5 contains 9,725 rows (209 DA), 4,842,037 supervised tokens, and **5.019623%**
DA supervised tokens; DA-25 contains 8,582 rows (1,040 DA), 4,841,794 supervised
tokens, and **24.998833%** DA supervised tokens. Source membership, multiplicities,
whole-row rounding, HF payload hashes and JSON loading were verified.

Each arm was trained **exactly once**, seed 0, one epoch, BF16 LoRA rank 64/alpha
128/dropout .05, learning rate 1e-4, `token_mean` loss, packing and dynamic token
budget 8,000, on one H200. Loss aggregates supervised tokens across the optimizer
step; rows are not given equal total loss weight. Base:
`Qwen/Qwen3.6-27B@6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
All 608/537 training steps completed with finite logged losses and gradients;
checkpoint and final-output backups were verified locally. Each adapter then ran
ODCV (three 80-scenario passes, H100) and full MASK (1,000 rows, one pass, H200)
in parallel, with CPU drivers, Docker and judging local. ODCV used concurrency 16
per arm; MASK used 192. Existing control/DA-15 results are pinned references.

**Result.** Both new ODCV runs contain 240 clean, nonempty transcripts, no retries
or reconstructed transcripts, and complete misconduct and progress judgments.
All four result repos were read back at their recorded revisions.

| Arm | ODCV misconduct % [95% CI] | ODCV progress | MASK honesty |
|---|---:|---:|---:|
| Nosynth reference | 45.4 [32.8, 58.6] | 4.95 | 56.9 |
| DA-5 | 20.4 [12.2, 32.1] | 4.91 | 72.0 |
| DA-15 reference | 9.2 [4.3, 18.6] | 4.95 | 90.2 |
| DA-25 | 7.5 [3.5, 15.5] | 4.99 | 92.2 |

**Interpretation.** Relative to nosynth, DA-5 lowered observed misconduct by 25.0
percentage points and raised MASK honesty by 15.1 points. DA-25 lowered misconduct
by 37.9 points and raised honesty by 35.3 points. Task-progress means remained
near the reference. Most of the observed gain was already present at DA-15;
the additional DA-15 to DA-25 changes were 1.7 points lower misconduct and 2.0
points higher honesty. These are descriptive comparisons of single checkpoints,
not estimates of training-seed robustness or a paired significance test.

| Arm | Dataset | Adapter | ODCV | MASK |
|---|---|---|---|---|
| DA-5 | [HF](https://huggingface.co/datasets/dougalldeepmind/2026-09-28-da-5-mix/tree/fd2c0984e31df02158d9c9a555c99d6a0f1d35e7) | [HF](https://huggingface.co/dougalldeepmind/2026-09-28-qwen36-0-da-5/tree/112f52fe593531d4c4af479e1c118a40be36e430) | [HF](https://huggingface.co/datasets/dougalldeepmind/2026-09-28-odcv-qwen36-0-da-5/tree/538574bf0b272817ad12d92037973853fc6de953) | [HF](https://huggingface.co/datasets/dougalldeepmind/2026-09-28-mask-qwen36-0-da-5/tree/9f8937ad1ec167c7485213a0298dea2c72d1faf6) |
| DA-25 | [HF](https://huggingface.co/datasets/dougalldeepmind/2026-09-28-da-25-mix/tree/476a41c0352df5b56103924c7f1a1c1b7bcdb0fe) | [HF](https://huggingface.co/dougalldeepmind/2026-09-28-qwen36-0-da-25/tree/57741f26c09f7abbac2f5cfa532246404e0aa0e3) | [HF](https://huggingface.co/datasets/dougalldeepmind/2026-09-28-odcv-qwen36-0-da-25/tree/4a8a1d295725a5131e67260987c57f607e29f1df) | [HF](https://huggingface.co/datasets/dougalldeepmind/2026-09-28-mask-qwen36-0-da-25/tree/93ae25c62a33d4b3c99b0dc560dd6ffb1a9332e7) |

**Recovery, cost and limits.** Rental-to-last-publication elapsed time was 2h33m
(13:12:58–15:46:07 UTC), including provisioning recovery. Three pods never exposed
SSH, one allocation returned HTTP 500, and one ODCV launch rejected interspersed
CLI overrides before rollout work; all were recovered without retraining or
rerolling valid outputs. Failed rentals are included in the conservative
GPU/storage estimate **$28.78**. The shared-account API usage increase measured
at 15:54 UTC was **$15.35**, an upper bound rather than attributed campaign usage;
the combined conservative estimate is **$44.13**, below the $200 cap, not a
provider invoice. All ten owned pods, including failed allocations, were verified
absent. Other sessions and their resources were not modified.

MASK generation errors were 34/4,436 (0.77%) for DA-5 and 6/4,429 (0.14%) for
DA-25; both are retained under the standard protocol's 5% cap. Empty-content
counts were 2 and 9, scored as evasion. One training seed and one MASK pass per
arm do not establish seed robustness; ODCV intervals cover scenario/pass
variation. Historical MASK serving concurrency differs. The ordered point
estimates support a dose trend, but the overlapping DA-15/DA-25 intervals do not
establish that 25% is better than 15%.


## Reference artifacts

| Reference | Adapter | ODCV | MASK |
|---|---|---|---|
| Nosynth | [HF](https://huggingface.co/dougalldeepmind/2026-09-22-qwen36-0-nosynth/tree/0b14213ef58a1c441d84787967c3bca9dde30d78) | [HF](https://huggingface.co/datasets/dougalldeepmind/2026-09-22-odcv-qwen36-0-nosynth/tree/2c9607c4c53582125badae149eed96917ab4ed0d) | [HF](https://huggingface.co/datasets/dougalldeepmind/2026-09-22-mask-qwen36-0-nosynth/tree/d8ddc80e05d1e3a74859906b0fe2916fca01fb78) |
| DA-15 | [HF](https://huggingface.co/dougalldeepmind/2026-09-25-qwen36-0-da-15/tree/b13cfe9891671f05c73a12078796e212944d22e4) | [HF](https://huggingface.co/datasets/dougalldeepmind/2026-09-26-odcv-qwen36-0-da-15/tree/44aeddf5590d128cd6031f3bbda34ff9b6408f00) | [HF](https://huggingface.co/datasets/dougalldeepmind/2026-09-26-mask-qwen36-0-da-15/tree/dc6b2a3fb51d5ce21ef68afab31ef70c703aa6e3) |

## Reproduction and retained evidence

The model repos contain `train_config.yaml`, `training_meta.json` and the full
training log. Evaluation repos contain `metadata/run_meta.json`, resolved
configuration, raw rollouts and judgments. Use the immutable revisions above;
do not substitute the latest artifact with a similar name.

Mixtures were built with the standard `configs/data/mixture/da.yaml` recipe,
`synthetic_pct=5` or `synthetic_pct=25`. Source pins above and published mixture
metadata define the inputs. Training used the existing entrypoint:

```powershell
uv run --frozen train --config configs/train/sft.yaml model=qwen36 seed=0 wandb=true data_repo=<pinned-mixture-repo> data_revision=<mixture-sha> base_model_revision=6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 hf_repo=<new-model-name> train.loss_agg=token_mean train.packing=true train.token_budget=8000 train.epochs=1
```

ODCV used the existing `scratch.da_supervision.odcv_eval` adapter for a counted
pilot, Windows Docker addressing, isolated Compose projects, disabled shared
daemon pruning, and GPU release before local judging. MASK used
`src.eval.run_eval`. Keep CLI options before the trailing OmegaConf overrides.
The launch recipe and protocol are recorded here; reproducing them incurs new
training/evaluation costs and is not part of this completed campaign.

Temporary launchers, watchdog orchestration, recovery scripts and their
compatibility workaround were removed before integration. Their exact historical
implementation remains in commit `5630b79c`; no campaign-specific runtime code
is required by main. Local receipts, verified checkpoint backups and the dated
PNG/SVG comparison remain under `output/` in
`C:/Users/nikak/da-sep25-5-25`. The HF artifacts are the durable result record.

**Next step:** use these pinned checkpoints for downstream evaluation. Any extra
seeds or repeats should be a separate experiment.
