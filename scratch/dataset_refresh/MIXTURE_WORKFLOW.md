# Final mixture preparation

Use this only after both synthetic corpora have passed their own audits and published exactly 716 rows as default `dataset.jsonl`. Obtain the actual immutable HF commit for each publication. The preparation helper performs no downloads, model calls, builds, publishing or training. It refuses missing/moving/placeholder revisions and mismatched corpus names.

Run from the repository root, replacing the bracketed values with the actual published repo IDs and full commits:

```powershell
uv run --no-sync python -m scratch.dataset_refresh.prepare_mixtures prepare --output output/refresh_mixture_preparation --low-repo <org/date-da-lowstakes-refresh-synth> --low-revision <full-40-character-HF-commit> --nonmoral-repo <org/date-nonmoral-advice-synth> --nonmoral-revision <full-40-character-HF-commit>
```

This writes undated `da-lowstakes-refresh.yaml` and `nonmoral-advice.yaml`, a frozen replay-proportions snapshot, `preparation.json`, and `BUILD_COMMANDS.txt`. Configs specify seed 0, Qwen/Qwen3.6-27B token counting, a strict 8192-token cap, exact 716 synthetic rows, and 10,000 total rows. `synthetic_pct: 7` is the rounded naming suffix; the actual share is 7.16%. Neither config contains `hf`, `filter`, or `reasoning_backfill`, so the existing builder only reads published data and builds locally. No generator is inherited from the base proportions file.

Build each arm with the existing CLI, as listed in `BUILD_COMMANDS.txt`:

```powershell
uv run mix --config output/refresh_mixture_preparation/da-lowstakes-refresh.yaml
uv run mix --config output/refresh_mixture_preparation/nonmoral-advice.yaml
```

Each command prints its timestamped output directory. The existing builder exits the process after writing its artifact; invoke it as the CLI, not an in-process function. The canonical synthetic intake is `dataset: org/repo` at the supplied revision, with native reasoning and `balance_by: trait_id`. A corpus with exactly 716 rows cannot silently drop an overlength row: the exact quota then fails. Do not substitute smoke outputs, change seeds, override counts, or reuse an older synthetic source to fill a shortfall.

Both arms reuse `dougalldeepmind/2026-09-08-nosynth-mix@7e991f58e86eff0b0a9f15a54ebeddfffb5b14dd/mixture.jsonl`. Largest-remainder replay quotas are: no_robots 2580; tulu3_if 1366; numinamath_cot 987; self_oss_instruct 988; smol_constraints 979; apigen_function_calling 978; smol_summarize 914; lima 291; longalign 201. Their sum is 9284. Per-source sampling and final shuffling use seed 0 independently of corpus identity. The subsequent audit verifies identical replay payloads **and positions**, rather than relying on that design claim alone.

Audit the two actual output files together:

```powershell
uv run --no-sync python -m scratch.dataset_refresh.validate_mixtures --mixtures <low-build-directory>/mixture.jsonl <nonmoral-build-directory>/mixture.jsonl --output output/refresh_mixture_preparation/mixture_validation.json
```

This read-only audit downloads the exact pinned replay file and uses the cached Qwen tokenizer. It checks all rows without truncation, actual assistant masks, both dosage counts, source quotas, duplicate synthetic prompts, replay identity and positions. It reports rendered tokens, supervised tokens, and separately tokenized reasoning/final fields. Row share is not token share or loss-weight share. This is not training or model evaluation.

Prepare publication cards separately, supplying the frozen generation config that produced each published corpus:

```powershell
uv run --no-sync python -m scratch.dataset_refresh.prepare_mixtures card --config output/refresh_mixture_preparation/da-lowstakes-refresh.yaml --mixture-dir <low-build-directory> --audit output/refresh_mixture_preparation/mixture_validation.json --synth-config <frozen-low-synth-run>/da-lowstakes-refresh/config.json
uv run --no-sync python -m scratch.dataset_refresh.prepare_mixtures card --config output/refresh_mixture_preparation/nonmoral-advice.yaml --mixture-dir <nonmoral-build-directory> --audit output/refresh_mixture_preparation/mixture_validation.json --synth-config <frozen-nonmoral-synth-run>/nonmoral-advice/config.json
```

The card step verifies the prepared config, complete non-smoke build, exact source counts, inherited replay-reasoning provenance, and current hashes of **both** audited mixtures. It writes `README.md`, `card_fields.json`, `card_front_matter.json`, and `publication_plan.json`, and copies the resolved mixture config, passing audit and frozen synthetic config into the build directory. The name is minted by `src.naming.mix_name` from the actual build date, e.g. `<build-date>-da-lowstakes-refresh-7-mix`; it is not dated at upload time.

For publication, use the repository's existing `push_run_dir` with the name from `publication_plan.json`, fields from `card_fields.json`, and front matter from `card_front_matter.json`. The environment's `HF_ORG` remains the destination authority. Recheck the saved audit hashes before upload if anything changed. This workflow helper intentionally has no publication command; production orchestration owns that action. Training and evaluation remain separately gated by the user's later confirmation.
