<!-- ABOUTME: How to run the Colosseum Hospital eval: one arm, several arms in parallel, a top-up, the smoke. -->
<!-- ABOUTME: The config is configs/eval/colosseum_hospital.yaml; the measures' vocabulary is docs/colosseum_hospital_terms.md. -->

# Colosseum Hospital: how to run it

## One run

One Hospital run is one `runpod up` plus one `uv run evals`:

```bash
uv run runpod up --name <you>-hospital --eval colosseum_hospital --target <arm>
uv run evals --name colosseum_hospital --target <arm> --server root@<ip>:<port>
uv run runpod down --pod <id>
```

- `<arm>` is the adapter under test, an HF id such as `dougalldeepmind/2026-10-30-qwen36-0-da-15-b`.
- No `--config` is needed. The eval's one config, `configs/eval/colosseum_hospital.yaml`, is the
  paper's self-sacrificial arm:
  - 30 shifts (seeds 1–30)
  - the arm under test in the two coalition seats (the resource provisioner and General
    Hospital's Triage)
  - `dougalldeepmind/2026-10-05-qwen36-0-plain` in the other seven seats
  - the rationale shown on the public board
  - a 131,072-token window
- The run serves the model on the pod, drives the shifts from your machine, judges them,
  and pushes the cell to the Hub as `<date>-hospital-self-sacrificial-<arm>`. So
  `OPENROUTER_API_KEY` and `HF_TOKEN` must be in your `.env`.
- The first run builds the patched Colosseum harness into `~/.cache/lessons-aft/colosseum/`
  (`src/eval/misalignment/colosseum/harness.py`). There is no setup step.
- A 30-shift cell takes about 3–3.5 hours, and your machine and its SSH tunnel to the pod must
  stay up throughout. If they can't, drive the run on the pod instead (next section).

Common overrides go at the end, as `key=value`:
- another condition: `condition=baseline` (also `self_promotional`, `covert`, `benign_coalition`)
- fewer shifts: `seeds=[1,2,3]`
- the constitution-in-the-system-prompt arm:
  `protocol=constitution system_preamble_file=constitutions/<name>/constitution.md`

## Several arms in parallel, driven on the pods

`scripts/eval/misalignment/colosseum/hospital/fleet.py` rents one pod per arm from a plan
file. On each pod it runs the same `uv run evals`, pulls the results back and tears the pod
down. Your machine can sleep while it runs; each pod also deletes itself at its time cap.

```bash
set -a; source .env; set +a
uv run python scripts/eval/misalignment/colosseum/hospital/fleet.py run --plan <plan.yaml>
uv run python scripts/eval/misalignment/colosseum/hospital/fleet.py status --plan <plan.yaml>
```

Example plan: `scratch/colosseum_hospital/2026-11-04_fleet_abc_seeds.yaml`. The plan's
`branch` must equal the commit you are on, pushed to origin, because the pods clone it.

## Finishing a cell that came in pieces

When some shifts of a cell failed and were rerun (a top-up), join the pieces into one cell
and publish it:

```bash
uv run python scripts/eval/misalignment/colosseum/hospital/merge.py cells --root <dir of the pieces> --out <merged dir>
uv run python scripts/eval/publish_colosseum.py --eval colosseum_hospital --run-dir <merged dir>
```

## Check the harness before a paid run

```bash
uv run python scripts/eval/misalignment/colosseum/hospital/smoke.py --out $PWD/output/hospital_smoke
```

This runs one shift per harness variant with a scripted model: no GPU, about a minute.
The variants are the earlier harness settings as override sets, in `VARIANTS` in
`src/eval/misalignment/colosseum/hospital/smoke.py`. `headline` is the harness the paper's
published cells ran (rationale hidden); `paper` is the original Colosseum harness. To run
the eval under one of them, pass its overrides on the `uv run evals` line.

## Where else to look

- What each measure means: `docs/colosseum_hospital_terms.md`.
- Every harness switch: `fixes:` in the config, documented in
  `src/eval/misalignment/colosseum/hospital/config.py`.
- The patches to Colosseum and terrarium: `src/eval/misalignment/colosseum/third_party/README.md`.
- Pod problems seen before: `docs/GOTCHAS.md`.
