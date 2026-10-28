# Petri constitution-tenet audit

90 scenarios, one per constitution tenet: nine principle families (P1–P9) of ten. The
auditor is Claude Sonnet 5 and the judge Gemini 3 Flash, both through OpenRouter. Each
scenario is one transcript of up to 25 turns, and the judge scores it on Petri's
`concerning` dimension (1–10).

The headline number is the share of scenarios scored **`concerning >= 4`**, the rubric's
"mild" anchor. It means "at least mildly concerning", not "harmful": the rubric puts clearly
harmful above 6. `summary.json` also records `>= 6` and the mean.

Every published run is a `dougalldeepmind/<date>-petri-<model>` dataset, for example
`2026-10-28-petri-qwen36-0-da-15`.

## What you need

- **`.env`** at the repo root with `OPENROUTER_API_KEY` (auditor and judge), `HF_TOKEN` and
  `HF_ORG=dougalldeepmind` (publishing). `run_audit.py` ignores stale shell exports of these
  and reads `.env`.
- **A GPU host with vLLM and the target's weights.** Petri is not a registered eval, so rent
  the box MASK uses, an H200:

  ```
  uv run runpod up matboz-petri-<arm> --eval mask --target <hf adapter or model> --push-env
  ```

  Note the `root@<ip>:<port>` it prints. Wait for `READY` in the pod's boot log before
  starting.
- **OpenRouter credit.** The auditor is most of the cost: one scenario measured about 250k
  input tokens and 18k output tokens on Sonnet 5. Check the balance before a full run.
  Running out partway leaves unscored seeds; resume them later (see below).
- Petri itself is **not** a project dependency. `run_audit.py` adds `inspect-petri==3.1.0` to
  the child process that runs the audit, through `uv run --with`, so there is nothing to
  install.

## 1. Smoke test: two seeds, minutes, cents

```
uv run python scripts/eval/audits/petri/run_audit.py \
    --server root@<ip>:<port> --target dougalldeepmind/<adapter> --limit 2
```

Logs land in `output/petri/<timestamp>_<adapter>/logs/*.eval`. Open one with
`uv run inspect view` and check that the target answered and the judge scored it.

## 2. Full run: all 90 seeds

```
uv run python scripts/eval/audits/petri/run_audit.py \
    --server root@<ip>:<port> --target dougalldeepmind/<adapter> --connections 24
```

- `--connections`: how many scenarios run at once (default 8). The October runs used 8–24.
  The GPU is billed by wall-clock either way, so more connections means a cheaper GPU bill.
- Without `--server`, the target is served on this machine, which must have a GPU.
- The driver aborts the audit if the target stops answering 3 health probes in a row, so a
  dead tunnel does not keep paying the auditor. Scored seeds stay in the log.

The pod is **not** torn down for you: `uv run runpod down --pod <id>` when finished.

### Constitution in the system prompt

To run an arm with the constitution prepended to every target system message:

```
uv run python scripts/eval/audits/petri/run_audit.py --server ... --target ... \
    --preamble constitutions/claude_distilled_09_principles/constitution.md
```

Verify the injection on a smoke run before paying for all 90. This reads the transcripts the
target actually received:

```
uv run python scripts/eval/audits/petri/verify_target_preamble.py \
    output/petri/<run>/logs/<log>.eval constitutions/claude_distilled_09_principles/constitution.md
```

It exits 0 only when every target system message starts with the file. If the context
window is too small for constitution plus scenario, raise `--context-window`: the October
plain+constitution run used 65536.

## 3. Resume or fill gaps

- **An interrupted run** (credit ran out, the tunnel died, Ctrl-C) resumes from its own log.
  Only unscored samples rerun:

  ```
  uv run python scripts/eval/audits/petri/run_audit.py --server ... --target ... \
      --retry output/petri/<run>/logs/<log>.eval --port <the port the log recorded>
  ```

  The target must be served on the same port as before. The driver checks this and refuses
  otherwise, because a mismatch makes the judge score silence as `concerning = 1`.
- **Seeds still missing across one or more runs:**

  ```
  uv run python scripts/eval/audits/petri/missing_seeds.py output/petri/<missing> output/petri/<run1> [<run2> ...]
  uv run python scripts/eval/audits/petri/run_audit.py --server ... --target ... --seeds output/petri/<missing>
  ```

## 4. Publish

```
uv run python scripts/eval/audits/petri/push_results.py \
    --run_dirs output/petri/<run> --target dougalldeepmind/<adapter> \
    --repo <date>-petri-<model>   # e.g. 2026-10-28-petri-qwen36-0-da-15
```

- When seeds were rerun in separate runs, pass them all as a list:
  `--run_dirs '[output/petri/<run1>,output/petri/<run2>]'`. Each seed takes its score from the
  newest log in which its target answered.
- A seed is **not counted** when its last call to the target failed. Otherwise a dead endpoint
  would read as a well-behaved model.
- The push refuses while any seed lacks a score. Rerun those seeds first, or name each one with
  `--allow_missing <seed-id>` to publish without it. The card and summary then list it.
- For a constitution arm, add `--preamble <file> --context_window <n>`: both are recorded in
  the card and `summary.json`.
- The repo gets the raw `.eval` log(s), `scores.json` (seed → `concerning`), `summary.json`
  (≥4, ≥6, mean, per family, settings) and a card.

## Editing the scenarios

The seed bodies live in `seeds/p1.yaml` … `seeds/p9.yaml`, keyed by tenet. After editing,
rebuild the 90 seed files the audit reads:

```
uv run python scripts/eval/audits/petri/build_seeds.py
```

`tests/test_petri_tenet_audit.py` fails if the built files and the YAML disagree, so commit
both. Changing a seed changes the benchmark, and results from before and after the change
are not comparable.

## Files

| file | role |
|---|---|
| `run_audit.py` | serve the target, run the audit, health-watch it |
| `task.py` | Petri's `audit` task, plus the `--preamble` injection |
| `scores.py` | read `concerning` per seed from `.eval` logs, dropping dead-endpoint seeds |
| `push_results.py` | publish a run to the Hub |
| `missing_seeds.py` | collect the seeds still lacking a score |
| `verify_target_preamble.py` | check the constitution reached the target |
| `build_seeds.py`, `seeds/` | the 90 scenarios |

Thin command-line mirrors of each live in `scripts/eval/audits/petri/`.
