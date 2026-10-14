<!-- ABOUTME: Navigation and reproduction boundaries for the October 2026 GPT-OSS MMLU campaign. -->
<!-- ABOUTME: Experiment orchestration stays in scratch; published artifacts pin the executed source. -->

# GPT-OSS MMLU campaign

Completed October 14, 2026. This is the matched **570-question screen**, not full
MMLU. Results, immutable HF links and interpretation limits are in
[the campaign index](../../docs/gptoss_campaigns.md) and [research log](../../docs/LOG.md).

These are one-off campaign drivers, not a second MMLU implementation. The scorer,
subset selection and publication contract come from `src/eval/capabilities/mmlu/`
and `src/eval/run_eval.py`; native rendering and accounting come from
`src/infra/endpoints/tinker*.py`. No production module should import this directory.

| File | Responsibility |
|---|---|
| `protocol.yaml` | Approved overrides of `configs/eval/mmlu.yaml` |
| `campaign.py` | Frozen selection, mock load qualification, bounded smoke and independent arm orchestration |
| `bridge.py` | Campaign budget/resource admission, native sampler retry policy and synthetic sampler |
| `host.py` | Owned Windows Docker driver and keep-awake/resource guard |
| `preflight.py` | Exact prompt-token audit before inference |
| `freeze.py` | Source manifest and archive for the executed checkout |
| `closeout.py` | Saved-result audit, paired statistics and publication-only recovery; no inference |
| `test_campaign.py` | Offline Linux shared-budget and reservation checks |

The executed inference source is `23d618faeae35318574fd8062b3461b702f78168`;
smoke used `2e43eeba6610c6289c69bb02a43cb58b917917ad`. Cleanup formatting is not
the executed source. Fetch the published source archives and resolved configs to
reproduce the historical protocol. Fixed dates, paths, checkpoints and container
names in these drivers describe that closed campaign; a new run requires its own
identity and authorization. Do not restart these owners to inspect results.

Run from the repository root. The local runtime was one Linux Docker container
with the locked CPU environment in `scratch/swebench_cpu_env` and the locked shim
environment in `src/infra/endpoints/tinker_env`. Offline tests (Linux CPU environment):

```bash
uv run --project scratch/swebench_cpu_env --frozen python -m pytest -q scratch/gptoss_mmlu/test_campaign.py
```

Raw records, traces, ledgers, qualification failures and publication receipts stay
under gitignored `output/gptoss_mmlu_20261014/` locally and in the pinned HF artifacts.
The mock load ledger is simulated, not paid spend. Empty final answers and other
parse failures remain wrong under the original final-content scoring protocol;
reasoning is retained separately. Control's failed parse-health gate must accompany
its score. No re-extraction or rescoring was performed during code cleanup.
