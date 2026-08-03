# Provider status

```text
[GPU ACTIVE] runpod | NVIDIA A100-SXM4-80GB | $1.49/h | provider balance $221.0447 | experiment GPU budget $78.0866 remaining | elapsed 01:17 | vLLM unreachable
```

Generated: 2026-08-01T12:41:11Z  (refresh reason: periodic)

Every figure below is labelled with its basis: **exact-provider-reported**,
**locally-calculated**, **estimated**, or **unavailable**. Nothing is invented;
an unavailable figure is reported as unavailable.

| Field | Value |
| --- | --- |
| Provider | runpod |
| GPU | NVIDIA A100-SXM4-80GB |
| Hourly price | 1.49 USD/h _(exact-provider-reported)_ |
| Instance ID | `avq7oc22a2izxx` |
| Instance state | RUNNING |
| Provider balance | 221.0447 USD _(exact-provider-reported)_ |
| Starting provider balance | 230.4385407536 USD _(exact-provider-reported)_ |
| Provider balance delta | 9.3938 USD _(locally-calculated)_ |
| Elapsed runtime | 01:17 (1.2842 hours _(locally-calculated)_) |
| Estimated infrastructure cost | 1.9134 USD _(locally-calculated)_ |
| Experiment GPU budget remaining | 78.0866 USD _(locally-calculated)_ |
| Wall-clock remaining | 48.716 hours _(locally-calculated)_ |
| Hard deadline | 2026-08-03T13:24:08Z |
| Model-server health | unreachable |
| SSH-tunnel health | down |
| Last successful API refresh | 2026-08-01T12:41:11Z |
| Cleanup-watchdog state | armed (deadline 2026-08-03T13:24:08Z) |
| Anthropic spend | 0 USD _(locally-calculated)_ |
| Anthropic budget remaining | unavailable _(nothing invented)_ |

## Hard limits

- GPU spend cap: $80
- Wall-clock cap: 50 h
- Idle shutdown: 30 min

## Raw artifacts

- [status.json](./status.json)
- [status-history.jsonl](./status-history.jsonl) (append-only)
- [run-state.json](./run-state.json)
