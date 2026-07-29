# Provider status

```text
[GPU ACTIVE] runpod | NVIDIA A100-SXM4-80GB | $1.49/h | provider balance unavailable | experiment GPU budget $24.3891 remaining | elapsed 02:48 | vLLM healthy
```

Generated: 2026-07-29T09:29:38Z  (refresh reason: periodic)

Every figure below is labelled with its basis: **exact-provider-reported**,
**locally-calculated**, **estimated**, or **unavailable**. Nothing is invented;
an unavailable figure is reported as unavailable.

| Field | Value |
| --- | --- |
| Provider | runpod |
| GPU | NVIDIA A100-SXM4-80GB |
| Hourly price | 1.49 USD/h _(exact-provider-reported)_ |
| Instance ID | `p397jthrc130o2` |
| Instance state | query-failed |
| Provider balance | unavailable _(nothing invented)_ |
| Starting provider balance | 108.1129952401 USD _(exact-provider-reported)_ |
| Provider balance delta | unavailable _(nothing invented)_ |
| Elapsed runtime | 02:48 (1.806 hours _(locally-calculated)_) |
| Estimated infrastructure cost | 15.6109 USD _(locally-calculated)_ |
| Experiment GPU budget remaining | 24.3891 USD _(locally-calculated)_ |
| Wall-clock remaining | 34.194 hours _(locally-calculated)_ |
| Hard deadline | 2026-07-30T19:41:17Z |
| Model-server health | healthy |
| SSH-tunnel health | listening |
| Last successful API refresh | 2026-07-29T09:20:14Z |
| Cleanup-watchdog state | armed (deadline 2026-07-30T19:41:17Z) |
| Anthropic spend | 0 USD _(locally-calculated)_ |
| Anthropic budget remaining | unavailable _(nothing invented)_ |

## Hard limits

- GPU spend cap: $40
- Wall-clock cap: 36 h
- Idle shutdown: 30 min

## Raw artifacts

- [status.json](./status.json)
- [status-history.jsonl](./status-history.jsonl) (append-only)
- [run-state.json](./run-state.json)
