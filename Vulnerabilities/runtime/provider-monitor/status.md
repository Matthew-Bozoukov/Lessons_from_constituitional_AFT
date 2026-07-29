# Provider status

```text
[GPU ACTIVE] runpod | NVIDIA A100-SXM4-80GB | $1.49/h | provider balance $94.8405 | experiment GPU budget $39.8435 remaining | elapsed 00:06 | vLLM unreachable
```

Generated: 2026-07-29T07:47:35Z  (refresh reason: periodic)

Every figure below is labelled with its basis: **exact-provider-reported**,
**locally-calculated**, **estimated**, or **unavailable**. Nothing is invented;
an unavailable figure is reported as unavailable.

| Field | Value |
| --- | --- |
| Provider | runpod |
| GPU | NVIDIA A100-SXM4-80GB |
| Hourly price | 1.49 USD/h _(exact-provider-reported)_ |
| Instance ID | `p397jthrc130o2` |
| Instance state | RUNNING |
| Provider balance | 94.8405 USD _(exact-provider-reported)_ |
| Starting provider balance | 108.1129952401 USD _(exact-provider-reported)_ |
| Provider balance delta | 13.2725 USD _(locally-calculated)_ |
| Elapsed runtime | 00:06 (0.1051 hours _(locally-calculated)_) |
| Estimated infrastructure cost | 0.1565 USD _(locally-calculated)_ |
| Experiment GPU budget remaining | 39.8435 USD _(locally-calculated)_ |
| Wall-clock remaining | 35.895 hours _(locally-calculated)_ |
| Hard deadline | 2026-07-30T19:41:17Z |
| Model-server health | unreachable |
| SSH-tunnel health | down |
| Last successful API refresh | 2026-07-29T07:47:35Z |
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
