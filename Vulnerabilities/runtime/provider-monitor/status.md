# Provider status

```text
[GPU ACTIVE] runpod | NVIDIA A100-SXM4-80GB | $1.49/h | provider balance $92.6871 | experiment GPU budget $37.6429 remaining | elapsed 02:34 | vLLM healthy
```

Generated: 2026-07-29T09:16:12Z  (refresh reason: periodic)

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
| Provider balance | 92.6871 USD _(exact-provider-reported)_ |
| Starting provider balance | 108.1129952401 USD _(exact-provider-reported)_ |
| Provider balance delta | 15.4259 USD _(locally-calculated)_ |
| Elapsed runtime | 02:34 (1.582 hours _(locally-calculated)_) |
| Estimated infrastructure cost | 2.3571 USD _(locally-calculated)_ |
| Experiment GPU budget remaining | 37.6429 USD _(locally-calculated)_ |
| Wall-clock remaining | 34.418 hours _(locally-calculated)_ |
| Hard deadline | 2026-07-30T19:41:17Z |
| Model-server health | healthy |
| SSH-tunnel health | listening |
| Last successful API refresh | 2026-07-29T09:16:12Z |
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
