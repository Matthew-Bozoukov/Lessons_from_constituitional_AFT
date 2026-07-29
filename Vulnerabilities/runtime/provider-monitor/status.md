# Provider status

```text
[GPU ACTIVE] runpod | NVIDIA A100-SXM4-80GB | $1.49/h | provider balance $95.2682 | experiment GPU budget $27.3436 remaining | elapsed 08:29 | vLLM unreachable
```

Generated: 2026-07-29T07:27:25Z  (refresh reason: periodic)

Every figure below is labelled with its basis: **exact-provider-reported**,
**locally-calculated**, **estimated**, or **unavailable**. Nothing is invented;
an unavailable figure is reported as unavailable.

| Field | Value |
| --- | --- |
| Provider | runpod |
| GPU | NVIDIA A100-SXM4-80GB |
| Hourly price | 1.49 USD/h _(exact-provider-reported)_ |
| Instance ID | `0vqb1gixqkqh5h` |
| Instance state | RUNNING |
| Provider balance | 95.2682 USD _(exact-provider-reported)_ |
| Starting provider balance | 108.1129952401 USD _(exact-provider-reported)_ |
| Provider balance delta | 12.8448 USD _(locally-calculated)_ |
| Elapsed runtime | 08:29 (8.4943 hours _(locally-calculated)_) |
| Estimated infrastructure cost | 12.6564 USD _(locally-calculated)_ |
| Experiment GPU budget remaining | 27.3436 USD _(locally-calculated)_ |
| Wall-clock remaining | 27.506 hours _(locally-calculated)_ |
| Hard deadline | 2026-07-30T10:57:46Z |
| Model-server health | unreachable |
| SSH-tunnel health | listening |
| Last successful API refresh | 2026-07-29T07:27:25Z |
| Cleanup-watchdog state | armed (deadline 2026-07-30T10:57:46Z) |
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
