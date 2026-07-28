# Provider status

```text
[NO GPU] runpod | NVIDIA A100 80GB PCIe | $1.19/h | provider balance $108.113 | experiment GPU budget $40 remaining | elapsed 00:00 | vLLM unreachable
```

Generated: 2026-07-28T21:30:48Z  (refresh reason: periodic)

Every figure below is labelled with its basis: **exact-provider-reported**,
**locally-calculated**, **estimated**, or **unavailable**. Nothing is invented;
an unavailable figure is reported as unavailable.

| Field | Value |
| --- | --- |
| Provider | runpod |
| GPU | NVIDIA A100 80GB PCIe |
| Hourly price | 1.19 USD/h _(exact-provider-reported)_ |
| Instance ID | none |
| Instance state | not-provisioned |
| Provider balance | 108.113 USD _(exact-provider-reported)_ |
| Starting provider balance | 108.1129952401 USD _(exact-provider-reported)_ |
| Provider balance delta | 0 USD _(locally-calculated)_ |
| Elapsed runtime | 00:00 (0 hours _(locally-calculated)_) |
| Estimated infrastructure cost | 0 USD _(locally-calculated)_ |
| Experiment GPU budget remaining | 40 USD _(locally-calculated)_ |
| Wall-clock remaining | unavailable _(nothing invented)_ |
| Hard deadline | not set (no instance) |
| Model-server health | unreachable |
| SSH-tunnel health | down |
| Last successful API refresh | 2026-07-28T21:30:48Z |
| Cleanup-watchdog state | armed (deadline ) |
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
