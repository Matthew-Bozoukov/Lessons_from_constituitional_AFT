---
title: "Infrastructure provider comparison and selection"
date: 2026-07-28
summary: "RunPod Secure Cloud A100 80GB PCIe at $1.19/h selected; Vast.ai is unusable because the account holds $0.00 and has no billing method, and the price gap does not justify funding it."
status: complete
---

# Infrastructure provider comparison and selection

Queried programmatically through each provider's official authenticated API on
2026-07-28. No provider website was scraped.

## Decision

**Selected: RunPod Secure Cloud, `NVIDIA A100 80GB PCIe`, $1.19/h on-demand.**

## Account state, which constrains the decision before price does

| Provider | Balance | Billing configured | Usable now |
| --- | --- | --- | --- |
| Vast.ai | **$0.00** (`credit` $0.00, `balance` $0.00) | `has_billing: false`, `can_pay: false` | **No** |
| RunPod | **$108.11** | yes, `currentSpendPerHr` $0, 0 active pods | Yes |

The Vast.ai account cannot rent anything at any price. This was established from
`GET /api/v0/users/current/` before any offer was considered.

## Qualifying offers

Filters applied to both providers, from the stated requirements: single GPU with
>= 80 GB VRAM, on-demand (not interruptible), verified/secure datacenter host,
reliability >= 99.5%, >= 150 GB usable disk, direct SSH ports, adequate download
bandwidth.

### Vast.ai - 8 qualifying offers

| Offer ID | GPU | VRAM | $/h | Reliability | Down Mbps | Disk GB | Location |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 29019282 | A100 SXM4 | 80 GB | **1.0427** | 0.9971 | 8526 | 750 | Czechia |
| 28705643 | RTX PRO 6000 SE | 96 GB | 1.7363 | 0.9994 | 18215 | 853 | Texas, US |
| 33665876 | RTX PRO 6000 SE | 96 GB | 1.7363 | 0.9990 | 16653 | 1122 | Texas, US |
| 29019369 | H100 SXM | 80 GB | 2.2693 | 0.9990 | 7658 | 2411 | Czechia |
| 34930660 | H100 SXM | 80 GB | 2.4027 | 0.9965 | 9561 | 1484 | Czechia |
| 27612880 | H200 NVL | 140 GB | 3.4049 | 0.9992 | 8179 | 2528 | Czechia |
| 21829059 | H200 NVL | 140 GB | 3.9004 | 0.9974 | 8556 | 1868 | Czechia |
| 36326110 | H200 | 140 GB | 4.0811 | 0.9977 | 6443 | 2495 | US |

Exactly **one** qualifying A100 80 GB host exists on Vast right now. The next
cheapest qualifying offer, at $1.7363/h, is more expensive than RunPod's A100.

### RunPod Secure Cloud - lowest on-demand price per GPU class

| GPU | VRAM | $/h | Considered |
| --- | --- | --- | --- |
| MI300X | 192 GB | 0.50 | Rejected: ROCm vLLM + LoRA toolchain risk, see below |
| **A100 PCIe** | **80 GB** | **1.19** | **Selected** |
| A100 SXM | 80 GB | 1.39 | Rejected: same VRAM class, $0.20/h more |
| RTX PRO 6000 (WK/Server) | 96 GB | 1.69 | Rejected on price |
| H100 PCIe | 80 GB | 1.99 | Rejected, see below |
| H100 NVL | 94 GB | 2.59 | Rejected on price |
| H100 SXM | 80 GB | 2.69 | Rejected on price |
| H200 SXM | 141 GB | 3.59 | Rejected on price |
| B200 | 180 GB | 5.98 | Rejected on price |
| B300 | 288 GB | 6.94 | Rejected on price |

## Why not fund Vast.ai

The instruction is to prefer Vast when a qualified Vast offer is *materially*
cheaper, and to use RunPod Secure Cloud when the saving is too small to justify
host variance. Quantified against the actual workload:

| | Vast A100 SXM | RunPod A100 PCIe |
| --- | --- | --- |
| Hourly | $1.0427 | $1.19 |
| Hourly delta | - | +$0.147 (**14.1%**) |
| Cost over the ~15 h estimated run | $15.64 | $17.85 |
| **Absolute saving from Vast** | **$2.21** | - |
| Hours available within the $40 cap | 38.4 h | 33.6 h |

The saving is $2.21 on the expected run, against these costs:

- The account must first be funded; Vast credit is non-refundable, so a minimum
  deposit likely exceeds the saving it would produce.
- Exactly one qualifying A100 host is currently listed. If it is rented before
  provisioning, the cheapest remaining qualifying Vast offer is $1.7363/h, which
  is **46% more expensive** than the selected RunPod option.
- RunPod is already funded at $108.11, far above the $40 hard cap, so budget
  headroom is not a differentiator.

**Conclusion: funding Vast.ai is not worth it.** A 14% hourly saving worth about
$2 does not justify a non-refundable deposit plus single-host exposure. RunPod
Secure Cloud is selected on host variance grounds, exactly as the decision rule
prescribes.

## Why not the cheaper or faster alternatives

**MI300X at $0.50/h** is less than half the price and has ample VRAM. Rejected:
the scientific requirement is BF16 inference of a Qwen3-32B LoRA adapter through
vLLM, and the ROCm path for vLLM LoRA serving carries a real risk of consuming
hours of the 36 h wall-clock budget on toolchain debugging. With expected spend
around $18 against a $40 cap, budget is not the binding constraint; reliability
is. Recorded as considered rather than overlooked.

**H100 PCIe at $1.99/h** would be faster per token. The instruction is not to use
an H100 merely for speed unless expected *total* cost is no greater than the best
suitable A100. At 1.67x the hourly price it would need to be more than 1.67x
faster end to end. Part of the run is fixed cost that does not scale with GPU
speed (provisioning, a ~65 GB model download, vLLM startup), and Petri audits at
concurrency 1 are latency-bound rather than throughput-bound. That threshold
cannot be established in advance, so the A100 is selected as instructed.

## Estimated total cost

| Phase | Est. hours |
| --- | --- |
| Provisioning and environment setup | 0.3 |
| Base + adapter download from Hugging Face | 0.4 |
| vLLM startup and preflight verification | 0.7 |
| Petri pilot, 4 audits | 1.5 |
| Focused discovery, 30 audits | 6.0 |
| Fixed evaluation and matched controls | 4.0 |
| Contingency | 2.0 |
| **Total** | **~14.9 h** |

At $1.19/h that is **~$17.75**, against a `MAX_GPU_SPEND_USD` cap of $40.00 and
a `MAX_WALL_CLOCK_HOURS` cap of 36 h. Storage on RunPod Secure Cloud is included
in the pod hourly rate for the container disk and attached volume at the sizes
required here; any separate volume charge is tracked in the cost ledger.

## Selection summary

| Field | Value |
| --- | --- |
| Provider | RunPod |
| Cloud type | Secure Cloud (datacenter, on-demand, not interruptible) |
| GPU | NVIDIA A100 80GB PCIe |
| VRAM | 80 GB |
| Hourly price | $1.19 |
| Storage price | included at required sizes; tracked separately if charged |
| Reliability | RunPod Secure Cloud datacenter tier |
| Bandwidth | datacenter; verified at provisioning time |
| Disk | >= 150 GB requested |
| Estimated total cost | ~$17.75 for ~14.9 h |
| Reason selected | Vast.ai account is unfunded and has no billing method, so no Vast offer is rentable. Among usable options this is the cheapest that meets every hard requirement. The 14% hourly saving available on Vast, worth about $2.21, does not justify a non-refundable deposit plus exposure to a single qualifying host. |

## Raw artifacts

- [vast-offers.json](../evidence/provider/vast-offers.json)
- [runpod-gpu-types.json](../evidence/provider/runpod-gpu-types.json)
- [account-state.json](../evidence/provider/account-state.json)
