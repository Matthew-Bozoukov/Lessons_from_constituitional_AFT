---
title: "Synthetic corpus superficial-pattern audit"
date: 2026-07-22
summary: "Scan–cluster–autorate notebook for rhetorical artifacts in the reasons-rich corpus."
model_id: qwen3-32b
checkpoint_id: post-sft-reasons-v1
training_stage: sft
run_id: pattern-audit-reasons-v1
dataset_version: reasons-rich-v1
status: needs-review
tags:
  - vulnerability-check
  - data-quality
  - pattern-audit
  - ood
---

# Synthetic corpus superficial-pattern audit

> [!WARNING]
> Fictional demonstration data. Frequencies below exist only to test the UI.

We used a three-pass **scan → cluster → autorate** procedure to look for
structural patterns that individual examples made hard to notice.

| Candidate pattern | Broad frequency | Strict frequency | Disposition |
| --- | ---: | ---: | --- |
| “Key consideration” opener | 18% | 11% | Filter and regenerate |
| Symmetrical three-part rationale | 24% | 9% | Monitor |
| Unnecessary emotional validation | 7% | 2% | Accept |
| Conversion-arc narrative | 5% | 4% | Exclude |

## Working interpretation

The corpus may teach stylistic regularities independently of the target
constitutional behavior. A stable downstream safety score therefore does not
rule out an unintended learned pattern. This needs separate auditing.

> [!CAUTION]
> Do not interpret low pattern frequency as proof that the training signal is
> clean. The autoraters themselves need calibration.

