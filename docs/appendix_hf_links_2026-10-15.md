<!-- ABOUTME: Immutable HF source catalogue for the October appendix figures. -->
<!-- ABOUTME: Generated from downloaded and SHA256-verified run metadata by package_appendix.py. -->
# HF links for the appendix

Every experiment link below is pinned. Dataset repositories contain raw/evaluation/metadata evidence; model repositories contain exported adapters. Index 0/1/2 means Base/Control/DA15. GPT-OSS SWE is a single three-arm repository.

**Provenance qualification:** GPT-OSS DA15 has no completed training metadata in its model export. Its configured-data link is from the historical launch configuration, not a recovered training receipt. The model card does bind the evaluated sampler identity. Optional MSM results have sampler provenance in their eval metadata; no separately verified adapter export is claimed.

## Qwen models and training mixtures

| Artifact | Immutable repository |
|---|---|
| qwen_control | [dougalldeepmind/2026-10-05-qwen36-0-plain @ 1565d4f3](https://huggingface.co/dougalldeepmind/2026-10-05-qwen36-0-plain/tree/1565d4f3356aa1b3bea1c503700164faa240ef3a) |
| qwen_control_data | [dougalldeepmind/2026-10-05-plain-mix @ 019e239c](https://huggingface.co/datasets/dougalldeepmind/2026-10-05-plain-mix/tree/019e239ce51f5adbbcaff4f92f26e63fcd31c080) |
| qwen_da15 | [dougalldeepmind/2026-10-05-qwen36-0-da-15 @ ce773a6b](https://huggingface.co/dougalldeepmind/2026-10-05-qwen36-0-da-15/tree/ce773a6b87db66bd569dd13071ec5e27201462f1) |
| qwen_da15_data | [dougalldeepmind/2026-10-05-da-15-mix @ f990959a](https://huggingface.co/datasets/dougalldeepmind/2026-10-05-da-15-mix/tree/f990959a8395573abccbb73e7441f951c0d33056) |

## Qwen evaluation runs and Arena answers

| Artifact | Immutable repository |
|---|---|
| arena_qwen_base | [dougalldeepmind/2026-10-13-ah-vs-qwen36 @ 99f48c89](https://huggingface.co/datasets/dougalldeepmind/2026-10-13-ah-vs-qwen36/tree/99f48c89e31c41626d9311a4f4a95fc965888302) |
| arena_qwen_base_qwen36 | [dougalldeepmind/2026-10-13-ah-qwen36 @ 0d220b94](https://huggingface.co/datasets/dougalldeepmind/2026-10-13-ah-qwen36/tree/0d220b9416c7ece98a185bad5bd32dc275f9fad5) |
| arena_qwen_base_qwen36_0_da_15 | [dougalldeepmind/2026-10-12-ah-qwen36-0-da-15 @ e15e030a](https://huggingface.co/datasets/dougalldeepmind/2026-10-12-ah-qwen36-0-da-15/tree/e15e030aaaf72fc05cad06bead91fa7284082553) |
| arena_qwen_base_qwen36_0_plain | [dougalldeepmind/2026-10-12-ah-qwen36-0-plain @ bc2fb7d0](https://huggingface.co/datasets/dougalldeepmind/2026-10-12-ah-qwen36-0-plain/tree/bc2fb7d0d17202357d8f04c3849354fee76ec0a4) |
| arena_qwen_control | [dougalldeepmind/2026-10-12-ah-vs-qwen36-0-plain @ 4f19416d](https://huggingface.co/datasets/dougalldeepmind/2026-10-12-ah-vs-qwen36-0-plain/tree/4f19416df56ca922b94fc3403dfe7dbea79ce698) |
| qwen36_mmlu_0 | [dougalldeepmind/2026-10-14-mmlu-qwen36 @ 6f41ca33](https://huggingface.co/datasets/dougalldeepmind/2026-10-14-mmlu-qwen36/tree/6f41ca3384b14c101d81f996065529021dc78eac) |
| qwen36_mmlu_1 | [dougalldeepmind/2026-10-14-mmlu-qwen36-0-plain @ a4556e5a](https://huggingface.co/datasets/dougalldeepmind/2026-10-14-mmlu-qwen36-0-plain/tree/a4556e5a8b2a24de05892133165fbef75035a88e) |
| qwen36_mmlu_2 | [dougalldeepmind/2026-10-14-mmlu-qwen36-0-da-15 @ da022054](https://huggingface.co/datasets/dougalldeepmind/2026-10-14-mmlu-qwen36-0-da-15/tree/da022054c494987bc1c75515851c49b55357eb12) |
| qwen36_swe_0 | [dougalldeepmind/2026-10-07-swebench-qwen36-lite-v5 @ 9e7447f0](https://huggingface.co/datasets/dougalldeepmind/2026-10-07-swebench-qwen36-lite-v5/tree/9e7447f072aaacfbb4919a8c6f62ac881c04004a) |
| qwen36_swe_1 | [dougalldeepmind/2026-10-07-swebench-qwen36-0-plain-lite-v5 @ 1148e0db](https://huggingface.co/datasets/dougalldeepmind/2026-10-07-swebench-qwen36-0-plain-lite-v5/tree/1148e0db51d4bbb7ac25d22be593a6bb77e63581) |
| qwen36_swe_2 | [dougalldeepmind/2026-10-07-swebench-qwen36-0-da-15-lite-v5 @ bde7963c](https://huggingface.co/datasets/dougalldeepmind/2026-10-07-swebench-qwen36-0-da-15-lite-v5/tree/bde7963c515c5f2e7be59fe50d6883cf48650e31) |

## GPT-OSS models and training mixtures

| Artifact | Immutable repository |
|---|---|
| gptoss_control | [dougalldeepmind/2026-10-06-gptoss120b-0-plain @ be6649f6](https://huggingface.co/dougalldeepmind/2026-10-06-gptoss120b-0-plain/tree/be6649f6d2c97a772b176fa8fea3082ac85c82d1) |
| gptoss_control_data | [dougalldeepmind/2026-10-05-plain-mix @ a5b11566](https://huggingface.co/datasets/dougalldeepmind/2026-10-05-plain-mix/tree/a5b115666913790be37482f83f6aa843c73e258a) |
| gptoss_da15 | [dougalldeepmind/2026-10-06-gptoss120b-0-da-15 @ ae42143e](https://huggingface.co/dougalldeepmind/2026-10-06-gptoss120b-0-da-15/tree/ae42143e50ebd04a6ff5cccd955bb90225d44083) |
| gptoss_da15_configured_data | [dougalldeepmind/2026-10-05-da-15-mix @ cf42d86b](https://huggingface.co/datasets/dougalldeepmind/2026-10-05-da-15-mix/tree/cf42d86b9a916a4a107e5a57720e3b181ccf1462) |

## GPT-OSS capability runs and Arena answers

| Artifact | Immutable repository |
|---|---|
| arena_gptoss | [dougalldeepmind/2026-10-14-ah-luna-low-gptoss120b-three-pairs @ 0ca407d8](https://huggingface.co/datasets/dougalldeepmind/2026-10-14-ah-luna-low-gptoss120b-three-pairs/tree/0ca407d879ac54516b1cc0c0a66a6b7dc79f0c6b) |
| arena_gptoss_base | [dougalldeepmind/2026-10-14-ah-gptoss120b-base @ ad9e5c9f](https://huggingface.co/datasets/dougalldeepmind/2026-10-14-ah-gptoss120b-base/tree/ad9e5c9f86b650318e31d56f18a7efabf1703429) |
| arena_gptoss_control | [dougalldeepmind/2026-10-14-ah-gptoss120b-0-plain @ ff601cfd](https://huggingface.co/datasets/dougalldeepmind/2026-10-14-ah-gptoss120b-0-plain/tree/ff601cfd84340770fe511d32023b867ee0fc7d64) |
| arena_gptoss_da15 | [dougalldeepmind/2026-10-14-ah-gptoss120b-0-da-15 @ 4bebdd1d](https://huggingface.co/datasets/dougalldeepmind/2026-10-14-ah-gptoss120b-0-da-15/tree/4bebdd1da589080668b129fd77a4e061a527768f) |
| gptoss120b_mmlu_0 | [dougalldeepmind/2026-10-14-mmlu-gptoss120b-base @ 345a70aa](https://huggingface.co/datasets/dougalldeepmind/2026-10-14-mmlu-gptoss120b-base/tree/345a70aa66775007425328a9b7c31c09f13ef390) |
| gptoss120b_mmlu_1 | [dougalldeepmind/2026-10-14-mmlu-gptoss120b-0-plain @ 862acf34](https://huggingface.co/datasets/dougalldeepmind/2026-10-14-mmlu-gptoss120b-0-plain/tree/862acf3443fb49172a8202637c6381421641fc65) |
| gptoss120b_mmlu_2 | [dougalldeepmind/2026-10-14-mmlu-gptoss120b-0-da-15 @ 36ea3662](https://huggingface.co/datasets/dougalldeepmind/2026-10-14-mmlu-gptoss120b-0-da-15/tree/36ea3662506b356c4d27bc360ffd2525106f426d) |
| gptoss120b_swe_0 | [dougalldeepmind/2026-10-13-gptoss120b-base-control-da15-swe-lite300-context65000 @ 4ccd7690](https://huggingface.co/datasets/dougalldeepmind/2026-10-13-gptoss120b-base-control-da15-swe-lite300-context65000/tree/4ccd7690774909418179133c1204a0a058f86cc8) |

## GPT-OSS misalignment results

| Artifact | Immutable repository |
|---|---|
| gptoss_odcv | [dougalldeepmind/2026-10-12-gptoss120b-base-control-da15-odcv-full56k @ ee25be03](https://huggingface.co/datasets/dougalldeepmind/2026-10-12-gptoss120b-base-control-da15-odcv-full56k/tree/ee25be032bfe137b1f61a896104e9d4260e54eac) |
| mask_Base | [dougalldeepmind/2026-10-06-mask-gptoss120b @ 081ecc21](https://huggingface.co/datasets/dougalldeepmind/2026-10-06-mask-gptoss120b/tree/081ecc21725aa8c83dd30bde61a644d5fa09f0c3) |
| mask_Control | [dougalldeepmind/2026-10-15-mask-gptoss120b-0-plain @ df0addf0](https://huggingface.co/datasets/dougalldeepmind/2026-10-15-mask-gptoss120b-0-plain/tree/df0addf0f7e40ae7109a41f575f1b0d628e9fd58) |
| mask_DA15 | [dougalldeepmind/2026-10-06-mask-gptoss120b-0-da-15 @ 4c1cbff1](https://huggingface.co/datasets/dougalldeepmind/2026-10-06-mask-gptoss120b-0-da-15/tree/4c1cbff18562c07727bbb02410b34991cdceceea) |
| mask_DA15+MSM | [dougalldeepmind/2026-10-06-mask-gptoss120b-0-da-msm-15 @ 874421e9](https://huggingface.co/datasets/dougalldeepmind/2026-10-06-mask-gptoss120b-0-da-msm-15/tree/874421e9c169a97d0e317263e5ee1b92361f8596) |
| mask_MSM | [dougalldeepmind/2026-10-06-mask-gptoss120b-0-msm @ a820a66a](https://huggingface.co/datasets/dougalldeepmind/2026-10-06-mask-gptoss120b-0-msm/tree/a820a66a928141ab7f2599e8eef8781381e6cf3c) |

## Synthetic source

| Artifact | Immutable repository |
|---|---|
| da_synthetic_source | [dougalldeepmind/2026-10-02-da-synth @ 305914d5](https://huggingface.co/datasets/dougalldeepmind/2026-10-02-da-synth/tree/305914d58627457002f61d86ba69866f6fa2b9a9) |

## Base models and original benchmark inputs

- [Qwen3.6-27B base weights and tokenizer](https://huggingface.co/Qwen/Qwen3.6-27B/tree/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9).
- [GPT-OSS-120B public tokenizer/configuration revision](https://huggingface.co/openai/gpt-oss-120b/tree/b5c939de8f754692c1647ca79fbf85e8c1e70f8a). Provider-served Base identity is `tinker://base`; this public revision is not proof of provider weight bytes or precision.
- [MMLU input](https://huggingface.co/datasets/cais/mmlu/tree/c30699e8356da336a370243923dbaf21066bb9fe): 10 test questions/subject, 57 subjects, five dev examples, seed 0.
- [SWE-bench Lite input](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite/tree/6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2): all 300 test problems.
- [Arena-Hard v2.0 vendored input](https://github.com/lmarena/arena-hard-auto/tree/196f6b826783b3da7310e361a805fa36f0be83f3): 500 hard and 250 creative prompts before the GPT-OSS judge exclusion.
- [ODCV vendored scenarios](https://github.com/McGill-DMaS/ODCV-Bench/tree/7353f1cf4b2579a3a8a5b8a5061d7c7d41f60668). Project harness patches and exact configs are retained with each run.
- [MASK vendored input](https://github.com/centerforaisafety/mask/tree/25e0b1201e6c928ebe69f7c5aad6fa9063a377ea). The Control publication contains frozen CSV hashes, historical scoring sources and raw request/response archives.

## Upstream sources of the plain mixture

These are the revisions documented by the plain mixture card. The released mixture/provenance files, rather than re-sampling upstream HEAD, define the training rows.

- [nvidia/Nemotron-SFT-Instruction-Following-Chat-v3](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Instruction-Following-Chat-v3/tree/be3b3e04ef605ac9d3f8f35b9d5a632f4a3a3402)
- [allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M/tree/7d6490e462285cf85d91eabea0f9a954fbddcd1f)
- [nvidia/Nemotron-SFT-Math-v4](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Math-v4/tree/84d42ad0cb960f07f951b9baa9ed2b46a5a18c66)
- [nvidia/OpenCodeReasoning](https://huggingface.co/datasets/nvidia/OpenCodeReasoning/tree/20a1ca19c0d050fe9057fc08339d6b370ec1c67a)
- [nvidia/Nemotron-SFT-Science-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Science-v2/tree/6536a5021222a94126968e8c92f29ee47fc8a7df)
