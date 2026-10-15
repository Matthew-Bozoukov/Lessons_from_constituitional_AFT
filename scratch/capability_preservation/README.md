# Capability and replication appendix figures

These scripts read pinned HF evidence. They never start inference, judging or training.
Use the existing project environment; no dependency upgrade is needed.

From the repository root, with its import path available:

1. `python scratch/capability_preservation/plot.py` validates the configured MMLU/SWE counts against immutable results, then renders both families' capability and pairwise Arena figures.
2. `python scratch/capability_preservation/plot_misalignment.py` renders the GPT-OSS ODCV and MASK figures, including the published Control closeout and its coverage qualification.
3. `python scratch/capability_preservation/appendix_sources.py` collects pinned evaluation metadata.
4. `python scratch/capability_preservation/appendix_lineage.py` collects model/data/answer-generation provenance. It explicitly records missing GPT-OSS DA15 training metadata.
5. `python scratch/capability_preservation/package_appendix.py` checks saved SHA256 hashes, builds the HF link catalogue, copies figures and prose into `output/appendix_20261015`, and verifies the ZIP readback.

`figures.yaml` and `misalignment.yaml` pin all numerical sources. The prose and captions are in `docs/appendix_capability_replication_2026-10-15.md`. Figures are local artifacts, not committed or uploaded. `audit_qwen.py` is the earlier offline diagnostic of saved Arena answers; its selected subsets must not replace benchmark scores.
