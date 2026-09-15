<!-- ABOUTME: Execution contract for low/high stakes edits of the original successful nonmoral corpus. -->
<!-- ABOUTME: Supersedes the failed private-stakes wrappers and unrelated grounded-revision pilot. -->
# Matched stakes from original684

User authorized autonomous creation/publication on September 10; no sample approval gate.
Question: does the magnitude of nonmoral consequences in otherwise matched SFT data
change alignment? User predicts no effect. No ODCV feedback enters generation/selection.

Parent: the exact 684 synthetic rows of the historical mixture at revision
`6364505df02b0020b030bf379bd42285a14de6a5`, SHA256
`0517ef85d288f14e42bc77f371d3e2e48866879b5824984feaf4a7451ce60561`.
Keep original domains, craft traits, system messages, instruction conflicts,
single-turn format, and complete original artifacts. This is a stakes intervention,
not a general repair of the original corpus. Source defects remain documented limitations.

Opus 4.8 produces both arms as literal minimal text edits through shared SynthDoc;
Sonnet 5 reviews the manipulation. Modify a plausible consequence of the actual task,
holding its causal mechanism constant while scaling loss from easily absorbed to
substantial. Professional settings are allowed. Do not attach unrelated fees, create
new moral dilemmas, change task difficulty, or silently relocate domains. Reasoning
is minimally adapted. After early pair drafts changed setting along with stakes,
the production representation was tightened: **one shared edit template**, with
only a numeric loss substituted between arms. The original recommendation remains
appropriate in both; magnitude strengthens or weakens the same craft consideration.
This tests exposure to different consequence magnitudes with matched substantive
reasoning, not whether an unconstrained teacher reasons differently at high stakes.
Thirteen earlier pairs already met the numeric-only equality rule and were retained.
All other earlier drafts remain in the audit, outside the final corpora.

Exact patch application rejects missing/ambiguous/overlapping spans. One initial
generation/review pass, then one targeted repair pass for specific defects; bounded
format retries. No general quality-rewrite loop or arbitrary acceptance-rate gate.
Independent agents audit separate batches; the root agent checks structure, source
preservation, and specific corrections. Final coverage is all 684 pairs:208 early,
225 middle/repair,233 later,18 root-reviewed pairs with no usable provider review.
Read scope is recorded: full original user and generated edits with affected source
passages for every pair; full original conversations for selected/suspect cases.
This is not infallible semantic validation. Target all 684 matched pairs; no duplicates, substituted IDs,
or unmodified rows counted as successful stakes edits. If unresolved cases remain,
publish their status honestly; do not call the corpus complete.

Driver: `uv run --no-sync python scratch/nonmoral/stakes.py --execute`.
After independent local audit, `--apply-audit` applies hash-bound literal corrections;
`--finalize` requires recorded correction clearance; `--assemble` verifies training
format; `--publish` uploads through the shared HF card/naming contract.
Resumable stage outputs and raw calls live under `output/nonmoral_stakes/20260910`.
Concurrency 12. Data exposure cap $60, within the existing $300 project ceiling;
prior attributable exposure $167.894061. Three-hour dispatch deadline, persistent
ledger, cooperative STOP_DISPATCH marker. A valid bare JSON response is normalized
locally without another paid call; raw responses remain intact. Windows file-replace
contention is retried locally. Earlier costs and any uncertain reservations remain
in the same ledger. No GPUs provisioned by this driver.

Deliverables: low and high public SynthDoc corpora, paired edit/review provenance,
and two mixtures each containing exactly the original 9,284 replay rows in their
original positions plus the matched 684 edited examples. Use shared naming and HF
publication helpers, public org dougalldeepmind. Full reasoning and final answer
supervision; verify tokenizer lengths/masks before declaring mixtures training-ready.

Downstream: two LoRAs, one seed per arm, identical settings on RunPod 2 H200s with
dynamic batching; common ODCV protocol, local CPU/Docker and RunPod GPU inference.
This document authorizes data execution; it does not itself start SFT/ODCV. The main
contrast is low versus high; either versus historical original also changes authoring.
Original 18.25% is a historical evaluation; that same checkpoint measured 13.75%
under the later common protocol. Do not treat them as different training results.

## Final audit findings

All684 pairs were authored;666 had accepted Sonnet reviews and18 had no usable
provider review, resolved through explicit local review.191 pairs needed local
corrections. Frequent problems were invented mandatory whole-run reruns/reprints,
external fees or review procedures, contradictions with original timing/context,
and implausible per-unit losses. Correct the generated consequence locally; never
replace the original task or silently repair the source corpus. Independent agents
cross-check these corrections. Old model reviews are retained as pre-correction
evidence and are not presented as reviews of the corrected text.

Both arms use the same wording apart from numerical amounts. Low/high therefore
mean relative consequence magnitudes, not independently validated perceived stakes.
The substantive reasoning is deliberately held fixed. This supports a controlled
stakes-exposure comparison; it does not measure how a fresh teacher would adapt its
reasoning to high stakes. Exact historical replay bytes and positions are preserved.
