<!-- ABOUTME: Provenance and caveats for no_claude_mentioned: the nine mid principles at a third -->
<!-- ABOUTME: of the length, model- and developer-neutral. Hand-derived 2026-09-06, not specgen. -->

# Rationale — no_claude_mentioned

**Derived from** `claude_distilled_12_principles_mid/constitution.md` (itself machine-distilled
from Anthropic's published *Claude's Constitution*, January 2026). Same nine principles, same
titles and order, same per-principle shape (statement, *Why*, two bullets, *When this does NOT
apply*), same priority preamble and the same style-guidance section verbatim. Each principle
is cut to roughly a third of its length by hand, keeping the claims that carry the most weight
and dropping elaboration and second examples.

**Two deliberate changes beyond length.** The source names its model ("you must never deny
being Claude") and its developer (Anthropic, in eleven places). Neither appears here: the
persona clause becomes "never deny being an AI or claim to be human when sincerely asked",
and Anthropic becomes "your developer". The organisms trained against this document are not
that model, and the base mixture's own instruction-tuning recipe (MSM, Appendix B.3) filters
out data where the model identifies as another model. Measured on 2026-09-06: 7 of 708 rows
of the DA baseline corpus mention "Claude", leaked through system prompts, user turns and
reasoning under principles 1 and 6.

**What this is for.** The `dat` recipe's alignment target (configs/data/synth/dat.yaml, from
2026-09-06). It makes two things comparable at once: constitution length (a third) and the
absence of model identity, against arms generated from the mid document. It is NOT a
specgen artifact: no coverage inventory, no explanation-ratio measurement; length and
claim coverage are reported as covariates by whoever compares arms.

**Known caveats.** Cutting by hand is a judgement; principles 1, 5 and 6 lose the most
(the pluralism-over-centralisation argument, the metaethical hierarchy, the moral-status
discussion). The chunker reads `## N.` headings, so the nine units are the same ids (t1–t9)
as the mid document.
