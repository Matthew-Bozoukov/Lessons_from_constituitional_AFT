# ABOUTME: Prompt templates for specgen's three LLM stages: extract, cluster, write.
# ABOUTME: Arm-varying slots are exactly n_principles (cluster) and token_budget/n_cues (write).

from __future__ import annotations

import hashlib

# Stage 1 — claim extraction. One call per source section; no arm-varying slots.
EXTRACT = """\
You are extracting atomic normative claims from one section of Anthropic's published
Claude constitution. Read ONLY the section text below; do not use any other knowledge
of this document.

<section title="{title}">
{text}
</section>

Extract every distinct normative claim — every statement about what the assistant
should do, must not do, may do, or how it should weigh considerations. Rules:
- One claim per entry, one sentence, self-contained, prescriptive.
- Split compound guidance into separate atomic claims.
- Skip pure exposition that carries no normative content.
- "anchor" is a VERBATIM span (at most 25 words) copied exactly from the section text
  that grounds the claim.
- "modality" is how the source states it: "never" | "always" (hard constraints),
  "prefer" (defaults), "weigh" (considerations to balance).

Return a JSON array only:
[{{"claim": "...", "anchor": "...", "modality": "weigh"}}]
"""

# Stage 2 — clustering. Exactly one arm-varying slot: n_principles.
CLUSTER = """\
Below is the complete inventory of normative claims extracted from Anthropic's Claude
constitution. Partition ALL of them into exactly {n_principles} principles.

<claims>
{claims}
</claims>

Rules:
- Every claim_id appears in exactly one cluster. No claim may be dropped or repeated.
- Group by normative theme — claims a single coherent principle could govern.
- "parent_priority" maps the cluster to the constitution's own top-level priority it
  most belongs to: "safe" | "ethical" | "compliant" | "helpful".
- "working_title" is a short prescriptive title for the future principle.

Return JSON only:
{{"clusters": [{{"working_title": "...", "parent_priority": "ethical",
                "claim_ids": ["eth-001", "eth-002"]}}]}}
"""

# Stage 3 — unit writing. Arm-varying slots: token_budget and n_cues (cue count varies
# with granularity by design — that is what the axis means, not a confound).
WRITE_UNIT = """\
Write one principle unit of a constitution for an AI assistant. It must be built from
the claims below and ONLY the claims below — every claim represented, nothing added
that the claims and anchors do not support.

<principle working_title="{title}">
{claims}
</principle>

Format (follow it exactly):

## <Principle title>
<statement — prescriptive, second person ("you"), 2-5 sentences covering the claims>

*Why:* <one paragraph of rationale grounded in the anchors>

{cue_block}*When this does NOT apply:* <2-3 sentences on the over-application / misfire
case — when following this principle too eagerly would itself be a failure>

Rules:
- Second-person prescriptive voice throughout. British spelling.
- Preserve each claim's modality: hard constraints stay absolute ("never", "do not");
  defaults and weighings stay hedged ("prefer", "weigh", "generally").
- Keep the statement COMPACT: it compresses many claims, so state the principle they
  share rather than enumerating them. Spend the saved tokens on explanation.
- The *Why:* paragraph plus the *When this does NOT apply:* block must together be
  LONGER than the statement and cues combined — at least 60% of the unit's tokens.
- HARD LIMIT: the whole unit must be about {token_budget} tokens — scale every block
  to that budget (at a small budget the statement is 1-2 sentences, the *Why:* a few
  tight sentences, the *When this does NOT apply:* 1-2 sentences). Do not run long.
- Output the unit markdown only — no numbering, no commentary.
"""

# Follow-up turn when a unit lands outside its token band or explanation ratio.
REVISE = """\
Your unit measured {measured} tokens against a budget of {token_budget}, with {expl_pct}%
of its tokens in the *Why:* + *When this does NOT apply:* blocks (need at least 60%).
Rewrite it to approximately {token_budget} tokens with at least 60% of them in those two
blocks. Keep the same structure, claim coverage, modality, and cue count — only rebalance
and compress or expand the prose. Output the unit markdown only.
"""


def cue_block(n_cues: int) -> str:
    """Render the behavioural-cue part of the unit template (empty at 0 cues)."""
    if n_cues == 0:
        return ""
    return (f"- <behavioural cue — one concrete, observable do/don't>  "
            f"(exactly {n_cues} such bullets)\n\n")


def hashes() -> dict[str, str]:
    """sha256 of every template, for meta.json provenance."""
    return {name: hashlib.sha256(tpl.encode()).hexdigest()
            for name, tpl in [("extract", EXTRACT), ("cluster", CLUSTER),
                              ("write_unit", WRITE_UNIT), ("revise", REVISE)]}
