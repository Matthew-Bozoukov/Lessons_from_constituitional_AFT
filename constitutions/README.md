<!-- ABOUTME: Constitution layout and active alignment targets. -->
<!-- ABOUTME: Abridged variants are model-neutral; historical and experimental documents remain archived. -->

# Constitutions

Each folder contains `constitution.md` (the alignment target), `rationale.md`
(provenance and caveats), and optionally a short metadata `README.md`.

## Current layout

```text
constitutions/
  claude_distilled_09_principles/              full-length, model/developer-neutral nine principles
  abridged/                                  shortened, model-neutral; used by DAT and delib
  archive/
    claude_distilled_12_principles_mid/        historical mid recipe (actually nine principles)
    claude_distilled_8_principles_v1/
    experimental/
      claude_distilled_04_principles_coarse/
      claude_distilled_07_principles_approved/
      claude_distilled_24_principles_fine/
```

The initial 2026-09-08 reorganisation changed paths without editing constitution text;
the subsequent nine-principle adaptation neutralised its model/developer references.
`abridged` was `no_claude_mentioned`. Its `_no_delib` twin (once `no_claude_mentioned_no_style`)
existed only to drop the closing response-style section for deliberative generation; on
2026-09-10 that section left EVERY constitution file -- it is a document type's tone guidance,
now the `style_guidance:` field of the synth config that uses it -- and the twin was retired.
A constitution is the alignment target and nothing else; the parser refuses a file that
still carries such a section.

Existing DA recipes continue to reference the archived mid document to preserve their
alignment target. Archiving it does **not** remove its Claude/Anthropic wording.
The full-length nine-principle variant is now model/developer-neutral. Its original
byte-pinned text remains in the archived mid document for historical reproduction.

Historical published artifacts and research logs retain their original paths. Active
configs and code point to the new locations. Configs may deliberately consume archived
documents to reproduce existing experiments.

## Generating constitutions

The one-off `scratch/specgen/` tool distills the published constitution at controlled
granularities with claim-level coverage accounting. See its README. New variants should
include a rationale documenting derivation, deliberate changes, and limitations.
