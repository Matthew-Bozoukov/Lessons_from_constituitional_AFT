<!-- ABOUTME: Metadata card for the craft-tensions spec: status, unit count, what consumes it. -->
<!-- ABOUTME: The spec itself is preferences.md; provenance and caveats are rationale.md. -->

# craft_tensions_09

| field | value |
|---|---|
| status | **experiment arm** — the non-moral counterpart to a constitution, for the deliberation-without-morality arm |
| units / traits | 9 self-contained tensions |
| source material | hand-authored, 2026-09-02 — not distilled from anything |
| date generated | 2026-09-02 |
| size | 224 words/unit mean (range 211–239); the 12-principle constitution runs 354 (329–434) |
| style guidance | 73 words (`## What a preference-aligned response looks like`) |
| alignment target | **none.** This is deliberately not one — see rationale.md |
| consumed by | `configs/data/synth/2026-09-02_nonmoral_deliberation.yaml` (segments into 9 traits) |

## Why it lives here and not in `constitutions/`

`constitutions/` holds alignment targets. This is not one, and filing it there would make the
directory's contract untrue. But it occupies the same *slot* in the generation pipeline — it is
parsed by `src/data/synth/constitution.py`, cut by `chunking: principle`, and injected into stage
prompts exactly as a constitution is. Hence a sibling directory with the same internal shape:
`preferences.md` + `rationale.md` + this card.

## The format contract it has to satisfy

`src/data/synth/constitution.py` parses it, so the shape is not cosmetic:

- `## N. Title` opens a unit. Nine of them, so `chunking: principle` yields nine traits.
- A `##` heading whose text contains "look" opens the style-guidance section, which becomes
  `{style_guidance}` in stage prompts. Here that is `## What a preference-aligned response looks
  like`.
- Everything before the numbered units is preamble. Under principle chunking the preamble
  **reaches no stage** (see `docs/BASELINES.md`), which is why each unit carries its own
  trade-off internally rather than deferring to a ranking section.
- The file must be UTF-8. The parser reads it explicitly as UTF-8; a locale-encoded write
  mojibakes every em-dash into prompts and changes the manifest's `constitution_sha256`.
