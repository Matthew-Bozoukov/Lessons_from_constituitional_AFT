<!-- ABOUTME: Full-length, model- and developer-neutral nine-principle constitution. -->
<!-- ABOUTME: Renamed and identity-neutralised on 2026-09-08; original bytes remain archived. -->

# claude_distilled_09_principles

| field | value |
|---|---|
| status | full-length model/developer-neutral adaptation |
| principles | 9 |
| source material | Anthropic's published constitution, via the recovered nine-principle mid snapshot; see rationale.md |
| changes | Claude identity becomes AI identity; Anthropic references become developer references; attribution moved out of model-facing text |
| updated | 2026-10-30 |
| historical original | `../archive/claude_distilled_12_principles_mid/constitution.md`, unchanged and source-hash-compatible |

This variant is no longer hash-identical to the original source corpus. Historical
reproduction configs deliberately use the archived original. Renaming this folder
does not switch DA from its separately configured archived constitution.

## About this document

Constitution (alignment target). A model- and developer-neutral adaptation of a distilled
nine-principle constitution. Used to steer response generation and to grade constitutional
alignment. See rationale.md for source attribution and the history of this adaptation.

constitution.md holds ONLY the priority/conflict-resolution section and the nine principles;
this note moved here on 2026-10-30 so the model-facing text carries nothing about the repo.
sha256(full_text) moved 8e273b472d945aa23efa6236886da5e1171bff2193ee31ff73489ca54c4f0edc ->
7726f860ad97a443c8b49e12c7fc8d8bcb69d2286dc5f85934153b87a5045c11; principle, paragraph and
bullet chunks are byte-identical (the chunker ignores text before `## 1.`), and the pair is
registered as equivalent in src/data/synth/ours/constitution.py EQUIVALENT_SHA256.
