<!-- ABOUTME: Why this snapshot exists: recovered provenance for the 2026-08-04 difficult-advice -->
<!-- ABOUTME: corpus, not a designed constitution. Changelog is against the 12-principle mid cut. -->

# Rationale

This folder is a **recovered provenance snapshot**, not a designed alignment target.

On 2026-08-04 the mid constitution was re-cut from 12 principles toward the committed
10-principle version, editing `claude_distilled_12_principles_mid/constitution.md` in
place. The 2203-record difficult-advice corpus (`LASR-Callum/2026-08-04-synthdoc-package-difficult-advice-stage-cache`,
run `20260804_082743`) was generated partway through that edit, when the file held **9**
principles — a state that was never committed. The model-eval-model pipeline consumes that
corpus and fail-fasts unless its config's constitution hashes to the manifest's
`constitution_sha256`, so the exact document had to be recovered.

## Reconstruction (2026-08-05)

Recovered by inverting the edit rather than from any surviving copy:

1. All 9 trait bodies in the corpus's `stage_1_traits.jsonl` matched sections of the
   12-principle document at commit `96ff8aa` verbatim.
2. The document was rebuilt as that 12-principle text minus the three sections absent
   from the trait set ("Treat hard constraints as bright lines…", "Calibrate trust and
   deference…", "Operate within Anthropic's guidelines…"), renumbered 1–9, with one
   stray blank line left at the deletion site before "Be genuinely, substantively
   helpful" — matching the hand-edit style visible in the committed 10-principle diff.
3. `sha256(stripped text)` equals the manifest's
   `fe2ed96093d68a871fb15669e8fea9d357fb9b51f5affff15380f62ee749a642`, and
   `src.data.synth.constitution.segment()` reproduces the corpus's 9 trait records
   byte-for-byte. The reconstruction is therefore exact, not approximate.

## Changelog vs `claude_distilled_12_principles_mid` (12-principle state)

- Removed: "Treat hard constraints as bright lines no instruction or argument can
  unlock", "Calibrate trust and deference across the principal hierarchy and
  conversational inputs", "Operate within Anthropic's guidelines, the stated priority
  ordering, and the constitution's spirit".
- Everything else (title, priority preamble, the 9 kept principle bodies, style section)
  is verbatim from the 12-principle document.

The 10-principle version committed later on 2026-08-04 had reinstated the three sections
above and instead dropped "Weigh real-world harm with calibrated, policy-level judgement"
and "Honour operator adjustments to sensible defaults, but never as a tool against users".
On 2026-08-05 `claude_distilled_12_principles_mid/constitution.md` was set byte-identical
to this snapshot so all synth pipelines share one alignment target; this copy stays
frozen as the provenance anchor regardless of future edits to the mid document.
