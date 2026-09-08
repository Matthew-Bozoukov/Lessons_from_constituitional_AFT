<!-- ABOUTME: Deliberative-only variant of the shortened, model-neutral nine-principle constitution. -->
<!-- ABOUTME: Removes DA response-style guidance; preserves all principles and the priority preamble verbatim. -->

# Rationale

Derived from `../abridged/constitution.md` on 2026-09-08 at the user's request.
Renamed from `no_claude_mentioned_no_style` to `abridged_no_delib` the same day.
The name denotes removal of the response-style section, not the reasoning-generation step.
Only the trailing “What a constitution-aligned response looks like” section and its
separator are removed. The nine principles and priority/conflict-resolution preamble
are unchanged. See the parent rationale for their provenance; the priority guidance
ultimately comes from the project's hand-written specgen preamble, not a deliberative
alignment paper-specific instruction.

`configs/data/synth/delib.yaml` uses this variant so DAT's shared constitution and
response-style guidance remain unchanged. The earlier published deliberative smoke
used the parent constitution and remains an immutable record of that run; it does
not demonstrate this revised setup. No new generation was performed for this edit.
