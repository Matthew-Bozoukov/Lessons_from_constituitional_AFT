<!-- ABOUTME: Prospective minimal decision-stakes recipe and single-smoke plan. -->
<!-- ABOUTME: Records alternatives, intended invariants and bounded full-generation path. -->
# Minimal decision-stakes variant: prospective plan

The preceding smoke produced 15/18 rows with coverage of all nine principles. It
showed mostly light prompt edits, but a final stakes judge sometimes credited safe
advice as reducing the stakes of the underlying decision. Drafting also omitted
facts that established the conflict, and one scenario's arithmetic reversed it.

Three options were considered. Tightening both judges preserves the largest portion
of the current config but pays to classify the same decision twice and leaves the
second classifier exposed to answer-dependent reasoning. Adding a scenario schema,
extra early judge or repair loop might improve observability but introduces more
moving parts than the user wants. The chosen option removes the final stakes stage,
retains the existing pre-answer magnitude/scope review, and clarifies the few source
and drafting instructions implicated by the failures.

No new operator, model role, content-revision pass or factual critic is added. Both
normal DA answer prompts, style guidance and lint criteria remain unchanged. The
existing all-Sonnet choice and no-substantive-retry policy remain. The source is still
only the nine full principles from the specified constitution, not old examples.

The source checks that its own facts support its claimed conflict, including any
numeric comparison, and considers ordinary downstream consequences for all affected
people. Drafting preserves ownership, consent, prices and intentions: implicit
values means no announced moral lesson, not concealed evidence. Prompt refinement
remains a local edit. The single judge rates the decision before advice, including
taking the shortcut and declining it; a polite refusal cannot lower this grade.
Missing consequential scale is unclear, but ordinary mundane gaps are not automatic
failures. Incoherent premise facts remain visible in its diagnostic note, without
turning this into an all-purpose quality gate.

Tradeoff: removing the final gate means the pipeline does not automatically catch
new serious facts invented in answers. That gate was not a reliable factual audit,
and normal DA does not have one. Full smoke reads must report such answer defects
separately; no claim of factual perfection or recovered ODCV performance follows.

Run exactly 18 new candidates, two per principle in one scenario call per principle,
with the existing cumulative $20 per-call budget guard. Previous exposure is
$12.287536; headroom $7.712464. No rerolls, replacements or mid-run prompt changes.
Keep the prospective checks already in the launch YAML: >=14 exports, every principle,
>=14 initially suitable cases, and no clear stakes/scope false accepts. Borderline
judgments are disclosed rather than silently counted as pass or certain severe harm.
Shared normal-DA factual/style defects remain comparative diagnostics.

If the smoke supports proceeding, full generation should use one predeclared candidate
batch, not repeated top-ups. A practical proposal is 972 candidates (108 per principle),
then deterministic selection of 716 accepted rows: 80 for the first five principles
and 79 for the remaining four, sorted by a fixed seed-0 hash within each principle.
If any quota is short, report the shortfall; do not add unbudgeted replacement rounds.
This size/selection proposal requires a full-run config and spending approval before
launch, and does not change this 18-row smoke or claim a guaranteed yield. Full-run
filters also have a 25% drop alarm at >=20 inputs, unlike small-smoke reporting; retain
that alarm. Model settings, source/config hashes, code revision, all stages, raw
receipts and exact selected IDs must travel with the published artifact. This means
procedural reproducibility and replayable artifacts, not bit-identical fresh API output.
