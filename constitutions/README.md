<!-- ABOUTME: Layout and conventions for constitutions/ — one folder per constitution, constitution.md + rationale.md. -->
<!-- ABOUTME: The current alignment target is claude_distilled_7_principles_approved/; superseded constitutions move to archive/. -->

# Constitutions

The alignment targets this repository trains toward and grades against. Data generation
(`configs/data/synthdoc.yaml`, `src/data/prompts.py`) points at these documents.

## Layout — one folder per constitution

```
constitutions/
  <descriptive_name>/       snake_case; states source, derivation, principle count and
                            status/version, e.g. claude_distilled_7_principles_approved
    constitution.md         the document itself (what generators and graders consume)
    rationale.md            why it says what it says: changelog, evidence, rejected
                            alternatives, scope limits
    README.md               short metadata card: status, principle/trait count, source
                            material, date generated, last updated, what consumes it
  archive/
    <descriptive_name>/     superseded constitutions, same folder shape
```

- **Folder names are self-describing**: a reader should learn the source material, how it
  was derived, how many principles/traits it carries, and its status without opening the
  folder (`claude_distilled_7_principles_approved`, `claude_distilled_8_principles_v1` —
  never a bare `claude_approved` or `v2`).
- **`constitution.md` is the artifact; `rationale.md` is its justification.** Every new
  constitution gets both — a constitution without a written rationale is not reviewable.
  `README.md` is the at-a-glance metadata card (keep it to one table).
- Both files start with a two-line ABOUTME header; `constitution.md` names its source and
  whether it is verbatim or distilled; `rationale.md` carries the changelog against the
  version it replaced.
- **Superseding:** move the old folder into `archive/` (git mv, so history follows), add an
  archived-banner to its `constitution.md` saying what replaced it and why it is kept, and
  update every config/test that pins it. Configs may keep pinning an archived constitution —
  reproducibility of an existing corpus beats freshness — but the pin must point into
  `archive/` so its status is visible.
- Anything derived from a constitution (prompt strings such as
  `src/data/prompts.py::CONSTITUTION_V2`, HF dataset cards) names the source folder, and a
  test guards against drift where feasible (see `tests/test_prompts.py`).

## Current constitutions

| folder | status | used by |
|---|---|---|
| `claude_distilled_7_principles_approved/` | **current alignment target** (7 principles, distilled from Anthropic's *Claude's Constitution*, Jan 2026) | `configs/data/difficult_advice_gen_v2.yaml`, `src/data/prompts.py::CONSTITUTION_V2` |
| `archive/claude_distilled_8_principles_v1/` | v1, superseded 2026-07-29 | still pinned by `configs/data/synthdoc.yaml` (its 8 principles = the 8 synthdoc traits) and `src/data/prompts.py::CONSTITUTION_V1` |
