<!-- ABOUTME: Layout and conventions for constitutions/ — one folder per constitution, constitution.md + rationale.md. -->
<!-- ABOUTME: The current alignment target is claude_distilled_07_principles_approved/; superseded constitutions move to archive/. -->

# Constitutions

The alignment targets this repository trains toward and grades against. Data generation
(`configs/data/synthdoc/difficult_advice.yaml`, `src/data/prompts.py`) points at these documents.

## Layout — one folder per constitution

```
constitutions/
  <descriptive_name>/       snake_case; states source, derivation, principle count and
                            status/version, e.g. claude_distilled_07_principles_approved
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
  folder (`claude_distilled_07_principles_approved`, `claude_distilled_24_principles_fine` —
  never a bare `claude_approved` or `v2`). Zero-pad the principle count to two digits so
  folders sort lexicographically by granularity.
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

## Generating new constitutions

New constitutions are produced by the one-off `specgen` tool (`scratch/specgen/`, see
its README), which distills the published Claude constitution into specs at controlled
granularities with claim-level coverage accounting. Selected outputs are promoted into
folders here following the layout above.

## Current constitutions

| folder | status | used by |
|---|---|---|
| `claude_distilled_07_principles_approved/` | **current alignment target** (7 principles, distilled from Anthropic's *Claude's Constitution*, Jan 2026) | `configs/data/difficult_advice_gen_v2.yaml`, `src/data/prompts.py::CONSTITUTION_V2` |
| `claude_distilled_04_principles_coarse/` | experiment arm (granularity study, machine-distilled by specgen, 2026-08-03) | nothing yet — spec-variation experiment |
| `claude_distilled_12_principles_mid/` | experiment arm; **default constitution for synthdoc data generation since 2026-08-03** (re-cut 12→10 on 2026-08-04, then set byte-identical on 2026-08-05 to the 9-principle generation-time snapshot below; folder name kept) | `configs/data/synthdoc/difficult_advice.yaml`, `configs/data/synthdoc/self_reflection.yaml` (9 traits) |
| `claude_distilled_09_principles_mid_20260804/` | frozen snapshot — the never-committed 9-principle interim state of the mid re-cut that the 2026-08-04 difficult-advice corpus (`LASR-Callum/synthdoc-v2-difficult-advice`) was generated against; reconstructed byte-exact (sha `fe2ed960…`), **never edit**. The mid folder above now carries identical bytes; this copy stays frozen as the provenance anchor | `configs/data/synthdoc/model_eval_model.yaml` |
| `claude_distilled_24_principles_fine/` | experiment arm (granularity study, machine-distilled by specgen, 2026-08-03) | nothing yet — spec-variation experiment |
| `archive/claude_distilled_8_principles_v1/` | v1, superseded 2026-07-29; synthdoc default until 2026-08-03 | `src/data/prompts.py::CONSTITUTION_V1`; pre-2026-08-03 synthdoc corpora |
