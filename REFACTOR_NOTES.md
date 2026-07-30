<!-- ABOUTME: Open questions and loose ends from the 2026-07-30 restructure, for Matthew -->
<!-- ABOUTME: to review. Add items as they come up; delete items once resolved. -->

# Refactor notes — for Matthew's review

Loose ends from the 2026-07-30 restructure (root `src/` + `scripts/` + `scratch/`
layout, `Visualizer/` → `dashboard/`, audit record removed at the tip). Add
items as they come up; delete each item once it is resolved.

1. **Possible redundancy in `src/data/`.** There exist several files related to
   data generation in `src/data/`. It is unclear which of these are now
   redundant (e.g. `generate_difficult_advice.py` seems to be superseded by the
   contents of `synthdoc/`).
