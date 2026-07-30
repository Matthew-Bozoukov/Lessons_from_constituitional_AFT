<!-- ABOUTME: Open questions and loose ends from the 2026-07-30 restructure, for Matthew -->
<!-- ABOUTME: to review. Add items as they come up; delete items once resolved. -->

# Refactor notes — for Matthew's review

Loose ends from the 2026-07-30 restructure (root `src/` + `scripts/` + `scratch/`
layout, `Visualizer/` → `dashboard/`, audit record removed at the tip). Add
items as they come up; delete each item once it is resolved.

1. **Possible redundancy in `src/data/`.** There exist several files related to
   data generation in `src/data/`. It is unclear which of these are now
   redundant (e.g. `generate_difficult_advice.py` seems to be superseded by the
   contents of `synthdoc/`). Remove the stuff that is no longer used.
2. Check the section "## The pipeline" in `CLAUDE.md` and make sure it is up to date, given the changes made after refactor note 1.
3. Check `scripts/`, it has a lot of boilerplate wrappers around functionality that could be resued several times in different pipelines (functionality that probably therefore belongs in `src/`), but also would be useful to run alone (e.g. `merge_lora.py`). Remove the files from `scripts/` that will never be used alone, and if there is a script that imports code that is *only* ran alone consider removing it from `src/` and moving it to `scripts/`. 
4. SFT appears to be doing backprop on full sequence, not only the tokens that the model is meant to generate (see slack DM Jamie->Matthew). Check this.
5. We were previously using gitignored `third_party/` to run stuff like ODCV direct from a third party cloned repo. This seems like an unstable way to do things but I am guessing we kinda have to because no proper package exists. Perhaps we should fully incorporate the relevant code into our codebase? Idk what best practices are here. 
6. We are missing HF code to upload output from some stages in the pipeline (e.g. eval results/sample transcripts like from ODCV). Once we have done that we need to adjust the dashboard to read in *all* data from HF by default.
7. I have no idea what's going on in `synthdoc/`. Kunwar says trust me bro.
8. I haven't ran anything with this refactored repo yet so the there is definitely gonna be some issues I haven't found. We should test it tomorrow.
9. I haven't reviewed Nika's petri and surf code. I am keen to do this next.
