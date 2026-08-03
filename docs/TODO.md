1. **Possible redundancy in `src/data/`.** There exist several files related to
   data generation in `src/data/`. It is unclear which of these are now
   redundant (e.g. `generate_difficult_advice.py` seems to be superseded by the
   contents of `synthdoc/`). Remove the stuff that is no longer used.
2. Check the section "## The pipeline" in `CLAUDE.md` and make sure it is up to date, given the changes made after refactor note 1.
3. Check `scripts/`, it has a lot of boilerplate wrappers around functionality that could be resued several times in different pipelines (functionality that probably therefore belongs in `src/`), but also would be useful to run alone (e.g. `merge_lora.py`). Remove the files from `scripts/` that will never be used alone, and if there is a script that imports code that is *only* ran alone consider removing it from `src/` and moving it to `scripts/`. 
4. We are missing HF code to upload output from some stages in the pipeline (e.g. eval results/sample transcripts like from ODCV). Once we have done that we need to adjust the dashboard to read in *all* data from HF by default.
5. Review and understand code in `synthdoc/`.
6. Review and understand code in `eval/vulnerabilities/`.
7. Centralise all configs into `configs/` (e.g. there should be no configs in `src/` but there currently are loads in `src/data/synthdoc/`).
8. Make `src/eval/vulnerabilities/` conform to the "Where code runs" model in `CLAUDE.md` (runs on the GPU pod from a clone of this repo, plain `uv run`): it currently has its own nested environment and workflow that predate the rule.
