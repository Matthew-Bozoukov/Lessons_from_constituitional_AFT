1. Check `scripts/`, it has a lot of boilerplate wrappers around functionality that could be resued several times in different pipelines (functionality that probably therefore belongs in `src/`), but also would be useful to run alone (e.g. `merge_lora.py`). Remove the files from `scripts/` that will never be used alone, and if there is a script that imports code that is *only* ran alone consider removing it from `src/` and moving it to `scripts/`. 
2. Adjust the dashboard to read in *all* data from HF by default.
3. Review and understand code in `synthdoc/`.
4. Review and understand code in `eval/vulnerabilities/`.
5. Centralise all configs into `configs/` (e.g. there should be no configs in `src/`; remaining: `src/eval/misalignment/internalization/control/configs/`).
6. Make `src/eval/vulnerabilities/` conform to the "Where code runs" model in `CLAUDE.md` (runs on the GPU pod from a clone of this repo, plain `uv run`): it currently has its own nested environment and workflow that predate the rule.
7. Serving assumes thinking mode on. vLLM's reasoning parser — the setting that splits a model's
`<think>` reasoning from its visible answer into separate fields — is only switched on for
`mode=think` arms. Each model family names its parser in `ModelProfile.serving`
(`src/utils.py`).

It stays off otherwise because model with thinking mode off never emits a closing `</think>` tag, and the
*thinking* vLLM parser treats everything as reasoning until it sees one — so it would swallow the entire
answer and return nothing visible.

This costs nothing today: every arm we serve is a thinking arm. If `nothink` or `default`
arms come back, decide per-mode behaviour deliberately and test it against a live endpoint.
Until then those modes split the reasoning out client-side with `split_think`, which handles
every response shape.
8. Execute the full 20/80 model-eval-model SFT run (1 epoch, ~10.5k examples, 2,100 docs = 420/cell) — plan, sizing math and past-run specifics in `docs/plan_full_sft_20_80.md` (blocked on: `hf_repo` at launch, spend sign-off ~$147 gen + ~$30 GPU). NOTE (2026-08-06): its mixture step should now use an interchange config (`path:` + `reasoning: native` — `metadata.supervise` is lifted automatically) instead of the plan's `convert_synthdoc_qwen.py` + `format: rendered` route; that converter was deleted 2026-08-06.
9. Retire legacy rendered mode from `build_mixture.py` (`reasoning: strip` / `format: rendered` and the `_render_without_think`/`_take_hf`/`_load_source_legacy` block) once the configs depending on it are archived to git history — they exist only to regenerate pre-2026-08-06 published artifacts. If the old v1 agentic corpora (multi-system turns, string-encoded tool_calls) are ever reused, port `_normalise` from git history (deleted convert_synthdoc_qwen.py) as a `sources/` adapter.
