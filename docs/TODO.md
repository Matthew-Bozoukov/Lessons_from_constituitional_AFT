1. Check `scripts/`, it has a lot of boilerplate wrappers around functionality that could be resued several times in different pipelines (functionality that probably therefore belongs in `src/`), but also would be useful to run alone (e.g. `merge_lora.py`). Remove the files from `scripts/` that will never be used alone, and if there is a script that imports code that is *only* ran alone consider removing it from `src/` and moving it to `scripts/`. 
2. Adjust the dashboard to read in *all* data from HF by default.
3. Review and understand code in `synthdoc/`.
4. Review and understand code in `eval/vulnerabilities/`.
5. Centralise all configs into `configs/` (e.g. there should be no configs in `src/`; remaining: `src/eval/misalignment/internalization/control/configs/`).
6. Make `src/eval/vulnerabilities/` conform to the "Where code runs" model in `CLAUDE.md` (runs on the GPU pod from a clone of this repo, plain `uv run`): it currently has its own nested environment and workflow that predate the rule.
7. Review `lmsys` and `arena_hard`. Give them better names and make sure that they don't rerun generations for targets or references that have already generated responses and make sure said responses are pushed to HF appropriately.
8. Serving assumes thinking mode on. vLLM's reasoning parser — the setting that splits a model's
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
9. Execute the full 20/80 model-eval-model SFT run (1 epoch, ~10.5k examples, 2,100 docs = 420/cell) — plan, sizing math and past-run specifics in `docs/plan_full_sft_20_80.md` (blocked on: `hf_repo` at launch, spend sign-off ~$147 gen + ~$30 GPU).
10. `configs/eval/swebench_mini_verified.yaml` asks for a 65536 window — inside Qwen3.6's 262144 trained window, but larger than anything booted on one 80GB card in bf16. Confirm on the first real run whether the KV cache allocates; if it does not, the lever is fp8 (KV 252k → 678k tokens, LOG 2026-07-29) or a bigger card, NOT a smaller window (long trajectories then abort and score unresolved, which reads as incapability). Record whichever value actually served.
9. `src/data/mixture/convert_synthdoc_qwen.py` still renders with the Qwen template at build
time. Its `_normalise` half (merge system turns, fix tool_calls, thinking→reasoning_content)
is exactly the model-agnostic interchange conversion — split that out as the converter, drop
the render half, and let train-time rendering (2026-08-06 policy) own the template. The
`format: rendered` mixture sources it feeds migrate to interchange rows at the same time.
