# Canary logit study (2026-10-11)

Callum's proposal (Group Callum Megadoc, "Callum Extra Meeting - October 9"): the canary word may be moving
in the LOGITS without ever being sampled, so measure the logprob the model assigns to the canary's first token
at every place a canary could begin, off-policy (every model on the plain control's transcripts) and on-policy
(each model on its own), and plot each arm as a difference from the control.

Five arms, all seed 0 on the 2026-10-05 plain base (`scratch/logit/arms.py` holds the repos):
plain (control), da, da-tools, da-canary, da-tools-canary. Two evals: MASK (pass 1) and ODCV lite (3 passes).

Pipeline (run from the repo root):

1. `uv run python -m scratch.logit.strip_canary_mix` — the da-tools-15 mix (no canary) on the plain base, from the
   10-09 da-tools-canary mix. Then train it: `uv run train --config configs/train/sft.yaml model=qwen36
   data_repo=dougalldeepmind/2026-10-11-da-tools-15-mix seed=0`, and `uv run evals --name {odcv,mask}` on it and on
   `2026-10-09-qwen36-0-da-tools-15-canary` (MASK with `max_generation_error_rate=0.02`).
2. `uv run python -m scratch.logit.transcripts --arm <arms>` — pulls each arm's ODCV + MASK run and writes one
   record per reasoning turn to `output/logit/turns/`.
3. `uv run python -m scratch.logit.judge --arm <arms>` — trace-level moral-deliberation grade (none / passing /
   substantial) per turn, Gemini 3 Flash, cached in `output/logit/judge_labels.jsonl`.
4. On a GPU host with the repo + `output/logit/turns/`: `uv run python -m scratch.logit.score --model <arm|base>
   --rollouts <arms>` — one teacher-forced pass per transcript (per ODCV segment), the canary's first-token logprob
   at every opener / paragraph / indented-line / mid-line sentence start, to `output/logit/scores/`.
5. `uv run python -m scratch.logit.plot` — the histograms (`output/figures/<date>_logit_canary_<eval>_<split>.png`)
   and the summary table (`<date>_logit_canary_results.md`).

Positions and targets: a paragraph start (after a newline, or the think opener) scores token `Vor`; a sentence
start after a space scores ` Vor` (`scratch/logit/positions.py`). Both logprobs are recorded at every position.
