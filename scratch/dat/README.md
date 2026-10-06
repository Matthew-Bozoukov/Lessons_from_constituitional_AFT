# dat: difficult agentic tasks, from scratch (draft, 2026-10-06)

An AI agent with a shell is deployed by an organisation, given a legitimate task by an operator,
and discovers in the course of the work that one principle of the constitution is at stake and
the right call is genuinely unclear. The row is the deployment, the request, the agent's own
exploration (real commands run in a Docker sandbox, real outputs), and a supervised final turn
(reasoning, message, at most one tool call).

Status: config, renderers and driver written; `--smoke --dry` runs (deal + prompt construction);
the paid path and the sandbox loop have not been run yet. The recipe is `dat.yaml` (REQUIREMENTS,
per-stage prompts, schemas, models, file types, clock); `prompts.py` only loads it and formats each
stage's inputs. Review `dat.yaml` first, or the review page (see the chat for the link).

## What was wrong before, and what this changes

| problem (2026-10-02 smoke, 29 rows; daa-7 corpus) | cause | here |
|---|---|---|
| 72% of rows had the operator ask for or lean toward the shortcut; 52% reduced to comply-vs-refuse | the scenario writer wrote a `shortcut` and the environment stage saw it; the reviser sharpened asks | no `shortcut` field exists; REQUIREMENTS 2 and 3 are in every writing stage; `revise` moves pressure out of the request and into records; `system` and `user` never see the facts; `select` drops rows whose pressure comes from the operator |
| tool results interpreted themselves ("Note: ...", a `ls` that says "no approval flow") in ~25/29 rows | one model wrote prompts, facts and outputs together, knowing the dilemma | `environment` fills per-file_type schemas (events, test cases, rows, keys); code renders the files; the model writes no text for structured file types |
| tools and scripts that existed only for the dilemma (`verify_checklist.py`, `risk_tolerance: aggressive`) | environment written from the answer | scripts are allowed but must be ordinary utilities the organisation would already have, with no flags that exist for the request, and each must run (its check command exits 0); `select` reads their source and output |
| system prompts carrying the rule that resolves the case | reviser "closing loopholes" | `system` does not know the facts or the request, so it cannot write the rule; REQUIREMENT 6 |
| reasoning template reciting the trait ("there's a real pull", "the honest path isn't X") | response stages primed with the shortcut | `respond` sees the principle and the transcript only; `rewrite` strikes set phrases and requires both readings weighed |
| the environment author knew the answer | | `select`, which knows the tension, reads every file and script output for anything that states or points at the conclusion (`settled_by`, quoted) and drops the row if it finds any; a blind `plant` judge was in the first draft and dropped: without knowing the tension it could only see overt remarks, which the structured file types already remove |

## Stages and what each is shown

| stage | sees | does not see | produces |
|---|---|---|---|
| write | principle, sector | | organisation, role, operator + goal, task, world facts (statement, bearing), two readings with costs, situation, not_obvious_because |
| revise | the situation, principle, REQUIREMENTS | | corrected situation, changes |
| environment | organisation, role, world facts (statements only), today | the prompts (not written yet), readings, situation, principle, which facts bear on the tension | directories; one file per entry with a file_type (log, test_report, csv, config, git_log, diff, json, ticket, email, doc, script) and the facts it carries; structured file types rendered by code; scripts must run (check command exits 0; a fact found by running one must appear in its output); mundane files; witness strings |
| system | organisation, role, kind of task, file listing (paths only) | file contents, operator's message, world facts, readings, situation, principle | system prompt |
| user | system prompt, operator + goal, task, file listing (paths only), today | file contents, world facts, readings, situation, principle | operator's message |
| explore | system, user, read-only shell, own outputs | facts, principle, situation | looks: thought, command, real output (cap 8); coverage appends one look per unseen tension fact |
| select | principle, situation, readings, intended facts, prompts, every file and each script's output, transcript | | tension_present, source (work / operator / both / none), unclear, principle_fit, realism, settled_by (files or outputs that state or point at the conclusion, quoted), keep |
| respond | principle, prompts, exploration, tools, style | situation, readings, facts | reasoning, response, at most one tool call |
| rewrite | the same + draft + REQUIREMENTS | situation, readings, facts | revised turn, changes |

`requirements` (dat.yaml) is injected verbatim into write, revise, environment, system, user,
select and rewrite. `stage_inputs()` in pipeline.py is the single place that decides what a stage is given.

## Deliberate non-changes

- The final turn is message + at most one call, as before. Shape variation (call only, several
  calls, no action) is a separate decision (memory: dat-status-and-shape-issue).
- `supervise: final`: exploration turns are context, not loss.
- Diversity is da's current recipe: the dealt sector, "distinct within this set", and nothing
  else at the prompt (da turned its avoid list off on 2026-10-02 because the writer copied the
  list's flavour). The 0.78 cosine gate is not reimplemented here; add it if repeats show on the smoke. If the recipe is promoted, it goes
  into the `ours` pipeline as stage kinds and gets them back.

## Running

```
uv run python scratch/dat/pipeline.py --smoke --dry     # deal + prompts, no spend
uv run python scratch/dat/pipeline.py --smoke           # 9 principles x 1 sector x 2 situations, Docker needed
uv run python scratch/dat/pipeline.py --total 750 --workers 8
```

Output under `output/synth_dat/<run>/`: `hands.jsonl`, `stage_write_revise.jsonl`, `rows.jsonl`
(every row with the stage it reached and why it was dropped), `dataset.jsonl` (kept rows),
`calls.jsonl`, `manifest.json` (yield per stage and trait, select source counts, settled_by counts,
cost). Nothing is pushed by this script.

Models (per stage in dat.yaml): Sonnet 4.5 for write, revise, environment, system, user, explore, respond,
rewrite; Gemini 3 Flash for select. Expected cost per attempted row roughly $0.40–0.60
at Sonnet; yield unknown until the smoke.

## What to read on the smoke

1. `manifest.json`: `select_source` (how many kept rows have source `work`), `select_unclear`,
   `settled_by_nonempty`, `stage_reached`.
2. The dropped rows' `dropped` reasons, and `select`'s `principle_fit` per trait: a trait that is
   repeatedly dropped there does not fit this shape.
3. Five kept rows end to end, against the eight requirements.

Earlier recipe: `archive/dat_2026-10-02.yaml` (the proposal smoked on 2026-10-02); `archive/prompts_py_pre_yaml.py` is
this recipe before its prompts moved into `dat.yaml` (identical content, kept for the diff).
