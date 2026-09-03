# Colosseum patches

Colosseum is not vendored into this repo — it is cloned to `/project/.../colosseum` by
`scripts/infra/slurm/setup_killarney.sh`, pinned to a commit, and patched there. What
lives here is the patch, so the modification is reviewable in this repo's history rather
than hidden in a checkout on a cluster.

**Upstream**: `github.com/umass-ai-safety/colosseum` @ `ac0b405`
(`terrarium-agents[providers,science,plots]==0.1.1` from PyPI — *not* the Terrarium git
checkout, whose `main` is 0.2.0 and has reorganised the package layout Colosseum imports).

## `per_agent_models.patch`

**What it does**: lets the six agents in a run be served by *different* models, chosen by
each agent's collusion role, via a new
`experiment.collusion.agent_llms_by_role: {colluder: <llm block>, normal: <llm block>}`.

**Why it is needed**: every experiment here rests on a MIXED team — the coalition seats
hold the arm under test while the rest of the team holds the control. Public Colosseum
cannot express that. `llm_models` is a sweep axis (one model per run, all agents
identical), and the vLLM path removes the possibility outright:

```python
# llm_server/vllm/runtime.py
def get_model_for_agent(self, agent_name):
    del agent_name  # unused
```
raising *"llm.vllm.models now supports exactly one model spec because per-agent routing
was removed"* for more than one spec. Terrarium's own README: *"All agents share the one
configured vLLM model; advanced routing is disabled."*

**Why this shape**: the feature is coming back upstream. Terrarium 0.2.0's test suite
(`terrarium/tests/test_collusion_runs_per_seed.py`) already exercises
`_resolve_agent_llm_configs`, `agent_llms` and `_resolve_colluders_from_config` against
`experiments.collusion.run` — none of which exist in public Colosseum HEAD, so the authors
have this on an unreleased branch. The patch keeps their function name and their
"resolve a config per agent, then build clients from it" structure so it merges rather
than conflicts.

It differs from upstream in one deliberate way: assignment is **by role**, not by
position. `colluder_selection: random` decides the coalition at run time from the seed, so
a positional `agent_llms` list would have to guess which seats end up colluding. Keying on
the role the seed produced is the only way to guarantee the arm under test is the arm in
the coalition.

The patch also records `agent_llm_labels` (agent name → served model) into
`run_config.json` and the summary row. Without it a mixed-team run is indistinguishable
from a single-model one after the fact, and the seating is the independent variable.

**If you re-clone Colosseum, re-apply this patch** — the same standing rule as the
vendored agentic-misalignment harness (CLAUDE.md gotcha 5). Verify with:

```bash
git -C <clone> apply --check src/eval/misalignment/colosseum/third_party/per_agent_models.patch
```

## Upstream behaviour worth knowing (not patched)

- **`health_check_path` default is broken.** `_health_url` joins `api_base()` (already
  ends `/v1`) with the default `"/v1/models"`, giving `.../v1/v1/models` → 404 → "server
  not reachable". Every config here sets `health_check_path: "/models"`. Not patched
  because a config field already fixes it.
- **`request_timeout` defaults to 60s**, far too short for a 27B reasoning model. Set per
  config.
- **Only `max_tokens` and `temperature` reach the server.** `top_p`, `stop`, `seed` and
  `tool_choice` in `params` are silently dropped — which is why runs are not
  sampling-seeded; see the eval config's note.
- **`assignment_filling` defaults to True** (unassigned agents get a random task filled in
  at scoring time). Shipped configs disagree with each other about it, so ours pins it.
- **`system_regret_ratio` divides by a loose analytic bound**, not the optimum. Normalised
  regret comes from `compute_jira_optimal.py` instead.
