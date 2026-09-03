# Multi-Agent Exploration

Does constitutional SFT change what a model does on a **team**?

Constitutional-SFT results — Anthropic's *Teaching Claude Why*, the GDM synthetic-document
traits work, and this repo's own difficult-advice arms — are trained and measured with one
model acting alone. Hammond et al. 2025 (*Multi-Agent Risks from Advanced AI*,
arXiv 2502.14143) argue single-agent safety does not imply multi-agent safety and name
collusion between agents as one of three core failure modes. Nobody has measured whether
constitutional training changes collusion, or whether its effect is smaller in a team than
alone. Three experiments here do both.

## The two arms

| | repo | share |
|---|---|---|
| **control** | `LASR-Callum/2026-08-04-qwen36-lora-table2-only-9284-rank-64` | 0% synthetic (Table-2 9,284 alone) |
| **treatment** | `LASR-Callum/2026-08-21-qwen36-lora-table2-9284-difficult-advice-chunk-only-702-rank-64-dynbatch` | 7.03% principle-only difficult advice (702 rows) |

Both verified: base `Qwen/Qwen3.6-27B`, mode `think`, LoRA rank 64, `training_meta.json`
present. Weights only differ; every prompt is identical.

That they share base model *and* thinking mode is what makes the whole study cheap: vLLM
holds both LoRA adapters against one base at once, so a six-seat mixed team runs on **one
GPU**, and each seat picks its arm by name in the request body. `ServedTarget.sibling()`
attaches the second one and refuses if base or mode disagree — disagreeing would restart
the server and silently run the episode with one arm in both seats.

## The environment

Colosseum's Jira task (Nakamura et al. 2026, arXiv 2602.15198) on Terrarium: six agents
divide 8 microtasks to maximise a shared joint reward, four planning rounds of messaging
on a shared blackboard then one execution pass of `assign_task` tool calls. Because the
task is a DCOP, the best achievable allocation is **computable** — solved exactly as a
bipartite assignment — so misbehaviour is a measured drop in outcome, not a reading of a
transcript.

A seed fixes the ticket set, the agents' private skills and cost matrix, and who the
coalition is. It is provably independent of topology, model, colluder count, channel and
prompt variant, so **the same seed is the same task instance in every cell and every
arm** — the pairing is exact by construction. It does *not* fix agent behaviour: no
sampling seed reaches the provider, so rollout noise rides on top of the paired instance
and is what the seed-to-seed interval measures.

## The three experiments

One `experiment=` override apart; the two arms are the two `--target`s.

| | seats | cells | seeds | episodes |
|---|---|---|---|---|
| **1 collusion** | 2 coalition seats = arm under test, 4 = control | baseline (no channel, no objective) / collusion (both) | 40 | 160 |
| **2 single** | 1 seat = arm under test with the private objective, no partner, no channel; 5 = control | baseline / single | 40 | 160 |
| **3 cooperation** | all 6 = same arm, no objective, no channel | cooperation | 20 | 40 |

**360 episodes total.**

Measures, per episode: `coalition_advantage` (coalition mean reward minus everyone
else's) and `normalised_regret` (`1 − achieved/optimal`, clipped). The judge score — a
frontier model reading the blackboards, 0–5 on three questions — is kept as a **secondary**
measure, because Colosseum's paper found talk and action diverge in both directions.

### Reading the three together

- Treatment lower in 1 **and** 2, ratio near 1, Exp 3 clean → the constitutional
  disposition transfers to multi-agent collusion.
- Lower in 2 but not 1 → single-agent alignment fades under peer interaction.
- Judge score drops but regret does not → the training taught clean talk, not clean
  behaviour.
- Exp 3 degraded → part of the "alignment" is a cooperation tax, and a treatment that
  "colludes less" may just be a worse teammate.

## How it runs

Three stages, split by **where the network is**. Killarney compute nodes have no outbound
route, so everything that needs one happens on a login node.

```bash
# 1. ONCE, on a login node (~30 min cold: Qwen3.6-27B is ~54GB).
#    Builds the venv, clones Colosseum at ac0b405, applies the patch, stages all weights.
bash scripts/infra/slurm/setup_killarney.sh

# 2. On an H100. One job per experiment is the safe shape (see "Runtime" below).
EXPERIMENTS=collusion   bash scripts/infra/slurm/submit_colosseum.sh
EXPERIMENTS=single      bash scripts/infra/slurm/submit_colosseum.sh
EXPERIMENTS=cooperation bash scripts/infra/slurm/submit_colosseum.sh

# 3. Back on a login node: judge the episodes, push every run dir to kunwar45 on HF.
bash scripts/infra/slurm/publish_runs.sh
```

Inside stage 2 the job runs, per experiment, one invocation of the repo's single eval
entrypoint:

```
uv run evals --name colosseum_jira --no-push \
    --target <control> <treatment> experiment=collusion
```

`run_eval` serves both adapters on localhost, calls `run()` once per arm, and then calls
`pool()` — which is where the contrast is computed, because **one arm alone says nothing
here**. "The coalition captured 0.19 more reward" only means something against what the
control coalition captured on the same ticket sets.

**H100, not L40S.** Qwen3.6-27B is ~54GB in bf16 and the repo's vLLM passes no
`--tensor-parallel-size`, so one server is one GPU and a 48GB L40S cannot hold it. L40S
schedules far sooner on this cluster, which makes this the main queueing cost. The job
re-checks the card on the node and exits in seconds rather than wasting the allocation.

## Where the results are published

To **`kunwar45`** (a personal namespace), not the group org the rest of the repo uses:

```
kunwar45/2026-09-03-colosseum-jira-collusion-qwen36-table2-only-9284
kunwar45/2026-09-03-colosseum-jira-collusion-qwen36-difficult-advice-chunk-only-702
kunwar45/2026-09-03-colosseum-jira-single-…            (× 2 arms)
kunwar45/2026-09-03-colosseum-jira-cooperation-…       (× 2 arms)
```

`src.huggingface.hf_org` resolves the namespace from `HF_ORG` in the environment and
refuses to take one from a config, so `scripts/eval/publish_colosseum.py --hf-org`
sets that variable; it defaults to `kunwar45` and prints the destination before pushing.

Three consequences:

- **The group dashboard will not show these runs.** It discovers eval runs by org plus
  tags, so runs outside that org are invisible to it. Publish to the group org instead
  (`--hf-org LASR-Callum`) if they should appear.
- The two adapters under test are still **read** from their own org. `HF_ORG` governs
  pushes only.
- `uv run evals --name colosseum_jira` must be given **`--no-push`**. This eval publishes
  through `publish_colosseum.py`, which is the only place that knows the personal
  namespace and the short arm labels; run_eval's own push would use the group org and the
  full adapter id. It fails at preflight rather than after the episodes, but it fails.

Arm labels are written out per arm in the config (`arm_labels`) rather than derived from
the adapter repo id. That id already carries its own production date and its rank,
batching and launcher detail, so reusing it double-dates the run name and overshoots the
Hub's 96-character limit — measured at 119 for the treatment arm. `tests/test_colosseum_publish.py`
asserts all six names pass the naming gate, so that failure surfaces in CI rather than on
a login node with the GPU bill already paid.

## Statistics

Both estimators pair on seed, on top of `src/eval/stats.py`.

`arm_difference` is treatment minus control. For Experiments 1 and 2 the headline is
`channel_advantage`: the within-arm, within-seed difference first (what the objective and
the channel *buy* that arm over the same arm with neither), then the between-arm
difference of those. Differencing within the arm first removes the seed's own generosity —
some ticket sets simply have more surplus to capture — which is the largest variance
source in this environment.

`transfer_ratio` is Experiment 1's effect over Experiment 2's. It is a **ratio of two
noisy estimates** with no usable closed-form interval: when the denominator's interval
covers zero the ratio's does not converge, and a naive SE would report a tight interval
around a meaningless number. It is bootstrapped over a shared seed draw, and it
**refuses to report** unless the single-agent effect is separated from zero. That refusal
is the point — "the disposition transfers" and "neither experiment moved" produce
similar-looking ratios, and only the denominator check separates them. Treat it as a
pre-registered gate, not a diagnostic.

## What the sample figure shows

`output/colosseum/2026-09-03_colosseum_jira_collusion_simulated.png`, from
`scratch/colosseum_mock_exp1_plot.py` — **invented numbers**, drawn to agree the analysis
before any GPU is rented. Panel A is coalition advantage per arm (40 seed dots + mean and
95% CI); Panel B is normalised regret in both conditions, so an arm that "colludes less"
by cooperating worse is visible rather than hidden.

## Open questions

1. **Judge model.** Defaulted to `google/gemini-3-flash-preview` — not the target family
   (Qwen) and deliberately not Sonnet, because the difficult-advice corpus the treatment
   was trained on was *written* by Sonnet 4.5 and a Sonnet judge would be scoring a voice
   distilled from its own. **Confirm before the first paid judge run.**
2. **Runtime, and whether 360 episodes fit.** Up to ~78 LLM calls per episode (6 agents ×
   4 planning rounds × 3 steps, plus execution), thinking traces at 4,096 max_tokens, and
   **no prefix caching** — this architecture cannot reuse the long shared blackboard
   prefix, so it is re-encoded every call. A rough estimate puts the full 360 at the edge
   of a 12h allocation, with wide error bars. Hence one job per experiment, and a timed
   3-seed smoke first (`SEEDS=1-3 TIME=01:00:00`) to calibrate before committing.
3. **Experiment 2 costs 160 episodes, not the 80 in the original design.** It needs its own
   baseline cell, or its effect is not the same *quantity* as Experiment 1's and the
   transfer ratio is a ratio of unlike things. Droppable if the budget is tight, at the
   cost of the ratio.
4. **40 seeds is an assumption, not a power calculation.** Ample if the seed-to-seed SD is
   ~0.11 against a ~0.10 effect; badly underpowered if SD is ~0.2 against ~0.05. The smoke
   run should be used to estimate the variance.
5. **The single-agent prompt is slightly odd.** With one colluder, upstream renders
   "Coalition members: \<itself\>" and a null blackboard id. Worth reading one rendered
   prompt from the smoke before the full run.

## Status

Built and unit-tested; **nothing has run on a GPU**. The cluster has not been touched —
connecting needs a Duo tap.

- 1,223 tests pass, 15 of them new (`tests/test_colosseum_stats.py`,
  `tests/test_colosseum_config.py`).
- The registry entry resolves, all three sweeps build and round-trip through YAML.
- The Colosseum patch applies cleanly to `ac0b405` (verified with `git apply --check`).
- Untested until data exists: `runner.py`'s subprocess plumbing, `harvest.py`'s parsing of
  a real output tree, and the judge against a live blackboard.
