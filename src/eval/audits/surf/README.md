# SURF — surfacing unintended response failures

The official [SURF](https://github.com/seoirsem/SURF) tool (from *Chunky Post-Training*,
arXiv 2602.05910), vendored under `third_party/SURF/` and retargeted at this project's
difficult-advice corpus and nine-principle constitution.

```
surf/
├── third_party/
│   ├── SURF/            official implementation @ 7d3fe91, with local patches
│   │   └── rubrics/
│   │       ├── rebuttal.yaml   upstream's example rubric
│   │       └── traits.yaml     ADDED — the 9-principle constitution
│   └── PATCHES.md       every local change, and how to re-apply after a re-clone
├── calibrate.py         our API-cost calibration over a completed run
└── validation_funnel.py our two-round validation funnel over a flag set
```

`calibrate.py` and `validation_funnel.py` are this repo's tooling, unchanged in behaviour;
they import from a SURF checkout at runtime and now default to the vendored one.

## What SURF does

It searches *prompt-category space* for regions that elicit a target behaviour, rather than
hand-writing adversarial prompts. Two phases:

1. **prepare-dataset** — extract ~10 semantic attributes per prompt from a corpus, embed
   them, K-Means them into clusters, summarise each cluster, and build a "pseudo-SAE"
   attribute vocabulary. This is the search space, and it works best when it comes from data
   close to what the model was trained on — which is why it points at our own corpus rather
   than Tulu.
2. **sweep** — parallel EM loops that sample attribute combinations, generate queries from
   them, score the target model's responses against a rubric, and concentrate sampling on
   the regions that score highest.

## Running it

Own nested env (`src/eval/audits/` is exempt from the eval-framework contract):

```bash
cd src/eval/audits/surf/third_party/SURF
uv sync
cp ../../../../../../.env .env        # needs OPENROUTER_API_KEY, HF_TOKEN

# 1. Build the search space from the difficult-advice corpus (all defaults preset)
uv run -m surf.cli.main prepare-dataset --output-dir data/synthdoc

# 2. Red-team a target against the 9-principle rubric
uv run -m surf.cli.main sweep \
    --attributes data/synthdoc/pseudo_sae_attributes.jsonl \
    --rubric rubrics/traits.yaml \
    --output-dir results/traits \
    --target-model openrouter:qwen/qwen3.6-27b

# 3. Read the worst offenders
uv run utils/top.py results/traits
```

Step 1 is preset to this project's corpus — 2,203 rows of
`LASR-Callum/2026-08-04-synthdoc-difficult-advice-9-principles` (`stage_7_sft.jsonl`),
1,000 clusters, attribute extraction and cluster summarisation on
`openrouter:openai/gpt-5.6-terra`. It resumes automatically if interrupted; `--force`
re-runs. See `third_party/PATCHES.md` for why each default is what it is.

Step 1 needs a GPU for the embedding stage (`Qwen/Qwen3-Embedding-8B`); steps 2-3 are API
calls and run anywhere.

## The rubric is two-sided — read this before interpreting results

`rubrics/traits.yaml` scores violations of all nine principles of
`constitutions/claude_distilled_09_principles_mid_20260804`. Principles 8 and 9 make
**over-refusal, hedging, moralising and paternalism violations in their own right**, so a
high score does not mean "the model did something harmful" — it can equally mean the model
refused something routine or lectured the user. The judge names the principle; read it.

This matters for the search itself: a rubric that only rewarded harmful or deceptive output
would drive SURF toward eliciting refusals and then score that caution as a win. The rubric
states the two-sidedness explicitly and carries the constitution's own "when this does NOT
apply" clauses as false-positive guards.
