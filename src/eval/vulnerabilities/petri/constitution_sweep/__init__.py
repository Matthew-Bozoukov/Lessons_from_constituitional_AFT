# ABOUTME: Constitution dose-sweep audit: seed building, uniform re-judging, stats, export, figures.
# ABOUTME: Self-contained under petri/ the way internalization/ is self-contained under misalignment/.
"""Petri audit of SFT arms against the constitution their training data came from.

Measures **frequency of violations** across four Qwen3.6-27B arms (0/10/20/40%
difficult-advice SFT) using the v1 distilled constitution decomposed into 44
atomic testable elements.

Why this is a subpackage rather than folded into `petri/build_export.py`: that
module builds the MSM focused-discovery export - one target, a validation-funnel
input, a replication chart. This one is arm-aware and epoch-aware, and pairs
scenarios across four models. They share a file format, not logic. The
`internalization/` eval sits under `misalignment/` on the same reasoning.

Pipeline, all from the repository root:

    # 1. seeds (already committed under seeds/; only needed if specs change)
    python -m ...constitution_sweep.seeds --specs configs/petri/seed_specs.yaml

    # 2. run the audits -> output/petri/logs/<arm>/*.eval   (see "Running the
    #    audits" in the module README: the orchestration is not in this repo)

    # 3. one judge, one transport, every arm
    python -m ...constitution_sweep.rejudge --logs output/petri/logs \
        --out output/petri/rejudged

    # 4. rates, intervals, McNemar, paired severity
    python -m ...constitution_sweep.analyse --rejudged output/petri/rejudged \
        --out output/petri/analysis

    # 5. figures + the greppable markdown mirror
    python -m ...constitution_sweep.plots --results output/petri/analysis/results.json \
        --out output/petri/analysis

    # 6. publishable export, then the Hub manifest + shards
    python -m ...constitution_sweep.export --logs output/petri/logs \
        --rejudged output/petri/rejudged --analysis output/petri/analysis \
        --out output/petri/exports/<date>-constitution-dose-sweep
    python -m ...constitution_sweep.manifest --export <that dir> \
        --meta configs/petri/manifest.yaml --commit <sha>

`scripts/run_petri_analysis.sh` chains steps 3-6.

**Read `adjudication.py` before trusting any number this produces.** Every rate
here is a judge opinion, and the run's own control seeds - which contain nothing
to violate - were flagged 17/8/36/45% across the four arms. The rates are upper
bounds until those flags are ruled on by a human.
"""
