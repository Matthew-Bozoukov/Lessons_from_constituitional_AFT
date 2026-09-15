# ABOUTME: Path-invoked equivalent of uv run synth, selecting the generation method from YAML.
# ABOUTME: Delegates to the shared CLI; supports constitutional and deliberative SFT recipes.
#
# Run: uv run scripts/data/synth/build_dataset.py --config configs/data/synth/da.yaml [--smoke]
#      uv run scripts/data/synth/build_dataset.py --config configs/data/synth/par.yaml [--smoke]
#      ... --ablate revise_responses   # ablation arm: run that stage's null-op instead
#      ... --ablate corpus         # skip the corpus-level checks (and their judging)
#      ... --estimate              # print the cost estimate instead of running

from __future__ import annotations

import sys

import fire
from src.data.synth import cli


def main(config: str, smoke: bool = False, resume: str | None = None,
         ablate: str | None = None, overrides: str | None = None,
         estimate: bool = False, measured: str | None = None,
         batch: bool = False) -> None:
    """Build a synthetic dataset from a pipeline config.

    Args:
        config: Path to the run YAML. Its `stages:` list (prompts included) defines
            the document type; `pipeline:` is the label recorded in the manifest.
        smoke: Merge the config's `smoke:` overrides -- tiny slice, full wiring.
        resume: Existing run directory to continue instead of starting fresh.
        overrides: Comma-separated OmegaConf dotlist applied over the YAML, e.g.
            "total_scenarios=144,id_prefix=b" -- keeps a one-off variant (a top-up, a
            different size) as one reproducible command rather than a forked config.
            Recorded in the run manifest either way.
        ablate: Comma-separated stage names to ablate (each must declare
            `ablate_with`); merged over the config's `ablate:` list.
        estimate: Print the cost estimate and exit without generating.
        measured: With --estimate: a smoke run's manifest.json, to price from real
            per-call token counts.
        batch: Route the bulk of every paid stage through OpenRouter's async batch
            API (50% token pricing, results within 24h): llm_json/llm_tagged batch
            their records, scenarios batches per steering wave (diversity preserved),
            and parse/lint rejects mop up interactively. Equivalent to `batch: true`
            in the config; a stage opts out with `batch: false` on its entry.
    """
    if estimate:
        cli.estimate(config, measured=measured, overrides=overrides, ablate=ablate, batch=batch)
        return
    if measured:
        raise ValueError("--measured requires --estimate")
    cli.run(config, smoke=smoke, resume=resume, ablate=ablate,
            overrides=overrides, batch=batch)


if __name__ == "__main__":
    fire.Fire(main, command=["--", "--help"]
              if any(arg in {"--help", "-h"} for arg in sys.argv[1:]) else None)
