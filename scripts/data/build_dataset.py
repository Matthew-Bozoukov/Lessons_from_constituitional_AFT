# ABOUTME: Thin CLI over src/data/synthdoc/pipeline.py -- THE entrypoint for synthetic
# ABOUTME: dataset generation. The config (incl. prompts) fully defines the document type.
#
# Run: uv run scripts/data/build_dataset.py --config configs/data/difficult_advice.yaml [--smoke]
#      uv run scripts/data/build_dataset.py --config configs/data/model_eval_model.yaml [--smoke]
#      ... --ablate final          # ablation arm: run the stage's null-op instead
#      ... --estimate              # print the cost estimate instead of running

from __future__ import annotations

import json

import fire
from dotenv import load_dotenv
from omegaconf import OmegaConf

from src.data.synthdoc import pipeline


def main(config: str, smoke: bool = False, resume: str | None = None,
         ablate: str | None = None, estimate: bool = False,
         measured: str | None = None) -> None:
    """Build a synthetic dataset from a pipeline config.

    Args:
        config: Path to the run YAML. Its `stages:` list (prompts included) defines
            the document type; `pipeline:` is the label recorded in the manifest.
        smoke: Merge the config's `smoke:` overrides -- tiny slice, full wiring.
        resume: Existing run directory to continue instead of starting fresh.
        ablate: Comma-separated stage names to ablate (each must declare
            `ablate_with`); merged over the config's `ablate:` list.
        estimate: Print the cost estimate and exit without generating.
        measured: With --estimate: a smoke run's manifest.json, to price from real
            per-call token counts.
    """
    load_dotenv()
    cfg = OmegaConf.to_container(OmegaConf.load(config), resolve=True)
    if ablate:
        cfg["ablate"] = sorted(set(cfg.get("ablate") or [])
                               | {a.strip() for a in str(ablate).split(",") if a.strip()})
    if estimate:
        print(json.dumps(pipeline.estimate(cfg, measured), indent=2))
        return
    pipeline.run(cfg, smoke=smoke, resume=resume)


if __name__ == "__main__":
    fire.Fire(main)
