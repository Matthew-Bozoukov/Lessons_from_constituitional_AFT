# ABOUTME: Fire entrypoint for the synthdoc_v2 difficult-advice pipeline.
# ABOUTME: uv run python -m src.data.synthdoc_v2.cli run --config configs/synthdoc_v2.yaml [--smoke]

from __future__ import annotations

import json

import fire
from dotenv import load_dotenv
from omegaconf import OmegaConf

from .estimate import estimate as _estimate
from .pipeline import run as _run


def run(config: str, smoke: bool = False, resume: str | None = None) -> None:
    """Run the six-stage pipeline.

    Args:
        config: Path to the run YAML.
        smoke: Validate wiring on 2 traits x 1 scenario.
        resume: Existing run directory to continue instead of starting fresh.
    """
    load_dotenv()
    cfg = OmegaConf.to_container(OmegaConf.load(config), resolve=True)
    _run(cfg, smoke=smoke, resume=resume)


def estimate(config: str, measured: str | None = None) -> None:
    """Print a cost estimate for a full run.

    Args:
        config: Path to the run YAML.
        measured: Optional manifest.json from a smoke run, to price from real token counts
            instead of assumptions.
    """
    cfg = OmegaConf.to_container(OmegaConf.load(config), resolve=True)
    print(json.dumps(_estimate(cfg, measured), indent=2))


def segment(constitution: str = "constitutions/claude_constitution_principles.md") -> None:
    """Print the traits the constitution segments into, without calling any model.

    Args:
        constitution: Path to the constitution markdown.
    """
    from .constitution import segment as _segment

    traits, style = _segment(constitution)
    for t in traits:
        print(f"{t.trait_id}  {t.name}")
        print(f"     {t.text[:150]}")
    print(f"\n{len(traits)} traits, {len(style)} chars of shared style guidance")


if __name__ == "__main__":
    fire.Fire({"run": run, "estimate": estimate, "segment": segment})
