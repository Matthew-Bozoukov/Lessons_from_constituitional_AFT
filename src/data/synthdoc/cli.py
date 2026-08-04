# ABOUTME: Fire entrypoint for the synthdoc difficult-advice pipeline.
# ABOUTME: uv run synthdoc run --config configs/data/synthdoc.yaml [--smoke]

from __future__ import annotations

import json

import fire
from dotenv import load_dotenv
from omegaconf import OmegaConf

from .estimate import estimate as _estimate
from .pipeline import run as _run
from .pipeline import topup as _topup


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


def topup(config: str, resume: str, traits, n: int = 25) -> None:
    """Bring specific traits up to a target number of stage-6 rewrites.

    Args:
        config: Path to the run YAML.
        resume: Run directory holding the stage snapshots.
        traits: Trait ids -- Fire hands this over as a tuple when it contains commas,
            so both "t5,t6" and a tuple are accepted.
        n: Target completed rewrites per trait.
    """
    load_dotenv()
    cfg = OmegaConf.to_container(OmegaConf.load(config), resolve=True)
    ids = list(traits) if isinstance(traits, (list, tuple)) else \
        [x.strip() for x in str(traits).split(",")]
    _topup(cfg, resume, [x for x in ids if x], n)


def estimate(config: str, measured: str | None = None) -> None:
    """Print a cost estimate for a full run.

    Args:
        config: Path to the run YAML.
        measured: Optional manifest.json from a smoke run, to price from real token counts
            instead of assumptions.
    """
    cfg = OmegaConf.to_container(OmegaConf.load(config), resolve=True)
    print(json.dumps(_estimate(cfg, measured), indent=2))


def segment(constitution: str = "constitutions/claude_distilled_12_principles_mid/constitution.md") -> None:
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


def main() -> None:
    fire.Fire({"run": run, "topup": topup, "estimate": estimate, "segment": segment})


if __name__ == "__main__":
    main()
