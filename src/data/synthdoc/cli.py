# ABOUTME: Fire entrypoint for the synthdoc difficult-advice and MEM pipelines.
# ABOUTME: uv run synthdoc run|mem --config configs/data/<name>.yaml [--smoke]

from __future__ import annotations

import json

import fire
from dotenv import load_dotenv
from omegaconf import OmegaConf

from .estimate import estimate as _estimate
from .estimate import estimate_mem as _estimate_mem
from .pipeline import run as _run
from .pipeline import run_mem as _run_mem
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


def mem(config: str, smoke: bool = False, resume: str | None = None) -> None:
    """Run the MEM (model-evaluates-model) pipeline over a completed run.

    Args:
        config: Path to the MEM run YAML.
        smoke: Validate wiring on 2 documents per enabled cell.
        resume: Existing run directory to continue instead of starting fresh.
    """
    load_dotenv()
    cfg = OmegaConf.to_container(OmegaConf.load(config), resolve=True)
    _run_mem(cfg, smoke=smoke, resume=resume)


def check(config: str, run_dir: str, sample: int | None = None) -> None:
    """Run the corpus validity checks over a MEM run and gate on the config's thresholds.

    Args:
        config: Path to the MEM run YAML (its `checks:` block supplies judge + gates).
        run_dir: The run directory holding the stage snapshots.
        sample: Override the number of documents the LLM-judged checks sample.

    Raises:
        SystemExit: Nonzero when any gated check fails; the full report is still
            written to <run_dir>/checks_report.json first.
    """
    load_dotenv()
    from .checks import run_checks

    cfg = OmegaConf.to_container(OmegaConf.load(config), resolve=True)
    _, ok = run_checks(run_dir, cfg, sample=sample)
    if not ok:
        raise SystemExit(1)


def estimate(config: str, measured: str | None = None) -> None:
    """Print a cost estimate for a full run (synthdoc or MEM, by config shape).

    Args:
        config: Path to the run YAML.
        measured: Optional manifest.json from a smoke run, to price from real token counts
            instead of assumptions.
    """
    cfg = OmegaConf.to_container(OmegaConf.load(config), resolve=True)
    est = _estimate_mem(cfg, measured) if "cells" in cfg else _estimate(cfg, measured)
    print(json.dumps(est, indent=2))


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
    fire.Fire({"run": run, "topup": topup, "mem": mem, "check": check,
               "estimate": estimate, "segment": segment})


if __name__ == "__main__":
    main()
