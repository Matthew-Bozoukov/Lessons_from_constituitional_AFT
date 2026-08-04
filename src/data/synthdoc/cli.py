# ABOUTME: Fire entrypoint for the synthdoc generation pipeline. The config's
# ABOUTME: `pipeline:` field names the document type (difficult_advice | mem).

from __future__ import annotations

import json

import fire
from dotenv import load_dotenv
from omegaconf import OmegaConf

from .estimate import estimate as _estimate
from .pipeline import run as _run
from .pipeline import topup as _topup


def _load(config: str) -> dict:
    """Load a run YAML into a plain dict."""
    return OmegaConf.to_container(OmegaConf.load(config), resolve=True)


def run(config: str, smoke: bool = False, resume: str | None = None) -> None:
    """Run the pipeline the config declares (difficult_advice or mem).

    Args:
        config: Path to the run YAML; its `pipeline:` field picks the document type
            (configs/data/difficult_advice.yaml or configs/data/mem.yaml).
        smoke: Validate wiring on a tiny slice (2 traits x 1 scenario, or 2 docs/cell).
        resume: Existing run directory to continue instead of starting fresh.
    """
    load_dotenv()
    _run(_load(config), smoke=smoke, resume=resume)


def topup(config: str, resume: str, traits, n: int = 25) -> None:
    """Bring specific traits up to a target number of stage-6 rewrites.

    Difficult-advice only: tops up an existing run's rewrites per trait.

    Args:
        config: Path to the run YAML (`pipeline: difficult_advice`).
        resume: Run directory holding the stage snapshots.
        traits: Trait ids -- Fire hands this over as a tuple when it contains commas,
            so both "t5,t6" and a tuple are accepted.
        n: Target completed rewrites per trait.
    """
    load_dotenv()
    cfg = _load(config)
    assert cfg.get("pipeline") == "difficult_advice", \
        "topup applies to difficult_advice runs only"
    ids = list(traits) if isinstance(traits, (list, tuple)) else \
        [x.strip() for x in str(traits).split(",")]
    _topup(cfg, resume, [x for x in ids if x], n)


def check(config: str, run_dir: str, sample: int | None = None) -> None:
    """Run the corpus validity checks over a MEM run and gate on the config's thresholds.

    MEM only: difficult-advice corpora are validated downstream by mixture/training
    checks.

    Args:
        config: Path to the run YAML (`pipeline: mem`; its `checks:` block supplies
            judge + gates).
        run_dir: The run directory holding the stage snapshots.
        sample: Override the number of documents the LLM-judged checks sample.

    Raises:
        SystemExit: Nonzero when any gated check fails; the full report is still
            written to <run_dir>/checks_report.json first.
    """
    load_dotenv()
    from .checks import run_checks

    cfg = _load(config)
    assert cfg.get("pipeline") == "mem", "check applies to mem runs only"
    _, ok = run_checks(run_dir, cfg, sample=sample)
    if not ok:
        raise SystemExit(1)


def estimate(config: str, measured: str | None = None) -> None:
    """Print a cost estimate for a full run of the config's pipeline.

    Args:
        config: Path to the run YAML.
        measured: Optional manifest.json from a smoke run, to price from real token
            counts instead of assumptions.
    """
    print(json.dumps(_estimate(_load(config), measured), indent=2))


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
    fire.Fire({"run": run, "topup": topup, "check": check,
               "estimate": estimate, "segment": segment})


if __name__ == "__main__":
    main()
