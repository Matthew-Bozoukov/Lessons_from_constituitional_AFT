# ABOUTME: Fire entrypoint for synthdoc's generation pipelines, one command group each:
# ABOUTME: uv run synthdoc da|mem <verb> --config configs/data/<pipeline>.yaml [--smoke]

from __future__ import annotations

import json

import fire
from dotenv import load_dotenv
from omegaconf import OmegaConf

from .difficult_advice import estimate as da_estimate
from .difficult_advice import pipeline as da_pipeline
from .mem import estimate as mem_estimate
from .mem import pipeline as mem_pipeline


def _load(config: str) -> dict:
    """Load a run YAML into a plain dict."""
    return OmegaConf.to_container(OmegaConf.load(config), resolve=True)


# --- difficult-advice --------------------------------------------------------------


def da_run(config: str, smoke: bool = False, resume: str | None = None) -> None:
    """Run the six-stage difficult-advice pipeline.

    Args:
        config: Path to the run YAML (configs/data/difficult_advice.yaml).
        smoke: Validate wiring on 2 traits x 1 scenario.
        resume: Existing run directory to continue instead of starting fresh.
    """
    load_dotenv()
    da_pipeline.run(_load(config), smoke=smoke, resume=resume)


def da_topup(config: str, resume: str, traits, n: int = 25) -> None:
    """Bring specific traits up to a target number of stage-6 rewrites.

    Args:
        config: Path to the run YAML.
        resume: Run directory holding the stage snapshots.
        traits: Trait ids -- Fire hands this over as a tuple when it contains commas,
            so both "t5,t6" and a tuple are accepted.
        n: Target completed rewrites per trait.
    """
    load_dotenv()
    ids = list(traits) if isinstance(traits, (list, tuple)) else \
        [x.strip() for x in str(traits).split(",")]
    da_pipeline.topup(_load(config), resume, [x for x in ids if x], n)


def da_estimate_cmd(config: str, measured: str | None = None) -> None:
    """Print a cost estimate for a full difficult-advice run.

    Args:
        config: Path to the run YAML.
        measured: Optional manifest.json from a smoke run, to price from real token
            counts instead of assumptions.
    """
    print(json.dumps(da_estimate.estimate(_load(config), measured), indent=2))


# --- MEM (model-evaluates-model) ---------------------------------------------------


def mem_run(config: str, smoke: bool = False, resume: str | None = None) -> None:
    """Run the MEM pipeline over a completed difficult-advice run.

    Args:
        config: Path to the MEM run YAML (configs/data/mem.yaml).
        smoke: Validate wiring on 2 documents per enabled cell.
        resume: Existing run directory to continue instead of starting fresh.
    """
    load_dotenv()
    mem_pipeline.run(_load(config), smoke=smoke, resume=resume)


def mem_check(config: str, run_dir: str, sample: int | None = None) -> None:
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
    from .mem.checks import run_checks

    _, ok = run_checks(run_dir, _load(config), sample=sample)
    if not ok:
        raise SystemExit(1)


def mem_estimate_cmd(config: str, measured: str | None = None) -> None:
    """Print a cost estimate for a full MEM run.

    Args:
        config: Path to the MEM run YAML.
        measured: Optional manifest.json from a smoke run, to price from real token
            counts instead of assumptions.
    """
    print(json.dumps(mem_estimate.estimate(_load(config), measured), indent=2))


# --- shared ------------------------------------------------------------------------


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
    da = {"run": da_run, "topup": da_topup, "estimate": da_estimate_cmd}
    mem = {"run": mem_run, "check": mem_check, "estimate": mem_estimate_cmd}
    fire.Fire({"da": da, "difficult-advice": da, "mem": mem, "segment": segment})


if __name__ == "__main__":
    main()
