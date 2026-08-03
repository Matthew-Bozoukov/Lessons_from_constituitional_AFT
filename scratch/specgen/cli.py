# ABOUTME: Fire entrypoint for the specgen constitution-granularity pipeline.
# ABOUTME: uv run scratch/specgen/cli.py <pin|extract|generate|metrics> --config scratch/specgen/specgen.yaml

from __future__ import annotations

import sys
from pathlib import Path

import fire
from dotenv import load_dotenv
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _cfg(config: str) -> dict:
    load_dotenv()
    return OmegaConf.to_container(OmegaConf.load(config), resolve=True)


def pin(config: str, file: str) -> None:
    """Pin the manually saved source constitution and write its hash lock."""
    from pipeline import pin as _pin

    _pin(_cfg(config), file)


def extract(config: str, smoke: bool = False) -> None:
    """Extract the shared claim inventory (run once; all arms build from it)."""
    from pipeline import extract as _extract

    _extract(_cfg(config), smoke=smoke)


def generate(config: str, arm: str | None = None, seeds: int | None = None,
             smoke: bool = False) -> None:
    """Generate specs for all arms (or one) across seeds; skips existing outputs."""
    from pipeline import generate as _generate

    _generate(_cfg(config), arm=arm, seeds=seeds, smoke=smoke)


def metrics(config: str, smoke: bool = False) -> None:
    """Metrics + hard-constraint checks + seed selection + comparison.md."""
    from metrics import run as _run

    _run(_cfg(config), smoke=smoke)


def main() -> None:
    fire.Fire({"pin": pin, "extract": extract, "generate": generate, "metrics": metrics})


if __name__ == "__main__":
    main()
