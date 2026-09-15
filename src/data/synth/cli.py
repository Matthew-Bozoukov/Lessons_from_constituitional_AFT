# ABOUTME: One synth command for generation methods, selected by the YAML method field.
# ABOUTME: Keeps existing constitutional SFT commands working and routes deliberative SFT runs.

from __future__ import annotations

import json
import sys
from pathlib import Path

import fire
from dotenv import load_dotenv
from omegaconf import OmegaConf

from .ours import cli as ours


def _config(config: str, overrides=None) -> dict:
    loaded = OmegaConf.load(config)
    if overrides:
        loaded = OmegaConf.merge(loaded, OmegaConf.from_dotlist(ours._csv(overrides)))
    cfg = OmegaConf.to_container(loaded, resolve=True)
    cfg["pipeline"] = Path(config).stem
    method = cfg.get("method", "ours")
    if method not in {"ours", "deliberative_alignment"}:
        raise ValueError(f"Unknown synth method: {method!r}")
    return cfg


def run(config: str, smoke: bool = False, resume: str | None = None,
        ablate: str | None = None, overrides: str | None = None,
        batch: bool = False) -> None:
    """Generate with the YAML's method (defaults to ours for existing configs)."""
    load_dotenv()
    cfg = _config(config, overrides)
    if cfg.get("method", "ours") == "ours":
        ours.run(config, smoke=smoke, resume=resume, ablate=ablate,
                 overrides=overrides, batch=batch)
        return
    if ablate or batch:
        raise ValueError("deliberative_alignment supports neither --ablate nor --batch")
    from .deliberative_alignment.pipeline import run as generate

    generate(cfg, smoke=smoke, resume=resume)


def estimate(config: str, measured: str | None = None, overrides: str | None = None,
             ablate: str | None = None, batch: bool = False) -> None:
    """Estimate a constitutional SFT recipe using its stage-level token assumptions."""
    cfg = _config(config, overrides)
    if cfg.get("method", "ours") != "ours":
        raise ValueError("Deliberative SFT has no token estimator; use a budgeted --smoke run")
    if ablate:
        cfg["ablate"] = sorted(set(cfg.get("ablate") or []) | set(ours._csv(ablate)))
    if batch:
        cfg["batch"] = True
    print(json.dumps(ours.pipeline.estimate(cfg, measured), indent=2))


def main() -> None:
    commands = {"run": run, "estimate": estimate, "topup": ours.topup,
                "check": ours.check, "checks": ours.checks,
                "segment": ours.segment, "chunkings": ours.chunkings}
    # Fire otherwise executes a paid command before rejecting leftover flags.
    ours._refuse_unknown_flags(commands)
    argv = sys.argv[1:]
    # Fire only treats help as its own flag AFTER the separator. An unseparated
    # --help after valid run arguments can otherwise execute a paid generation first.
    if "--help" in argv or "-h" in argv:
        target = commands.get(argv[0]) if argv else None
        fire.Fire(target or commands, command=["--", "--help"],
                  name=f"synth {argv[0]}" if target else "synth")
        return
    fire.Fire(commands)


if __name__ == "__main__":
    main()
