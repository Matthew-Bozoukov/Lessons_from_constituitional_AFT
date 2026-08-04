# ABOUTME: Eval-framework entrypoint for ODCV-Bench: containerized agent rollouts against a
# ABOUTME: served target (reached at the docker host address), then multi-judge scoring.

from __future__ import annotations

import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

from src.eval.misalignment.odcv import odcv_judge, odcv_rollout


def container_host_address() -> str:
    """Where a container reaches the machine it runs on.

    Linux docker exposes the host at the default bridge gateway; Docker Desktop
    (macOS/Windows) has no host-side bridge interface and provides the special
    `host.docker.internal` name instead (both patterns proven in docs/LOG.md).
    """
    return "host.docker.internal" if sys.platform == "darwin" else "172.17.0.1"


def _bridge_url(base_url: str, address: str | None = None) -> str:
    """Rewrite a localhost endpoint to the container-reachable host address."""
    address = address or container_host_address()
    return base_url.replace("localhost", address).replace("127.0.0.1", address)


def run(target, cfg, out_dir: Path) -> dict:
    """Run ODCV-Bench rollouts + judging against a ServedTarget (CLAUDE.md contract).

    Returns:
        The parsed results.json summary from the judged rollout.
    """
    cfg = OmegaConf.merge(cfg)  # private copy
    cfg.model = target.model_name
    cfg.model_key = target.spec.model_key
    cfg.base_url = _bridge_url(target.base_url)
    cfg.output_root = str(out_dir)

    # The rollout/judge mains load their config from a path (their resume/caching keys
    # off it), so materialize the per-target config rather than passing objects around.
    cfg_path = out_dir / "odcv_config.yaml"
    OmegaConf.save(cfg, cfg_path)
    smoke = bool(cfg.get("smoke", False))

    odcv_rollout.main(config=str(cfg_path), smoke=smoke)
    rollout_dir = max((out_dir / target.spec.model_key).glob("*/"), key=lambda p: p.name)
    odcv_judge.main(rollout_dir=str(rollout_dir), config=str(cfg_path),
                    max_workers=int(cfg.get("judge_workers", 8)), smoke=smoke)

    return json.loads((rollout_dir / "results.json").read_text())
