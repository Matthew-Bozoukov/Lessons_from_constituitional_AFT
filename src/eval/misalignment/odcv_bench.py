# ABOUTME: Eval-framework entrypoint for ODCV-Bench: containerized agent rollouts against a
# ABOUTME: served target (reached at the docker-bridge address), then multi-judge scoring.

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf

from src.eval.misalignment import odcv_judge, odcv_rollout

# Scenario containers reach the host's vLLM via the default docker bridge, not localhost.
_DOCKER_BRIDGE = "172.17.0.1"


def _bridge_url(base_url: str) -> str:
    """Rewrite a localhost endpoint to the docker-bridge address (pure; unit-tested)."""
    return base_url.replace("localhost", _DOCKER_BRIDGE).replace("127.0.0.1", _DOCKER_BRIDGE)


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
