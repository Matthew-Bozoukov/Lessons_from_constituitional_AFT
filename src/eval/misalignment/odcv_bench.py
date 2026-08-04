# ABOUTME: Eval-framework entrypoint for ODCV-Bench: containerized agent rollouts against a
# ABOUTME: served target (reached at the docker host address), then multi-judge scoring.

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from omegaconf import OmegaConf

from src.eval.misalignment import odcv_judge, odcv_rollout


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


def _fail(problem: str, *, because: str, fix: str) -> None:
    raise SystemExit(
        f"\nODCV preflight failed: {problem}\n"
        f"  Why it matters: {because}\n"
        f"  Fix: {fix}\n"
        "  (ODCV is the one eval that must run where docker runs — a laptop with Docker\n"
        "  Desktop or a vast.ai instance. RunPod pods are unprivileged containers and can\n"
        "  never pass these checks; see docs/LOG.md 2026-08-03.)")


def docker_preflight() -> None:
    """Fail fast, with a specific remedy, on every docker capability ODCV needs.

    Checks run in dependency order, each one assuming the previous passed: binary →
    daemon → compose plugin → network creation. The last is the decisive one: a daemon
    can run bridgeless on an unprivileged host and still be useless for ODCV, whose
    every scenario creates its own Compose project network.
    """
    if not shutil.which("docker"):
        _fail("no `docker` binary on PATH",
              because="every ODCV scenario runs two containers (executor + orchestrator)",
              fix="install Docker (Docker Desktop on macOS), or run on a vast.ai instance")

    info = subprocess.run(["docker", "info"], capture_output=True, text=True)
    if info.returncode != 0:
        err = (info.stderr or info.stdout).strip()[-400:]
        if "permission denied" in err.lower():
            _fail("the docker daemon refused this user (permission denied)",
                  because="rollouts must create/destroy containers without sudo",
                  fix="`sudo usermod -aG docker $USER && newgrp docker` (replication.md)")
        _fail(f"the docker daemon is not reachable:\n    {err}",
              because="no daemon means no scenario containers at all",
              fix="start Docker (Desktop app / `systemctl start docker`); on an "
                  "unprivileged container host the daemon cannot start at all")

    compose = subprocess.run(["docker", "compose", "version"], capture_output=True, text=True)
    if compose.returncode != 0:
        _fail("`docker compose` (v2 plugin) is missing",
              because="each scenario is brought up as its own Compose project",
              fix="install the compose plugin (ships with Docker Desktop; "
                  "`apt install docker-compose-plugin` on linux)")

    probe = f"odcv-preflight-{uuid.uuid4().hex[:8]}"
    create = subprocess.run(["docker", "network", "create", probe],
                            capture_output=True, text=True)
    subprocess.run(["docker", "network", "rm", probe], capture_output=True)
    if create.returncode != 0:
        _fail("the daemon runs but CANNOT create networks:\n    "
              f"{(create.stderr or create.stdout).strip()[-300:]}",
              because="every scenario gets its own Compose network (with an "
                      "orchestrator_api alias the harness hardcodes); without network "
                      "creation no rollout can start",
              fix="use a host with full docker privileges — vast.ai per the playbook. "
                  "This is exactly how RunPod pods fail: daemon up bridgeless, "
                  "networks impossible")


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
