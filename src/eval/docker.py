# ABOUTME: Shared docker glue for evals whose rollouts run in containers (EvalSpec.needs_docker):
# ABOUTME: a driver-side preflight that refuses unusable hosts with a specific remedy.

from __future__ import annotations

import shutil
import subprocess
import uuid


def _fail(problem: str, *, because: str, fix: str) -> None:
    raise SystemExit(
        f"\nDocker preflight failed: {problem}\n"
        f"  Why it matters: {because}\n"
        f"  Fix: {fix}\n"
        "  (Docker evals must drive where docker works — a laptop with Docker Desktop or\n"
        "  a vast.ai instance. RunPod pods are unprivileged containers and can never pass\n"
        "  these checks; see docs/LOG.md 2026-08-03.)")


def docker_preflight() -> None:
    """Fail fast, with a specific remedy, on every docker capability container evals need.

    Checks run in dependency order, each one assuming the previous passed: binary →
    daemon → compose plugin → network creation. The last is the decisive one: a daemon
    can run bridgeless on an unprivileged host and still be useless for scenario
    containers — ODCV, for one, creates a Compose project network per scenario.
    """
    if not shutil.which("docker"):
        _fail("no `docker` binary on PATH",
              because="docker evals run their rollouts in scenario containers "
                      "(ODCV: executor + orchestrator per scenario)",
              fix="install Docker (Docker Desktop on macOS), or run on a vast.ai instance")

    info = subprocess.run(["docker", "info"], capture_output=True, text=True)
    if info.returncode != 0:
        err = (info.stderr or info.stdout).strip()[-400:]
        if "permission denied" in err.lower():
            _fail("the docker daemon refused this user (permission denied)",
                  because="rollouts must create/destroy containers without sudo",
                  fix="`sudo usermod -aG docker $USER && newgrp docker`")
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

    probe = f"docker-preflight-{uuid.uuid4().hex[:8]}"
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
