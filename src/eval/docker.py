# ABOUTME: Shared docker glue for evals whose rollouts run in containers (EvalSpec.needs_docker):
# ABOUTME: a driver-side preflight that refuses unusable hosts with a specific remedy.

from __future__ import annotations

import json
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
              fix="use a host with full docker privileges — a laptop with Docker Desktop. "
                  "This is exactly how RunPod pods fail: daemon up bridgeless, "
                  "networks impossible")


# What Docker hands user-defined networks out of when the daemon config names no
# `default-address-pools`: 172.17.0.0/12 as /16s (16, the first of them the default bridge)
# and 192.168.0.0/16 as /20s (16). Thirty-one networks, and a Compose project that declares
# two (ODCV: `default` + `internal_net`) exhausts them at sixteen scenarios in flight.
_DEFAULT_POOL_CAPACITY = 31


def network_capacity(info_json: str | None = None) -> int:
    """How many user-defined networks this daemon can hold at once, from `docker info`.

    Args:
        info_json: The `docker info --format '{{json .}}'` text; fetched when None.
            Injectable so the arithmetic is unit-tested without a daemon.
    """
    if info_json is None:
        try:
            out = subprocess.run(["docker", "info", "--format", "{{json .}}"],
                                 capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            return 0
        if out.returncode != 0:
            return 0
        info_json = out.stdout
    try:
        pools = json.loads(info_json).get("DefaultAddressPools") or []
    except ValueError:
        return 0
    if not pools:
        return _DEFAULT_POOL_CAPACITY
    total = 0
    for pool in pools:
        base, size = str(pool.get("Base", "")), int(pool.get("Size", 0))
        prefix = int(base.rsplit("/", 1)[-1]) if "/" in base else 0
        if size >= prefix > 0:
            total += 2 ** (size - prefix)
    return total


def require_network_capacity(networks: int, *, because: str) -> int:
    """Refuse a run whose concurrency would exhaust the daemon's address pools.

    The failure this pre-empts looks like a flaky harness rather than a config limit:
    `compose up` builds the images, then dies creating the project network with
    "all predefined address pools have been fully subnetted", and the cell is a hole
    with no container. At 32 ODCV scenarios in flight on a default Docker Desktop
    (2026-09-06) about half of every wave failed that way, and one retry per pass
    cannot close a gap that size.

    Args:
        networks: Networks the run will hold at once (ODCV: 2 x concurrency).
        because: Who needs them, for the message.

    Returns:
        The capacity, for the caller's log line.
    """
    capacity = network_capacity()
    # 0 means the daemon did not answer or said nothing parseable — unknown, not
    # insufficient; the generic preflight is what refuses an unreachable daemon.
    if capacity and networks > capacity:
        _fail(f"this run needs up to {networks} docker networks at once but the daemon's "
              f"address pools hold {capacity}",
              because=because,
              fix='raise the pool in the daemon config (Docker Desktop: Settings > Docker '
                  'Engine, or ~/.docker/daemon.json) — e.g. "default-address-pools": '
                  '[{"base": "10.200.0.0/14", "size": 24}] gives 1024 /24 networks — and '
                  "restart the daemon; or lower `concurrency`")
    return capacity
