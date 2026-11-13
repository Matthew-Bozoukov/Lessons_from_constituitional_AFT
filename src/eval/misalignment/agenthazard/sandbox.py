# ABOUTME: Docker side of AgentHazard: the sandbox image, a per-run internal network whose only exit
# ABOUTME: is a one-port proxy to the model, the isolation check, episode containers, a disk watchdog.
"""Safety boundary for an eval whose tasks do real RCE, exfiltration, persistence and sabotage.

- Each episode gets a fresh container from `build_image`, removed after its logs are copied out.
  No bind mounts or volumes, no host env: what an episode sees is the image plus the
  ANTHROPIC_* variables naming the model, with a dummy token.
- Episode containers join ONE per-run network created `--internal`, which has no route out of the
  Docker VM. The only other member is a proxy container (same image, `socat`) that is ALSO on the
  default bridge and forwards exactly one port, alias `model:8000`, to the tunnelled vLLM port on
  the host. So a container reaches the model and nothing else: no internet, no LAN, no host
  services. `check_isolation` proves this before any episode runs, and the run refuses to start
  otherwise.
- Writes are bounded: /root and /tmp are size-capped tmpfs, `--memory`, `--cpus` and
  `--pids-limit` cap the rest, and `DiskWatchdog` kills any container whose writable layer grows
  past a limit (a resource-exhaustion task writing elsewhere would otherwise grow the Docker VM's
  disk into the host's free space) and aborts the run if the host's free disk falls too low.
"""

from __future__ import annotations

import asyncio
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path

ASSETS = Path(__file__).parent / "assets"
PROXY_ALIAS = "model"
PROXY_PORT = 8000
LABEL = "agenthazard.run"


async def run_cmd(
    argv: list[str], timeout: float | None = None, stdin: bytes | None = None
) -> tuple[int, bytes, bytes]:
    """Run a command without a shell; (returncode, stdout, stderr). rc -9 on our timeout."""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE
        if stdin is not None
        else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(stdin), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return -9, b"", b"driver timeout"
    return proc.returncode, out, err


def build_image(tag: str, claude_code_version: str) -> str:
    """Build the sandbox image (layer-cached), returning its image id. Raises on a failed build."""
    b = subprocess.run(
        [
            "docker",
            "build",
            "-t",
            tag,
            "--build-arg",
            f"CLAUDE_CODE_VERSION={claude_code_version}",
            str(ASSETS),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="backslashreplace",
    )
    if b.returncode != 0:
        raise RuntimeError(
            f"docker build of the AgentHazard sandbox failed:\n{b.stdout[-3000:]}\n{b.stderr[-3000:]}"
        )
    got = subprocess.run(
        ["docker", "run", "--rm", "--network", "none", tag, "claude", "--version"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if claude_code_version not in got:
        raise RuntimeError(
            f"image {tag} reports Claude Code {got!r}, expected {claude_code_version}"
        )
    return subprocess.run(
        ["docker", "image", "inspect", "-f", "{{.Id}}", tag],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@dataclass
class RunNetwork:
    """The per-run internal network plus its one-port egress proxy. Context manager."""

    image: str
    host: str  # where the proxy reaches the model: host.docker.internal / 172.17.0.1
    port: int  # the tunnelled vLLM port on the host
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])

    @property
    def network(self) -> str:
        return f"agenthazard-{self.run_id}"

    @property
    def proxy(self) -> str:
        return f"agenthazard-proxy-{self.run_id}"

    @property
    def base_url(self) -> str:
        """ANTHROPIC_BASE_URL as seen from inside an episode container."""
        return f"http://{PROXY_ALIAS}:{PROXY_PORT}"

    def __enter__(self) -> RunNetwork:
        sh = lambda *a: subprocess.run(
            list(a), capture_output=True, text=True, check=True
        )  # noqa: E731
        sh(
            "docker",
            "network",
            "create",
            "--internal",
            "--label",
            f"{LABEL}={self.run_id}",
            self.network,
        )
        try:
            sh(
                "docker", "run", "-d", "--rm", "--name", self.proxy, "--label", f"{LABEL}={self.run_id}",
                "--add-host", "host.docker.internal:host-gateway", "--memory", "256m", "--pids-limit", "256",
                self.image, "socat", f"TCP-LISTEN:{PROXY_PORT},fork,reuseaddr", f"TCP:{self.host}:{self.port}",
            )  # fmt: skip
            sh(
                "docker",
                "network",
                "connect",
                "--alias",
                PROXY_ALIAS,
                self.network,
                self.proxy,
            )
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc) -> None:
        # Every container this run labelled (episodes cut off by an abort included), then the proxy.
        ids = subprocess.run(
            ["docker", "ps", "-aq", "--filter", f"label={LABEL}={self.run_id}"],
            capture_output=True,
            text=True,
        ).stdout.split()
        if ids:
            subprocess.run(["docker", "rm", "-f", *ids], capture_output=True)
        subprocess.run(["docker", "rm", "-f", self.proxy], capture_output=True)
        subprocess.run(["docker", "network", "rm", self.network], capture_output=True)


def docker_args(
    image: str, name: str, network: RunNetwork, env: dict[str, str], limits
) -> list[str]:
    """`docker run` argv for one episode container (no mounts, internal network, capped)."""
    argv = [
        "docker", "run", "-d", "--name", name, "--label", f"{LABEL}={network.run_id}",
        "--network", network.network, "--hostname", "devbox",
        "--memory", str(limits.memory), "--cpus", str(limits.cpus), "--pids-limit", str(limits.pids),
        "--security-opt", "no-new-privileges",
        "--tmpfs", f"/root:rw,exec,size={limits.tmpfs_root}", "--tmpfs", f"/tmp:rw,exec,size={limits.tmpfs_tmp}",
    ]  # fmt: skip
    for k, v in env.items():
        argv += ["-e", f"{k}={v}"]
    return argv + [image, "sleep", "infinity"]


CHECKS = {
    # name: (shell command, must_succeed)
    "model_reachable": (
        f"curl -s -m 10 -o /dev/null -w '%{{{{http_code}}}}' http://{PROXY_ALIAS}:{PROXY_PORT}/v1/models | grep -q 200",
        True,
    ),
    "internet_ip_blocked": ("curl -s -m 6 -o /dev/null http://1.1.1.1", False),
    "internet_dns_blocked": ("curl -s -m 6 -o /dev/null https://huggingface.co", False),
    "host_gateway_blocked": (
        "curl -s -m 6 -o /dev/null http://host.docker.internal:{port}/v1/models",
        False,
    ),
    "bridge_gateway_blocked": (
        "curl -s -m 6 -o /dev/null http://172.17.0.1:{port}/v1/models",
        False,
    ),
}


async def check_isolation(image: str, network: RunNetwork, limits) -> dict[str, bool]:
    """Run CHECKS from a throwaway episode-shaped container; raise unless every one holds."""
    name = f"agenthazard-check-{network.run_id}"
    rc, _, err = await run_cmd(
        docker_args(image, name, network, {}, limits), timeout=60
    )
    if rc != 0:
        raise RuntimeError(
            f"isolation check container failed to start: {err.decode()[-500:]}"
        )
    try:
        results = {}
        for key, (cmd, must) in CHECKS.items():
            rc, _, _ = await run_cmd(
                ["docker", "exec", name, "bash", "-c", cmd.format(port=network.port)],
                timeout=30,
            )
            results[key] = (rc == 0) == must
    finally:
        await run_cmd(["docker", "rm", "-f", name], timeout=60)
    if not all(results.values()):
        raise RuntimeError(
            f"AgentHazard sandbox isolation check FAILED, refusing to run episodes: {results}"
        )
    return results


_SIZE = re.compile(r"^([\d.]+)\s*([kMGT]?B)")
_UNIT = {"B": 1, "kB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12}


def parse_size(text: str) -> float:
    """Bytes from `docker ps --size`'s '12.3MB (virtual 1.2GB)' (the writable-layer part)."""
    m = _SIZE.match(text.strip())
    return float(m.group(1)) * _UNIT[m.group(2)] if m else 0.0


@dataclass
class DiskWatchdog:
    """Kill episode containers whose writable layer passes `max_bytes`; flag low host disk."""

    run_id: str
    max_bytes: float
    min_host_free_bytes: float
    interval_s: float = 5.0
    killed: set[str] = field(default_factory=set)
    abort_reason: str | None = None

    async def loop(self) -> None:
        while True:
            free = shutil.disk_usage("/").free
            if free < self.min_host_free_bytes:
                self.abort_reason = f"host free disk {free / 1e9:.1f} GB below {self.min_host_free_bytes / 1e9:.0f} GB"
            rc, out, _ = await run_cmd(
                [
                    "docker",
                    "ps",
                    "--size",
                    "--filter",
                    f"label={LABEL}={self.run_id}",
                    "--format",
                    "{{.Names}}\t{{.Size}}",
                ],
                timeout=60,
            )
            if rc == 0:
                for line in out.decode().splitlines():
                    name, _, size = line.partition("\t")
                    if (
                        name.startswith("agenthazard-ep-")
                        and parse_size(size) > self.max_bytes
                    ):
                        self.killed.add(name)
                        await run_cmd(["docker", "kill", name], timeout=60)
            await asyncio.sleep(self.interval_s)
