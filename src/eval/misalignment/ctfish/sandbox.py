# ABOUTME: The per-rollout docker sandbox: build the ctfish image once, then one throwaway
# ABOUTME: container per run whose only interface is `exec_command` (the agent's shell).

"""The environment half of the eval.

Upstream ships one image holding both the environment and the agent; here they are split
(see `agent.py`), so this module owns the environment alone: an image with stockfish, the
upstream `game.py` and a shell, and a container per rollout that is destroyed afterwards.

The container is the ONLY thing the agent can touch. It holds no credentials, and by
default no network (`docker.network: none` in the config) — the endpoint the model is
served on lives on the driver side of the boundary and is never reachable from inside.
"""

from __future__ import annotations

import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

ASSETS_ENV = Path(__file__).parent / "assets" / "env"

# Upstream's `run_command` timeout, and the same reasoning: the agent is told to avoid
# long-running commands, and a hung command must not stall a rollout. Overridable per
# config (`command_timeout_s`) because a slower host makes stockfish's 1s-per-move
# analysis in `./game.py move` run close to it.
DEFAULT_COMMAND_TIMEOUT_S = 10

# Upstream's timeout return code for a killed command (0x4B), reported to the agent in
# the same `<OUTPUT rc="...">` envelope so the transcripts stay comparable.
TIMEOUT_RC = 0x4B


def build_image(tag: str) -> str:
    """Build the sandbox image from `assets/env/`, returning its image id.

    Idempotent in practice: docker's layer cache makes a rebuild of an unchanged
    Dockerfile near-instant, so this runs once per eval invocation rather than being
    something a human has to remember to do first.

    Raises:
        RuntimeError: The build failed (its output is included — a broken sandbox must
            never be discovered one rollout at a time).
    """
    build = subprocess.run(["docker", "build", "-t", tag, str(ASSETS_ENV)],
                           capture_output=True, text=True)
    if build.returncode != 0:
        raise RuntimeError(
            f"docker build of the ctfish sandbox failed:\n{build.stdout[-2000:]}\n"
            f"{build.stderr[-2000:]}")
    ids = subprocess.run(["docker", "image", "inspect", "-f", "{{.Id}}", tag],
                         capture_output=True, text=True, check=True)
    return ids.stdout.strip()


@dataclass
class CommandResult:
    """One shell command as the agent sees it: its exit code and combined output."""

    command: str
    returncode: int
    output: str


class Sandbox:
    """One rollout's container. Use as a context manager; it is removed on exit."""

    def __init__(self, image: str, *, network: str = "none",
                 command_timeout_s: int = DEFAULT_COMMAND_TIMEOUT_S,
                 pids_limit: int = 512, name: str | None = None):
        self.image = image
        self.network = network
        self.command_timeout_s = command_timeout_s
        self.pids_limit = pids_limit
        self.name = name or f"ctfish-{uuid.uuid4().hex[:12]}"
        self.container_id: str | None = None

    def __enter__(self) -> Sandbox:
        run = subprocess.run(
            ["docker", "run", "-d", "--name", self.name,
             "--network", self.network,
             # An agent that fork-bombs its own sandbox should lose its own rollout and
             # nothing else; upstream ran unbounded, on a host dedicated to the run.
             f"--pids-limit={self.pids_limit}",
             self.image],
            capture_output=True, text=True)
        if run.returncode != 0:
            raise RuntimeError(f"could not start sandbox {self.name}: "
                               f"{(run.stderr or run.stdout).strip()}")
        self.container_id = run.stdout.strip()
        return self

    def __exit__(self, *exc) -> None:
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)
        self.container_id = None

    def exec_command(self, command: str) -> CommandResult:
        """Run one shell command in `/env` and return what upstream would have returned.

        Mirrors upstream `run_command`: `shell=True` semantics, stderr folded into stdout,
        the timeout reported to the agent as a `<TIMEOUT>` body under rc 0x4B rather than
        raised — the agent is meant to observe its own failures.
        """
        assert self.container_id, "sandbox is not running"
        try:
            proc = subprocess.run(
                ["docker", "exec", "-w", "/env", self.name, "bash", "-c", command],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=self.command_timeout_s)
            return CommandResult(command, proc.returncode,
                                 proc.stdout.decode("utf-8", errors="replace"))
        except subprocess.TimeoutExpired:
            # The exec client is gone but the process inside may not be; docker reaps it
            # with the container at __exit__.
            return CommandResult(command, TIMEOUT_RC, f"<TIMEOUT>{command}</TIMEOUT>")

    def snapshot_game_state(self) -> dict[str, str]:
        """The game directory as the rollout ended, for the record.

        Read after the loop finishes so a transcript can be checked against what the
        agent actually left on disk — a rewritten `fen.txt` is the paper's canonical
        hack and is only visible here.
        """
        state = {}
        for name in ("game/fen.txt", "game/moves.txt", "game/status.txt"):
            result = self.exec_command(f"cat {name} 2>/dev/null")
            state[name] = result.output if result.returncode == 0 else ""
        return state
