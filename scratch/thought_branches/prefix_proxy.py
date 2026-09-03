# ABOUTME: An OpenAI-compatible shim that replays a rollout's first k assistant turns
# ABOUTME: verbatim and then forwards to the live model — live-environment branching.

"""How to branch a trajectory whose continuation depends on a container.

To resample ODCV step k and still get a real ending, the environment has to reach the
same state the original rollout was in at step k. Replaying commands out-of-band is
fragile: the scenario's own scripts write files, mutate state, and sometimes depend on
timing, so a hand-driven replay diverges quietly.

The trick is to leave the container completely alone and lie to it about the model. The
ODCV executor is an ordinary chat loop against `OPENAI_BASE_URL`; point that at this
proxy instead. For the first k assistant turns the proxy hands back the assistant
messages the original rollout produced, so the container runs exactly the same commands
and reaches exactly the same state. From turn k it forwards to the real model, which now
free-runs from a genuinely reproduced world. The container never learns anything changed,
the tool results after the branch are real, and the trajectory reaches an ending the ODCV
judges can score without any special-casing.

Turn accounting is by CONTENT, not by call count: each request carries the whole history,
so the proxy counts the assistant messages in the incoming payload and serves
`forced[that many]`. A retry, a dropped connection or a harness that re-sends the same
turn therefore replays identically instead of sliding the branch point.

Usage:

    forced = forced_messages(traj, upto_step=bp.step_index)
    with PrefixProxy(forced, upstream_base_url=..., upstream_key=..., model=...) as proxy:
        # point the container at proxy.container_url() and run the scenario as usual
        ...

Stdlib only, so it adds no dependency and starts in milliseconds.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from scratch.thought_branches.trajectory import Trajectory


def forced_messages(traj: Trajectory, upto_step: int) -> list[dict]:
    """The assistant turns to replay before handing control to the live model.

    Args:
        traj: The trajectory being branched.
        upto_step: Transcript step index of the branch point; assistant turns strictly
            before it are forced.

    Returns:
        OpenAI assistant messages, in order, ready to be served back verbatim.
    """
    out: list[dict] = []
    for s in traj.steps:
        if s.index >= upto_step:
            break
        if not s.is_assistant:
            continue
        msg: dict[str, Any] = {"role": "assistant", "content": s.content or ""}
        if s.reason:
            msg["reasoning_content"] = s.reason
        if s.calls:
            msg["tool_calls"] = [
                {
                    "id": c.call_id or f"call_{s.index}_{i}",
                    "type": "function",
                    "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                }
                for i, c in enumerate(s.calls)
            ]
        out.append(msg)
    return out


def container_host_address(platform: str = "") -> str:
    """The address a Docker container uses to reach a server on its host.

    Mirrors the rule ODCV already relies on: Docker Desktop resolves
    `host.docker.internal`, plain Linux Docker needs the bridge gateway.

    Args:
        platform: `sys.platform` value; read from the running interpreter when empty.

    Returns:
        The hostname to put in the container's `OPENAI_BASE_URL`.
    """
    import sys

    plat = platform or sys.platform
    return "172.17.0.1" if plat.startswith("linux") else "host.docker.internal"


@dataclass
class ProxyStats:
    """What the proxy did, for a branch run's provenance.

    Attributes:
        forced_served: Recorded turns handed back.
        live_served: Turns forwarded to the real model.
        errors: Upstream failures, as strings.
        diverged: True if a request arrived with FEWER assistant messages than a request
            already served — the container restarted its loop, so the replay is suspect.
    """

    forced_served: int = 0
    live_served: int = 0
    errors: list[str] = field(default_factory=list)
    diverged: bool = False


class PrefixProxy:
    """A local OpenAI-compatible endpoint that forces a prefix, then goes live.

    Attributes:
        forced: Assistant messages to replay, in order.
        upstream_base_url: Where live turns are forwarded.
        upstream_key: API key for upstream.
        model: Model name reported in responses and sent upstream.
        port: Host port; 0 picks a free one.
    """

    def __init__(
        self,
        forced: list[dict],
        upstream_base_url: str,
        upstream_key: str = "EMPTY",
        model: str = "model",
        port: int = 0,
        timeout: float = 600.0,
    ) -> None:
        self.forced = list(forced)
        self.upstream_base_url = upstream_base_url.rstrip("/")
        self.upstream_key = upstream_key
        self.model = model
        self.timeout = timeout
        self.stats = ProxyStats()
        self._max_seen = -1
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("0.0.0.0", port), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    # -- lifecycle ---------------------------------------------------------------

    def __enter__(self) -> "PrefixProxy":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    def base_url(self, host: str = "127.0.0.1") -> str:
        """The base URL to hand a client on this machine."""
        return f"http://{host}:{self.port}/v1"

    def container_url(self) -> str:
        """The base URL to hand a Docker container on this host."""
        return f"http://{container_host_address()}:{self.port}/v1"

    # -- request handling --------------------------------------------------------

    def _reply_for(self, body: dict) -> dict:
        """Decide this turn: replay a recorded message, or forward upstream."""
        msgs = body.get("messages") or []
        n_assistant = sum(1 for m in msgs if m.get("role") == "assistant")
        with self._lock:
            if n_assistant < self._max_seen:
                self.stats.diverged = True
            self._max_seen = max(self._max_seen, n_assistant)
        if n_assistant < len(self.forced):
            with self._lock:
                self.stats.forced_served += 1
            return self._envelope(
                self.forced[n_assistant],
                "tool_calls" if self.forced[n_assistant].get("tool_calls") else "stop",
            )
        with self._lock:
            self.stats.live_served += 1
        return self._forward(body)

    def _envelope(self, message: dict, finish_reason: str) -> dict:
        """Wrap a stored assistant message in an OpenAI chat-completion response."""
        return {
            "id": f"chatcmpl-forced-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model,
            "choices": [
                {"index": 0, "message": message, "finish_reason": finish_reason}
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    def _forward(self, body: dict) -> dict:
        """Send a live turn upstream and return its response verbatim."""
        body = dict(body)
        body.setdefault("model", self.model)
        body.pop("stream", None)
        req = urllib.request.Request(
            f"{self.upstream_base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.upstream_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:400]
            with self._lock:
                self.stats.errors.append(f"HTTP {e.code}: {detail}")
            raise
        except Exception as e:  # noqa: BLE001
            with self._lock:
                self.stats.errors.append(str(e)[:200])
            raise

    def _handler(self):
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args) -> None:  # noqa: A003
                """Silence per-request logging; the harness's own logs are the record."""

            def _send(self, code: int, payload: dict) -> None:
                data = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:  # noqa: N802
                # The executor health-checks /v1/models before it starts.
                if self.path.rstrip("/").endswith("/models"):
                    self._send(
                        200,
                        {
                            "object": "list",
                            "data": [{"id": proxy.model, "object": "model"}],
                        },
                    )
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                except json.JSONDecodeError:
                    self._send(400, {"error": "bad json"})
                    return
                try:
                    self._send(200, proxy._reply_for(body))
                except Exception as e:  # noqa: BLE001
                    self._send(
                        502,
                        {"error": {"message": str(e)[:400], "type": "upstream_error"}},
                    )

        return Handler
